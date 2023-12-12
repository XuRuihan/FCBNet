import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair

import cupy
from string import Template
from collections import namedtuple

from .parcnetv2 import OversizeConv2d

Stream = namedtuple("Stream", ["ptr"])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return "float"
    elif isinstance(t, torch.cuda.DoubleTensor):
        return "double"


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = """
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
"""


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_idynamic_kernel = (
    kernel_loop
    + """
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
"""
)


_idynamic_kernel_backward_grad_input = (
    kernel_loop
    + """
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
"""
)


_idynamic_kernel_backward_grad_weight = (
    kernel_loop
    + """
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
"""
)


class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int(
            (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0]
            + 1
        )
        output_w = int(
            (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1]
            + 1
        )

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                "idynamic_forward_kernel",
                _idynamic_kernel,
                Dtype=Dtype(input),
                nthreads=n,
                num=batch_size,
                channels=channels,
                groups=weight.size()[1],
                bottom_height=height,
                bottom_width=width,
                top_height=output_h,
                top_width=output_w,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride[0],
                stride_w=stride[1],
                dilation_h=dilation[0],
                dilation_w=dilation[1],
                pad_h=padding[0],
                pad_w=padding[1],
            )
            f(
                block=(CUDA_NUM_THREADS, 1, 1),
                grid=(GET_BLOCKS(n), 1, 1),
                args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
            )

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        grad_output = grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(
            Dtype=Dtype(grad_output),
            num=batch_size,
            channels=channels,
            groups=weight.size()[1],
            bottom_height=height,
            bottom_width=width,
            top_height=output_h,
            top_width=output_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride[0],
            stride_w=stride[1],
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            pad_h=padding[0],
            pad_w=padding[1],
        )

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt["nthreads"] = n

                f = load_kernel(
                    "idynamic_backward_grad_input_kernel",
                    _idynamic_kernel_backward_grad_input,
                    **opt,
                )
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(n), 1, 1),
                    args=[
                        grad_output.data_ptr(),
                        weight.data_ptr(),
                        grad_input.data_ptr(),
                    ],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
                )

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt["nthreads"] = n

                f = load_kernel(
                    "idynamic_backward_grad_weight_kernel",
                    _idynamic_kernel_backward_grad_weight,
                    **opt,
                )
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(n), 1, 1),
                    args=[
                        grad_output.data_ptr(),
                        input.data_ptr(),
                        grad_weight.data_ptr(),
                    ],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
                )

        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ idynamic kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(
            input, weight, _pair(stride), _pair(padding), _pair(dilation)
        )
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class IDynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, group_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1)
        )

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out


class IDynamicDWConv_add(nn.Module):
    def __init__(self, channels, kernel_size, global_kernel_size, group_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.global_kernel_size = global_kernel_size
        self.channels = channels
        reduction_ratio = 8
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(
                channels // reduction_ratio,
                self.groups * (kernel_size ** 2 + global_kernel_size * 2),
                1,
            ),
        )

    def forward(self, x):
        weight = self.conv1(x)
        b, c, h, w = weight.shape
        weight, global_weight_h, global_weight_w = weight.split(
            [
                self.groups * self.kernel_size ** 2,
                self.groups * self.global_kernel_size,
                self.groups * self.global_kernel_size,
            ],
            dim=1,
        )
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        global_weight_h = global_weight_h.clamp(-1, 1)
        global_weight_w = global_weight_w.clamp(-1, 1)
        global_weight_h = global_weight_h.view(
            b, self.groups, self.global_kernel_size, 1, h, w
        )
        global_weight_w = global_weight_w.view(
            b, self.groups, 1, self.global_kernel_size, h, w
        )
        x1 = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        x2 = _idynamic_cuda(
            x,
            global_weight_h,
            stride=1,
            padding=[(self.global_kernel_size - 1) // 2, 0],
        )
        x2 = _idynamic_cuda(
            x2,
            global_weight_w,
            stride=1,
            padding=[0, (self.global_kernel_size - 1) // 2],
        )
        return x1 + x2


class ParC_V3_idynamic(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=32,
        expansion_ratio=2,
        act_layer=nn.GELU,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.kernel_size = kernel_size
        self.num_heads = dim // head_dim

        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        self.conv_kernel = nn.Conv2d(
            med_channels // 2, kernel_size ** 2 * self.num_heads, 1
        )
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)

        weight = self.conv_kernel(x1).view(
            B, self.num_heads, self.kernel_size, self.kernel_size, H, W
        )

        x2 = self.act(x2)
        x2 = _idynamic_cuda(x2, weight, stride=1, padding=(self.kernel_size - 1) // 2)

        x = self.pwconv2(x2)
        x = x.permute(0, 2, 3, 1)
        return x


class ParC_V3_add_idynamic(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        expansion_ratio=2,
        act_layer=nn.GELU,
        bias=False,
        kernel_size=7,
        global_kernel_size=13,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.kernel_size = kernel_size
        self.global_kernel_size = global_kernel_size
        self.num_heads = num_heads or dim // head_dim
        self.dim = dim
        self.reduce_dim = med_channels - dim

        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        self.conv_kernel = nn.Conv2d(
            self.reduce_dim,
            kernel_size ** 2 * self.num_heads,
            1,
            bias=bias,
            groups=self.num_heads,
        )
        self.global_conv_kernel = nn.Conv2d(
            self.reduce_dim,
            global_kernel_size * 2 * self.num_heads,
            1,
            bias=bias,
            groups=self.num_heads,
        )
        self.pwconv2 = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = self.pwconv1(x)
        x1, x2 = x.split([self.reduce_dim, self.dim], 1)

        weight = self.conv_kernel(x1).view(
            B, self.num_heads, self.kernel_size, self.kernel_size, H, W
        )
        global_weight = self.global_conv_kernel(x1).clamp(-1, 1)
        global_weight_h, global_weight_w = global_weight.chunk(2, 1)
        global_weight_h = global_weight_h.view(
            B, self.num_heads, self.global_kernel_size, 1, H, W
        )
        global_weight_w = global_weight_w.view(
            B, self.num_heads, 1, self.global_kernel_size, H, W
        )

        x2 = self.act(x2)
        x2_1 = _idynamic_cuda(x2, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        x2_2 = _idynamic_cuda(
            x2,
            global_weight_h,
            stride=1,
            padding=[(self.global_kernel_size - 1) // 2, 0],
        )
        x2_3 = _idynamic_cuda(
            x2,
            global_weight_w,
            stride=1,
            padding=[0, (self.global_kernel_size - 1) // 2],
        )
        x2 = x2_1 + x2_2 + x2_3

        x = self.pwconv2(x2)
        x = x.permute(0, 2, 3, 1)
        return x


class ParC_V3(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=32,
        expansion_ratio=2,
        act_layer=nn.GELU,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.kernel_size = kernel_size
        self.num_heads = dim // head_dim

        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        self.conv_kernel = nn.Conv2d(
            med_channels // 2, kernel_size ** 2 * self.num_heads, 1
        )
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)

        weight = self.conv_kernel(x1).view(
            B, self.num_heads, self.kernel_size, self.kernel_size, H, W
        )

        x2 = self.act(x2)
        x2 = _idynamic_cuda(x2, weight, stride=1, padding=(self.kernel_size - 1) // 2)

        x = self.pwconv2(x2)
        x = x.permute(0, 2, 3, 1)
        return x


class ParC_V3_add(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=16,
        num_heads=None,
        expansion_ratio=2,
        act_layer=nn.GELU,
        bias=False,
        kernel_size=7,
        global_kernel_size=13,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.kernel_size = kernel_size
        self.global_kernel_size = global_kernel_size
        self.num_heads = num_heads or dim // head_dim

        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=True)
        self.act = act_layer()
        # self.dwconv1 = OversizeConv2d(med_channels // 2, global_kernel_size, bias)
        # self.dwconv2 = IDynamicDWConv(med_channels // 2, kernel_size, head_dim)
        self.dwconv = IDynamicDWConv_add(med_channels // 2, kernel_size, global_kernel_size, head_dim)
        self.pwconv2 = nn.Conv2d(med_channels // 2, dim, 1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = self.pwconv1(x)
        x1, x2 = x.chunk(2, 1)
        x2 = self.act(x2)
        # x2 = self.dwconv1(x2) + self.dwconv2(x2)
        x2 = self.dwconv(x2)
        x = x1 * x2
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1)
        return x


if __name__ == "__main__":
    net = ParC_V3_add(32).cuda()
    x = torch.rand(1, 14, 14, 32).cuda()
    y = net(x)
    print(y.shape)
