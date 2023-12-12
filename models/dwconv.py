import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair

import cupy
from string import Template
from collections import namedtuple

Stream = namedtuple("Stream", ["ptr"])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return "float"
    elif isinstance(t, torch.cuda.DoubleTensor):
        return "double"
    else:
        raise NotImplementedError


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
          const int offset_weight = (g * ${kernel_h} + kh) * ${kernel_w} + kw;
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
            const int offset = ((n * ${groups} + c) * ${top_height} + h_out) * ${top_width} + w_out;
            const int offset_weight = (g * ${kernel_h} + kh) * ${kernel_w} + kw;
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
    const int kh = (index / ${kernel_w}) % ${kernel_h};
    const int kw = index % ${kernel_w};
    const int g = (index / ${kernel_h} / ${kernel_w}) % ${groups};
    const int n = index / ${kernel_h} / ${kernel_w} / ${groups};
    ${Dtype} value = 0;
    #pragma unroll
    for (int n = 0; n < ${num}; ++n){
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        #pragma unroll
        for (int h_in = -${pad_h} + kh * ${dilation_h}; h_in + (${kernel_h} - kh - 1) * ${dilation_h} < ${bottom_height} + ${pad_h}; h_in += ${stride_h}) {
          #pragma unroll
          for (int w_in = -${pad_w} + kw * ${dilation_w}; w_in + (${kernel_w} - kw - 1) * ${dilation_w} < ${bottom_width} + ${pad_w}; w_in += ${stride_w}) {
            if (h_in >= 0 && h_in < ${bottom_height} && w_in >= 0 && w_in < ${bottom_width}) {
              const int top_offset = ((n * ${channels} + c) * ${top_height} + h_in)
                    * ${top_width} + w_in;  // should be h_out and w_out here
              const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
                    * ${bottom_width} + w_in;
              value += top_diff[top_offset] * bottom_data[bottom_offset];
            }
          }
        }
      }
    }
    buffer_data[index] = value;
  }
}
"""
)


class _dwconv(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 4 and weight.is_cuda
        assert weight.size()[1] == 1 and weight.size()[0] == input.size()[1]
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
                groups=weight.size()[0],
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
            groups=weight.size()[0],
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


def _dwconv_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    if input.is_cuda:
        out = _dwconv.apply(
            input, weight, _pair(stride), _pair(padding), _pair(dilation)
        )
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class DWConv(nn.Module):
    def __init__(self, channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.channels = channels
        assert bias is False
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=bias
        )

    def forward(self, x):
        weight = self.conv.weight
        out = _dwconv_cuda(x, weight, stride=1, padding=self.padding)
        return out


if __name__ == "__main__":
    net1 = DWConv(3, 7, 3).cuda()
    net2 = nn.Conv2d(3, 3, 7, padding=3, groups=3, bias=False).cuda()
    x1 = torch.arange(2 * 3 * 6 * 5., requires_grad=True).reshape(2, 3, 6, 5).cuda()
    x1.retain_grad()
    x2 = torch.arange(2 * 3 * 6 * 5., requires_grad=True).reshape(2, 3, 6, 5).cuda()
    x2.retain_grad()
    nn.init.constant_(net1.conv.weight, 1)
    nn.init.constant_(net2.weight, 1)
    y1 = net1(x1)
    y2 = net2(x2)
    assert y1.allclose(y2)
    y1.sum().backward()
    y2.sum().backward()
    assert x1.grad.allclose(x2.grad)
    assert net1.conv.weight.grad.allclose(net2.weight.grad)
