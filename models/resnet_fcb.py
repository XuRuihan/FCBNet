"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import (
    DropBlock2d,
    DropPath,
    AvgPool2dSame,
    create_attn,
    create_classifier,
)
from timm.models.helpers import build_model_with_cfg, checkpoint_seq
from timm.models.registry import register_model

__all__ = [
    "ResNet",
    "BasicBlock",
    "Bottleneck",
]  # model_registry will add each entrypoint fn to this


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv1",
        "classifier": "fc",
        **kwargs,
    }


default_cfgs = {
    # ResNet and Wide ResNet
    "resnet10t": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth",
        input_size=(3, 176, 176),
        pool_size=(6, 6),
        test_crop_pct=0.95,
        test_input_size=(3, 224, 224),
        first_conv="conv1.0",
    ),
    "resnet14t": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth",
        input_size=(3, 176, 176),
        pool_size=(6, 6),
        test_crop_pct=0.95,
        test_input_size=(3, 224, 224),
        first_conv="conv1.0",
    ),
    "resnet18": _cfg(url="https://download.pytorch.org/models/resnet18-5c106cde.pth"),
    "resnet18d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
    "resnet34": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth"
    ),
    "resnet34d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
    "resnet26": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth",
        interpolation="bicubic",
    ),
    "resnet26d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
    "resnet26t": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
        input_size=(3, 256, 256),
        pool_size=(8, 8),
        crop_pct=0.94,
    ),
    "resnet50": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
        interpolation="bicubic",
        crop_pct=0.95,
    ),
    "resnet50d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
    "resnet50t": _cfg(url="", interpolation="bicubic", first_conv="conv1.0"),
    "resnet101": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth",
        interpolation="bicubic",
        crop_pct=0.95,
    ),
    "resnet101d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
        input_size=(3, 256, 256),
        pool_size=(8, 8),
        crop_pct=1.0,
        test_input_size=(3, 320, 320),
    ),
    "resnet152": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth",
        interpolation="bicubic",
        crop_pct=0.95,
    ),
    "resnet152d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
        input_size=(3, 256, 256),
        pool_size=(8, 8),
        crop_pct=1.0,
        test_input_size=(3, 320, 320),
    ),
    "resnet200": _cfg(url="", interpolation="bicubic"),
    "resnet200d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
        input_size=(3, 256, 256),
        pool_size=(8, 8),
        crop_pct=1.0,
        test_input_size=(3, 320, 320),
    ),
    "wide_resnet50_2": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth",
        interpolation="bicubic",
    ),
    "wide_resnet101_2": _cfg(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"
    ),
    # ResNeXt
    "resnext50_32x4d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth",
        interpolation="bicubic",
        crop_pct=0.95,
    ),
    "resnext50d_32x4d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth",
        interpolation="bicubic",
        first_conv="conv1.0",
    ),
    "resnext101_32x4d": _cfg(url=""),
    "resnext101_32x8d": _cfg(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"
    ),
    "resnext101_64x4d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth",
        interpolation="bicubic",
        crop_pct=1.0,
        test_input_size=(3, 288, 288),
    ),
}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return (
        aa_layer(stride)
        if issubclass(aa_layer, nn.AvgPool2d)
        else aa_layer(channels=channels, stride=stride)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
        )
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(
            aa_layer, channels=first_planes, stride=stride, enable=use_aa
        )

        self.conv2 = nn.Conv2d(
            first_planes,
            outplanes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class FourierFilter(nn.Module):
    def __init__(self, dim, height, width, kernel_size=7):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.complex_weight = nn.Parameter(
            torch.randn(
                [dim, kernel_size, kernel_size // 2 + 1, 2], dtype=torch.float32
            )
            * 0.2
        )
        self.register_buffer("mask", self.get_mask(self.height, self.width))

    def get_mask(self, h, w):
        index_x = torch.min(torch.arange(0.1, h), torch.arange(h, 0, -1))
        index_y = torch.arange(w)
        mask = torch.max(index_x.unsqueeze(1), index_y.unsqueeze(0))
        mask = mask < self.kernel_size / 2 + 0.1
        return mask

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x.float(), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        # print(x[0].shape, self.mask.shape, self.mask.sum(), weight.shape)
        weight = torch.zeros_like(x[0]).masked_scatter(self.mask, weight)
        x = x * (1 + weight)
        x = torch.fft.irfft2(x, s=(H, W), norm="ortho")
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, height={self.height}, width={self.width}, kernel_size={self.kernel_size}"


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_layer=nn.GELU,
        act2_layer=nn.Identity,
        bias=False,
        feature_resolution=7,
        kernel_size=7,
        padding=3,
        stride=1,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=bias)
        self.filterconv1 = nn.Conv2d(dim, med_channels, 1, bias=bias)
        self.act1 = act1_layer()

        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.filter = FourierFilter(
            med_channels, feature_resolution, feature_resolution // 2 + 1
        )
        self.pool = (
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()
        )

        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(med_channels, dim, 1, bias=bias)

    def forward(self, x):
        x1 = self.pwconv1(x)
        x1 = self.act1(x1)
        x1 = self.dwconv(x1)
        x1 = self.pool(x1)

        x2 = self.filterconv1(x)
        x2 = self.act1(x2)
        x2 = self.filter(x2)
        x2 = self.pool(x2)

        x = x1 * x2
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        feature_resolution=7,
    ):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        assert first_planes == width
        self.conv2 = SepConv(
            first_planes, feature_resolution=feature_resolution, stride=1 if use_aa else stride  # no group, no dilation
        )
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=p,
                dilation=first_dilation,
                bias=False,
            ),
            norm_layer(out_channels),
        ]
    )


def downsample_avg(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = (
            AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        )
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[
            pool,
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
        ]
    )


def drop_blocks(drop_prob=0.0):
    return [
        None,
        None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25)
        if drop_prob
        else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00)
        if drop_prob
        else None,
    ]


def make_blocks(
    block_fn,
    channels,
    block_repeats,
    inplanes,
    reduce_first=1,
    output_stride=32,
    down_kernel_size=1,
    avg_down=False,
    drop_block_rate=0.0,
    drop_path_rate=0.0,
    feature_resolutions=[56, 56, 28, 14],
    **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(
        zip(channels, block_repeats, drop_blocks(drop_block_rate))
    ):
        stage_name = f"layer{stage_idx + 1}"  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get("norm_layer"),
            )
            downsample = (
                downsample_avg(**down_kwargs)
                if avg_down
                else downsample_conv(**down_kwargs)
            )
        feature_resolution = feature_resolutions[stage_idx]

        block_kwargs = dict(
            reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs
        )
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = (
                drop_path_rate * net_block_idx / (net_num_blocks - 1)
            )  # stochastic depth linear decay rule
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None,
                    feature_resolution=feature_resolution,
                    **block_kwargs,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
            feature_resolution = feature_resolution // stride

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(
            dict(num_chs=inplanes, reduction=net_stride, module=stage_name)
        )

    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        global_pool="avg",
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        replace_stem_pool=False,
        block_reduce_first=1,
        down_kernel_size=1,
        avg_down=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=0.0,
        zero_init_last=True,
        block_args=None,
    ):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False
                    ),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(
                        stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False
                    ),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(
                        stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False
                    ),
                ]
            )
        else:
            self.conv1 = nn.Conv2d(
                in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module="act1")]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv2d(
                            inplanes,
                            inplanes,
                            3,
                            stride=1 if aa_layer else 2,
                            padding=1,
                            bias=False,
                        ),
                        create_aa(aa_layer, channels=inplanes, stride=2)
                        if aa_layer is not None
                        else None,
                        norm_layer(inplanes),
                        act_layer(inplace=True),
                    ],
                )
            )
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(
                        *[
                            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                            aa_layer(channels=inplanes, stride=2),
                        ]
                    )
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        feature_resolutions = [56, 56, 28, 14]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            feature_resolutions=feature_resolutions,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, "zero_init_last"):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^conv1|bn1|maxpool",
            blocks=r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)",
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True
            )
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def resnet10t(pretrained=False, **kwargs):
    """Constructs a ResNet-10-T model."""
    model_args = dict(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet10t", pretrained, **model_args)


@register_model
def resnet14t(pretrained=False, **kwargs):
    """Constructs a ResNet-14-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[1, 1, 1, 1],
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet14t", pretrained, **model_args)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet("resnet18", pretrained, **model_args)


@register_model
def resnet18d(pretrained=False, **kwargs):
    """Constructs a ResNet-18-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet18d", pretrained, **model_args)


@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet("resnet34", pretrained, **model_args)


@register_model
def resnet34d(pretrained=False, **kwargs):
    """Constructs a ResNet-34-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet34d", pretrained, **model_args)


@register_model
def resnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model."""
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet("resnet26", pretrained, **model_args)


@register_model
def resnet26t(pretrained=False, **kwargs):
    """Constructs a ResNet-26-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet26t", pretrained, **model_args)


@register_model
def resnet26d(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet26d", pretrained, **model_args)


@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet("resnet50", pretrained, **model_args)


@register_model
def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet50d", pretrained, **model_args)


@register_model
def resnet50t(pretrained=False, **kwargs):
    """Constructs a ResNet-50-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet50t", pretrained, **model_args)


@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet("resnet101", pretrained, **model_args)


@register_model
def resnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet101d", pretrained, **model_args)


@register_model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet("resnet152", pretrained, **model_args)


@register_model
def resnet152d(pretrained=False, **kwargs):
    """Constructs a ResNet-152-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet152d", pretrained, **model_args)


@register_model
def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model."""
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], **kwargs)
    return _create_resnet("resnet200", pretrained, **model_args)


@register_model
def resnet200d(pretrained=False, **kwargs):
    """Constructs a ResNet-200-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnet200d", pretrained, **model_args)


@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128, **kwargs)
    return _create_resnet("wide_resnet50_2", pretrained, **model_args)


@register_model
def wide_resnet101_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128, **kwargs)
    return _create_resnet("wide_resnet101_2", pretrained, **model_args)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs
    )
    return _create_resnet("resnext50_32x4d", pretrained, **model_args)


@register_model
def resnext50d_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample"""
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    )
    return _create_resnet("resnext50d_32x4d", pretrained, **model_args)


@register_model
def resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs
    )
    return _create_resnet("resnext101_32x4d", pretrained, **model_args)


@register_model
def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs
    )
    return _create_resnet("resnext101_32x8d", pretrained, **model_args)


@register_model
def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt101-64x4d model."""
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4, **kwargs
    )
    return _create_resnet("resnext101_64x4d", pretrained, **model_args)


if __name__ == "__main__":
    net = resnet50()
    x = torch.rand(2, 3, 224, 224)
    y = net(x)
    print(y.shape)
