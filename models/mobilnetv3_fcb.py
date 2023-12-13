""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.efficientnet_blocks import SqueezeExcite, InvertedResidual
from timm.models.efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.models.features import FeatureInfo, FeatureHooks
from timm.models.helpers import build_model_with_cfg, pretrained_cfg_for_features, checkpoint_seq
from timm.models.layers import SelectAdaptivePool2d, Linear, create_conv2d, get_act_fn, get_norm_act_layer
from timm.models.registry import register_model

__all__ = ['MobileNetV3', 'MobileNetV3Features']


from timm.models.efficientnet_builder import _log_info_if
from timm.models.efficientnet_blocks import num_groups, CondConvResidual, DepthwiseSeparableConv, EdgeResidual, ConvBnAct
from timm.models.layers import make_divisible, DropPath


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
        mask = self.get_mask(H, W // 2 + 1).to(x.device)
        # print(x[0].shape, self.mask.shape, self.mask.sum(), weight.shape)
        weight = torch.zeros_like(x[0]).masked_scatter(mask, weight)
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
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        out_channels = out_channels or dim
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
        self.pwconv2 = nn.Conv2d(med_channels, out_channels, 1, bias=bias)

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


class FCBInvertedResidual(InvertedResidual):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super().__init__(in_chs, out_chs, dw_kernel_size, stride, dilation, group_size, pad_type,
            noskip, exp_ratio, exp_kernel_size, pw_kernel_size, act_layer,
            norm_layer, se_layer, conv_kwargs, drop_path_rate)
        self.fourier = True if self.has_skip else False
        if self.fourier:
            exp_ratio *= 1.25
            norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            conv_kwargs = conv_kwargs or {}
            mid_chs = make_divisible(in_chs * exp_ratio)
            groups = num_groups(group_size, mid_chs)

            # Point-wise expansion
            self.conv_pw = create_conv2d(in_chs, mid_chs * 2, exp_kernel_size, padding=pad_type, **conv_kwargs)
            self.bn1 = norm_act_layer(mid_chs * 2, inplace=True)

            # Depth-wise convolution
            # self.conv_dw = create_conv2d(
            #     mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            #     groups=groups, padding=pad_type, **conv_kwargs)
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, kernel_size=7, stride=stride, padding=3,
                groups=mid_chs, bias=False,
            )
            self.filter = FourierFilter(mid_chs, 7, 4)
            self.bn2 = norm_act_layer(mid_chs, inplace=True)

            # # Squeeze-and-excitation
            # self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()
            self.se = nn.Identity()

            # Point-wise linear projection
            self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
            self.bn3 = nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre')
        else:  # location == 'bottleneck', block output
            return dict(module='', hook_type='')

    def forward(self, x):
        if not self.fourier:
            return super().forward(x)
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x1, x2 = x.chunk(2, 1)
        x1 = self.conv_dw(x1)
        x2 = self.filter(x2)
        x = x1 * x2
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        x = self.drop_path(x) + shortcut
        return x


class FCBEfficientNetBuilder(EfficientNetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        if 'force_in_chs' in ba and ba['force_in_chs']:
            # NOTE this is a hack to work around mismatch in TF EdgeEffNet impl
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    # adjust se_ratio by expansion ratio if calculating se channels from block input
                    se_ratio /= ba.get('exp_ratio', 1.0)
                if self.se_has_ratio:
                    ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
                else:
                    ba['se_layer'] = self.se_layer

        if bt == 'ir':
            _log_info_if('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = CondConvResidual(**ba) if ba.get('num_experts', 0) else FCBInvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            _log_info_if('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt

        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenetv3_large_075': _cfg(url=''),
    'mobilenetv3_large_100': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
    'mobilenetv3_large_100_miil': _cfg(
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.),
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth'),
    'mobilenetv3_large_100_miil_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_in21k_miil-d71cc17b.pth',
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.), num_classes=11221),

    'mobilenetv3_small_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth',
        interpolation='bicubic'),

    'mobilenetv3_rw': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth',
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95, pool_size=(8, 8)),

    "lcnet_035": _cfg(),
    "lcnet_050": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth',
        interpolation='bicubic',
    ),
    "lcnet_075": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth',
        interpolation='bicubic',
    ),
    "lcnet_100": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth',
        interpolation='bicubic',
    ),
    "lcnet_150": _cfg(),
}


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(
            self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280,
            head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True,
            round_chs_fn=round_channels, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = FCBEfficientNetBuilder(
            output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.flatten(x)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(
            self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3,
            stem_size=16, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels,
            se_from_exp=True, act_layer=None, norm_layer=None, se_layer=None, drop_rate=0., drop_path_rate=0.):
        super(MobileNetV3Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = FCBEfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer,
            drop_path_rate=drop_path_rate, feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_mnv3(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = MobileNetV3
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'head_bias', 'global_pool')
        model_cls = MobileNetV3Features
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid'),
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetv3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNetV3
    Paper: `FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining`
        - https://arxiv.org/abs/2006.02049
    FIXME untested, this is a preliminary impl of some FBNet-V3 variants.
    """
    vl = variant.split('_')[-1]
    if vl in ('a', 'b'):
        stem_size = 16
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k5_s2_e4_c24', 'ir_r3_k5_s1_e2_c24'],
            ['ir_r1_k5_s2_e5_c40_se0.25', 'ir_r4_k5_s1_e3_c40_se0.25'],
            ['ir_r1_k5_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c120_se0.25', 'ir_r5_k5_s1_e3_c120_se0.25'],
            ['ir_r1_k3_s2_e6_c184_se0.25', 'ir_r5_k5_s1_e4_c184_se0.25', 'ir_r1_k5_s1_e6_c224_se0.25'],
            ['cn_r1_k1_s1_c1344'],
        ]
    elif vl == 'd':
        stem_size = 24
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k3_s2_e5_c24', 'ir_r5_k3_s1_e2_c24'],
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r4_k3_s1_e3_c40_se0.25'],
            ['ir_r1_k3_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c128_se0.25', 'ir_r6_k5_s1_e3_c128_se0.25'],
            ['ir_r1_k3_s2_e6_c208_se0.25', 'ir_r5_k5_s1_e5_c208_se0.25', 'ir_r1_k5_s1_e6_c240_se0.25'],
            ['cn_r1_k1_s1_c1440'],
        ]
    elif vl == 'g':
        stem_size = 32
        arch_def = [
            ['ds_r3_k3_s1_e1_c24'],
            ['ir_r1_k5_s2_e4_c40', 'ir_r4_k5_s1_e2_c40'],
            ['ir_r1_k5_s2_e4_c56_se0.25', 'ir_r4_k5_s1_e3_c56_se0.25'],
            ['ir_r1_k5_s2_e5_c104', 'ir_r4_k3_s1_e3_c104'],
            ['ir_r1_k3_s1_e5_c160_se0.25', 'ir_r8_k5_s1_e3_c160_se0.25'],
            ['ir_r1_k3_s2_e6_c264_se0.25', 'ir_r6_k5_s1_e5_c264_se0.25', 'ir_r2_k5_s1_e6_c288_se0.25'],
            ['cn_r1_k1_s1_c1728'],
        ]
    else:
        raise NotImplemented
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.95)
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=round_chs_fn)
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1984,
        head_bias=False,
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        se_from_exp=False,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


@register_model
def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100_miil(pretrained=False, **kwargs):
    """ MobileNet V3
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100_miil_in21k(pretrained=False, **kwargs):
    """ MobileNet V3, 21k pretraining
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil_in21k', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_050(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet V3 """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_b(pretrained=False, **kwargs):
    """ FBNetV3-B """
    model = _gen_fbnetv3('fbnetv3_b', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_d(pretrained=False, **kwargs):
    """ FBNetV3-D """
    model = _gen_fbnetv3('fbnetv3_d', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_g(pretrained=False, **kwargs):
    """ FBNetV3-G """
    model = _gen_fbnetv3('fbnetv3_g', pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_035(pretrained=False, **kwargs):
    """ PP-LCNet 0.35"""
    model = _gen_lcnet('lcnet_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_050(pretrained=False, **kwargs):
    """ PP-LCNet 0.5"""
    model = _gen_lcnet('lcnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_075(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_100(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_150(pretrained=False, **kwargs):
    """ PP-LCNet 1.5"""
    model = _gen_lcnet('lcnet_150', 1.5, pretrained=pretrained, **kwargs)
    return model


if __name__ == "__main__":
    net = mobilenetv3_large_100()
    print(sum(p.numel() for p in net.parameters()))
    x = torch.rand(2, 3, 224, 224)
    y = net(x)
    print(y.shape)
