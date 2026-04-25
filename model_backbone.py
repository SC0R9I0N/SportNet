import math
import os
from typing import Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from typing import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath

import cv2

# =========================
#  MogaNet backbone
# =========================

def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        x = self.feat_decompose(x)
        g = self.gate(x)
        v = self.value(x)
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class MogaNet(nn.Module):
    arch_zoo = {
        **dict.fromkeys(['xt', 'x-tiny', 'xtiny'],
                        {'embed_dims': [32, 64, 96, 192],
                         'depths': [3, 3, 10, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [3, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 160, 320, 512],
                         'depths': [4, 6, 22, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 160, 320, 640],
                         'depths': [4, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['xl', 'x-large', 'xlarge'],
                        {'embed_dims': [96, 192, 480, 960],
                         'depths': [6, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 num_classes=1000,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 head_init_scale=1.,
                 patch_sizes=[3, 3, 3, 3],
                 stem_norm_type='BN',
                 conv_norm_type='BN',
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv'],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 fork_feat=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo)
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == 'LN'
        assert len(patchembed_types) == self.num_stages
        self.fork_feat = fork_feat
        self.frozen_stages = frozen_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type='GELU',
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_type=conv_norm_type,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_type=attn_act_type,
                    attn_force_fp32=attn_force_fp32,
                ) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

        if self.fork_feat:
            self.head = nn.Identity()
        else:
            self.num_classes = num_classes
            self.head = nn.Linear(self.embed_dims[-1], num_classes) \
                if num_classes > 0 else nn.Identity()

            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.init_cfg = init_cfg

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                              blocks.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)
            if self.fork_feat:
                outs.append(x)

        if self.fork_feat:
            return outs
        else:
            return x

    def forward_head(self, x):
        return self.head(x.mean(dim=[2, 3]))

    def forward(self, x):
        x = self.forward_features(x)
        if self.fork_feat:
            return x
        else:
            return self.forward_head(x)


class MogaNetFeat(MogaNet):
    """Backbone for dense prediction (fork_feat=True)."""
    def __init__(self, **kwargs):
        super().__init__(fork_feat=True, **kwargs)


# =========================
#  Heatmap head (Simple)
# =========================

class HeatmapHead(nn.Module):
    """
    Simplified version of MMPose HeatmapHead, configured to behave like
    TopdownHeatmapSimpleHead (no deconv, just 1x1 conv).
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: Optional[Sequence[int]] = None,
                 deconv_kernel_sizes: Optional[Sequence[int]] = None,
                 conv_out_channels: Optional[Sequence[int]] = None,
                 conv_kernel_sizes: Optional[Sequence[int]] = None,
                 final_layer: dict = dict(kernel_size=1)):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'have the same length.')
            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'have the same length.')
            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = nn.Conv2d(
                in_channels=cfg['in_channels'],
                out_channels=cfg['out_channels'],
                kernel_size=cfg.get('kernel_size', 1),
                stride=cfg.get('stride', 1),
                padding=cfg.get('padding', 0),
                bias=cfg.get('bias', True),
            )
        else:
            self.final_layer = nn.Identity()

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size}')
            layers.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        # Ported from MMPose HeatmapHead to support old TopdownHeatmapSimpleHead
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3 and isinstance(self.conv_layers, nn.Sequential):
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    k_new = k
            else:
                k_new = k
            state_dict[prefix + k_new] = v


# =========================
#  Top-down pose model
# =========================

class TopDownPoseModel(nn.Module):
    def __init__(self, backbone: nn.Module, keypoint_head: nn.Module, in_index: int = 3):
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.in_index = in_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # list of 4 feature maps
        feat = feats[self.in_index]
        heatmaps = self.keypoint_head((feat,))
        return heatmaps


# =========================
#  Inference utilities
# =========================

CHECKPOINT_PATH = "moganet_b_ap2d_384x288.pth"
INPUT_HEIGHT = 384
INPUT_WIDTH = 288
NUM_KEYPOINTS = 17

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[TopDownPoseModel] = None


def load_model() -> TopDownPoseModel:
    global _model
    if _model is not None:
        return _model

    backbone = MogaNetFeat(arch='base', in_channels=3)

    # Match the checkpoint: 3 deconv layers (512→256→256→256) + final 1x1 conv
    head = HeatmapHead(
        in_channels=512,
        out_channels=NUM_KEYPOINTS,
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=None,
        conv_kernel_sizes=None,
        final_layer=dict(kernel_size=1, bias=True),
    )
    model = TopDownPoseModel(backbone, head, in_index=3)

    import numpy as np

    ckpt = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Load with strict=True to catch any mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys: {unexpected}")

    model.to(_device)
    model.eval()
    _model = model
    return model

# =========================
#  BBox + ID utilities
# =========================

DET_ROOT = "pose_2d/det_result"
SEQ_LEN = 17388  # frames per sequence


def load_det_results(det_json_path: str) -> dict:
    with open(det_json_path, "r") as f:
        data = json.load(f)
    return {entry["image_id"]: entry["bbox"] for entry in data}


def filename_to_image_id(filename: str) -> int:
    # Simply convert filename to integer, which removes leading zeros
    # 00000000001.jpg -> 1
    # 00000018260.jpg -> 18260
    # 10000000001.jpg -> 10000000001
    raw = int(os.path.splitext(filename)[0])
    return raw


def get_bbox_for_image(image_path: str) -> List[float]:
    filename = os.path.basename(image_path)
    image_id = filename_to_image_id(filename)

    parent = os.path.basename(os.path.dirname(image_path))
    if "train" in parent:
        det_file = os.path.join(DET_ROOT, "ap2d_train_det.json")
    elif "valid" in parent:
        det_file = os.path.join(DET_ROOT, "ap2d_valid_det.json")
    elif "test" in parent:
        det_file = os.path.join(DET_ROOT, "ap2d_test_det.json")
    else:
        raise ValueError(f"Cannot determine dataset split from path: {image_path}")

    det_map = load_det_results(det_file)

    if image_id not in det_map:
        raise KeyError(f"image_id {image_id} not found in {det_file}")

    return det_map[image_id]  # [x, y, w, h]

# =========================
#  Affine + Crop
# =========================

def get_third_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size):
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = np.array([0, src_w * -0.5], np.float32)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = [src_dir[0] * cs - src_dir[1] * sn,
               src_dir[0] * sn + src_dir[1] * cs]

    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_third_point(src[0, :], src[1, :])
    dst[2:, :] = get_third_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def crop_and_resize(img: Image.Image, center, scale):
    trans = get_affine_transform(center, scale, 0, (INPUT_WIDTH, INPUT_HEIGHT))
    img_np = np.array(img)
    cropped = cv2.warpAffine(
        img_np,
        trans,
        (INPUT_WIDTH, INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR
    )
    return cropped, trans

def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    x, y, w, h = get_bbox_for_image(image_path)
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    # Adjust bbox scale to target aspect ratio (width:height = 288:384 = 0.75)
    # Then apply padding
    aspect_ratio = INPUT_WIDTH / INPUT_HEIGHT  # 288/384 = 0.75
    if w / h > aspect_ratio:
        # bbox is too wide, increase height
        h = w / aspect_ratio
    else:
        # bbox is too tall, increase width
        w = h * aspect_ratio

    padding = 1.25
    scale = np.array([w / 200.0 * padding, h / 200.0 * padding], dtype=np.float32)

    cropped, trans = crop_and_resize(img, center, scale)

    cropped = cropped.astype(np.float32) / 255.0
    cropped = (cropped - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
    cropped = cropped.transpose(2, 0, 1)

    tensor = torch.from_numpy(cropped).unsqueeze(0).float()
    return tensor.to(_device), center, scale, trans, (orig_w, orig_h), cropped

def decode_heatmaps(heatmaps: torch.Tensor,
                    center: np.ndarray,
                    scale: np.ndarray,
                    trans: np.ndarray) -> np.ndarray:

    heatmaps = heatmaps.detach().cpu().numpy()
    N, K, H, W = heatmaps.shape
    assert N == 1

    keypoints = np.zeros((K, 3), dtype=np.float32)
    inv_trans = cv2.invertAffineTransform(trans)

    for k in range(K):
        hm = heatmaps[0, k]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = idx[0], idx[1]
        conf = hm[y, x]

        # Apply sub-pixel refinement in heatmap space
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = hm[y, x + 1] - hm[y, x - 1]
            dy = hm[y + 1, x] - hm[y - 1, x]
            x = x + 0.25 * np.sign(dx)
            y = y + 0.25 * np.sign(dy)

        # Scale from heatmap space (72x96) to input/crop space (288x384)
        # INPUT_WIDTH=288, INPUT_HEIGHT=384, heatmap is H x W
        scale_x = INPUT_WIDTH / W   # 288 / 72 = 4
        scale_y = INPUT_HEIGHT / H  # 384 / 96 = 4
        x_crop = x * scale_x
        y_crop = y * scale_y

        # Transform from crop coordinates to original image coordinates
        pt = np.array([x_crop, y_crop], dtype=np.float32)
        x_img, y_img = affine_transform(pt, inv_trans)

        keypoints[k, 0] = x_img
        keypoints[k, 1] = y_img
        keypoints[k, 2] = conf

    return keypoints

# ------------ Helper Methods to show the keypoints that get generated on the image -------------------------

def draw_keypoints_on_crop(crop_img: np.ndarray, kpts: np.ndarray, save_path="debug_crop_kpts.jpg"):
    """
    crop_img: HxWx3 uint8 array (your cropped image)
    kpts: (17, 3) array in crop-space BEFORE inverse affine
    """
    img = crop_img.copy()

    for (x, y, conf) in kpts:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    Image.fromarray(img).save(save_path)
    print(f"[OK] Saved crop keypoint overlay → {save_path}")

def draw_keypoints_on_original(orig_img: np.ndarray, kpts: np.ndarray, save_path="debug_orig_kpts.jpg"):
    """
    orig_img: HxWx3 uint8 array (original image)
    kpts: (17, 3) array in original-image coordinates AFTER inverse affine
    """
    img = orig_img.copy()

    for (x, y, conf) in kpts:
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    Image.fromarray(img).save(save_path)
    print(f"[OK] Saved original keypoint overlay → {save_path}")

# ---------------------------- End of Helper Methods ----------------------------------

def get_keypoints(image_path: str) -> np.ndarray:
    model = load_model()

    # Preprocess → now also returns the normalized crop (CHW) for shape reference
    inp, center, scale, trans, orig_size, cropped_chw = preprocess_image(image_path)

    orig_img = np.array(Image.open(image_path).convert("RGB"))
    crop_img = (cropped_chw.transpose(1, 2, 0) * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN)
    crop_img = np.clip(crop_img * 255.0, 0, 255).astype(np.uint8)

    with torch.no_grad():
        heatmaps = model(inp)  # (1, 17, H_hm, W_hm)

    # 1. Decode into ORIGINAL image coordinates
    keypoints = decode_heatmaps(heatmaps, center, scale, trans)

    # 2. Decode into CROP-SPACE for debugging (and scale to crop size)
    heat_np = heatmaps[0].cpu().numpy()  # (17, H_hm, W_hm)
    K, H_hm, W_hm = heat_np.shape
    crop_kpts = []
    for k in range(K):
        hm = heat_np[k]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        conf = hm[y, x]

        # scale from heatmap space → crop space (288x384)
        scale_x = INPUT_WIDTH / W_hm  # 288 / 72 = 4
        scale_y = INPUT_HEIGHT / H_hm  # 384 / 96 = 4
        x_vis = x * scale_x
        y_vis = y * scale_y

        crop_kpts.append([x_vis, y_vis, conf])
    crop_kpts = np.array(crop_kpts)

    # 3. Save overlays
    DEBUG_VIS = False

    if DEBUG_VIS:
        draw_keypoints_on_crop(crop_img, crop_kpts, save_path="debug_crop_kpts.jpg")
        draw_keypoints_on_original(orig_img, keypoints, save_path="debug_orig_kpts.jpg")
        print("[OK] Saved debug_crop_kpts.jpg and debug_orig_kpts.jpg")
        print(next(model.parameters()).device)

    return keypoints

# ------------------- Helper Method to test the image cropping --------------------------

def debug_save_crop(image_path: str, save_path: str = "debug_crop.jpg"):
    img = Image.open(image_path).convert("RGB")

    # bbox: [x, y, w, h]
    x, y, w, h = get_bbox_for_image(image_path)

    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    # Adjust bbox scale to target aspect ratio
    aspect_ratio = INPUT_WIDTH / INPUT_HEIGHT  # 288/384 = 0.75
    if w / h > aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio

    padding = 1.25
    scale = np.array([w / 200.0 * padding, h / 200.0 * padding], dtype=np.float32)

    cropped, trans = crop_and_resize(img, center, scale)

    Image.fromarray(cropped).save(save_path)
    print(f"Saved crop to {save_path}")

# --------------------------- End of Helper Method ---------------------------

if __name__ == "__main__":
    kpts = get_keypoints("pose_2d/valid_set/20000000056.jpg")
    print("Keypoints shape:", kpts.shape)
    print(kpts)
    debug_save_crop("pose_2d/valid_set/20000000056.jpg")
