# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
# from abc import ABCMeta, abstractmethod
# from inspect import signature
# from typing import List, Optional, Tuple

# import torch
# from mmcv.ops import batched_nms
# from mmengine.config import ConfigDict
# from mmengine.model import BaseModule, constant_init
# from mmengine.structures import InstanceData
# from torch import Tensor

# from mmdet.structures import SampleList
# from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
#                                    scale_boxes)
# from mmdet.utils import InstanceList, OptMultiConfig
# from ..test_time_augs import merge_aug_results
# from ..utils import (filter_scores_and_topk, select_single_mlvl,
#                      unpack_gt_instances)

# # Copyright (c) OpenMMLab. All rights reserved.
# import warnings
# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn as nn
# from mmengine.structures import InstanceData
# from torch import Tensor

# from mmdet.registry import MODELS, TASK_UTILS
# from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
# from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
#                          OptInstanceList, OptMultiConfig)
# from ..task_modules.prior_generators import (AnchorGenerator,
#                                              anchor_inside_flags)
# from ..task_modules.samplers import PseudoSampler
# from ..utils import images_to_levels, multi_apply, unmap
# from .base_dense_head import BaseDenseHead

# # Copyright (c) OpenMMLab. All rights reserved.
# import torch.nn as nn
# from mmcv.cnn import ConvModule

# from mmdet.registry import MODELS
# from .anchor_head import AnchorHead


import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import InstanceList, OptInstanceList

from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import (images_to_levels, multi_apply, unmap,
                                filter_scores_and_topk, select_single_mlvl)
from mmdet.models.dense_heads import AnchorHead


import os
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath

from projects.ViTDet.vitdet.vit import window_partition, window_unpartition, Attention, Mlp
from mmdet.models.backbones.vheat_models.vHeat import LayerNorm2d, HeatBlock
from mmdet.models.backbones.vheat_models.vHeatK import HeatKBlock
from timm.models.layers import trunc_normal_


class BCHW2BHWC(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class BHWC2BCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class LayerNorm2d_ex(nn.Module):

    def __init__(
        self,
        num_features,
        eps=1e-6,
        input_mode='BCHW',
        output_mode=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps)
        self.input_mode = input_mode
        self.output_mode = output_mode if (output_mode is not None) else input_mode

    def forward(self, x):
        if self.input_mode == 'BCHW': x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        if self.output_mode == 'BCHW': x = x.permute(0, 3, 1, 2)
        return x


class F_Interpolate(nn.Module):

    def __init__(
        self,
        size=None,
        mode='nearest',
    ):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        with_cp=True,
    ):
        super().__init__()
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            return self.conv(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class AttnBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
        with_cp=True,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_cfg=act_cfg)

        self.window_size = window_size
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.norm1(x)
            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)
            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@MODELS.register_module()
class HEPRetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 # 
                 mmt_min=0.0,
                 mmt_max=1.2,
                 mmt_base=1.0,
                 mmt_mean=0.0,
                 mmt_std=1.0,
                 mmt_encode_mode='base',
                 loss_mmt_reg=dict(type='L1Loss', loss_weight=1.0),
                 mmt_reg_channels=1,
                 # 
                 # loss_mmt_label=dict(
                 #     type='FocalLoss',
                 #     use_sigmoid=True,
                 #     gamma=2.0,
                 #     alpha=0.25,
                 #     loss_weight=1.0),
                 loss_mmt_label=None,
                 mmt_label_channels=12,
                 # 
                 mmt_use_fpn=False,     # 使用局部注意力回归动量
                 mmt_use_gloattn=True,  # 使用全局注意力回归动量
                 mmt_in_channels=768,
                 hw_shape=[15, 30],
                 stacked_blocks=2,
                 block_type='HeatKBlock',
                 feat_fusion_mode='cat',
                 drop_path=0.1,
                 mlp_ratio=4.0,
                 post_norm=False,
                 layer_scale=None,
                 # pretrained=None,
                 # pretrained_src=None,
                 with_cp=True,
                 **kwargs):
        assert stacked_convs >= 0, \
            '`stacked_convs` must be non-negative integers, ' \
            f'but got {stacked_convs} instead.'
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        self.mmt_base = mmt_base
        self.mmt_mean = mmt_mean
        self.mmt_std = mmt_std
        self.mmt_encode_mode = mmt_encode_mode
        self.use_sigmoid_mmt = loss_mmt_reg.get('use_sigmoid', False)
        self.use_mmt_reg = (loss_mmt_reg is not None)
        self.mmt_reg_channels = mmt_reg_channels

        self.use_mmt_label = (loss_mmt_label is not None)
        self.mmt_label_channels = mmt_label_channels

        self.mmt_use_fpn = mmt_use_fpn
        self.mmt_use_gloattn = mmt_use_gloattn
        self.mmt_in_channels = mmt_in_channels
        self.hw_shape = hw_shape
        self.stacked_blocks = stacked_blocks
        self.block_type = block_type if isinstance(block_type, str) else ''
        self.feat_fusion_mode = feat_fusion_mode
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.post_norm = post_norm
        self.layer_scale = layer_scale
        # self.pretrained = pretrained
        # self.pretrained_src = pretrained_src
        self.with_cp = with_cp

        self.eps = 1e-6

        super().__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        if self.use_mmt_reg:                                                                        # mmt
            self.loss_mmt_reg = MODELS.build(loss_mmt_reg)
        if self.use_mmt_label:                                                                      # mmt_label
            self.loss_mmt_label = MODELS.build(loss_mmt_label)

    # def load_pretrained(self, ckpt=""):
    #     _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
    #     print(f"Successfully load ckpt {ckpt}")

    #     if self.pretrained_src == 'simmim':
    #         freq_embed_name = 'neck.freq_embed'
    #         weight_name_prefix = 'neck.blocks.'
    #         len_prefix = len(weight_name_prefix)

    #         pretrained_freq_embed = _ckpt['model'][freq_embed_name]
    #         if pretrained_freq_embed.shape[:2] != self.freq_embed.shape[:2]:
    #             resized_freq_embed = pretrained_freq_embed.permute(2, 0, 1).contiguous().unsqueeze(0)
    #             resized_freq_embed = F.interpolate(
    #                 resized_freq_embed, size=(self.freq_embed.shape[0], self.freq_embed.shape[1]), mode='bicubic'
    #             ).squeeze().permute(1, 2, 0).contiguous()
    #         else:
    #             resized_freq_embed = pretrained_freq_embed
    #         print('freq_embed: {} -> {}'.format(pretrained_freq_embed.shape, resized_freq_embed.shape))

    #         self.freq_embed.data.copy_(resized_freq_embed.to(self.freq_embed.device))
    #         print('self.freq_embed: device: {}, requires_grad: {}'.format(self.freq_embed.device, self.freq_embed.requires_grad))

    #         new_weights = {}
    #         weights_keys = list(_ckpt['model'].keys())
    #         for k in range(self.stacked_blocks):
    #             for weight_name in weights_keys:
    #                 if weight_name[:len_prefix] == weight_name_prefix:
    #                     new_weights[weight_name[len_prefix:]] = _ckpt['model'][weight_name]
    #                     print('{} -> {}'.format(weight_name, weight_name[len_prefix:]))

    #     elif self.pretrained_src == 'backbone':
    #         weight_name_prefix = 'layers.3.1'
    #         len_prefix = len(weight_name_prefix)

    #         new_weights = {}
    #         weights_keys = list(_ckpt['model'].keys())
    #         for k in range(self.stacked_blocks):
    #             for weight_name in weights_keys:
    #                 if weight_name[:len_prefix] == weight_name_prefix:
    #                     new_weights[str(k) + weight_name[len_prefix:]] = _ckpt['model'][weight_name]
    #                     print('{} -> {}'.format(weight_name, str(k) + weight_name[len_prefix:]))

    #     else:
    #         new_weights = {}

    #     incompatibleKeys = self.mmt_reg_blocks.load_state_dict(new_weights, strict=False)
    #     print(incompatibleKeys)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            in_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors * self.bbox_coder.encode_size, 3, padding=1)

        # self.mmt_use_fpn只负责self.use_mmt_reg
        if self.use_mmt_reg and self.mmt_use_fpn:
            self.mmt_reg_convs = nn.ModuleList()

            for j in range(self.stacked_convs):
                self.mmt_reg_convs.append(
                    ConvModule(
                        in_channels,
                        in_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

            self.retina_mmt_reg = nn.Conv2d(
                in_channels, self.num_base_priors * self.mmt_reg_channels, 3, padding=1)

        # self.mmt_use_gloattn负责(self.use_mmt_reg or self.use_mmt_label)
        if (self.use_mmt_reg or self.use_mmt_label) and self.mmt_use_gloattn:
            self.mmt_reg_blocks = nn.ModuleList()
            mmt_in_channels = self.mmt_in_channels

            if self.stacked_blocks == 0:
                self.mmt_reg_blocks.append(nn.Identity())

            elif self.block_type == 'ConvBlock':
                for k in range(self.stacked_blocks):
                    self.mmt_reg_blocks.append(
                        ConvBlock(
                            mmt_in_channels,
                            mmt_in_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            with_cp=self.with_cp))

            elif self.block_type == 'AttnBlock':
                self.mmt_reg_blocks.append(BCHW2BHWC())     # FPN gives BCHW but AttnBlock uses BHWC

                for k in range(self.stacked_blocks):
                    self.mmt_reg_blocks.append(
                        AttnBlock(
                            dim=mmt_in_channels,
                            num_heads=mmt_in_channels // 32,
                            mlp_ratio=self.mlp_ratio,
                            drop_path=self.drop_path,
                            with_cp=self.with_cp,
                        ))

                self.mmt_reg_blocks.append(LayerNorm2d_ex(
                    num_features=mmt_in_channels, eps=1e-6, input_mode='BHWC', output_mode='BCHW'))

            elif self.block_type == 'HeatBlock':
                for k in range(self.stacked_blocks):
                    self.mmt_reg_blocks.append(
                        HeatBlock(
                            res=self.hw_shape[1],
                            hidden_dim=mmt_in_channels,
                            drop_path=self.drop_path,
                            norm_layer=LayerNorm2d,
                            use_checkpoint=self.with_cp,
                            mlp_ratio=self.mlp_ratio,
                            post_norm=self.post_norm,
                            layer_scale=self.layer_scale,
                            infer_mode=False,
                        ))
                self.mmt_reg_blocks.append(LayerNorm2d_ex(
                    num_features=mmt_in_channels, eps=1e-6, input_mode='BCHW', output_mode='BCHW'))

            elif self.block_type == 'HeatKBlock':
                for k in range(self.stacked_blocks):
                    self.mmt_reg_blocks.append(
                        HeatKBlock(
                            res=self.hw_shape[1],
                            hidden_dim=mmt_in_channels,
                            drop_path=self.drop_path,
                            norm_layer=LayerNorm2d,
                            use_checkpoint=self.with_cp,
                            mlp_ratio=self.mlp_ratio,
                            post_norm=self.post_norm,
                            layer_scale=self.layer_scale,
                            infer_mode=False,
                            feat_fusion_mode=self.feat_fusion_mode,
                        ))
                self.mmt_reg_blocks.append(LayerNorm2d_ex(
                    num_features=mmt_in_channels, eps=1e-6, input_mode='BCHW', output_mode='BCHW'))

            else:
                self.mmt_reg_blocks.append(nn.Identity())

            if 'Heat' in self.block_type and self.stacked_blocks > 0:
                self.freq_embed = nn.Parameter(
                    torch.zeros(self.hw_shape[0], self.hw_shape[1], mmt_in_channels),
                    requires_grad=True)
                trunc_normal_(self.freq_embed, std=.02)
            else:
                self.freq_embed = None

            if self.use_mmt_reg:
                self.mmt_reg_fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(mmt_in_channels, self.mmt_reg_channels),
                )
            if self.use_mmt_label:
                self.mmt_label_fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(mmt_in_channels, self.mmt_label_channels),
                )

            self.apply(self._init_weights)

            # if 'Heat' in self.block_type and self.stacked_blocks > 0 and self.pretrained is not None:
            #     assert os.path.exists(self.pretrained)
            #     self.load_pretrained(self.pretrained)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        if self.use_mmt_reg and self.mmt_use_fpn: mmt_reg_feat = x                                  # mmt

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        if self.use_mmt_reg and self.mmt_use_fpn:                                                   # mmt
            for mmt_reg_conv in self.mmt_reg_convs:
                mmt_reg_feat = mmt_reg_conv(mmt_reg_feat)
            mmt_reg_pred = self.retina_mmt_reg(mmt_reg_feat)
        else:
            mmt_reg_pred = None

        return cls_score, bbox_pred, mmt_reg_pred                                                   # mmt

    def forward(self, x):
        fpn_feats, backbone_feat = x

        cls_scores, bbox_preds, mmt_reg_preds_use_fpn = multi_apply(self.forward_single, fpn_feats)

        # self.mmt_use_gloattn负责(self.use_mmt_reg or self.use_mmt_label)
        if (self.use_mmt_reg or self.use_mmt_label) and self.mmt_use_gloattn:
            mmt_reg_feat = backbone_feat

            # # 调整self.freq_embed的大小（可忽略）
            # if self.freq_embed is None:
            #     resized_freq_embed = self.freq_embed
            # elif self.freq_embed.shape[:2] != mmt_reg_feat.shape[2:]:
            #     resized_freq_embed = self.freq_embed.permute(2, 0, 1).contiguous().unsqueeze(0)
            #     resized_freq_embed = F.interpolate(
            #         resized_freq_embed, size=(mmt_reg_feat.shape[2], mmt_reg_feat.shape[3]), mode='bicubic'
            #     ).squeeze().permute(1, 2, 0).contiguous()
            # else:
            #     resized_freq_embed = self.freq_embed

            # 回归单个mmt
            for block in self.mmt_reg_blocks:
                if 'Heat' in block.__class__.__name__:
                    mmt_reg_feat = block(mmt_reg_feat, self.freq_embed)
                else:
                    mmt_reg_feat = block(mmt_reg_feat)

            # 在维度上展开回归的mmt使其与featmap等大
            if self.use_mmt_reg:
                mmt_reg_preds_use_gloattn = self.mmt_reg_fc(mmt_reg_feat)
                mmt_reg_preds = []
                num_levels = len(cls_scores)
                for level in range(num_levels):
                    cls_score = cls_scores[level]
                    B, C, H, W = cls_score.shape
                    temp = mmt_reg_preds_use_gloattn[:, None, :, None, None].repeat(
                        1, self.num_base_priors, 1, H, W).flatten(1, 2)

                    if mmt_reg_preds_use_fpn[level] is not None:
                        # OLD: 如果FPN也回归了mmt，那么将两者相加
                        # temp += mmt_reg_preds_use_fpn[level]

                        # NEW: 如果FPN也回归了mmt，那么将cls_score最大值索引不等于0的那些mmt
                        #     替换为mmt_reg_preds_use_fpn[level]。适用于多目标检测
                        cls_score = cls_score.reshape(B, self.num_base_priors, self.cls_out_channels, H, W)
                        max_values, max_indices = cls_score.max(dim=2, keepdim=False)
                        replace_mask = (max_indices > 0)
                        temp[replace_mask] = mmt_reg_preds_use_fpn[level][replace_mask]

                    mmt_reg_preds.append(temp)
            else:
                mmt_reg_preds = mmt_reg_preds_use_fpn

            if self.use_mmt_label:
                mmt_label_scores_use_gloattn = self.mmt_label_fc(mmt_reg_feat)
                mmt_label_scores = []
                num_levels = len(cls_scores)
                for level in range(num_levels):
                    cls_score = cls_scores[level]
                    B, C, H, W = cls_score.shape
                    temp = mmt_label_scores_use_gloattn[:, None, :, None, None].repeat(
                        1, self.num_base_priors, 1, H, W).flatten(1, 2)
                    mmt_label_scores.append(temp)
            else:
                mmt_label_scores = [None, ] * len(cls_scores)
        else:
            mmt_reg_preds = mmt_reg_preds_use_fpn
            mmt_label_scores = [None, ] * len(cls_scores)

        return cls_scores, bbox_preds, mmt_reg_preds, mmt_label_scores

    def mmt_encode_base(self, mmt_gts) -> Tensor:
        return (torch.log(mmt_gts / self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_base(self, mmt_preds) -> Tensor:
        return torch.exp(mmt_preds * self.mmt_std + self.mmt_mean) * self.mmt_base

    def mmt_encode_direct(self, mmt_gts) -> Tensor:
        return ((mmt_gts - self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_direct(self, mmt_preds) -> Tensor:
        return (mmt_preds * self.mmt_std + self.mmt_mean) + self.mmt_base

    def mmt_encode_sigmoid(self, mmt_gts) -> Tensor:
        mmt_norm = torch.clamp((mmt_gts - self.mmt_min) / (self.mmt_max - self.mmt_min),
                               min = self.eps, max = 1 - self.eps)
        return mmt_norm

    def mmt_decode_sigmoid(self, mmt_preds) -> Tensor:
        return torch.sigmoid(mmt_preds) * (self.mmt_max - self.mmt_min) + self.mmt_min

    # ##### ##### ##### ##### ##### #####   from anchor_head.py   ##### ##### ##### ##### ##### ##### #


    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        if self.use_mmt_reg:                                                                        # mmt
            mmt_reg_targets = anchors.new_zeros(num_valid_anchors, self.mmt_reg_channels)
            mmt_reg_weights = anchors.new_zeros(num_valid_anchors, self.mmt_reg_channels)
        else:
            mmt_reg_targets = None
            mmt_reg_weights = None

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        if self.use_mmt_label:                                                                      # mmt_label
            mmt_labels = anchors.new_full((num_valid_anchors, ),
                                          self.mmt_label_channels,
                                          dtype=torch.long)
            mmt_label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        else:
            mmt_labels = None
            mmt_label_weights = None

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if self.use_mmt_reg:                                                                    # mmt
                # 由于loss_mmt_reg无法使用`IouLoss`, `GIouLoss`等损失函数，因此必须对gt预编码而非对pred预解码！
                if self.mmt_encode_mode == 'base':
                    pos_mmt_reg_targets = self.mmt_encode_base(sampling_result.pos_gt_mmt_regs)
                elif self.mmt_encode_mode == 'direct':
                    pos_mmt_reg_targets = self.mmt_encode_direct(sampling_result.pos_gt_mmt_regs)
                elif self.mmt_encode_mode == 'sigmoid':
                    pos_mmt_reg_targets = self.mmt_encode_sigmoid(sampling_result.pos_gt_mmt_regs)
                else:
                    raise NotImplementedError
                mmt_reg_targets[pos_inds, :] = pos_mmt_reg_targets
                mmt_reg_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']

            if self.use_mmt_label:                                                                  # mmt_label
                mmt_labels[pos_inds] = sampling_result.pos_gt_mmt_labels
                if self.train_cfg['pos_weight'] <= 0:
                    mmt_label_weights[pos_inds] = 1.0
                else:
                    mmt_label_weights[pos_inds] = self.train_cfg['pos_weight']

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_targets = unmap(mmt_reg_targets, num_total_anchors, inside_flags)
                mmt_reg_weights = unmap(mmt_reg_weights, num_total_anchors, inside_flags)
            if self.use_mmt_label:                                                                  # mmt_label
                mmt_labels = unmap(
                    mmt_labels, num_total_anchors, inside_flags,
                    fill=self.mmt_label_channels)  # fill bg label
                mmt_label_weights = unmap(
                    mmt_label_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, 
                mmt_reg_targets, mmt_reg_weights,                                                   # mmt
                mmt_labels, mmt_label_weights,                                                      # mmt_label
                pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  `PseudoSampler`, `avg_factor` is usually equal to the number
                  of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_mmt_reg_targets, all_mmt_reg_weights,                                                  # mmt
         all_mmt_labels, all_mmt_label_weights,                                                     # mmt_label
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:11]
        rest_results = list(results[11:])  # user-added return values
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        if self.use_mmt_reg:                                                                        # mmt
            mmt_reg_targets_list = images_to_levels(all_mmt_reg_targets,
                                                    num_level_anchors)
            mmt_reg_weights_list = images_to_levels(all_mmt_reg_weights,
                                                    num_level_anchors)
        else:
            mmt_reg_targets_list = [None, ] * len(bbox_targets_list)
            mmt_reg_weights_list = [None, ] * len(bbox_weights_list)

        if self.use_mmt_label:                                                                      # mmt_label
            mmt_labels_list         = images_to_levels(all_mmt_labels,
                                                       num_level_anchors)
            mmt_label_weights_list  = images_to_levels(all_mmt_label_weights,
                                                       num_level_anchors)
        else:
            mmt_labels_list         = [None, ] * len(labels_list)
            mmt_label_weights_list  = [None, ] * len(label_weights_list)

        res = (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
               mmt_reg_targets_list, mmt_reg_weights_list,                                          # mmt
               mmt_labels_list, mmt_label_weights_list,                                             # mmt_label
               avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            mmt_reg_pred: Optional[Tensor],                                         # mmt
                            mmt_label_score: Optional[Tensor],                                      # mmt_label
                            anchors: Tensor, labels: Tensor, label_weights: Tensor, bbox_targets: Tensor, bbox_weights: Tensor,
                            mmt_reg_targets: Optional[Tensor], mmt_reg_weights: Optional[Tensor],   # mmt
                            mmt_labels: Optional[Tensor], mmt_label_weights: Optional[Tensor],      # mmt_label
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)

        # mmt loss
        if self.use_mmt_reg:                                                                        # mmt
            mmt_reg_targets = mmt_reg_targets.reshape(-1, self.mmt_reg_channels)
            mmt_reg_weights = mmt_reg_weights.reshape(-1, self.mmt_reg_channels)
            mmt_reg_pred = mmt_reg_pred.permute(0, 2, 3, 1).reshape(-1, self.mmt_reg_channels)
            # 由于loss_mmt_reg无法使用`IouLoss`, `GIouLoss`等损失函数，因此必须对gt预编码而非对pred预解码！
            # 如果对gt进行了sigmoid预编码、但后续使用L1Loss/L2Loss等而非使用BCELoss，必须也对mmt_reg_pred预编码！
            if self.mmt_encode_mode == 'sigmoid' and not self.use_sigmoid_mmt:
                mmt_reg_pred = torch.sigmoid(mmt_reg_pred)
            loss_mmt_reg = self.loss_mmt_reg(
                mmt_reg_pred, mmt_reg_targets, mmt_reg_weights, avg_factor=avg_factor)
        else:
            loss_mmt_reg = torch.zeros(1, device=loss_bbox.device)

        if self.use_mmt_label:                                                                      # mmt_label
            mmt_labels = mmt_labels.reshape(-1)
            mmt_label_weights = mmt_label_weights.reshape(-1)
            mmt_label_score = mmt_label_score.permute(0, 2, 3, 1).reshape(-1, self.mmt_label_channels)
            loss_mmt_label = self.loss_mmt_label(
                mmt_label_score, mmt_labels, mmt_label_weights, avg_factor=avg_factor)
        else:
            loss_mmt_label = torch.zeros(1, device=loss_cls.device)

        return loss_cls, loss_bbox, loss_mmt_reg, loss_mmt_label

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            mmt_reg_preds: List[Optional[Tensor]],                                                  # mmt
            mmt_label_scores: List[Optional[Tensor]],                                               # mmt_label
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mmt_reg_targets_list, mmt_reg_weights_list,                                                # mmt
         mmt_labels_list, mmt_label_weights_list,                                                   # mmt_label
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_mmt_reg, losses_mmt_label = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            mmt_reg_preds,                                                                          # mmt
            mmt_label_scores,                                                                       # mmt_label
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mmt_reg_targets_list, mmt_reg_weights_list,                                             # mmt
            mmt_labels_list, mmt_label_weights_list,                                                # mmt_label
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, 
                    loss_mmt_reg=losses_mmt_reg,                                                    # mmt
                    loss_mmt_label=losses_mmt_label,                                                # mmt_label
                    )


    # ##### ##### ##### ##### ##### ##### from base_dense_head.py ##### ##### ##### ##### ##### ##### #


    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get(
            'sampling_results', None)
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.mmt_regs = sampling_result.pos_gt_mmt_regs                                     # mmt
            pos_info.mmt_labels = sampling_result.pos_gt_mmt_labels                                 # mmt
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    # def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict: pass

    # @abstractmethod
    # def loss_by_feat(self, **kwargs) -> dict: pass

    # def loss_and_predict(
    #     self,
    #     x: Tuple[Tensor],
    #     batch_data_samples: SampleList,
    #     proposal_cfg: Optional[ConfigDict] = None
    # ) -> Tuple[dict, InstanceList]: pass

    # def predict(self,
    #             x: Tuple[Tensor],
    #             batch_data_samples: SampleList,
    #             rescale: bool = False) -> InstanceList: pass

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        mmt_reg_preds: List[Optional[Tensor]],                                      # mmt
                        mmt_label_scores: List[Optional[Tensor]],                                   # mmt_label
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if self.use_mmt_reg: assert len(cls_scores) == len(mmt_reg_preds)                           # mmt
        if self.use_mmt_label: assert len(cls_scores) == len(mmt_label_scores)                      # mmt_label

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_pred_list = select_single_mlvl(mmt_reg_preds, img_id, detach=True)
            else:
                mmt_reg_pred_list = [None, ] * len(bbox_pred_list)
            if self.use_mmt_label:                                                                  # mmt_label
                mmt_label_score_list = select_single_mlvl(mmt_label_scores, img_id, detach=True)
            else:
                mmt_label_score_list = [None, ] * len(cls_score_list)

            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                mmt_reg_pred_list=mmt_reg_pred_list,                                                # mmt
                mmt_label_score_list=mmt_label_score_list,                                          # mmt_label
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                mmt_reg_pred_list: List[Optional[Tensor]],                          # mmt
                                mmt_label_score_list: List[Optional[Tensor]],                       # mmt_label
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_mmt_preds = []                                                                         # mmt
        mlvl_mmt_labels = []                                                                        # mmt_label
        mlvl_mmt_scores = []                                                                        # mmt_label
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred,
                        mmt_reg_pred,                                                               # mmt
                        mmt_label_score,                                                            # mmt_label
                        score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, 
                              mmt_reg_pred_list,                                                    # mmt
                              mmt_label_score_list,                                                 # mmt_label
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if self.use_mmt_reg: assert cls_score.size()[-2:] == mmt_reg_pred.size()[-2:]           # mmt
            if self.use_mmt_label: assert cls_score.size()[-2:] == mmt_label_score.size()[-2:]      # mmt_label

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_pred = mmt_reg_pred.permute(1, 2, 0).reshape(-1, self.mmt_reg_channels)
            else:
                mmt_reg_pred = torch.zeros((len(bbox_pred), self.mmt_reg_channels), device=bbox_pred.device)

            if self.use_mmt_label:                                                                  # mmt_label
                mmt_label_score = mmt_label_score.permute(1, 2, 0).reshape(-1, self.mmt_label_channels)
                mmt_score, mmt_label = torch.max(mmt_label_score.sigmoid(), dim=-1)
            else:
                mmt_label = torch.zeros((len(bbox_pred), ), dtype=torch.long, device=bbox_pred.device)
                mmt_score = torch.zeros((len(bbox_pred), ), dtype=torch.float, device=bbox_pred.device)

            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            mmt_reg_pred = mmt_reg_pred[keep_idxs]                                                  # mmt
            mmt_label = mmt_label[keep_idxs]                                                        # mmt_label
            mmt_score = mmt_score[keep_idxs]                                                        # mmt_label

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_mmt_preds.append(mmt_reg_pred)                                                     # mmt
            mlvl_mmt_labels.append(mmt_label)                                                       # mmt_label
            mlvl_mmt_scores.append(mmt_score)                                                       # mmt_label
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        mmt_pred = torch.cat(mlvl_mmt_preds)                                                        # mmt
        if not self.use_mmt_reg:
            mmts = mmt_pred
        elif self.mmt_encode_mode == 'base':
            mmts = self.mmt_decode_base(mmt_pred)
        elif self.mmt_encode_mode == 'direct':
            mmts = self.mmt_decode_direct(mmt_pred)
        elif self.mmt_encode_mode == 'sigmoid':
            mmts = self.mmt_decode_sigmoid(mmt_pred)
        else:
            raise NotImplementedError

        results = InstanceData()
        results.bboxes = bboxes
        results.mmts = mmts                                                                         # mmt
        results.mmt_labels = torch.cat(mlvl_mmt_labels)                                             # mmt_label
        results.mmt_scores = torch.cat(mlvl_mmt_scores)                                             # mmt_label
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        # NEW!
        w_shift = img_meta.get('w_shift', 0)
        if w_shift > 0:
            img_shape = img_meta.get('img_shape')
            assert w_shift == img_shape[1]

            r_shift_px = img_meta.get('r_shift_px')
            assert r_shift_px is not None

            l_shift_px = r_shift_px - w_shift

            bboxes = results.bboxes
            bboxes_x_ctr = (bboxes[:, 0] + bboxes[:, 2]) * 0.5              # results['gt_bboxes']: xmin, ymin, xmax, ymax
            r_ind = ((bboxes_x_ctr - r_shift_px) >= 0)                      # 判断向右平移的逆过程是否未越界

            bboxes[r_ind, 0::2] -= r_shift_px
            bboxes[~r_ind, 0::2] -= l_shift_px
            results.bboxes = bboxes

        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results

    # def aug_test(self,
    #              aug_batch_feats,
    #              aug_batch_img_metas,
    #              rescale=False,
    #              with_ori_nms=False,
    #              **kwargs): pass
