# # Copyright (c) OpenMMLab. All rights reserved.
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


from mmcv.cnn import build_norm_layer
from mmdet.utils import OptMultiConfig
from mmdet.models.backbones.hepv2_transformer import Block

from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader


@MODELS.register_module()
class HEPv3RetinaHead(AnchorHead):
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

    def __init__(
        self,
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
        # 
        mmt_in_channels: int = 768,
        mmt_use_fpn: bool = False,                          # ViC使用局部注意力回归动量
        mmt_use_gloattn: bool = True,                       # ViC使用全局注意力回归动量
        mmt_label_use_gloattn: bool = False,                # ViC使用全局注意力回归全局标签
        # 
        trans_cfg=dict(
            embed_dim=384, # 512,
            depth=8,
            num_heads=12, # 16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            with_cp=False,
        ),
        # 
        phithe_base: Union[float, List[float]] = 45.0,
        phithe_mean: float = 0.0,
        phithe_std: float = 1.0,
        easy_scale: Union[float, List[float]] = 10.0,
        # 
        mmt_min: float = 0.0,
        mmt_max: float = 1.2,
        mmt_base: float = 1.0,
        mmt_mean: float = 0.0,
        mmt_std: float = 1.0,
        mmt_encode_mode: str = 'base',
        mmt_reg_channels: int = 1,
        # 
        len_seq: int = 640,
        use_hit_token_for_mmt: bool = False,                # ANT使用hit_token回归动量
        use_mmt_token: bool = True,                         # ANT使用mmt_token回归动量
        use_mmt_token_for_label: bool = False,              # ANT使用mmt_token回归全局标签
        backbone_out_eng: bool = True,
        backbone_out_phithe: bool = False,
        # 
        loss_phithe_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_mmt_reg=dict(type='L1Loss', loss_weight=1.0),
        # 
        # loss_mmt_label=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        loss_mmt_label=None,
        mmt_label_channels: int = 12,
        # 
        nms_mode: str = 'phithe', # 'bbox',
        phithe_nms_thr: float = 9.0,
        score_thr: float = 0.0,
        max_per_img: int = 1,
        # 
        phithe_source: str = 'mix',
        mmt_source: str = 'mix',
        # 
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        assert stacked_convs >= 0, \
            '`stacked_convs` must be non-negative integers, ' \
            f'but got {stacked_convs} instead.'
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_classes = num_classes

        self.mmt_in_channels = mmt_in_channels
        self.mmt_use_fpn = mmt_use_fpn
        self.mmt_use_gloattn = mmt_use_gloattn
        self.mmt_label_use_gloattn = mmt_label_use_gloattn
        self.use_mmt_reg = mmt_use_fpn or mmt_use_gloattn

        self.trans_cfg = trans_cfg

        # self.phithe_pos_thresh = phithe_pos_thresh / 180 * torch.pi
        self.phithe_base = self.float_to_list(phithe_base, deg_to_rad=True)
        self.phithe_mean = phithe_mean
        self.phithe_std = phithe_std
        self.easy_scale = self.float_to_list(easy_scale, deg_to_rad=False)
        self.phithe_reg_channels = 2

        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        self.mmt_base = mmt_base
        self.mmt_mean = mmt_mean
        self.mmt_std = mmt_std
        self.mmt_encode_mode = mmt_encode_mode
        self.use_sigmoid_mmt = loss_mmt_reg.get('use_sigmoid', False)
        self.mmt_reg_channels = mmt_reg_channels

        self.len_seq = len_seq
        self.use_hit_token_for_mmt = use_hit_token_for_mmt
        self.use_mmt_token = use_mmt_token
        self.use_mmt_token_for_label = use_mmt_token_for_label
        self.backbone_out_eng = backbone_out_eng
        self.backbone_out_phithe = backbone_out_phithe

        self.use_mmt_label = (mmt_label_use_gloattn or use_mmt_token_for_label)
        self.mmt_label_channels = mmt_label_channels

        self.nms_mode = nms_mode
        self.phithe_nms_thr = phithe_nms_thr / 180 * torch.pi
        self.score_thr = score_thr
        self.max_per_img = max_per_img

        self.phithe_source = phithe_source
        self.mmt_source = mmt_source

        self.width = 960
        self.height = 480
        self.eps = 1e-6

        super().__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_phithe_reg = MODELS.build(loss_phithe_reg)
        self.loss_mmt_reg = MODELS.build(loss_mmt_reg)

        if self.use_mmt_label:
            self.loss_mmt_label = MODELS.build(loss_mmt_label)

        self.fp16_enabled = False

    def float_to_list(self, x, deg_to_rad=False):
        if isinstance(x, float) or isinstance(x, int):
            if deg_to_rad:
                return [x / 180 * torch.pi] * self.num_classes
            else:
                return [x] * self.num_classes
        elif isinstance(x, list) or isinstance(x, tuple):
            assert len(x) == self.num_classes
            if deg_to_rad:
                return [temp / 180 * torch.pi for temp in x]
            else:
                return x
        else:
            raise NotImplementedError

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

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            else:
                _state_dict = ckpt
            incompatibleKeys = self.load_state_dict(_state_dict, False)
            print(incompatibleKeys)

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

        if self.mmt_use_fpn:
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

        embed_dim = self.trans_cfg['embed_dim']
        depth = self.trans_cfg['depth']
        num_heads = self.trans_cfg['num_heads']
        mlp_ratio = self.trans_cfg['mlp_ratio']
        qkv_bias = self.trans_cfg['qkv_bias']
        drop_path_rate = self.trans_cfg['drop_path_rate']
        norm_cfg = self.trans_cfg['norm_cfg']
        act_cfg = self.trans_cfg['act_cfg']
        with_cp = self.trans_cfg['with_cp']

        in_channels = self.mmt_in_channels
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)

        if self.mmt_use_gloattn or self.mmt_label_use_gloattn:
            self.vic_embed = nn.Linear(in_channels, embed_dim, bias=True)

        if self.backbone_out_eng:
            self.decoder_embed_eng = nn.Linear(in_channels, embed_dim, bias=True)
        if self.backbone_out_phithe:
            self.decoder_embed_phithe = nn.Linear(in_channels, embed_dim, bias=True)

        dpr = [drop_path_rate] * depth  # [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
            ) for i in range(depth)
        ])

        self.decoder_norm = build_norm_layer(norm_cfg, embed_dim)[1]

        self.dense_cls        = nn.Linear(embed_dim, self.cls_out_channels)
        self.dense_phithe_reg = nn.Linear(embed_dim, self.phithe_reg_channels)

        if self.use_hit_token_for_mmt:
            self.dense_hit_mmt_reg = nn.Linear(embed_dim, self.mmt_reg_channels)
        if self.use_mmt_token:
            self.dense_mmt_reg = nn.Linear(embed_dim, self.mmt_reg_channels)
        if self.use_mmt_token_for_label:
            self.dense_mmt_label = nn.Linear(embed_dim, self.mmt_label_channels)

        if self.mmt_use_gloattn:
            self.mmt_reg_fc = nn.Linear(embed_dim, self.mmt_reg_channels)
        if self.mmt_label_use_gloattn:
            self.mmt_label_fc = nn.Linear(embed_dim, self.mmt_label_channels)

        self.apply(self._init_weights)

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
        if self.mmt_use_fpn: mmt_reg_feat = x                                  # mmt

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        if self.mmt_use_fpn:                                                   # mmt
            for mmt_reg_conv in self.mmt_reg_convs:
                mmt_reg_feat = mmt_reg_conv(mmt_reg_feat)
            mmt_reg_pred = self.retina_mmt_reg(mmt_reg_feat)
        else:
            mmt_reg_pred = None

        return cls_score, bbox_pred, mmt_reg_pred                                                   # mmt

    def forward(self, x):
        backbone_outs, backbone_outs_others, fpn_feats, backbone_feat = x

        feat = backbone_outs[-1]
        # assert feat.requires_grad
        feat = self.decoder_embed(feat)

        feat_main = feat[:, :-1, :]
        feat_mmt = feat[:, -1:, :]

        backbone_x = backbone_outs_others['x']
        # assert not backbone_x.requires_grad

        flags = backbone_x[..., 0:1]
        # x_eng = backbone_x[..., 1]
        # x_phi = backbone_x[..., 2]
        # x_the = backbone_x[..., 3]
        # x_time = backbone_x[..., 4]
        flags_0 = backbone_x[..., 0]

        B, N, C = feat_main.shape

        if self.backbone_out_eng:
            y_eng = backbone_outs_others['y_eng']
            # assert y_eng.requires_grad
            y_eng = self.decoder_embed_eng(y_eng)
        else:
            y_eng = 0.0

        if self.backbone_out_phithe:
            y_phithe = backbone_outs_others['y_phithe']
            # assert y_phithe.requires_grad
            y_phithe = self.decoder_embed_phithe(y_phithe)
        else:
            y_phithe = 0.0

        x = feat_main + y_eng + y_phithe

        flags = (flags % self.len_seq + self.eps).to(dtype=torch.long)          # 浮点数改整数以防出错
        attn_mask_main = (flags != flags.transpose(-2, -1)) * (-1e9)            # B, N, N
        attn_mask_mmt = (torch.abs(flags_0) < self.eps) * (-1e9)                # B, N

        if self.mmt_use_gloattn or self.mmt_label_use_gloattn:
            feat_vic = backbone_feat.flatten(2).transpose(-2, -1)
            feat_vic = self.vic_embed(feat_vic)
            B, HW, C = feat_vic.shape

            x = torch.cat([x, feat_mmt, feat_vic], dim=1)                       # B, N + 1 + HW, C
            attn_mask = torch.zeros(B, N + 1 + HW, N + 1 + HW, device=x.device, requires_grad=False)
        else:
            x = torch.cat([x, feat_mmt], dim=1)                                 # B, N + 1, C
            attn_mask = torch.zeros(B, N + 1, N + 1, device=x.device, requires_grad=False)

        attn_mask[:, :N, :N] = attn_mask_main
        attn_mask[:,  N, :N] = attn_mask_mmt
        attn_mask[:, :N,  N] = attn_mask_mmt
        attn_mask[:, N:, N:] = 1

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, attn_mask)

        feat_x = self.decoder_norm(x)

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

        feat_x_main = feat_x[:, :N, :]
        feat_x_mmt = feat_x[:, N:N+1, :]
        if self.mmt_use_gloattn or self.mmt_label_use_gloattn:
            mmt_reg_feat = feat_x[:, N+1:, :].mean(dim=1)

        cls_scores_ant = []
        phithe_reg_preds_ant = []
        mmt_reg_preds_ant = []
        mmt_label_scores_ant = []

        cls_score = self.dense_cls(feat_x_main)
        phithe_reg_pred = self.dense_phithe_reg(feat_x_main)

        if self.use_mmt_token:
            mmt_reg_pred = self.dense_mmt_reg(feat_x_mmt).repeat(1, N, 1)
        if self.use_hit_token_for_mmt:
            hit_mmt_reg_pred = self.dense_hit_mmt_reg(feat_x_main)
            if self.use_mmt_token:
                # max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=False)
                # replace_mask = (max_indices > 0)
                # mmt_reg_pred[replace_mask] = hit_mmt_reg_pred[replace_mask]
                max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=True)
                replace_mask = (max_indices > 0)
                mmt_reg_pred = mmt_reg_pred * (~replace_mask) + hit_mmt_reg_pred * replace_mask
            else:
                mmt_reg_pred = hit_mmt_reg_pred
        if self.use_mmt_token_for_label:
            mmt_label_score = self.dense_mmt_label(feat_x_mmt).repeat(1, N, 1)
        else:
            mmt_label_score = None

        cls_scores_ant.append(cls_score)
        phithe_reg_preds_ant.append(phithe_reg_pred)
        mmt_reg_preds_ant.append(mmt_reg_pred)
        mmt_label_scores_ant.append(mmt_label_score)

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

        cls_scores, bbox_preds, mmt_reg_preds_use_fpn = multi_apply(self.forward_single, fpn_feats)

        if self.mmt_use_gloattn:
            # 在维度上展开回归的mmt使其与featmap等大
            mmt_reg_preds_use_gloattn = self.mmt_reg_fc(mmt_reg_feat)
            mmt_reg_preds = []
            num_levels = len(cls_scores)
            for level in range(num_levels):
                cls_score = cls_scores[level]
                B, C, H, W = cls_score.shape
                temp = mmt_reg_preds_use_gloattn[:, None, :, None, None].repeat(
                    1, self.num_base_priors, 1, H, W).flatten(1, 2)

                if mmt_reg_preds_use_fpn[level] is not None:
                    # 如果FPN也回归了mmt，那么将cls_score最大值索引不等于0的那些mmt
                    #     替换为mmt_reg_preds_use_fpn[level]。适用于多目标检测
                    cls_score = cls_score.reshape(B, self.num_base_priors, self.cls_out_channels, H, W)
                    max_values, max_indices = cls_score.max(dim=2, keepdim=False)
                    replace_mask = (max_indices > 0)
                    # temp[replace_mask] = mmt_reg_preds_use_fpn[level][replace_mask]
                    temp = temp * (~replace_mask) + mmt_reg_preds_use_fpn[level] * replace_mask

                mmt_reg_preds.append(temp)
        else:
            mmt_reg_preds = mmt_reg_preds_use_fpn

        if self.mmt_label_use_gloattn:
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

        return (cls_scores, bbox_preds, mmt_reg_preds, mmt_label_scores,
                cls_scores_ant, phithe_reg_preds_ant, mmt_reg_preds_ant, mmt_label_scores_ant)


    def mmt_encode_sigmoid(self, mmt_gts) -> torch.Tensor:
        mmt_norm = torch.clamp((mmt_gts - self.mmt_min) / (self.mmt_max - self.mmt_min),
                               min = self.eps, max = 1 - self.eps)
        return mmt_norm

    def mmt_decode_sigmoid(self, mmt_preds) -> torch.Tensor:
        return torch.sigmoid(mmt_preds) * (self.mmt_max - self.mmt_min) + self.mmt_min


    def mmt_encode_direct(self, mmt_gts) -> torch.Tensor:
        return ((mmt_gts - self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_direct(self, mmt_preds) -> torch.Tensor:
        return (mmt_preds * self.mmt_std + self.mmt_mean) + self.mmt_base


    def mmt_encode_base(self, mmt_gts) -> torch.Tensor:
        return (torch.log(mmt_gts / self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_base(self, mmt_preds) -> torch.Tensor:
        return torch.exp(mmt_preds * self.mmt_std + self.mmt_mean) * self.mmt_base


    def phithe_encode_base(self, phithe_gts, phithe_anchors, label_anchors) -> torch.Tensor:
        """
        Args:
            phithe_gts: torch.Tensor [N, 2]
            phithe_anchors: torch.Tensor [N, 2]
            label_anchors: torch.Tensor [N]
        Returns:
            encoded_phithe: torch.Tensor [N, 2]
        """
        base_1 = torch.tensor(self.phithe_base, device=label_anchors.device)[label_anchors]
        base_2 = base_1.unsqueeze(-1).repeat(1, 2)
        return ((phithe_gts - phithe_anchors) / base_2 - self.phithe_mean) / self.phithe_std

    def phithe_decode_base(self, phithe_preds, phithe_anchors, label_anchors) -> torch.Tensor:
        """
        Args:
            phithe_preds: torch.Tensor [N, 2]
            phithe_anchors: torch.Tensor [N, 2]
            label_anchors: torch.Tensor [N]
        Returns:
            decoded_phithe: torch.Tensor [N, 2]
        """
        base_1 = torch.tensor(self.phithe_base, device=label_anchors.device)[label_anchors]
        base_2 = base_1.unsqueeze(-1).repeat(1, 2)
        return phithe_anchors + base_2 * (phithe_preds * self.phithe_std + self.phithe_mean)


    # def bbox_to_phithe(self, bbox_gts) -> torch.Tensor:
    #     # phi: (0, w) -> (-π, π); the: (0, h) -> (0, π);
    #     x_ctr = (bbox_gts[:, 2::4] + bbox_gts[:, 0::4]) * 0.5
    #     y_ctr = (bbox_gts[:, 3::4] + bbox_gts[:, 1::4]) * 0.5
    #     phi = torch.clamp((x_ctr / self.width - 0.5) * 2 * torch.pi,
    #                       min = -torch.pi+self.eps, max = torch.pi-self.eps)
    #     the = torch.clamp( y_ctr / self.height           * torch.pi,
    #                       min =          +self.eps, max = torch.pi-self.eps)
    #     return torch.cat([phi, the], dim=1)

    def phithe_to_bbox(self, decoded_phithe_preds, label_anchors) -> torch.Tensor:
        local_device = decoded_phithe_preds.device

        w_px = torch.tensor([
            30, 30, 24, 24, 20, 20,         # empty
            20,                             # empty
            15, 15, 12, 12, 10, 10, 
            10,                             # empty
            8, 8, 8, 8, 8, 
            8, 8, 8, 8, 
            8, 8, 8, 8, 8, 
            8, 8, 8, 8, 8, 8, 8, 8, 
            8, 8, 8, 8, 8, 8, 8, 8, 
            8, 8, 8, 8, 8, 
            8, 8, 8, 8, 
            8, 8, 8, 8, 8, 
            10,                             # empty
            10, 10, 12, 12, 15, 15, 
            20,                             # empty
            20, 20, 24, 24, 30, 30,         # empty
        ], device=local_device)
        h_px = torch.tensor([
            8, 8, 8, 8, 7, 7,               # empty
            7,                              # empty
            6, 6, 6, 6, 5, 5, 
            5,                              # empty
            5, 5, 5, 5, 5, 
            6, 6, 6, 6, 
            7, 7, 7, 7, 7, 
            8, 8, 8, 8, 8, 8, 8, 8, 
            8, 8, 8, 8, 8, 8, 8, 8, 
            7, 7, 7, 7, 7, 
            6, 6, 6, 6, 
            5, 5, 5, 5, 5, 
            5,                              # empty
            5, 5, 6, 6, 6, 6, 
            7,                              # empty
            7, 7, 8, 8, 8, 8,               # empty
        ], device=local_device)
        hh_px_2D = torch.tensor([[
            8, 16, 24, 32, 39, 46, 
            53, 
            59, 65, 71, 77, 82, 87, 
            92, 
            97, 102, 107, 112, 117, 
            123, 129, 135, 141, 
            148, 155, 162, 169, 176, 
            184, 192, 200, 208, 216, 224, 232, 240, 
            248, 256, 264, 272, 280, 288, 296, 304, 
            311, 318, 325, 332, 339, 
            345, 351, 357, 363, 
            368, 373, 378, 383, 388, 
            393, 
            398, 403, 409, 415, 421, 427, 
            434, 
            441, 448, 456, 464, 472, 480, 
        ]], device=local_device)

        x_ctr_2D = (decoded_phithe_preds[:, 0::2] / (2 * torch.pi) + 0.5) % 1.0 * self.width
        y_ctr_2D = (decoded_phithe_preds[:, 1::2] /      torch.pi       ) % 1.0 * self.height
        ind = torch.sum((y_ctr_2D - hh_px_2D) >= 0, dim=1)

        easy_scale = torch.tensor(self.easy_scale, device=local_device)[label_anchors]
        w_cell_2D = (w_px[ind] * easy_scale).unsqueeze(-1)
        h_cell_2D = (h_px[ind] * easy_scale).unsqueeze(-1)

        x_min = x_ctr_2D - w_cell_2D / 2
        y_min = y_ctr_2D - h_cell_2D / 2
        x_max = x_ctr_2D + w_cell_2D / 2
        y_max = y_ctr_2D + h_cell_2D / 2
        return torch.cat([x_min, y_min, x_max, y_max], dim=1)


    # ##### ##### ##### ##### ##### #####   from anchor_head.py   ##### ##### ##### ##### ##### ##### #


    def get_anchors_ant(
        self,
        batch_img_metas: List[dict],
        device,
    ):
        B = len(batch_img_metas)
        N = self.len_seq
        anchors_flag = torch.zeros([B, N])
        anchors_phi = torch.zeros([B, N])
        anchors_the = torch.zeros([B, N]) + 0.5 * torch.pi

        for i, img_metas in enumerate(batch_img_metas):
            n = sum(img_metas['n_hit_all'])
            anchors_flag[i, :n] = torch.ones(n)
            anchors_phi[i, :n] = torch.tensor(img_metas['m_phi_all'])
            anchors_the[i, :n] = torch.tensor(img_metas['m_the_all'])

        # 为了避免RuntimeError: CUDA error: device-side assert triggered
        # 不能把覆写放在GPU上操作，而是在覆写完return时再传到GPU上
        return (anchors_flag.to(device=device),
                anchors_phi.to(device=device),
                anchors_the.to(device=device))

    def get_targets_ant(
        self,
        anchors_flag,
        anchors_phi,
        anchors_the,
        batch_gt_instances: InstanceList,
        device,
    ):
        B = len(batch_gt_instances)
        N = self.len_seq
        assert len(anchors_flag) == len(anchors_phi) == len(anchors_the) == B

        # 创建空的targets和weights
        labels = torch.zeros([B, N], dtype=torch.long, device=device) + self.num_classes
        label_weights = torch.zeros([B, N], device=device)
        phithe_reg_targets = torch.zeros([B, N, self.phithe_reg_channels], device=device)
        phithe_reg_weights = torch.zeros([B, N, self.phithe_reg_channels], device=device)
        mmt_reg_targets = torch.zeros([B, N, self.mmt_reg_channels], device=device)
        mmt_reg_weights = torch.zeros([B, N, self.mmt_reg_channels], device=device)

        if self.use_mmt_token_for_label:
            mmt_labels = torch.zeros([B, N], dtype=torch.long, device=device) + self.mmt_label_channels
            mmt_label_weights = torch.ones([B, N], device=device)
            mmt_label_avg_factor = torch.sum(mmt_label_weights)
        else:
            mmt_labels = None
            mmt_label_weights = None
            mmt_label_avg_factor = 0

        # 逐个gt填充进targets和weights
        for i, gt_instances in enumerate(batch_gt_instances):
            # gt_bboxes = gt_instances['bboxes']
            gt_labels = gt_instances['labels']              # shape: G
            gt_phithe_regs = gt_instances['phithe_regs']    # shape: G, 2
            gt_mmt_regs = gt_instances['mmt_regs']          # shape: G, 1

            G = len(gt_labels)

            gt_phi = gt_phithe_regs[:, 0:1]                 # shape: G, 1
            gt_the = gt_phithe_regs[:, 1:2]                 # shape: G, 1
            anchors_phi_i = anchors_phi[i:i+1]              # shape: 1, N
            anchors_the_i = anchors_the[i:i+1]              # shape: 1, N

            # 1. 对于gt_label中的每个元素，假设其值为k，取phithe_base第k个元素的值，构成一个大小[G]的torch Tensor
            #    (e.g. self.phithe_base = [45, 15], gt_label = [0, 1, 1], base_1 = [45, 15, 15])
            base_1 = torch.tensor(self.phithe_base, device=device)[gt_labels]
            base_2 = base_1.unsqueeze(-1).repeat(1, N)      # shape: G, N

            # 2. 计算abs_phi=abs(anchors_phi-gt_phi)和abs_the=abs(anchors_the-gt_the)，得到两个大小[G, N]的torch Tensor；
            #    计算anchors_distance=abs_phi+abs_the，得到一个大小[G, N]的torch Tensor。
            abs_phi = torch.abs(anchors_phi_i - gt_phi)     # shape: G, N
            abs_the = torch.abs(anchors_the_i - gt_the)     # shape: G, N
            anchors_distance = abs_phi + abs_the            # shape: G, N

            # 3. 判断哪些是正样本anchor：
            #    计算anchors_positive=(abs_phi<phithe_base) & (abs_the<phithe_base)，得到一个大小[G, N]的torch Tensor；
            #    对于G行N列的anchors_positive表中的每一列，按列取逻辑或，构成一个大小[N]的torch Tensor。
            anchors_positive = (abs_phi < base_2) & (abs_the < base_2)
            pos_anchors = torch.any(anchors_positive, dim=0)

            # 4. 建立一个anchors_distance的副本anchors_distance_clone，对于anchors_positive中的每个False，将anchors_distance_clone中的对应值改为1e6
            anchors_distance_clone = anchors_distance.clone()
            anchors_distance_clone[~anchors_positive] = 1e6

            # 5. 判断每个anchor负责哪个gt：对于G行N列的anchors_distance_clone表中的每一列，查找最小值所在的行编号，构成一个大小[N]的torch Tensor
            _, anchors_gt = torch.min(anchors_distance_clone, dim=0)

            # 6. 正样本anchor的保底方案：对于G行N列的anchors_distance表前num_hits列中的每一行，假设其行编号为i，查找最小值所在的列编号j，将anchors_gt[j]置为i，将pos_anchors[j]置为True
            valid_anchors = (anchors_flag[i] > 0.5)
            num_hits = int(valid_anchors.sum())
            valid_distance = anchors_distance[:, :num_hits]  # shape: G, num_hits

            # 对每行找最小值所在的列索引
            _, row_min_indices = torch.min(valid_distance, dim=1)
            # 创建行索引 [0, 1, ..., G-1]
            row_indices = torch.arange(G, device=device)

            # 更新anchors_gt和pos_anchors
            anchors_gt[row_min_indices] = row_indices
            pos_anchors[row_min_indices] = True
            assert not anchors_gt.requires_grad
            assert not pos_anchors.requires_grad

            # 剩下的都标记为negative
            neg_anchors = ~pos_anchors
            # 超出num_hits的，标记为invalid
            pos_anchors &= valid_anchors
            neg_anchors &= valid_anchors

            # 7. 对于anchors_gt中的每个元素，假设其值为i，
            #    取gt_label第i个元素，构成一个大小为[N]的torch Tensor；
            #    取gt_phi、gt_the第i个元素，构成一个大小为[N, 2]的torch Tensor；
            #    取gt_mmt_regs第i个元素，构成一个大小为[N, 1]的torch Tensor。
            label_anchors = gt_labels[anchors_gt]

            phi_gts = gt_phi[anchors_gt]
            the_gts = gt_the[anchors_gt]
            phithe_gts = torch.cat([phi_gts, the_gts], dim=1)

            mmt_gts = gt_mmt_regs[anchors_gt]

            # 对gt进行编码
            priors = torch.cat([anchors_phi_i, anchors_the_i], dim=0).transpose(0, 1)
            encoded_phithe = self.phithe_encode_base(phithe_gts, priors, label_anchors)
            if self.mmt_encode_mode == 'base':
                encoded_mmt = self.mmt_encode_base(mmt_gts)
            elif self.mmt_encode_mode == 'direct':
                encoded_mmt = self.mmt_encode_direct(mmt_gts)
            elif self.mmt_encode_mode == 'sigmoid':
                encoded_mmt = self.mmt_encode_sigmoid(mmt_gts)
            else:
                raise NotImplementedError

            # 填充进targets和weights
            labels[i, pos_anchors] = label_anchors[pos_anchors]
            label_weights[i, valid_anchors] = 1.0
            phithe_reg_targets[i] = encoded_phithe
            phithe_reg_weights[i, pos_anchors] = 1.0
            mmt_reg_targets[i] = encoded_mmt
            mmt_reg_weights[i, pos_anchors] = 1.0

            if self.use_mmt_token_for_label:
                mmt_labels[i] = gt_instances['mmt_labels'][0]

        avg_factors = [
            torch.sum(label_weights),
            torch.sum(phithe_reg_weights) / self.phithe_reg_channels,
            torch.sum(mmt_reg_weights) / self.mmt_reg_channels,
            mmt_label_avg_factor,
        ]

        return (labels, label_weights, phithe_reg_targets, phithe_reg_weights,
                mmt_reg_targets, mmt_reg_weights,
                mmt_labels, mmt_label_weights,
                avg_factors)

    def loss_by_feat_ant(
            self,
            cls_scores: List[Tensor],
            phithe_reg_preds: List[Tensor],
            mmt_reg_preds: List[Tensor],
            mmt_label_scores: List[Optional[Tensor]],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None):

        device = cls_scores[0].device

        anchors_flag, anchors_phi, anchors_the = self.get_anchors_ant(
            batch_img_metas, device=device)
        cls_reg_targets = self.get_targets_ant(
            anchors_flag,
            anchors_phi,
            anchors_the,
            batch_gt_instances,
            device=device)
        (labels, label_weights, phithe_reg_targets, phithe_reg_weights,
         mmt_reg_targets, mmt_reg_weights,
         mmt_labels, mmt_label_weights,
         avg_factors) = cls_reg_targets

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_scores[-1].reshape(-1, self.cls_out_channels)
        losses_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=int(avg_factors[0]))

        # regression loss
        phithe_reg_targets = phithe_reg_targets.reshape(-1, self.phithe_reg_channels)
        phithe_reg_weights = phithe_reg_weights.reshape(-1, self.phithe_reg_channels)
        phithe_reg_pred = phithe_reg_preds[-1].reshape(-1, self.phithe_reg_channels)
        losses_phithe_reg = self.loss_phithe_reg(
            phithe_reg_pred, phithe_reg_targets, phithe_reg_weights, avg_factor=int(avg_factors[1]))

        # mmt loss
        # 如果对gt进行了sigmoid预编码、但后续使用L1Loss/L2Loss等而非使用BCELoss，必须也对mmt_reg_pred预编码！
        if self.mmt_encode_mode == 'sigmoid' and not self.use_sigmoid_mmt:
            mmt_reg_pred = torch.sigmoid(mmt_reg_pred)
        mmt_reg_targets = mmt_reg_targets.reshape(-1, self.mmt_reg_channels)
        mmt_reg_weights = mmt_reg_weights.reshape(-1, self.mmt_reg_channels)
        mmt_reg_pred = mmt_reg_preds[-1].reshape(-1, self.mmt_reg_channels)
        losses_mmt_reg = self.loss_mmt_reg(
            mmt_reg_pred, mmt_reg_targets, mmt_reg_weights, avg_factor=int(avg_factors[2]))

        if self.use_mmt_token_for_label:
            mmt_labels = mmt_labels.reshape(-1)
            mmt_label_weights = mmt_label_weights.reshape(-1)
            mmt_label_score = mmt_label_scores[-1].reshape(-1, self.mmt_label_channels)
            losses_mmt_label = self.loss_mmt_label(
                mmt_label_score, mmt_labels, mmt_label_weights, avg_factor=int(avg_factors[3]))
        else:
            losses_mmt_label = torch.zeros(1, device=losses_cls.device)

        return (losses_cls, losses_phithe_reg, losses_mmt_reg, losses_mmt_label)


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

        if self.mmt_label_use_gloattn:
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

            if self.mmt_label_use_gloattn:
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
            if self.mmt_label_use_gloattn:
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

        if self.mmt_label_use_gloattn:
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

        if self.mmt_label_use_gloattn:
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
            cls_scores_ant: List[Tensor],
            phithe_reg_preds_ant: List[Tensor],
            mmt_reg_preds_ant: List[Tensor],
            mmt_label_scores_ant: List[Optional[Tensor]],
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

        losses_cls_vic, losses_bbox_vic, losses_mmt_reg_vic, losses_mmt_label_vic = multi_apply(
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

        losses_cls_ant, losses_phithe_reg_ant, losses_mmt_reg_ant, losses_mmt_label_ant = self.loss_by_feat_ant(
            cls_scores_ant,
            phithe_reg_preds_ant,
            mmt_reg_preds_ant,
            mmt_label_scores_ant,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)

        return dict(
            loss_cls_vic=losses_cls_vic,
            loss_bbox_vic=losses_bbox_vic,
            loss_mmt_reg_vic=losses_mmt_reg_vic,                                                    # mmt
            loss_mmt_label_vic=losses_mmt_label_vic,                                                # mmt_label
            loss_cls_ant=losses_cls_ant,
            loss_phithe_reg_ant=losses_phithe_reg_ant,
            loss_mmt_reg_ant=losses_mmt_reg_ant,
            loss_mmt_label_ant=losses_mmt_label_ant,
        )


    # ##### ##### ##### ##### ##### ##### from base_dense_head.py ##### ##### ##### ##### ##### ##### #


    def _predict_by_feat_single_ant(self,
                                cls_score_list: List[Tensor],
                                phithe_reg_pred_list: List[Tensor],
                                mmt_reg_pred_list: List[Tensor],
                                mmt_label_score_list: List[Optional[Tensor]],
                                img_meta: Optional[dict] = None,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        device = cls_score_list[-1].device

        anchors_flag, anchors_phi, anchors_the = self.get_anchors_ant(
            [img_meta, ], device=device)
        anchors_flag = anchors_flag.transpose(0, 1)                                         # shape: N, 1
        priors = torch.cat([anchors_phi, anchors_the], dim=0).transpose(0, 1)               # shape: N, 2

        cls_score = cls_score_list[-1].reshape(-1, self.cls_out_channels)                   # shape: N, 2
        phithe_reg_pred = phithe_reg_pred_list[-1].reshape(-1, self.phithe_reg_channels)    # shape: N, 2
        mmt_reg_pred = mmt_reg_pred_list[-1].reshape(-1, self.mmt_reg_channels)             # shape: N, 1

        if self.use_mmt_token_for_label:
            mmt_label_score = mmt_label_score_list[-1].reshape(-1, self.mmt_label_channels)
            mmt_scores, mmt_labels = torch.max(mmt_label_score.sigmoid(), dim=-1)
        else:
            mmt_labels = torch.zeros((len(mmt_reg_pred), ), dtype=torch.long, device=device)
            mmt_scores = torch.zeros((len(mmt_reg_pred), ), dtype=torch.float, device=device)

        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        scores *= anchors_flag
        max_scores, max_labels = torch.max(scores, dim=-1)
        decoded_phithe = self.phithe_decode_base(phithe_reg_pred, priors, max_labels)

        bboxes = self.phithe_to_bbox(decoded_phithe, max_labels)
        if self.mmt_encode_mode == 'base':
            mmts = self.mmt_decode_base(mmt_reg_pred)
        elif self.mmt_encode_mode == 'direct':
            mmts = self.mmt_decode_direct(mmt_reg_pred)
        elif self.mmt_encode_mode == 'sigmoid':
            mmts = self.mmt_decode_sigmoid(mmt_reg_pred)
        else:
            raise NotImplementedError

        if self.nms_mode == 'phithe':
            keep_idxs = self.nms_for_phithe(
                max_scores, max_labels, decoded_phithe,
                self.phithe_nms_thr, self.score_thr, self.max_per_img, device)

            results = InstanceData()
            results.bboxes = bboxes[keep_idxs]
            results.mmts = mmts[keep_idxs]
            results.mmt_labels = mmt_labels[keep_idxs]
            results.mmt_scores = mmt_scores[keep_idxs]
            results.scores = max_scores[keep_idxs]
            results.labels = max_labels[keep_idxs]

            return (results, max_scores, max_labels, bboxes, mmts, mmt_labels, mmt_scores)

        elif self.nms_mode == 'bbox':
            results = InstanceData()
            results.bboxes = bboxes
            results.mmts = mmts
            results.mmt_labels = mmt_labels
            results.mmt_scores = mmt_scores
            results.scores = max_scores
            results.labels = max_labels

            return (self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
                img_meta=img_meta), max_scores, max_labels, bboxes, mmts, mmt_labels, mmt_scores)

        else:
            raise NotImplementedError

    def nms_for_phithe(self,
                       scores, labels, phithe_preds, # priors,
                       phithe_nms_thr, score_thr, max_per_img, device):
        """
        张量化实现的NMS操作，基于phi/theta坐标

        Args:
            scores: torch.Tensor [N] 置信度
            labels: torch.Tensor [N] 类别标签
            phithe_preds: torch.Tensor [N, 2] phi和theta预测值
            # priors: torch.Tensor [N, 2] phi和theta先验值
            phithe_nms_thr: float NMS阈值
            score_thr: float 置信度阈值
            max_per_img: int 每张图片最大保留数

        Returns:
            keep_indices: torch.Tensor [N] 需要保留的索引（已按scores降序排列）
        """
        N = len(scores)

        # 对scores进行降序排列，并按这个排列索引重排对应的phithe_preds和priors
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        sorted_phithe_preds = phithe_preds[sorted_indices]
        # sorted_priors = priors[sorted_indices]

        keep_mask = torch.ones(N, dtype=torch.bool, device=device)
        keep_num = 0

        for i in range(N):
            # 如果已被抑制
            if not keep_mask[i]: continue

            # 最后一个必定不用抑制
            if i == N - 1: break

            # 当对应的score值小于score_thr时，包括当前的剩余keep_mask均标记为False
            if sorted_scores[i] < score_thr:
                keep_mask[i:] = False
                break

            # 当标为True的个数已达到max_per_img时，剩余keep_mask均标记为False
            keep_num += 1
            if keep_num >= max_per_img:
                keep_mask[i+1:] = False
                break

            # 类似于NMS操作的：
            # 对于一个score值对应的一对(phi_pred_high, the_pred_high)值和label_high值，
            # 遍历所有更小score值对应的(phi_pred_low, the_pred_low)值和label_low值，
            #     如果满足label_high==label_low，
            #         且同时满足abs(phi_pred_low-phi_pred_high)<phithe_nms_thr & abs(the_pred_low-the_pred_high)<phithe_nms_thr，
            #             则其keep_mask标记为False。
            label_diff = torch.abs(sorted_labels[i+1:] - sorted_labels[i])
            phi_pred_diff = torch.abs(sorted_phithe_preds[i+1:, 0] - sorted_phithe_preds[i, 0])
            the_pred_diff = torch.abs(sorted_phithe_preds[i+1:, 1] - sorted_phithe_preds[i, 1])

            label_close = (label_diff < self.eps)
            pred_close = (phi_pred_diff < phithe_nms_thr) & (the_pred_diff < phithe_nms_thr)

            suppress = (label_close & pred_close)

            # 保留已标记的False，新增~suppress标记的False
            keep_mask[i+1:] = keep_mask[i+1:] & (~suppress)

        keep_indices = sorted_indices[keep_mask]
        return keep_indices


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

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        mmt_reg_preds: List[Optional[Tensor]],                                      # mmt
                        mmt_label_scores: List[Optional[Tensor]],                                   # mmt_label
                        cls_scores_ant: List[Tensor],
                        phithe_reg_preds_ant: List[Tensor],
                        mmt_reg_preds_ant: List[Tensor],
                        mmt_label_scores_ant: List[Optional[Tensor]],
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
        if self.mmt_label_use_gloattn: assert len(cls_scores) == len(mmt_label_scores)              # mmt_label

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
            if self.mmt_label_use_gloattn:                                                          # mmt_label
                mmt_label_score_list = select_single_mlvl(mmt_label_scores, img_id, detach=True)
            else:
                mmt_label_score_list = [None, ] * len(cls_score_list)

            cls_score_ant_list = select_single_mlvl(
                cls_scores_ant, img_id, detach=True)
            phithe_reg_pred_ant_list = select_single_mlvl(
                phithe_reg_preds_ant, img_id, detach=True)
            mmt_reg_pred_ant_list = select_single_mlvl(
                mmt_reg_preds_ant, img_id, detach=True)

            if self.use_mmt_token_for_label:
                mmt_label_score_ant_list = select_single_mlvl(mmt_label_scores_ant, img_id, detach=True)
            else:
                mmt_label_score_ant_list = [None, ] * len(cls_score_ant_list)

            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list          = cls_score_list,
                bbox_pred_list          = bbox_pred_list,
                mmt_reg_pred_list       = mmt_reg_pred_list,
                mmt_label_score_list    = mmt_label_score_list,
                cls_score_ant_list          = cls_score_ant_list,
                phithe_reg_pred_ant_list    = phithe_reg_pred_ant_list,
                mmt_reg_pred_ant_list       = mmt_reg_pred_ant_list,
                mmt_label_score_ant_list    = mmt_label_score_ant_list,
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
                                cls_score_ant_list: List[Tensor],
                                phithe_reg_pred_ant_list: List[Tensor],
                                mmt_reg_pred_ant_list: List[Tensor],
                                mmt_label_score_ant_list: List[Optional[Tensor]],
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
            if self.mmt_label_use_gloattn: assert cls_score.size()[-2:] == mmt_label_score.size()[-2:]      # mmt_label

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_pred = mmt_reg_pred.permute(1, 2, 0).reshape(-1, self.mmt_reg_channels)
            else:
                mmt_reg_pred = torch.zeros((len(bbox_pred), self.mmt_reg_channels), device=bbox_pred.device)

            if self.mmt_label_use_gloattn:                                                          # mmt_label
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

        results_vic = InstanceData()
        results_vic.bboxes = bboxes
        results_vic.mmts = mmts                                                                         # mmt
        results_vic.mmt_labels = torch.cat(mlvl_mmt_labels)                                             # mmt_label
        results_vic.mmt_scores = torch.cat(mlvl_mmt_scores)                                             # mmt_label
        results_vic.scores = torch.cat(mlvl_scores)
        results_vic.labels = torch.cat(mlvl_labels)

        post_vic = self._bbox_post_process(
            results=results_vic,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        (post_ant, 
         max_scores_ant, max_labels_ant, bboxes_ant, 
         mmts_ant, mmt_labels_ant, mmt_scores_ant) = self._predict_by_feat_single_ant(
            cls_score_ant_list,
            phithe_reg_pred_ant_list,
            mmt_reg_pred_ant_list,
            mmt_label_score_ant_list,
            img_meta=img_meta,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms)

        results_mix = InstanceData()
        results_mix.bboxes = torch.cat([bboxes, bboxes_ant], dim=0)
        results_mix.mmts = torch.cat([mmts, mmts_ant], dim=0)
        results_mix.mmt_labels = torch.cat([torch.cat(mlvl_mmt_labels), mmt_labels_ant], dim=0)
        results_mix.mmt_scores = torch.cat([torch.cat(mlvl_mmt_scores), mmt_scores_ant], dim=0)
        results_mix.scores = torch.cat([torch.cat(mlvl_scores), max_scores_ant], dim=0)
        results_mix.labels = torch.cat([torch.cat(mlvl_labels), max_labels_ant], dim=0)

        post_mix = self._bbox_post_process(
            results=results_mix,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        results = InstanceData()
        results.bboxes =       self.x_choose(post_vic.bboxes,     post_ant.bboxes,     post_mix.bboxes,     self.phithe_source)
        results.bboxes_confi = self.x_choose(post_vic.scores,     post_ant.scores,     post_mix.scores,     self.phithe_source)
        results.mmts =         self.x_choose(post_vic.mmts,       post_ant.mmts,       post_mix.mmts,       self.mmt_source)
        results.mmts_confi =   self.x_choose(post_vic.scores,     post_ant.scores,     post_mix.scores,     self.mmt_source)
        results.mmt_labels =   self.x_choose(post_vic.mmt_labels, post_ant.mmt_labels, post_mix.mmt_labels, self.mmt_source)
        results.mmt_scores =   self.x_choose(post_vic.mmt_scores, post_ant.mmt_scores, post_mix.mmt_scores, self.mmt_source)
        results.scores =       self.x_choose(post_vic.scores,     post_ant.scores,     post_mix.scores,     self.phithe_source)
        results.labels =       self.x_choose(post_vic.labels,     post_ant.labels,     post_mix.labels,     self.phithe_source)
        return results

    def x_choose(self, x_vic, x_ant, x_mix, source):
        if source == 'vic':
            return x_vic
        elif source == 'ant':
            return x_ant
        elif source == 'mix':
            return x_mix
        else:
            raise NotImplementedError

