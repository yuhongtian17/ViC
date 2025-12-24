from typing import List, Optional, Union, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                                multi_apply)

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmcv.cnn import build_norm_layer
from mmdet.models.backbones.hepv2_transformer import Block

from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader


@MODELS.register_module()
class HEPv2DenseHead(BaseDenseHead):

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 768,
        # 
        use_fc_head: bool = False,
        num_shared_fcs: int = 2,
        fc_out_channels: int = 768,
        # 
        embed_dim=384, # 512,
        depth=8,
        num_heads=12, # 16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        with_cp=False,
        # 
        # phithe_pos_thresh: float = 45.0,
        phithe_base: Union[float, List[float]] = 45.0,
        phithe_mean: float = 0.0,
        phithe_std: float = 1.0,
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
        use_mmt_token: bool = True,
        use_hit_token_for_mmt: bool = False,
        backbone_out_eng: bool = True,
        backbone_out_phithe: bool = False,
        # 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
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
        phithe_nms_thr: float = 3.0,
        score_thr: float = 0.0,
        max_per_img: int = 1,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        super().__init__()
        self.init_cfg=init_cfg

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.use_fc_head = use_fc_head
        self.num_shared_fcs = num_shared_fcs
        self.fc_out_channels = fc_out_channels

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        # self.phithe_pos_thresh = phithe_pos_thresh / 180 * torch.pi
        if isinstance(phithe_base, float) or isinstance(phithe_base, int):
            self.phithe_base = [phithe_base / 180 * torch.pi] * self.num_classes
        elif isinstance(phithe_base, list) or isinstance(phithe_base, tuple):
            assert len(phithe_base) == self.num_classes
            self.phithe_base = [temp / 180 * torch.pi for temp in phithe_base]
        else:
            raise NotImplementedError

        self.phithe_mean = phithe_mean
        self.phithe_std = phithe_std
        self.phithe_reg_channels = 2

        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        self.mmt_base = mmt_base
        self.mmt_mean = mmt_mean
        self.mmt_std = mmt_std
        self.mmt_encode_mode = mmt_encode_mode
        self.use_sigmoid_mmt = loss_mmt_reg.get('use_sigmoid', False)
        self.mmt_reg_channels = mmt_reg_channels

        self.use_mmt_label = (loss_mmt_label is not None)
        self.mmt_label_channels = mmt_label_channels

        self.len_seq = len_seq
        self.use_mmt_token = use_mmt_token
        self.use_hit_token_for_mmt = use_hit_token_for_mmt
        self.backbone_out_eng = backbone_out_eng
        self.backbone_out_phithe = backbone_out_phithe

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_phithe_reg = MODELS.build(loss_phithe_reg)
        self.loss_mmt_reg = MODELS.build(loss_mmt_reg)

        if self.use_mmt_label:
            self.loss_mmt_label = MODELS.build(loss_mmt_label)

        self.phithe_nms_thr = phithe_nms_thr / 180 * torch.pi
        self.score_thr = score_thr
        self.max_per_img = max_per_img
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        if self.use_fc_head:
            self._fc_init_layers()
        else:
            self._trans_init_layers(
                in_channels,
                embed_dim,
                depth,
                num_heads,
                mlp_ratio,
                qkv_bias,
                drop_path_rate,
                norm_cfg,
                act_cfg,
                with_cp,
            )

        self.width = 960
        self.height = 480
        self.easy_scale = 10
        self.eps = 1e-6

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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


    def _fc_init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)

        self.shared_fcs = nn.ModuleList()
        if self.num_shared_fcs > 0:
            for i in range(self.num_shared_fcs):
                fc_in_channels = self.in_channels if i == 0 else self.fc_out_channels
                self.shared_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        else:
            last_layer_dim = self.in_channels

        self.dense_cls        = nn.Linear(last_layer_dim, self.cls_out_channels)
        self.dense_phithe_reg = nn.Linear(last_layer_dim, self.phithe_reg_channels)

        if self.use_mmt_token:
            self.dense_mmt_reg = nn.Linear(last_layer_dim, self.mmt_reg_channels)
        if self.use_hit_token_for_mmt:
            self.dense_hit_reg = nn.Linear(last_layer_dim, self.mmt_reg_channels)
        if self.use_mmt_label:
            self.dense_mmt_label = nn.Linear(last_layer_dim, self.mmt_label_channels)

        self.apply(self._init_weights)

    def _fc_forward(self, inputs):
        # 来自HEPv2Transformer: backbone_outs是List[Tensor], 每个Tensor形状为[B, N, C]
        backbone_outs, backbone_outs_others = inputs
        feat_x = backbone_outs[-1]
        flags = backbone_outs_others['x'][..., 0:1]
        assert not flags.requires_grad

        if self.num_shared_fcs > 0:
            for fc in self.shared_fcs:
                feat_x = self.relu(fc(feat_x))

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

        if self.use_mmt_token:
            feat_x_main = feat_x[:, :-1, :]
            feat_x_mmt = feat_x[:, -1:, :]
        else:
            feat_x_main = feat_x
            feat_x_mask = (flags > self.eps)
            feat_x_mmt = (feat_x * feat_x_mask).sum(dim=1, keepdim=True) / feat_x_mask.sum(dim=1, keepdim=True)

        N = self.len_seq

        cls_scores = []
        phithe_reg_preds = []
        mmt_reg_preds = []
        mmt_label_scores = []

        cls_score = self.dense_cls(feat_x_main)
        phithe_reg_pred = self.dense_phithe_reg(feat_x_main)

        if self.use_mmt_token:
            mmt_reg_pred = self.dense_mmt_reg(feat_x_mmt).repeat(1, N, 1)
        if self.use_hit_token_for_mmt:
            hit_reg_pred = self.dense_hit_reg(feat_x_main)
            if self.use_mmt_token:
                # max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=False)
                # replace_mask = (max_indices > 0)
                # mmt_reg_pred[replace_mask] = hit_reg_pred[replace_mask]
                max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=True)
                replace_mask = (max_indices > 0)
                mmt_reg_pred = mmt_reg_pred * (~replace_mask) + hit_reg_pred * replace_mask
            else:
                mmt_reg_pred = hit_reg_pred
        if self.use_mmt_label:
            mmt_label_score = self.dense_mmt_label(feat_x_mmt).repeat(1, N, 1)
        else:
            mmt_label_score = None

        cls_scores.append(cls_score)
        phithe_reg_preds.append(phithe_reg_pred)
        mmt_reg_preds.append(mmt_reg_pred)
        mmt_label_scores.append(mmt_label_score)

        return cls_scores, phithe_reg_preds, mmt_reg_preds, mmt_label_scores


    def _trans_init_layers(
        self,
        in_channels=768,
        embed_dim=384, # 512,
        depth=8,
        num_heads=12, # 16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        with_cp=False,
    ):
        """Initialize layers of the head."""
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)

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

        if self.use_mmt_token:
            self.dense_mmt_reg = nn.Linear(embed_dim, self.mmt_reg_channels)
        if self.use_hit_token_for_mmt:
            self.dense_hit_reg = nn.Linear(embed_dim, self.mmt_reg_channels)
        if self.use_mmt_label:
            self.dense_mmt_label = nn.Linear(embed_dim, self.mmt_label_channels)

        self.apply(self._init_weights)

    def _trans_forward(self, inputs):
        # 来自HEPv2Transformer: backbone_outs是List[Tensor], 每个Tensor形状为[B, N, C]
        backbone_outs, backbone_outs_others = inputs

        feat = backbone_outs[-1]
        # assert feat.requires_grad
        feat = self.decoder_embed(feat)
        if self.use_mmt_token:
            feat_main = feat[:, :-1, :]
            feat_mmt = feat[:, -1:, :]
        else:
            feat_main = feat

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
        if self.use_mmt_token:
            x = torch.cat([x, feat_mmt], dim=1)                                 # B, N + 1, C
            attn_mask = torch.zeros(B, N + 1, N + 1, device=x.device, requires_grad=False)
            attn_mask[:, :-1, :-1] = attn_mask_main
            attn_mask[:,  -1, :-1] = attn_mask_mmt
            attn_mask[:, :-1,  -1] = attn_mask_mmt
        else:
            attn_mask = attn_mask_main

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, attn_mask)

        feat_x = self.decoder_norm(x)

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

        if self.use_mmt_token:
            feat_x_main = feat_x[:, :-1, :]
            feat_x_mmt = feat_x[:, -1:, :]
        else:
            feat_x_main = feat_x
            feat_x_mask = (flags > self.eps)
            feat_x_mmt = (feat_x * feat_x_mask).sum(dim=1, keepdim=True) / feat_x_mask.sum(dim=1, keepdim=True)

        N = self.len_seq

        cls_scores = []
        phithe_reg_preds = []
        mmt_reg_preds = []
        mmt_label_scores = []

        cls_score = self.dense_cls(feat_x_main)
        phithe_reg_pred = self.dense_phithe_reg(feat_x_main)

        if self.use_mmt_token:
            mmt_reg_pred = self.dense_mmt_reg(feat_x_mmt).repeat(1, N, 1)
        if self.use_hit_token_for_mmt:
            hit_reg_pred = self.dense_hit_reg(feat_x_main)
            if self.use_mmt_token:
                # max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=False)
                # replace_mask = (max_indices > 0)
                # mmt_reg_pred[replace_mask] = hit_reg_pred[replace_mask]
                max_values, max_indices = torch.max(cls_score, dim=-1, keepdim=True)
                replace_mask = (max_indices > 0)
                mmt_reg_pred = mmt_reg_pred * (~replace_mask) + hit_reg_pred * replace_mask
            else:
                mmt_reg_pred = hit_reg_pred
        if self.use_mmt_label:
            mmt_label_score = self.dense_mmt_label(feat_x_mmt).repeat(1, N, 1)
        else:
            mmt_label_score = None

        cls_scores.append(cls_score)
        phithe_reg_preds.append(phithe_reg_pred)
        mmt_reg_preds.append(mmt_reg_pred)
        mmt_label_scores.append(mmt_label_score)

        return cls_scores, phithe_reg_preds, mmt_reg_preds, mmt_label_scores


    def forward(self, inputs):
        if self.use_fc_head:
            return self._fc_forward(inputs)
        else:
            return self._trans_forward(inputs)


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

    def phithe_to_bbox(self, decoded_phithe_preds) -> torch.Tensor:
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
        w_cell_2D = w_px[ind].unsqueeze(-1) * self.easy_scale
        h_cell_2D = h_px[ind].unsqueeze(-1) * self.easy_scale

        x_min = x_ctr_2D - w_cell_2D / 2
        y_min = y_ctr_2D - h_cell_2D / 2
        x_max = x_ctr_2D + w_cell_2D / 2
        y_max = y_ctr_2D + h_cell_2D / 2
        return torch.cat([x_min, y_min, x_max, y_max], dim=1)


    # ##### ##### ##### ##### ##### #####   from anchor_head.py   ##### ##### ##### ##### ##### ##### #


    def get_anchors(
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

    def get_targets(
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

        if self.use_mmt_label:
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

            if self.use_mmt_label:
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

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            phithe_reg_preds: List[Tensor],
            mmt_reg_preds: List[Tensor],
            mmt_label_scores: List[Optional[Tensor]],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        device = cls_scores[0].device

        anchors_flag, anchors_phi, anchors_the = self.get_anchors(
            batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
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

        if self.use_mmt_label:
            mmt_labels = mmt_labels.reshape(-1)
            mmt_label_weights = mmt_label_weights.reshape(-1)
            mmt_label_score = mmt_label_scores[-1].reshape(-1, self.mmt_label_channels)
            losses_mmt_label = self.loss_mmt_label(
                mmt_label_score, mmt_labels, mmt_label_weights, avg_factor=int(avg_factors[3]))
        else:
            losses_mmt_label = torch.zeros(1, device=losses_cls.device)

        return dict(loss_cls=losses_cls, loss_phithe_reg=losses_phithe_reg, loss_mmt_reg=losses_mmt_reg,
                    loss_mmt_label=losses_mmt_label)


    # ##### ##### ##### ##### ##### ##### from base_dense_head.py ##### ##### ##### ##### ##### ##### #


    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        phithe_reg_preds: List[Tensor],
                        mmt_reg_preds: List[Tensor],
                        mmt_label_scores: List[Optional[Tensor]],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            phithe_reg_pred_list = select_single_mlvl(
                phithe_reg_preds, img_id, detach=True)
            mmt_reg_pred_list = select_single_mlvl(
                mmt_reg_preds, img_id, detach=True)

            if self.use_mmt_label:
                mmt_label_score_list = select_single_mlvl(mmt_label_scores, img_id, detach=True)
            else:
                mmt_label_score_list = [None, ] * len(cls_score_list)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                phithe_reg_pred_list=phithe_reg_pred_list,
                mmt_reg_pred_list=mmt_reg_pred_list,
                mmt_label_score_list=mmt_label_score_list,
                img_meta=img_meta)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                phithe_reg_pred_list: List[Tensor],
                                mmt_reg_pred_list: List[Tensor],
                                mmt_label_score_list: List[Optional[Tensor]],
                                img_meta: dict) -> InstanceData:

        device = cls_score_list[-1].device

        anchors_flag, anchors_phi, anchors_the = self.get_anchors(
            [img_meta, ], device=device)
        anchors_flag = anchors_flag.transpose(0, 1)                                         # shape: N, 1
        priors = torch.cat([anchors_phi, anchors_the], dim=0).transpose(0, 1)               # shape: N, 2

        cls_score = cls_score_list[-1].reshape(-1, self.cls_out_channels)                   # shape: N, 2
        phithe_reg_pred = phithe_reg_pred_list[-1].reshape(-1, self.phithe_reg_channels)    # shape: N, 2
        mmt_reg_pred = mmt_reg_pred_list[-1].reshape(-1, self.mmt_reg_channels)             # shape: N, 1

        if self.use_mmt_label:
            mmt_label_score = mmt_label_score_list[-1].reshape(-1, self.mmt_label_channels)
            _, mmt_labels = torch.max(mmt_label_score, dim=-1)
        else:
            mmt_labels = torch.zeros((len(mmt_reg_pred), ), dtype=torch.long, device=device)

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

        bboxes = self.phithe_to_bbox(decoded_phithe)
        if self.mmt_encode_mode == 'base':
            mmts = self.mmt_decode_base(mmt_reg_pred)
        elif self.mmt_encode_mode == 'direct':
            mmts = self.mmt_decode_direct(mmt_reg_pred)
        elif self.mmt_encode_mode == 'sigmoid':
            mmts = self.mmt_decode_sigmoid(mmt_reg_pred)
        else:
            raise NotImplementedError

        keep_idxs = self.nms_for_phithe(
            max_scores, max_labels, decoded_phithe, # priors,
            self.phithe_nms_thr, self.score_thr, self.max_per_img, device)

        results = InstanceData()
        results.bboxes = bboxes[keep_idxs]
        results.mmts = mmts[keep_idxs]
        results.mmt_labels = mmt_labels[keep_idxs]
        results.scores = max_scores[keep_idxs]
        results.labels = max_labels[keep_idxs]

        return results

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

            # # 对于一个score值对应的一对(phi_pred_high, the_pred_high)值和一对(phi_priors_high, the_prior_high)值，
            # # 遍历所有更小score值对应的(phi_pred_low, the_pred_low)值和(phi_priors_low, the_prior_low)值，
            # #     如果满足abs(phi_priors_low-phi_priors_high)<phithe_nms_thr & abs(the_priors_low-the_priors_high)<phithe_nms_thr，
            # #         那么如果它同时满足abs(phi_pred_low-phi_pred_high)<phithe_nms_thr*2 & abs(the_pred_low-the_pred_high)<phithe_nms_thr*2，
            # #             则其keep_mask标记为False；
            # #     如果不满足，
            # #         那么如果它同时满足abs(phi_pred_low-phi_pred_high)<phithe_nms_thr & abs(the_pred_low-the_pred_high)<phithe_nms_thr，
            # #             则其keep_mask标记为False。
            # phi_priors_diff = torch.abs(sorted_priors[i+1:, 0] - sorted_priors[i, 0])
            # the_priors_diff = torch.abs(sorted_priors[i+1:, 1] - sorted_priors[i, 1])
            # phi_pred_diff = torch.abs(sorted_phithe_preds[i+1:, 0] - sorted_phithe_preds[i, 0])
            # the_pred_diff = torch.abs(sorted_phithe_preds[i+1:, 1] - sorted_phithe_preds[i, 1])

            # prior_close = (phi_priors_diff < phithe_nms_thr) & (the_priors_diff < phithe_nms_thr)
            # pred_close_2x = (phi_pred_diff < phithe_nms_thr * 2) & (the_pred_diff < phithe_nms_thr * 2)
            # pred_close_1x = (phi_pred_diff < phithe_nms_thr) & (the_pred_diff < phithe_nms_thr)

            # suppress = (prior_close & pred_close_2x) | (~prior_close & pred_close_1x)

            # 保留已标记的False，新增~suppress标记的False
            keep_mask[i+1:] = keep_mask[i+1:] & (~suppress)

        keep_indices = sorted_indices[keep_mask]
        return keep_indices

