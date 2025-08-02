from typing import List, Optional, Tuple, Dict

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


@MODELS.register_module()
class HEPv2DenseHead(BaseDenseHead):

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 768,
        # 
        num_shared_fcs: int = 2,
        fc_out_channels: int = 768,
        phithe_pos_thresh: float = 45.0,
        phithe_base: float = 45.0,
        phithe_mean: float = 0.0,
        phithe_std: float = 1.0,
        # 
        mmt_min: float = 0.0,
        mmt_max: float = 1.2,
        mmt_base: float = 1.0,
        mmt_mean: float = 0.0,
        mmt_std: float = 1.0,
        # 
        len_seq: int = 640,
        use_mmt_token: bool = True,
        # 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_phithe_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_mmt_reg=dict(type='L1Loss', loss_weight=1.0),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01),
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_shared_fcs = num_shared_fcs
        self.fc_out_channels = fc_out_channels

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self.phithe_pos_thresh = phithe_pos_thresh / 180 * torch.pi
        self.phithe_base = phithe_base / 180 * torch.pi
        self.phithe_mean = phithe_mean
        self.phithe_std = phithe_std
        self.phithe_reg_channels = 2

        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        self.mmt_base = mmt_base
        self.mmt_mean = mmt_mean
        self.mmt_std = mmt_std
        self.mmt_reg_channels = 1

        self.len_seq = len_seq
        self.use_mmt_token = use_mmt_token

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_phithe_reg = MODELS.build(loss_phithe_reg)
        self.loss_mmt_reg = MODELS.build(loss_mmt_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        self._init_layers()

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

    def _init_layers(self):
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
        self.dense_mmt_reg    = nn.Linear(last_layer_dim, self.mmt_reg_channels)
        self.apply(self._init_weights)

    def forward(self, inputs):
        """ """
        if not self.use_mmt_token:
            # 来自HEPv2Transformer: backbone_outs是List[Tensor], 每个Tensor形状为[B, N, C]
            backbone_outs, backbone_outs_others = inputs
            feat = backbone_outs[-1]
            flags = backbone_outs_others['x'][..., 0:1]
            assert not flags.requires_grad
        else:
            feat = inputs[-1]

        if self.num_shared_fcs > 0:
            for fc in self.shared_fcs:
                feat = self.relu(fc(feat))

        if not self.use_mmt_token:
            feat_main = feat
            feat_mask = (flags > self.eps)
            feat_mmt = (feat * feat_mask).sum(dim=1, keepdim=True) / feat_mask.sum(dim=1, keepdim=True)
        else:
            feat_main = feat[:, :-1, :]
            feat_mmt = feat[:, -1:, :]

        N = self.len_seq

        cls_scores = []
        phithe_reg_preds = []
        mmt_reg_preds = []

        cls_scores.append(self.dense_cls(feat_main))
        phithe_reg_preds.append(self.dense_phithe_reg(feat_main))
        mmt_reg_preds.append(self.dense_mmt_reg(feat_mmt).repeat(1, N, 1))

        return cls_scores, phithe_reg_preds, mmt_reg_preds


    def mmt_encode_base(self, mmt_gts) -> torch.Tensor:
        return (torch.log(mmt_gts / self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_base(self, mmt_preds) -> torch.Tensor:
        return torch.exp(mmt_preds * self.mmt_std + self.mmt_mean) * self.mmt_base


    def phithe_encode_base(self, phithe_gts, phithe_anchors) -> torch.Tensor:
        return ((phithe_gts - phithe_anchors) / self.phithe_base - self.phithe_mean) / self.phithe_std

    def phithe_decode_base(self, phithe_preds, phithe_anchors) -> torch.Tensor:
        return phithe_anchors + self.phithe_base * (phithe_preds * self.phithe_std + self.phithe_mean)


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

        # 逐个gt填充进targets和weights
        for i, gt_instances in enumerate(batch_gt_instances):
            # gt_bboxes = gt_instances['bboxes']
            gt_labels = gt_instances['labels']              # shape: 1
            gt_phithe_regs = gt_instances['phithe_regs']    # shape: 1, 2
            gt_mmt_regs = gt_instances['mmt_regs']          # shape: 1, 1

            anchors_phi_i = anchors_phi[i]
            anchors_the_i = anchors_the[i]
            gt_phi, gt_the = gt_phithe_regs[0]

            # 简单的判断正负锚框的方法
            # phi和the偏差都不超过6°的标记为positive
            pos_anchors = (torch.abs(anchors_phi_i - gt_phi) < self.phithe_pos_thresh) & \
                          (torch.abs(anchors_the_i - gt_the) < self.phithe_pos_thresh)
            # 取phi+the偏差最小的也标记为positive
            min_values, min_indices = torch.min(
                torch.abs(anchors_phi_i - gt_phi) +
                torch.abs(anchors_the_i - gt_the), dim=-1)
            pos_anchors[min_indices] = True
            # 剩下的都标记为negative
            neg_anchors = ~pos_anchors

            # 超出num_hits的，标记为invalid
            valid_anchors = (anchors_flag[i] > 0.5)
            pos_anchors &= valid_anchors
            neg_anchors &= valid_anchors

            # 对gt进行编码
            priors = torch.vstack([anchors_phi_i, anchors_the_i]).transpose(0, 1)
            encoded_phithe = self.phithe_encode_base(gt_phithe_regs, priors)
            encoded_mmt = self.mmt_encode_base(gt_mmt_regs[0])

            # 填充进targets和weights
            labels[i, pos_anchors] = gt_labels[0]
            label_weights[i, valid_anchors] = 1.0
            phithe_reg_targets[i] = encoded_phithe
            phithe_reg_weights[i, pos_anchors] = 1.0
            mmt_reg_targets[i] = encoded_mmt
            mmt_reg_weights[i, :] = 1.0

        avg_factors = [
            torch.sum(label_weights),
            torch.sum(phithe_reg_weights) / self.phithe_reg_channels,
            torch.sum(mmt_reg_weights) / self.mmt_reg_channels,
        ]

        return (labels, label_weights, phithe_reg_targets, phithe_reg_weights,
                mmt_reg_targets, mmt_reg_weights,
                avg_factors)

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            phithe_reg_preds: List[Tensor],
            mmt_reg_preds: List[Tensor],
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
        mmt_reg_targets = mmt_reg_targets.reshape(-1, self.mmt_reg_channels)
        mmt_reg_weights = mmt_reg_weights.reshape(-1, self.mmt_reg_channels)
        mmt_reg_pred = mmt_reg_preds[-1].reshape(-1, self.mmt_reg_channels)
        losses_mmt_reg = self.loss_mmt_reg(
            mmt_reg_pred, mmt_reg_targets, mmt_reg_weights, avg_factor=int(avg_factors[2]))

        return dict(loss_cls=losses_cls, loss_phithe_reg=losses_phithe_reg, loss_mmt_reg=losses_mmt_reg)


    # ##### ##### ##### ##### ##### ##### from base_dense_head.py ##### ##### ##### ##### ##### ##### #


    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        phithe_reg_preds: List[Tensor],
                        mmt_reg_preds: List[Tensor],
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

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                phithe_reg_pred_list=phithe_reg_pred_list,
                mmt_reg_pred_list=mmt_reg_pred_list,
                img_meta=img_meta)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                phithe_reg_pred_list: List[Tensor],
                                mmt_reg_pred_list: List[Tensor],
                                img_meta: dict) -> InstanceData:

        device = cls_score_list[-1].device

        anchors_flag, anchors_phi, anchors_the = self.get_anchors(
            [img_meta, ], device=device)
        anchors_flag = anchors_flag.transpose(0, 1)                                         # shape: N, 1
        priors = torch.cat([anchors_phi, anchors_the], dim=0).transpose(0, 1)               # shape: N, 2

        cls_score = cls_score_list[-1].reshape(-1, self.cls_out_channels)                   # shape: N, 2
        phithe_reg_pred = phithe_reg_pred_list[-1].reshape(-1, self.phithe_reg_channels)    # shape: N, 2
        mmt_reg_pred = mmt_reg_pred_list[-1].reshape(-1, self.mmt_reg_channels)             # shape: N, 1

        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        scores *= anchors_flag
        results = filter_scores_and_topk(
            scores, 0.0, 1,
            dict(phithe_reg_pred=phithe_reg_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results                               # shape: 1

        phithe_reg_pred = filtered_results['phithe_reg_pred']                               # shape: 1, 2
        priors = filtered_results['priors']                                                 # shape: 1, 2
        mmt_reg_pred = mmt_reg_pred[keep_idxs]                                              # shape: 1, 1

        bboxes = self.phithe_to_bbox(self.phithe_decode_base(phithe_reg_pred, priors))
        mmts = self.mmt_decode_base(mmt_reg_pred)

        assert scores.size(0) == labels.size(0) == bboxes.size(0) == mmts.size(0) == 1
        assert scores.dim() == labels.dim() == 1
        assert bboxes.dim() == mmts.dim() == 2

        results = InstanceData()
        results.bboxes = bboxes
        results.mmts = mmts
        results.scores = scores
        results.labels = labels

        return results

