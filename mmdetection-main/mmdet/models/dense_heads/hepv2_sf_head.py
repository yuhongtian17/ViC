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
class HEPv2SFHead(BaseDenseHead):

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

        self.loss_bbox = self.loss_phithe_reg
        self.reg_decoded_bbox = False
        self.use_sigmoid_bbox = loss_phithe_reg.get('use_sigmoid', False)
        self.use_mmt_reg = True
        self.use_sigmoid_mmt_reg = loss_mmt_reg.get('use_sigmoid', False)

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

        # if self.num_shared_fcs > 0:
        #     for fc in self.shared_fcs:
        #         feat = self.relu(fc(feat))

        if not self.use_mmt_token:
            # feat_main = feat
            feat_mask = (flags > self.eps)
            feat_mmt = (feat * feat_mask).sum(dim=1) / feat_mask.sum(dim=1)
        else:
            # feat_main = feat[:, :-1, :]
            feat_mmt = feat[:, -1, :]

        # N = self.len_seq

        cls_scores = []
        phithe_reg_preds = []
        mmt_reg_preds = []

        cls_scores.append(self.dense_cls(feat_mmt))
        phithe_reg_preds.append(self.dense_phithe_reg(feat_mmt))
        mmt_reg_preds.append(self.dense_mmt_reg(feat_mmt))

        return cls_scores, phithe_reg_preds, mmt_reg_preds


    def bbox_encode_norm(self, bbox_gts):
        x_ctr_norm = torch.clamp((bbox_gts[:, 2::4] + bbox_gts[:, 0::4]) * 0.5 / self.width,
                                 min = self.eps, max = 1 - self.eps)
        y_ctr_norm = torch.clamp((bbox_gts[:, 3::4] + bbox_gts[:, 1::4]) * 0.5 / self.height,
                                 min = self.eps, max = 1 - self.eps)
        return torch.cat([x_ctr_norm, y_ctr_norm], dim=1)

    def bbox_decode(self, bbox_preds):
        local_device = bbox_preds.device

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

        bbox_preds_norm = torch.sigmoid(bbox_preds)
        x_ctr_2D = bbox_preds_norm[:, 0::2] * self.width
        y_ctr_2D = bbox_preds_norm[:, 1::2] * self.height
        ind = torch.sum((y_ctr_2D - hh_px_2D) >= 0, dim=1)
        w_cell_2D = w_px[ind].unsqueeze(-1) * self.easy_scale
        h_cell_2D = h_px[ind].unsqueeze(-1) * self.easy_scale

        x_min = x_ctr_2D - w_cell_2D / 2
        y_min = y_ctr_2D - h_cell_2D / 2
        x_max = x_ctr_2D + w_cell_2D / 2
        y_max = y_ctr_2D + h_cell_2D / 2
        return torch.cat([x_min, y_min, x_max, y_max], dim=1)

    def mmt_encode_norm(self, mmt_gts) -> Tensor:
        # return (torch.log(mmt_gts) - self.mmt_mean) / self.mmt_std
        mmt_norm = torch.clamp(mmt_gts / self.mmt_max,
                               min = self.eps, max = 1 - self.eps)
        return mmt_norm

    def mmt_decode(self, mmt_preds) -> Tensor:
        # return torch.exp(mmt_preds * self.mmt_std + self.mmt_mean)
        return torch.sigmoid(mmt_preds) * self.mmt_max


    def _get_targets_single(
        self,
        gt_instances: InstanceData,
    ) -> Tuple[Tensor, Tensor]:

        gt_labels = gt_instances.labels
        gt_bboxes = gt_instances.bboxes
        gt_mmt_regs = gt_instances.mmt_regs

        if not self.reg_decoded_bbox:
            bbox_targets = self.bbox_encode_norm(gt_bboxes)
        else:
            bbox_targets = gt_bboxes

        mmt_reg_targets = self.mmt_encode_norm(gt_mmt_regs)

        return gt_labels, bbox_targets, mmt_reg_targets


    def get_targets(
        self,
        batch_gt_instances: InstanceList,
    ) -> Tuple[List[Tensor], List[Tensor]]:

        return multi_apply(self._get_targets_single, batch_gt_instances)


    def loss_by_feat(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        mmt_reg_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        labels, bbox_targets, mmt_reg_targets = self.get_targets(batch_gt_instances)
        avg_factor = len(labels)                                                                    # B

        flatten_cls_scores = cls_scores[-1]                                                         # shape: B, num_classes
        flatten_bbox_preds = bbox_preds[-1]                                                         # shape: B, 2

        flatten_labels = torch.cat(labels)                                                          # shape: B
        flatten_bbox_targets = torch.cat(bbox_targets)                                              # shape: B, 2

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=avg_factor)

        if self.reg_decoded_bbox:
            # 如果使用IoULoss等，需要对bbox_pred解码
            flatten_bbox_preds = self.bbox_decode(flatten_bbox_preds)
        elif not self.use_sigmoid_bbox:
            # 如果使用L1Loss等（而非使用CrossEntropyLoss且sigmoid=True），需要对bbox_pred归一化
            flatten_bbox_preds = torch.sigmoid(flatten_bbox_preds)
        else:
            pass

        loss_bbox = self.loss_bbox(
            flatten_bbox_preds, flatten_bbox_targets, avg_factor=avg_factor)

        # mmt loss
        if self.use_mmt_reg:                                                                        # mmt
            flatten_mmt_reg_preds = mmt_reg_preds[-1]                                               # shape: B, 1
            flatten_mmt_reg_targets = torch.cat(mmt_reg_targets)                                    # shape: B, 1
            if not self.use_sigmoid_mmt_reg:
                # 如果使用L1Loss等（而非使用CrossEntropyLoss且sigmoid=True），需要对mmt_reg_pred归一化
                flatten_mmt_reg_preds = torch.sigmoid(flatten_mmt_reg_preds)
            loss_mmt_reg = self.loss_mmt_reg(
                flatten_mmt_reg_preds, flatten_mmt_reg_targets, avg_factor=avg_factor)
        else:
            loss_mmt_reg = torch.zeros(1, device=loss_bbox.device)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_mmt_reg=loss_mmt_reg)


    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        mmt_reg_preds: List[Tensor],                                                # mmt
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
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_pred_list = select_single_mlvl(mmt_reg_preds, img_id, detach=True)
            else:
                mmt_reg_pred_list = [None, ] * len(bbox_pred_list)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                mmt_reg_pred_list=mmt_reg_pred_list,                                                # mmt
                img_meta=img_meta,
                rescale=rescale)
            result_list.append(results)
        return result_list


    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                mmt_reg_pred_list: List[Tensor],                                    # mmt
                                img_meta: dict,
                                rescale: bool = False) -> InstanceData:

        mlvl_bbox_preds = []
        mlvl_mmt_preds = []                                                                         # mmt
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, bbox_pred, mmt_reg_pred) in \
                enumerate(zip(cls_score_list, bbox_pred_list, mmt_reg_pred_list)):                  # mmt

            bbox_pred = bbox_pred.reshape(-1, 2)

            if self.use_mmt_reg:                                                                    # mmt
                mmt_reg_pred = mmt_reg_pred.reshape(-1, 1)
            else:
                mmt_reg_pred = torch.zeros((len(bbox_pred), 1), device=bbox_pred.device)

            cls_score = cls_score.reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            score_pred, label_pred = torch.max(scores, 1)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_mmt_preds.append(mmt_reg_pred)
            mlvl_scores.append(score_pred)
            mlvl_labels.append(label_pred)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        bboxes = self.bbox_decode(bbox_pred)
        
        mmt_pred = torch.cat(mlvl_mmt_preds)                                                        # mmt
        mmts = self.mmt_decode(mmt_pred) if self.use_mmt_reg else mmt_pred                          # mmt

        results = InstanceData()
        results.bboxes = bboxes
        results.mmts = mmts                                                                         # mmt
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return results

