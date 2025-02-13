# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Optional
from numpy import ndarray

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptMultiConfig, OptInstanceList)
from mmdet.models.utils import (multi_apply, select_single_mlvl)
from mmdet.structures.bbox import scale_boxes
from .base_dense_head import BaseDenseHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_


# class LayerNorm2d(nn.LayerNorm):
#     def forward(self, x: torch.Tensor):
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         return x


@MODELS.register_module()
class HEPSFHead(BaseDenseHead):

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: ConfigType = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        width: int = 960,
        height: int = 480,
        easy_scale: float = 10.0,
        mmt_mean: float = 0.0,
        mmt_std: float = 1.0,
        mmt_max: float = 1.2,
        loss_mmt_reg: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        self.fp16_enabled = False
        self.eps = 1e-6

        self.reg_decoded_bbox = ('IoU' in loss_bbox.get('type'))
        self.use_sigmoid_bbox = loss_bbox.get('use_sigmoid', False)
        self.width = width
        self.height = height
        self.easy_scale = easy_scale
        self.mmt_mean = mmt_mean                                                                    # mmt
        self.mmt_std = mmt_std                                                                      # mmt
        self.mmt_max = mmt_max                                                                      # mmt
        self.use_mmt_reg = (loss_mmt_reg is not None)                                               # mmt
        if self.use_mmt_reg:                                                                        # mmt
            self.loss_mmt_reg = MODELS.build(loss_mmt_reg)
            self.use_sigmoid_mmt_reg = loss_mmt_reg.get('use_sigmoid', False)

        # The feature maps from backbones have been normalized before joining the output list.
        # Maybe those from FPNs have not been normalized.
        self.sf_pool = nn.Sequential(
            # LayerNorm2d(self.in_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.sf_cls = nn.Linear(self.in_channels, self.cls_out_channels)
        self.sf_reg = nn.Linear(self.in_channels, 2)
        if self.use_mmt_reg: self.sf_mmt_reg = nn.Linear(self.in_channels, 1)

        self.apply(self._init_weights)
 
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each scale \
            level, each is a 4D-tensor, the channel number is num_points * 4.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
            after classification and regression conv layers, some
            models needs these features like FCOS.
        """
        x = self.sf_pool(x)
        cls_feat = x
        reg_feat = x
        if self.use_mmt_reg: mmt_reg_feat = x                                                       # mmt

        cls_score = self.sf_cls(cls_feat)
        bbox_pred = self.sf_reg(reg_feat)
        if self.use_mmt_reg:                                                                        # mmt
            mmt_reg_pred = self.sf_mmt_reg(mmt_reg_feat)
        else:
            mmt_reg_pred = None

        return cls_score, bbox_pred, mmt_reg_pred


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

        # self._bbox_post_process()
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        return results


    # TODO refactor aug_test
    def aug_test(self,
                 aug_batch_feats: List[Tensor],
                 aug_batch_img_metas: List[List[Tensor]],
                 rescale: bool = False) -> List[ndarray]:
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            aug_batch_img_metas (list[list[dict]]): the outer list indicates
                test-time augs (multiscale, flip, etc.) and the inner list
                indicates images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(
            aug_batch_feats, aug_batch_img_metas, rescale=rescale)

