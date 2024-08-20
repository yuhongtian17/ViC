# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from .utils import weighted_loss


def bboxes_to_phi_the(bboxes, image_w, image_h):
    bboxes_x_ctr = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    bboxes_y_ctr = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    phi = bboxes_x_ctr / image_w * 2 * torch.pi - torch.pi      # phi: [-pi, pi)
    the = bboxes_y_ctr / image_h * torch.pi - 0.5 * torch.pi    # the: [-pi/2, pi/2)
    return phi, the


def points_to_phi_the(points, image_w, image_h):
    points_x = points[:, 0]
    points_y = points[:, 1]
    phi = points_x / image_w * 2 * torch.pi - torch.pi          # phi: [-pi, pi)
    the = points_y / image_h * torch.pi - 0.5 * torch.pi        # the: [-pi/2, pi/2)
    return phi, the


def get_angle(phi_1, the_1, phi_2, the_2):
    vector1 = torch.vstack((torch.cos(phi_1) * torch.cos(the_1), torch.sin(phi_1) * torch.cos(the_1), torch.sin(the_1)))    # (3, n)
    vector2 = torch.vstack((torch.cos(phi_2) * torch.cos(the_2), torch.sin(phi_2) * torch.cos(the_2), torch.sin(the_2)))    # (3, n)
    product = torch.clamp(torch.sum(vector1 * vector2, dim=0), min=-1.0, max=1.0)                                           # (1, n)
    # angle = torch.arccos(product) # nan!
    angle_x4 = (product - 1) ** 2
    return angle_x4


@weighted_loss
def abiou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7,
               iou_weight: float = 1.0,
               alpha: float = 1.0, image_w: int = 960, image_h: int = 480) -> Tensor:
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    pred_phi, pred_the = bboxes_to_phi_the(pred, image_w, image_h)
    target_phi, target_the = bboxes_to_phi_the(target, image_w, image_h)
    ab_ctr = get_angle(pred_phi, pred_the, target_phi, target_the)

    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    # cw = enclose_wh[:, 0]
    # ch = enclose_wh[:, 1]

    # c2 = cw**2 + ch**2 + eps

    # b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    # b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    # b2_x1, b2_y1 = target[:, 0], target[:, 1]
    # b2_x2, b2_y2 = target[:, 2], target[:, 3]

    # left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    # right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    # rho2 = left + right

    # # DIoU
    # dious = ious - rho2 / c2
    # loss = 1 - dious

    enclose_phi_1, enclose_the_1 = points_to_phi_the(enclose_x1y1, image_w, image_h)
    enclose_phi_2, enclose_the_2 = points_to_phi_the(enclose_x2y2, image_w, image_h)
    ab_enclose = get_angle(enclose_phi_1, enclose_the_1, enclose_phi_2, enclose_the_2) + eps

    loss = iou_weight * (1 - ious) + alpha * ab_ctr / ab_enclose
    return loss


@MODELS.register_module()
class ABIoULoss(nn.Module):
    r"""Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 iou_weight: float = 1.0,
                 alpha: float = 1.0, image_w: int = 960, image_h: int = 480) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.iou_weight = iou_weight
        self.alpha = alpha
        self.image_w = image_w
        self.image_h = image_h

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * abiou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            iou_weight=self.iou_weight,
            alpha=self.alpha,
            image_w=self.image_w,
            image_h=self.image_h,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
