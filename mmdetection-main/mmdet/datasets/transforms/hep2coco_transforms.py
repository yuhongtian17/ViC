# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type

from mmdet.datasets.transforms import RandomShift


@TRANSFORMS.register_module()
class RandomCyclicShift(BaseTransform):
    """Similar with ``class RandomShift``.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Args:
        prob (float): Probability of shifts. Defaults to 0.5.
    """

    def __init__(self,
                 prob: float = 0.5,
                 r_shift_ratio: float = 0.5) -> None:
        assert 0 <= prob <= 1
        assert 0 <= r_shift_ratio <= 1
        self.prob = prob
        self.r_shift_ratio = r_shift_ratio

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def cyclic_shift(self, img: np.ndarray, r_shift_px: int) -> np.ndarray:
        h, w = img.shape[:2]
        new_img = img.copy()
        new_img[:, r_shift_px:w] = img[:, 0:(w - r_shift_px)]
        new_img[:, 0:r_shift_px] = img[:, (w - r_shift_px):w]
        return new_img

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if self._random_prob() < self.prob:
            # img_shape = results['img'].shape[:2]
            h, w = results['img'].shape[:2]

            r_shift_px = int(w * self.r_shift_ratio)
            if r_shift_px == 0: r_shift_px = random.randint(0, w)               # 正数，向右平移多少像素
            l_shift_px = r_shift_px - w                                         # 负数，向左平移多少像素

            # shift bboxes
            # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/horizontal_boxes.py
            if results.get('gt_bboxes', None) is not None:
                bboxes = results['gt_bboxes'].tensor
                bboxes_x_ctr = (bboxes[:, 0] + bboxes[:, 2]) * 0.5              # results['gt_bboxes']: xmin, ymin, xmax, ymax
                r_ind = ((bboxes_x_ctr + r_shift_px) < w)                       # 判断向右平移是否未越界

                bboxes[r_ind, 0::2] += r_shift_px
                bboxes[~r_ind, 0::2] += l_shift_px
                results['gt_bboxes'].tensor = bboxes

            # TODO: support mask and semantic segmentation maps.

            # shift masks
            # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/mask/structures.py#L506
            if results.get('gt_masks', None) is not None:
                masks = results['gt_masks'].masks
                masks = masks.transpose((1, 2, 0))
                masks = self.cyclic_shift(masks, r_shift_px)
                masks = masks.transpose((2, 0, 1))
                results['gt_masks'].masks = masks

            # shift segs
            # 似乎是一个与results['img']相同大小的np.ndarray
            if results.get('gt_seg_map', None) is not None:
                results['gt_seg_map'] = self.cyclic_shift(results['gt_seg_map'], r_shift_px)

            # shift img
            results['img'] = self.cyclic_shift(results['img'], r_shift_px)

            results['w_shift'] = w
            results['r_shift_px'] = r_shift_px

        else:
            results['w_shift'] = 0
            results['r_shift_px'] = 0

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'r_shift_ratio={self.r_shift_ratio})'
        return repr_str

