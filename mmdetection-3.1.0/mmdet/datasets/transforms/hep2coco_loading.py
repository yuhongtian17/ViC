# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.formatting import PackDetInputs

from tools.dataset_converters.root_to_utils import load_rgb


@TRANSFORMS.register_module()
class LoadImageFromHEPeng(LoadImageFromFile):
    """Load an image from ``results['m_eng']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    data in ``results['m_eng']``.

    Required Keys:

    - height
    - width
    - n_hit
    - m_eng
    - xyxy
    - m_time

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """
    def __init__(self,
                 # https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/loading.py
                 to_float32: bool = True,
                 with_time: int = 0,
                 bg_version: Optional[str] = None,
                 snr_db: float = 10.0,
                 # json_path: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(to_float32=to_float32, **kwargs)

        self.with_time = with_time
        self.bg_version = bg_version
        self.snr_db = snr_db
        # self.json_path = json_path

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # img = results['img']
        img = load_rgb(
            single_image = results,
            with_time = self.with_time,
            bg_version = self.bg_version,
            snr_db = self.snr_db,
            # json_path = self.json_path,
        )
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class HEPLoadAnnotations(LoadAnnotations):
    def __init__(
            self,
            with_mmt: bool = True,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.with_mmt = with_mmt

    def _load_mmts(self, results: dict) -> None:
        gt_mmt_regs = []
        for instance in results.get('instances', []):
            gt_mmt_regs.append([instance['p_RM'], ])

        if self.box_type is None:
            results['gt_mmt_regs'] = np.array(gt_mmt_regs, dtype=np.float32).reshape((-1, 1))
        else:
            results['gt_mmt_regs'] = torch.tensor(gt_mmt_regs, dtype=torch.float32).reshape((-1, 1))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_mmt:                                                                           # mmt
            self._load_mmts(results)                                                                # mmt
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_mmt={self.with_mmt}, '                                                   # mmt
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class HEPPackDetInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_mmt_regs': 'mmt_regs',
    }

