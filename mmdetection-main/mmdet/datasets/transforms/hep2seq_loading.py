# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch

from mmengine.structures import InstanceData
from mmcv.transforms import to_tensor
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import get_box_type

from mmdet.datasets.transforms.loading import LoadImageFromNDArray, LoadAnnotations
from mmdet.datasets.transforms.formatting import PackDetInputs


@TRANSFORMS.register_module()
class LoadSeqFromHEPv2(BaseTransform):

    def __init__(self,
                 to_float32: bool = True,
                 use_random_permutation: bool = False,
                 use_cyclic_phi: bool = False,
                 len_seq: int = 640,
                 eng_top: float = 0.0,
                 hit_mask_eng: float = 0.0,
                 hit_mask_phithe: float = 0.0,
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.use_random_permutation = use_random_permutation
        self.use_cyclic_phi = use_cyclic_phi
        self.len_seq = len_seq

        self.eng_top = eng_top
        self.hit_mask_eng = hit_mask_eng
        self.hit_mask_phithe = hit_mask_phithe

        if backend_args is not None:
            self.backend_args = backend_args.copy()
        else:
            self.backend_args = None

    def check_phi(self, phi):
        byd_min = (phi < -np.pi)
        byd_max = (phi >= np.pi)
        phi[byd_min] = phi[byd_min] + 2 * np.pi
        phi[byd_max] = phi[byd_max] - 2 * np.pi
        return phi

    def create_hit_mask_for_flags(self, flags, eng):
        n_hit = len(flags)

        n_eng_top = max(int(n_hit * self.eng_top), 2)
        eng_top_indices = np.argsort(eng)[-n_eng_top:]

        n_hit_mask_eng = max(int(n_eng_top * self.hit_mask_eng), 1) \
            if self.hit_mask_eng > 0 else 0
        n_hit_mask_phithe = max(int(n_eng_top * self.hit_mask_phithe), 1) \
            if self.hit_mask_phithe > 0 else 0

        arr = np.random.rand(n_eng_top)
        arr_indices_eng     = np.argsort(arr)[0             :n_hit_mask_eng]
        arr_indices_phithe  = np.argsort(arr)[n_hit_mask_eng:n_hit_mask_eng+n_hit_mask_phithe]
        mask_eng            = eng_top_indices[arr_indices_eng]
        mask_phithe         = eng_top_indices[arr_indices_phithe]

        flags[mask_eng]    -= self.len_seq
        flags[mask_phithe] -= (self.len_seq * 2)
        return flags

    def transform(self, results: dict) -> Optional[dict]:
        seq = np.zeros([self.len_seq, 5])
        n_hit_all = []
        # m_eng_all = []
        m_phi_all = []
        m_the_all = []

        n_hit = results['n_hit']
        m_eng = results['m_eng']
        m_phi = results['m_phi']
        m_the = results['m_the']
        m_time = results['m_time']

        repeat = int(self.len_seq / n_hit)
        np_ones = np.ones(n_hit)
        np_eng = np.array(m_eng)
        np_phi = np.array(m_phi)
        np_the = np.array(m_the)
        np_time = np.array(m_time)

        for i in range(repeat):
            if self.use_random_permutation:
                permutation = np.random.permutation(n_hit)
            else:
                permutation = np.arange(n_hit)

            if self.use_cyclic_phi:
                np_phi_new = self.check_phi(np_phi + 2 * np.pi * i / repeat)
            else:
                np_phi_new = np_phi

            if self.eng_top > 0:
                np_flags_new = self.create_hit_mask_for_flags(
                    np_ones + i, np_eng)
            else:
                np_flags_new = np_ones + i

            b = n_hit * i
            e = n_hit * (i + 1)
            seq[b:e] = np.vstack([
                np_flags_new[permutation],
                np_eng[permutation],
                np_phi_new[permutation],
                np_the[permutation],
                np_time[permutation],
            ]).T
            n_hit_all.append(n_hit)
            # m_eng_all += (np_eng[permutation]).tolist()
            m_phi_all += (np_phi[permutation]).tolist()
            m_the_all += (np_the[permutation]).tolist()

        if self.to_float32:
            seq = seq.astype(np.float32)

        results['seq'] = seq
        results['n_hit_all'] = n_hit_all
        # results['m_eng_all'] = m_eng_all
        results['m_phi_all'] = m_phi_all
        results['m_the_all'] = m_the_all

        img = np.zeros([480, 960, 3])
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['scale_factor'] = 1.0

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'use_random_permutation={self.use_random_permutation}, '
        repr_str += f'use_cyclic_phi={self.use_cyclic_phi}, '
        repr_str += f'len_seq={self.len_seq}, '
        repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class HEPv2LoadAnnotations(BaseTransform):

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_phithe: bool = True,
        with_mmt: bool = True,
        with_mmt_label: bool = True,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_phithe = with_phithe
        self.with_mmt = with_mmt
        self.with_mmt_label = with_mmt_label

        self.box_type = None

        if backend_args is not None:
            self.backend_args = backend_args.copy()
        else:
            self.backend_args = None

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_phithes(self, results: dict) -> None:
        gt_phithe_regs = []
        for instance in results.get('instances', []):
            gt_phithe_regs.append([instance['phi_RM'], instance['the_RM']])

        results['gt_phithe_regs'] = np.array(gt_phithe_regs, dtype=np.float32).reshape((-1, 2))

    def _load_mmts(self, results: dict) -> None:
        gt_mmt_regs = []
        for instance in results.get('instances', []):
            gt_mmt_regs.append([instance['p_RM'], ])

        results['gt_mmt_regs'] = np.array(gt_mmt_regs, dtype=np.float32).reshape((-1, 1))

    def _load_mmt_labels(self, results: dict) -> None:
        gt_mmt_labels = []
        for instance in results.get('instances', []):
            gt_mmt_labels.append(instance['p_RM_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_mmt_labels'] = np.array(
            gt_mmt_labels, dtype=np.int64)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_phithe:
            self._load_phithes(results)
        if self.with_mmt:
            self._load_mmts(results)
        if self.with_mmt_label:
            self._load_mmt_labels(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_phithe={self.with_phithe}, '
        repr_str += f'with_mmt={self.with_mmt}, '
        repr_str += f'with_mmt_label={self.with_mmt_label}, '
        repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class HEPv2PackDetInputs(BaseTransform):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_phithe_regs': 'phithe_regs',
        'gt_mmt_regs': 'mmt_regs',
        'gt_mmt_labels': 'mmt_labels',
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor',
                            'n_hit_all', 'm_phi_all', 'm_the_all', )):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'seq' in results:
            seq = results['seq']
            packed_results['inputs'] = to_tensor(seq).contiguous()

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key in results:
                instance_data[self.mapping_table[key]] = to_tensor(results[key])

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class HEPv2PackInputs(BaseTransform):

    mapping_table = {}

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', )):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'seq' in results:
            seq = results['seq']
            packed_results['inputs'] = to_tensor(seq).contiguous()

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key in results:
                instance_data[self.mapping_table[key]] = to_tensor(results[key])

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

