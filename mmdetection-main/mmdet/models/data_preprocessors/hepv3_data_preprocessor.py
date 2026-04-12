# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
import torch
from mmengine.utils import is_seq_of
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS

from mmdet.models.data_preprocessors import DetDataPreprocessor


@MODELS.register_module()
class HEPv3DataPreprocessor(DetDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data_seq = self.cast_data(data)  # type: ignore
        inputs_seq = data_seq['inputs_seq']

        # Process data with `pseudo_collate`.
        if is_seq_of(inputs_seq, torch.Tensor):
            batch_inputs = []
            for batch_input in inputs_seq:
                batch_inputs.append(batch_input)
            inputs_seq = torch.stack(batch_inputs)
        # Process data with `default_collate`.
        elif isinstance(inputs_seq, torch.Tensor):
            pass
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': (inputs_seq, inputs), 'data_samples': data_samples}

