# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmengine.model import BaseDataPreprocessor
from mmengine.utils import is_seq_of

from mmdet.registry import MODELS


@MODELS.register_module()
class HEPv2DataPreprocessor(BaseDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs, data_samples = data['inputs'], data['data_samples']

        # Process data with `pseudo_collate`.
        if is_seq_of(inputs, torch.Tensor):
            batch_inputs = []
            for batch_input in inputs:
                batch_inputs.append(batch_input)
            inputs = torch.stack(batch_inputs)
        # Process data with `default_collate`.
        elif isinstance(inputs, torch.Tensor):
            pass
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        return {'inputs': inputs, 'data_samples': data_samples}

