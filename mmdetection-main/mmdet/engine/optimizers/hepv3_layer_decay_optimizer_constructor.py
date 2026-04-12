# Copyright (c) OpenMMLab. All rights reserved.
"""HEPv3 Layer Decay Optimizer Constructor with custom learning rate for
specific modules.

Supports both layer-wise learning rate decay (from LayerDecayOptimizerConstructor)
and custom_keys for setting different learning rates for specific modules.
"""
import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_hepv3(var_name: str, max_layer_id: int) -> int:
    """Get the layer id for HEPv3 backbone (similar to ViT structure).

    Args:
        var_name (str): The parameter name.
        max_layer_id (int): Maximum layer id.
    Returns:
        int: The layer id for learning rate scaling.
    """
    if var_name.startswith('backbone'):
        if 'patch_embed' in var_name or 'pos_embed' in var_name:
            return 0
        elif 'token_embed' in var_name or 'mmt_token' in var_name:
            return 0
        elif '.blocks.' in var_name:
            layer_id = int(var_name.split('.')[2]) + 1
            return layer_id
        elif 'outnorm' in var_name:
            return max_layer_id
        else:
            return max_layer_id + 1
    else:
        return max_layer_id + 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class HEPv3LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Optimizer constructor for HEPv3 that supports layer-wise decay and
    custom learning rates for specific modules.

    Use paramwise_cfg with:
        - decay_rate, decay_type, num_layers: for layer-wise decay
        - custom_keys: dict of module_name_prefix -> dict(lr_mult=x)
          e.g. custom_keys={'backbone.vic': dict(lr_mult=0.2)}
    """

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """Add all parameters of module to the params list."""
        logger = MMLogger.get_current_instance()

        parameter_groups = {}
        paramwise_cfg = self.paramwise_cfg or {}

        # Extract custom_keys for modules with custom learning rate
        custom_keys = paramwise_cfg.get('custom_keys', {})
        num_layers = paramwise_cfg.get('num_layers', 12) + 2
        decay_rate = paramwise_cfg.get('decay_rate', 0.7)
        decay_type = paramwise_cfg.get('decay_type', 'layer_wise')

        logger.info(f'HEPv3LayerDecayOptimizerConstructor paramwise_cfg: '
                    f'{paramwise_cfg}')
        logger.info(f'Custom keys (lr override): {list(custom_keys.keys())}')

        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this param matches any custom_keys prefix
            custom_cfg = None
            for prefix, cfg in custom_keys.items():
                if name.startswith(prefix) or name == prefix:
                    custom_cfg = cfg
                    break

            if custom_cfg is not None:
                # Use custom learning rate and weight_decay
                custom_lr_mult = custom_cfg.get('lr_mult', 1.0)
                if 'weight_decay' in custom_cfg:
                    custom_weight_decay = custom_cfg['weight_decay']
                else:
                    custom_weight_decay = weight_decay * custom_cfg.get(
                        'decay_mult', 1.0)
                group_name = f'custom_lr{custom_lr_mult}_wd{custom_weight_decay}'
                if group_name not in parameter_groups:
                    parameter_groups[group_name] = {
                        'weight_decay': custom_weight_decay,
                        'params': [],
                        'param_names': [],
                        'lr_scale': custom_lr_mult,
                        'group_name': group_name,
                        'lr': self.base_lr * custom_lr_mult,
                    }
                parameter_groups[group_name]['params'].append(param)
                parameter_groups[group_name]['param_names'].append(name)
                continue

            # Use layer-wise decay logic
            if name.startswith('backbone.blocks') and 'norm' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            elif 'pos_embed' in name:
                group_name = 'no_decay_pos_embed'
                this_weight_decay = 0
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_layer_id_for_hepv3(name, paramwise_cfg.get(
                'num_layers', 12))
            group_name = f'layer_{layer_id}_{group_name}'
            this_lr_multi = 1.

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - 1 - layer_id)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr * this_lr_multi,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')

        params.extend(parameter_groups.values())
