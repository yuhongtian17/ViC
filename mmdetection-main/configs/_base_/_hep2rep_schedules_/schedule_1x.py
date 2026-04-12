# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer (verified through experiments)
# ref: https://github.com/open-mmlab/mmdetection/blob/main/projects/ViTDet/configs/lsj-100e_coco-instance.py
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='HEPv3LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
        'custom_keys': {
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone.vic': dict(lr_mult=0.2, decay_mult=0.5),
            'neck': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.cls_convs': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.reg_convs': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.retina_cls': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.retina_reg': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.mmt_reg_convs': dict(lr_mult=0.2, decay_mult=0.5),
            'bbox_head.retina_mmt_reg': dict(lr_mult=0.2, decay_mult=0.5),
        },
    },
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
