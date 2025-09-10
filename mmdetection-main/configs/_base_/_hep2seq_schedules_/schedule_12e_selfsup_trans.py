# training schedule
num_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs-1,
        by_epoch=True,
        begin=1,
        end=num_epochs,
        convert_to_iter_based=True)
]

# optimizer
# ref: https://github.com/open-mmlab/mmdetection/blob/main/projects/ViTDet/configs/lsj-100e_coco-instance.py
# ref: https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=4e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    ))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2048)
