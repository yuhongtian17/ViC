_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/_datasets_/abla_sc/hep2coco-sc80_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

pretrained = 'data/pretrained/swin_tiny_patch4_window7_224.pth'

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True, # False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5),
    bbox_head=dict(
        num_classes=4),
    test_cfg=dict(
        score_thr=0.00,
        max_per_img=1))

train_dataloader = dict(
    batch_size=16,
    num_workers=8)
val_dataloader = dict(
    batch_size=16,
    num_workers=2)
test_dataloader = dict(
    batch_size=16,
    num_workers=2)

# init_lr: coco bs=2*8 <=> hep2coco bs=16*4

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-5, # 1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
