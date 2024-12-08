_base_ = './retinanet_r50_fpn_1x_hep2coco.py'

pretrained = 'data/pretrained/vheat_tiny_512.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        # copied from https://github.com/MzeroMiko/vHeat/blob/main/detection/configs/vheat/mask_rcnn_fpn_coco_tiny.py
        type='MMDET_VHEAT',
        drop_path=0.1,
        post_norm=False,
        depths=(2, 2, 6, 2),
        dims=96,
        # out_indices=(0, 1, 2, 3),
        out_indices=(1, 2, 3),
        img_size=512,
        pretrained=pretrained,
        use_checkpoint=True,
        # ##### ##### swin-tiny! ##### ##### #
        embed_dims=96,
        # depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        with_cp=True, # False,
        convert_weights=True,
        init_cfg=None,
    ),
    # neck=dict(in_channels=[96, 192, 384, 768]))
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))

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
