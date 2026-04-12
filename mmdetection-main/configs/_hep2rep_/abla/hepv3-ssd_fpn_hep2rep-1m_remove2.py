_base_ = [
    # '../../_base_/_hep2rep_models_/hepv3-ssd_fpn.py',
    '../../_base_/_hep2rep_datasets_/hep2rep-1m_detection.py',
    '../../_base_/_hep2rep_schedules_/schedule_1x.py', '../../_base_/default_runtime.py'
]

pretrained = 'data/pretrained/all_pretrained.pth'

# model settings
model = dict(
    type='HEPv2SSD',
    data_preprocessor=dict(
        type='HEPv3DataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='HEPv3Transformer',
        num_embeds_engphithe=[768, 360, 180],
        embed_dim_engphithe=[768, 384, 384],
        # 
        # use_mmt_token=True,
        out_eng=True,
        out_phithe=False,
        out_vic=False,  # True,
        # 
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_cp=True,
        # 
        out_indices=None,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained),
        vic_cfg=dict(
            type='MMDET_VHEAT',
            feat_fusion_mode='cat',
            drop_path_rate=0.1,
            post_norm=False,
            depths=(2, 2, 6, 2),
            dims=96,
            out_indices=(),  # (1, 2, 3),
            img_size=512,
            use_checkpoint=True,
        ),
        mix_mode='post',                                    # ViC特征在ANT每层的融合位置（'pre' or 'post' or None，None指不进行融合）
        output_feature_map_indices=None,                    # 融合层的编号（默认None指全部12层均进行融合）
    ),
    neck=None,
    bbox_head=dict(
        type='HEPv2DenseHead',
        num_classes=2,
        in_channels=768,
        # 
        use_fc_head=False,
        num_shared_fcs=2,
        fc_out_channels=768,
        # 
        embed_dim=384, # 512,
        depth=8,
        num_heads=12, # 16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        with_cp=True,
        # 
        phithe_base=45.0,
        phithe_mean=0.0,
        phithe_std=1.0,
        easy_scale=10.0,
        # 
        mmt_min=0.0,
        mmt_max=1.2,
        mmt_base=1.0,
        mmt_mean=0.0,
        mmt_std=1.0,
        mmt_encode_mode='base',
        mmt_reg_channels=1,
        # 
        len_seq=640,
        use_mmt_token=True,
        use_hit_token_for_mmt=False,
        backbone_out_eng=True,
        backbone_out_phithe=False,
        # 
        loss_mmt_label=None,
        mmt_label_channels=12,
        # 
        nms_mode='phithe',
        phithe_nms_thr=9.0,
        score_thr=0.0,
        max_per_img=1,
        init_cfg=None,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.00, # 0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1,
    ))
