pretrained = None

# model settings
model = dict(
    type='HEPv2SSD',
    data_preprocessor=dict(
        type='HEPv2DataPreprocessor',
    ),
    backbone=dict(
        type='HEPv2Transformer',
        num_embeds_engphithe=[768, 360, 180],
        embed_dim_engphithe=[768, 384, 384],
        # 
        use_mmt_token=True,
        out_eng=True,
        out_phithe=False,
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
        init_cfg=None,
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
        phithe_nms_thr=3.0,
        score_thr=0.0,
        max_per_img=1,
        init_cfg=None,
    ))
