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
        type='HEPv2DenseTransformerHead',
        num_classes=2,
        in_channels=768,
        # 
        embed_dim=384, # 512,
        depth=8,
        num_heads=12, # 16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        with_cp=True,
        # 
        phithe_pos_thresh=45.0,
        phithe_base=45.0,
        # 
        len_seq=640,
        use_mmt_token=True,
        backbone_out_eng=True,
        backbone_out_phithe=False,
        init_cfg=None,
    ))
