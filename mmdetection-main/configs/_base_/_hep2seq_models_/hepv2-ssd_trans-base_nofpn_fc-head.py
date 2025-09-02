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
        out_eng=False,
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
        type='HEPv2DenseFCHead',
        num_classes=2,
        in_channels=768,
        # 
        num_shared_fcs=2,
        fc_out_channels=768,
        phithe_pos_thresh=45.0,
        phithe_base=45.0,
        # 
        len_seq=640,
        use_mmt_token=True,
    ))
