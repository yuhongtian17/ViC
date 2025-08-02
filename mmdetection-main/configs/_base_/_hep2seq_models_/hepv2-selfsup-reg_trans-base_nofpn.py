pretrained = None

# model settings
model = dict(
    type='HEPv2SelfSupervisor',
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
        drop_path_rate=0.0,
        with_cp=True,
        # 
        out_indices=None,
        init_cfg=None,
    ),
    neck=None,
    head=dict(
        type='HEPv2SelfSupervisorREGHead',
        in_channels=768,
        len_seq=640,
        use_mmt_token=True,
        backbone_out_eng=True,
        backbone_out_phithe=False,
        recover_eng=False,
        recover_phithe=True,
        # 
        embed_dim=384,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        with_cp=False,
        # 
        num_classes_eng=12,
        num_classes_phi=24,
        num_classes_the=12,
        init_cfg=None,
    ))
