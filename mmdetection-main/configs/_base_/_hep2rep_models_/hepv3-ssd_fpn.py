pretrained = 'data/pretrained/all_pretrained.pth'

# model settings
model = dict(
    type='HEPv3SSD',
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
        out_vic=True,
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
            out_indices=(1, 2, 3),
            img_size=512,
            use_checkpoint=True,
        ),
        mix_mode='post',                                    # ViC特征在ANT每层的融合位置（'pre' or 'post' or None，None指不进行融合）
        output_feature_map_indices=None,                    # 融合层的编号（默认None指全部12层均进行融合）
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768], # [256, 512, 1024, 2048],
        out_channels=256,
        start_level=0, # 1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='HEPv3RetinaHead',
        num_classes=2, # 80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # 
        mmt_in_channels=768,
        mmt_use_fpn=False,                                  # ViC使用局部注意力回归动量
        mmt_use_gloattn=False,                              # ViC使用全局注意力回归动量
        mmt_label_use_gloattn=False,                        # ViC使用全局注意力回归全局标签
        # 
        trans_cfg=dict(
            embed_dim=384, # 512,
            depth=8,
            num_heads=12, # 16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            with_cp=True,
        ),
        # 
        phithe_base=45.0,                                   # positive anchor的锥角范围（含光子的情况改为[45.0, 15.0]）
        phithe_mean=0.0,
        phithe_std=1.0,
        easy_scale=10.0,                                    # 将phi/the转换为bbox时，所在晶体大小向bbox大小的扩大倍数
        # 
        mmt_min=0.0,                                        # 动量最小值
        mmt_max=1.2,                                        # 动量最大值
        mmt_base=1.0,                                       # 动量基准值
        mmt_mean=0.0,
        mmt_std=1.0,
        mmt_encode_mode='base',                             # 动量的编解码方案
        mmt_reg_channels=1,
        # 
        len_seq=640,
        use_hit_token_for_mmt=False,                        # ANT使用hit_token回归动量
        use_mmt_token=True,                                 # ANT使用mmt_token回归动量
        use_mmt_token_for_label=False,                      # ANT使用mmt_token回归全局标签
        backbone_out_eng=True,
        backbone_out_phithe=False,
        # 
        # loss_mmt_label=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        loss_mmt_label=None,
        mmt_label_channels=12,                              # 全局标签的类别数
        # 
        nms_mode='phithe',                                  # ANT的NMS后处理方案（'phithe' or 'bbox'）
        phithe_nms_thr=9.0,                                 # ANT使用phithe NMS时的抑制锥角范围
        score_thr=0.0,
        max_per_img=1,                                      # phithe NMS的最多保留数量
        # 
        phithe_source='mix',                                # phi/the的采信分支（'vic' or 'ant' or 'mix'）
        mmt_source='ant',                                   # 动量的采信分支（'vic' or 'ant' or 'mix'）
        # 
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            # type='PseudoSampler'),  # Focal loss should use PseudoSampler
            type='HEPPseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.00, # 0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1,                                      # bbox NMS的最多保留数量
    ))
