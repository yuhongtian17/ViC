_base_ = [
    '../../_base_/_hep2seq_models_/hepv2-ssd_trans-base_nofpn_trans-head.py',
    '../../_base_/_hep2seq_datasets_/abla/hep2seq-rew_detection.py',
    '../../_base_/_hep2seq_schedules_/schedule_1x_ssd_trans.py', '../../_base_/default_runtime.py'
]

pretrained = "work_dirs/selfsup_50x.pth"

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint=pretrained),
    ),
    bbox_head=dict(
        mmt_base=0.7,
        mmt_min=0.2,
        mmt_max=1.2,
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint=pretrained),
    ))

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
