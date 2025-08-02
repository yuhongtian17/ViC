_base_ = [
    '../../_base_/_hep2seq_models_/hepv2-ssd_trans-base_nofpn.py',
    '../../_base_/_hep2seq_datasets_/abla/hep2seq-rew_detection.py',
    '../../_base_/_hep2seq_schedules_/schedule_1x_ssd_trans.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=None,
    ),
    bbox_head=dict(
        mmt_base=0.7,
        mmt_min=0.2,
        mmt_max=1.2,
    ))
