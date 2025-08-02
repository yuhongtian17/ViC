_base_ = [
    '../../_base_/_hep2seq_models_/hepv2-ssd_trans-base_nofpn.py',
    '../../_base_/_hep2seq_datasets_/abla/hep2seq-1m_detection.py',
    '../../_base_/_hep2seq_schedules_/schedule_1x_ssd_trans.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=None,
    ))
