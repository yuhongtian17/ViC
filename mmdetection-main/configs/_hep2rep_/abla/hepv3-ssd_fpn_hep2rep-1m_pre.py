_base_ = [
    '../../_base_/_hep2rep_models_/hepv3-ssd_fpn.py',
    '../../_base_/_hep2rep_datasets_/hep2rep-1m_detection.py',
    '../../_base_/_hep2rep_schedules_/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        mix_mode='pre',                                    # ViC特征在ANT每层的融合位置（'pre' or 'post' or None，None指不进行融合）
    ))
