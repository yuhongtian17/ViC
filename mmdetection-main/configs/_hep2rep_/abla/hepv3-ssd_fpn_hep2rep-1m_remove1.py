_base_ = [
    '../../_base_/_hep2rep_models_/hepv3-ssd_fpn.py',
    '../../_base_/_hep2rep_datasets_/hep2rep-1m_detection.py',
    '../../_base_/_hep2rep_schedules_/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        mmt_use_fpn=False,                                  # ViC使用局部注意力回归动量
        mmt_use_gloattn=False,                               # ViC使用全局注意力回归动量
        mmt_label_use_gloattn=False,                        # ViC使用全局注意力回归全局标签
        # 
        phithe_source='mix',                                # phi/the的采信分支（'vic' or 'ant' or 'mix'）
        mmt_source='ant',                                   # 动量的采信分支（'vic' or 'ant' or 'mix'）
    ))
