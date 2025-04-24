_base_ = [
    '../../_base_/_hep2coco_models_/hep-retinanet_vheatk-tiny_fpn.py',
    '../../_base_/_hep2coco_datasets_/hep2coco-rew_detection.py',
    '../../_base_/_hep2coco_schedules_/schedule_1x_rtf.py', '../../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        mmt_base=0.7,
        mmt_min=0.2,
        mmt_max=1.2,
    ))
