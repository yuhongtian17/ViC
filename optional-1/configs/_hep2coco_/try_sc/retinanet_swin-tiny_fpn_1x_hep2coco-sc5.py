_base_ = [
    '../_base_/_hep2coco_models_/retinanet_swin-tiny_fpn.py',
    '../_base_/_try_datasets_/try_sc/hep2coco-sc5_detection.py',
    '../_base_/_hep2coco_schedules_/schedule_1x_rst.py', '../_base_/default_runtime.py'
]
