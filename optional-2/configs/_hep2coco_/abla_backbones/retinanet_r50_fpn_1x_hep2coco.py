_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/_hep2coco_datasets_/hep2coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        type='HEPRetinaHead',
        num_classes=2),
    test_cfg=dict(
        score_thr=0.00,
        max_per_img=1))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001))
