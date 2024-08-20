_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/_datasets_/hep2coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=4),
    test_cfg=dict(
        score_thr=0.00,
        max_per_img=1))

train_dataloader = dict(
    batch_size=16,
    num_workers=8)
val_dataloader = dict(
    batch_size=16,
    num_workers=2)
test_dataloader = dict(
    batch_size=16,
    num_workers=2)

# init_lr: coco bs=2*8 <=> hep2coco bs=16*4

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001))
