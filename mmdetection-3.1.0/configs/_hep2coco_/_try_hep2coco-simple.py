_base_ = './try_abc/atss_swin-tiny_fpn_ab1_1x_hep2coco.py'

# dataset settings
dataset_type = 'Hep2CocoDataset'
data_root = 'data/HEP2COCO/bbox_scale_10/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromHEPeng', backend_args=backend_args,
         bg_version='black_randn'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        ann_file='Nm_1m__s00000001__e00100000.json',
        data_prefix=dict(img='./'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
