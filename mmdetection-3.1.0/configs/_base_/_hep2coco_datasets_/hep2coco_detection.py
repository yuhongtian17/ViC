# dataset settings
dataset_type = 'Hep2CocoDataset'
data_root = 'data/HEP2COCO/bbox_scale_10/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromHEPeng', backend_args=backend_args,
         bg_version='black_randn', snr_db=10.0),
    dict(type='HEPLoadAnnotations', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='HEPPackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromHEPeng', backend_args=backend_args,
         bg_version='black_randn_seed', snr_db=10.0),
    dict(type='HEPLoadAnnotations', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(
        type='HEPPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

ann_files=[
    'Nm_1m__b00000001__e00100000.json',
    'Nm_1m__b00100001__e00200000.json',
    'Nm_1m__b00200001__e00300000.json',
    'Nm_1m__b00300001__e00400000.json',
    'Nm_1m__b00400001__e00500000.json',
    'Nm_1m__b00500001__e00600000.json',
    'Nm_1m__b00600001__e00700000.json',
    'Nm_1m__b00700001__e00800000.json',
    'Nm_1m__b00800001__e00900000.json',
    'Nm_1m__b00900001__e00986343.json']
train_dataset_base = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_train2017.json',
    # data_prefix=dict(img='train2017/'),
    ann_file='',
    data_prefix=dict(img='./'),
    p_RM_thr=0.0,
    btr_eng_thr=0.0,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
train_datasets = []

for i in range(1, len(ann_files)):
    temp = train_dataset_base.copy()
    temp['ann_file'] = ann_files[i]
    train_datasets.append(temp)

# print(train_datasets)

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/dataset_wrappers.py
        type='ConcatDataset',
        datasets=train_datasets))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='val2017/'),
        ann_file=ann_files[0],
        data_prefix=dict(img='./'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + ann_files[0],
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
