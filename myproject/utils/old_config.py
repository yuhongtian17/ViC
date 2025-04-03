_base_ = [
    '../_base_/_hep2coco_models_/hep-retinanet_vheatk-tiny_fpn.py',
    '../_base_/_hep2coco_datasets_/hep2coco_detection.py',
    '../_base_/_hep2coco_schedules_/schedule_1x_rtf.py', '../_base_/default_runtime.py'
]

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

dataset_type = 'Hep2CocoDataset'
data_root = 'data/HEP2COCO/bbox_scale_10/'
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromHEP', backend_args=backend_args,
         bg_version='black_randn_seed', snr_db=10.0),
    dict(type='HEPLoadAnnotations', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(
        type='HEPPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_ann_files = []
test_dataset_base = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_val2017.json',
    # data_prefix=dict(img='val2017/'),
    ann_file='',
    data_prefix=dict(img='./'),
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args)
test_datasets = []

for test_i in range(0, len(test_ann_files)):
    temp = test_dataset_base.copy()
    temp['ann_file'] = test_ann_files[test_i]
    test_datasets.append(temp)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/dataset_wrappers.py
        type='ConcatDataset',
        datasets=test_datasets))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + test_ann_files[0],
    outfile_prefix='./work_dirs/coco_detection/test')
