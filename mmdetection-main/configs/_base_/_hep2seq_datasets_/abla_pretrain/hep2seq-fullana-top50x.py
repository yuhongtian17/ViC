# dataset settings
dataset_type = 'Hep2SeqDataset'
data_root = 'data/HEP2COCO/Nm_fullana/'

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
    dict(type='LoadSeqFromHEPv2',
         use_random_permutation=True,
         use_cyclic_phi=True,
         len_seq=640,
         eng_top=0.5,
         hit_mask_eng=0.0,
         hit_mask_phithe=0.2),
    dict(type='HEPv2PackInputs'),
]

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

ann_files = [
    'Nm_fullana__b00000001__e00100000.json',
    'Nm_fullana__b00100001__e00200000.json',
    'Nm_fullana__b00200001__e00300000.json',
    'Nm_fullana__b00300001__e00400000.json',
    'Nm_fullana__b00400001__e00500000.json',
    'Nm_fullana__b00500001__e00600000.json',
    'Nm_fullana__b00600001__e00700000.json',
    'Nm_fullana__b00700001__e00800000.json',
    'Nm_fullana__b00800001__e00900000.json',
    'Nm_fullana__b00900001__e01000000.json',
    'Nm_fullana__b01000001__e01100000.json',
    'Nm_fullana__b01100001__e01200000.json',
    'Nm_fullana__b01200001__e01300000.json',
    'Nm_fullana__b01300001__e01400000.json',
    'Nm_fullana__b01400001__e01500000.json',
    'Nm_fullana__b01500001__e01600000.json',
    'Nm_fullana__b01600001__e01700000.json',
    'Nm_fullana__b01700001__e01800000.json',
    'Nm_fullana__b01800001__e01900000.json',
    'Nm_fullana__b01900001__e02000000.json',
    'Nm_fullana__b02000001__e02100000.json',
    'Nm_fullana__b02100001__e02200000.json',
    'Nm_fullana__b02200001__e02300000.json',
    'Nm_fullana__b02300001__e02400000.json',
    'Nm_fullana__b02400001__e02500000.json',
    'Nm_fullana__b02500001__e02600000.json',
    'Nm_fullana__b02600001__e02700000.json',
    'Nm_fullana__b02700001__e02800000.json',
    'Nm_fullana__b02800001__e02900000.json',
    'Nm_fullana__b02900001__e03000000.json',
    'Nm_fullana__b03000001__e03100000.json',
    'Nm_fullana__b03100001__e03200000.json',
    'Nm_fullana__b03200001__e03300000.json',
    'Nm_fullana__b03300001__e03400000.json',
    'Nm_fullana__b03400001__e03500000.json',
    'Nm_fullana__b03500001__e03600000.json',
    'Nm_fullana__b03600001__e03700000.json',
    'Nm_fullana__b03700001__e03800000.json',
    'Nm_fullana__b03800001__e03900000.json',
    'Nm_fullana__b03900001__e04000000.json',
    'Nm_fullana__b04000001__e04100000.json',
    'Nm_fullana__b04100001__e04200000.json',
    'Nm_fullana__b04200001__e04300000.json',
    'Nm_fullana__b04300001__e04400000.json',
    'Nm_fullana__b04400001__e04500000.json',
    'Nm_fullana__b04500001__e04600000.json',
    'Nm_fullana__b04600001__e04700000.json',
    'Nm_fullana__b04700001__e04800000.json',
    'Nm_fullana__b04800001__e04900000.json',
    'Nm_fullana__b04900001__e05000000.json',
    'Nm_fullana__b05000001__e05100000.json',
    'Nm_fullana__b05100001__e05200000.json',
    'Nm_fullana__b05200001__e05300000.json',
    'Nm_fullana__b05300001__e05400000.json',
    'Nm_fullana__b05400001__e05500000.json',
    'Nm_fullana__b05500001__e05600000.json',
    'Nm_fullana__b05600001__e05700000.json',
    'Nm_fullana__b05700001__e05800000.json',
    'Nm_fullana__b05800001__e05900000.json',
    'Nm_fullana__b05900001__e06000000.json',
    'Nm_fullana__b06000001__e06100000.json',
    'Nm_fullana__b06100001__e06200000.json',
    'Nm_fullana__b06200001__e06300000.json',
    'Nm_fullana__b06300001__e06400000.json',
    'Nm_fullana__b06400001__e06500000.json',
    'Nm_fullana__b06500001__e06600000.json',
    'Nm_fullana__b06600001__e06700000.json',
    'Nm_fullana__b06700001__e06800000.json',
    'Nm_fullana__b06800001__e06900000.json',
    'Nm_fullana__b06900001__e07000000.json',
    'Nm_fullana__b07000001__e07100000.json',
    'Nm_fullana__b07100001__e07200000.json',
    'Nm_fullana__b07200001__e07300000.json',
    'Nm_fullana__b07300001__e07400000.json',
    'Nm_fullana__b07400001__e07500000.json',
    'Nm_fullana__b07500001__e07600000.json',
    'Nm_fullana__b07600001__e07700000.json',
    'Nm_fullana__b07700001__e07800000.json',
    'Nm_fullana__b07800001__e07900000.json',
    'Nm_fullana__b07900001__e08000000.json',
    'Nm_fullana__b08000001__e08100000.json',
    'Nm_fullana__b08100001__e08200000.json',
    'Nm_fullana__b08200001__e08300000.json',
    'Nm_fullana__b08300001__e08400000.json',
    'Nm_fullana__b08400001__e08500000.json',
    'Nm_fullana__b08500001__e08600000.json',
    'Nm_fullana__b08600001__e08700000.json',
    'Nm_fullana__b08700001__e08800000.json',
    'Nm_fullana__b08800001__e08900000.json',
    'Nm_fullana__b08900001__e09000000.json',
    'Nm_fullana__b09000001__e09100000.json',
    'Nm_fullana__b09100001__e09200000.json',
    'Nm_fullana__b09200001__e09300000.json',
    'Nm_fullana__b09300001__e09400000.json',
    'Nm_fullana__b09400001__e09500000.json',
    'Nm_fullana__b09500001__e09600000.json',
    'Nm_fullana__b09600001__e09700000.json',
    'Nm_fullana__b09700001__e09800000.json',
    'Nm_fullana__b09800001__e09900000.json',
    'Nm_fullana__b09900001__e10000000.json',
    'Nm_fullana__b10000001__e10100000.json',
    'Nm_fullana__b10100001__e10200000.json',
    'Nm_fullana__b10200001__e10300000.json',
    'Nm_fullana__b10300001__e10400000.json',
    'Nm_fullana__b10400001__e10500000.json',
    'Nm_fullana__b10500001__e10600000.json',
    'Nm_fullana__b10600001__e10700000.json',
    'Nm_fullana__b10700001__e10800000.json',
    'Nm_fullana__b10800001__e10900000.json',
    'Nm_fullana__b10900001__e11000000.json',
    'Nm_fullana__b11000001__e11093276.json',
]
train_dataset_base = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_train2017.json',
    # data_prefix=dict(img='train2017/'),
    ann_file='',
    data_prefix=dict(img='./'),
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
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/dataset_wrappers.py
        type='ConcatDataset',
        datasets=train_datasets))
