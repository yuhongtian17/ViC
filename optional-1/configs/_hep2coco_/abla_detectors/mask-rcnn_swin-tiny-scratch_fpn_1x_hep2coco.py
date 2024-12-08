_base_ = './mask-rcnn_swin-tiny_fpn_1x_hep2coco.py'

model = dict(
    backbone=dict(
        init_cfg=None))
