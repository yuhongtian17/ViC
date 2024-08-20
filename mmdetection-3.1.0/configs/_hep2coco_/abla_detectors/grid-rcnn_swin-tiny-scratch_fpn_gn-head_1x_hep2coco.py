_base_ = './grid-rcnn_swin-tiny_fpn_gn-head_1x_hep2coco.py'

model = dict(
    backbone=dict(
        init_cfg=None))
