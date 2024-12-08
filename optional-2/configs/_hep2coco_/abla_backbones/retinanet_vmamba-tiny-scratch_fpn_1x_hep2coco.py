_base_ = './retinanet_vmamba-tiny_fpn_1x_hep2coco.py'

model = dict(
    backbone=dict(
        pretrained=None))
