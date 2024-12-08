_base_ = '../retinanet_swin-tiny_fpn_1x_hep2coco.py'

pretrained = 'data/pretrained/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
