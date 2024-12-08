_base_ = '../abla_detectors/atss_swin-tiny_fpn_1x_hep2coco.py'

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=False,
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
