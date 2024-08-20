_base_ = '../abla_detectors/retinanet_swin-tiny_fpn_1x_hep2coco.py'

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0)))
