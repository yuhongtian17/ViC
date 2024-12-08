_base_ = '../abla_detectors/retinanet_swin-tiny_fpn_1x_hep2coco.py'

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='ABIoULoss', loss_weight=2.0,
                       alpha=1.0, image_w=960, image_h=480)))
