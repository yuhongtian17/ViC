# original code by Mingrui Wu

import os
import argparse
import pickle
import datetime
from tqdm import tqdm

import numpy as np
from pycocotools.coco import COCO

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from tools.dataset_converters.root_to_json import visualization


class HEP_visual(object):
    """
    转化为高能方式的评估
    """
    def __init__(self, json, output_dir):
        # 工作路径
        self.json = json
        self.output_dir = output_dir

        self.ann_coco = COCO(self.json)                                         # 读取json文件
        self.ann_coco_imgids = self.ann_coco.getImgIds()                        # 读取json文件的imgids列表
        self.ann_coco_annids = self.ann_coco.getAnnIds()                        # 读取json文件的annids列表

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def visual(self, ind_start, ind_end):
        for ind in range(ind_start, ind_end):
            image_id = self.ann_coco_imgids[ind]
            single_image = self.ann_coco.loadImgs(ids=image_id)[0]

            ann_id = self.ann_coco.getAnnIds(imgIds=image_id)
            gts = self.ann_coco.loadAnns(ids=ann_id)

            visualization(
                single_image = single_image, 
                gts = gts, 
                output_dir = self.output_dir, 
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type = str, default = "./data/HEP2COCO/bbox_scale_5/Nm_1m__s00000001__e00100000.json", help = "json file")  # 目前仅支持单一文件输入
    parser.add_argument("--output_dir", type = str, default = "./work_dirs/visual_gts/", help = "output directory")
    parser.add_argument("--ind_start", type = int, default = 0, help = "index start")
    parser.add_argument("--ind_end", type = int, default = 100, help = "index end")
    opt = parser.parse_args()

    HEP_visual(
        json = opt.json, 
        output_dir = opt.output_dir, 
    ).visual(
        ind_start = opt.ind_start, 
        ind_end = opt.ind_end, 
    )

