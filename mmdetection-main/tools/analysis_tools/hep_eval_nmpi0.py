import os
import argparse
# from pycocotools.coco import COCO
import json
import pickle
import datetime
from tqdm import tqdm

import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font

from tools.dataset_converters.root_to_utils import visualization, str_to_numbers


class hep_eval(object):
    """
    转化为高能方式的评估
    """
    def __init__(self, 
                 pkl_path: str, 
                 json_path: str, 
                 # 
                 # num_classes: int = 2, 
                 mmt_min: float = 0.2, 
                 mmt_max: float = 1.4, 
                 # gt_per_image: int = 3, 
                 # 
                 need_excel: int = 0, 
                 excel_path: str = "", 
                 # 
                 need_visual: int = 0, 
                 visual_path: str = "", 
                 visual_ignore_gt: bool = False, 
                 visual_ignore_pred: bool = False, 
                 ):
        self.pkl_path = pkl_path
        self.json_path = json_path
        # 
        # self.num_classes = num_classes
        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        # self.gt_per_image = gt_per_image
        # 
        self.need_excel = need_excel
        self.excel_path = excel_path
        # 
        self.need_visual = need_visual
        self.visual_path = visual_path
        self.visual_ignore_gt = visual_ignore_gt
        self.visual_ignore_pred = visual_ignore_pred

        if self.json_path[-5:] == '.json':
            print("Now loading json ...")
            json_t1 = datetime.datetime.now()

            self.ann_coco = json.load(open(self.json_path, 'r'))                # json.load()

            json_t2 = datetime.datetime.now()
            print("[json]   : open \"{}\" successfully! time: {}".format(
                self.json_path, json_t2 - json_t1))

        else:
            jsons_raw = os.listdir(self.json_path)
            jsons = []
            for temp in jsons_raw:
                if temp[-5:] == '.json': jsons.append(temp)
            jsons.sort()

            data_dict = {}
            data_dict['images'] = []
            data_dict['annotations'] = []

            for temp in jsons:
                print("Now loading json ...")
                json_t1 = datetime.datetime.now()

                json_path_temp = os.path.join(self.json_path, temp)
                ann_coco_temp = json.load(open(json_path_temp, 'r'))            # json.load()

                json_t2 = datetime.datetime.now()
                print("[json]   : open \"{}\" successfully! time: {}".format(
                    json_path_temp, json_t2 - json_t1))

                data_dict['images'] += ann_coco_temp['images']
                data_dict['annotations'] += ann_coco_temp['annotations']
            self.ann_coco = data_dict

        self.len_ann_coco = len(self.ann_coco['images'])

        # self.ann_coco = COCO(self.json)                                         # 读取json文件
        # self.ann_coco_imgids = self.ann_coco.getImgIds()                        # 读取json文件的imgids列表
        # self.ann_coco_annids = self.ann_coco.getAnnIds()                        # 读取json文件的annids列表

        print("Now loading pickle ...")
        pickle_t1 = datetime.datetime.now()

        self.result_all = pickle.load(open(self.pkl_path, 'rb'))                # 读取pkl文件

        pickle_t2 = datetime.datetime.now()
        print("[pickle] : open \"{}\" successfully! time: {}".format(
            self.pkl_path, pickle_t2 - pickle_t1))

        self.len_results_all = len(self.result_all)
        assert self.len_results_all == self.len_ann_coco                        # 只考虑不使用测试数据增强的情况

        print("len_ann_coco: {}; len_results_all: {}".format(self.len_ann_coco, self.len_results_all))
        print()

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

        self.eps = 1e-6
        self.efficiency_list = [100.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]

        self.num_phi_bin = 12
        self.num_the_bin = 12
        self.num_mmt_bin = int((self.mmt_max - self.mmt_min + self.eps) / 0.1)


    def evaluate_mab_mre(self):
        book = Workbook()                                                       # 创建一个新的Excel文件
        sheet = book.active                                                     # 选择或创建一个工作表
        sheet_font = Font(name='Dengxian', size=11, bold=False, italic=False)   # 我们的默认字体
        sheet_col  = ['A', 'B', 'C', 
                      'D', 'E', 'F', 'G', 
                      'H', 'I', 
                      'J', 'K', 'L', 
                      'M', 'N', 'O', 
                      ]
        sheet_head = ['runid', 'evtid', 'image_id', 
                      'gt_label', 'phi_RM', 'the_RM', 'p_RM', 
                      'pred_score', 'pred_label', 
                      'pred_phi [-pi, pi)', 'pred_the [0, pi)', 'angular_bias [0, 180]', 
                      'pred_mmt', 'absolute_error (GeV/c)', 'relative_error_gt (%)', 
                      ]
        for i in range(len(sheet_head)):
            sheet[sheet_col[i] + '1'] = sheet_head[i]                           # 写入'A1', 'B1', ...
            sheet[sheet_col[i] + '1'].font = sheet_font                         # 修改字体

        sheet_i = 2

        gt_phi_all = []
        gt_the_all = []
        gt_mmt_all = []
        score_all = []
        ab_all = []
        re_all = []

        score_gam = []
        ab_gam = []
        re_gam = []

        pred_num_list = []
        pred_num_Nm_list = []

        for event_i in tqdm(range(self.len_ann_coco)):
            # image_id = self.ann_coco_imgids[event_i]                            # pkl文件与json文件的图片顺序一致
            # single_image = self.ann_coco.loadImgs(ids=image_id)[0]              # loadImgs()返回一个list。只需要1张图片
            # ann_id = self.ann_coco.getAnnIds(imgIds=image_id)[0]                # getAnnIds()返回一个list。每张图片只需要1个gt
            # single_gt = self.ann_coco.loadAnns(ids=ann_id)[0]                   # loadAnns()返回一个list。每张图片只需要1个gt

            single_image = self.ann_coco['images'][event_i]
            single_gt = self.ann_coco['annotations'][event_i * 3]               # 固定有1个反中子和2个光子

            image_runid = int(single_image.get('runid', -1))
            image_evtid = int(single_image.get('evtid', -1))
            image_id = int(single_image['id'])

            gt_category = int(single_gt['category_id'])                         # 'category_id' 从1开始计数，label从0开始计数
            gt_phi      = float(single_gt["phi_RM"])
            gt_the      = float(single_gt["the_RM"])
            gt_mmt      = float(single_gt['p_RM'])
            assert gt_category == 1                                             # 检验root_to_json代码与hep_eval代码有无出入
            assert image_id == int(single_gt['image_id'])

            # ##### ##### ##### ##### ##### Gamma1 ##### ##### ##### ##### ##### #

            single_gt_gam1 = self.ann_coco['annotations'][event_i * 3 + 1]

            gt_category_gam1 = int(single_gt_gam1['category_id'])
            gt_phi_gam1      = float(single_gt_gam1["phi_RM"])
            gt_the_gam1      = float(single_gt_gam1["the_RM"])
            gt_mmt_gam1      = float(single_gt_gam1['p_RM'])
            assert gt_category_gam1 == 2
            assert image_id == int(single_gt_gam1['image_id'])

            # ##### ##### ##### ##### ##### Gamma2 ##### ##### ##### ##### ##### #

            single_gt_gam2 = self.ann_coco['annotations'][event_i * 3 + 2]

            gt_category_gam2 = int(single_gt_gam2['category_id'])
            gt_phi_gam2      = float(single_gt_gam2["phi_RM"])
            gt_the_gam2      = float(single_gt_gam2["the_RM"])
            gt_mmt_gam2      = float(single_gt_gam2['p_RM'])
            assert gt_category_gam2 == 2
            assert image_id == int(single_gt_gam2['image_id'])

            # 过滤无效图片
            if gt_mmt < self.mmt_min or gt_mmt > self.mmt_max:
                continue
            if gt_mmt_gam1 < self.mmt_min or gt_mmt_gam1 > self.mmt_max:
                continue
            if gt_mmt_gam2 < self.mmt_min or gt_mmt_gam2 > self.mmt_max:
                continue

            (_image_id, pred_score, pred_label, pred_phi, pred_the, pred_mmt,
             pred_score_gam1, pred_label_gam1, pred_phi_gam1, pred_the_gam1, pred_mmt_gam1,
             pred_score_gam2, pred_label_gam2, pred_phi_gam2, pred_the_gam2, pred_mmt_gam2,
             pred_num, pred_num_Nm) = \
                self.result_to_pred(self.result_all[event_i])
            assert image_id == _image_id

            # mab统计
            angular_bias = self.get_angle(pred_phi, pred_the - 0.5 * np.pi, gt_phi, gt_the - 0.5 * np.pi)

            # mmt统计
            absolute_error = abs(pred_mmt - gt_mmt)
            # relative_error_gt = absolute_error / gt_mmt * 100.0
            # relative_error_pred = absolute_error  / pred_mmt * 100.0
            relative_error_gt = (absolute_error + self.eps) / (gt_mmt + self.eps) * 100.0
            # relative_error_pred = (absolute_error + self.eps) / (pred_mmt + self.eps) * 100.0

            gt_phi_all.append(gt_phi)
            gt_the_all.append(gt_the)
            gt_mmt_all.append(gt_mmt)
            score_all.append(pred_score)
            ab_all.append(angular_bias)
            re_all.append(relative_error_gt)

            row = [image_runid, image_evtid, image_id, 
                   gt_category - 1, gt_phi, gt_the, gt_mmt, 
                   pred_score, pred_label, 
                   pred_phi, pred_the, angular_bias, 
                   pred_mmt, absolute_error, relative_error_gt, 
                   ]
            sheet.append(row)
            for col in sheet_col: sheet[col + str(sheet_i)].font = sheet_font   # 例如image_0对应'A2', 'B2', ...

            sheet_i += 1

            # 双光子交叉匹配
            angular_bias_11 = self.get_angle(pred_phi_gam1, pred_the_gam1 - 0.5 * np.pi, gt_phi_gam1, gt_the_gam1 - 0.5 * np.pi)
            angular_bias_22 = self.get_angle(pred_phi_gam2, pred_the_gam2 - 0.5 * np.pi, gt_phi_gam2, gt_the_gam2 - 0.5 * np.pi)
            angular_bias_12 = self.get_angle(pred_phi_gam1, pred_the_gam1 - 0.5 * np.pi, gt_phi_gam2, gt_the_gam2 - 0.5 * np.pi)
            angular_bias_21 = self.get_angle(pred_phi_gam2, pred_the_gam2 - 0.5 * np.pi, gt_phi_gam1, gt_the_gam1 - 0.5 * np.pi)
            if angular_bias_12 + angular_bias_21 < angular_bias_11 + angular_bias_22:
                pred_score_gam_, pred_label_gam_, pred_phi_gam_, pred_the_gam_, pred_mmt_gam_ = pred_score_gam1, pred_label_gam1, pred_phi_gam1, pred_the_gam1, pred_mmt_gam1
                pred_score_gam1, pred_label_gam1, pred_phi_gam1, pred_the_gam1, pred_mmt_gam1 = pred_score_gam2, pred_label_gam2, pred_phi_gam2, pred_the_gam2, pred_mmt_gam2
                pred_score_gam2, pred_label_gam2, pred_phi_gam2, pred_the_gam2, pred_mmt_gam2 = pred_score_gam_, pred_label_gam_, pred_phi_gam_, pred_the_gam_, pred_mmt_gam_

                angular_bias_11 = angular_bias_21
                angular_bias_22 = angular_bias_12

            absolute_error_11 = abs(pred_mmt_gam1 - gt_mmt_gam1)
            relative_error_11 = (absolute_error_11 + self.eps) / (gt_mmt_gam1 + self.eps) * 100.0
            absolute_error_22 = abs(pred_mmt_gam2 - gt_mmt_gam2)
            relative_error_22 = (absolute_error_22 + self.eps) / (gt_mmt_gam2 + self.eps) * 100.0

            score_gam.append(pred_score_gam1)
            ab_gam.append(angular_bias_11)
            re_gam.append(relative_error_11)

            score_gam.append(pred_score_gam2)
            ab_gam.append(angular_bias_22)
            re_gam.append(relative_error_22)

            pred_num_list.append(pred_num)
            pred_num_Nm_list.append(pred_num_Nm)

            # ##### ##### ##### ##### ##### Gamma1 ##### ##### ##### ##### ##### #

            row = [image_runid, image_evtid, image_id, 
                   gt_category_gam1 - 1, gt_phi_gam1, gt_the_gam1, gt_mmt_gam1, 
                   pred_score_gam1, pred_label_gam1, 
                   pred_phi_gam1, pred_the_gam1, angular_bias_11, 
                   pred_mmt_gam1, absolute_error_11, relative_error_11, 
                   ]
            sheet.append(row)
            for col in sheet_col: sheet[col + str(sheet_i)].font = sheet_font   # 例如image_0对应'A2', 'B2', ...

            sheet_i += 1

            # ##### ##### ##### ##### ##### Gamma2 ##### ##### ##### ##### ##### #

            row = [image_runid, image_evtid, image_id, 
                   gt_category_gam2 - 1, gt_phi_gam2, gt_the_gam2, gt_mmt_gam2, 
                   pred_score_gam2, pred_label_gam2, 
                   pred_phi_gam2, pred_the_gam2, angular_bias_22, 
                   pred_mmt_gam2, absolute_error_22, relative_error_22, 
                   ]
            sheet.append(row)
            for col in sheet_col: sheet[col + str(sheet_i)].font = sheet_font   # 例如image_0对应'A2', 'B2', ...

            sheet_i += 1
            # 双光子交叉匹配 END

        num_valid_event = (sheet_i - 2) // 3                                    # 固定有1个反中子和2个光子
        print("num valid event:", num_valid_event)
        print()

        np_gt_phi_all = np.array(gt_phi_all)
        np_gt_the_all = np.array(gt_the_all)
        np_gt_mmt_all = np.array(gt_mmt_all)
        np_score_all = np.array(score_all)
        np_ab_all = np.array(ab_all)
        np_re_all = np.array(re_all)

        print("mean_angular_bias:", np.mean(np_ab_all))
        print("mean_relative_error:", np.mean(np_re_all))
        print()

        mab_with_efficiency_list = []
        for efficiency in self.efficiency_list:
            num_with_efficiency = int(num_valid_event * efficiency / 100.0)     # 计算保留多少事例
            indices = np.argsort(np_score_all)[-num_with_efficiency:]           # 找到被保留事例的索引
            mab_with_efficiency = np.mean(np_ab_all[indices])                   # 求平均
            mab_with_efficiency_list.append(mab_with_efficiency)

        print("efficiency_list:", self.efficiency_list)
        print("mab_with_efficiency_list:", mab_with_efficiency_list)
        print()

        # 光子测量
        print("# ##### ##### ##### ##### ##### Gamma ##### ##### ##### ##### ##### #")
        print()

        np_score_gam = np.array(score_gam)
        np_ab_gam = np.array(ab_gam)
        np_re_gam = np.array(re_gam)

        print("mean_angular_bias:", np.mean(np_ab_gam))
        print("mean_relative_error:", np.mean(np_re_gam))
        print()

        mab_with_efficiency_list = []
        for efficiency in self.efficiency_list:
            num_with_efficiency = int(2 * num_valid_event * efficiency / 100.0)     # 计算保留多少事例
            indices = np.argsort(np_score_gam)[-num_with_efficiency:]           # 找到被保留事例的索引
            mab_with_efficiency = np.mean(np_ab_gam[indices])                   # 求平均
            mab_with_efficiency_list.append(mab_with_efficiency)

        print("efficiency_list:", self.efficiency_list)
        print("mab_with_efficiency_list:", mab_with_efficiency_list)
        print()

        print("pred_num: 0-{}, 1-{}, 2-{}, 3--{}".format(
            pred_num_list.count(0),
            pred_num_list.count(1),
            pred_num_list.count(2),
            pred_num_list.count(3),
        ))
        print("pred_num_Nm: 0-{}, 1--{}, 2-{}, 3-{}".format(
            pred_num_Nm_list.count(0),
            pred_num_Nm_list.count(1),
            pred_num_Nm_list.count(2),
            pred_num_Nm_list.count(3),
        ))

        print("# ##### ##### ##### ##### ##### Gamma ##### ##### ##### ##### ##### #")
        print()
        # 光子测量 END

        self.get_bin_mean('phi', -np.pi,       np.pi,        self.num_phi_bin, np_gt_phi_all, np_ab_all, np_re_all)
        self.get_bin_mean('the', 0,            np.pi,        self.num_the_bin, np_gt_the_all, np_ab_all, np_re_all)
        self.get_bin_mean('mmt', self.mmt_min, self.mmt_max, self.num_mmt_bin, np_gt_mmt_all, np_ab_all, np_re_all)

        if self.need_excel:
            book.save(self.excel_path)
            book.close()
            print("[openpyxl] : write to \"{}\" successfully!".format(self.excel_path))


    def result_to_pred(self, result):
        image_h, image_w = result["img_shape"]
        image_id = result["img_id"]
        pred_scores = result["pred_instances"]["scores"]          # 以下均已按scores降序排列！
        pred_labels = result["pred_instances"]["labels"]
        pred_bboxes = result["pred_instances"]["bboxes"]
        pred_mmts   = result["pred_instances"]["mmts"]            # 要求必须含有动量预测

        pred_num = 0
        pred_num_Nm = 1

        if len(pred_labels) == 0:
            pred_score = pred_score_gam1 = pred_score_gam2 = -self.eps
            pred_label = pred_label_gam1 = pred_label_gam2 = 0
            pred_phi = pred_phi_gam1 = pred_phi_gam2 = 0.0
            pred_the = pred_the_gam1 = pred_the_gam2 = 0.5 * np.pi
            pred_mmt = pred_mmt_gam1 = pred_mmt_gam2 = -self.eps
            pred_num = 0
        else:
            # 必须至少有1个反中子和2个光子，且我们假设反中子的优先级低于光子
            if len(pred_labels) == 1:
                pred_scores = pred_scores[[0, 0, 0]]
                pred_labels = pred_labels[[0, 0, 0]]
                pred_bboxes = pred_bboxes[[0, 0, 0]]
                pred_mmts   = pred_mmts[[0, 0, 0]]
                pred_num = 1
            elif len(pred_labels) == 2:
                pred_scores = pred_scores[[0, 1, 1]]
                pred_labels = pred_labels[[0, 1, 1]]
                pred_bboxes = pred_bboxes[[0, 1, 1]]
                pred_mmts   = pred_mmts[[0, 1, 1]]
                pred_num = 2
            else:
                pred_num = 3

            # 如果预测结果中反中子数量为0，则我们假设第2顺位为反中子
            if pred_labels.tolist().count(0) == 0:
                pred_labels[2] = 0
                pred_num_Nm = 0
            # 如果预测结果中光子数量为0，则我们假设第0、1顺位为光子
            elif pred_labels.tolist().count(1) == 0:
                pred_labels[0] = 1
                pred_labels[1] = 1
                pred_num_Nm = 3
            # 如果预测结果中光子数量为1，则我们假设第1个反中子实际为光子
            elif pred_labels.tolist().count(1) == 1:
                if pred_labels[0] == 1:
                    pred_labels[1] = 1
                else:
                    pred_labels[0] = 1
                pred_num_Nm = 2
            else:
                pred_num_Nm = 1

            pred_mask_gam = (pred_labels == 1)
            pred_scores_gam = pred_scores[pred_mask_gam]
            pred_labels_gam = pred_labels[pred_mask_gam]
            pred_bboxes_gam = pred_bboxes[pred_mask_gam]
            pred_mmts_gam   = pred_mmts[pred_mask_gam]

            # ##### ##### ##### ##### ##### Gamma1 ##### ##### ##### ##### ##### #

            pred_score_gam1 = float(pred_scores_gam[0])
            pred_label_gam1 = int(pred_labels_gam[0])
            pred_bbox_gam1 = pred_bboxes_gam[0]
            pred_x_ctr_gam1 = (pred_bbox_gam1[0] + pred_bbox_gam1[2]) * 0.5
            pred_y_ctr_gam1 = (pred_bbox_gam1[1] + pred_bbox_gam1[3]) * 0.5
            pred_phi_gam1 = float(pred_x_ctr_gam1 / image_w * 2 * np.pi - np.pi)
            pred_the_gam1 = float(pred_y_ctr_gam1 / image_h * np.pi)
            pred_mmt_gam1 = float(pred_mmts_gam[0, 0])

            # ##### ##### ##### ##### ##### Gamma2 ##### ##### ##### ##### ##### #

            pred_score_gam2 = float(pred_scores_gam[1])
            pred_label_gam2 = int(pred_labels_gam[1])
            pred_bbox_gam2 = pred_bboxes_gam[1]
            pred_x_ctr_gam2 = (pred_bbox_gam2[0] + pred_bbox_gam2[2]) * 0.5
            pred_y_ctr_gam2 = (pred_bbox_gam2[1] + pred_bbox_gam2[3]) * 0.5
            pred_phi_gam2 = float(pred_x_ctr_gam2 / image_w * 2 * np.pi - np.pi)
            pred_the_gam2 = float(pred_y_ctr_gam2 / image_h * np.pi)
            pred_mmt_gam2 = float(pred_mmts_gam[1, 0])

            # ##### ##### ##### ##### #####   Nm   ##### ##### ##### ##### ##### #

            pred_mask_Nm = (pred_labels == 0)
            pred_scores = pred_scores[pred_mask_Nm]
            pred_labels = pred_labels[pred_mask_Nm]
            pred_bboxes = pred_bboxes[pred_mask_Nm]
            pred_mmts   = pred_mmts[pred_mask_Nm]

            pred_score = float(pred_scores[0])                              # 注意必须强制类型转换，否则写入excel会报错
            pred_label = int(pred_labels[0])
            pred_bbox = pred_bboxes[0]                                      # 格式为[xmin, ymin, xmax, ymax]
            pred_x_ctr = (pred_bbox[0] + pred_bbox[2]) * 0.5
            pred_y_ctr = (pred_bbox[1] + pred_bbox[3]) * 0.5
            pred_phi = float(pred_x_ctr / image_w * 2 * np.pi - np.pi)      # phi: [-pi, pi)
            pred_the = float(pred_y_ctr / image_h * np.pi)                  # the: [0, pi)
            pred_mmt = float(pred_mmts[0, 0])

        return (image_id, pred_score, pred_label, pred_phi, pred_the, pred_mmt,
                pred_score_gam1, pred_label_gam1, pred_phi_gam1, pred_the_gam1, pred_mmt_gam1,
                pred_score_gam2, pred_label_gam2, pred_phi_gam2, pred_the_gam2, pred_mmt_gam2,
                pred_num, pred_num_Nm)


    def get_bin_mean(self, hint, gt_min, gt_max, num_bin, gt_array, ab_array, re_array):
        len_bin = (gt_max - gt_min) / num_bin
        num_with_bin_list = []
        mab_with_bin_list = []
        mre_with_bin_list = []
        for i in range(num_bin):
            bin_min = i * len_bin + gt_min
            bin_max = (i + 1) * len_bin + gt_min
            bin_mask = (gt_array >= bin_min) & (gt_array < bin_max)

            num_with_bin_list.append(int(np.sum(bin_mask)))
            mab_with_bin_list.append(np.mean(ab_array[bin_mask]))
            mre_with_bin_list.append(np.mean(re_array[bin_mask]))

        print("{}: min {}, max {}, num {}".format(hint, gt_min, gt_max, num_bin))
        print("num_with_bin_list:", num_with_bin_list)
        print("mab_with_bin_list:", mab_with_bin_list)
        print("mre_with_bin_list:", mre_with_bin_list)
        print()


    def get_angle(self, phi_1, the_1, phi_2, the_2):
        # 球坐标系中夹角计算。输入：phi_1, the_1, phi_2, the_2 in rad。输出：angle_deg in deg。
        vector1 = (np.cos(phi_1) * np.cos(the_1), np.sin(phi_1) * np.cos(the_1), np.sin(the_1))
        vector2 = (np.cos(phi_2) * np.cos(the_2), np.sin(phi_2) * np.cos(the_2), np.sin(the_2))
        product = np.clip(np.dot(vector1, vector2), a_min=-1.0, a_max=1.0)
        angle_deg = np.degrees(np.arccos(product))
        assert angle_deg >= 0.0 and angle_deg <= 180.0
        return angle_deg


    def visual(self):
        while True:
            input_str = input("Input image ids:\n")

            if input_str[:4] == 'None' or input_str[:4] == 'exit':
                break
            elif input_str[:5] == 'range':
                numbers = str_to_numbers(input_str[5:], dtype=int)
                image_id_list = [i for i in range(*numbers)]
            else:
                image_id_list = str_to_numbers(input_str, dtype=int)

            for image_id in tqdm(image_id_list):
                ind = image_id - 1
                single_pred = None if self.visual_ignore_pred else self.result_all[ind]

                single_image = self.ann_coco['images'][ind]
                single_gt = self.ann_coco['annotations'][ind * 3]               # 固定有1个反中子和2个光子

                # TODO: visualization with both Nm and Gamma
                visualization(
                    single_image = single_image, 
                    single_gt = single_gt, 
                    single_pred = single_pred, 
                    output_dir = self.visual_path, 
                    with_hint = False, 
                )

            if True: print()


    def main(self):
        if self.need_visual <= 1: self.evaluate_mab_mre()
        if self.need_visual >= 1: self.visual()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/results_ep12.pkl", help = "pkl path")
    parser.add_argument("--json_path", type = str, default = "./data/HEP2COCO/bbox_scale_10/Nm_1m__b00000001__e00100000.json", help = "json path")
    # 
    # parser.add_argument("--num_classes", type = int, default = 2, help = "")
    parser.add_argument("--mmt_min", type = float, default = 0.2, help = "")
    parser.add_argument("--mmt_max", type = float, default = 1.4, help = "")
    # parser.add_argument("--gt_per_image", type = int, default = 1, help = "")
    # 
    parser.add_argument("--need_excel", type = int, default = 0, help = "")
    parser.add_argument("--excel_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/results_ep12.xlsx", help = "excel path")
    # 
    parser.add_argument("--need_visual", type = int, default = 0, help = "0: only eval; 1: eval and visual; 2: only visual")
    parser.add_argument("--visual_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/", help = "visualization output path")
    parser.add_argument("--visual_ignore_gt", type = int, default = 0, help = "whether to ignore gt when visualizing")
    parser.add_argument("--visual_ignore_pred", type = int, default = 0, help = "whether to ignore pred when visualizing")
    opt = parser.parse_args()

    hep_eval(
        pkl_path = opt.pkl_path, 
        json_path = opt.json_path, 
        # 
        # num_classes = opt.num_classes, 
        mmt_min = opt.mmt_min, 
        mmt_max = opt.mmt_max, 
        # gt_per_image = opt.gt_per_image, 
        # 
        need_excel = opt.need_excel, 
        excel_path = opt.excel_path, 
        # 
        need_visual = opt.need_visual, 
        visual_path = opt.visual_path, 
        visual_ignore_gt = opt.visual_ignore_gt, 
        visual_ignore_pred = opt.visual_ignore_pred, 
    ).main()

