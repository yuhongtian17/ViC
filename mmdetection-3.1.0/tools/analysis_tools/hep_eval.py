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


class HEP_eval(object):
    """
    转化为高能方式的评估
    """
    def __init__(self, 
                 pkl, json, 
                 cls_ignore, 
                 output_dir, excel_name, 
                 visual_ind, visual_end, 
                 visual_gt_ignore, visual_pred_ignore, 
                 ):
        # 工作路径
        self.pkl = pkl
        self.json = json
        self.cls_ignore = cls_ignore
        self.output_dir = output_dir
        self.excel_name = excel_name
        self.visual_ind = visual_ind
        self.visual_end = visual_end
        self.visual_gt_ignore = visual_gt_ignore
        self.visual_pred_ignore = visual_pred_ignore

        self.ann_coco = COCO(self.json)                                         # 读取json文件
        self.ann_coco_imgids = self.ann_coco.getImgIds()                        # 读取json文件的imgids列表
        self.ann_coco_annids = self.ann_coco.getAnnIds()                        # 读取json文件的annids列表

        t1 = datetime.datetime.now()

        self.result_all = pickle.load(open(self.pkl, 'rb'))                     # 读取pkl文件

        t2 = datetime.datetime.now()
        print("[pickle] : open file successfully! time: {}".format(t2 - t1))

        self.all_count    = np.array([1e-6] * 4)
        self.right_count  = np.array([1e-6] * 4)

        self.ab_sum       = np.array([0.0] * 4)

        self.ab_countunit = 3.0
        self.ab_countlist = np.array([[0] * int(180.0 / self.ab_countunit + 1)] * 4)
        
        self.ae_sum       = np.array([0.0] * 4)
        self.re_sum       = np.array([0.0] * 4)

        # self.re_countunit = 0.05
        # self.re_countlist = np.array([[0] * int(1.0   / self.re_countunit + 1)] * 4)


    def evaluate_acc_mab_eng(self):
        book = Workbook()                                                       # 创建一个新的Excel文件
        sheet = book.active                                                     # 选择或创建一个工作表
        sheet_font = Font(name='Dengxian', size=11, bold=False, italic=False)   # 我们的默认字体
        sheet_col  = ['A', 'B', 'C', 'D', 'E', 
                      'F', 
                      'G', 'H', 
                      'I', 'J', 'K', 
                      'L', 'M', 'N']
        sheet_head = ['id', 'category_id', 'phi_RM', 'the_RM', 'p_RM', 
                      'pred_score', 
                      'pred_label', 'flag {0, 1}', 
                      'pred_phi [-pi, pi)', 'pred_the [0, pi)', 'angular_bias [0, 180]', 
                      'pred_eng', 'absolute_error (GeV/c)', 'relative_error']
        for i in range(len(sheet_head)):
            sheet[sheet_col[i] + '1'] = sheet_head[i]                           # 写入'A1', 'B1', ...
            sheet[sheet_col[i] + '1'].font = sheet_font                         # 修改字体

        i = 0

        for result_per_event in tqdm(self.result_all):
            image_h, image_w = result_per_event["img_shape"]
            pred_scores = result_per_event["pred_instances"]["scores"]          # 以下均已按scores降序排列！
            pred_labels = result_per_event["pred_instances"]["labels"]
            pred_bboxes = result_per_event["pred_instances"]["bboxes"]
            pred_engs   = result_per_event["pred_instances"].get("engs", [0.0] * len(pred_bboxes))

            if len(pred_bboxes) == 0:
                pred_score = 1e-6
                pred_label = 0
                pred_phi = 0.0
                pred_the = 0.5 * np.pi
                pred_eng = 0.0
            else:
                pred_score = float(pred_scores[0])                              # 注意必须强制类型转换，否则写入excel会报错
                pred_label = 0 if self.cls_ignore else int(pred_labels[0])
                pred_bbox = pred_bboxes[0]                                      # 格式为[xmin, ymin, xmax, ymax]
                pred_x_ctr = (pred_bbox[0] + pred_bbox[2]) * 0.5
                pred_y_ctr = (pred_bbox[1] + pred_bbox[3]) * 0.5
                pred_phi = float(pred_x_ctr / image_w * 2 * np.pi - np.pi)      # phi: [-pi, pi)
                pred_the = float(pred_y_ctr / image_h * np.pi)                  # the: [0, pi)
                pred_eng = float(pred_engs[0])                                  # 注意必须强制类型转换，否则写入excel会报错

            image_id = self.ann_coco_imgids[i]                                  # pkl文件与json文件的图片顺序一致
            ann_id = self.ann_coco.getAnnIds(imgIds=image_id)[0]                # getAnnIds()返回一个list。每张图片只需要一个gt
            gt = self.ann_coco.loadAnns(ids=ann_id)[0]                          # loadAnns()返回一个list。每张图片只需要一个gt

            gt_id       = gt['id']
            gt_category = 1 if self.cls_ignore else gt['category_id']
            gt_phi      = float(gt["phi_RM"])
            gt_the      = float(gt["the_RM"])
            gt_eng      = float(gt['p_RM'])

            gt_label = int(gt_category - 1)                                     # 'category_id' 从1开始计数

            (self.all_count)[gt_label] += 1
            flag = int(pred_label == gt_label)
            if pred_label == gt_label: (self.right_count)[gt_label] += 1        # acc统计

            angular_bias = self.get_angle(pred_phi, pred_the - 0.5 * np.pi, gt_phi, gt_the - 0.5 * np.pi)
            (self.ab_sum)[gt_label] += angular_bias                             # mab统计

            ab_ind = int(angular_bias / self.ab_countunit)
            (self.ab_countlist)[gt_label, ab_ind] += 1                          # ab_countlist

            absolute_error = abs(pred_eng - gt_eng)
            relative_error = absolute_error / gt_eng
            (self.ae_sum)[gt_label] += absolute_error                           # eng统计
            (self.re_sum)[gt_label] += relative_error                           # eng统计

            # re_ind = int(abs_eng_error / self.re_countunit) if (abs_eng_error < 1.0) else -1
            # (self.re_countlist)[gt_label, re_ind] += 1                          # re_countlist

            row = [gt_id, gt_category, gt_phi, gt_the, gt_eng, 
                   pred_score, 
                   pred_label, flag, 
                   pred_phi, pred_the, angular_bias, 
                   pred_eng, absolute_error, relative_error]
            sheet.append(row)
            for col in sheet_col: sheet[col + str(i + 2)].font = sheet_font     # 例如image_0对应'A2', 'B2', ...

            i += 1

        acc = self.right_count / self.all_count
        mab = self.ab_sum / self.all_count
        mae = self.ae_sum / self.all_count
        mre = self.re_sum / self.all_count
        print()
        print("raw count:", self.right_count.astype(int).tolist(), self.all_count.astype(int).tolist())
        print("accuracy:", acc)
        print()
        print("mean angular_bias:", mab)
        print("ab_countlist:", self.ab_countlist.astype(int).tolist())
        print()
        print("mean absolute_error:", mae)
        print("mean relative_error:", mre)
        print()

        output_excel_path = os.path.join(self.output_dir, self.excel_name)
        book.save(output_excel_path)
        book.close()
        print("[openpyxl] : write to \"{}\" successfully!".format(output_excel_path))

        return i


    def get_angle(self, phi_1, the_1, phi_2, the_2):
        # 球坐标系中夹角计算。输入：phi_1, the_1, phi_2, the_2 in rad。输出：angle_deg in deg。
        vector1 = (np.cos(phi_1) * np.cos(the_1), np.sin(phi_1) * np.cos(the_1), np.sin(the_1))
        vector2 = (np.cos(phi_2) * np.cos(the_2), np.sin(phi_2) * np.cos(the_2), np.sin(the_2))
        product = np.clip(np.dot(vector1, vector2), a_min=-1.0, a_max=1.0)
        angle_deg = np.degrees(np.arccos(product))
        assert angle_deg >= 0.0 and angle_deg <= 180.0
        return angle_deg


    def visual(self):
        if self.visual_end <= self.visual_ind:
            self.visual_end = self.visual_ind + 1

        for ind in tqdm(range(self.visual_ind, self.visual_end)):
            single_pred = None if self.visual_pred_ignore else self.result_all[ind]

            image_id = self.ann_coco_imgids[ind]
            single_image = self.ann_coco.loadImgs(ids=image_id)[0]

            ann_id = self.ann_coco.getAnnIds(imgIds=image_id)
            gts = None if self.visual_gt_ignore else self.ann_coco.loadAnns(ids=ann_id)

            visualization(
                single_image = single_image, 
                gts = gts, 
                single_pred = single_pred, 
                output_dir = self.output_dir, 
                with_hint = False, 
            )


    def main(self):
        if self.visual_ind < 0: self.evaluate_acc_mab_eng()
        else: self.visual()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type = str, default = "./work_dirs/retinanet_swin-tiny_fpn_1x_hep2coco/results_ep12.pkl", help = "pkl file")
    parser.add_argument("--json", type = str, default = "./data/HEP2COCO/bbox_scale_10/Nm_1m__s00000001__e00100000.json", help = "json file")  # 目前仅支持单一文件输入
    parser.add_argument("--cls_ignore", type = bool, default = False, help = "ignore class info")
    parser.add_argument("--output_dir", type = str, default = "./work_dirs/retinanet_swin-tiny_fpn_1x_hep2coco/", help = "output directory")
    parser.add_argument("--excel_name", type = str, default = "results_ep12.xlsx", help = "excel filename")
    parser.add_argument("--visual_ind", type = int, default = -1, help = "visual index")
    parser.add_argument("--visual_end", type = int, default = -1, help = "visual index end")
    parser.add_argument("--visual_gt_ignore", type = bool, default = False, help = "ignore gt when visualizing")
    parser.add_argument("--visual_pred_ignore", type = bool, default = False, help = "ignore pred when visualizing")
    opt = parser.parse_args()

    HEP_eval(
        pkl = opt.pkl, 
        json = opt.json, 
        cls_ignore = opt.cls_ignore, 
        output_dir = opt.output_dir, 
        excel_name = opt.excel_name, 
        visual_ind = opt.visual_ind, 
        visual_end = opt.visual_end, 
        visual_gt_ignore = opt.visual_gt_ignore, 
        visual_pred_ignore = opt.visual_pred_ignore, 
    ).main()

