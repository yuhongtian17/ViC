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
                 mmt_min: float = 0.0, 
                 mmt_max: float = 1.2, 
                 gt_per_image: int = 1, 
                 # 
                 need_excel: int = 0, 
                 excel_path: str = "", 
                 # 
                 need_visual: int = 0, 
                 visual_path: str = "", 
                 visual_gt_ignore: bool = False, 
                 visual_pred_ignore: bool = False, 
                 ):
        self.pkl_path = pkl_path
        self.json_path = json_path
        # 
        # self.num_classes = num_classes
        self.mmt_min = mmt_min
        self.mmt_max = mmt_max
        self.gt_per_image = gt_per_image
        # 
        self.need_excel = need_excel
        self.excel_path = excel_path
        # 
        self.need_visual = need_visual
        self.visual_path = visual_path
        self.visual_gt_ignore = visual_gt_ignore
        self.visual_pred_ignore = visual_pred_ignore

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

        self.score_all = []
        self.ab_all = []
        self.mmt_gt_all = []
        self.mmt_pred_all = []

        self.eps = 1e-6
        self.efficiency_list = [100.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]

        self.len_mmt_bin = 0.10                                                 # 绘制gt-pred图时的单位统计区间长度
        self.num_mmt_bin = int((mmt_max - self.eps) / self.len_mmt_bin + 1)

        print("len_ann_coco: {}; len_results_all: {}".format(self.len_ann_coco, self.len_results_all))
        print("len_mmt_bin: {}; num_mmt_bin: {}".format(self.len_mmt_bin, self.num_mmt_bin))
        print()


    def evaluate_acc_mab_mmt(self):
        book = Workbook()                                                       # 创建一个新的Excel文件
        sheet = book.active                                                     # 选择或创建一个工作表
        sheet_font = Font(name='Dengxian', size=11, bold=False, italic=False)   # 我们的默认字体
        sheet_col  = ['A', 'B', 'C', 
                      'D', 'E', 'F', 'G', 
                      'H', 'I', 
                      'J', 'K', 'L', 
                      'M', 'N', 'O', 'P', 
                      ]
        sheet_head = ['runid', 'evtid', 'image_id', 
                      'gt_label', 'phi_RM', 'the_RM', 'p_RM', 
                      'pred_score', 'pred_label', 
                      'pred_phi [-pi, pi)', 'pred_the [0, pi)', 'angular_bias [0, 180]', 
                      'pred_mmt', 'absolute_error (GeV/c)', 'relative_error_gt (%)', 'relative_error_pred (%)', 
                      ]
        for i in range(len(sheet_head)):
            sheet[sheet_col[i] + '1'] = sheet_head[i]                           # 写入'A1', 'B1', ...
            sheet[sheet_col[i] + '1'].font = sheet_font                         # 修改字体

        sheet_i = 2

        for event_i in tqdm(range(self.len_ann_coco)):
            # image_id = self.ann_coco_imgids[event_i]                            # pkl文件与json文件的图片顺序一致
            # single_image = self.ann_coco.loadImgs(ids=image_id)[0]              # loadImgs()返回一个list。只需要1张图片
            # ann_id = self.ann_coco.getAnnIds(imgIds=image_id)[0]                # getAnnIds()返回一个list。每张图片只需要1个gt
            # single_gt = self.ann_coco.loadAnns(ids=ann_id)[0]                   # loadAnns()返回一个list。每张图片只需要1个gt

            single_image = self.ann_coco['images'][event_i]
            single_gt = self.ann_coco['annotations'][event_i * self.gt_per_image]

            image_runid = int(single_image['runid'])
            image_evtid = int(single_image["evtid"])
            image_id = int(single_image['id'])

            gt_category = int(single_gt['category_id'])
            gt_label    = int(gt_category - 1)                                  # 'category_id' 从1开始计数，label从0开始计数
            gt_phi      = float(single_gt["phi_RM"])
            gt_the      = float(single_gt["the_RM"])
            gt_mmt      = float(single_gt['p_RM'])
            assert image_id == int(single_gt['image_id'])

            # 过滤无效图片
            if gt_category > 1:
                # 理论上必定仅保留gt为反中子的图片。检验root_to_json代码与hep_eval代码有无出入
                print("ERROR! image_id: {}, ann_id: {}".format(single_gt['image_id'], single_gt['id']))
                continue
            if gt_mmt < self.mmt_min or gt_mmt > self.mmt_max:
                continue

            # _image_id, pred_score, pred_label, pred_phi, pred_the, pred_mmt = \
            #     self.result_to_pred(self.result_all[event_i])
            # assert image_id == _image_id

            pred_score = -self.eps
            pred_label = 0
            pred_phi = 0.0
            pred_the = 0.5 * np.pi
            pred_mmt = -self.eps

            for j in range(self.len_results_all // self.len_ann_coco):
                _image_id, _pred_score, _pred_label, _pred_phi, _pred_the, _pred_mmt = \
                    self.result_to_pred(self.result_all[event_i + j * self.len_ann_coco])
                assert image_id == _image_id
                if _pred_score > pred_score:
                    pred_score = _pred_score
                    pred_label = _pred_label
                    pred_phi = _pred_phi
                    pred_the = _pred_the
                    pred_mmt = _pred_mmt

            # mab统计
            angular_bias = self.get_angle(pred_phi, pred_the - 0.5 * np.pi, gt_phi, gt_the - 0.5 * np.pi)

            self.score_all.append(pred_score)
            self.ab_all.append(angular_bias)

            # mmt统计
            absolute_error = abs(pred_mmt - gt_mmt)
            # relative_error_gt = absolute_error / gt_mmt * 100.0
            # relative_error_pred = absolute_error  / pred_mmt * 100.0
            relative_error_gt = (absolute_error + self.eps) / (gt_mmt + self.eps) * 100.0
            relative_error_pred = (absolute_error + self.eps) / (pred_mmt + self.eps) * 100.0

            self.mmt_gt_all.append(gt_mmt)
            self.mmt_pred_all.append(pred_mmt)

            row = [image_runid, image_evtid, image_id, 
                   gt_label, gt_phi, gt_the, gt_mmt, 
                   pred_score, pred_label, 
                   pred_phi, pred_the, angular_bias, 
                   pred_mmt, absolute_error, relative_error_gt, relative_error_pred, 
                   ]
            sheet.append(row)
            for col in sheet_col: sheet[col + str(sheet_i)].font = sheet_font   # 例如image_0对应'A2', 'B2', ...

            sheet_i += 1

        print("valid count:", sheet_i - 2)
        print()

        orig_score_all = np.array(self.score_all)
        orig_ab_all = np.array(self.ab_all)
        print("mean angular_bias:", np.mean(orig_ab_all))

        mab_with_efficiency_list = []
        num_event = len(orig_score_all)
        for efficiency in self.efficiency_list:
            num_event_with_efficiency = int(num_event * efficiency / 100.0)
            indices = np.argsort(orig_score_all)[-num_event_with_efficiency:]
            mab_with_efficiency = np.mean(orig_ab_all[indices])
            mab_with_efficiency_list.append(mab_with_efficiency)

        print("efficiency_list:", self.efficiency_list)
        print("mab_with_efficiency_list:", mab_with_efficiency_list)
        print()

        # 记录所有动量真实值和预测值的列表
        orig_mmt_gt_all = np.array(self.mmt_gt_all)
        orig_mmt_pred_all = np.array(self.mmt_pred_all)

        self.get_gt_pred_mean(orig_mmt_gt_all, orig_mmt_pred_all)

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

        if len(pred_labels) == 0:
            pred_score = -self.eps
            pred_label = 0
            pred_phi = 0.0
            pred_the = 0.5 * np.pi
            pred_mmt = -self.eps
        else:
            if (0 in pred_labels) and (1 in pred_labels):                   # 如果既有反中子又有光子，则过滤光子
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

        return image_id, pred_score, pred_label, pred_phi, pred_the, pred_mmt


    def get_gt_pred_mean(self, mmt_gt_array, mmt_pred_array):
        ae = np.abs(mmt_pred_array - mmt_gt_array)
        # re_gt = ae / mmt_gt_array * 100.0
        # re_pred = ae / mmt_pred_array * 100.0
        re_gt = (ae + self.eps) / (mmt_gt_array + self.eps) * 100.0
        re_pred = (ae + self.eps) / (mmt_pred_array + self.eps) * 100.0
        print("mAE: {} GeV/c".format(np.mean(ae)))
        print("mRE_gt: {} %".format(np.mean(re_gt)))
        print("mRE_pred: {} %".format(np.mean(re_pred)))
        print()

        # 将所有动量真实值和预测值计算它们对应ind
        mmt_gt_ind = self.get_mmt_ind(mmt_gt_array)
        mmt_pred_ind = self.get_mmt_ind(mmt_pred_array)

        # 如果以gt_ind为标准
        gbin_count = []
        # 绘制pred-gt图所需的参数。计算各统计区间内的均值、标准差
        gbin_gt_mean = []
        gbin_gt_std = []
        gbin_pred_mean = []
        gbin_pred_std = []

        for gbin_i in range(self.num_mmt_bin):
            mask_gbin_i = (mmt_gt_ind == gbin_i)
            temp_count = np.sum(mask_gbin_i)
            if temp_count == 0: continue

            temp_gt = mmt_gt_array[mask_gbin_i]
            temp_pred = mmt_pred_array[mask_gbin_i]

            gbin_count.append(temp_count)
            gbin_gt_mean.append(np.mean(temp_gt))
            gbin_gt_std.append(np.std(temp_gt))
            gbin_pred_mean.append(np.mean(temp_pred))
            gbin_pred_std.append(np.std(temp_pred))

        print("gbin_count     :", gbin_count)
        print("gbin_gt_mean   :", gbin_gt_mean)
        print("gbin_gt_std    :", gbin_gt_std)
        print("gbin_pred_mean :", gbin_pred_mean)
        print("gbin_pred_std  :", gbin_pred_std)
        print()

        # 如果以pred_ind为标准
        pbin_count = []
        # 绘制pred-gt图所需的参数。计算各统计区间内的均值、标准差
        pbin_gt_mean = []
        pbin_gt_std = []
        pbin_pred_mean = []
        pbin_pred_std = []

        for pbin_i in range(self.num_mmt_bin):
            mask_pbin_i = (mmt_pred_ind == pbin_i)
            temp_count = np.sum(mask_pbin_i)
            if temp_count == 0: continue

            temp_gt = mmt_gt_array[mask_pbin_i]
            temp_pred = mmt_pred_array[mask_pbin_i]

            pbin_count.append(temp_count)
            pbin_gt_mean.append(np.mean(temp_gt))
            pbin_gt_std.append(np.std(temp_gt))
            pbin_pred_mean.append(np.mean(temp_pred))
            pbin_pred_std.append(np.std(temp_pred))

        print("pbin_count     :", pbin_count)
        print("pbin_gt_mean   :", pbin_gt_mean)
        print("pbin_gt_std    :", pbin_gt_std)
        print("pbin_pred_mean :", pbin_pred_mean)
        print("pbin_pred_std  :", pbin_pred_std)
        print()


    def get_mmt_ind(self, mmt_array):
        mmt_ind = np.clip(mmt_array / self.len_mmt_bin, a_min=0, a_max=self.num_mmt_bin - 0.99)
        mmt_ind = np.array(mmt_ind, dtype=int)
        return mmt_ind


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
                single_pred = None if self.visual_pred_ignore else self.result_all[ind]

                single_image = self.ann_coco['images'][ind]
                single_gt = self.ann_coco['annotations'][ind * self.gt_per_image]

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
        if self.need_visual <= 1: self.evaluate_acc_mab_mmt()
        if self.need_visual >= 1: self.visual()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/results_ep12.pkl", help = "pkl path")
    parser.add_argument("--json_path", type = str, default = "./data/HEP2COCO/bbox_scale_10/Nm_1m__b00000001__e00100000.json", help = "json path")
    # 
    # parser.add_argument("--num_classes", type = int, default = 2, help = "")
    parser.add_argument("--mmt_min", type = float, default = 0.0, help = "")
    parser.add_argument("--mmt_max", type = float, default = 1.2, help = "")
    parser.add_argument("--gt_per_image", type = int, default = 1, help = "")
    # 
    parser.add_argument("--need_excel", type = int, default = 0, help = "")
    parser.add_argument("--excel_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/results_ep12.xlsx", help = "excel path")
    # 
    parser.add_argument("--need_visual", type = int, default = 0, help = "0: only eval; 1: eval and visual; 2: only visual")
    parser.add_argument("--visual_path", type = str, default = "./work_dirs/hep-retinanet_vheatk-tiny_fpn_1x_hep2coco/", help = "visualization output path")
    parser.add_argument("--visual_gt_ignore", type = int, default = 0, help = "whether to ignore gt when visualizing")
    parser.add_argument("--visual_pred_ignore", type = int, default = 0, help = "whether to ignore pred when visualizing")
    opt = parser.parse_args()

    hep_eval(
        pkl_path = opt.pkl_path, 
        json_path = opt.json_path, 
        # 
        # num_classes = opt.num_classes, 
        mmt_min = opt.mmt_min, 
        mmt_max = opt.mmt_max, 
        gt_per_image = opt.gt_per_image, 
        # 
        need_excel = opt.need_excel, 
        excel_path = opt.excel_path, 
        # 
        need_visual = opt.need_visual, 
        visual_path = opt.visual_path, 
        visual_gt_ignore = opt.visual_gt_ignore, 
        visual_pred_ignore = opt.visual_pred_ignore, 
    ).main()

