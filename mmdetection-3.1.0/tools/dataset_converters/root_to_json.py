import os
import argparse
import json
import datetime
from tqdm import tqdm
from typing import Optional, List

import math
import numpy as np
import uproot
import mmcv


def create_bg_np(
    h, w, c,
    bg_version: Optional[str] = None,
    snr_db: float = 30.0,
) -> np.ndarray:
    # mean & std from ImageNet:
    # mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    # rgb to bgr
    # mean.reverse()
    std.reverse()

    snr = 10 ** (snr_db / 20)
    seed = 0

    if bg_version == 'black_randn':
        img_np = np.random.randn(h, w, c) * np.array(std) / snr
    elif bg_version == 'black_randn_seed':
        np.random.seed(seed)
        img_np = np.random.randn(h, w, c) * np.array(std) / snr
    elif bg_version == 'white':
        img_np = np.zeros((h, w, c)) + 255
    else:
        img_np = np.zeros((h, w, c))

    return img_np


def eng_to_rgb_np(eng):
    # if not isinstance(eng, np.ndarray):
    #     eng = np.array([eng])

    eng_log10 = np.log10(eng)

    index_low =            (eng_log10 <  -2.3)                                          # 小于5e-3 GeV
    index_mid = np.vstack([(eng_log10 >= -2.3), (eng_log10 <  -1.3)]).all(axis=0)
    index_high =                                (eng_log10 >= -1.3)                     # 大于5e-2 GeV

    r = np.zeros_like(eng)
    g = np.zeros_like(eng)
    b = np.zeros_like(eng)

    if index_low.any():
        rgb_norm =   np.clip((eng_log10[index_low] + 3.3), a_min=0, a_max=1) ** 0.5     # [-3.3, -2.3) -> [0, 1)
        b[index_low] = rgb_norm * 255 + 1
    if index_mid.any():
        rgb_norm =           (eng_log10[index_mid] + 2.3)                    ** 0.6     # [-2.3, -1.3) -> [0, 1)
        g[index_mid] = rgb_norm * 255 + 1
    if index_high.any():
        #                                    in rad: np.arctan(3) = 1.2490457723982544
        rgb_norm = np.arctan((eng_log10[index_high] + 1.3) * 2.5) / 1.2490457723982544  # [-1.3, -0.1) -> [0, 1)
        r[index_high] = rgb_norm * 255 + 1

    return r, g, b


def load_rgb(
    single_image: dict,
    bg_version: Optional[str] = None,
    snr_db: float = 30.0,
) -> np.ndarray:
    """加载RGB值
    要求：
    single_image具有这些键: 'height', 'width', 'n_hit', 'm_eng', 'xyxy'
    """

    h = single_image['height']
    w = single_image['width']
    c = 3  # B, G, R

    img_np = create_bg_np(h, w, c, bg_version, snr_db)

    n_hit = single_image['n_hit']
    m_eng = single_image['m_eng']
    xyxy = single_image['xyxy']

    r_array, g_array, b_array = eng_to_rgb_np(np.array(m_eng))  # Here `m_eng` is a list!

    for i in range(n_hit):
        r, g, b = r_array[i], g_array[i], b_array[i]
        xmin, ymin, xmax, ymax = xyxy[i]
        img_np[ymin:ymax, xmin:xmax, :] = b, g, r

    return img_np


def visualization(
    single_image: dict,
    gts: List[dict] = None,
    single_pred: dict = None,
    output_dir: str = "./",
    with_hint: bool = True,
) -> np.ndarray:
    """可视化函数（图像保存到本地）
    要求：
    single_image具有这些键: 'file_name', 'height', 'width', 'n_hit', 'm_eng', 'xyxy'
    gts的列表元素具有这些键: 'p_RM', 'bbox'
    single_pred具有这些键: 'pred_instances.bboxes'
    """
    file_name = single_image['file_name']  # e.g. 'Nm_1m_00000001.png'

    t1 = datetime.datetime.now()

    img_np = load_rgb(single_image, bg_version='white')

    t2 = datetime.datetime.now()

    mmcv.imwrite(img_np, output_dir + "raw_" + file_name)
    if with_hint: print("[numpy]  : write BGR value successfully! time: {}".format(t2 - t1))

    # 可视化gt框
    if isinstance(gts, list) and len(gts) > 0:
        gt_eng = gts[0]['p_RM']
        output_img_path = output_dir + "gt_{:04}MeV_".format(int(gt_eng * 1000)) + file_name

        bboxes_list = []
        for gt in gts:
            x1, y1, w1, h1 = gt['bbox']
            bboxes_list.append([x1, y1, x1 + w1, y1 + h1])
        bboxes = np.array(bboxes_list)

        t3 = datetime.datetime.now()

        # https://github.com/open-mmlab/mmcv/blob/main/mmcv/visualization/image.py
        img_np = mmcv.imshow_bboxes(
            img = img_np,
            bboxes = bboxes,
            # colors = (0, 0, 255),
            colors = 'red',
            # top_k: int = -1,
            thickness = 2,
            show = False,
            # win_name: str = '',
            # wait_time: int = 0,
            out_file = output_img_path,
        )

        t4 = datetime.datetime.now()
        if with_hint: print("[mmcv]   : write to \"{}\" successfully! time: {}".format(output_img_path, t4 - t3))

    # 可视化pred框
    if single_pred is not None:
        pred_eng = single_pred["pred_instances"].get("engs", [0.0])
        output_img_path = output_dir + "pred_{:04}MeV_".format(int(pred_eng[0] * 1000)) + file_name

        single_pred_bboxes = single_pred["pred_instances"]["bboxes"]
        bboxes = np.array(single_pred_bboxes)

        t5 = datetime.datetime.now()

        # https://github.com/open-mmlab/mmcv/blob/main/mmcv/visualization/image.py
        img_np = mmcv.imshow_bboxes(
            img = img_np,
            bboxes = bboxes,
            # colors = (255, 0, 0),
            colors = 'blue',
            # top_k: int = -1,
            thickness = 2,
            show = False,
            # win_name: str = '',
            # wait_time: int = 0,
            out_file = output_img_path,
        )

        t6 = datetime.datetime.now()
        if with_hint: print("[mmcv]   : write to \"{}\" successfully! time: {}".format(output_img_path, t6 - t5))

    return img_np


def phithe_to_xywh_np(phi, the, width=960, height=480):
    """Turn (phi, the) to (x_ctr, y_ctr, w, h) with numpy

    params:
        phi :           float or 1D numpy.ndarray
        the :           float or 1D numpy.ndarray
        width :         int, default = 960
        height :        int, default = 480

    return:
        x_ctr :         float or 1D numpy.ndarray
        y_ctr :         float or 1D numpy.ndarray
        w_cell :        int or 1D numpy.ndarray
        h_cell :        int or 1D numpy.ndarray
        xmin_cell :     int or 1D numpy.ndarray
        ymax_cell :     int or 1D numpy.ndarray
    """
    phi_np_flag = isinstance(phi, np.ndarray)
    the_np_flag = isinstance(the, np.ndarray)
    assert phi_np_flag == the_np_flag

    # 先转换成1D矢量
    if not phi_np_flag:
        phi = np.array([phi])
        the = np.array([the])
    # 再判断越界
    assert phi.ndim == 1 and np.vstack([phi >= -np.pi, phi < np.pi]).all()
    assert the.ndim == 1 and np.vstack([the >= 0, the < np.pi]).all()

    half_width = width * 0.5
    x_ctr = phi / np.pi * half_width + half_width     # float or 1D numpy.ndarray
    y_ctr = the / np.pi * height                      # 1D numpy.ndarray

    w_px = np.array([
        30, 30, 24, 24, 20, 20,         # empty
        20,                             # empty
        15, 15, 12, 12, 10, 10, 
        10,                             # empty
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        8, 8, 8, 8, 
        8, 8, 8, 8, 8, 
        10,                             # empty
        10, 10, 12, 12, 15, 15, 
        20,                             # empty
        20, 20, 24, 24, 30, 30,         # empty
    ])
    h_px = np.array([
        8, 8, 8, 8, 7, 7,               # empty
        7,                              # empty
        6, 6, 6, 6, 5, 5, 
        5,                              # empty
        5, 5, 5, 5, 5, 
        6, 6, 6, 6, 
        7, 7, 7, 7, 7, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        8, 8, 8, 8, 8, 8, 8, 8, 
        7, 7, 7, 7, 7, 
        6, 6, 6, 6, 
        5, 5, 5, 5, 5, 
        5,                              # empty
        5, 5, 6, 6, 6, 6, 
        7,                              # empty
        7, 7, 8, 8, 8, 8,               # empty
    ])
    hh_px = np.array([
        8, 16, 24, 32, 39, 46, 
        53, 
        59, 65, 71, 77, 82, 87, 
        92, 
        97, 102, 107, 112, 117, 
        123, 129, 135, 141, 
        148, 155, 162, 169, 176, 
        184, 192, 200, 208, 216, 224, 232, 240, 
        248, 256, 264, 272, 280, 288, 296, 304, 
        311, 318, 325, 332, 339, 
        345, 351, 357, 363, 
        368, 373, 378, 383, 388, 
        393, 
        398, 403, 409, 415, 421, 427, 
        434, 
        441, 448, 456, 464, 472, 480, 
    ])

    y_ctr_2D = y_ctr[:, np.newaxis]     # 转换为2D矢量，shape (:, 1)
    hh_px_2D = hh_px[np.newaxis, :]     # 转换为2D矢量，shape (1, :)
    ind = np.sum((y_ctr_2D - hh_px_2D) >= 0, axis=1)
    w_cell = w_px[ind]
    h_cell = h_px[ind]
    xmin_cell = (x_ctr / w_cell).astype(int) * w_cell
    ymax_cell = hh_px[ind]

    if not phi_np_flag:                 # 转换回浮点数/整数！
        x_ctr = float(x_ctr[0])
        y_ctr = float(y_ctr[0])
        w_cell = int(w_cell[0])
        h_cell = int(h_cell[0])
        xmin_cell = int(xmin_cell[0])
        ymax_cell = int(ymax_cell[0])

    return x_ctr, y_ctr, w_cell, h_cell, xmin_cell, ymax_cell


def str_to_floats(s, default_return = None):
    num_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']     # 数字字符
    i_start = 0                                                             # 字符串开头指示符
    i_end = 0                                                               # 字符串结束指示符
    floats = []

    s += '_'                                                                # 使s必定以非数字字符结尾，确保最后一个数字能够输出

    for i, c in enumerate(s):
        if c in num_chars:                                                  # 如果是数字字符
            i_end = i + 1                                                   # ！将结束指示符向后推一格！
        else:                                                               # 否则不是数字字符
            if i_end > i_start:                                             # 此时，如果有效字符串长度大于0
                floats.append(float(s[i_start:i_end]))                      # 将字符串转换为浮点数存入bbox_scales列表
            i_start = i + 1                                                 # ！将开头指示符向后推一格！

    if len(floats) == 0: floats = default_return
    print("floats:", floats)
    return floats


def strides_and_ratios_to_wh(strides, ratios):
    w = np.array([0.0])
    h = np.array([0.0])

    if strides is not None:
        w = (w[:, np.newaxis] + np.array(strides)).flatten()
        h = (h[:, np.newaxis] + np.array(strides)).flatten()

        if ratios is not None:
            h_ratios = np.sqrt(np.array(ratios))
            w_ratios = 1 / h_ratios
            w = (w[:, np.newaxis] * w_ratios).flatten()
            h = (h[:, np.newaxis] * h_ratios).flatten()

    wh = np.hstack((w[:, np.newaxis], h[:, np.newaxis]))
    print("len(wh):", len(wh))
    for single_wh in wh: print(single_wh)
    return wh


def root_to_json(
    srcfile: str,
    destroot: str,
    split_size: int = 100000,
    image_id_offset: int = 0,
    ann_id_offset: int = 0,
    width: int = 960,
    height: int = 480,
    strides: Optional[List[float]] = None,
    ratios: Optional[List[float]] = None,
    scales: List[float] = [5.0],
    cls_ignore: bool = False,
    p_RM_thr: float = 0.0,
):
    t1 = datetime.datetime.now()

    tree = uproot.open(srcfile + ":TreeAna")

    t2 = datetime.datetime.now()
    print("[uproot] : open file successfully! time: {}".format(t2 - t1))

    data = tree.arrays(library="np")

    t3 = datetime.datetime.now()
    print("[numpy]  : read data successfully! time: {}".format(t3 - t2))

    # for key, value in data.items():
    #     print(key, len(value))

    flag_cc_array = data['flag_cc']
    flag_SB_array = data['flag_SB']

    p_RM_array = data['p_RM']
    phi_RM_array = data['phi_RM']
    the_RM_array = data['the_RM']

    n_hit_array = data['n_hit']
    m_eng_arrays = data['m_eng']
    m_phi_arrays = data['m_phi']
    m_the_arrays = data['m_the']

    num_event = len(n_hit_array)
    print("num_event:", num_event)

    split_num = (num_event // split_size) + 1

    _path, _file = os.path.split(srcfile)                                   # 分割文件夹与文件
    _filename, _fileext = os.path.splitext(_file)                           # 分割文件名与文件后缀。最后只需要文件名！

    # json文件的文件夹名
    bbox_scales_str = "bbox_scale"
    for single_scale in scales: bbox_scales_str += ('_' + str(int(single_scale)))
    if cls_ignore: bbox_scales_str += '_nocls'
    if p_RM_thr > 0.0: bbox_scales_str += ('_' + str(int(p_RM_thr * 1000)) + 'MeV')
    destfolder = os.path.join(destroot, bbox_scales_str)

    if not os.path.exists(destfolder):
        os.makedirs(destfolder)

    wh = strides_and_ratios_to_wh(strides, ratios)

    # 特别注意：json相关编号均从1开始计数。
    #     single_image['file_name'], single_image['id'], single_obj['image_id'], single_obj['id'],
    #     以及json文件的se命名均采用十进制8位数image_id。
    #     为保证唯一性，应手动指定合适的偏置。
    image_i = image_id_offset
    ann_i = ann_id_offset
    data_dict_checked = False

    for j in range(split_num):

        # json文件的基础信息
        data_dict = {}
        info = {"description": "HEP2COCO 2024 Dataset",
                "url": "",
                "version": "0.01a",
                "year": 2024,
                "contributor": "Hongtian Yu, Yangu Li, Mingrui Wu, Letian Shen",
                "date_created": "2024/05/01"}
        categories = [
            {'id': 1, 'name': 'Nm', 'supercategory': 'Nm'},
            {'id': 2, 'name': 'Lmdm', 'supercategory': 'Lmdm'},
            {'id': 3, 'name': 'Np', 'supercategory': 'Np'},
            {'id': 4, 'name': 'Lmdp', 'supercategory': 'Lmdp'},
            ]
        data_dict['info'] = info
        data_dict['categories'] = categories
        data_dict['images'] = []
        data_dict['annotations'] = []

        # json文件的详细信息
        i_start = j * split_size
        i_end = min(num_event, (j + 1) * split_size)
        print("i_start: {}, i_end: {}".format(i_start, i_end))              # 0~100000, 100000~200000, ...

        # 验证image_i和ann_i的起始值；验证有效图片数
        image_i_start = image_i + 1
        ann_i_start = ann_i + 1
        valid_image_count = 0

        for i in tqdm(range(i_start, i_end)):

            # 注意此处必须先进行强制类型转换，否则json.dump()会报错不支持的数据类型
            flag_cc_i = int(flag_cc_array[i])
            flag_SB_i = int(flag_SB_array[i])

            p_RM_i = float(p_RM_array[i])
            phi_RM_i = float(phi_RM_array[i])
            the_RM_i = float(the_RM_array[i])

            n_hit_i = int(n_hit_array[i])
            m_eng_array = m_eng_arrays[i].astype(float)
            m_phi_array = m_phi_arrays[i].astype(float)
            m_the_array = m_the_arrays[i].astype(float)

            if p_RM_i > p_RM_thr:
                valid_image_count += 1
            else:
                image_i += 1
                ann_i += len(wh) * len(scales)
                continue

            category_id = 1 if cls_ignore else (flag_cc_i + flag_SB_i + 1)

            _r1, _r2, w_array, h_array, xmin_array, ymax_array = phithe_to_xywh_np(m_phi_array, m_the_array, width=width, height=height)
            xmax_array = xmin_array + w_array
            ymin_array = ymax_array - h_array
            xyxy_array = np.vstack([xmin_array, ymin_array, xmax_array, ymax_array]).T

            image_i += 1

            single_image = {}
            single_image['file_name'] = _filename + "_{:08}".format(image_i) + ".png"
            single_image['id'] = image_i
            single_image['width'] = width
            single_image['height'] = height
            single_image['n_hit'] = n_hit_i
            single_image['m_eng'] = m_eng_array.tolist()
            # single_image['m_phi'] = m_phi_array.tolist()
            # single_image['m_the'] = m_the_array.tolist()
            single_image['xyxy'] = xyxy_array.astype(int).tolist()
            data_dict['images'].append(single_image)

            for single_wh in wh:
                single_w, single_h = single_wh

                for single_scale in scales:
                    x_ctr_float, y_ctr_float, w_int, h_int, _r5, _r6 = phithe_to_xywh_np(phi_RM_i, the_RM_i, width=width, height=height)

                    w_ex = (single_w * single_scale) if (single_w > 0.0) else (w_int * single_scale)
                    h_ex = (single_h * single_scale) if (single_h > 0.0) else (h_int * single_scale)
                    xmin_ex = x_ctr_float - w_ex * 0.5
                    xmax_ex = x_ctr_float + w_ex * 0.5
                    ymin_ex = y_ctr_float - h_ex * 0.5
                    ymax_ex = y_ctr_float + h_ex * 0.5
                    area_ex = w_ex * h_ex

                    ann_i += 1

                    single_obj = {}
                    single_obj['area'] = area_ex
                    single_obj['category_id'] = category_id
                    single_obj['segmentation'] = [[xmin_ex, ymin_ex, 
                                                   xmax_ex, ymin_ex, 
                                                   xmax_ex, ymax_ex, 
                                                   xmin_ex, ymax_ex]]
                    single_obj['iscrowd'] = 0
                    single_obj['bbox'] = xmin_ex, ymin_ex, w_ex, h_ex
                    single_obj['image_id'] = image_i
                    single_obj['id'] = ann_i
                    single_obj['p_RM'] = p_RM_i
                    single_obj['phi_RM'] = phi_RM_i
                    single_obj['the_RM'] = the_RM_i
                    data_dict['annotations'].append(single_obj)

            if not data_dict_checked:
                # 检查数值
                print(data_dict)
                # # 检查变量类型
                # print("\nsingle_image:")
                # for key, value in single_image.items(): print(key, type(value))
                # print("\nsingle_obj:")
                # for key, value in single_obj.items(): print(key, type(value))
                # print("")
                # # 检查可视化
                # visualization(single_image=data_dict['images'][0], gts=data_dict['annotations'])
                data_dict_checked = True

        # 验证image_i和ann_i的结束值；验证有效图片数
        image_i_end = image_i
        ann_i_end = ann_i
        print("image_i_start: {}, image_i_end: {}; ann_i_start: {}, ann_i_end: {}".format(image_i_start, image_i_end, ann_i_start, ann_i_end))
        print("valid_image_count: {}".format(valid_image_count))

        # json文件的文件名
        destfile = os.path.join(destfolder, _filename 
                                + "__s{:08}".format(image_i_start) 
                                + "__e{:08}".format(image_i_end) 
                                + ".json")
        t4 = datetime.datetime.now()

        with open(destfile, 'w') as f_out:
            json.dump(data_dict, f_out)

        t5 = datetime.datetime.now()
        print("[json]   : write to \"{}\" successfully! time: {}".format(destfile, t5 - t4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcfile", type = str, default = "./data/BESIII_training_sample/Nm_1m.root", help = "srcfile")
    parser.add_argument("--destroot", type = str, default = "./data/HEP2COCO/", help = "destroot")
    parser.add_argument("--split_size", type = int, default = 100000, help = "split size")
    parser.add_argument("--image_id_offset", type = int, default = 0, help = "image_id offset")
    parser.add_argument("--ann_id_offset", type = int, default = 0, help = "ann_id offset")
    parser.add_argument("--width", type = int, default = 960, help = "width")
    parser.add_argument("--height", type = int, default = 480, help = "height")
    parser.add_argument("--strides", type = str, default = "", help = "strides")
    parser.add_argument("--ratios", type = str, default = "", help = "ratios")
    parser.add_argument("--scales", type = str, default = "5.0", help = "scales")
    parser.add_argument("--cls_ignore", type = bool, default = False, help = "whether to ignore class metainfo")
    parser.add_argument("--p_RM_thr", type = float, default = 0.0, help = "threshold of p_RM")
    opt = parser.parse_args()

    strides = str_to_floats(opt.strides, None)
    ratios = str_to_floats(opt.ratios, None)
    scales = str_to_floats(opt.scales, [5.0])

    root_to_json(
        srcfile = opt.srcfile,
        destroot = opt.destroot,
        split_size = opt.split_size,
        image_id_offset = opt.image_id_offset,
        ann_id_offset = opt.ann_id_offset,
        width = opt.width,
        height = opt.height,
        strides = strides,
        ratios = ratios,
        scales = scales,
        cls_ignore = opt.cls_ignore,
        p_RM_thr = opt.p_RM_thr,
    )

