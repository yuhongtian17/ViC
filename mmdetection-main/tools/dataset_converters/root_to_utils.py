import datetime
from typing import Optional

import numpy as np
import mmcv


def str_to_numbers(
    s,
    dtype = float,
    default_return = None,
):
    """
    将字符串s转换为数值列表。
    如果不包含任何数值，则返回None。

    （代码写法比较简陋）
    """
    num_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']          # 数字字符
    if dtype == float: num_chars.append('.')
    i_begin = 0                                                             # 字符串开头指示符
    i_end = 0                                                               # 字符串结束指示符
    numbers = []

    s += '<E>'                                                              # 使s必定以非数字字符结尾，确保最后一个数字能够输出

    for i, c in enumerate(s):
        if c in num_chars:                                                  # 如果是数字字符
            i_end = i + 1                                                   # ！将结束指示符向后推一格！
        else:                                                               # 否则不是数字字符
            if i_end > i_begin:                                             # 此时，如果有效字符串长度大于0
                numbers.append(dtype(s[i_begin:i_end]))                     # 将字符串转换为浮点数存入bbox_scales列表
            i_begin = i + 1                                                 # ！将开头指示符向后推一格！

    if len(numbers) == 0: numbers = default_return
    print("numbers:", numbers)
    return numbers


def create_bg_np(
    h, w, c,
    bg_version: Optional[str] = None,
    snr_db: float = 10.0,
) -> np.ndarray:
    # mean & std from ImageNet:
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    # rgb to bgr
    mean.reverse()
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

    index_low  = (eng_log10 <  -2.3)                                                    # 小于5e-3 GeV
    index_mid  = (eng_log10 >= -2.3) & (eng_log10 <  -1.3)
    index_high =                       (eng_log10 >= -1.3)                              # 大于5e-2 GeV

    r = np.zeros_like(eng)
    g = np.zeros_like(eng)
    b = np.zeros_like(eng)

    if index_low.any():
        rgb_norm =   np.clip((eng_log10[index_low] + 3.3), a_min=0, a_max=1) ** 0.5     # [-3.3, -2.3) -> [0, 1)
        b[index_low]  = rgb_norm * 255 + 1
    if index_mid.any():
        rgb_norm =           (eng_log10[index_mid] + 2.3)                    ** 0.6     # [-2.3, -1.3) -> [0, 1)
        g[index_mid]  = rgb_norm * 255 + 1
    if index_high.any():
        #                                    in rad: np.arctan(3) = 1.2490457723982544
        rgb_norm = np.arctan((eng_log10[index_high] + 1.3) * 2.5) / 1.2490457723982544  # [-1.3, -0.1) -> [0, 1)
        r[index_high] = rgb_norm * 255 + 1

    return r, g, b


def load_rgb(
    single_image: dict,
    bg_version: Optional[str] = None,
    snr_db: float = 10.0,
) -> np.ndarray:
    """
    加载RGB值。要求：
    single_image具有这些键: 'height', 'width', 'n_hit', 'm_eng', 'xyxy'
    """

    h = single_image['height']
    w = single_image['width']
    c = 3  # B, G, R

    img_np = create_bg_np(
        h, w, c, 
        bg_version = bg_version, 
        snr_db = snr_db, 
    )

    n_hit = single_image['n_hit']
    m_eng = single_image['m_eng']
    xyxy = single_image['xyxy']

    r_array, g_array, b_array = eng_to_rgb_np(np.array(m_eng))  # Here `m_eng` is a list!

    for i in range(n_hit):
        r, g, b = r_array[i], g_array[i], b_array[i]
        xmin, ymin, xmax, ymax = xyxy[i]
        img_np[ymin:ymax, xmin:xmax, 0:3] = np.array([b, g, r])

    return img_np


def visualization(
    single_image: dict,
    single_gt: dict = None,
    single_pred: dict = None,
    output_dir: str = "./",
    with_hint: bool = True,
) -> np.ndarray:
    """可视化函数（图像保存到本地）
    要求：
    single_image具有这些键: 'file_name', 'height', 'width', 'n_hit', 'm_eng', 'xyxy'
    single_gt   具有这些键: 'p_RM', 'bbox'
    single_pred 具有这些键: 'pred_instances.bboxes'
    """
    file_name = single_image['file_name']

    t1 = datetime.datetime.now()

    img_np = load_rgb(single_image, bg_version='white')

    t2 = datetime.datetime.now()

    mmcv.imwrite(img_np, output_dir + "raw_" + file_name)
    if with_hint: print("[numpy]  : write BGR value successfully! time: {}".format(t2 - t1))

    # 可视化gt框
    if single_gt is not None:
        gt_eng = single_gt['p_RM']
        output_img_path = output_dir + "gt_{:04}MeV_".format(int(gt_eng * 1000)) + file_name

        x1, y1, w1, h1 = single_gt['bbox']
        bboxes = np.array([[x1, y1, x1 + w1, y1 + h1]])

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

