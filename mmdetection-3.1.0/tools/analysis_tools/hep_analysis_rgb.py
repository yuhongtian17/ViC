import os
import argparse
import json
import datetime
from tqdm import tqdm

import math
import numpy as np
import uproot
import mmcv

from tools.dataset_converters.root_to_json import *


def hep_analysis_rgb(
    srcfile,
    width = 960,
    height = 480,
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

    # flag_cc_array = data['flag_cc']
    # flag_SB_array = data['flag_SB']

    # p_RM_array = data['p_RM']
    # phi_RM_array = data['phi_RM']
    # the_RM_array = data['the_RM']

    n_hit_array = data['n_hit']
    m_eng_arrays = data['m_eng']
    m_phi_arrays = data['m_phi']
    m_the_arrays = data['m_the']

    num_event = len(n_hit_array)
    print("num_event:", num_event)

    # 一般计算
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    r2_sum = 0.0
    g2_sum = 0.0
    b2_sum = 0.0
    pixel_sum = num_event * width * height

    r_mean = r_sum / pixel_sum
    g_mean = g_sum / pixel_sum
    b_mean = b_sum / pixel_sum
    r2_mean = r2_sum / pixel_sum
    g2_mean = g2_sum / pixel_sum
    b2_mean = b2_sum / pixel_sum

    # 特殊计算
    sp_r2_sum = 0.0
    sp_g2_sum = 0.0
    sp_b2_sum = 0.0
    r_pixel_sum = 0
    g_pixel_sum = 0
    b_pixel_sum = 0

    eps = 1e-6

    sp_r_mean = r_sum / (r_pixel_sum + eps)
    sp_g_mean = g_sum / (g_pixel_sum + eps)
    sp_b_mean = b_sum / (b_pixel_sum + eps)
    sp_r2_mean = sp_r2_sum / (r_pixel_sum + eps)
    sp_g2_mean = sp_g2_sum / (g_pixel_sum + eps)
    sp_b2_mean = sp_b2_sum / (b_pixel_sum + eps)

    mean_flag = False

    while True:
        for i in tqdm(range(num_event)):

            # 注意此处必须先进行强制类型转换，否则json.dump()会报错不支持的数据类型
            # flag_cc_i = int(flag_cc_array[i])
            # flag_SB_i = int(flag_SB_array[i])

            # p_RM_i = float(p_RM_array[i])
            # phi_RM_i = float(phi_RM_array[i])
            # the_RM_i = float(the_RM_array[i])

            n_hit_i = int(n_hit_array[i])
            m_eng_array = m_eng_arrays[i].astype(float)
            m_phi_array = m_phi_arrays[i].astype(float)
            m_the_array = m_the_arrays[i].astype(float)

            r_array, g_array, b_array = eng_to_rgb_np(m_eng_array)
            r_inds = (r_array > 0)
            g_inds = (g_array > 0)
            b_inds = (b_array > 0)

            _r1, _r2, w_array, h_array, xmin_array, ymax_array = phithe_to_xywh_np(m_phi_array, m_the_array, width=width, height=height)

            if not mean_flag:
                r_sum += np.sum(r_array * w_array * h_array)
                g_sum += np.sum(g_array * w_array * h_array)
                b_sum += np.sum(b_array * w_array * h_array)
                r_pixel_sum += np.sum(w_array[r_inds] * h_array[r_inds])
                g_pixel_sum += np.sum(w_array[g_inds] * h_array[g_inds])
                b_pixel_sum += np.sum(w_array[b_inds] * h_array[b_inds])
            else:
                empty_pixel_temp = width * height - np.sum(w_array * h_array)
                r2_sum += (np.sum(((r_array - r_mean) ** 2) * w_array * h_array) + (r_mean ** 2) * empty_pixel_temp)
                g2_sum += (np.sum(((g_array - g_mean) ** 2) * w_array * h_array) + (g_mean ** 2) * empty_pixel_temp)
                b2_sum += (np.sum(((b_array - b_mean) ** 2) * w_array * h_array) + (b_mean ** 2) * empty_pixel_temp)
                sp_r2_sum += np.sum(((r_array[r_inds] - sp_r_mean) ** 2) * w_array[r_inds] * h_array[r_inds])
                sp_g2_sum += np.sum(((g_array[g_inds] - sp_g_mean) ** 2) * w_array[g_inds] * h_array[g_inds])
                sp_b2_sum += np.sum(((b_array[b_inds] - sp_b_mean) ** 2) * w_array[b_inds] * h_array[b_inds])

        if not mean_flag:
            r_mean = r_sum / pixel_sum
            g_mean = g_sum / pixel_sum
            b_mean = b_sum / pixel_sum
            sp_r_mean = r_sum / (r_pixel_sum + eps)
            sp_g_mean = g_sum / (g_pixel_sum + eps)
            sp_b_mean = b_sum / (b_pixel_sum + eps)
            print("pixel_sum, r_pixel_sum, g_pixel_sum, b_pixel_sum: [{}, {}, {}, {}]".format(
                pixel_sum, r_pixel_sum, g_pixel_sum, b_pixel_sum))
            print("r_mean, g_mean, b_mean: [{}, {}, {}]".format(
                r_mean, g_mean, b_mean))
            print("sp_r_mean, sp_g_mean, sp_b_mean: [{}, {}, {}]".format(
                sp_r_mean, sp_g_mean, sp_b_mean))
            mean_flag = True
        else:
            r2_mean = r2_sum / pixel_sum
            g2_mean = g2_sum / pixel_sum
            b2_mean = b2_sum / pixel_sum
            sp_r2_mean = sp_r2_sum / (r_pixel_sum + eps)
            sp_g2_mean = sp_g2_sum / (g_pixel_sum + eps)
            sp_b2_mean = sp_b2_sum / (b_pixel_sum + eps)
            print("r_std, g_std, b_std: [{}, {}, {}]".format(
                np.sqrt(r2_mean), np.sqrt(g2_mean), np.sqrt(b2_mean)))
            print("sp_r_std, sp_g_std, sp_b_std: [{}, {}, {}]".format(
                np.sqrt(sp_r2_mean), np.sqrt(sp_g2_mean), np.sqrt(sp_b2_mean)))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcfile", type = str, default = "./data/BESIII_training_sample/Nm_1m.root", help = "srcfile")
    parser.add_argument("--width", type = int, default = 960, help = "width")
    parser.add_argument("--height", type = int, default = 480, help = "height")
    opt = parser.parse_args()

    hep_analysis_rgb(
        srcfile = opt.srcfile,
        width = opt.width,
        height = opt.height,
    )

