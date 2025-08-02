# Default:
# python ./tools/dataset_converters/root_to_json.py
# 
# Others:
# python ./tools/dataset_converters/root_to_json.py --srcroot "./data/BESIII_training_sample/Nm_1m.root" --df_prefix "Nm_1m" --fn_prefix "Nm_1m"
# python ./tools/dataset_converters/root_to_json.py --srcroot "./data/BESIII_training_sample/Nm_reweight_mom.root" --df_prefix "Nm_rew" --fn_prefix "Nm_rew"
# python ./tools/dataset_converters/root_to_json.py --srcroot "./data/BESIII_training_sample/Nm_mergepi0_tof_train.root" --df_prefix "Nm_pi0" --fn_prefix "Nm_pi0" --split_size 100000 --Nm_scale 10.0 --Gamma_scale 5.0

import os
import argparse
import json
import datetime
from tqdm import tqdm

import numpy as np
import uproot

from tools.dataset_converters.root_to_utils import *


def create_single_obj(
    width,
    height,
    easy_scale,
    image_i,
    ann_i,
    category_id,
    flag_cc_i,
    flag_SB_i,
    p_RM_i,
    phi_RM_i,
    the_RM_i,
):
    x_ctr_float, y_ctr_float, w_int, h_int, _r5, _r6 = phithe_to_xywh_np(phi_RM_i, the_RM_i, width=width, height=height)

    w_ex = w_int * easy_scale
    h_ex = h_int * easy_scale
    xmin_ex = x_ctr_float - w_ex * 0.5
    xmax_ex = x_ctr_float + w_ex * 0.5
    ymin_ex = y_ctr_float - h_ex * 0.5
    ymax_ex = y_ctr_float + h_ex * 0.5
    area_ex = w_ex * h_ex

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
    single_obj['flag_cc'] = flag_cc_i
    single_obj['flag_SB'] = flag_SB_i
    single_obj['p_RM'] = p_RM_i
    single_obj['phi_RM'] = phi_RM_i
    single_obj['the_RM'] = the_RM_i

    return single_obj


def root_to_json(
    srcroot: str,
    destroot: str,
    df_prefix: str = "bbox_scale_10",
    fn_prefix: str = "Nm_1m",
    split_size: int = 100000,
    width: int = 960,
    height: int = 480,
    Nm_scale: float = 10.0,
    Gamma_scale: float = 0.0,
):
    # 如果srcroot本身是一个文件
    if srcroot[-5:] == '.root':
        _path, _file = os.path.split(srcroot)                               # 分割文件夹与文件
        srcroot = _path
        srcfiles = [_file, ]
        num_srcfiles = 1

    # 否则列举srcroot中的文件名，选出其中拓展名为'.root'的，按顺序排列，共1229个
    else:
        srcfiles_raw = os.listdir(srcroot)
        srcfiles = []
        for temp in srcfiles_raw:
            if temp[-5:] == '.root': srcfiles.append(temp)
        srcfiles.sort()
        num_srcfiles = len(srcfiles)

    print("num_srcfiles:", num_srcfiles)

    # json文件的文件夹名
    if len(df_prefix) > 0:
        destfolder = os.path.join(destroot, df_prefix)
    else:
        destfolder = os.path.join(destroot, 'Nm{}x_Gamma{}x'.format(int(Nm_scale), int(Gamma_scale)))

    if not os.path.exists(destfolder):
        os.makedirs(destfolder)

    destfilename_list = []

    # 创建一个json文件的内存副本
    # json文件的基础信息
    data_dict = {}
    info = {"description": "HEP2COCO 2025 Dataset",
            "url": "",
            "version": "0.02a",
            "year": 2025,
            "contributor": "Hongtian Yu, Yangu Li, Yuyang Huang, Zhaoyang Yuan",
            "date_created": "2025/06/01"}
    categories = [
        {'id': 1, 'name': 'Nm',    'supercategory': 'Nm'},
        {'id': 2, 'name': 'Gamma', 'supercategory': 'Gamma'},
    ]
    data_dict['info'] = info
    data_dict['categories'] = categories
    data_dict['images'] = []
    data_dict['annotations'] = []

    # 特别注意：json相关编号均从1开始计数。
    #     single_image['file_name'], single_image['id'], single_obj['image_id'], single_obj['id'],
    #     以及json文件的begin-end命名均采用十进制8位数image_id。
    image_i = 0
    ann_i = 0
    data_dict_checked = False

    split_size_remain = split_size
    image_i_begin_ind = 0
    ann_i_begin_ind = 0

    for srcfile_i, srcfile in enumerate(srcfiles):                          # 对于srcfiles里的每个root文件
        t1 = datetime.datetime.now()

        tree = uproot.open(os.path.join(srcroot, srcfile) + ":TreeAna")

        t2 = datetime.datetime.now()
        print("[uproot] : open file successfully! time: {}".format(t2 - t1))

        data = tree.arrays(array_cache=None, library="np")                  # do not use a cache

        t3 = datetime.datetime.now()
        print("[numpy]  : read data successfully! time: {}".format(t3 - t2))

        # for key, value in data.items():
        #     print(key, len(value))

        runid_array = data['runid']
        evtid_array = data['evtid']
        flag_cc_array = data['flag_cc']
        flag_SB_array = data['flag_SB']
        p_RM_array = data['p_RM']
        phi_RM_array = data['phi_RM']
        the_RM_array = data['the_RM']

        if Gamma_scale > 0.0:
            p_gam1_array = data['p_gam1']
            phi_gam1_array = data['phi_gam1']
            the_gam1_array = data['the_gam1']
            p_gam2_array = data['p_gam2']
            phi_gam2_array = data['phi_gam2']
            the_gam2_array = data['the_gam2']
        else:
            p_gam1_array = None
            phi_gam1_array = None
            the_gam1_array = None
            p_gam2_array = None
            phi_gam2_array = None
            the_gam2_array = None

        n_hit_array = data['n_hit']
        m_eng_arrays = data['m_eng']
        m_phi_arrays = data['m_phi']
        m_the_arrays = data['m_the']
        m_time_arrays = data['m_time']

        num_events = len(n_hit_array)
        print("num_events:", num_events)

        # 检查root文件长度
        assert num_events == len(runid_array)
        assert num_events == len(evtid_array)
        assert num_events == len(flag_cc_array)
        assert num_events == len(flag_SB_array)
        assert num_events == len(p_RM_array)
        assert num_events == len(phi_RM_array)
        assert num_events == len(the_RM_array)

        if Gamma_scale > 0.0:
            assert num_events == len(p_gam1_array)
            assert num_events == len(phi_gam1_array)
            assert num_events == len(the_gam1_array)
            assert num_events == len(p_gam2_array)
            assert num_events == len(phi_gam2_array)
            assert num_events == len(the_gam2_array)

        assert num_events == len(m_eng_arrays)
        assert num_events == len(m_phi_arrays)
        assert num_events == len(m_the_arrays)
        assert num_events == len(m_time_arrays)

        num_events_remain = num_events
        event_i_begin_ind = 0

        while True:
            # 计算event_i的头尾
            event_i_range = min(num_events_remain, split_size_remain)
            event_i_end_ind = event_i_begin_ind + event_i_range

            # 对于每张图片
            for event_i in tqdm(range(event_i_begin_ind, event_i_end_ind)):

                # 检查溢出
                if image_i > 99999999 or ann_i > 99999999:
                    raise NotImplementedError

                # 注意此处必须先进行强制类型转换，否则json.dump()会报错不支持的数据类型
                # flag_cc: +1为正粒子，0为光子，-1为反粒子
                # flag_SB: 1为中子，0为光子
                runid_i = int(runid_array[event_i])
                evtid_i = int(evtid_array[event_i])
                flag_cc_i = int(flag_cc_array[event_i])
                flag_SB_i = int(flag_SB_array[event_i])
                p_RM_i = float(p_RM_array[event_i])
                phi_RM_i = float(phi_RM_array[event_i])
                the_RM_i = float(the_RM_array[event_i])

                # 目前不需要四动量
                # px_RM_i = float(p4_RM_array[event_i].member('fP').member('fX'))
                # py_RM_i = float(p4_RM_array[event_i].member('fP').member('fY'))
                # pz_RM_i = float(p4_RM_array[event_i].member('fP').member('fZ'))
                # E_RM_i = float(p4_RM_array[event_i].member('fE'))

                if Gamma_scale > 0.0:
                    p_gam1_i = float(p_gam1_array[event_i])
                    phi_gam1_i = float(phi_gam1_array[event_i])
                    the_gam1_i = float(the_gam1_array[event_i])
                    p_gam2_i = float(p_gam2_array[event_i])
                    phi_gam2_i = float(phi_gam2_array[event_i])
                    the_gam2_i = float(the_gam2_array[event_i])

                n_hit_i = int(n_hit_array[event_i])
                m_eng_array = m_eng_arrays[event_i].astype(float)
                m_phi_array = m_phi_arrays[event_i].astype(float)
                m_the_array = m_the_arrays[event_i].astype(float)
                m_time_array = m_time_arrays[event_i].astype(int)

                # 根据time, the, phi进行多级升序排序
                sort_indices = np.lexsort((m_phi_array, m_the_array, m_time_array))
                m_eng_array = m_eng_array[sort_indices]
                m_phi_array = m_phi_array[sort_indices]
                m_the_array = m_the_array[sort_indices]
                m_time_array = m_time_array[sort_indices]

                image_i += 1

                x_ctr_array, y_ctr_array, w_array, h_array, xmin_array, ymax_array = phithe_to_xywh_np(m_phi_array, m_the_array, width=width, height=height)
                xmax_array = xmin_array + w_array
                ymin_array = ymax_array - h_array
                xyxy_array = np.vstack([xmin_array, ymin_array, xmax_array, ymax_array]).T

                single_image = {}
                # e.g. Nm_1m__ro0__ev0__00000001.png
                single_image['file_name'] = "{}__ro{}__ev{}__{:08}.png".format(fn_prefix, srcfile_i, event_i, image_i)
                single_image['runid'] = runid_i
                single_image['evtid'] = evtid_i
                single_image['id'] = image_i
                single_image['width'] = width
                single_image['height'] = height
                single_image['n_hit'] = n_hit_i
                single_image['m_eng'] = m_eng_array.tolist()
                single_image['m_phi'] = m_phi_array.tolist()
                single_image['m_the'] = m_the_array.tolist()
                single_image['xyxy'] = xyxy_array.astype(int).tolist()
                single_image['m_time'] = m_time_array.tolist()
                data_dict['images'].append(single_image)

                if Nm_scale > 0.0:
                    ann_i += 1
                    single_obj = create_single_obj(width, height, Nm_scale, image_i, ann_i, 1, 
                                                   flag_cc_i, flag_SB_i, p_RM_i, phi_RM_i, the_RM_i)
                    data_dict['annotations'].append(single_obj)

                if Gamma_scale > 0.0:
                    ann_i += 1
                    single_obj = create_single_obj(width, height, Gamma_scale, image_i, ann_i, 2, 
                                                   0, 0, p_gam1_i, phi_gam1_i, the_gam1_i)
                    data_dict['annotations'].append(single_obj)

                if Gamma_scale > 0.0:
                    ann_i += 1
                    single_obj = create_single_obj(width, height, Gamma_scale, image_i, ann_i, 2, 
                                                   0, 0, p_gam2_i, phi_gam2_i, the_gam2_i)
                    data_dict['annotations'].append(single_obj)

                # 检查数值
                if not data_dict_checked:
                    print(data_dict)
                    data_dict_checked = True

            split_size_remain -= event_i_range
            num_events_remain -= event_i_range
            event_i_begin_ind = event_i_end_ind

            # 检查是否可以输出json文件
            json_end_flag = (split_size_remain == 0)
            all_end_flag = (srcfile_i == num_srcfiles - 1 and num_events_remain == 0)
            if json_end_flag or all_end_flag:
                # e.g. Nm_1m__b00000001__e00100000.json
                destfilename = "{}__b{:08}__e{:08}.json".format(fn_prefix, image_i_begin_ind + 1, image_i)
                destfilename_list.append(destfilename)
                destfile = os.path.join(destfolder, destfilename)
                t4 = datetime.datetime.now()

                with open(destfile, 'w') as f_out:
                    json.dump(data_dict, f_out)

                t5 = datetime.datetime.now()
                print("[json]   : write to \"{}\" successfully! time: {}".format(destfile, t5 - t4))
                print("ann_i: {} -> {}".format(ann_i_begin_ind + 1, ann_i))
                print()

                data_dict['images'] = []                                    # 重置data_dict['images']
                data_dict['annotations'] = []                               # 重置data_dict['annotations']
                split_size_remain = split_size                              # 重置split_size_remain
                image_i_begin_ind = image_i                                 # 重置image_i_begin_ind
                ann_i_begin_ind = ann_i                                     # 重置ann_i_begin_ind

            # 检查是否遍历完root文件
            if num_events_remain == 0: break

    for temp in destfilename_list:
        print("\'{}\',".format(temp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcroot", type = str, default = "./data/BESIII_training_sample/Nm_1m.root", help = "srcroot")
    parser.add_argument("--destroot", type = str, default = "./data/HEP2COCO/", help = "destroot")
    parser.add_argument("--df_prefix", type = str, default = "bbox_scale_10", help = "destfolder prefix")
    parser.add_argument("--fn_prefix", type = str, default = "Nm_1m", help = "filename prefix")
    parser.add_argument("--split_size", type = int, default = 100000, help = "split size")
    parser.add_argument("--width", type = int, default = 960, help = "width")
    parser.add_argument("--height", type = int, default = 480, help = "height")
    parser.add_argument("--Nm_scale", type = float, default = 10.0, help = "Nm scale")
    parser.add_argument("--Gamma_scale", type = float, default = 0.0, help = "Gamma scale")
    # 
    opt = parser.parse_args()

    root_to_json(
        srcroot = opt.srcroot,
        destroot = opt.destroot,
        df_prefix = opt.df_prefix,
        fn_prefix = opt.fn_prefix,
        split_size = opt.split_size,
        width = opt.width,
        height = opt.height,
        Nm_scale = opt.Nm_scale,
        Gamma_scale = opt.Gamma_scale,
    )

