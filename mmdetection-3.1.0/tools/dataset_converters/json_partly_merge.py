# original code by Mingrui Wu

import os
import argparse
import json
import datetime


def json_partly_merge(
    json_srcroot,
    json_destfile,
    part_offset = 0,
    part_num = 25000,
):
    srcfiles = os.listdir(json_srcroot)
    print(srcfiles)
    # >>> ['Nm_1m__s00000001__e00100000.json', 'Np_1m__s02000001__e02100000.json', 'Lmdm_1m__s01000001__e01100000.json', 'Lmdp_1m__s03000001__e03100000.json']

    # 同root_to_json.py
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

    for srcfile in srcfiles:
        json_srcfile = os.path.join(json_srcroot, srcfile)
        with open(json_srcfile, 'r') as f_in:
            t1 = datetime.datetime.now()

            src_content = json.load(f_in)

            t2 = datetime.datetime.now()
            print("[json]   : read from \"{}\" successfully! time: {}".format(json_srcfile, t2 - t1))

            print(len(src_content['images']), len(src_content['annotations']))
            # >>> 100000 100000

            data_dict['images'].extend(src_content['images'][part_offset:(part_offset + part_num)])
            data_dict['annotations'].extend(src_content['annotations'][part_offset:(part_offset + part_num)])

    print(len(data_dict['images']), len(data_dict['annotations']))
    # >>> 100000 100000

    t4 = datetime.datetime.now()

    with open(json_destfile, 'w') as f_out:
        json.dump(data_dict, f_out)

    t5 = datetime.datetime.now()
    print("[json]   : write to \"{}\" successfully! time: {}".format(json_destfile, t5 - t4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_srcroot", type = str, default = "./data/HEP2COCO/bbox_scale_10/valset/", help = "filefolder of old json files")
    parser.add_argument("--json_destfile", type = str, default = "./data/HEP2COCO/bbox_scale_10/val.json", help = "path of new json file")
    parser.add_argument("--part_offset", type = int, default = 0, help = "index to start in each source file")
    parser.add_argument("--part_num", type = int, default = 50000, help = "number of events to use in each source file")
    opt = parser.parse_args()

    json_partly_merge(
        json_srcroot = opt.json_srcroot,
        json_destfile = opt.json_destfile,
        part_offset = opt.part_offset,
        part_num = opt.part_num,
    )

