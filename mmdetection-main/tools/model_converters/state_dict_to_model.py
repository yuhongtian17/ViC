# python ./tools/model_converters/state_dict_to_model.py --destpath "/workspace/all-data/work_dirs/epoch_12_1035_2300_embed.pth" --old_keyword "backbone.token_embedding" --new_keyword "backbone.token_embed" --rename "state_dict"
# ./tools/dist_test.sh ./configs/_hep2seq_/abla_pretrain/hepv2-ssd_trans-base-mae_nofpn_1x_hep2seq_8xbs128.py ./work_dirs/epoch_12_1035_2300_embed.pth 4 --out ./work_dirs/results_hepv2.pkl
# python ./tools/analysis_tools/hep_eval.py --pkl_path ./work_dirs/results_hepv2.pkl --json_path ./data/HEP2COCO/Nm_1m/Nm_1m__b00000001__e00100000.json


import argparse
import torch


def state_dict_to_model(
    srcpath: str,
    destpath: str,
    old_keyword: str,
    new_keyword: str,
    rename: str,
):
    len_old_keyword = len(old_keyword)

    srcfile = torch.load(srcpath, map_location='cpu')
    print("Open file \"{}\" successfully!".format(srcpath))
    destfile = {}

    weights_keys = list(srcfile['state_dict'].keys())
    for weight_name in weights_keys:
        if weight_name[:len_old_keyword] == old_keyword:
            new_weight_name = new_keyword + weight_name[len_old_keyword:]
            destfile[new_weight_name] = srcfile['state_dict'][weight_name]
            print('{} -> {}'.format(weight_name, new_weight_name))
        else:
            destfile[weight_name] = srcfile['state_dict'][weight_name]

    srcfile.pop('state_dict')
    srcfile[rename] = destfile
    print('state_dict -> {}'.format(rename))

    torch.save(srcfile, destpath)
    print("Write to \"{}\" successfully!".format(destpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcpath", type = str, default = "/workspace/all-data/work_dirs/+hepv2_vit-base_eng768c-211add_nofpn_2fc768c45t45b_1x_hep2seq-rp-cp_bs128x8-lr1e-3/epoch_12.pth", help = "source pth file")
    parser.add_argument("--destpath", type = str, default = "/workspace/all-data/work_dirs/trained_ep12.pth", help = "destination pth file")
    parser.add_argument("--old_keyword", type = str, default = "backbone.", help = "old keyword")
    parser.add_argument("--new_keyword", type = str, default = "", help = "new keyword")
    parser.add_argument("--rename", type = str, default = "model", help = "rename state_dict")
    opt = parser.parse_args()

    state_dict_to_model(
        srcpath = opt.srcpath,
        destpath = opt.destpath,
        old_keyword = opt.old_keyword,
        new_keyword = opt.new_keyword,
        rename = opt.rename,
    )
