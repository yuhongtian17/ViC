# python ./tools/model_converters/state_dict_to_model.py --srcpath "./work_dirs/epoch_12.pth"       --destpath "./work_dirs/epoch_12_temp1.pth" --old_keyword "backbone." --new_keyword ""
# python ./tools/model_converters/state_dict_to_model.py --srcpath "./work_dirs/epoch_12_temp1.pth" --destpath "./work_dirs/epoch_12_temp2.pth" --old_keyword "head.decoder_embed" --new_keyword "decoder_embed"
# python ./tools/model_converters/state_dict_to_model.py --srcpath "./work_dirs/epoch_12_temp2.pth" --destpath "./work_dirs/epoch_12_temp3.pth" --old_keyword "head.blocks" --new_keyword "decoder_blocks"
# python ./tools/model_converters/state_dict_to_model.py --srcpath "./work_dirs/epoch_12_temp3.pth" --destpath "./work_dirs/selfsup_50x.pth"    --old_keyword "head.outnorm7" --new_keyword "decoder_norm"


import argparse
import torch


def state_dict_to_model(
    srcpath: str,
    destpath: str,
    old_dict: str,
    old_keyword: str,
    new_dict: str,
    new_keyword: str,
):
    len_old_keyword = len(old_keyword)

    srcfile = torch.load(srcpath, map_location='cpu')
    print("Open file \"{}\" successfully!".format(srcpath))
    destfile = {}

    weights_keys = list(srcfile[old_dict].keys())
    for weight_name in weights_keys:
        if weight_name[:len_old_keyword] == old_keyword:
            new_weight_name = new_keyword + weight_name[len_old_keyword:]
            destfile[new_weight_name] = srcfile[old_dict][weight_name]
            print('{} -> {}'.format(weight_name, new_weight_name))
        else:
            destfile[weight_name] = srcfile[old_dict][weight_name]

    srcfile.pop(old_dict)
    srcfile[new_dict] = destfile
    print('{} -> {}'.format(old_dict, new_dict))

    torch.save(srcfile, destpath)
    print("Write to \"{}\" successfully!".format(destpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcpath", type = str, default = "./work_dirs/epoch_12.pth", help = "source pth file")
    parser.add_argument("--destpath", type = str, default = "./work_dirs/selfsup_best50x.pth", help = "destination pth file")
    parser.add_argument("--old_dict", type = str, default = "state_dict", help = "old state_dict")
    parser.add_argument("--old_keyword", type = str, default = "backbone.", help = "old keyword")
    parser.add_argument("--new_dict", type = str, default = "state_dict", help = "new state_dict")
    parser.add_argument("--new_keyword", type = str, default = "", help = "new keyword")
    opt = parser.parse_args()

    state_dict_to_model(
        srcpath = opt.srcpath,
        destpath = opt.destpath,
        old_dict = opt.old_dict,
        old_keyword = opt.old_keyword,
        new_dict = opt.new_dict,
        new_keyword = opt.new_keyword,
    )
