# 'mmt_token' -> 'backbone.mmt_token'
# 'token_embed...' -> 'backbone.token_embed...'
# 'blocks...' -> 'backbone.blocks...'
# 'outnorm11...' -> 'backbone.outnorm11...'
# 'mask_token' -> 'head.mask_token'
# 'decoder...' -> 'head.decoder...'
# 'classifier...' -> 'head.classifier....'


import argparse
import torch


OLD_KEYWORDS = [
    'mmt_token',
    'token_embed',
    'blocks',
    'outnorm11',
    'mask_token',
    'decoder',
    'classifier',
]
NEW_KEYWORDS = [
    'backbone.mmt_token',
    'backbone.token_embed',
    'backbone.blocks',
    'backbone.outnorm11',
    'head.mask_token',
    'head.decoder',
    'head.classifier',
]


def model_to_state_dict(
    srcpath: str,
    destpath: str,
    old_dict: str,
    new_dict: str,
):
    assert len(OLD_KEYWORDS) == len(NEW_KEYWORDS)
    n = len(OLD_KEYWORDS)

    srcfile = torch.load(srcpath, map_location='cpu', weights_only=False)
    print("Open file \"{}\" successfully!".format(srcpath))
    destfile = {}

    weights_keys = list(srcfile[old_dict].keys())
    for weight_name in weights_keys:
        is_new = False

        for i in range(n):
            old_keyword = OLD_KEYWORDS[i]
            new_keyword = NEW_KEYWORDS[i]
            len_old_keyword = len(old_keyword)

            if weight_name[:len_old_keyword] == old_keyword:
                new_weight_name = new_keyword + weight_name[len_old_keyword:]
                destfile[new_weight_name] = srcfile[old_dict][weight_name]
                print('{} -> {}'.format(weight_name, new_weight_name))

                is_new = True
                break

        if not is_new:
            destfile[weight_name] = srcfile[old_dict][weight_name]
            print('KEEP {}'.format(weight_name))

    srcfile.pop(old_dict)
    srcfile[new_dict] = destfile
    print()
    print('{} -> {}'.format(old_dict, new_dict))

    torch.save(srcfile, destpath)
    print("Write to \"{}\" successfully!".format(destpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcpath", type = str, default = "./work_dirs/selfsup_50x.pth", help = "source pth file")
    parser.add_argument("--destpath", type = str, default = "./work_dirs/epoch_12_50x.pth", help = "destination pth file")
    parser.add_argument("--old_dict", type = str, default = "state_dict", help = "old state_dict")
    parser.add_argument("--new_dict", type = str, default = "state_dict", help = "new state_dict")
    opt = parser.parse_args()

    model_to_state_dict(
        srcpath = opt.srcpath,
        destpath = opt.destpath,
        old_dict = opt.old_dict,
        new_dict = opt.new_dict,
    )
