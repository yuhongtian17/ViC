import os
import torch
import torch.nn.functional as F
import argparse


def parse_option():
    parser = argparse.ArgumentParser('Interpolate pretrained checkpoint for downstream tasks.', add_help=False)
    parser.add_argument('--base_pth', type=str, required=True, help='', )
    parser.add_argument('--pt_pth', type=str, required=True, help='path to pretrained pth', )
    parser.add_argument('--pt_size', type=int, default=224, )
    parser.add_argument('--tg_pth', type=str, required=True, help='path to save target pth', )
    parser.add_argument('--tg_size', type=int, default=512, )
    args, unparsed = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_option()

    base_pth = torch.load(args.base_pth, map_location='cpu', weights_only=False)
    pt_pth = torch.load(args.pt_pth, map_location='cpu', weights_only=False)

    for i in range(4):
        print("Layer {}:".format(i))
        interpolate_size = (args.tg_size - args.pt_size) // (2**(i + 2))
        tmp = pt_pth['model']['freq_embed.{}'.format(i)]
        print("Shape before interpolation:", tmp.shape)
        tmp = F.pad(tmp.permute(2, 0, 1), (0, interpolate_size, 0, interpolate_size), 'constant', 0)
        pt_pth['model']['freq_embed.{}'.format(i)] = tmp.permute(1, 2, 0)
        print("Shape after interpolation:", tmp.permute(1, 2, 0).shape)

    destfile = base_pth['state_dict']

    weights_keys = list(pt_pth['model'].keys())
    for weight_name in weights_keys:
        new_weight_name = 'vic.' + weight_name
        destfile[new_weight_name] = pt_pth['model'][weight_name]
        print('{} -> {}'.format(weight_name, new_weight_name))

    base_pth.pop('state_dict')
    base_pth['state_dict'] = destfile

    torch.save(base_pth, args.tg_pth)
    print("Finished! Saved to {}.".format(args.tg_pth))


# ref: https://github.com/MzeroMiko/vHeat/blob/main/classification/interpolate4downstream.py
# python ./tools/model_converters/all_pretrained.py --base_pth "./data/pretrained/selfsup_50x.pth" --pt_pth "./data/pretrained/vHeat_tiny.pth" --tg_pth "./data/pretrained/all_pretrained.pth"

