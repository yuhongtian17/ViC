# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader

from mmdet.registry import MODELS

from typing import Optional
import torch.utils.checkpoint as cp


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        B, N, C = x.shape
        # B, N, N_ = mask.shape

        # qkv with shape (3, B, nHead, N, C // nHead)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, N, C // nHead)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn += mask[:, None, :, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, N,
                            -1).permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type='GELU'),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        with_cp=False,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, mask):

        def _inner_forward(x, mask):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, mask)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.mlp(x)

            return x + identity

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, mask)
        else:
            x = _inner_forward(x, mask)

        return x


class TokenEmbed_engtime(nn.Module):

    def __init__(self,
                 num_embeds_eng=768,
                 num_embeds_time=1,
                 embed_dim=768,
                 eng_min=5e-4,
                 eng_max=2.0,
                 eps=1e-6):
        super().__init__()

        self.num_embeds_eng = num_embeds_eng
        self.num_embeds_time = num_embeds_time
        self.eng_min = eng_min
        self.eng_max = eng_max
        self.eps = eps

        self.embed_layer = nn.Embedding(num_embeds_eng * num_embeds_time, embed_dim)

    def forward(self, x_eng: torch.Tensor, x_time: torch.Tensor):
        log10_ind = torch.clamp(
            torch.log10(x_eng / self.eng_min) / math.log10(self.eng_max / self.eng_min),
            min=0, max=1-self.eps,
        ) * self.num_embeds_eng

        log10_ind += torch.clamp(
            x_time,
            min=0, max=self.num_embeds_time-self.eps,
        ) * self.num_embeds_eng

        log10_ind = log10_ind.to(dtype=torch.long)
        return self.embed_layer(log10_ind)


class TokenEmbed_phithe(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()

        self.width = 960
        self.height = 480
        self.num_embeds_grid = 7008

        self.embed_layer = nn.Embedding(self.num_embeds_grid, embed_dim)

    def forward(self, x_phi: torch.Tensor, x_the: torch.Tensor):
        B, N = x_phi.shape
        x_phi = x_phi.flatten()
        x_the = x_the.flatten().unsqueeze(-1)
        local_device = x_phi.device

        w_px = torch.tensor([
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
        ], device=local_device)
        h_px = torch.tensor([
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
        ], device=local_device)
        hh_px_2D = torch.tensor([[
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
        ]], device=local_device)

        sum_grids = torch.tensor([
            0,
            32, 64, 104, 144, 192, 240,
            288,
            352, 416, 496, 576, 672, 768,
            864,
            984, 1104, 1224, 1344, 1464,
            1584, 1704, 1824, 1944,
            2064, 2184, 2304, 2424, 2544,
            2664, 2784, 2904, 3024, 3144, 3264, 3384, 3504,
            3624, 3744, 3864, 3984, 4104, 4224, 4344, 4464,
            4584, 4704, 4824, 4944, 5064,
            5184, 5304, 5424, 5544,
            5664, 5784, 5904, 6024, 6144,
            6240,
            6336, 6432, 6512, 6592, 6656, 6720,
            6768,
            6816, 6864, 6904, 6944, 6976, 7008, 
        ], device=local_device)

        half_width = self.width * 0.5
        x_ctr_1D = x_phi / torch.pi * half_width + half_width
        y_ctr_2D = x_the / torch.pi * self.height
        ind = torch.sum((y_ctr_2D - hh_px_2D) >= 0, dim=1)
        grid_ind = (x_ctr_1D / w_px[ind] + sum_grids[ind]).reshape(B, N).to(dtype=torch.long)

        return self.embed_layer(grid_ind)


class TokenEmbed_eng(nn.Module):

    def __init__(self,
                 num_embeds=768,
                 embed_dim=768,
                 eng_min=5e-4,
                 eng_max=2.0,
                 eps=1e-6):
        super().__init__()

        self.num_embeds = num_embeds
        self.eng_min = eng_min
        self.eng_max = eng_max
        self.eps = eps

        self.embed_layer = nn.Embedding(num_embeds, embed_dim)

    def forward(self, x: torch.Tensor):
        log10_ind = torch.clamp(
            torch.log10(x / self.eng_min) / math.log10(self.eng_max / self.eng_min),
            min=0, max=1-self.eps,
        ) * self.num_embeds

        log10_ind = log10_ind.to(dtype=torch.long)
        return self.embed_layer(log10_ind)


class TokenEmbed_rad(nn.Module):

    def __init__(self,
                 num_embeds=360,
                 embed_dim=384,
                 rad_min=-torch.pi,
                 rad_max=torch.pi,
                 eps=1e-6):
        super().__init__()

        self.num_embeds = num_embeds
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.eps = eps

        self.embed_layer = nn.Embedding(num_embeds, embed_dim)

    def forward(self, x: torch.Tensor):
        ind = (x - self.rad_min) / (self.rad_max - self.rad_min) % 1.0 * self.num_embeds
        ind = ind.to(dtype=torch.long)
        return self.embed_layer(ind)


class TokenEmbed_time(nn.Module):

    def __init__(self, num_embeds=21, embed_dim=21):
        super().__init__()

        self.embed_layer = nn.Embedding(num_embeds, embed_dim)

    def forward(self, x: torch.Tensor):
        ind = x.to(dtype=torch.long)
        return self.embed_layer(ind)


@MODELS.register_module()
class HEPv2Transformer(BaseModule):
    """HEPv2 Transformer is from
    Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self,
                 num_embeds_engphithe=[768, 360, 180],
                 embed_dim_engphithe=[768, 384, 384],
                 # 
                 use_mmt_token=True,
                 out_eng=False,
                 out_phithe=False,
                 # 
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 with_cp=False,
                 # 
                 out_indices=None,
                 init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

        self.token_embed_eng = TokenEmbed_eng(num_embeds_engphithe[0], embed_dim_engphithe[0])
        self.token_embed_phi = TokenEmbed_rad(num_embeds_engphithe[1], embed_dim_engphithe[1], -torch.pi, torch.pi)
        self.token_embed_the = TokenEmbed_rad(num_embeds_engphithe[2], embed_dim_engphithe[2], 0, torch.pi)

        self.use_mmt_token = use_mmt_token
        if self.use_mmt_token:
            self.mmt_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.out_eng = out_eng
        self.out_phithe = out_phithe

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
            ) for i in range(depth)
        ])

        self.out_indices = out_indices if out_indices is not None else [depth - 1]
        for i in self.out_indices:
            layer = build_norm_layer(norm_cfg, embed_dim)[1]
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        self.eps = 1e-6

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            else:
                _state_dict = ckpt
            incompatibleKeys = self.load_state_dict(_state_dict, False)
            print(incompatibleKeys)

    def forward(self, x):
        outs_others = dict()

        # 如果没有mmt_token，则需要更多信息辅助mmt回归
        if not self.use_mmt_token:
            outs_others['x'] = x.clone()
        elif self.out_eng or self.out_phithe:
            outs_others['x'] = x.clone()
        else:
            pass

        B, N, C = x.shape
        assert C == 5                                                           # flags, eng, phi, the, time
        flags = x[..., 0:1]
        x_eng = x[..., 1]
        x_phi = x[..., 2]
        x_the = x[..., 3]
        # x_time = x[..., 4]
        flags_0 = x[..., 0]

        y_eng = self.token_embed_eng(x_eng)
        y_phi = self.token_embed_phi(x_phi)
        y_the = self.token_embed_the(x_the)
        y_phithe = torch.cat([y_phi, y_the], dim=2)

        # 如果没有mmt_token，则需要更多信息辅助mmt回归
        if self.out_eng:
            outs_others['y_eng']    = y_eng.clone()
        if self.out_phithe:
            outs_others['y_phithe'] = y_phithe.clone()

        x = y_eng + y_phithe

        flags = (flags + self.eps).to(dtype=torch.long)                         # 浮点数改整数以防出错
        attn_mask_main = (flags != flags.transpose(-2, -1)) * (-1e9)            # B, N, N
        attn_mask_mmt = (flags_0 < self.eps) * (-1e9)                           # B, N
        if self.use_mmt_token:
            x = torch.cat([x, self.mmt_token.repeat(B, 1, 1)], dim=1)           # B, N + 1, C
            attn_mask = torch.zeros(B, N + 1, N + 1, device=x.device, requires_grad=False)
            attn_mask[:, :-1, :-1] = attn_mask_main
            attn_mask[:,  -1, :-1] = attn_mask_mmt
            attn_mask[:, :-1,  -1] = attn_mask_mmt
        else:
            attn_mask = attn_mask_main

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x)
                outs.append(out)

        if len(outs_others) > 0:
            return outs, outs_others
        else:
            return outs

