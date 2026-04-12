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

from typing import Union
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


class TokenEmbed(nn.Module):

    def __init__(self,
                 num_embeds: int = 768,
                 embed_dim: int = 768,
                 num_min: float = 5e-4,
                 num_max: float = 2.0,
                 log_base: float = 10.0,
                 eps: float = 1e-6):
        super().__init__()

        self.num_embeds = num_embeds
        self.num_min = num_min
        self.num_max = num_max
        self.log_base = log_base
        self.eps = eps

        self.embed_layer = nn.Embedding(num_embeds, embed_dim)

    def logba(self, base: float, a: Union[float, torch.Tensor]):
        if isinstance(a, (int, float)):
            a = max(a, self.eps)
            return math.log10(a) / math.log10(base)
        else:
            a = torch.clamp(a, min=self.eps)
            return torch.log10(a) / math.log10(base)

    def forward(self, x: torch.Tensor):
        if self.log_base > 0:
            num_min = self.logba(self.log_base, self.num_min)
            num_max = self.logba(self.log_base, self.num_max)
            x = self.logba(self.log_base, x)
        else:
            num_min = self.num_min
            num_max = self.num_max

        ind = torch.clamp(
            (x - num_min) / (num_max - num_min),
            min=0, max=1-self.eps,
        ) * self.num_embeds

        ind = ind.to(dtype=torch.long)
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

        self.token_embed_eng = TokenEmbed(num_embeds_engphithe[0], embed_dim_engphithe[0])
        self.token_embed_phi = TokenEmbed(num_embeds_engphithe[1], embed_dim_engphithe[1], -torch.pi, torch.pi, log_base=0)
        self.token_embed_the = TokenEmbed(num_embeds_engphithe[2], embed_dim_engphithe[2], 0, torch.pi, log_base=0)

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
        outs_others['x'] = x.clone()

        B, N, C = x.shape
        flags = x[..., 0:1]
        x_eng = x[..., 1]
        x_phi = x[..., 2]
        x_the = x[..., 3]
        # x_time = x[..., 4]
        # x_grid = x[..., 5:7]
        flags_0 = x[..., 0]

        y_eng = self.token_embed_eng(x_eng)
        y_phi = self.token_embed_phi(x_phi)
        y_the = self.token_embed_the(x_the)
        y_phithe = torch.cat([y_phi, y_the], dim=2)

        # 预训练需要更多信息辅助
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

        return outs, outs_others

