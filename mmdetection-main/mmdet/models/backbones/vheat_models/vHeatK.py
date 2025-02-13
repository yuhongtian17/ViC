import torch
import torch.nn as nn
import torch.nn.functional as F

from .vHeat import LayerNorm2d, Heat2D, HeatBlock, AdditionalInputSequential, vHeat


class HeatK2D(Heat2D):
    def __init__(
        self, 
        infer_mode=False, 
        res=14, 
        dim=96, 
        hidden_dim=96, 
        feat_fusion_mode=None, 
        **kwargs, 
    ):
        super().__init__(
            infer_mode=infer_mode, 
            res=res, 
            dim=dim, 
            hidden_dim=hidden_dim, 
            **kwargs, 
        )
        self.infer_mode = False  # Not implemented yet for self.infer_mode = True
        self.feat_fusion_mode = feat_fusion_mode
        if self.feat_fusion_mode == 'cat':
            self.feat_to_k = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)

        if self.feat_fusion_mode == 'cat' or self.feat_fusion_mode == 'add':
            feat = x.permute(0, 2, 3, 1) # B, H, W, C

        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1) # B, H, W, C

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        # if self.infer_mode:
        #     x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        # else:
        #     weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
        #     x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp) # exp decay

        # freq_embed shape: H, W, C
        # feat       shape: B, H, W, C
        # weight_exp shape: H, W
        if self.feat_fusion_mode == 'cat':
            freq_embed = self.to_k[0](freq_embed) + self.feat_to_k(feat)
            weight_exp = torch.pow(weight_exp[None, :, :, None], self.to_k[1](freq_embed))
            x = x * weight_exp
        elif self.feat_fusion_mode == 'add':
            freq_embed = freq_embed + feat
            weight_exp = torch.pow(weight_exp[None, :, :, None], self.to_k(freq_embed))
            x = x * weight_exp
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp) # exp decay

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)

        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class HeatKBlock(HeatBlock):
    def __init__(
        self, 
        infer_mode=False, 
        res=14, 
        dim=96, 
        hidden_dim=96, 
        feat_fusion_mode=None, 
        **kwargs, 
    ):
        super().__init__(
            infer_mode=infer_mode, 
            res=res, 
            dim=dim, 
            hidden_dim=hidden_dim, 
            **kwargs, 
        )
        self.op = HeatK2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode, 
                          feat_fusion_mode=feat_fusion_mode)


class vHeatK(vHeat):
    def __init__(
        self, 
        feat_fusion_mode=None, 
        **kwargs, 
    ):
        self.feat_fusion_mode = feat_fusion_mode
        super().__init__(**kwargs)

    # @staticmethod
    def make_layer(
        self,
        res=14,
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=True, # False,
        norm_layer=LayerNorm2d,
        post_norm=False, # True,
        layer_scale=None,
        downsample=nn.Identity(),
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatKBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
                feat_fusion_mode=self.feat_fusion_mode,
                **kwargs,
            ))

        return AdditionalInputSequential(
            *blocks, 
            downsample,
        )

