import torch
import torch.nn as nn

import math
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.models.backbones.hepv2_transformer import Block


@MODELS.register_module()
class HEPv2SelfSupervisorREGHead(BaseModule):

    def __init__(
        self,
        in_channels=768,
        len_seq=640,
        use_mmt_token=True,
        backbone_out_eng=True,
        backbone_out_phithe=False,
        recover_eng=False,
        recover_phithe=True,
        # 
        embed_dim=384, # 512,
        depth=8,
        num_heads=12, # 16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        with_cp=False,
        # 
        num_classes_eng=12,
        num_classes_phi=24,
        num_classes_the=12,
        loss_eng=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_phi=dict(
            type='L1Loss',
            loss_weight=1.0),
        loss_the=dict(
            type='L1Loss',
            loss_weight=1.0),
        init_cfg=None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.len_seq = len_seq
        self.use_mmt_token = use_mmt_token
        self.backbone_out_eng = backbone_out_eng
        self.backbone_out_phithe = backbone_out_phithe
        self.recover_eng = recover_eng
        self.recover_phithe = recover_phithe

        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.backbone_out_eng:
            self.decoder_embed_eng = nn.Linear(in_channels, embed_dim, bias=True)
            if self.recover_eng:
                self.mask_token_eng = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.backbone_out_phithe:
            self.decoder_embed_phithe = nn.Linear(in_channels, embed_dim, bias=True)
            if self.recover_phithe:
                self.mask_token_phithe = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_classes_eng = num_classes_eng
        self.num_classes_phi = num_classes_phi
        self.num_classes_the = num_classes_the

        self.phithe_base = 0.25 * torch.pi
        self.phithe_mean = 0.0
        self.phithe_std = 1.0
        self.mmt_base = 1.0
        self.mmt_mean = 0.0
        self.mmt_std = 1.0

        if self.recover_eng:
            self.classifier_eng = nn.Linear(embed_dim, 1, bias=True)
            self.loss_eng = MODELS.build(loss_eng)

        if self.recover_phithe:
            self.classifier_phi = nn.Linear(embed_dim, 1, bias=True)
            self.classifier_the = nn.Linear(embed_dim, 1, bias=True)
            self.loss_phi = MODELS.build(loss_phi)
            self.loss_the = MODELS.build(loss_the)

        dpr = [drop_path_rate] * depth  # [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.decoder_blocks = nn.ModuleList([
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

        self.decoder_norm = build_norm_layer(norm_cfg, embed_dim)[1]

        self.fp16_enabled = False
        self.apply(self._init_weights)
        self.eps = 1e-6

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        # 来自HEPv2Transformer: backbone_outs是List[Tensor], 每个Tensor形状为[B, N, C]
        backbone_outs, backbone_outs_others = inputs

        feat = backbone_outs[-1]
        # assert feat.requires_grad
        feat = self.decoder_embed(feat)
        if self.use_mmt_token:
            feat_main = feat[:, :-1, :]
            feat_mmt = feat[:, -1:, :]
        else:
            feat_main = feat

        backbone_x = backbone_outs_others['x']
        # assert not backbone_x.requires_grad

        flags = backbone_x[..., 0:1]
        x_eng = backbone_x[..., 1]
        x_phi = backbone_x[..., 2]
        x_the = backbone_x[..., 3]
        # x_time = backbone_x[..., 4]
        flags_0 = backbone_x[..., 0]

        B, N, C = feat_main.shape
        mask_token = self.mask_token.repeat(B, N, 1)
        hit_mask = (flags_0 < -self.eps)
        feat_main[hit_mask] = mask_token[hit_mask]

        if self.backbone_out_eng:
            y_eng = backbone_outs_others['y_eng']
            # assert y_eng.requires_grad
            y_eng = self.decoder_embed_eng(y_eng)
        else:
            y_eng = 0.0

        if self.backbone_out_phithe:
            y_phithe = backbone_outs_others['y_phithe']
            # assert y_phithe.requires_grad
            y_phithe = self.decoder_embed_phithe(y_phithe)
        else:
            y_phithe = 0.0

        if self.recover_eng:
            hit_mask_eng = ((flags_0 < -self.eps) & (flags_0 > -self.len_seq))
        else:
            hit_mask_eng = None

        if self.recover_phithe:
            hit_mask_phithe = ((flags_0 < -self.len_seq) & (flags_0 > -self.len_seq * 2))
        else:
            hit_mask_phithe = None

        if self.backbone_out_eng and self.recover_eng:
            mask_token_eng = self.mask_token_eng.repeat(B, N, 1)
            y_eng[hit_mask_eng] = mask_token_eng[hit_mask_eng]

        if self.backbone_out_phithe and self.recover_phithe:
            mask_token_phithe = self.mask_token_phithe.repeat(B, N, 1)
            y_phithe[hit_mask_phithe] = mask_token_phithe[hit_mask_phithe]

        x = feat_main + y_eng + y_phithe

        flags = (flags % self.len_seq + self.eps).to(dtype=torch.long)          # 浮点数改整数以防出错
        attn_mask_main = (flags != flags.transpose(-2, -1)) * (-1e9)            # B, N, N
        attn_mask_mmt = (torch.abs(flags_0) < self.eps) * (-1e9)                # B, N
        if self.use_mmt_token:
            x = torch.cat([x, feat_mmt], dim=1)                                 # B, N + 1, C
            attn_mask = torch.zeros(B, N + 1, N + 1, device=x.device, requires_grad=False)
            attn_mask[:, :-1, :-1] = attn_mask_main
            attn_mask[:,  -1, :-1] = attn_mask_mmt
            attn_mask[:, :-1,  -1] = attn_mask_mmt
        else:
            attn_mask = attn_mask_main

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, attn_mask)

        feat = self.decoder_norm(x)
        if self.use_mmt_token: feat = feat[:, :-1, :]                           # 去掉mmt_token

        if self.recover_eng:
            pred_eng = self.classifier_eng(feat)
            target_eng = self.mmt_encode_base(x_eng + self.eps ** 2)            # 防止log计算出现nan
        else:
            pred_eng = None
            target_eng = None

        if self.recover_phithe:
            pred_phi = self.classifier_phi(feat)
            pred_the = self.classifier_the(feat)
            target_phi = self.phithe_encode_base(x_phi, 0.0)
            target_the = self.phithe_encode_base(x_the, 0.5 * torch.pi)
        else:
            pred_phi = None
            pred_the = None
            target_phi = None
            target_the = None

        preds = [pred_eng, pred_phi, pred_the]
        targets = [target_eng, target_phi, target_the]
        masks = [hit_mask_eng, hit_mask_phithe, hit_mask_phithe]

        return preds, targets, masks


    def mmt_encode_base(self, mmt_gts) -> torch.Tensor:
        return (torch.log(mmt_gts / self.mmt_base) - self.mmt_mean) / self.mmt_std

    def mmt_decode_base(self, mmt_preds) -> torch.Tensor:
        return torch.exp(mmt_preds * self.mmt_std + self.mmt_mean) * self.mmt_base


    def phithe_encode_base(self, phithe_gts, phithe_anchors) -> torch.Tensor:
        return ((phithe_gts - phithe_anchors) / self.phithe_base - self.phithe_mean) / self.phithe_std

    def phithe_decode_base(self, phithe_preds, phithe_anchors) -> torch.Tensor:
        return phithe_anchors + self.phithe_base * (phithe_preds * self.phithe_std + self.phithe_mean)


    def get_angle(self, phi_1, the_1, phi_2, the_2):
        # 球坐标系中夹角计算。输入：phi_1, the_1, phi_2, the_2 in rad。输出：angle_deg in deg。
        vector1 = (torch.cos(phi_1) * torch.cos(the_1), torch.sin(phi_1) * torch.cos(the_1), torch.sin(the_1))
        vector2 = (torch.cos(phi_2) * torch.cos(the_2), torch.sin(phi_2) * torch.cos(the_2), torch.sin(the_2))
        product = torch.clamp(vector1[0] * vector2[0] + 
                              vector1[1] * vector2[1] + 
                              vector1[2] * vector2[2], min=-1.0, max=1.0)
        angle_deg = torch.arccos(product) / torch.pi * 180.0
        # assert angle_deg >= 0.0 and angle_deg <= 180.0
        return angle_deg


    def eng_to_ind(self, x: torch.Tensor,
                   num_embeds=768,
                   eng_min=5e-4,
                   eng_max=2.0):
        log10_ind = torch.clamp(
            torch.log10(x / eng_min) / math.log10(eng_max / eng_min),
            min=0, max=1-self.eps,
        ) * num_embeds

        log10_ind = log10_ind.to(dtype=torch.long)
        return log10_ind

    def rad_to_ind(self, x: torch.Tensor,
                   num_embeds=360,
                   rad_min=-torch.pi,
                   rad_max=torch.pi):
        ind = (x - rad_min) / (rad_max - rad_min) % 1.0 * num_embeds
        ind = ind.to(dtype=torch.long)
        return ind


    def loss_single(self,
                    loss_name: str,
                    pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor):
        pred = pred.squeeze(-1)
        assert pred.shape == target.shape == mask.shape

        if loss_name == 'eng':
            decoded_pred   = self.mmt_decode_base(pred)
            decoded_target = self.mmt_decode_base(target)
            decoded_pred_ind   = self.eng_to_ind(decoded_pred,   self.num_classes_eng)
            decoded_target_ind = self.eng_to_ind(decoded_target, self.num_classes_eng)
            loss_model = self.loss_eng
        elif loss_name == 'phi':
            decoded_pred   = self.phithe_decode_base(pred, 0.0)
            decoded_target = self.phithe_decode_base(target, 0.0)
            decoded_pred_ind   = self.rad_to_ind(decoded_pred,   self.num_classes_phi, -torch.pi, torch.pi)
            decoded_target_ind = self.rad_to_ind(decoded_target, self.num_classes_phi, -torch.pi, torch.pi)
            loss_model = self.loss_phi
        elif loss_name == 'the':
            decoded_pred   = self.phithe_decode_base(pred, 0.5 * torch.pi)
            decoded_target = self.phithe_decode_base(target, 0.5 * torch.pi)
            decoded_pred_ind   = self.rad_to_ind(decoded_pred,   self.num_classes_the, 0, torch.pi)
            decoded_target_ind = self.rad_to_ind(decoded_target, self.num_classes_the, 0, torch.pi)
            loss_model = self.loss_the
        else:
            raise NotImplementedError

        correct_mask = (decoded_pred_ind[mask] == decoded_target_ind[mask])
        correct_count = int(torch.sum(correct_mask))
        all_count = int(torch.sum(mask))
        accuracy = torch.tensor(correct_count / all_count)

        pred = pred.reshape(-1)
        target = target.reshape(-1)
        mask = mask.reshape(-1)
        losses = loss_model(pred, target, mask, avg_factor=all_count)

        return losses, accuracy, decoded_pred, decoded_target

    def loss(self, pred, target, mask):
        if self.recover_eng:
            mask_eng = mask[0]
            loss_eng, acc_eng, decoded_eng_pred, decoded_eng_target = self.loss_single('eng', pred[0], target[0], mask_eng)
            re = torch.abs(decoded_eng_pred - decoded_eng_target) / decoded_eng_target
            mre_eng = torch.sum(re[mask_eng]) / torch.sum(mask_eng)
            count_eng = torch.sum(mask_eng)
        else:
            loss_eng = torch.zeros(1, requires_grad=False)
            acc_eng = -torch.ones(1, requires_grad=False)
            mre_eng = -torch.ones(1, requires_grad=False)
            count_eng = torch.zeros(1, requires_grad=False)

        if self.recover_phithe:
            mask_phi = mask[1]
            mask_the = mask[2]
            loss_phi, acc_phi, decoded_phi_pred, decoded_phi_target = self.loss_single('phi', pred[1], target[1], mask_phi)
            loss_the, acc_the, decoded_the_pred, decoded_the_target = self.loss_single('the', pred[2], target[2], mask_the)
            ab = self.get_angle(decoded_phi_pred,
                                decoded_the_pred - 0.5 * torch.pi,
                                decoded_phi_target,
                                decoded_the_target - 0.5 * torch.pi)
            mab_phithe = torch.sum(ab[mask_phi]) / torch.sum(mask_phi)
            count_phithe = torch.sum(mask_the)
        else:
            loss_phi = torch.zeros(1, requires_grad=False)
            loss_the = torch.zeros(1, requires_grad=False)
            acc_phi = -torch.ones(1, requires_grad=False)
            acc_the = -torch.ones(1, requires_grad=False)
            mab_phithe = -torch.ones(1, requires_grad=False)
            count_phithe = torch.zeros(1, requires_grad=False)

        return dict(
            loss_eng=loss_eng,
            loss_phi=loss_phi,
            loss_the=loss_the,
            # 
            acc_eng=acc_eng,
            acc_phi=acc_phi,
            acc_the=acc_the,
            # 
            mre_eng=mre_eng,
            count_eng=count_eng,
            mab_phithe=mab_phithe,
            count_phithe=count_phithe,
        )

