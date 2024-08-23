import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from functools import partial
from fairscale.nn.checkpoint import checkpoint_wrapper

from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.attention import MultiScaleBlock, MultiScaleDecoderBlock
from slowfast.models.av_attention import TemporalBlock, SpatialBlock
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import
from . import stem_helper
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CSTS(nn.Module):
    """
    Multiscale Vision Transformers with Audio-Visual Fusion
    """

    def __init__(self, cfg):
        super(CSTS, self).__init__()
        # ============================= Get parameters =============================
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  # False

        # =============================== Prepare input ===============================
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES  # 8
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D  # default false
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        self.patch_stride_audio = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
            self.patch_stride_audio = [1] + self.patch_stride_audio

        # =============================== Prepare output ===============================
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM  # 96
        embed_dim_audio = cfg.MVIT.EMBED_DIM  # 96

        # =============================== Prepare backbone ===============================
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS  # True
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # 0
        depth = cfg.MVIT.DEPTH
        depth_audio = 4
        drop_path_rate = cfg.MVIT.DROPPATH_RATE  # 0.2
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.spatial_audio_attn = cfg.MVIT.SPATIAL_AUDIO_ATTN
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        # =============================== Input embedding ===============================
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.patch_embed_audio = stem_helper.PatchEmbed(
            dim_in=1,  # audio input channel
            dim_out=embed_dim_audio,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        self.input_dims_audio = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [self.input_dims[i] // self.patch_stride[i] for i in range(len(self.input_dims))]
        self.patch_dims_audio = [self.input_dims_audio[i] // self.patch_stride_audio[i] for i in range(len(self.input_dims_audio))]
        num_patches = math.prod(self.patch_dims)
        num_patches_audio = math.prod(self.patch_dims_audio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Positional embedding
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # size (1, 1, 96)
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches
            pos_embed_dim_audio = num_patches_audio

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_dims[0], embed_dim))
            self.pos_embed_spatial_audio = nn.Parameter(torch.zeros(1, self.patch_dims_audio[1] * self.patch_dims_audio[2], embed_dim_audio))
            self.pos_embed_temporal_audio = nn.Parameter(torch.zeros(1, self.patch_dims_audio[0], embed_dim_audio))
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))
            self.pos_embed_audio = nn.Parameter(torch.zeros(1, pos_embed_dim_audio, embed_dim_audio))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        # =============================== Visual encoder ===============================
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        self.stage_end = []  # save the number of blocks before downsampling

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            self.stage_end.append(cfg.MVIT.POOL_Q_STRIDE[i][0]-1)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:  # if there's a stride in q
                    _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]]
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None  # None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:  # False
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(embed_dim, dim_mul[i + 1], divisor=round_width(num_heads, head_mul[i + 1]), )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        # =============================== Audio encoder ===============================
        # 4 layers
        embed_dim_audio = [96, 192, 384, 768]
        dim_out_audio = [192, 384, 768, 768]
        num_heads_audio = [1, 2, 4, 8]
        pool_q_audio = [[], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        pool_kv_audio = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        stride_q_audio = [[], [1, 2, 2], [1, 2, 2], [1, 2, 2]]
        stride_kv_audio = [[1, 8, 8], [1, 4, 4], [1, 2, 2], [1, 1, 1]]
        dpr_audio = [x.item() for x in torch.linspace(0, drop_path_rate, depth_audio)]

        assert len(embed_dim_audio) == depth_audio

        self.blocks_audio = nn.ModuleList()
        for i in range(depth_audio):
            attention_block = MultiScaleBlock(
                dim=embed_dim_audio[i],
                dim_out=dim_out_audio[i],
                num_heads=num_heads_audio[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=0.,
                norm_layer=norm_layer,
                kernel_q=pool_q_audio[i] if len(pool_q_audio) > i else [],
                kernel_kv=pool_kv_audio[i] if len(pool_kv_audio) > i else [],
                stride_q=stride_q_audio[i] if len(stride_q_audio) > i else [],
                stride_kv=stride_kv_audio[i] if len(stride_kv_audio) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks_audio.append(attention_block)

        # =============================== Audio-Visual Fusion ===============================
        token_dim = self.blocks[-1].dim_out

        if 'nce' in self.cfg.MODEL.LOSS_FUNC:
            # before nce loss
            self.vision_proj = nn.Linear(token_dim, 256)
            self.audio_proj = nn.Linear(token_dim, 256)

        # spatial pooling
        self.vision_pool = nn.Conv3d(token_dim, token_dim, kernel_size=(1, 8, 8), stride=1)
        self.audio_pool = nn.Conv3d(token_dim, token_dim, kernel_size=(1, 8, 8), stride=1)
        self.audio_pool2 = nn.Conv3d(token_dim, token_dim, kernel_size=(1, 8, 8), stride=1)

        # ======================= Temporal Fusion ======================
        self.temporal_fusion = TemporalBlock(
            dim=token_dim,
            dim_out=token_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=self.drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
            kernel_q=[1, 1, 1],
            kernel_kv=[1, 1, 1],  # consider [3, 3, 3]
            stride_q=[1, 1, 1],
            stride_kv=[1, 1, 1],
            mode=mode,
            has_cls_embed=self.cls_embed_on,
            pool_first=pool_first
        )

        # =============================== Spatial Fusion ===============================
        self.spatial_fusion = SpatialBlock(
            dim=token_dim,
            dim_out=token_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=self.drop_rate,
            drop_path=0.0,
            norm_layer=norm_layer,
            kernel_q=[1, 1, 1],
            kernel_kv=[1, 1, 1],  # consider [3, 3, 3]
            stride_q=[1, 1, 1],
            stride_kv=[1, 1, 1],
            mode=mode,
            has_cls_embed=self.cls_embed_on,
            pool_first=pool_first,
            return_audio_attn=self.spatial_audio_attn
        )

        # =============================== TransDecoder ===============================
        decode_dim_in = [768, 768, 384, 192]
        # decode_dim_in = [768*2, 768, 384, 192]  # if two features are concatenated along channels
        decode_dim_out = [768, 384, 192, 96]
        decode_num_heads = [8, 4, 4, 2]
        decode_dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, 4)]
        decode_kernel_q = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_kernel_kv = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_stride_q = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 1, 1]]  # upsample stride
        decode_stride_kv = [[1, 2, 2], [1, 4, 4], [1, 8, 8], [1, 16, 16]]  # lagekv
        for i in range(len(decode_dim_in)):
            decoder_block = MultiScaleDecoderBlock(
                dim=decode_dim_in[i],
                dim_out=decode_dim_out[i],
                num_heads=decode_num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=0,
                norm_layer=norm_layer,
                kernel_q=decode_kernel_q[i] if len(decode_kernel_q) > i else [],
                kernel_kv=decode_kernel_kv[i] if len(decode_kernel_kv) > i else [],
                stride_q=decode_stride_q[i] if len(decode_stride_q) > i else [],
                stride_kv=decode_stride_kv[i] if len(decode_stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )

            setattr(self, f'decode_block{i+1}', decoder_block)

        self.classifier = nn.Conv3d(96, 1, kernel_size=1)

        # =============================== Initialization ===============================
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            trunc_normal_(self.pos_embed_spatial_audio, std=0.02)
            trunc_normal_(self.pos_embed_temporal_audio, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.pos_embed_audio, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class", "cls_token"}
                else:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"}
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, y, return_embed=False, return_spatial_attn=False, return_temporal_attn=False):
        inpt = x[0]  # size (B, 3, 8, 256, 256)
        x = self.patch_embed(inpt)  # size (B, 16384, 96)  16384 = 4*64*64
        y = self.patch_embed_audio(y)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]  # 4
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]  # 64
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]  # 64
        B, N, C = x.shape  # B, 16384, 96

        T_audio = self.cfg.DATA.NUM_FRAMES // self.patch_stride_audio[0]  # 4
        H_audio = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride_audio[1]  # 64
        W_audio = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride_audio[2]  # 64
        B_audio, N_audio, C_audio = x.shape  # B, 16384, 96

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1) \
                        + torch.repeat_interleave(self.pos_embed_temporal, self.patch_dims[1] * self.patch_dims[2], dim=1)
            pos_embed_audio = self.pos_embed_spatial_audio.repeat(1, self.patch_dims_audio[0], 1) \
                              + torch.repeat_interleave(self.pos_embed_temporal_audio, self.patch_dims_audio[1] * self.patch_dims_audio[2], dim=1)
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
            y = y + pos_embed_audio
        else:
            x = x + self.pos_embed
            y = y + self.pos_embed_audio

        if self.drop_rate:
            x = self.pos_drop(x)
            y = self.pos_drop(y)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        thw_audio = [T_audio, H_audio, W_audio]
        inter_feat = [[x, thw]]  # record features to be integrated in decoder

        # Block1, video layer * 1 + audio layer * 1
        for i, blk in enumerate(self.blocks[:1]):
            x, thw = blk(x, thw)
        inter_feat.append([x, thw])
        for i, blk in enumerate(self.blocks_audio[:1]):  # 16 layers, 8 layers, 4 layers
            y, thw_audio = blk(y, thw_audio)

        # Block2, video layer * 2 + audio layer * 2
        for i, blk in enumerate(self.blocks[1:3]):
            x, thw = blk(x, thw)
        inter_feat.append([x, thw])
        for i, blk in enumerate(self.blocks_audio[1:2]):  # 4 layers
            y, thw_audio = blk(y, thw_audio)

        # Block3, video layer * 11 + audio layer * 11
        for i, blk in enumerate(self.blocks[3:14]):
            x, thw = blk(x, thw)
        inter_feat.append([x, thw])
        for i, blk in enumerate(self.blocks_audio[2:3]):  # 4 layers
            y, thw_audio = blk(y, thw_audio)

        # Block4, video layer * 2 + audio layer * 2
        for i, blk in enumerate(self.blocks[14:]):
            x, thw = blk(x, thw)
        for i, blk in enumerate(self.blocks_audio[3:]):  # 4 layers
            y, thw_audio = blk(y, thw_audio)

        # ==================================== Spatial-Temporal Fusion (Parallel) ====================================
        # Spatial
        B_a, N_a, C_a = y.size()
        x_spatial = x  # (B, 256, 768)
        y_ori_fold = y.reshape(B_a, *thw_audio, C_a)  # (B, 4, 8, 8, 768)
        y_spatial = y_ori_fold

        y_spatial = self.audio_pool(y_spatial.permute(0, 4, 1, 2, 3))  # (B, 768, 4, 1, 1)
        y_spatial = y_spatial.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, 4, 768)
        y_tmp = y_spatial

        av_spatial = torch.cat([x_spatial, y_spatial], dim=1)  # (B, 260, 768)
        if not self.spatial_audio_attn and not return_spatial_attn:
            av_spatial, _ = self.spatial_fusion(av_spatial, thw_shape=thw)  # (B, 260, 768)
        elif not self.spatial_audio_attn and return_spatial_attn:
            av_spatial, _, spatial_attn = self.spatial_fusion(av_spatial, thw_shape=thw, return_spatial_attn=return_spatial_attn)
        else:
            av_spatial, _, audio_attn = self.spatial_fusion(av_spatial, thw_shape=thw)  # (B, 260, 768)
        x_spatial = av_spatial[:, :x.size(1), :]  # (B, 256, 768)
        y_spatial = av_spatial[:, x.size(1):, :]  # (B, 4, 768)

        # Temporal
        B_v, N_v, C_v = x.size()
        x_ori_fold = x.reshape(B_v, *thw, C_v)  # (B, 4, 8, 8, 768)
        x_temporal = x_ori_fold
        if self.spatial_audio_attn:
            audio_attn = audio_attn.mean(dim=1).unsqueeze(-1)
            x_temporal = x_temporal * audio_attn

        x_temporal = self.vision_pool(x_temporal.permute(0, 4, 1, 2, 3))  # (B, 768, 4, 1, 1)
        x_temporal = x_temporal.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, 4, 768)
        y_temporal = self.audio_pool2(y_ori_fold.permute(0, 4, 1, 2, 3))  # (B, 768, 4, 1, 1)
        y_temporal = y_temporal.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, 4, 768)

        av_temporal = torch.cat([x_temporal, y_temporal], dim=1)  # (B, 8, 768)
        if not return_temporal_attn:
            av_temporal, _ = self.temporal_fusion(av_temporal, thw_shape=(2, 2, 2))
        else:
            av_temporal, _, temporal_attn = self.temporal_fusion(av_temporal, thw_shape=(2, 2, 2), return_temporal_attn=return_temporal_attn)

        # Reweight
        x_weights = av_temporal[:, :x_temporal.size(1), :]  # (B, 4, 768)
        x_tmp = x_spatial.reshape(B_v, *thw, C_v)  # (B, 4, 8, 8, 768)
        x_reweight = x_tmp * x_weights.unsqueeze(2).unsqueeze(3)  # (B, 4, 8, 8, 768)
        x_reweight = x_reweight.reshape(B_v, N_v, C_v)  # (B, 256, 768)  # input to decoder

        y_weights = av_temporal[:, x_temporal.size(1):, :]  # (B, 4, 768)
        y_reweight = y_ori_fold * y_weights.unsqueeze(2).unsqueeze(3)  # (B, 4, 8, 8, 768)
        y_reweight = y_reweight.reshape(B_a, N_a, C_a)  # (B, 256, 768)
        # ============================================================================================================


        # Decoder (Transformer)
        feat, thw = self.decode_block1(x_reweight, thw)  # (B, 1024, 768)  1024 = 4*16*16
        # feat, thw = self.decode_block1(av_fuse, thw)  # (B, 1024, 768)  1024 = 4*16*16
        feat = feat + (inter_feat[-1][0] if self.global_embed_on is False else inter_feat[-1][0][:, self.global_embed_num:, :])

        feat, thw = self.decode_block2(feat, thw)  # (B, 4096, 384)  4096 = 4*32*32
        feat = feat + (inter_feat[-2][0] if self.global_embed_on is False else inter_feat[-2][0][:, self.global_embed_num:, :])

        feat, thw = self.decode_block3(feat, thw)  # (B, 16384, 192)  16384 = 4*64*64
        feat = feat + (inter_feat[-3][0] if self.global_embed_on is False else inter_feat[-3][0][:, self.global_embed_num:, :])

        feat, thw = self.decode_block4(feat, thw)  # (B, 32768, 96)  16384 = 8*64*64
        feat = feat.reshape(feat.size(0), *thw, feat.size(2)).permute(0, 4, 1, 2, 3)
        en_feat, thw = inter_feat[-4]
        en_feat = en_feat.reshape(en_feat.size(0), *thw, en_feat.size(2)).permute(0, 4, 1, 2, 3)
        feat = feat + F.interpolate(en_feat, size=(thw[0]*2, thw[1], thw[2]), mode='trilinear')

        feat = self.classifier(feat)

        if not return_embed and not return_spatial_attn and not return_temporal_attn:
            return feat
        elif not return_embed and (return_spatial_attn or return_temporal_attn):
            variable = [feat]
            if return_spatial_attn:
                variable.append(spatial_attn)
            if return_temporal_attn:
                variable.append(temporal_attn)
            return variable
        else:
            x_return = x_reweight.mean(dim=1)
            y_return = y_reweight.mean(dim=1)
            x_return = self.vision_proj(x_return)
            y_return = self.audio_proj(y_return)

            return [feat, x_return, y_return]

