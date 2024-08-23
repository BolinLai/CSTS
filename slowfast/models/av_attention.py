import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models.common import DropPath, Mlp


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True,
                   has_global_embed=False, global_embed_num=1, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    # assert not (has_cls_embed and has_global_embed), "Can't use both class embedding and global embedding."
    if has_global_embed:
        global_tok, tensor = tensor[:, :, :global_embed_num, :], tensor[:, :, global_embed_num:, :]
    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous())

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if has_global_embed:
        tensor = torch.cat((global_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        has_global_embed=False,
        global_embed_num=1,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        reverse=False
    ):
        super().__init__()
        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.has_global_embed = has_global_embed
        self.global_embed_num = global_embed_num
        self.reverse = reverse
        # assert not (has_cls_embed and has_global_embed), "Can't use both class embedding and global embedding."
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)  # this implementation is the same as the pretrained weight
        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim * 2, dim)  # if output is dim
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (pool_op(kernel_q, stride_q, padding_q, ceil_mode=False) if len(kernel_q) > 0 else None)
            self.pool_k = (pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None)
            self.pool_v = (pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None)
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(head_dim, head_dim, kernel_q, stride=stride_q, padding=padding_q, groups=head_dim, bias=False)
                if len(kernel_q) > 0 else None)
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim, bias=False)
                if len(kernel_kv) > 0 else None)
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim, bias=False)
                if len(kernel_kv) > 0 else None)
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape, return_temporal_attn=False):
        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(q, self.pool_q, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_q if hasattr(self, "norm_q") else None)
        k, k_shape = attention_pool(k, self.pool_k, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_k if hasattr(self, "norm_k") else None)
        v, v_shape = attention_pool(v, self.pool_v, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_v if hasattr(self, "norm_v") else None)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        N = q.shape[2]
        if not self.reverse:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
        else:
            v_reverse = torch.cat((v[:, :, 4:, :], v[:, :, :4, :]), dim=2)  # reverse the video and audio values
            x = (attn @ v_reverse).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        if not return_temporal_attn:
            return x, q_shape
        else:
            return x, q_shape, attn


class TemporalBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        has_global_embed=False,
        global_embed_num=1,
        pool_first=False,
        reverse=False
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        # assert not (has_cls_embed and has_global_embed), "Can't use both class embedding and global embedding."
        self.attn = TemporalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            has_global_embed=has_global_embed,
            global_embed_num=global_embed_num,
            mode=mode,
            pool_first=pool_first,
            reverse=reverse
        )

        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.has_global_embed = has_global_embed
        self.global_embed_num = global_embed_num
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        # self.pool_skip = (
        #     nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
        #     if len(kernel_skip) > 0 else None)
        self.pool_skip = None

    def forward(self, x, thw_shape, return_temporal_attn=False):
        if not return_temporal_attn:
            x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        else:
            x_block, thw_shape_new, temporal_attn = self.attn(self.norm1(x), thw_shape, return_temporal_attn=return_temporal_attn)
        x_res, _ = attention_pool(x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed,
                                  has_global_embed=self.has_global_embed, global_embed_num=self.global_embed_num)
        x = x_res + self.drop_path(x_block)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        if not return_temporal_attn:
            return x, thw_shape_new
        else:
            return x, thw_shape_new, temporal_attn


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        has_global_embed=False,
        global_embed_num=1,
        mode="conv",  # Options include `conv`, `avg`, and `max`.
        pool_first=False,  # If True, perform pool before projection.
        return_audio_attn=False
    ):
        super().__init__()
        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.has_global_embed = has_global_embed
        self.global_embed_num = global_embed_num
        self.return_audio_attn = return_audio_attn
        # assert not (has_cls_embed and has_global_embed), "Can't use both class embedding and global embedding."
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)  # this implementation is the same as the pretrained weight
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (pool_op(kernel_q, stride_q, padding_q, ceil_mode=False) if len(kernel_q) > 0 else None)
            self.pool_k = (pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None)
            self.pool_v = (pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None)
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(head_dim, head_dim, kernel_q, stride=stride_q, padding=padding_q, groups=head_dim, bias=False)
                if len(kernel_q) > 0 else None)
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim, bias=False)
                if len(kernel_kv) > 0 else None)
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim, bias=False)
                if len(kernel_kv) > 0 else None)
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape, return_spatial_attn=False):
        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(q, self.pool_q, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_q if hasattr(self, "norm_q") else None)
        k, k_shape = attention_pool(k, self.pool_k, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_k if hasattr(self, "norm_k") else None)
        v, v_shape = attention_pool(v, self.pool_v, thw_shape, has_cls_embed=self.has_cls_embed, has_global_embed=self.has_global_embed,
                                    global_embed_num=self.global_embed_num, norm=self.norm_v if hasattr(self, "norm_v") else None)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # in-frame attention for 4x8x8 and 4x1
        offset = torch.full(size=(attn.size()[2:]), fill_value=1e8, device=attn.device)
        T, HW = thw_shape[0], thw_shape[1] * thw_shape[2]
        THW = T * HW
        for t in range(T):
            offset[HW*t:HW*(t+1), HW*t:HW*(t+1)] = 0
            offset[HW*t:HW*(t+1), THW+t] = 0
            offset[THW+t, HW*t:HW*(t+1)] = 0
            offset[THW+t, THW+t] = 0

        attn = attn - offset
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        if not self.return_audio_attn and not return_spatial_attn:
            return x, q_shape
        elif not self.return_audio_attn and return_spatial_attn:  # for visualization
            return x, q_shape, attn
        else:  # output audio attention mask to reweight temporal fusion
            T, H, W = thw_shape[0], thw_shape[1], thw_shape[2]
            HW = H * W
            THW = T * HW
            audio_attn = [attn[:, :, THW+t, HW*t:HW*(t+1)] for t in range(T)]
            audio_attn = torch.stack(audio_attn, dim=2)
            audio_attn_max = audio_attn.max(dim=-1, keepdim=True)[0]
            audio_attn_min = audio_attn.min(dim=-1, keepdim=True)[0]
            audio_rescale = (audio_attn - audio_attn_min) / (audio_attn_max - audio_attn_min + 1e-8)
            audio_rescale = audio_rescale.reshape(audio_rescale.size(0), audio_rescale.size(1), T, H, W)
            return x, q_shape, audio_rescale


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        has_global_embed=False,
        global_embed_num=1,
        pool_first=False,
        return_audio_attn=False
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        # assert not (has_cls_embed and has_global_embed), "Can't use both class embedding and global embedding."
        self.attn = SpatialAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            has_global_embed=has_global_embed,
            global_embed_num=global_embed_num,
            mode=mode,
            pool_first=pool_first,
            return_audio_attn=return_audio_attn
        )

        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.has_global_embed = has_global_embed
        self.global_embed_num = global_embed_num
        self.return_audio_attn = return_audio_attn
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        # self.pool_skip = (
        #     nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
        #     if len(kernel_skip) > 0 else None)
        self.pool_skip = None

    def forward(self, x, thw_shape, return_spatial_attn=False):
        if not self.return_audio_attn and not return_spatial_attn:
            x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        elif not self.return_audio_attn and return_spatial_attn:  # for visualization
            x_block, thw_shape_new, spatial_attn = self.attn(self.norm1(x), thw_shape, return_spatial_attn=True)
        else:  # for reweighting to temporal fusion
            x_block, thw_shape_new, audio_attn = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed,
                                  has_global_embed=self.has_global_embed, global_embed_num=self.global_embed_num)
        x = x_res + self.drop_path(x_block)

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        if not self.return_audio_attn and not return_spatial_attn:
            return x, thw_shape_new
        elif not self.return_audio_attn and return_spatial_attn:
            return x, thw_shape_new, spatial_attn
        else:
            return x, thw_shape_new, audio_attn
