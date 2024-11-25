import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.cnn.bricks.transformer import FFN, build_dropout
# from mmcv.cnn.utils.weight_init import trunc_normal_
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader
# from mmcv.runner import BaseModule, ModuleList
# from mmcv.utils import to_2tuple
from mmengine.utils import to_2tuple

# from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
# from ...utils import get_root_logger
from mmengine.logging import MMLogger
from mmdet.registry import MODELS
# from ..builder import BACKBONES
# from ..utils.ckpt_convert import giga_converter
# from ..utils.transformer import PatchEmbed, PatchMerging
from ..layers import PatchEmbed, PatchMerging

class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.
    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.out_drop = DROPOUT_LAYERS.build(dropout_layer)
        self.out_drop = build_dropout(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class LocalWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        # self.w_msa = MultiheadAttention(
        #     embed_dims=embed_dims,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=proj_drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=0),
        #     init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape, keep_token_indices):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            shifted_query = query
            attn_mask = None


        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        ########
        query_windows = query_windows.view(B, -1, self.window_size, self.window_size, C)

        K = keep_token_indices.shape[1]
        if attn_mask is not None:
            attn_mask_sparse = torch.gather(attn_mask, sparse_grad=True, dim=1,
                                     index=keep_token_indices.view(B, K, 1, 1).repeat(1, 1, self.window_size**2,
                                                                                         self.window_size**2))

            attn_mask_sparse =  attn_mask_sparse.view(-1, self.window_size**2, self.window_size**2)

        else:
            attn_mask_sparse = None

        query_windows_sparse = torch.gather(query_windows, sparse_grad=True, dim=1,
                                            index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
                                                                                          self.window_size, C))
        query_windows_sparse = query_windows_sparse.view(-1, self.window_size**2, C)


        # # This is a memory-save way
        # query_windows_sparse = query_windows.view(-1, self.window_size**2, C)


        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows_sparse, mask=attn_mask_sparse)
        # attn_windows = self.w_msa(query_windows_sparse, )

        #####
        attn_windows = attn_windows.view(B, -1, self.window_size, self.window_size, C)
        attn_windows = query_windows.scatter(src=attn_windows, dim=1,
                                     index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
                                                                                         self.window_size, C))
        #####

        # # This is a memory-save way
        # attn_windows = attn_windows.view(B, -1, self.window_size, self.window_size, C)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)


        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class LocalBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(LocalBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.window_size = window_size

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = LocalWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        # self.attn = MultiheadAttention(
        #     embed_dims=embed_dims,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False,
            init_cfg=None)

    def forward(self, x, hw_shape, keep_token_indices):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)

            # B, L, C = x.shape
            # H, W = hw_shape
            # assert L == H * W, 'input feature has wrong size'
            # x = x.view(B, H, W, C)
            #
            # # pad feature maps to multiples of window size
            # pad_r = (self.window_size - W % self.window_size) % self.window_size
            # pad_b = (self.window_size - H % self.window_size) % self.window_size
            # #
            # # # x_pooled = self.pooling(x.permute(0, 3, 1, 2))
            # # x_pooled = F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=self.window_size, stride=self.window_size, padding=(3, 3))
            #
            # x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            # H_pad, W_pad = x.shape[1], x.shape[2]
            #
            # query_windows = self.window_partition(x).view(-1, self.window_size ** 2, C)
            # x = query_windows

            x = self.attn(x, hw_shape, keep_token_indices)
            # x = self.attn(x)

            # x = self.window_reverse_swin(x, H_pad, W_pad)
            #
            # x = x.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size,
            #            self.window_size, -1)
            # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
            #
            # if pad_r > 0 or pad_b:
            #     x = x[:, :H, :W, :].contiguous()
            # x = x.view(B, H * W, C)

            x = x + identity

            identity = x
            x = self.norm2(x)


            ###
            B, L, C = x.shape
            H, W = hw_shape
            assert L == H * W, 'input feature has wrong size'
            query = x.view(B, H, W, C)

            # pad feature maps to multiples of window size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
            H_pad = query.shape[1]
            W_pad = query.shape[2]
            # H_pad, W_pad = query.shape[1], query.shape[2]                                            self.window_size,

            # nW*B, window_size, window_size, C
            query_windows = self.window_partition(query)
            # nW*B, window_size*window_size, C
            query_windows = query_windows.view(-1, self.window_size ** 2, C)

            query_windows = query_windows.view(B, -1, self.window_size, self.window_size, C)

            K = keep_token_indices.shape[1]

            query_windows_sparse = torch.gather(query_windows, sparse_grad=True, dim=1,
                                                index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1,
                                                                                                    self.window_size,
                                                                                                    self.window_size,
                                                                                                    C))

            #
            # identity = identity.view(B, H, W, C)
            #
            #
            # identity = F.pad(identity, (0, 0, 0, pad_r, 0, pad_b))
            #
            #
            # # nW*B, window_size, window_size, C
            # identity = self.window_partition(identity)
            # # nW*B, window_size*window_size, C
            # identity = identity.view(-1, self.window_size ** 2, C)
            #
            # identity = identity.view(B, -1, self.window_size, self.window_size, C)
            #
            # identity = torch.gather(identity, sparse_grad=True, dim=1,
            #                                     index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1,
            #                                                                                         self.window_size,
            #                                                                                         self.window_size,
            #                                                                                         C))
            # print(query_windows_sparse.shape, identity.shape)
            query_ffn = self.ffn(query_windows_sparse, identity=identity)

            ###
            # x = self.ffn(x, identity=identity)

            ###

            #####
            query_ffn = query_ffn.view(B, -1, self.window_size, self.window_size, C)
            x = query_windows.scatter(src=query_ffn, dim=1,
                                                 index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1,
                                                                                                     self.window_size,
                                                                                                     self.window_size,
                                                                                                     C))


            # merge windows
            x = x.view(-1, self.window_size, self.window_size, C)

            # B H' W' C
            x = self.window_reverse(x, H_pad, W_pad)
            if pad_r > 0 or pad_b:
                x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)
            x = x + identity
            ####

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

    def window_reverse_swin(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            # nW = mask.shape[0]
            # attn = attn.view(B // nW, nW, self.num_heads, N,
            #                  N) + mask.unsqueeze(1).unsqueeze(0)
            # attn = attn.view(-1, self.num_heads, N, N)
            attn += mask.unsqueeze(1)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# class ShiftWindowMSA(BaseModule):
#     """Shifted Window Multihead Self-Attention Module.
#
#     Args:
#         embed_dims (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         window_size (int): The height and width of the window.
#         shift_size (int, optional): The shift step of each window towards
#             right-bottom. If zero, act as regular window-msa. Defaults to 0.
#         qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
#             Default: True
#         qk_scale (float | None, optional): Override default qk scale of
#             head_dim ** -0.5 if set. Defaults: None.
#         attn_drop_rate (float, optional): Dropout ratio of attention weight.
#             Defaults: 0.
#         proj_drop_rate (float, optional): Dropout ratio of output.
#             Defaults: 0.
#         dropout_layer (dict, optional): The dropout_layer used before output.
#             Defaults: dict(type='DropPath', drop_prob=0.).
#         init_cfg (dict, optional): The extra config for initialization.
#             Default: None.
#     """
#
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  window_size,
#                  shift_size=0,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  attn_drop_rate=0,
#                  proj_drop_rate=0,
#                  dropout_layer=dict(type='DropPath', drop_prob=0.),
#                  init_cfg=None):
#         super().__init__(init_cfg)
#
#         self.window_size = window_size
#         self.shift_size = shift_size
#         assert 0 <= self.shift_size < self.window_size
#
#         self.w_msa = WindowMSA(
#             embed_dims=embed_dims,
#             num_heads=num_heads,
#             window_size=to_2tuple(window_size),
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop_rate=attn_drop_rate,
#             proj_drop_rate=proj_drop_rate,
#             init_cfg=None)
#
#         self.drop = build_dropout(dropout_layer)
#
#     def forward(self, query, hw_shape, keep_token_indices):
#         B, L, C = query.shape
#         H, W = hw_shape
#         assert L == H * W, 'input feature has wrong size'
#         query = query.view(B, H, W, C)
#
#         # pad feature maps to multiples of window size
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
#         H_pad, W_pad = query.shape[1], query.shape[2]
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_query = torch.roll(
#                 query,
#                 shifts=(-self.shift_size, -self.shift_size),
#                 dims=(1, 2))
#
#             # calculate attention mask for SW-MSA
#             img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size,
#                               -self.shift_size), slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size,
#                               -self.shift_size), slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1
#
#             # nW, window_size, window_size, 1
#             mask_windows = self.window_partition(img_mask)
#             mask_windows = mask_windows.view(
#                 -1, self.window_size * self.window_size)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0,
#                                               float(-100.0)).masked_fill(
#                                                   attn_mask == 0, float(0.0))
#         else:
#             shifted_query = query
#             attn_mask = None
#
#         # nW*B, window_size, window_size, C
#         query_windows = self.window_partition(shifted_query)
#         # nW*B, window_size*window_size, C
#         query_windows = query_windows.view(-1, self.window_size**2, C)
#
#         # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
#         attn_windows = self.w_msa(query_windows, mask=attn_mask)
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size,
#                                          self.window_size, C)
#
#         # B H' W' C
#         shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(
#                 shifted_x,
#                 shifts=(self.shift_size, self.shift_size),
#                 dims=(1, 2))
#         else:
#             x = shifted_x
#
#         if pad_r > 0 or pad_b:
#             x = x[:, :H, :W, :].contiguous()
#
#         x = x.view(B, H * W, C)
#
#         x = self.drop(x)
#         return x
#
#     def window_reverse(self, windows, H, W):
#         """
#         Args:
#             windows: (num_windows*B, window_size, window_size, C)
#             H (int): Height of image
#             W (int): Width of image
#         Returns:
#             x: (B, H, W, C)
#         """
#         window_size = self.window_size
#         B = int(windows.shape[0] / (H * W / window_size / window_size))
#         x = windows.view(B, H // window_size, W // window_size, window_size,
#                          window_size, -1)
#         x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#         return x
#
#     def window_partition(self, x):
#         """
#         Args:
#             x: (B, H, W, C)
#         Returns:
#             windows: (num_windows*B, window_size, window_size, C)
#         """
#         B, H, W, C = x.shape
#         window_size = self.window_size
#         x = x.view(B, H // window_size, window_size, W // window_size,
#                    window_size, C)
#         windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
#         windows = windows.view(-1, window_size, window_size, C)
#         return windows


class GlobalMSA(BaseModule):
    """Global Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.pooling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.msa = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            # window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]


        shifted_query = query
        attn_mask = None

        global_query = self.pooling(query)
        B, H, W, C = global_query.shape

        global_query = global_query.view(B, -1, C)

        attn_global = self.msa(global_query, mask=attn_mask)


        x = attn_global.view(B, H * W, C)

        x = self.drop(x)
        return x


class LocalMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None


        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x


# class LocalBlock(BaseModule):
#     """"
#     Args:
#         embed_dims (int): The feature dimension.
#         num_heads (int): Parallel attention heads.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         window_size (int, optional): The local window scale. Default: 7.
#         shift (bool, optional): whether to shift window or not. Default False.
#         qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
#         qk_scale (float | None, optional): Override default qk scale of
#             head_dim ** -0.5 if set. Default: None.
#         drop_rate (float, optional): Dropout rate. Default: 0.
#         attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
#         drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
#         act_cfg (dict, optional): The config dict of activation function.
#             Default: dict(type='GELU').
#         norm_cfg (dict, optional): The config dict of normalization.
#             Default: dict(type='LN').
#         with_cp (bool, optional): Use checkpoint or not. Using checkpoint
#             will save some memory while slowing down the training speed.
#             Default: False.
#         init_cfg (dict | list | None, optional): The init config.
#             Default: None.
#     """
#
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  feedforward_channels,
#                  window_size=7,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  with_cp=False,
#                  init_cfg=None):
#
#         super(LocalBlock, self).__init__()
#
#         self.init_cfg = init_cfg
#         self.with_cp = with_cp
#
#         self.window_size = window_size
#
#         self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
#         self.attn = MultiheadAttention(
#             embed_dims=embed_dims,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop_rate,
#             proj_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             init_cfg=None)
#
#         self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
#         self.ffn = FFN(
#             embed_dims=embed_dims,
#             feedforward_channels=feedforward_channels,
#             num_fcs=2,
#             ffn_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             act_cfg=act_cfg,
#             add_identity=True,
#             init_cfg=None)
#
#     def forward(self, x):
#         def _inner_forward(x):
#             identity = x
#             x = self.norm1(x)
#             x = self.attn(x)
#
#             x = x + identity
#
#             identity = x
#             x = self.norm2(x)
#             x = self.ffn(x, identity=identity)
#
#             return x
#
#         if self.with_cp and x.requires_grad:
#             x = cp.checkpoint(_inner_forward, x)
#         else:
#             x = _inner_forward(x)
#
#         return x
#
#     def window_reverse(self, windows, H, W):
#         """
#         Args:
#             windows: (num_windows*B, window_size, window_size, C)
#             H (int): Height of image
#             W (int): Width of image
#         Returns:
#             x: (B, H, W, C)
#         """
#         window_size = self.window_size
#         B = int(windows.shape[0] / (H * W / window_size / window_size))
#         x = windows.view(B, H // window_size, W // window_size, window_size,
#                          window_size, -1)
#         x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#         return x
#
#     def window_partition(self, x):
#         """
#         Attention:This is different from Swin
#         Args:
#             x: (B, H, W, C)
#         Returns:
#             windows: (B, num_windows, window_size, window_size, C)
#         """
#         B, H, W, C = x.shape
#         window_size = self.window_size
#         x = x.view(B, H // window_size, window_size, W // window_size,
#                    window_size, C)
#         windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
#         windows = windows.view(B, -1, window_size, window_size, C)
#         return windows

class GlobalBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(GlobalBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class BlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 top_k = 1.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.global_blocks = ModuleList()
        for i in range(depth):
            block = GlobalBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.global_blocks.append(block)

        self.local_blocks = ModuleList()
        for i in range(depth):
            block = LocalBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.local_blocks.append(block)
        self.pooling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.downsample = downsample
        self.window_size = window_size
        self.score_net = nn.Linear(self.window_size**2*embed_dims, 1)
        self.score_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.top_k = top_k

    def forward(self, x, hw_shape):
        keep_token_indices, x_score, _ = self.score_generator(x, hw_shape)

        # x = x + x_att_global * (1-x_score)


        # transformer for Local level
        # x.shape [B, H, W, C]

        # B, nW, window_size, window_size, C
        # query_windows = self.window_partition(x).view(B, -1,  self.window_size, self.window_size, C)

        # K = keep_token_indices.shape[1]

        # B, K, window_size, window_size, C
        # query_windows_sparse = torch.gather(query_windows, sparse_grad=True, dim=1,
        #                                     index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
        #                                                                                   self.window_size, C))

        # query_windows_sparse = x
        #
        # query_windows_sparse = query_windows_sparse.view(-1, self.window_size**2, C)

        #
        # import copy
        # haha = copy.deepcopy(query_windows.detach())
        # qws = query_windows_sparse.view(B, K, self.window_size, self.window_size, -1)
        # print(haha.shape, qws.shape, keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
        #                                                                                   self.window_size, C).shape)
        # qws += 0
        # haha = haha.scatter(src=qws, dim=1,
        #                              index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
        #                                                                                  self.window_size, C))
        # print(torch.sum(haha-query_windows))

        query_windows_sparse = x
        for block in self.local_blocks:
            query_windows_sparse = block(query_windows_sparse, hw_shape, keep_token_indices)

        # print(query_windows_sparse.shape, x_score.shape)


        # print(query_windows_sparse.shape, x_score.shape)
        # query_windows_sparse = query_windows_sparse * x_score

        # reverse the sparse
        # query_windows_sparse = query_windows_sparse.view(B, K, self.window_size, self.window_size, -1)
        # C = query_windows_sparse.shape[-1]
        # query_windows = query_windows.scatter(src=query_windows_sparse, dim=1,
        #                              index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1, self.window_size,
        #                                                                                  self.window_size, C))
        # query_windows = query_windows_sparse
        # query_windows = query_windows.view(-1, self.window_size, self.window_size, C)
        #
        # x = self.window_reverse_swin(query_windows, H_pad, W_pad)
        #
        # x = x.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size,
        #                  self.window_size, -1)
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
        #
        # if pad_r > 0 or pad_b:
        #     x = x[:, :H, :W, :].contiguous()
        # x = x.view(B, H*W, C)

        # re-parameter
        x_score = 1 + x_score - x_score.detach()
        # print(query_windows_sparse.shape, x_score.shape)
        query_windows_sparse = query_windows_sparse * x_score

        x = query_windows_sparse

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


    def score_generator(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        #
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        x_pooled = self.pooling(x.permute(0, 3, 1, 2))
        # x_pooled = F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=self.window_size, stride=self.window_size, padding=(3, 3))

        H_pad, W_pad = x.shape[1], x.shape[2]



        # H_pooled, W_pooled = x_pooled.shape[2], x_pooled.shape[3]

        # make ScoreNet work
        x_mean = F.interpolate(x_pooled, scale_factor=self.window_size, mode='bilinear').permute(0, 2, 3, 1)
        x_residual = x - x_mean
        # if pad_r > 0 or pad_b:
        #     x_residual[:, H:, W:, :] *= 0

        x_residual_windows = self.window_partition(x_residual)
        x_residual_windows =  x_residual_windows.view(-1, self.window_size**2*C)


        x_windows_score = self.score_net(x_residual_windows)

        # x_windows_score = torch.randn(B*(H_pad//self.window_size)*(W_pad//self.window_size), 1).to(x.device)
        # x_windows_score = torch.randn(x_residual_windows.shape[0], 1).to(x_residual_windows.device)

        # B, nH, nW, C
        x_windows_score = self.window_reverse(x_windows_score, H_pad, W_pad)

        x_windows_score = F.softmax(x_windows_score.flatten(1), dim=1)


        # # B, K
        _, keep_token_indices = x_windows_score.topk(dim=1, k=int(self.top_k*(H_pad // self.window_size) * (W_pad // self.window_size)))



        # # B, H, W, C
        x_score = x_windows_score.view(B, 1, H_pad // self.window_size, 1, W_pad // self.window_size, 1).repeat(1, C, 1,
                                                                                                                self.window_size,
                                                                                                                1,
                                                                                                                self.window_size).view(
            B, C, H_pad, W_pad).permute(0, 2, 3, 1)

        if pad_r > 0 or pad_b:
            x_score = x_score[:, :H, :W, :].contiguous().view(B, -1, C)
        else:
            x_score = x_score.contiguous().view(B, -1, C)


        # # viz
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as CM
        # import uuid
        # print(x_score.shape)
        # # print(x_score.view(H, W, C)[:, :, 0])
        # print(torch.topk(x_score.squeeze().view(H, W, C)[:, :, 0], 3))
        # plt.imsave(str(uuid.uuid4())+".png", x_score.squeeze().view(H, W, C)[:-2, :-2, 0].cpu().numpy(), cmap='Reds')


        return keep_token_indices, x_score, None

        # return keep_token_indices, x_score, x_att_global, x_global

    def window_reverse(self, windows, H, W):
        """
        Attention: This is different from the Swin.
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, -1).contiguous()

        return x

    def window_reverse_swin(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


@MODELS.register_module()
class GigaTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 top_k=(0.7, 0.6, 0.5, 0.5),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(GigaTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = BlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                top_k = top_k[i],
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GigaTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = giga_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs
