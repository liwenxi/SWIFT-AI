import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmdet.vtpack.layers import DynamicGrainedEncoder
from mmdet.vtpack.layers.sparse_ops import batched_sparse_attention, batched_sparse_gemm

from mmdet.registry import MODELS
__all__ = [
    "pvt_dge_s124_tiny_256", "pvt_dge_s124_small_256",
    "pvt_dge_s124_medium_256", "pvt_dge_s124_large_256",
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def complexity(self, num_inputs, num_queries):
        comp = num_queries * self.fc1.in_features * self.fc1.out_features  # fc1
        comp += num_queries * self.fc1.out_features  # act
        comp += num_queries * self.fc2.in_features * self.fc2.out_features  # fc2
        return comp

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def complexity(self, num_inputs, num_queries):
        num_channels = self.dim
        comp = num_queries * (num_channels ** 2)  # q embed
        if self.sr_ratio > 1:
            comp += num_inputs * (num_channels ** 2)  # sr
            comp += num_inputs / (self.sr_ratio ** 2) * num_channels  # norm
        comp += (num_inputs / (self.sr_ratio ** 2) * (num_channels ** 2)) * 2  # kv embed
        comp += num_queries * num_inputs / (self.sr_ratio ** 2) * num_channels  # attention
        comp += num_queries * num_inputs / (self.sr_ratio ** 2) * self.num_heads * 3  # softmax
        comp += num_queries * num_channels ** 2  # proj
        return comp

    def forward(self, x, q, H, W, q_lengths):
        B, N, C = x.shape
        q = self.q(q).reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 3, 0, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 3, 0, 1, 4)
        k, v = kv[0], kv[1]

        if not self.training:
            x = batched_sparse_attention(q, k, v, q_lengths, self.scale)
            x = x.transpose(0, 1).reshape(-1, C)
        else:
            if (q_lengths.max() - q_lengths.min()) == 0:
                q = q.reshape(self.num_heads, B, -1, C // self.num_heads)
                attn = (q @ k.transpose(-1, -2)) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = (attn @ v).permute(1, 2, 0, 3).reshape(-1, C)
            else:
                kv_lengths = q_lengths.new_full([B], kv.shape[3])
                k = k.reshape(self.num_heads, -1, C // self.num_heads)
                v = v.reshape(self.num_heads, -1, C // self.num_heads)
                attn = batched_sparse_gemm(q, k, q_lengths, kv_lengths, False, True) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = batched_sparse_gemm(attn, v, q_lengths, kv_lengths, False, False)
                x = x.transpose(0, 1).reshape(-1, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, split_sizes=[2, 1]):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dge = DynamicGrainedEncoder(in_channels=dim, split_sizes=split_sizes, complexity_handler=self.complexity)

    def complexity(self, num_inputs, num_queries):
        comp = num_inputs * self.dim * 2  # norm1 and norm2
        comp += self.attn.complexity(num_inputs, num_queries)
        comp += self.mlp.complexity(num_inputs, num_queries)
        return comp

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.dge.compress(x, H, W)
        q = q + self.drop_path(self.attn(self.norm1(x), self.norm1(q), H, W, self.dge.states["batches"]))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        x = self.dge.decompress(q)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def complexity(self, num_inputs, num_queries):
        comp = num_inputs * self.proj.in_channels * self.proj.out_channels
        comp += num_inputs / (self.patch_size[0] * self.patch_size[1]) * self.proj.out_channels  # norm
        return comp

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

# @MODELS.register_module()
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
                 split_sizes=[[2, 1], [2, 1], [2, 1], [2, 1]]):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i], split_sizes=split_sizes[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3]) if num_classes > 0 else nn.Identity()

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.apply(self._init_dynamic_weights)

        # feature attributes
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {f"res{2+i}": dim for i, dim in enumerate(embed_dims)}
        self._size_divisibility = 32 * max(split_sizes[-1])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_dynamic_weights(self, m):
        if isinstance(m, DynamicGrainedEncoder):
            m.router.init_parameters()

    def complexity(self, x):
        N = x.shape[1] * x.shape[2]
        comp = self.embed_dims[-1] * self.num_classes  # classifier
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            comp += patch_embed.complexity(N, N)
            N = N / (patch_embed.patch_size[0] * patch_embed.patch_size[1])

        comp_static, comp_dynamic = [], []

        def append_complexity(m):
            if isinstance(m, DynamicGrainedEncoder):
                comp = m.get_complexity()
                comp_static.append(comp["static"])
                comp_dynamic.append(comp["dynamic"])

        self.apply(append_complexity)
        comp_static = (sum(comp_static) + comp).mean()
        comp_dynamic = (sum(comp_dynamic) + comp).mean()
        return {"static": comp_static, "dynamic": comp_dynamic}

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wdc = {"cls_token"}
        # no_wdc.update({f"pos_embed{i+1}" for i in range(self.num_stages)})
        return no_wdc

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        x = self.forward_features(x)

        # if self.F4:
        #     x = x[3:4]
        x = x[1:4]
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def forward_features(self, x):
    outs = []

    B = x.shape[0]

    for i in range(self.num_stages):
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        pos_embed = getattr(self, f"pos_embed{i + 1}")
        pos_drop = getattr(self, f"pos_drop{i + 1}")
        block = getattr(self, f"block{i + 1}")
        x, (H, W) = patch_embed(x)
        if i == self.num_stages - 1:
            pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
        else:
            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

        x = pos_drop(x + pos_embed)
        for blk in block:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

    return outs


def forward(self, x):
    x = self.forward_features(x)

    if self.F4:
        x = x[3:4]

    return x

@MODELS.register_module()
class pvt_tiny(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_tiny, self).__init__(
            img_size=256, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], split_sizes=[[4, 2, 1], [4, 2, 1], [4, 2, 1], [4, 2, 1]])


@MODELS.register_module()
class pvt_small(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_small, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])


@MODELS.register_module()
class pvt_small_f4(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_small_f4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, F4=True, pretrained=kwargs['pretrained'])


@MODELS.register_module()
class pvt_medium(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_medium, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1], pretrained=kwargs['pretrained'])


@MODELS.register_module()
class pvt_large(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_large, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1], pretrained=kwargs['pretrained'])