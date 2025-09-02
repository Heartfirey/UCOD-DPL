import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_learnable_query=False, return_attn_map=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_learnable_query = use_learnable_query
        self.return_attn_map = return_attn_map

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_learnable_query:
            self.learnable_query = nn.Parameter(torch.randn(1, 1, dim))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        if self.use_learnable_query:
            query = self.learnable_query.expand(B, -1, -1)
        else:
            query = x
        q = self.q(query).reshape(B, query.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x)
            if kv.numel() == 0:
                return x
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, query.shape[1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.return_attn_map:
            return x, attn
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_learnable_query=False, return_attn_map=False):
        super().__init__()
        self.return_attn_map = return_attn_map
        self.use_learnable_query = use_learnable_query
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            sr_ratio=sr_ratio,use_learnable_query=use_learnable_query, return_attn_map=return_attn_map)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        attn_out = self.attn(self.norm1(x), H, W)
        if self.return_attn_map:
            features, attn = attn_out
        else:
            features = attn_out
        if self.use_learnable_query:
            x = self.attn.learnable_query.expand(x.shape[0], -1, -1) 
        x = x + self.drop_path(features)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.return_attn_map:
            return x, attn
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.norm_mlp = norm_layer(dim)

    def forward(self, query, context):
        """
        query: (B, Q, C)
        context: (B, N, C)
        """
        q = self.norm_q(query)
        kv = self.norm_kv(context)

        # MultiheadAttention expects (B, L, D)
        attn_output, _ = self.attn(q, kv, kv)  # cross-attention
        x = query + self.drop_path(attn_output)  # residual

        mlp_out = self.mlp(self.norm_mlp(x))
        x = x + self.drop_path(mlp_out)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

