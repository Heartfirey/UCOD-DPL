import torch
import torch.nn as nn

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

    def forward(self, x, H, W):
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
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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
    

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 use_learnable_query=False, num_queries=1, return_attn_map=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_learnable_query = use_learnable_query
        self.return_attn_map = return_attn_map
        self.num_queries = num_queries

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_learnable_query:
            self.learnable_query = nn.Parameter(torch.randn(1, num_queries, dim))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, query=None, context=None, H=None, W=None):
        B, N_ctx, C = context.shape

        if self.use_learnable_query:
            query = self.learnable_query.expand(B, -1, -1)  # [B, N_query, C]
        elif query is None:
            raise ValueError("Query must be provided if not using learnable query.")

        B, N_query, _ = query.shape

        # Q from query
        q = self.q(query).reshape(B, N_query, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # K, V from context
        if self.sr_ratio > 1:
            context_ = context.permute(0, 2, 1).reshape(B, C, H, W)
            context_ = self.sr(context_).reshape(B, C, -1).permute(0, 2, 1)
            context_ = self.norm(context_)
        else:
            context_ = context

        kv = self.kv(context_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, L_ctx, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N_query, L_ctx]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_query, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if self.return_attn_map:
            return out, attn
        return out
