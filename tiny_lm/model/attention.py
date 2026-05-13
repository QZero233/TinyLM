import math
from typing import Optional, Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import Softmax, Linear
from .embedding import RoPE
from .kv_cache import KVCacheState

class SPDAAttention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.softmax = Softmax()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        pre_softmax = (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        if mask is not None:
            mask_val = torch.zeros_like(pre_softmax)
            mask_val[~mask] = -torch.inf
            pre_softmax += mask_val

        return self.softmax(pre_softmax) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: Optional[int] = None, rope_theta: Optional[float] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)

        self.attention = SPDAAttention()

        if rope_theta is not None:
            self.rope = RoPE(rope_theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: bool = True,
                rope_token_positions: Optional[torch.IntTensor] = None,
                kv_cache_state: Optional[KVCacheState] = None) -> torch.Tensor:
        batch_dim_shape = q.shape[:-2]
        proj_q = self.linear_q(q)
        proj_k = self.linear_k(k)
        proj_v = self.linear_v(v)

        # 如果有KV Cache，那么需要把缺失的KV拼接上去，因为这里的输入序列长度只有未缓存的部分
        if kv_cache_state is not None:
            cached_k, cached_v = kv_cache_state.k, kv_cache_state.v
            kv_cache_state.append_kv(proj_k, proj_v)

            if cached_k is not None:
                proj_k = torch.cat([cached_k, proj_k], dim=-2)
                proj_v = torch.cat([cached_v, proj_v], dim=-2)

        # (...batch, seq_len, d_model) -> (...batch, seq_len, num_heads, d_k) -> (...batch, num_heads, seq_len, d_k)
        qs = proj_q.reshape((*batch_dim_shape, proj_q.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)
        ks = proj_k.reshape((*batch_dim_shape, proj_k.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)
        vs = proj_v.reshape((*batch_dim_shape, proj_v.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)

        mask = None
        if causal_mask:
            q_len = proj_q.shape[-2]
            k_len = proj_k.shape[-2]
            q_start = k_len - q_len
            q_positions = torch.arange(q_len, device=proj_q.device) + q_start
            k_positions = torch.arange(k_len, device=proj_q.device)
            mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            mask = mask.broadcast_to((*proj_q.shape[:-2], self.num_heads, q_len, k_len))

        if self.rope is not None:
            qs = self.rope(qs, rope_token_positions)
            ks = self.rope(ks, rope_token_positions)
        attention_res = self.attention(qs, ks, vs, mask)

        # (...batch, num_heads, seq_len, d_k) -> (...batch, seq_len, num_heads, d_k) -> (...batch, seq_len, d_model)
        attention_res = attention_res.transpose(-2, -3)
        attention_res = attention_res.reshape((*batch_dim_shape, attention_res.shape[-3], -1))

        return self.linear_out(attention_res)


class TorchMultiHeadAttention(MultiHeadAttention):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: bool = True,
                rope_token_positions: Optional[torch.IntTensor] = None,
                kv_cache_state: Optional[KVCacheState] = None) -> torch.Tensor:
        if kv_cache_state is not None:
            raise NotImplementedError("TorchMultiHeadAttention does not support kv_cache_state")

        batch_dim_shape = q.shape[:-2]
        proj_q = self.linear_q(q)
        proj_k = self.linear_k(k)
        proj_v = self.linear_v(v)

        qs = proj_q.reshape((*batch_dim_shape, proj_q.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)
        ks = proj_k.reshape((*batch_dim_shape, proj_k.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)
        vs = proj_v.reshape((*batch_dim_shape, proj_v.shape[-2], self.num_heads, self.d_k)).transpose(-2, -3)

        if self.rope is not None:
            qs = self.rope(qs, rope_token_positions)
            ks = self.rope(ks, rope_token_positions)

        attention_res = F.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal_mask,
        )

        attention_res = attention_res.transpose(-2, -3)
        attention_res = attention_res.reshape((*batch_dim_shape, attention_res.shape[-3], -1))
        return self.linear_out(attention_res)
