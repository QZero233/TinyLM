import math
from typing import Any, Optional, List, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .norm import RMSNorm
from .attention import MultiHeadAttention
from .common import SwiGLU, Linear
from .embedding import Embedding
from .kv_cache import KVCacheState, TransformerKVCache

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024,
                 theta: float = 0.5, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, kv_cache_state: Optional[KVCacheState] = None) -> torch.Tensor:
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        norm1_x = self.norm1(x)
        x = x + self.multi_head_attn(norm1_x, norm1_x, norm1_x, kv_cache_state=kv_cache_state)
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int,
                 num_heads: int, d_ff: int,
                 theta: float = 0.5, gradient_checkpoint: bool = True,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.gradient_checkpoint = gradient_checkpoint

        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length,theta) for _ in range(num_layers)])
        self.embedding = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, kv_cache: Optional[TransformerKVCache]) -> torch.Tensor:
        # Input: (batch_size, seq_len)
        # Output: (batch_size, seq_len, vocab_size)
        if kv_cache is not None:
            cached_len = len(kv_cache)
            x = x[..., cached_len:]
        x = self.embedding(x)

        for i, layer in enumerate(self.transformer_layers):
            kv_cache_state = kv_cache[i] if kv_cache is not None else None
            if self.training and self.gradient_checkpoint:
                x = checkpoint(layer, x, kv_cache_state=kv_cache_state, use_reentrant=False)
            else:
                x = layer(x, kv_cache_state=kv_cache_state)
        x = self.norm(x)
        x = self.linear(x)
        return x

    def resize_embedding(self, new_size: int):
        origin_vocab_size, d_model = self.embedding.embedding_matrix.shape
        assert new_size > origin_vocab_size

        new_embedding_weight = torch.empty((new_size, d_model))
        nn.init.trunc_normal_(new_embedding_weight, mean=0, std=1, a=-3, b=3)
        new_embedding_weight[:origin_vocab_size, :] = self.embedding.embedding_matrix
        self.embedding.embedding_matrix = nn.Parameter(new_embedding_weight)

        new_linear_weight = torch.empty((new_size, d_model))
        init_std = 2 / (d_model + new_size)
        nn.init.trunc_normal_(new_linear_weight, mean=0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
        new_linear_weight[:origin_vocab_size, :] = self.linear.weights
        self.linear.weights = nn.Parameter(new_linear_weight)