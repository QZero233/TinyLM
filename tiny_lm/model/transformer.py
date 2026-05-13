import math
from typing import Any, Optional, List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .norm import RMSNorm
from .attention import MultiHeadAttention, TorchMultiHeadAttention
from .common import SwiGLU, TorchSwiGLU
from .embedding import Embedding
from .kv_cache import KVCacheState, TransformerKVCache
from .lora import LoRALinear, LoRAConfig
from .utils import replace_submodule

def _build_attention(d_model: int, num_heads: int, max_seq_len: int, theta: float,
                     pytorch_impl: bool) -> nn.Module:
    if pytorch_impl:
        return TorchMultiHeadAttention(d_model, num_heads, max_seq_len, theta)
    return MultiHeadAttention(d_model, num_heads, max_seq_len, theta)


def _build_ffn(d_model: int, d_ff: int, pytorch_impl: bool) -> nn.Module:
    if pytorch_impl:
        return TorchSwiGLU(d_model, d_ff)
    return SwiGLU(d_model, d_ff)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024,
                 theta: float = 0.5, pytorch_impl: bool = False, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.multi_head_attn = _build_attention(d_model, num_heads, max_seq_len, theta, pytorch_impl)
        self.ff = _build_ffn(d_model, d_ff, pytorch_impl)

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
                 theta: float = 0.5, gradient_checkpoint: bool = True, pytorch_impl: bool = False,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.gradient_checkpoint = gradient_checkpoint
        self.pytorch_impl = pytorch_impl

        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta, pytorch_impl) for _ in range(num_layers)]
        )
        self.embedding = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, kv_cache: Optional[TransformerKVCache]=None) -> torch.Tensor:
        # Input: (batch_size, seq_len)
        # Output: (batch_size, seq_len, vocab_size)
        if self.pytorch_impl and kv_cache is not None:
            raise NotImplementedError("pytorch_impl=True does not support kv_cache")

        if kv_cache is not None:
            cached_len = len(kv_cache)
            x = x[..., cached_len:]
        x = self.embedding(x)

        for i, layer in enumerate(self.transformer_layers):
            kv_cache_state = kv_cache[i] if kv_cache is not None else None
            if self.training and self.gradient_checkpoint and i % 4 == 0:
                x = checkpoint(layer, x, kv_cache_state=kv_cache_state, use_reentrant=False)
            else:
                x = layer(x, kv_cache_state=kv_cache_state)
        x = self.norm(x)
        # 使用Shared Weight，和Embedding共享权重
        x = x @ self.embedding.embedding_matrix.T
        return x

    def resize_embedding(self, new_size: int):
        origin_vocab_size, d_model = self.embedding.embedding_matrix.shape
        if new_size == origin_vocab_size:
            return

        old_embedding_weight = self.embedding.embedding_matrix
        new_embedding_weight = torch.empty(
            (new_size, d_model),
            device=old_embedding_weight.device,
            dtype=old_embedding_weight.dtype,
        )
        keep_size = min(origin_vocab_size, new_size)
        new_embedding_weight[:keep_size, :] = old_embedding_weight[:keep_size, :]
        if new_size > origin_vocab_size:
            nn.init.trunc_normal_(new_embedding_weight[origin_vocab_size:, :], mean=0, std=1, a=-3, b=3)
        self.embedding.embedding_matrix = nn.Parameter(new_embedding_weight)
        self.embedding.num_embeddings = new_size

    def _freeze_params(self):
        for name, param in self.named_parameters():
            if name.startswith("embedding."):
                continue
            param.requires_grad = False

    def adapt_lora(self, lora_configs: List[LoRAConfig]):
        self._freeze_params()
        # 替换对应的Linear层
        for config in lora_configs:
            old_linear = self.get_submodule(config.module_name)
            lora_linear = LoRALinear(old_linear, config.b, config.a)
            lora_linear.b.requires_grad = True
            lora_linear.a.requires_grad = True
            replace_submodule(self, config.module_name, lora_linear)
