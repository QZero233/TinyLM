from typing import Any, Optional, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim

        self.embedding_matrix = nn.Parameter(torch.zeros((num_embeddings, embedding_dim)))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0, std=0.02)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # token_ids: (batch_size, sequence_length)
        # result: (batch_size, sequence_length, embedding_dim)
        vocab_size = self.embedding_matrix.shape[0]
        token_ids_one_hot = F.one_hot(token_ids, vocab_size).to(dtype=self.embedding_matrix.dtype)
        return token_ids_one_hot @ self.embedding_matrix

class RoPE(nn.Module):
    _rope_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert d_k % 2 == 0, "d_k must be even number"
        self.max_seq_len = max_seq_len

        if max_seq_len in RoPE._rope_cache:
            sin_seq, cos_seq = RoPE._rope_cache[max_seq_len]
        else:
            k = d_k // 2
            seq_range = torch.arange(max_seq_len).reshape((max_seq_len, 1)).broadcast_to((max_seq_len, k))
            theta_exp = (((torch.arange(k) + 1) * 2 - 2) / d_k).reshape((1, k)).broadcast_to((max_seq_len, k))
            theta_tensor = torch.tensor([theta]).reshape((1, 1)).broadcast_to((max_seq_len, k))
            theta_exp = torch.pow(theta_tensor, theta_exp)
            theta_final = seq_range / theta_exp

            sin_seq = torch.sin(theta_final).to(device=device)
            cos_seq = torch.cos(theta_final).to(device=device)

            RoPE._rope_cache[max_seq_len] = (sin_seq, cos_seq)

        self.register_buffer("sin_buffer", sin_seq)
        self.register_buffer("cos_buffer", cos_seq)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len

        if token_positions is None:
            token_positions = torch.arange(seq_len).broadcast_to((*x.shape[:-2], -1))
        # (...batch, seq_len, k)
        sin_slice = self.sin_buffer[token_positions, :]
        cos_slice = self.cos_buffer[token_positions, :]
        # 先处理奇数位的，构造cos sin cos sin的组合
        odd_slice = torch.empty((*sin_slice.shape[:-1], 2 * sin_slice.shape[-1]), device=x.device, dtype=x.dtype)
        odd_slice[..., 0::2] = cos_slice
        odd_slice[..., 1::2] = -sin_slice
        odd_prod = x * odd_slice
        odd_result = odd_prod.reshape((*odd_prod.shape[:-1], -1, 2)).sum(dim=-1)
        # 再处理偶数位的，构造-sin cos -sin cos
        even_slice = torch.empty((*sin_slice.shape[:-1], 2 * sin_slice.shape[-1]), device=x.device, dtype=x.dtype)
        even_slice[..., 0::2] = sin_slice
        even_slice[..., 1::2] = cos_slice
        even_prod = x * even_slice
        even_result = even_prod.reshape((*even_prod.shape[:-1], -1, 2)).sum(dim=-1)
        # 最后拼接起来
        result = torch.empty_like(x)
        result[..., 0::2] = odd_result
        result[..., 1::2] = even_result

        return result


