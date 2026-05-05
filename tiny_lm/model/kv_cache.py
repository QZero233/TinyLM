from typing import Optional

import torch


class KVCacheState:
    def __init__(self):
        # k,v: (...batch, seq, d_model)
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def append_kv(self, new_k: torch.Tensor, new_v: torch.Tensor):
        # new_k, new_v: (...batch, inc_len, d_model)
        assert len(new_k.shape) >= 2, "Expected shape: (...batch, inc_len, d_model)"
        if self.k is None:
            self.k = new_k.clone().detach()
            self.v = new_v.clone().detach()
            return

        assert len(new_k.shape) == len(self.k.shape)
        self.k = torch.cat([self.k, new_k], dim=-2)
        self.v = torch.cat([self.v, new_v], dim=-2)

class TransformerKVCache:
    def __init__(self, num_layers: int):
        self.kv_cache_states = [KVCacheState() for _ in range(num_layers)]

    # 返回已经缓存的序列长度
    def __len__(self) -> int:
        state = self.kv_cache_states[0]
        if state.k is None:
            return 0
        return state.k.shape[-2]

    def __getitem__(self, item):
        return self.kv_cache_states[item]