import bisect
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class _ShardInfo:
    path: str
    token_count: int
    sample_count: int


class TinyLMZhDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16", full_random: bool = True):
        self.shards: List[_ShardInfo] = []
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.full_random = full_random
        self._memmap_cache: dict[int, np.memmap] = {}
        self._cumulative_sample_ends: List[int] = []

        total_samples = 0
        for root, _, files in os.walk(data_dir):
            for file in sorted(files):
                if file.endswith(".bin"):
                    path = os.path.join(root, file)
                    token_count = os.path.getsize(path) // self.dtype.itemsize
                    sample_count = token_count - seq_len
                    if sample_count > 0:
                        self.shards.append(
                            _ShardInfo(path=path, token_count=token_count, sample_count=sample_count)
                        )
                        total_samples += sample_count
                        self._cumulative_sample_ends.append(total_samples)

        if not self.shards:
            raise ValueError(f"No usable .bin token files found in {data_dir}")

        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples

    def _map_index(self, idx: int) -> int:
        if not self.full_random or self.total_samples <= 1:
            return idx
        return random.randrange(self.total_samples)

    def _get_memmap(self, shard_idx: int) -> np.memmap:
        memmap = self._memmap_cache.get(shard_idx)
        if memmap is None:
            shard = self.shards[shard_idx]
            # Open shards lazily so dataset init does not map every file up front.
            memmap = np.memmap(shard.path, dtype=self.dtype, mode="r", shape=(shard.token_count,))
            self._memmap_cache[shard_idx] = memmap
        return memmap

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")

        global_idx = self._map_index(idx)
        shard_idx = bisect.bisect_right(self._cumulative_sample_ends, global_idx)
        prev_end = 0 if shard_idx == 0 else self._cumulative_sample_ends[shard_idx - 1]
        shard_offset = global_idx - prev_end

        target_batch = self._get_memmap(shard_idx)
        shard = self.shards[shard_idx]
        assert shard_offset < shard.sample_count

        x = target_batch[shard_offset:shard_offset + self.seq_len].astype(np.int64)
        y = target_batch[shard_offset + 1:shard_offset + self.seq_len + 1].astype(np.int64)

        return torch.LongTensor(x), torch.LongTensor(y)
