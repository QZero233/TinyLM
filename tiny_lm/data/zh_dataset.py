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
    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16",
                 full_random: bool = True, train: bool = True):
        self.shards: List[_ShardInfo] = []
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.full_random = full_random
        self.train = train
        self._memmap_cache: dict[int, np.memmap] = {}
        self._cumulative_sample_ends: List[int] = []
        self._group_size = self.seq_len + 1
        self._index_map: List[int] = []
        self._epoch_offset = 0
        self._valid_max_samples = 500
        self._valid_selected_shard_idx: int | None = None

        shard_candidates: List[_ShardInfo] = []
        for root, _, files in os.walk(data_dir):
            for file in sorted(files):
                if file.endswith(".bin"):
                    path = os.path.join(root, file)
                    token_count = os.path.getsize(path) // self.dtype.itemsize
                    if self.full_random:
                        # Random mode: sample one non-overlapping group with size (seq_len + 1).
                        sample_count = token_count // self._group_size
                    else:
                        # Deterministic mode keeps the original sliding-window behavior.
                        sample_count = token_count - seq_len
                    if sample_count > 0:
                        shard_candidates.append(_ShardInfo(path=path, token_count=token_count, sample_count=sample_count))

        if not shard_candidates:
            raise ValueError(f"No usable .bin token files found in {data_dir}")

        if self.train:
            self.shards = shard_candidates
            for shard in self.shards:
                self._cumulative_sample_ends.append(
                    (self._cumulative_sample_ends[-1] if self._cumulative_sample_ends else 0) + shard.sample_count
                )
            total_samples = self._cumulative_sample_ends[-1]
        else:
            selected_shard = random.choice(shard_candidates)
            self.shards = [selected_shard]
            self._valid_selected_shard_idx = shard_candidates.index(selected_shard)
            total_samples = min(self._valid_max_samples, selected_shard.sample_count)
            self.shards[0] = _ShardInfo(
                path=selected_shard.path,
                token_count=selected_shard.token_count,
                sample_count=total_samples,
            )
            self._cumulative_sample_ends = [total_samples]

        self.total_samples = total_samples
        if self.full_random:
            if self.train:
                self.reset()
            else:
                self._index_map = list(range(self.total_samples))
                random.shuffle(self._index_map)
                self._epoch_offset = random.randint(0, self.seq_len)

    def __len__(self) -> int:
        return self.total_samples

    def reset(self):
        if not self.full_random or not self.train:
            return
        self._index_map = list(range(self.total_samples))
        random.shuffle(self._index_map)
        self._epoch_offset = random.randint(0, self.seq_len)

    def _map_index(self, idx: int) -> int:
        if not self.full_random:
            return idx
        return self._index_map[idx]

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

        if self.full_random:
            start = shard_offset * self._group_size + self._epoch_offset
            max_start = shard.token_count - self._group_size
            if start > max_start:
                start %= (max_start + 1)
        else:
            start = shard_offset

        x = target_batch[start:start + self.seq_len].astype(np.int64)
        y = target_batch[start + 1:start + self.seq_len + 1].astype(np.int64)

        return torch.LongTensor(x), torch.LongTensor(y)
