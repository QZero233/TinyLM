from io import TextIOWrapper
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

import os

from .. import BPETokenizer

from multiprocessing import Pool

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.LongTensor, torch.LongTensor]:
    size = len(x)
    legal_start_index = np.arange(size - context_length)
    start_index = np.random.choice(legal_start_index, size=batch_size, replace=False)
    chosen_x = [x[start: start+context_length+1] for start in start_index]
    chosen_x = np.stack(chosen_x, axis=0)
    data = chosen_x[:, :-1]
    label = chosen_x[:, 1:]
    return torch.LongTensor(data, device=device), torch.LongTensor(label, device=device)

class TinyStoryDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, train: bool = True):
        self.raw_data_list: List[np.ndarray] = []
        prefix = "train_" if train else "valid_"
        for file in sorted(os.listdir(data_dir)):
            if file.startswith(prefix) and file.endswith(".npy"):
                self.raw_data_list.append(np.load(os.path.join(data_dir, file)))

        self.seq_len = seq_len
        # 确保不会出现跨越块boundary的情况
        self.raw_data_size: List[int] = [len(data) - seq_len for data in self.raw_data_list]

    @staticmethod
    def _batch_tokenize(f: TextIOWrapper, result_file_prefix: str, tokenizer: BPETokenizer, batch_size: int, total_size: int):
        i = 0
        split_num = total_size // batch_size + 1
        while f.tell() < total_size:
            print(f"Working ({i + 1}/{split_num + 1})")

            part_text = f.read(batch_size)

            print(f"Read with tell {f.tell()}")

            part_token_ids = tokenizer.encode(part_text)
            part_token_ids = np.array(part_token_ids, dtype=np.int32)
            file_name = f"{result_file_prefix}_{i}.npy"
            np.save(file_name, part_token_ids)
            print(f"Saved {file_name} ({i+1}/{split_num+1})")
            i += 1

    @staticmethod
    def tokenize_origin_data(data_dir: str, tokenizer: BPETokenizer):
        train_data_txt = os.path.join(data_dir, "owt_train.txt")
        valid_data_txt = os.path.join(data_dir, "owt_valid.txt")

        with open(train_data_txt, "r") as train_f, open(valid_data_txt, "r") as valid_f:
            batch_size = 3_0000_0000
            TinyStoryDataset._batch_tokenize(train_f, os.path.join(data_dir, "train"), tokenizer, batch_size, os.path.getsize(train_data_txt))
            TinyStoryDataset._batch_tokenize(valid_f, os.path.join(data_dir, "valid"), tokenizer, batch_size, os.path.getsize(valid_data_txt))

    def __len__(self):
        return sum(self.raw_data_size) - self.seq_len

    def __getitem__(self, idx):
        # 首先定位到这个idx在第几个slice里面
        remain_size = idx
        target_idx = -1
        for idx, size in enumerate(self.raw_data_size):
            if remain_size < size:
                target_idx = idx
                break

            remain_size -= size

        assert target_idx >= 0
        target_batch = self.raw_data_list[target_idx]
        assert remain_size + self.seq_len < len(target_batch)

        x = target_batch[remain_size:remain_size+self.seq_len]
        y = target_batch[remain_size+1:remain_size+self.seq_len+1]

        return torch.LongTensor(x), torch.LongTensor(y)



