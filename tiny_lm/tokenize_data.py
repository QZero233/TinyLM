from multiprocessing.pool import Pool
import argparse

from tiny_lm import BPETokenizer
import numpy as np

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def init_worker():
    global tokenizer
    tokenizer = BPETokenizer.from_files(os.path.join(PROJECT_ROOT, "saved_gpt_tokenizer"))

def process_data(file: str, result_file: str, start: int, size: int, i: int, total: int):
    print(f"Start {i} / {total}")
    with open(file, "r") as f:
        f.seek(start)
        text = f.read(size)
        print(f"Worker {i} Read with length {len(text)}")
        part_token_ids = tokenizer.encode(text)
        part_token_ids = np.array(part_token_ids)
        print(f"Worker {i} {result_file} saving....")
        np.save(result_file, part_token_ids)
        print(f"Worker {i}, Saved {result_file} ({i}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a text file into shard .npy files")
    parser.add_argument("--data_file", type=str, default=os.path.join(PROJECT_ROOT, "data", "owt_valid.txt"))
    parser.add_argument("--result_prefix", type=str, default=os.path.join(PROJECT_ROOT, "data", "valid"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1_0000_0000)
    args = parser.parse_args()

    workers = args.workers
    batch_size = args.batch_size
    data_file = args.data_file
    result_prefix = args.result_prefix

    data_length = os.path.getsize(data_file)
    tasks = []
    total_split = data_length // batch_size + 1
    for i in range(total_split):
        start = i * batch_size
        end = min(start+batch_size, data_length)
        if start >= end:
            break

        size = end-start+1
        target_filename = f"{result_prefix}_{i}.npy"
        if os.path.exists(target_filename):
            print(f"File {target_filename} exists, skip it")
            continue
        tasks.append((data_file, target_filename, start, size, i, total_split))


    with Pool(workers, initializer=init_worker) as pool:
        pool.starmap(process_data, tasks)
