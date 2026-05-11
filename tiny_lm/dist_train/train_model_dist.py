import os
import os.path
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tiny_lm import TinyStoryDataset, get_default_model_config, get_default_optimizer_config, \
    init_model_from_checkpoint, init_optimizer
from .train_model import load_tokenizer, train_epoch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dist_setup():
    dist.init_process_group("nccl")
    local_rank = int (os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed TinyLM training")
    parser.add_argument("--tokenizer_dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_gpt_tokenizer"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint", "new_epoch_0_0", "9999.cpt"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--checkpoint_base_dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulate", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    args = parser.parse_args()

    local_rank = dist_setup()
    torch.set_float32_matmul_precision('high')

    tokenizer = load_tokenizer(args.tokenizer_dir)
    batch_size = args.batch_size
    config = get_default_model_config(len(tokenizer.vocab))
    optimizer_config = get_default_optimizer_config()
    if args.context_length is not None:
        config.context_length = args.context_length
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff

    model = init_model_from_checkpoint(config, args.checkpoint if args.checkpoint else None)
    t = 0
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = init_optimizer(optimizer_config, model)

    model.train()
    model = torch.compile(model)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    dataset = TinyStoryDataset(args.data_dir, seq_len=config.context_length, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Rank {local_rank} ready!")
    for i in range(args.epochs):
        print(f"Start epoch {i}")
        checkpoint_dir = os.path.join(args.checkpoint_base_dir, f"new_epoch_1_{i}")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        train_epoch(dataloader, model, optimizer, checkpoint_dir, t, args.gradient_accumulate)
