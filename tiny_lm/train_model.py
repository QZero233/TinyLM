import errno
import os
import os.path
import shutil
import time
import argparse
from typing import Optional, Tuple, List
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tiny_lm import Transformer, BPETokenizer, AdamW, load_checkpoint, TinyStoryDataset, cross_entropy_loss, \
    gradient_clip, save_checkpoint, cosine_lr_scheduler

MIN_FREE_SPACE_BYTES = 5 * 1024 ** 3
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LMConfig:
    def __init__(self, vocab_size: int = 1, context_length: int = 1,
                 num_layers: int = 1, d_model: int = 1, num_heads: int = 1, d_ff: int = 1, theta: float = 0,
                 lr: float = 0, beta_1: float = 0, beta_2: float = 0, weight_decay: float = 0, eps: float = 1e-6,
                 gradient_checkpoint: bool = True,
                 batch_size: int = 1):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.eps = eps

        self.gradient_checkpoint = gradient_checkpoint

        self.batch_size = batch_size

def init_model(config: LMConfig, checkpoint: Optional[str]) -> torch.nn.Module:
    model = Transformer(config.vocab_size, config.context_length, config.num_layers, config.d_model,
                        config.num_heads, config.d_ff, config.theta, gradient_checkpoint=config.gradient_checkpoint)

    if checkpoint is not None:
        load_checkpoint(checkpoint, model, None)

    return model

def init_optimizer(config: LMConfig, model: torch.nn.Module, checkpoint: Optional[str]) -> Tuple[torch.optim.Optimizer, int]:
    optimizer = AdamW(model.parameters(), config.lr, (config.beta_1, config.beta_2), config.weight_decay, config.eps)
    t = 0
    if checkpoint is not None:
        t = load_checkpoint(checkpoint, None, optimizer)
    return optimizer, t

def load_tokenizer(tokenizer_dir: str) -> BPETokenizer:
    return BPETokenizer.from_files(tokenizer_dir)

def _param_estimate(config: LMConfig):
    batch = config.batch_size
    vocab, d_model, layers, d_ff = config.vocab_size, config.d_model, config.num_layers, config.d_ff
    param_fp32_num = 2*vocab*d_model+d_model+layers*(2*d_model+4*d_model**2+3*d_model*d_ff)
    print(f"Need fp32 param num {param_fp32_num}, memory {4*param_fp32_num/1024**3} GB")
    seq = config.context_length
    train_fp32_num = 4 * param_fp32_num + 2 * batch * seq * vocab + (layers + 2) * batch * seq * d_model
    print(f"Train need {train_fp32_num}, memory {4*train_fp32_num/1024**3} GB")

def _cleanup_old_checkpoints(checkpoint_root: str, min_free_space_bytes: int, protected_file: Optional[str] = None) -> Tuple[List[str], int]:
    free_space = shutil.disk_usage(checkpoint_root).free
    if free_space >= min_free_space_bytes:
        return [], free_space

    protected_file = os.path.abspath(protected_file) if protected_file is not None else None
    cpt_files: List[str] = []
    for root, _, files in os.walk(checkpoint_root):
        for file_name in files:
            if file_name.endswith(".cpt"):
                cpt_files.append(os.path.join(root, file_name))

    cpt_files.sort(key=os.path.getmtime)
    removed: List[str] = []
    for file_path in cpt_files:
        abs_path = os.path.abspath(file_path)
        if protected_file is not None and abs_path == protected_file:
            continue
        try:
            os.remove(file_path)
            removed.append(file_path)
        except OSError as e:
            print(f"Skip removing {file_path}, err: {e}")
            continue

        free_space = shutil.disk_usage(checkpoint_root).free
        if free_space >= min_free_space_bytes:
            break

    return removed, free_space

def train_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: Optimizer, checkpoint_dir: str,
                last_train_step: int, gradient_accumulate: int = 1):
    print("Data loader size: ", len(dataloader))
    n=len(dataloader)
    total_loss = 0
    opt_step = 0
    last_report_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        current_global_step = opt_step + last_train_step
        lr = cosine_lr_scheduler(current_global_step, 3e-5, 3e-6, 2000, 60000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x = x.cuda()
        y = y.cuda()

        logits = model(x, None)
        loss = cross_entropy_loss(logits, y)
        loss = loss / gradient_accumulate
        loss.backward()

        if (i + 1) % gradient_accumulate == 0:
            gradient_clip(model.parameters(), m=1)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

        total_loss += loss.cpu().item()

        if i % (5 * gradient_accumulate) == 0:
            print(f"({i+1}/{n+1}) Total loss {total_loss * gradient_accumulate}, average loss {total_loss / (i + 1) * gradient_accumulate},"
                  f" last step loss {loss.cpu().item() * gradient_accumulate}, {time.time() - last_report_time} since last report,"
                  f"global step {current_global_step}")
            last_report_time = time.time()
        if (i + 1) % 2000 == 0:
            checkpoint_file = os.path.join(checkpoint_dir, f"{i}.cpt")
            checkpoint_root = os.path.dirname(os.path.normpath(checkpoint_dir))
            removed, free_space = _cleanup_old_checkpoints(checkpoint_root, MIN_FREE_SPACE_BYTES)
            if removed:
                print(f"Cleanup before save: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")

            try:
                with open(checkpoint_file, "wb") as f:
                    save_checkpoint(model, optimizer, current_global_step, f)
            except OSError as e:
                if e.errno != errno.ENOSPC:
                    raise
                print("Disk full while saving checkpoint, removing old checkpoints and retrying...")
                removed, free_space = _cleanup_old_checkpoints(checkpoint_root, MIN_FREE_SPACE_BYTES)
                if removed:
                    print(f"Cleanup on ENOSPC: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")
                if free_space < MIN_FREE_SPACE_BYTES:
                    raise RuntimeError(
                        f"Insufficient disk space after cleanup: {free_space / 1024 ** 3:.2f} GB free, "
                        f"need at least {MIN_FREE_SPACE_BYTES / 1024 ** 3:.2f} GB."
                    ) from e
                with open(checkpoint_file, "wb") as f:
                    save_checkpoint(model, optimizer, current_global_step, f)

            removed, free_space = _cleanup_old_checkpoints(
                checkpoint_root, MIN_FREE_SPACE_BYTES, protected_file=checkpoint_file
            )
            if removed:
                print(f"Cleanup after save: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")
            if free_space < MIN_FREE_SPACE_BYTES:
                print(
                    f"Warning: free disk space is {free_space / 1024 ** 3:.2f} GB, "
                    f"below required {MIN_FREE_SPACE_BYTES / 1024 ** 3:.2f} GB."
                )
            print(f"({i+1}/{n+1}) Saved checkpoint at {checkpoint_file}")
        # if i % 10 == 0:
        #     _export_param_grad(model)

def _get_model_config(vocab_size: int) -> LMConfig:
    d_model = 1024
    num_layers = 16
    d_ff = 2752
    num_heads = 16
    context_length = 1024
    batch_size = 16

    return LMConfig(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers, d_model=d_model,
                      num_heads=num_heads, d_ff=d_ff, theta=10000,
                      lr=1e-3, beta_1=0.9, beta_2=0.95, weight_decay=0.1, gradient_checkpoint=True, batch_size=batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyLM")
    parser.add_argument("--tokenizer_dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint", "saved_gpt_tokenizer"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint", "new_epoch_5B_0", "9999.cpt"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--checkpoint_base_dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulate", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    tokenizer = load_tokenizer(args.tokenizer_dir)

    config = _get_model_config(len(tokenizer.vocab))
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
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    _param_estimate(config)

    # 当前训练数据量：4.5B
    checkpoint = args.checkpoint if args.checkpoint else None
    model = init_model(config, checkpoint)
    optimizer, t = init_optimizer(config, model, None)

    model.train()
    model.to("cuda")
    model = torch.compile(model)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    dataset = TinyStoryDataset(args.data_dir, seq_len=config.context_length, train=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for i in range(args.epochs):
        print(f"Start epoch {i}")
        checkpoint_dir = os.path.join(args.checkpoint_base_dir, f"new_epoch_5B_new_{i}")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        train_epoch(dataloader, model, optimizer, checkpoint_dir, t, args.gradient_accumulate)
