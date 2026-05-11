import errno
import os
import os.path
import shutil
import time
import argparse
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tiny_lm import ModelConfig, get_dataset, load_train_config, init_model_from_checkpoint, \
    init_optimizer_from_checkpoint, cross_entropy_loss, gradient_clip, save_checkpoint, cosine_lr_scheduler, \
    DEFAULT_TRAIN_CONFIG

MIN_FREE_SPACE_BYTES = 5 * 1024 ** 3
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _param_estimate(config: ModelConfig) -> None:
    batch = config.batch_size
    vocab, d_model, layers, d_ff = config.vocab_size, config.d_model, config.num_layers, config.d_ff
    param_fp32_num = 2*vocab*d_model+d_model+layers*(2*d_model+4*d_model**2+3*d_model*d_ff)
    print(f"Need fp32 param num {param_fp32_num}, memory {4*param_fp32_num/1024**3} GB")
    seq = config.context_length
    train_fp32_num = 4 * param_fp32_num + 2 * batch * seq * vocab + (layers + 2) * batch * seq * d_model
    print(f"Train need {train_fp32_num}, memory {4*train_fp32_num/1024**3} GB")

def _save_checkpoint_with_cleanup(
    model: torch.nn.Module,
    optimizer: Optimizer,
    checkpoint_file_path: str,
) -> None:
    def cleanup_old_checkpoints(protected_file: str | None = None):
        free_space = shutil.disk_usage(checkpoint_root).free
        if free_space >= MIN_FREE_SPACE_BYTES:
            return [], free_space

        protected_file = os.path.abspath(protected_file) if protected_file is not None else None
        cpt_files: list[str] = []
        for root, _, files in os.walk(checkpoint_root):
            for file_name in files:
                if file_name.endswith(".cpt"):
                    cpt_files.append(os.path.join(root, file_name))

        cpt_files.sort(key=os.path.getmtime)
        removed: list[str] = []
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
            if free_space >= MIN_FREE_SPACE_BYTES:
                break

        return removed, free_space

    checkpoint_root = os.path.dirname(os.path.normpath(checkpoint_file_path))
    global_step = int(os.path.splitext(os.path.basename(checkpoint_file_path))[0])
    removed, free_space = cleanup_old_checkpoints()
    if removed:
        print(f"Cleanup before save: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")

    try:
        with open(checkpoint_file_path, "wb") as f:
            save_checkpoint(model, optimizer, global_step, f)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        print("Disk full while saving checkpoint, removing old checkpoints and retrying...")
        removed, free_space = cleanup_old_checkpoints()
        if removed:
            print(f"Cleanup on ENOSPC: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")
        if free_space < MIN_FREE_SPACE_BYTES:
            raise RuntimeError(
                f"Insufficient disk space after cleanup: {free_space / 1024 ** 3:.2f} GB free, "
                f"need at least {MIN_FREE_SPACE_BYTES / 1024 ** 3:.2f} GB."
            ) from e
        with open(checkpoint_file_path, "wb") as f:
            save_checkpoint(model, optimizer, global_step, f)

    removed, free_space = cleanup_old_checkpoints(protected_file=checkpoint_file_path)
    if removed:
        print(f"Cleanup after save: removed {len(removed)} old checkpoints, free space {free_space / 1024 ** 3:.2f} GB")
    if free_space < MIN_FREE_SPACE_BYTES:
        print(
            f"Warning: free disk space is {free_space / 1024 ** 3:.2f} GB, "
            f"below required {MIN_FREE_SPACE_BYTES / 1024 ** 3:.2f} GB."
        )

def train_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: Optimizer, checkpoint_dir: str,
                last_train_step: int, gradient_accumulate: int = 1) -> None:
    print("Data loader size: ", len(dataloader))
    n=len(dataloader)
    total_loss = 0
    opt_step = 0
    last_report_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        current_global_step = opt_step + last_train_step
        lr = cosine_lr_scheduler(current_global_step, 3e-4, 3e-5, 2000, 60000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x = x.cuda()
        y = y.cuda()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                  f"global step {current_global_step}, lr {lr}")
            last_report_time = time.time()
        if (i + 1) % 2000 == 0:
            checkpoint_file_path = os.path.join(checkpoint_dir, f"{current_global_step}.cpt")
            _save_checkpoint_with_cleanup(model, optimizer, checkpoint_file_path)
            print(f"({i + 1}/{n + 1}) Saved checkpoint at {checkpoint_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyLM")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    train_config, config, optimizer_config = load_train_config(args.config)
    _param_estimate(config)

    checkpoint = train_config.checkpoint if train_config.checkpoint else None
    model = init_model_from_checkpoint(config, checkpoint)
    optimizer, t = init_optimizer_from_checkpoint(optimizer_config, model, checkpoint)

    t = 0

    model.train()
    model.to("cuda")
    model = torch.compile(model)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    dataset = get_dataset(
        train_config.data_dir,
        seq_len=config.context_length,
        zh_token_dtype=train_config.zh_token_dtype,
        zh_fold=train_config.zh_fold,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for i in range(train_config.epochs):
        print(f"Start epoch {i}")
        checkpoint_dir = os.path.join(train_config.checkpoint_base_dir, f"72M_2_{i}")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        train_epoch(dataloader, model, optimizer, checkpoint_dir, t, train_config.gradient_accumulate)
