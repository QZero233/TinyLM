import argparse
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import wandb

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from tiny_lm import (
    DEFAULT_TRAIN_CONFIG,
    cross_entropy_loss,
    cosine_lr_scheduler,
    get_dataset,
    gradient_clip,
    init_model_from_checkpoint,
    init_optimizer_from_checkpoint,
    load_train_config,
)
from tiny_lm.train_model import _param_estimate, _save_checkpoint_with_cleanup

DEFAULT_SEED = 20260527


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _dist_setup() -> tuple[torch.device, int, int, int, str]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if _is_npu_available():
        torch.npu.set_device(local_rank)
        backend = "hccl"
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size, backend


def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _autocast_context(device: torch.device):
    if device.type in ("cuda", "npu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _ddp_model(model: torch.nn.Module, device: torch.device, local_rank: int, world_size: int) -> torch.nn.Module:
    if world_size == 1:
        return model
    if device.type in ("cuda", "npu"):
        return DDP(model, device_ids=[local_rank], output_device=local_rank)
    return DDP(model)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _distributed_average(value: float, count: int, device: torch.device) -> float:
    totals = torch.tensor([value, float(count)], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_count = int(totals[1].item())
    if total_count == 0:
        return float("nan")
    return float(totals[0].item() / total_count)


def _eval_validation_loss(
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for x, y in valid_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(), _autocast_context(device):
            logits = model(x, None)
            loss = cross_entropy_loss(logits, y)
        batch_count = x.shape[0]
        total_loss += loss.item() * batch_count
        total_count += batch_count

    avg_loss = _distributed_average(total_loss, total_count, device)
    model.train()
    return avg_loss


def train_epoch_ddp(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optimizer,
    checkpoint_dir: str,
    last_train_step: int,
    device: torch.device,
    rank: int,
    gradient_accumulate: int = 1,
    fix_lr: float | None = None,
    wandb_run = None,
    checkpoint_save_accum_steps: int = 200,
    valid_dataloader: DataLoader | None = None,
    valid_steps: int = 200,
) -> int:
    if _is_rank0(rank):
        print("Data loader size per rank: ", len(dataloader))

    n = len(dataloader)
    total_loss = 0.0
    accum_loss = 0.0
    accum_count = 0
    opt_step = 0
    last_report_time = time.time()

    for i, (input_ids, labels) in enumerate(dataloader):
        current_global_step = opt_step + last_train_step
        lr = fix_lr if fix_lr is not None else cosine_lr_scheduler(current_global_step, 3e-4, 3e-5, 2000, 60000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        should_sync = (i + 1) % gradient_accumulate == 0
        sync_context = nullcontext()
        if isinstance(model, DDP) and not should_sync:
            sync_context = model.no_sync()

        with sync_context:
            with _autocast_context(device):
                logits = model(input_ids, None)
                ce_loss = cross_entropy_loss(logits, labels)
            loss = ce_loss / gradient_accumulate
            loss.backward()

        accum_loss += ce_loss.item()
        accum_count += 1
        total_loss += loss.detach().cpu().item()

        if _is_rank0(rank) and (i + 1) % (5 * gradient_accumulate) == 0:
            accum_avg = accum_loss / accum_count if accum_count > 0 else 0.0
            print(
                f"({i + 1}/{n + 1}) Total loss {total_loss * gradient_accumulate}, "
                f"average loss {total_loss / (i + 1) * gradient_accumulate}, "
                f"accumulation avg loss {accum_avg}, {time.time() - last_report_time} since last report, "
                f"global step {current_global_step}, lr {lr}"
            )
            last_report_time = time.time()

        if should_sync:
            gradient_clip(model.parameters(), m=1)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1
            current_step = opt_step + last_train_step
            accum_avg = accum_loss / accum_count if accum_count > 0 else 0.0
            if wandb_run is not None:
                wandb_run.log({
                    "train/lr_accum_step": lr,
                    "train/loss_accum_step": accum_avg,
                }, step=current_step)
            accum_loss = 0.0
            accum_count = 0

            if opt_step > 0 and opt_step % checkpoint_save_accum_steps == 0:
                checkpoint_file_path = os.path.join(checkpoint_dir, f"{current_step}.cpt")
                if _is_rank0(rank):
                    _save_checkpoint_with_cleanup(_unwrap_model(model), optimizer, checkpoint_file_path)
                    print(f"({i + 1}/{n + 1}) Saved checkpoint at {checkpoint_file_path} (accum_step={opt_step})")
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if valid_dataloader is not None and opt_step > 0 and opt_step % valid_steps == 0:
                valid_loss = _eval_validation_loss(model, valid_dataloader, device)
                if wandb_run is not None:
                    wandb_run.log({"valid/loss_accum_step": valid_loss}, step=current_step)
                if _is_rank0(rank):
                    print(
                        f"({i + 1}/{n + 1}) Validation average loss {valid_loss} "
                        f"(global_step={current_step}, accum_step={opt_step})"
                    )

    return last_train_step + opt_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Train TinyLM")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device, rank, local_rank, world_size, backend = _dist_setup()
    _seed_everything(args.seed)

    train_config, config, optimizer_config = load_train_config(args.config)
    if _is_rank0(rank):
        _param_estimate(config)
        print(f"Distributed backend: {backend}, world_size: {world_size}")
        print(f"Rank {rank} using device: {device}, local_rank: {local_rank}")

    checkpoint = train_config.checkpoint if train_config.checkpoint else None
    model = init_model_from_checkpoint(config, checkpoint).to(device)
    optimizer_checkpoint = None if optimizer_config.reset else checkpoint
    optimizer, step = init_optimizer_from_checkpoint(optimizer_config, model, optimizer_checkpoint)
    _move_optimizer_state(optimizer, device)
    model.train()
    model = _ddp_model(model, device, local_rank, world_size)

    experiment_dir = os.path.join(train_config.checkpoint_base_dir, train_config.project_name)
    if _is_rank0(rank):
        os.makedirs(experiment_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    wandb_run = None
    if _is_rank0(rank):
        wandb_run = wandb.init(
            project=train_config.project_name,
            name=f"{train_config.project_name}_ddp_pretrain",
            config={
                "training": asdict(train_config),
                "model": asdict(config),
                "optimizer": asdict(optimizer_config),
                "backend": backend,
                "world_size": world_size,
                "device": str(device),
            },
        )

    train_dataset = get_dataset(
        train_config.data_dir,
        seq_len=config.context_length,
        zh_token_dtype=train_config.zh_token_dtype,
        full_random=True,
        train=True,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)

    valid_dataset = get_dataset(
        train_config.data_dir,
        seq_len=config.context_length,
        zh_token_dtype=train_config.zh_token_dtype,
        full_random=True,
        train=False,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler)

    try:
        for epoch in range(train_config.epochs):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            if _is_rank0(rank):
                print(f"Start epoch {epoch}")
            step = train_epoch_ddp(
                train_dataloader,
                model,
                optimizer,
                experiment_dir,
                step,
                device,
                rank,
                train_config.gradient_accumulate,
                train_config.fix_lr,
                wandb_run,
                train_config.checkpoint_save_accum_steps,
                valid_dataloader,
                train_config.valid_steps,
            )
            if hasattr(train_dataset, "reset"):
                _seed_everything(args.seed + epoch + 1)
                train_dataset.reset()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _cleanup_dist()


if __name__ == "__main__":
    main()
