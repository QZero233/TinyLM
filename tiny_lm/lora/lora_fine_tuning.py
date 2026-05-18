import argparse
import math
import os
import re
import time
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tiny_lm import (
    DEFAULT_TRAIN_CONFIG,
    LoRAConfig,
    cross_entropy_loss,
    gradient_clip,
    init_model_from_checkpoint,
    init_model,
    init_optimizer,
    load_checkpoint,
    load_lora_train_config,
    load_lora_trainable_checkpoint,
    save_checkpoint,
    save_lora_trainable_checkpoint,
)
from tiny_lm.model.transformer import Transformer
from tiny_lm.lora.sft_dataset import SFTJsonlDataset
from tiny_lm.load_config import get_tokenizer


PAIR_FILE_PATTERN = re.compile(r"^epoch_(\d+)_step_(\d+)_(model|lora)\.cpt$")


def _encode_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text)
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
        try:
            return tokenizer.tokenizer.encode(text, bos=False, eos=False)
        except TypeError:
            return tokenizer.tokenizer.encode(text)
    if hasattr(tokenizer, "sp_model") and hasattr(tokenizer.sp_model, "encode"):
        return tokenizer.sp_model.encode(text)
    encoded = tokenizer(text, add_special_tokens=False)
    return encoded["input_ids"]


def _build_lora_configs(config, r: int, device: str) -> list[LoRAConfig]:
    lora_configs: list[LoRAConfig] = []
    for i in range(config.num_layers):
        for component in ["linear_q", "linear_k", "linear_v", "linear_out"]:
            b = torch.zeros(config.d_model, r, device=device)
            a = torch.zeros(r, config.d_model, device=device)

            init_std = 2 / (r + config.d_model)
            nn.init.trunc_normal_(b, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
            nn.init.trunc_normal_(a, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))

            lora_configs.append(LoRAConfig(f"transformer_layers.{i}.multi_head_attn.{component}", b, a))
    return lora_configs


def _cosine_lr_scheduler_with_floor(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return lr_min + (lr_max - lr_min) * t / t_w
    if t <= t_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)
    return lr_min


def _get_tokenizer_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size"):
        return int(tokenizer.vocab_size)
    return int(len(tokenizer))


def _ensure_model_vocab_compatible(model: Transformer, tokenizer_vocab_size: int, auto_resize_embedding: bool) -> None:
    model_vocab_size = int(model.embedding.embedding_matrix.shape[0])
    if tokenizer_vocab_size == model_vocab_size:
        return
    if not auto_resize_embedding:
        raise ValueError(
            f"tokenizer vocab size ({tokenizer_vocab_size}) does not match model vocab size ({model_vocab_size})"
        )
    model.resize_embedding(tokenizer_vocab_size)
    print(f"[Vocab] resized embedding from {model_vocab_size} to {tokenizer_vocab_size}")


def eval_valid_loss(model: torch.nn.Module, valid_dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_num = 0
    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.cuda()
            y = y.cuda()
            logits = model(x, None)
            ce_loss = cross_entropy_loss(logits, y, ignore_label=-100)
            total_loss += ce_loss.item()
            total_num += 1
    return total_loss / max(total_num, 1)


def _load_step_from_lora_checkpoint(lora_checkpoint: str) -> int:
    if not lora_checkpoint:
        return 0
    try:
        return load_lora_trainable_checkpoint(lora_checkpoint, model=None)
    except Exception as e:
        print(f"Warning: failed to load step from lora checkpoint {lora_checkpoint}: {e}")
        return 0


def _load_step_from_full_checkpoint(checkpoint: str) -> int:
    if not checkpoint:
        return 0
    try:
        state = torch.load(checkpoint, map_location="cpu")
        return int(state.get("t", 0))
    except Exception as e:
        print(f"Warning: failed to load step from checkpoint {checkpoint}: {e}")
        return 0


def cleanup_old_lora_checkpoints(save_root: str, max_keep: int = 5) -> None:
    lora_files: list[tuple[int, int, str]] = []
    for file_name in os.listdir(save_root):
        match = PAIR_FILE_PATTERN.match(file_name)
        if not match:
            continue
        if match.group(3) != "lora":
            continue
        epoch = int(match.group(1))
        step = int(match.group(2))
        lora_files.append((step, epoch, os.path.join(save_root, file_name)))

    if len(lora_files) <= max_keep:
        return

    lora_files.sort(reverse=True)  # newer first by (step, epoch)
    removed_count = 0
    for _, _, file_path in lora_files[max_keep:]:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_count += 1
    if removed_count > 0:
        print(f"[Checkpoint] Removed {removed_count} old lora checkpoints, keep latest {max_keep}")


def _print_trainable_update_ratios(model: torch.nn.Module, param_before: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in param_before:
            continue
        param_after = param.detach()
        real_update_ratio = (param_after - param_before[name]).norm().item() / (param_after.norm().item() + 1e-12)
        print(f"[OptimizerUpdateRatio] {name}: {real_update_ratio}")


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    valid_dataloader: DataLoader,
    valid_steps: int,
    checkpoint_save_steps: int,
    fix_lr: float | None,
    lr_scheduler_warmup_steps: int,
    lr_scheduler_total_steps: int,
    lr_scheduler_min_lr: float,
    lr_scheduler_max_lr: float,
    print_optimizer_update_ratio: bool,
    save_root: str,
    epoch: int,
    full_finetune: bool,
    global_step: int,
    writer: SummaryWriter,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    steps = 0
    last_report_time = time.time()

    for i, (x, y) in enumerate(dataloader):
        x = x.cuda()
        y = y.cuda()

        if fix_lr is not None:
            current_lr = fix_lr
        else:
            current_lr = _cosine_lr_scheduler_with_floor(
                global_step, lr_scheduler_max_lr, lr_scheduler_min_lr, lr_scheduler_warmup_steps, lr_scheduler_total_steps
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, None)
            ce_loss = cross_entropy_loss(logits, y, ignore_label=-100)
        ce_loss.backward()
        param_before = None
        if print_optimizer_update_ratio and (global_step + 1) % 10 == 0:
            param_before = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
        gradient_clip(model.parameters(), m=1)
        optimizer.step()
        if param_before is not None:
            _print_trainable_update_ratios(model, param_before)
        optimizer.zero_grad()

        current_global_step = global_step + 1
        writer.add_scalar("lora/train_loss_step", ce_loss.item(), current_global_step)
        writer.add_scalar("lora/lr_step", current_lr, current_global_step)
        global_step = current_global_step
        if valid_steps > 0 and global_step % valid_steps == 0:
            valid_loss = eval_valid_loss(model, valid_dataloader)
            writer.add_scalar("lora/valid_loss_step", valid_loss, global_step)
            print(f"[Valid] global_step {global_step}, ce loss {valid_loss}")
            model.train()

        if checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
            lora_ckpt = os.path.join(save_root, f"epoch_{epoch}_step_{global_step}_lora.cpt")
            if full_finetune:
                save_checkpoint(model, None, global_step, lora_ckpt)
            else:
                save_lora_trainable_checkpoint(model, global_step, lora_ckpt)
            print(f"[Checkpoint] Saved lora checkpoint: {lora_ckpt}")
            cleanup_old_lora_checkpoints(save_root, max_keep=20)

        total_loss += ce_loss.item()
        steps += 1
        if (i + 1) % 20 == 0:
            now = time.time()
            elapsed_sec = now - last_report_time
            last_report_time = now
            print(
                f"Step {i + 1}/{len(dataloader)}, global_step {global_step}, "
                f"avg loss {total_loss / steps}, last {ce_loss.item()}, "
                f"{elapsed_sec:.3f}s since last report"
            )

    return total_loss / max(steps, 1), global_step


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for TinyLM")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args()

    train_config, config, optimizer_config, lora_train_config = load_lora_train_config(args.config)

    tokenizer = get_tokenizer(train_config.tokenizer_path)
    tokenizer_vocab_size = _get_tokenizer_vocab_size(tokenizer)
    full_finetune = bool(lora_train_config.full_finetune)
    lora_configs = [] if full_finetune else _build_lora_configs(config, lora_train_config.r, "cuda")
    base_model_checkpoint = lora_train_config.base_model_checkpoint if lora_train_config.base_model_checkpoint else None
    if lora_train_config.lora_checkpoint:
        if full_finetune:
            model = init_model_from_checkpoint(config, lora_train_config.lora_checkpoint)
            _ensure_model_vocab_compatible(model, tokenizer_vocab_size, lora_train_config.auto_resize_embedding)
            model = model.to("cuda")
        else:
            model = init_model(config)
            _ensure_model_vocab_compatible(model, tokenizer_vocab_size, lora_train_config.auto_resize_embedding)
            model = model.to("cuda")
            if base_model_checkpoint:
                load_checkpoint(base_model_checkpoint, model, None)
            model.adapt_lora(lora_configs)
            load_lora_trainable_checkpoint(lora_train_config.lora_checkpoint, model)
    else:
        model = init_model(config)
        _ensure_model_vocab_compatible(model, tokenizer_vocab_size, lora_train_config.auto_resize_embedding)
        model = model.to("cuda")
        if base_model_checkpoint:
            load_checkpoint(base_model_checkpoint, model, None)
        if not full_finetune:
            model.adapt_lora(lora_configs)

    optimizer = init_optimizer(optimizer_config, model)
    if optimizer_config.reset:
        global_step = 0
    elif full_finetune:
        global_step = _load_step_from_full_checkpoint(lora_train_config.lora_checkpoint)
    else:
        global_step = _load_step_from_lora_checkpoint(lora_train_config.lora_checkpoint)
    if global_step > lora_train_config.lr_scheduler_total_steps:
        print(f"[LR] global_step {global_step} exceeds total schedule steps {lora_train_config.lr_scheduler_total_steps}, lr will stay at {lora_train_config.lr_scheduler_min_lr}")

    dataset = SFTJsonlDataset(
        data_path=lora_train_config.data_dir,
        tokenizer=tokenizer,
        context_length=config.context_length,
        split="train",
        mask_question=lora_train_config.mask_question,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataset = SFTJsonlDataset(
        data_path=lora_train_config.data_dir,
        tokenizer=tokenizer,
        context_length=config.context_length,
        split="valid",
        max_samples=500,
        mask_question=lora_train_config.mask_question,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    save_root = os.path.join(lora_train_config.checkpoint_base_dir, train_config.project_name)
    os.makedirs(save_root, exist_ok=True)
    tensorboard_dir = os.path.join("/root/tf-logs", f"{train_config.project_name}_lora")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    for epoch in range(train_config.epochs):
        avg_loss, global_step = train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            valid_steps=lora_train_config.valid_steps,
            checkpoint_save_steps=lora_train_config.checkpoint_save_steps,
            fix_lr=lora_train_config.fix_lr,
            lr_scheduler_warmup_steps=lora_train_config.lr_scheduler_warmup_steps,
            lr_scheduler_total_steps=lora_train_config.lr_scheduler_total_steps,
            lr_scheduler_min_lr=lora_train_config.lr_scheduler_min_lr,
            lr_scheduler_max_lr=lora_train_config.lr_scheduler_max_lr,
            print_optimizer_update_ratio=lora_train_config.print_optimizer_update_ratio,
            save_root=save_root,
            epoch=epoch,
            full_finetune=full_finetune,
            global_step=global_step,
            writer=writer,
        )
        writer.add_scalar("lora/train_loss_epoch", avg_loss, epoch + 1)
        print(f"Epoch {epoch} finished, avg loss {avg_loss}, global_step {global_step}")

    writer.close()


if __name__ == "__main__":
    main()
