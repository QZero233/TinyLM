import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from tiny_lm import DEFAULT_TRAIN_CONFIG, get_dataset, load_train_config
from tiny_lm.hf.model_def import get_model


TENSORBOARD_ROOT = "/root/tf-logs"


class CausalLMDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.dataset[idx]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def reset(self) -> None:
        if hasattr(self.dataset, "reset"):
            self.dataset.reset()


class ResetDatasetCallback(TrainerCallback):
    def __init__(self, dataset: CausalLMDataset):
        self.dataset = dataset

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs: Any):
        self.dataset.reset()


class TinyLMHFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return (loss, outputs) if return_outputs else loss


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _resolve_resume_checkpoint(config_checkpoint: str, output_dir: str, explicit_resume: str | None) -> str | None:
    if explicit_resume:
        return explicit_resume

    if config_checkpoint:
        checkpoint_path = Path(config_checkpoint)
        if checkpoint_path.is_dir():
            return str(checkpoint_path)
        if checkpoint_path.exists():
            print(f"Skip non-HF checkpoint from config: {checkpoint_path}")

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        print(f"Resume from latest HF checkpoint: {last_checkpoint}")
    return last_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyLM with Hugging Face Transformers Trainer")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--logging-dir", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    train_config, model_config, optimizer_config = load_train_config(args.config)
    output_dir = args.output_dir or os.path.join(
        train_config.checkpoint_base_dir,
        f"{train_config.project_name}_hf",
    )
    logging_dir = args.logging_dir or os.path.join(TENSORBOARD_ROOT, f"{train_config.project_name}_hf")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    model = get_model(args.config)
    if model_config.gradient_checkpoint:
        model.config.use_cache = False

    train_dataset = CausalLMDataset(
        get_dataset(
            train_config.data_dir,
            seq_len=model_config.context_length,
            zh_token_dtype=train_config.zh_token_dtype,
            full_random=True,
            train=True,
        )
    )
    eval_dataset = None
    if not args.no_eval:
        eval_dataset = CausalLMDataset(
            get_dataset(
                train_config.data_dir,
                seq_len=model_config.context_length,
                zh_token_dtype=train_config.zh_token_dtype,
                full_random=True,
                train=False,
            )
        )

    fixed_lr = train_config.fix_lr is not None
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=train_config.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulate,
        learning_rate=train_config.fix_lr if fixed_lr else optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
        adam_beta1=optimizer_config.beta_1,
        adam_beta2=optimizer_config.beta_2,
        adam_epsilon=optimizer_config.eps,
        max_grad_norm=1.0,
        lr_scheduler_type="constant" if fixed_lr else "cosine",
        warmup_steps=0 if fixed_lr else 2000,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=train_config.checkpoint_save_accum_steps,
        save_total_limit=10,
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=train_config.valid_steps if eval_dataset is not None else None,
        prediction_loss_only=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        bf16=not args.no_bf16 and _bf16_supported(),
        gradient_checkpointing=model_config.gradient_checkpoint,
        torch_compile=args.torch_compile,
        dataloader_pin_memory=torch.cuda.is_available(),
        save_safetensors=True,
    )

    trainer = TinyLMHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[ResetDatasetCallback(train_dataset)],
    )

    resume_checkpoint = _resolve_resume_checkpoint(
        train_config.checkpoint,
        output_dir,
        args.resume_from_checkpoint,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
