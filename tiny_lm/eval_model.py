import argparse
import math
import random
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from tiny_lm import (
    DEFAULT_TRAIN_CONFIG,
    LoRAConfig,
    ModelConfig,
    TrainConfig,
    TransformerKVCache,
    cross_entropy_loss,
    get_dataset,
    get_tokenizer,
    init_model,
    init_model_from_checkpoint,
    load_checkpoint,
    load_lora_trainable_checkpoint,
    load_train_config,
)


device = "cpu"
TokenizerT = Any


def _apply_repetition_penalty(logits: torch.Tensor, input_token_ids: List[int], repetition_penalty: float) -> torch.Tensor:
    if repetition_penalty <= 1.0 or len(input_token_ids) == 0:
        return logits

    unique_token_ids = set(input_token_ids)
    token_index = torch.tensor(list(unique_token_ids), dtype=torch.long, device=logits.device)
    selected = logits[token_index]
    logits[token_index] = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
    return logits


def _predict_next_token(
    input_token_ids: List[int],
    model: torch.nn.Module,
    temperature: float = 1,
    top_p: float = 0.9,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    kv_cache: Optional[TransformerKVCache] = None,
) -> int:
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device)

    model.eval()
    with torch.no_grad():
        output = model(input_token_ids_tensor, kv_cache=kv_cache)
        output /= max(temperature, 1e-5)

        logits = output[-1]
        logits = _apply_repetition_penalty(logits, input_token_ids, repetition_penalty)

        if greedy:
            max_token_id = torch.argmax(logits)
            return max_token_id.item()

    prob = torch.softmax(logits, dim=-1)
    prob_map: List[Tuple[float, int]] = [(prob[i].item(), i) for i in range(prob.shape[-1])]
    prob_map.sort(reverse=True)

    accumulated_prob = 0.0
    end_index = 0
    for i in range(len(prob_map)):
        accumulated_prob += prob_map[i][0]
        if accumulated_prob >= top_p:
            end_index = i + 1
            break

    new_distribution: List[Tuple[float, int]] = []
    for i in range(end_index):
        new_distribution.append((prob_map[i][0] / accumulated_prob, prob_map[i][1]))

    if not new_distribution:
        return torch.argmax(logits).item()

    _, res_token_id = random.choices(new_distribution, weights=[p[0] for p in new_distribution], k=1)[0]
    return res_token_id


def _encode_text(tokenizer: TokenizerT, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _get_eos_token_id(tokenizer: TokenizerT) -> Optional[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return None


def _build_lora_configs(model_config: ModelConfig, r: int, device: str = "cpu") -> List[LoRAConfig]:
    lora_configs: List[LoRAConfig] = []
    for i in range(model_config.num_layers):
        for component in ["linear_q", "linear_k", "linear_v", "linear_out"]:
            b = torch.zeros(model_config.d_model, r, device=device)
            a = torch.zeros(r, model_config.d_model, device=device)
            init_std = 2 / (r + model_config.d_model)
            torch.nn.init.trunc_normal_(b, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
            torch.nn.init.trunc_normal_(a, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
            lora_configs.append(LoRAConfig(f"transformer_layers.{i}.multi_head_attn.{component}", b, a))
    return lora_configs


def load_model_for_inference(
    model_config: ModelConfig,
    checkpoint: str,
    lora_checkpoint: str = "",
    lora_base_checkpoint: str = "",
    lora_r: int = 8,
    lora_full_finetune: bool = False,
    device: str = "cpu",
) -> torch.nn.Module:
    model_config.pytorch_impl = False

    if lora_checkpoint:
        if lora_full_finetune:
            model = init_model_from_checkpoint(model_config, lora_checkpoint)
        else:
            model = init_model_from_checkpoint(model_config, lora_base_checkpoint or checkpoint)
            model = model.to(device)
            lora_configs = _build_lora_configs(model_config, lora_r, device)
            model.adapt_lora(lora_configs)
            try:
                load_lora_trainable_checkpoint(lora_checkpoint, model)
            except Exception:
                from tiny_lm import load_lora_configs
                loaded_lora = load_lora_configs(lora_checkpoint)
                loaded_map = {cfg.module_name: cfg for cfg in loaded_lora}
                for cfg in lora_configs:
                    if cfg.module_name in loaded_map:
                        cfg.b.copy_(loaded_map[cfg.module_name].b.to(cfg.b.device))
                        cfg.a.copy_(loaded_map[cfg.module_name].a.to(cfg.a.device))
    else:
        model = init_model_from_checkpoint(model_config, checkpoint)
        model = model.to(device)

    model.eval()
    return model


def generate(
    model: torch.nn.Module,
    tokenizer: TokenizerT,
    input_token_ids: List[int],
    max_seq_len: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    kv_cache: Optional[TransformerKVCache] = None,
) -> dict:
    if not input_token_ids:
        raise ValueError("input_token_ids is empty")

    token_ids = list(input_token_ids)
    eos_token_id = _get_eos_token_id(tokenizer) if eos_token_id is None else eos_token_id

    while len(token_ids) < max_seq_len and (eos_token_id is None or token_ids[-1] != eos_token_id):
        next_id = _predict_next_token(
            token_ids,
            model,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
            kv_cache=kv_cache,
        )
        token_ids.append(next_id)

    return {
        "text": tokenizer.decode(token_ids),
        "input_token_ids": input_token_ids,
        "output_token_ids": token_ids[len(input_token_ids):],
        "full_token_ids": token_ids,
    }


def _auto_regression(
    prompt: str,
    max_seq_len: int,
    model: torch.nn.Module,
    tokenizer: TokenizerT,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    kv_cache: Optional[TransformerKVCache] = None,
) -> str:
    input_token_ids = _encode_text(tokenizer, prompt)
    result = generate(
        model,
        tokenizer,
        input_token_ids,
        max_seq_len=max_seq_len,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        kv_cache=kv_cache,
    )
    return result["text"]


def _eval_valid_loss(model: torch.nn.Module, config: ModelConfig, train_config: TrainConfig,
                     eval_batch_size: int) -> None:
    valid_data = get_dataset(
        train_config.data_dir,
        seq_len=config.context_length,
        zh_token_dtype=train_config.zh_token_dtype,
        full_random=False,
    )
    data_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=True)
    total_loss = 0.0
    total_num = 0

    model.eval()
    n = len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            logits = model(x, None)
            loss = cross_entropy_loss(logits, y)

        total_loss += loss.item()
        total_num += 1
        print(f"({i + 1}/{n}) Current average loss {total_loss / total_num}")

    print(f"Final loss {total_loss / total_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or generate with TinyLM")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--mode", type=str, choices=["generate", "valid_loss"], default="generate")
    parser.add_argument("--prompt", type=str, default="美国首都是")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    train_config, config, _ = load_train_config(args.config)
    config.pytorch_impl = False
    tokenizer = get_tokenizer(train_config.tokenizer_path)

    checkpoint = train_config.checkpoint or None
    model = load_model_for_inference(config, checkpoint, device=device)

    if args.mode == "valid_loss":
        _eval_valid_loss(model, config, train_config, args.eval_batch_size)
    else:
        kv_cache = TransformerKVCache(config.num_layers)
        print(
            _auto_regression(
                args.prompt,
                max_seq_len=args.max_seq_len,
                model=model,
                tokenizer=tokenizer,
                repetition_penalty=args.repetition_penalty,
                greedy=args.greedy,
                kv_cache=kv_cache,
            )
        )
