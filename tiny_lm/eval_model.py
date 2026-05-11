import argparse
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from tiny_lm import (
    DEFAULT_TRAIN_CONFIG,
    ModelConfig,
    TransformerKVCache,
    cross_entropy_loss,
    get_dataset,
    get_tokenizer,
    init_model_from_checkpoint,
    load_train_config,
)


device = "cuda"


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
    temperature: float = 0.9,
    top_p: float = 0.9,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    kv_cache: Optional[TransformerKVCache] = None,
) -> int:
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device)

    model.eval()
    with torch.no_grad():
        output = model(input_token_ids_tensor, kv_cache=kv_cache)
        output /= temperature

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


def _get_eos_token_id(tokenizer) -> Optional[int]:
    if hasattr(tokenizer, "eos_id"):
        return tokenizer.eos_id

    if hasattr(tokenizer, "encode"):
        try:
            eos_ids = tokenizer.encode("<|endoftext|>")
        except Exception:
            eos_ids = []
        if len(eos_ids) == 1:
            return eos_ids[0]

    return None


def _auto_regression(
    prompt: str,
    max_seq_len: int,
    model: torch.nn.Module,
    tokenizer,
    repetition_penalty: float = 1.0,
    kv_cache: Optional[TransformerKVCache] = None,
) -> str:
    token_ids = tokenizer.encode(prompt)
    eos_token_id = _get_eos_token_id(tokenizer)

    while len(token_ids) < max_seq_len and (eos_token_id is None or token_ids[-1] != eos_token_id):
        next_id = _predict_next_token(
            token_ids,
            model,
            repetition_penalty=repetition_penalty,
            kv_cache=kv_cache,
        )
        token_ids.append(next_id)
        if len(token_ids) % 20 == 0:
            print(tokenizer.decode(token_ids))

    return tokenizer.decode(token_ids)


def _eval_valid_loss(model: torch.nn.Module, config: ModelConfig, train_config, eval_batch_size: int):
    valid_data = get_dataset(
        train_config.dataset_type,
        train_config.data_dir,
        seq_len=config.context_length,
        zh_token_dtype=train_config.zh_token_dtype,
        zh_fold=train_config.zh_fold,
        train=False,
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["generate", "valid_loss"], default="generate")
    parser.add_argument("--prompt", type=str, default="美国首都是")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    train_config, config, _ = load_train_config(args.config)
    tokenizer = get_tokenizer(train_config.tokenizer_name)

    checkpoint = args.checkpoint if args.checkpoint is not None else train_config.checkpoint
    checkpoint = checkpoint if checkpoint else None
    model = init_model_from_checkpoint(config, checkpoint)
    model = model.to(device)

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
                kv_cache=kv_cache,
            )
        )
