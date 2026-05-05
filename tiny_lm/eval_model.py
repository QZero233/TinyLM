import random
import os
import argparse
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from tiny_lm import TinyStoryDataset, cross_entropy_loss, BPETokenizer, TransformerKVCache
from tiny_lm.train_model import LMConfig, load_tokenizer, init_model, _get_model_config

device="cuda"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    kv_cache: Optional[TransformerKVCache] = None
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

    # logits: (vocab_size,)
    prob = torch.softmax(logits, dim=-1)
    prob_map: List[Tuple[float, int]] = [(prob[i].item(), i) for i in range(prob.shape[-1])]
    prob_map.sort()
    prob_map.reverse()

    accumulated_prob = 0
    end_index = 0
    for i in range(len(prob_map)):
        accumulated_prob += prob_map[i][0]

        if accumulated_prob >= top_p:
            end_index = i + 1
            break

    new_distribution: List[Tuple[float, int]] = []
    for i in range(end_index):
        new_distribution.append((prob_map[i][0] / accumulated_prob, prob_map[i][1]))

    if len(new_distribution) == 0:
        return torch.argmax(logits).item()

    _, res_token_id = random.choices(new_distribution, weights=[p[0] for p in new_distribution], k=1)[0]
    return res_token_id

def _auto_regression(
    prompt: str,
    max_seq_len: int,
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    repetition_penalty: float = 1.0,
    kv_cache: Optional[TransformerKVCache] = None
) -> str:
    assert len(tokenizer.encode("<|endoftext|>")) == 1
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    token_ids = tokenizer.encode(prompt)

    while len(token_ids) < max_seq_len and token_ids[-1] != eos_token_id:
        next_id = _predict_next_token(
            token_ids,
            model,
            repetition_penalty=repetition_penalty,
            kv_cache=kv_cache
        )
        token_ids.append(next_id)
        if len(token_ids) % 20 == 0:
            print(tokenizer.decode(token_ids))

    return tokenizer.decode(token_ids)

def _eval_valid_loss(model: torch.nn.Module, config: LMConfig):
    valid_data = TinyStoryDataset(os.path.join(PROJECT_ROOT, "data"), seq_len=config.context_length, train=False)
    data_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
    total_loss = 0
    total_num = 0

    model.eval()
    n = len(data_loader)
    for i, (x, y) in enumerate(data_loader):
        x=x.to(device=device)
        y=y.to(device=device)

        with torch.no_grad():
            logits = model(x, None)
            loss = cross_entropy_loss(logits, y)

        total_loss += loss.item()
        total_num += 1
        print(f"({i + 1}/{n}) Current average loss {total_loss / total_num}")

    print(f"Final loss {total_loss / total_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or generate with TinyLM")
    parser.add_argument("--tokenizer_dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_gpt_tokenizer"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoint", "4.5B.cpt"))
    parser.add_argument("--mode", type=str, choices=["generate", "valid_loss"], default="generate")
    parser.add_argument("--prompt", type=str, default="The capital of the United States is a place called")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    args = parser.parse_args()

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

    model = init_model(config, args.checkpoint if args.checkpoint else None)
    model = model.to(device)

    if args.mode == "valid_loss":
        _eval_valid_loss(model, config)
    else:
        kv_cache = TransformerKVCache(config.num_layers)
        print(_auto_regression(args.prompt, max_seq_len=args.max_seq_len, model=model, tokenizer=tokenizer, repetition_penalty=1.05, kv_cache=kv_cache))
