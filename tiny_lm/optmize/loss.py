from typing import Optional, Tuple

import torch
from torch import nn

def _cross_entropy_loss_origin(logits: torch.Tensor, label: torch.LongTensor, ignore_label: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    # logits: (...batch, seq_len, vocab_size)
    # label: (...batch, seq_len)
    # return: (...batch, seq_len)
    assert ignore_label is None or ignore_label < 0

    if ignore_label is None:
        ignore_label = -1

    # Mask为True的位置才会保留
    mask = label != ignore_label
    # 把ignore label替换成一个合法的token
    safe_label = label.clone()
    safe_label[~mask] = 0

    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_adjust = logits - logits_max
    logits_exp_sum_log = torch.log(torch.exp(logits_adjust).sum(dim=-1, keepdim=True))
    logits_selected = logits_adjust.gather(dim=-1, index=safe_label.unsqueeze(-1))

    return mask * (logits_exp_sum_log - logits_selected).squeeze(-1), mask.sum().item()

def cross_entropy_loss(logits: torch.Tensor, label: torch.LongTensor, ignore_label: Optional[int] = None) -> torch.Tensor:
    origin_loss, nums = _cross_entropy_loss_origin(logits, label, ignore_label)
    return origin_loss.sum() / nums

def perplexity(logits: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
    # return: (...batch)
    origin_loss = _cross_entropy_loss_origin(logits, label)
    return torch.exp(origin_loss.sum(dim=-1))