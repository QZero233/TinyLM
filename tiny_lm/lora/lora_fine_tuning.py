import math

from torch.utils.data import DataLoader

from tiny_lm.train_model import load_tokenizer
from tiny_lm import LoRAConfig, cross_entropy_loss, gradient_clip, save_checkpoint, save_lora_configs, \
    get_default_model_config, get_default_optimizer_config, init_model_from_checkpoint, init_optimizer
import torch
from torch import nn
import numpy as np

import os


def print_parameter_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"[Grad] {name}: None")
        else:
            print(f"[Grad] {name}: {param.grad}")

class RepeatInstFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "/root/autodl-tmp/TinyLM/fine_tuning_data/lima_1025"):
        data_file = os.path.join(data_dir, "repeat.npy")
        prompt_len_file = os.path.join(data_dir, "prompt_len.npy")
        self.raw_data = np.load(data_file)
        self.prompt_len_data = np.load(prompt_len_file)
        assert self.raw_data.shape[0] == len(self.prompt_len_data)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        data = self.raw_data[idx, :]
        x = data[:-1]
        y = data[1:]

        prompt_len = self.prompt_len_data[idx]
        return torch.LongTensor(x), torch.LongTensor(y), torch.IntTensor([prompt_len])

if __name__ == "__main__":
    old_tokenizer = load_tokenizer("/root/autodl-tmp/CS336_A1/saved_gpt_tokenizer")
    new_tokenizer = load_tokenizer("/root/autodl-tmp/CS336_A1/saved_gpt_inst_tokenizer")

    config = get_default_model_config(len(old_tokenizer.vocab))
    optimizer_config = get_default_optimizer_config()
    optimizer_config.lr = 3e-4
    model = init_model_from_checkpoint(config, "/root/autodl-tmp/TinyLM/checkpoint/4.5B.cpt")

    model.resize_embedding(len(new_tokenizer.vocab))
    model.train()
    model.to("cuda")

    r = 8
    lora_config_list = []

    for i in range(config.num_layers):
        for component in ["linear_q", "linear_k", "linear_v", "linear_out"]:
            b = torch.zeros(config.d_model, r).cuda()
            a = torch.zeros(r, config.d_model).cuda()

            init_std = 2 / (r + config.d_model)
            nn.init.trunc_normal_(b, mean=0, std=init_std, a=-3 * math.sqrt(init_std),
                                  b=3 * math.sqrt(init_std))
            nn.init.trunc_normal_(a, mean=0, std=init_std, a=-3 * math.sqrt(init_std),
                                  b=3 * math.sqrt(init_std))

            lora_config_list.append(LoRAConfig(f"transformer_layers.{i}.multi_head_attn.{component}", b, a))
    model.adapt_lora(lora_config_list)

    optimizer = init_optimizer(optimizer_config, model)

    batch_size = 16
    dataset = RepeatInstFinetuneDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    n = len(dataloader)
    print(f"Total {n} batches")

    data_length = 1024
    length_mask_ranges = torch.arange(data_length).broadcast_to((batch_size, data_length)).cuda()

    pad_encode = new_tokenizer.encode("<|pad|>")
    assert len(pad_encode) == 1
    pad_token_id = pad_encode[0]

    total_loss = 0
    checkpoint_dir = "/root/autodl-tmp/TinyLM/checkpoint_train"
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(20):
        for i, (x, y, prompt_len) in enumerate(dataloader):
            x=x.cuda()
            y=y.cuda()
            prompt_len=prompt_len.cuda()

            current_batch_size = prompt_len.shape[0]

            # 只有ans部分才算loss
            prompt_mask = length_mask_ranges[:current_batch_size] < prompt_len
            pad_mask = y == pad_token_id
            mask = prompt_mask | pad_mask
            y[mask] = -100

            logits = model(x, None)
            loss = cross_entropy_loss(logits, y, ignore_label=-100)
            loss.backward()

            # if (i + 1) % 10 == 0:
            #     print(f"\n===== Gradient dump at step {i + 1} =====")
            #     print_parameter_gradients(model)
            #     print("===== End gradient dump =====\n")

            gradient_clip(model.parameters(), m=1)
            optimizer.step()
            optimizer.zero_grad()

            # if i % 20 == 0:
            total_loss += loss.item()
            print(f"Step {i+1}/{n}, total loss {total_loss}, avg loss {total_loss/(i+1)}, last step loss {loss.item()}")

        checkpoint_file = os.path.join(checkpoint_dir, f"repeat_{epoch}.cpt")
        lora_config_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}_lora")
        save_checkpoint(model, None, 0, checkpoint_file)
        save_lora_configs(lora_config_list, lora_config_dir)
        print(f"Saved checkpoint at {checkpoint_file}")

        print(f"Finish epoch {epoch}")
