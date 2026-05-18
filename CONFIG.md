# 配置文件说明

TinyLM 使用 JSON 格式的配置文件管理所有训练/推理参数。文件包含三个顶级节：`training`、`model`、`optimizer`。

## 示例

```json
{
  "training": {
    "project_name": "zh_pretrain_139M",
    "tokenizer_path": "data/chatglm3_tokenizer",
    "checkpoint": "checkpoint/zh_pretrain_139M/72300.cpt",
    "data_dir": "data/zh_data",
    "zh_token_dtype": "uint16",
    "checkpoint_base_dir": "checkpoint",
    "epochs": 20,
    "gradient_accumulate": 8,
    "checkpoint_save_accum_steps": 400,
    "valid_steps": 200,
    "fix_lr": null,
    "print_optimizer_update_ratio": false,
    "lora": { ... }
  },
  "model": { ... },
  "optimizer": { ... }
}
```

---

## `training` — 训练配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project_name` | string | — | 项目名称，用于命名 checkpoint 子目录 |
| `tokenizer_path` | string | — | 分词器目录路径 |
| `checkpoint` | string | "" | 初始 checkpoint 路径。设为空字符串或不存在的路径时从随机初始化开始 |
| `data_dir` | string | — | 预训练数据的 `.bin` 文件所在目录 |
| `zh_token_dtype` | string | "uint16" | 预训练数据的 token dtype（uint16 / uint32 / int32 / int64） |
| `checkpoint_base_dir` | string | — | checkpoint 保存的根目录，最终保存到 `{checkpoint_base_dir}/{project_name}/` |
| `epochs` | int | — | 训练轮数 |
| `gradient_accumulate` | int | — | 梯度累积步数。实际 batch size = batch_size × gradient_accumulate |
| `checkpoint_save_accum_steps` | int | 200 | 每多少累积步保存一次 checkpoint |
| `valid_steps` | int | 200 | 每多少步在验证集上评估一次 loss |
| `fix_lr` | float \| null | null | 固定学习率（不为 null 时覆盖 scheduler） |
| `print_optimizer_update_ratio` | bool | false | 是否打印每个参数的实际更新比率 |

### `training.lora` — LoRA / SFT 子配置

当配置了 `lora` 子节时，使用 `tiny_lm.lora.lora_fine_tuning` 进行微调。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | int | — | LoRA rank。`full_finetune=true` 时忽略 |
| `checkpoint_base_dir` | string | — | 微调 checkpoint 保存根目录 |
| `data_dir` | string | — | SFT 数据目录（需包含 `sft_data.jsonl` 和 `sft_data_test.jsonl`） |
| `base_model_checkpoint` | string | — | 基础预训练模型的 checkpoint 路径 |
| `lora_checkpoint` | string | "" | 已有的 LoRA/SFT checkpoint 路径，用于恢复训练 |
| `full_finetune` | bool | false | true = 全量微调，false = LoRA 微调 |
| `fix_lr` | float \| null | null | 固定学习率 |
| `valid_steps` | int | 200 | 验证间隔步数 |
| `checkpoint_save_steps` | int | 200 | checkpoint 保存间隔步数 |
| `mask_question` | bool | true | 是否在 SFT loss 计算中 mask 掉 question 部分（只计算 answer 的 loss） |
| `auto_resize_embedding` | bool | true | 当 tokenizer 词表大小与模型不匹配时自动 resize embedding |
| `print_optimizer_update_ratio` | bool | false | 是否打印更新比率 |
| `lr_scheduler_warmup_steps` | int | 3600 | 学习率预热步数 |
| `lr_scheduler_total_steps` | int | 120000 | 学习率调度总步数 |
| `lr_scheduler_min_lr` | float | 3e-6 | 最小学习率（cosine 调度底值） |
| `lr_scheduler_max_lr` | float | 1e-5 | 最大学习率（预热后到达的峰值） |

---

## `model` — 模型架构配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vocab_size` | int | 1 | 词表大小 |
| `context_length` | int | 1024 | 最大序列长度 |
| `num_layers` | int | 8 | Transformer 层数 |
| `d_model` | int | 512 | 模型隐层维度 |
| `num_heads` | int | 8 | 注意力头数（需整除 d_model） |
| `d_ff` | int | 1024 | 前馈网络隐层维度（SwiGLU 的实际中间维度为此值的 2 倍） |
| `theta` | float | 10000 | RoPE 频率基数 |
| `gradient_checkpoint` | bool | false | 是否使用梯度 checkpoint 节省显存 |
| `batch_size` | int | 16 | 单步 batch size |
| `pytorch_impl` | bool | false | 是否使用 PyTorch 原生实现（`F.scaled_dot_product_attention`） |

## `optimizer` — 优化器配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr` | float | 1e-3 | 学习率 |
| `beta_1` | float | 0.9 | AdamW beta1 |
| `beta_2` | float | 0.95 | AdamW beta2 |
| `weight_decay` | float | 0.1 | 权重衰减 |
| `eps` | float | 1e-6 | AdamW epsilon |
| `reset` | bool | false | 是否重置优化器状态（从 checkpoint 恢复时忽略旧优化器状态） |
