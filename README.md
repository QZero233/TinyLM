<div align="center">

# 🧠 TinyLM

### 从零实现的 Transformer 中文预训练语言模型

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](./pyproject.toml)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/model-139M%20Transformer-orange.svg)](./CONFIG.md#model--模型架构配置)
[![Runner](https://img.shields.io/badge/runner-uv-8A2BE2.svg)](https://github.com/astral-sh/uv)

`Transformer from scratch` · `Pre-training` · `SFT / LoRA` · `KV Cache Inference`

</div>

---

## 项目简介

TinyLM 是一个从零实现的 Transformer 语言模型项目，基于 **PyTorch**，在 **中文百科类数据**（百度百科 + 中文维基百科）上预训练了约 **139M 参数** 的模型，并进行了指令微调。

| 版本 | 参数量 | 数据 | 状态 |
|------|--------|------|------|
| **当前分支** | **139M**（d_model=768, layers=16） | 百度百科 + 中文维基 | ✅ 可用 |
| legacy\_305M 分支 | 305M（d_model=1024, layers=16） | OpenAI WebText（英文） | 📦 已归档 |

> 旧版 305M 英文模型的代码已归档至 `legacy_305M` 分支。

## 数据准备

### 下载预训练数据

预训练数据为百度百科和中文维基百科的 tokenized `.bin` 文件，从 ModelScope 下载：

https://www.modelscope.cn/datasets/wdndev/tiny_llm_dataset

下载后放入 `data/zh_data/` 目录：

```
data/
├── zh_data/
│   ├── baidubaike_563w_1.bin
│   ├── baidubaike_563w_2.bin
│   ├── baidubaike_563w_3.bin
│   ├── baidubaike_563w_4.bin
│   ├── baidubaike_563w_5.bin
│   └── wikipedia-cn.bin
├── sft_data/
│   ├── sft_data.jsonl
│   └── sft_data_test.jsonl
└── chatglm3_tokenizer/
    ├── tokenizer.model
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

### 下载 SFT 数据

在同一数据集页面下载 SFT 用的 JSONL 文件，放入 `data/sft_data/` 目录。每行格式为：

```json
{"question": "...", "answer": "..."}
```

### 分词器

使用 ChatGLM3 的 BPE 分词器，已包含在 `data/chatglm3_tokenizer/` 目录中。

## 环境准备

```bash
uv sync
```

## 配置文件

所有训练参数通过 JSON 配置文件管理。详见 [CONFIG.md](./CONFIG.md)。

默认配置：`configs/train_zh.json`。各字段含义详见 [CONFIG.md](./CONFIG.md)。

## 预训练

预训练参数已在 JSON 中配置好，直接启动即可：

```bash
uv run python -m tiny_lm.train_model --config configs/train_zh.json
```

### 从 Checkpoint 恢复训练

修改 JSON 中的 `checkpoint` 字段指向已有的 `.cpt` 文件：

```json
{
  "training": {
    "checkpoint": "/path/to/checkpoint.cpt",
    ...
  }
}
```

> 传入空字符串或不存在的路径时从随机初始化开始。

## 指令微调

微调同样通过 JSON 配置，使用 `lora` 子节控制微调参数。通过 `full_finetune` 字段切换全量微调和 LoRA：

```json
{
  "training": {
    "lora": {
      "full_finetune": true,
      "base_model_checkpoint": "/path/to/pretrained_checkpoint.cpt",
      ...
    }
  }
}
```

### 全量微调（SFT）

```bash
uv run python -m tiny_lm.lora.lora_fine_tuning --config configs/train_zh.json
```

需要设置：
- `lora.full_finetune: true`
- `lora.base_model_checkpoint`：指向预训练 checkpoint
- `lora.data_dir`：SFT 数据目录（JSONL 文件所在目录）

### LoRA 微调

同样一条命令，切换 `full_finetune` 即可：

```json
{
  "training": {
    "lora": {
      "full_finetune": false,
      "r": 8,
      ...
    }
  }
}
```

```bash
uv run python -m tiny_lm.lora.lora_fine_tuning --config configs/train_zh.json
```

> **注意**：LoRA 效果通常不如全量微调，推荐优先使用全量微调。LoRA 模式下学习率建议设为全量的 1/10～1/100。

### 从 Checkpoint 恢复微调

全量微调恢复：设置 `lora.lora_checkpoint` 指向已有的 SFT checkpoint。

LoRA 恢复：同样设置 `lora.lora_checkpoint` 指向已有的 LoRA checkpoint（需要配套设置 `lora.base_model_checkpoint`）。

## 生成与评估

使用 `eval_model.py` 进行文本生成或验证集 loss 评估：

```bash
# 文本生成
uv run python -m tiny_lm.eval_model --config configs/train_zh.json --prompt "中国的首都是" --max_seq_len 256

# 验证集 loss
uv run python -m tiny_lm.eval_model --config configs/train_zh.json --mode valid_loss
```

## WebUI

提供浏览器交互界面：

```bash
uv run python webui/app.py --config configs/train_zh.json --port 6008
```

支持 base checkpoint 和 LoRA checkpoint 两种模式。

## 分布式训练

⚠️ **当前处于 TODO 状态，暂不可用。**

分布式训练代码位于 `tiny_lm/dist_train/`，尚未适配当前版本。

## 项目亮点

- 从零实现 Transformer 核心模块（多头注意力、RoPE、RMSNorm、SwiGLU）
- 完整的预训练 + SFT + LoRA 微调流程
- 推理阶段 KV Cache 增量解码
- 统一的 JSON 配置文件管理所有训练参数
- 基于 `uv` 的简洁运行方式

## 项目来源

本项目代码来源于 **Stanford CS336 Assignment 1**，在原开源协议前提下进行了结构调整与功能扩展。

- GitHub：https://github.com/QZero233/TinyLM
- ModelScope（权重与数据）：https://www.modelscope.cn/models/QZero233/tiny_lm
