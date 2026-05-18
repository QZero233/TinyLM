import argparse
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror
from urllib import request as urlrequest
from typing import Any, Optional

import torch

from tiny_lm import (
    LoRAConfig,
    get_tokenizer,
    init_model,
    init_model_from_checkpoint,
    load_checkpoint,
    load_lora_configs,
    load_lora_trainable_checkpoint,
    load_train_config,
)


HTML_PAGE = """<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>TinyLM WebUI</title>
  <style>
    :root {
      --bg: #f5f4ef;
      --card: #fffef8;
      --line: #ddd6c8;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-2: #115e59;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif;
      background: radial-gradient(circle at 10% 0%, #fbf8e8 0%, var(--bg) 45%), var(--bg);
      color: var(--ink);
      min-height: 100vh;
      padding: 24px;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 10px 24px rgba(17, 24, 39, 0.08);
    }
    h1 { margin: 0 0 16px; font-size: 24px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .full { grid-column: 1 / -1; }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 100px; resize: vertical; }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      font-weight: 700;
      cursor: pointer;
      transition: background .18s ease;
    }
    button:hover { background: var(--accent-2); }
    .status { margin: 8px 0; color: var(--muted); min-height: 20px; }
    .output {
      border: 1px solid var(--line);
      border-radius: 10px;
      min-height: 120px;
      padding: 12px;
      white-space: pre-wrap;
      background: #fffeff;
    }
    .token-panel {
      margin-top: 12px;
      display: grid;
      gap: 12px;
    }
    .token-box {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      background: #fff;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 72px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }
    .token-title {
      margin: 0 0 6px;
      font-size: 13px;
      color: var(--muted);
    }
    .tiny { font-size: 12px; color: var(--muted); }
    @media (max-width: 760px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>TinyLM 推理 WebUI</h1>
    <div class=\"row\">
      <div class=\"full\">
        <label>Mode</label>
        <select id=\"mode\">
          <option value=\"base\">Base</option>
          <option value=\"lora\">LoRA</option>
        </select>
      </div>
      <div class=\"full\">
        <label>Checkpoint</label>
        <select id=\"checkpoint\"></select>
      </div>
      <div class=\"full\" id=\"lora_checkpoint_row\" style=\"display:none;\">
        <label>LoRA Checkpoint</label>
        <select id=\"lora_checkpoint\"></select>
      </div>
      <div class=\"full\" id=\"lora_special_token_row\" style=\"display:none;\">
        <label><input id=\"lora_add_special_tokens\" type=\"checkbox\" checked /> Add special token (<|user|> ... <|assistant|>)</label>
      </div>
      <div>
        <label>Max Seq Len</label>
        <input id=\"max_seq_len\" type=\"number\" min=\"1\" value=\"32\" />
      </div>
      <div>
        <label>Temperature</label>
        <input id=\"temperature\" type=\"number\" step=\"0.1\" min=\"0.1\" value=\"1.0\" />
      </div>
      <div>
        <label>Top P</label>
        <input id=\"top_p\" type=\"number\" step=\"0.05\" min=\"0.01\" max=\"1\" value=\"0.9\" />
      </div>
      <div>
        <label>Repetition Penalty</label>
        <input id=\"repetition_penalty\" type=\"number\" step=\"0.1\" min=\"1\" value=\"1.0\" />
      </div>
      <div class=\"full\">
        <label><input id=\"greedy\" type=\"checkbox\" checked /> Greedy</label>
      </div>
      <div class=\"full\">
        <label>Prompt</label>
        <textarea id=\"prompt\">中国首都是</textarea>
      </div>
    </div>
    <button id=\"run\">生成</button>
    <div class=\"status\" id=\"status\"></div>
    <div class=\"output\" id=\"output\"></div>
    <div class=\"token-panel\">
      <div>
        <div class=\"token-title\">输入 token ids</div>
        <div class=\"token-box\" id=\"input_token_ids\"></div>
      </div>
      <div>
        <div class=\"token-title\">输出 token ids</div>
        <div class=\"token-box\" id=\"output_token_ids\"></div>
      </div>
    </div>
    <p class=\"tiny\" id=\"meta\"></p>
  </div>

  <script>
    const checkpointSel = document.getElementById('checkpoint');
    const modeSel = document.getElementById('mode');
    const loraCheckpointRow = document.getElementById('lora_checkpoint_row');
    const loraCheckpointSel = document.getElementById('lora_checkpoint');
    const loraSpecialTokenRow = document.getElementById('lora_special_token_row');
    const loraAddSpecialTokens = document.getElementById('lora_add_special_tokens');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const inputTokenIdsEl = document.getElementById('input_token_ids');
    const outputTokenIdsEl = document.getElementById('output_token_ids');
    const metaEl = document.getElementById('meta');
    const runBtn = document.getElementById('run');
    let checkpointPayload = null;

    async function loadCheckpoints() {
      const res = await fetch('/api/checkpoints');
      const data = await res.json();
      checkpointPayload = data;
      checkpointSel.innerHTML = '';
      for (const item of data.checkpoints) {
        const opt = document.createElement('option');
        opt.value = item.path;
        opt.textContent = item.label;
        checkpointSel.appendChild(opt);
      }
      if (data.default_checkpoint) checkpointSel.value = data.default_checkpoint;

      loraCheckpointSel.innerHTML = '';
      for (const item of (data.lora_checkpoints || [])) {
        const opt = document.createElement('option');
        opt.value = item.path;
        opt.textContent = item.label;
        loraCheckpointSel.appendChild(opt);
      }
      if (data.default_lora_checkpoint) loraCheckpointSel.value = data.default_lora_checkpoint;

      metaEl.textContent = `config: ${data.config_path} | device: ${data.device}`;
      updateModeUI();
    }

    function updateModeUI() {
      const mode = modeSel.value;
      loraCheckpointRow.style.display = mode === 'lora' ? '' : 'none';
      loraSpecialTokenRow.style.display = mode === 'lora' ? '' : 'none';
    }

    async function runGenerate() {
      runBtn.disabled = true;
      statusEl.textContent = '推理中...';
      outputEl.textContent = '';
      inputTokenIdsEl.textContent = '';
      outputTokenIdsEl.textContent = '';
      try {
        const mode = modeSel.value;
        let loraCheckpoint = '';
        if (mode === 'lora') {
          loraCheckpoint = loraCheckpointSel.value;
          if (!loraCheckpoint) throw new Error('请选择 LoRA checkpoint');
        }
        const payload = {
          mode: mode,
          checkpoint: checkpointSel.value,
          lora_checkpoint: loraCheckpoint,
          lora_add_special_tokens: loraAddSpecialTokens.checked,
          prompt: document.getElementById('prompt').value,
          max_seq_len: Number(document.getElementById('max_seq_len').value),
          temperature: Number(document.getElementById('temperature').value),
          top_p: Number(document.getElementById('top_p').value),
          repetition_penalty: Number(document.getElementById('repetition_penalty').value),
          greedy: document.getElementById('greedy').checked,
        };
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'request failed');
        outputEl.textContent = data.text;
        inputTokenIdsEl.textContent = JSON.stringify(data.input_token_ids || []);
        outputTokenIdsEl.textContent = JSON.stringify(data.output_token_ids || []);
        statusEl.textContent = `完成，耗时 ${data.elapsed_sec.toFixed(2)}s`;
      } catch (err) {
        statusEl.textContent = `失败: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    }

    runBtn.addEventListener('click', runGenerate);
    modeSel.addEventListener('change', updateModeUI);
    loadCheckpoints();
  </script>
</body>
</html>
"""


@dataclass
class LoadedModel:
    checkpoint: str
    lora_checkpoint: str
    model: torch.nn.Module


class TinyLMService:
    def __init__(self, config_path: str, device: str = "cpu"):
        self.config_path = os.path.abspath(config_path)
        self.device = device
        self.train_config, self.model_config, _ = load_train_config(self.config_path)
        self.model_config.pytorch_impl = False
        self.tokenizer = get_tokenizer(self.train_config.tokenizer_path)
        self.checkpoint_dir = os.path.abspath(
            os.path.join(self.train_config.checkpoint_base_dir, self.train_config.project_name)
        )
        self.lora_checkpoint_dir: Optional[str] = None
        self.lora_r: Optional[int] = None
        self.lora_base_model_checkpoint: Optional[str] = None
        self.lora_full_finetune: bool = False
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        lora_cfg = raw_cfg.get("training", {}).get("lora", {})
        if isinstance(lora_cfg, dict):
            lora_base_dir = lora_cfg.get("checkpoint_base_dir", "")
            if lora_base_dir:
                self.lora_checkpoint_dir = os.path.abspath(os.path.join(lora_base_dir, self.train_config.project_name))
            r = lora_cfg.get("r", None)
            if isinstance(r, int) and r > 0:
                self.lora_r = r
            base_model_checkpoint = lora_cfg.get("base_model_checkpoint", "")
            if base_model_checkpoint:
                self.lora_base_model_checkpoint = os.path.abspath(base_model_checkpoint)
            self.lora_full_finetune = bool(lora_cfg.get("full_finetune", False))
        self._model_lock = threading.Lock()
        self._loaded: Optional[LoadedModel] = None

    def _build_lora_configs(self) -> list[LoRAConfig]:
        if self.lora_r is None or self.lora_r <= 0:
            raise ValueError("Invalid lora.r in config for lora mode")
        r = self.lora_r
        cfg = self.model_config
        lora_configs: list[LoRAConfig] = []
        for i in range(cfg.num_layers):
            for component in ["linear_q", "linear_k", "linear_v", "linear_out"]:
                b = torch.zeros(cfg.d_model, r, device=self.device)
                a = torch.zeros(r, cfg.d_model, device=self.device)
                init_std = 2 / (r + cfg.d_model)
                torch.nn.init.trunc_normal_(b, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
                torch.nn.init.trunc_normal_(a, mean=0.0, std=init_std, a=-3 * math.sqrt(init_std), b=3 * math.sqrt(init_std))
                lora_configs.append(LoRAConfig(f"transformer_layers.{i}.multi_head_attn.{component}", b, a))
        return lora_configs

    def _scan_checkpoints(self) -> list[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return []

        valid: list[str] = []
        for dirpath, _, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file.endswith(".cpt"):
                    valid.append(os.path.abspath(os.path.join(dirpath, file)))

        valid.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return valid

    def _scan_lora_checkpoints(self) -> list[str]:
        if not self.lora_checkpoint_dir or not os.path.isdir(self.lora_checkpoint_dir):
            return []

        checkpoints: list[str] = []
        for dirpath, _, files in os.walk(self.lora_checkpoint_dir):
            for file in files:
                if file.endswith(".cpt"):
                    checkpoints.append(os.path.abspath(os.path.join(dirpath, file)))

        checkpoints.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return checkpoints

    def _checkpoint_looks_compatible(self, checkpoint_path: str) -> bool:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model_state = state.get("model", state)
            emb = model_state.get("embedding.embedding_matrix")
            if emb is not None and len(emb.shape) == 2 and emb.shape[1] != self.model_config.d_model:
                return False
            return True
        except Exception:
            return False

    def checkpoints(self) -> dict[str, Any]:
        ckpts = [p for p in self._scan_checkpoints() if self._checkpoint_looks_compatible(p)]
        lora_ckpts = self._scan_lora_checkpoints()
        default_ckpt = ckpts[0] if ckpts else None
        default_lora_checkpoint = lora_ckpts[0] if lora_ckpts else None

        return {
            "config_path": self.config_path,
            "device": self.device,
            "default_checkpoint": default_ckpt,
            "default_lora_checkpoint": default_lora_checkpoint,
            "checkpoints": [
                {
                    "path": p,
                    "label": os.path.relpath(p, "/root/autodl-tmp/TinyLM"),
                }
                for p in ckpts
            ],
            "lora_checkpoints": [
                {
                    "path": p,
                    "label": os.path.relpath(p, "/root/autodl-tmp/TinyLM"),
                }
                for p in lora_ckpts
            ],
        }

    def _ensure_model(self, checkpoint: str, lora_checkpoint: str = "") -> torch.nn.Module:
        checkpoint = os.path.abspath(checkpoint) if checkpoint else ""
        lora_checkpoint = os.path.abspath(lora_checkpoint) if lora_checkpoint else ""
        with self._model_lock:
            effective_checkpoint = checkpoint
            if lora_checkpoint and not self.lora_full_finetune:
                if not self.lora_base_model_checkpoint:
                    raise ValueError("lora mode requires training.lora.base_model_checkpoint in config")
                effective_checkpoint = self.lora_base_model_checkpoint

            if (
                self._loaded is not None
                and self._loaded.checkpoint == effective_checkpoint
                and self._loaded.lora_checkpoint == lora_checkpoint
            ):
                return self._loaded.model

            if lora_checkpoint:
                if self.lora_full_finetune:
                    # In full_finetune mode, lora checkpoint is a full model checkpoint.
                    model = init_model(self.model_config)
                    load_checkpoint(lora_checkpoint, model, None)
                    model = model.to(self.device)
                else:
                    model = init_model_from_checkpoint(self.model_config, self.lora_base_model_checkpoint)
                    model = model.to(self.device)
                    lora_configs = self._build_lora_configs()
                    model.adapt_lora(lora_configs)
                    try:
                        load_lora_trainable_checkpoint(lora_checkpoint, model)
                    except Exception:
                        # Backward compatibility for old LoRA checkpoint format.
                        loaded_lora = load_lora_configs(lora_checkpoint)
                        loaded_map = {cfg.module_name: cfg for cfg in loaded_lora}
                        for cfg in lora_configs:
                            if cfg.module_name in loaded_map:
                                cfg.b.copy_(loaded_map[cfg.module_name].b.to(cfg.b.device))
                                cfg.a.copy_(loaded_map[cfg.module_name].a.to(cfg.a.device))
            else:
                if not checkpoint:
                    raise ValueError("checkpoint is required in base mode")
                model = init_model_from_checkpoint(self.model_config, checkpoint)
                model = model.to(self.device)
            model.eval()
            self._loaded = LoadedModel(checkpoint=effective_checkpoint, lora_checkpoint=lora_checkpoint, model=model)
            return model

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, input_token_ids: list[int], repetition_penalty: float) -> torch.Tensor:
        if repetition_penalty <= 1.0 or len(input_token_ids) == 0:
            return logits
        token_index = torch.tensor(list(set(input_token_ids)), dtype=torch.long, device=logits.device)
        selected = logits[token_index]
        logits[token_index] = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
        return logits

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _get_eos_token_id(self) -> Optional[int]:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return None

    def _predict_next(
        self,
        input_token_ids: list[int],
        model: torch.nn.Module,
        temperature: float,
        top_p: float,
        greedy: bool,
        repetition_penalty: float,
        kv_cache=None,
    ) -> int:
        input_token_ids_tensor = torch.tensor(input_token_ids, device=self.device)
        with torch.no_grad():
            output = model(input_token_ids_tensor, kv_cache=kv_cache)
            output = output / max(temperature, 1e-5)
            logits = output[-1]
            logits = self._apply_repetition_penalty(logits, input_token_ids, repetition_penalty)
            if greedy:
                return int(torch.argmax(logits).item())

        prob = torch.softmax(logits, dim=-1)
        prob_map = [(float(prob[i].item()), i) for i in range(prob.shape[-1])]
        prob_map.sort(reverse=True)

        accumulated_prob = 0.0
        end_index = 0
        for i, (p, _) in enumerate(prob_map):
            accumulated_prob += p
            if accumulated_prob >= top_p:
                end_index = i + 1
                break

        if end_index <= 0:
            return int(torch.argmax(logits).item())

        distribution = [(prob_map[i][0] / accumulated_prob, prob_map[i][1]) for i in range(end_index)]
        _, token_id = random.choices(distribution, weights=[p[0] for p in distribution], k=1)[0]
        return int(token_id)

    def generate(
        self,
        checkpoint: str,
        lora_checkpoint: str,
        prompt: str,
        max_seq_len: int,
        temperature: float,
        top_p: float,
        greedy: bool,
        repetition_penalty: float,
    ) -> dict[str, Any]:
        model = self._ensure_model(checkpoint, lora_checkpoint=lora_checkpoint)
        input_token_ids = self._encode_text(prompt)
        if not input_token_ids:
            raise ValueError("Prompt encoded to empty token ids")
        token_ids = list(input_token_ids)

        eos_token_id = self._get_eos_token_id()

        while len(token_ids) < max_seq_len and (eos_token_id is None or token_ids[-1] != eos_token_id):
            next_id = self._predict_next(
                token_ids,
                model,
                temperature=temperature,
                top_p=top_p,
                greedy=greedy,
                repetition_penalty=repetition_penalty,
                kv_cache=None,
            )
            token_ids.append(next_id)

        return {
            "text": self.tokenizer.decode(token_ids),
            "input_token_ids": input_token_ids,
            "output_token_ids": token_ids[len(input_token_ids):],
            "full_token_ids": token_ids,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: TinyLMService

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/checkpoints":
            self._send_json(HTTPStatus.OK, self.service.checkpoints())
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            checkpoint = str(payload.get("checkpoint", "")).strip()
            mode = str(payload.get("mode", "base")).strip().lower()
            lora_checkpoint = str(payload.get("lora_checkpoint", "")).strip()
            lora_add_special_tokens = bool(payload.get("lora_add_special_tokens", True))
            prompt = str(payload.get("prompt", ""))
            if mode == "lora":
                if not lora_checkpoint:
                    raise ValueError("lora mode requires lora_checkpoint")
                if lora_add_special_tokens:
                    prompt = f"<|user|>{prompt}<|assistant|>"
            if mode != "lora" and not checkpoint:
                raise ValueError("checkpoint is required")
            if not prompt:
                raise ValueError("prompt is required")

            max_seq_len = int(payload.get("max_seq_len", 32))
            temperature = float(payload.get("temperature", 1.0))
            top_p = float(payload.get("top_p", 0.9))
            greedy = bool(payload.get("greedy", True))
            repetition_penalty = float(payload.get("repetition_penalty", 1.0))

            started = time.time()
            result = self.service.generate(
                checkpoint=checkpoint,
                lora_checkpoint=lora_checkpoint if mode == "lora" else "",
                prompt=prompt,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_p=top_p,
                greedy=greedy,
                repetition_penalty=repetition_penalty,
            )
            elapsed_sec = time.time() - started
            result["elapsed_sec"] = elapsed_sec
            self._send_json(HTTPStatus.OK, result)
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})


class ForwardRequestHandler(BaseHTTPRequestHandler):
    target_host: str = "127.0.0.1"
    target_port: int = 6007

    def _forward(self) -> None:
        body = b""
        if self.command in ("POST", "PUT", "PATCH"):
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length > 0 else b""

        target_url = f"http://{self.target_host}:{self.target_port}{self.path}"
        req = urlrequest.Request(target_url, data=body if body else None, method=self.command)

        for key, value in self.headers.items():
            lower_key = key.lower()
            if lower_key in ("host", "content-length", "connection"):
                continue
            req.add_header(key, value)

        try:
            with urlrequest.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    lower_key = key.lower()
                    if lower_key in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urlerror.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "text/plain; charset=utf-8"))
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            msg = f"Forward error: {e}".encode("utf-8")
            self.send_response(HTTPStatus.BAD_GATEWAY)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_PUT(self):
        self._forward()

    def do_DELETE(self):
        self._forward()

    def do_PATCH(self):
        self._forward()

    def do_OPTIONS(self):
        self._forward()


def start_forward_server(listen_host: str = "0.0.0.0", listen_port: int = 6006,
                         target_host: str = "127.0.0.1", target_port: int = 6007) -> None:
    ForwardRequestHandler.target_host = target_host
    ForwardRequestHandler.target_port = target_port
    try:
        server = ThreadingHTTPServer((listen_host, listen_port), ForwardRequestHandler)
    except OSError as e:
        print(f"[ForwardServer] failed to bind {listen_host}:{listen_port}: {e}")
        return

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[ForwardServer] http://{listen_host}:{listen_port} -> http://{target_host}:{target_port}")


def run_server(config_path: str, host: str, port: int, device: str):
    start_forward_server(listen_host=host, listen_port=6006, target_host="127.0.0.1", target_port=6007)
    service = TinyLMService(config_path=config_path, device=device)
    RequestHandler.service = service

    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"TinyLM WebUI running: http://{host}:{port}")
    print(f"Config: {os.path.abspath(config_path)}")
    print(f"Device: {device}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM WebUI")
    parser.add_argument("--config", type=str, default="/root/autodl-tmp/TinyLM/configs/train_zh.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_server(args.config, args.host, args.port, args.device)
