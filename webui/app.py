import argparse
import json
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

from tiny_lm import TransformerKVCache, get_tokenizer, init_model_from_checkpoint, load_train_config


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
        <label>Checkpoint</label>
        <select id=\"checkpoint\"></select>
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
    <p class=\"tiny\" id=\"meta\"></p>
  </div>

  <script>
    const checkpointSel = document.getElementById('checkpoint');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const metaEl = document.getElementById('meta');
    const runBtn = document.getElementById('run');

    async function loadCheckpoints() {
      const res = await fetch('/api/checkpoints');
      const data = await res.json();
      checkpointSel.innerHTML = '';
      for (const item of data.checkpoints) {
        const opt = document.createElement('option');
        opt.value = item.path;
        opt.textContent = item.label;
        checkpointSel.appendChild(opt);
      }
      if (data.default_checkpoint) checkpointSel.value = data.default_checkpoint;
      metaEl.textContent = `config: ${data.config_path} | device: ${data.device}`;
    }

    async function runGenerate() {
      runBtn.disabled = true;
      statusEl.textContent = '推理中...';
      outputEl.textContent = '';
      try {
        const payload = {
          checkpoint: checkpointSel.value,
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
        statusEl.textContent = `完成，耗时 ${data.elapsed_sec.toFixed(2)}s`;
      } catch (err) {
        statusEl.textContent = `失败: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    }

    runBtn.addEventListener('click', runGenerate);
    loadCheckpoints();
  </script>
</body>
</html>
"""


@dataclass
class LoadedModel:
    checkpoint: str
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
        self._model_lock = threading.Lock()
        self._loaded: Optional[LoadedModel] = None

    def _scan_checkpoints(self) -> list[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return []

        valid: list[str] = []
        for dirpath, _, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file.endswith(".cpt"):
                    valid.append(os.path.abspath(os.path.join(dirpath, file)))

        def sort_key(path: str) -> tuple[int, float]:
            base = os.path.splitext(os.path.basename(path))[0]
            step = -1
            if base.isdigit():
                step = int(base)
            return (step, os.path.getmtime(path))

        valid.sort(key=sort_key, reverse=True)
        return valid

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
        default_ckpt = None
        if self.train_config.checkpoint and os.path.isfile(self.train_config.checkpoint):
            configured = os.path.abspath(self.train_config.checkpoint)
            if configured in ckpts:
                default_ckpt = configured
        elif ckpts:
            default_ckpt = ckpts[0]

        return {
            "config_path": self.config_path,
            "device": self.device,
            "default_checkpoint": default_ckpt,
            "checkpoints": [
                {
                    "path": p,
                    "label": os.path.relpath(p, "/root/autodl-tmp/TinyLM"),
                }
                for p in ckpts
            ],
        }

    def _ensure_model(self, checkpoint: str) -> torch.nn.Module:
        checkpoint = os.path.abspath(checkpoint)
        with self._model_lock:
            if self._loaded is not None and self._loaded.checkpoint == checkpoint:
                return self._loaded.model

            model = init_model_from_checkpoint(self.model_config, checkpoint)
            model = model.to(self.device)
            model.eval()
            self._loaded = LoadedModel(checkpoint=checkpoint, model=model)
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
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
            try:
                return tokenizer.tokenizer.encode(text, bos=False, eos=False)
            except TypeError:
                return tokenizer.tokenizer.encode(text)
        if hasattr(tokenizer, "sp_model") and hasattr(tokenizer.sp_model, "encode"):
            return tokenizer.sp_model.encode(text)
        if hasattr(tokenizer, "encode"):
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(text)
        encoded = tokenizer(text, add_special_tokens=False)
        return encoded["input_ids"]

    def _get_eos_token_id(self) -> Optional[int]:
        tokenizer = self.tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
        if hasattr(tokenizer, "eos_id"):
            return tokenizer.eos_id
        return None

    def _predict_next(
        self,
        input_token_ids: list[int],
        model: torch.nn.Module,
        temperature: float,
        top_p: float,
        greedy: bool,
        repetition_penalty: float,
        kv_cache: Optional[TransformerKVCache],
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
        prompt: str,
        max_seq_len: int,
        temperature: float,
        top_p: float,
        greedy: bool,
        repetition_penalty: float,
    ) -> str:
        model = self._ensure_model(checkpoint)
        token_ids = self._encode_text(prompt)
        if not token_ids:
            raise ValueError("Prompt encoded to empty token ids")

        eos_token_id = self._get_eos_token_id()
        kv_cache = TransformerKVCache(self.model_config.num_layers)

        while len(token_ids) < max_seq_len and (eos_token_id is None or token_ids[-1] != eos_token_id):
            next_id = self._predict_next(
                token_ids,
                model,
                temperature=temperature,
                top_p=top_p,
                greedy=greedy,
                repetition_penalty=repetition_penalty,
                kv_cache=kv_cache,
            )
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids)


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
            prompt = str(payload.get("prompt", ""))
            if not checkpoint:
                raise ValueError("checkpoint is required")
            if not prompt:
                raise ValueError("prompt is required")

            max_seq_len = int(payload.get("max_seq_len", 32))
            temperature = float(payload.get("temperature", 1.0))
            top_p = float(payload.get("top_p", 0.9))
            greedy = bool(payload.get("greedy", True))
            repetition_penalty = float(payload.get("repetition_penalty", 1.0))

            started = time.time()
            text = self.service.generate(
                checkpoint=checkpoint,
                prompt=prompt,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_p=top_p,
                greedy=greedy,
                repetition_penalty=repetition_penalty,
            )
            elapsed_sec = time.time() - started
            self._send_json(HTTPStatus.OK, {"text": text, "elapsed_sec": elapsed_sec})
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
