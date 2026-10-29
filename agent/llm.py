"""Minimal multi-provider chat client (stdlib only): OpenAI-compatible, Anthropic, replay.

OpenAI-compatible works for DeepSeek / Qwen(DashScope) / Moonshot / GLM / SiliconFlow /
vLLM / Ollama and friends — just set base_url and model, e.g.:

    https://api.deepseek.com/v1        deepseek-chat
    https://dashscope.aliyuncs.com/compatible-mode/v1   qwen-plus
    https://api.moonshot.cn/v1         kimi-k2-0905-preview
    https://open.bigmodel.cn/api/paas/v4                glm-4.6
    https://api.siliconflow.cn/v1      deepseek-ai/DeepSeek-V3
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_DEFAULT_KEY_ENVS = (
    "KERNEL_AGENT_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPUAI_API_KEY",
    "SILICONFLOW_API_KEY",
    "ANTHROPIC_API_KEY",
)


class LLMError(RuntimeError):
    pass


class ReplayModel:
    """Offline model: returns canned responses in order (used by tests and demos)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def next(self) -> str:
        if self.index >= len(self.responses):
            return json.dumps({"action": "finish", "summary": "replay responses exhausted"})
        response = self.responses[self.index]
        self.index += 1
        if isinstance(response, dict):  # allow readable JSON objects in replay files
            return json.dumps(response, ensure_ascii=False)
        return response


def resolve_api_key(provider: str = "openai", explicit: str = "", env_name: str = "") -> str:
    if explicit:
        return explicit
    names = []
    if env_name:
        names.append(env_name)
    names.extend(_DEFAULT_KEY_ENVS)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _http_json(url: str, payload: dict, headers: dict, timeout: float, retries: int = 2) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries:
                last_error = "HTTP %s: %s" % (exc.code, detail)
                time.sleep(2 ** attempt)
                continue
            raise LLMError("HTTP %s from %s: %s" % (exc.code, url, detail)) from exc
        except Exception as exc:
            if attempt < retries:
                last_error = str(exc)
                time.sleep(2 ** attempt)
                continue
            raise LLMError("request to %s failed: %s" % (url, exc)) from exc
    raise LLMError("request to %s failed: %s" % (url, last_error))


def chat(
    messages,
    system: str = None,
    provider: str = "openai",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 300,
    replay: ReplayModel = None,
) -> dict:
    if provider == "replay":
        return {"text": (replay or ReplayModel([])).next(), "usage": {}}

    if provider == "openai":
        if not base_url:
            raise LLMError("openai provider needs --base-url (e.g. https://api.deepseek.com/v1)")
        if not model:
            raise LLMError("openai provider needs --model")
        payload = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra = os.environ.get("KERNEL_AGENT_LLM_EXTRA")
        if extra:
            try:
                extra_payload = json.loads(extra)
                if isinstance(extra_payload, dict):
                    payload.update(extra_payload)
            except ValueError:
                pass
        data = _http_json(
            base_url.rstrip("/") + "/chat/completions",
            payload,
            {"Authorization": "Bearer %s" % api_key},
            timeout,
        )
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape: %s" % json.dumps(data)[:300]) from exc
        return {"text": text or "", "usage": data.get("usage", {}) or {},
                "finish_reason": choice.get("finish_reason")}

    if provider == "anthropic":
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": list(messages),
        }
        if system:
            payload["system"] = system
        data = _http_json(
            (base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages",
            payload,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout,
        )
        blocks = data.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        return {"text": text, "usage": data.get("usage", {}) or {}}

    raise LLMError("unknown provider: %r" % provider)