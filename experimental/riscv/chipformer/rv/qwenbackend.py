"""Host backends for the Qwen coprocessor device (linux.py, QWEN_BASE).

A backend is a plain callable: prompt string in, answer string out.
The device is deliberately backend-blind — today these are:

  * ``echo``     — deterministic, offline; proves the plumbing end to end.
  * ``lmstudio`` — whatever model is *already loaded* in LM Studio
                   (auto-detected via /v1/models; never triggers a load).

The endgame backend is the Qwenputer checkpoint's own language layers
(the same model whose chip layers run the guest CPU); it drops in behind
this same callable when the in-weights Linux run lands, with no guest or
device change. See docs/linux.md.
"""

import json
import os
import urllib.request


def echo_backend(prompt: str) -> str:
    return f"qwen-echo: {prompt}"


def lmstudio_backend(base_url: str | None = None, proxy: str | None = None):
    """Return a backend that asks the model already loaded in LM Studio.

    base_url: LM Studio root (default env QWEN_LM_URL, else
    http://127.0.0.1:1234 — right when the runner sits next to LM Studio,
    e.g. on the Mac). proxy: explicit HTTP proxy (env QWEN_LM_PROXY); from
    the VM the tailnet route is the :3130 variant of HTTPS_PROXY.
    """
    base = (base_url or os.environ.get("QWEN_LM_URL")
            or "http://127.0.0.1:1234").rstrip("/")
    proxy = proxy or os.environ.get("QWEN_LM_PROXY")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy else urllib.request.ProxyHandler({}))

    def call(path: str, payload: dict | None = None, timeout: float = 30) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read())

    models = call("/v1/models", timeout=15).get("data") or []
    if not models:
        raise RuntimeError(f"no model loaded in LM Studio at {base}")
    model = models[0]["id"]

    def ask(prompt: str) -> str:
        out = call("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.environ.get("QWEN_TEMP", "0.7")),
            "max_tokens": int(os.environ.get("QWEN_MAX_TOKENS", "220")),
        }, timeout=600)
        return out["choices"][0]["message"]["content"].strip() or "[qwen: empty answer]"

    ask.model = model  # for the runner's log line
    return ask


def make_backend(name: str, lm_url: str | None = None):
    if name == "echo":
        return echo_backend
    if name == "lmstudio":
        return lmstudio_backend(lm_url)
    raise ValueError(f"unknown qwen backend {name!r}")
