"""Host backends for the Qwen coprocessor device (linux.py, QWEN_BASE).

A backend is a plain callable: prompt string in, answer string out.
The device is deliberately backend-blind — today these are:

  * ``echo``     — deterministic, offline; proves the plumbing end to end.
  * ``lmstudio`` — whatever model is *already loaded* in LM Studio
                   (auto-detected via /v1/models; never triggers a load).
  * ``builtin``  — the preconfigured seat for the harness: **Qwen 3.5 4B,
                   the model whose own weights host this machine's CPU**,
                   resolved by family from LM Studio's model list and
                   pinned (a pristine/base build wins over fine-tunes
                   when several match). This is the harness's default:
                   hermes-agent's brain is the built-in weights, not
                   whatever happens to be loaded. ``QWEN_MODEL`` pins an
                   exact model id instead, and ``QWEN_FAMILY`` overrides
                   the family substring — the two knobs that keep a
                   fused/GGUF build seated under its new name.

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


def lmstudio_backend(base_url: str | None = None, proxy: str | None = None,
                     model: str | None = None):
    """Return a backend that asks a model in LM Studio.

    base_url: LM Studio root (default env QWEN_LM_URL, else
    http://127.0.0.1:1234 — right when the runner sits next to LM Studio,
    e.g. on the Mac). proxy: explicit HTTP proxy (env QWEN_LM_PROXY); from
    the VM the tailnet route is the :3130 variant of HTTPS_PROXY.
    model: pin an exact model id (or env QWEN_MODEL); without a pin the
    first listed model is used (the original auto-detect behavior).
    A pin of "auto" (the dashboard's word) also means unpinned.
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
        try:
            with opener.open(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            raise RuntimeError(f"LM Studio request to {base}{path} failed: {e}")

    ids = [m.get("id") for m in (call("/v1/models", timeout=15).get("data") or [])
           if m.get("id")]
    if not ids:
        raise RuntimeError(f"no model loaded in LM Studio at {base}")
    pin = model or os.environ.get("QWEN_MODEL")
    if pin == "auto":  # the dashboard's word for "unpinned", not a model id
        pin = None
    model = pin or ids[0]
    if model not in ids:
        raise RuntimeError(f"pinned model {model!r} is not available in "
                           f"LM Studio at {base} (have: {', '.join(ids)})")
    try:
        temperature = float(os.environ.get("QWEN_TEMP", "0.7"))
        max_tokens = int(os.environ.get("QWEN_MAX_TOKENS", "220"))
    except ValueError as e:
        raise RuntimeError(f"bad QWEN_TEMP/QWEN_MAX_TOKENS in the environment: {e}")

    def ask(prompt: str) -> str:
        out = call("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, timeout=600)
        try:
            content = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeError(f"unexpected chat response from {model!r} at "
                               f"{base}: {str(out)[:200]!r}")
        return content.strip() or "[qwen: empty answer]"

    ask.model = model  # for the runner's log line
    return ask


# The built-in weights: Qwen 3.5 4B, the base model of the Qwenputer
# checkpoint — the CPU running the guest lives in these weights, so the
# harness is preconfigured to think with them too.
BUILTIN_FAMILY = "qwen3.5-4b"


def pick_builtin_model(ids, pin=None, family=BUILTIN_FAMILY):
    """Pick the seated model id from LM Studio's model list (pure: no I/O).

    An exact pin wins when it names a listed id; otherwise the family
    substring decides (a pristine/base build wins over tagged
    fine-tunes when several match). Raises RuntimeError instead of
    serving a different model — the harness must think with the
    built-in weights or say so. After a fuse/GGUF conversion the new
    build keeps working here as long as its LM Studio id still carries
    the family substring (e.g. a qwen3.5-4b-qwenputer GGUF does); under
    a wholly new name, seat it with QWEN_MODEL/QWEN_FAMILY.
    """
    ids = [i for i in ids if i]
    if pin and pin != "auto":
        if pin not in ids:
            raise RuntimeError(
                f'pinned model {pin!r} is not loaded '
                f'(have: {", ".join(ids) or "no models"})')
        return pin
    family = (family or BUILTIN_FAMILY).lower()
    fam = [i for i in ids if family in i.lower()]
    if not fam:
        raise RuntimeError(
            f"the built-in weights ({family}) are not available in "
            f"LM Studio (have: {', '.join(ids) or 'no models'}); load a "
            f"matching build first, or point QWEN_MODEL/QWEN_FAMILY at "
            f"the fused build's id")
    fam.sort(key=lambda i: ("3d-coder" in i.lower(), i))
    return fam[0]


def builtin_backend(base_url: str | None = None, proxy: str | None = None):
    """The harness's preconfigured brain: Qwen 3.5 4B in LM Studio.

    Resolves the family from the model list and pins it: a build whose
    id carries no fine-tune tag (a pristine/base Qwen 3.5 4B) wins over
    tagged fine-tunes when both are present; otherwise the first family
    match serves. Raises when no Qwen 3.5 4B is available at all — the
    harness must think with the built-in weights or say so, never
    silently fall back to a different model. QWEN_MODEL pins an exact
    id and QWEN_FAMILY overrides the family substring, so a fused or
    GGUF-converted build keeps working under its new name.
    """
    base = (base_url or os.environ.get("QWEN_LM_URL")
            or "http://127.0.0.1:1234").rstrip("/")
    proxy = proxy or os.environ.get("QWEN_LM_PROXY")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy else urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/v1/models")
    try:
        with opener.open(req, timeout=15) as r:
            data = json.loads(r.read()).get("data") or []
    except Exception as e:
        raise RuntimeError(f"LM Studio request to {base}/v1/models failed: {e}")
    ids = [m.get("id") for m in data if m.get("id")]
    try:
        choice = pick_builtin_model(
            ids, pin=os.environ.get("QWEN_MODEL"),
            family=os.environ.get("QWEN_FAMILY") or BUILTIN_FAMILY)
    except RuntimeError as e:
        raise RuntimeError(f"{e} at {base}")
    return lmstudio_backend(base_url, proxy, model=choice)


# A prompt from inside the guest harness always carries this phrase
# (agent.sh's baked-in system prompt) - the bridge uses it to tell a
# harness prompt from a terminal user's question.
AGENT_MARK = "TOOL run <command> - you will get"

_BRIDGE_HINT = (
    " [This question arrives from a terminal on a small linux machine "
    "that has its own agent, hermes-agent, which can run shell commands "
    "and use files inside that machine. If answering well needs that "
    "work done, reply with exactly one line: AGENT: <the task for "
    "hermes-agent>. You will then receive its report and can answer.]")


FETCH_MARK = "FETCHURL "
FETCH_MAX_BYTES = 4096


def fetch_page(url: str) -> str:
    """Fetch a page for the guest agent (its TOOL fetch verb).

    The guest has no network, so the host fetches on its behalf:
    http(s) only, no credentials in the URL, 15 s timeout, 4 KB cap,
    flattened to one console-safe line. Failures come back as a
    ``[fetch: ...]`` notice, never an exception, so the session
    survives a dead link.
    """
    url = (url or "").strip().split()[0] if (url or "").strip() else ""
    try:
        if not url or not url.lower().startswith(("http://", "https://")):
            return f"[fetch: refused (http(s) only): {url[:80]}]"
        if "@" in url.split("/", 3)[2]:
            return "[fetch: refused (credentials in URL)]"
        req = urllib.request.Request(
            url, headers={"User-Agent": "qwenputer-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read(FETCH_MAX_BYTES + 1)
        if len(raw) > FETCH_MAX_BYTES:
            raw = raw[:FETCH_MAX_BYTES]
        text = " ".join(raw.decode("utf-8", "replace").split())
        return text[:2000] or "[fetch: page was empty]"
    except Exception as e:
        return f"[fetch: failed: {str(e)[:160]}]"


def fetch_backend_wrapper(base):
    """Wrap a backend so ``FETCHURL <url>`` prompts are fetched by the
    host instead of asked of the model. The guest harness sends exactly
    that marker (TOOL fetch); everything else passes straight through.
    No guest or device change: the marker rides the existing QWEN:
    console protocol and needs no fastlinux.c change.
    """
    def ask(prompt: str) -> str:
        if isinstance(prompt, str) and prompt.startswith(FETCH_MARK):
            return fetch_page(prompt[len(FETCH_MARK):])
        return base(prompt)

    ask.model = getattr(base, "model", "fetch")
    return ask


def agent_bridge_backend(base):
    """Wrap a backend so the model in the seat can *use* the guest's
    hermes-agent: an ordinary question is offered the choice, and if the
    model replies ``AGENT: <task>`` the wrapper returns the
    ``[[RUN-AGENT: <task>]]`` marker instead. The guest-side ask program
    sees the marker, runs hermes-agent right there in the same Linux
    partition (same files), and asks again with a ``HERMES REPORT:``
    folded in, which this wrapper passes straight to the base model.
    Prompts from inside the harness itself always go straight to the
    base model - delegation is one level deep, never recursive.
    """
    def ask(prompt: str) -> str:
        if AGENT_MARK in prompt or "HERMES REPORT:" in prompt:
            return base(prompt)
        ans = str(base(prompt + _BRIDGE_HINT))
        flat = " ".join(ans.split())
        i = flat.find("AGENT:")
        if i >= 0:
            task = flat[i + 6:].strip()[:200]
            if task:
                return f"[[RUN-AGENT: {task}]]"
        return ans

    ask.model = getattr(base, "model", "agent-bridge")
    return ask


def make_backend(name: str, lm_url: str | None = None):
    if name == "echo":
        return echo_backend
    if name == "lmstudio":
        return lmstudio_backend(lm_url)
    if name == "builtin":
        return builtin_backend(lm_url)
    raise ValueError(f"unknown qwen backend {name!r}")
