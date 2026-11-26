# Qwenputer-X86 — the x86 bare-metal track of Qwenputer

**What this is.** Upstream [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)
(MIT) proved a Qwen3-0.6B can be the *only* program on a bare-metal x86-64
machine: a 24 KB pure-assembly Multiboot kernel, no Linux, no libc, booting in
under 100 ms to a chat REPL at 134.9 tok/s on KVM. But upstream's machine can
only *talk*.

Qwenputer — this project's RISC-V track — proved the complementary half: a
language model next to a **real computer** (exact calc, a filesystem, an
agent harness, a Linux machine) does things a chat-only model can't.

**Qwenputer-X86 lines the two up.** Same upstream kernel, verified
byte-for-byte at import, plus the Qwenputer computer layer ported into the
kernel itself — in assembly, no dependencies added:

| Command | What it does |
|---|---|
| `/calc <expr>` | Exact signed 64-bit integer arithmetic: `+ - * / %`, parentheses, unary minus, correct precedence. `/calc 123456789*987654321` → `121932631112635269`. Divide-by-zero and bad input report an error, never a crash (the `INT64_MIN / -1` idiv fault is guarded). |
| `/put <name> <text>` | Store text in the RAM filesystem (32 files × 255 B) |
| `/get <name>` | Print a stored file |
| `/del <name>` | Delete a stored file |
| `/list` / `/files` | List stored files with sizes |
| `/info` / `/about` / `/qp` | Qwenputer-X86 status |
| `/qhelp` | Qwenputer command help |

The prompt is now `qwenputer>`, the banner reads Qwenputer-X86, and `/help`
lists both upstream chat commands and the Qwenputer commands. Everything
upstream had — SMP, KV-cache multi-turn chat, `/think`, `/bench`, batch
cmdline prompts, VGA + serial — is untouched.

Kernel size: **27,132 bytes** (upstream: 24,252; the computer layer costs
2,880 bytes).

## Host tools (`qwenputer/`)

| File | What it is |
|---|---|
| `qpcalc.py` | Python oracle for `/calc` — identical grammar and 64-bit wrap/idiv semantics; the kernel is tested against it. |
| `test_qpcalc.py` | Oracle unit tests: fixed cases + 500-case fuzz vs Python big-int arithmetic. |
| `qp_serial_test.py` | Boots the real OS in QEMU (no batch prompt, so no inference runs) and exercises every Qwenputer command over serial: 16 checks. |
| `qp_agent.py` | Host bridge: serial session class, an **agentshell-style scheduler** (JSON-lines jobs file, due jobs fire into the OS), and an **agent loop** — model emits `CALC:` / `PUT:` / `GET:` lines, the kernel executes them exactly, results feed back. The RISC-V track's gadget pattern, on a real x86 CPU. |

## Build & test

```bash
sudo apt install nasm binutils qemu-system-x86   # build tools
make os FAST_Q8=1                                # 27,132 B kernel
make os-check

# model (one-time, ~614 MiB)
python3 -m venv venv && venv/bin/pip install numpy safetensors huggingface_hub
venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-0.6B', allow_patterns=['*.safetensors','tokenizer.json','config.json'], \
  local_dir='models/qwen3-0.6b-bf16')"
venv/bin/python scripts/export_weights.py --model-dir models/qwen3-0.6b-bf16 \
  --out models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16

python3 qwenputer/test_qpcalc.py                 # oracle tests
python3 qwenputer/qp_serial_test.py --model models/qwen3-0.6b-q8_0.asmbin \
  --log boot-logs/qp-serial-test.out             # boots the OS, 16 checks
make os-run MODEL=models/qwen3-0.6b-q8_0.asmbin SMP=16   # chat + computer
```

## Honest status

- Import baseline verified 2026-11-25: upstream builds at exactly its
  published sizes (OS 24,252 B, inference 14,712 B, trainer 32,944 B).
- Qwenputer kernel builds at 27,132 B; command behaviour is verified by
  `qp_serial_test.py` in QEMU (TCG on the build VM — no inference runs in
  that test; KVM hosts run the full chat at upstream speeds).
- Chat parity smoke, 2026-11-25: a batch prompt under TCG produced
  token-identical output on this kernel and on a pristine upstream build
  (25 prefill tokens, 1 decode token, same generated text) — the computer
  layer does not perturb inference.
- Chat inference itself is upstream's code, unchanged; its parity suite
  (`scripts/verify_os.py`) still applies.
- Not yet: persistence (the RAM filesystem dies at poweroff — same as the
  KV-cache), larger models (the transformer is hard-wired to Qwen3-0.6B
  geometry), framebuffer graphics. See `UPGRADES.md` for the roadmap and
  how this track relates to the RISC-V in-weights track.

## Verified hosts

Built and boot-tested on three machines on 2026-11-25 — Muse VM (TCG),
Mac Studio M5 Max (TCG, 4.28 tok/s decode), and Robin, a Ryzen 9 4900H
mini-PC, under real KVM via Docker: **56.2 tok/s decode / 72.9 tok/s
prefill**, matching upstream's published live figure with the Qwenputer
layer compiled in. The exported model file is byte-identical on all
three hosts. Full table, logs and host runners (`run_mac_test.sh`,
`run_robin_test.sh`, `Dockerfile.qptest`): see `UPGRADES.md` →
"Verified hosts".

## v1.1: Qwen3-1.7B + agent framework v2

The same kernel also runs **Qwen3-1.7B** (geometry comes from the model
header; bind the 1.7B `.asmbin` and give QEMU `-m 4G`). The exporter
handles sharded safetensors now. `qp_agent.py` is a full agent framework:
six tools, scheduler (raw + agent jobs, one-shot/repeating), session
transcripts, and persistent memory synced through the kernel FS —
verified end-to-end on Robin under KVM (see `UPGRADES.md`).

## v1.2: agent framework v3 (mined from Hermes)

`qp_agent.py` v3 adds the Hermes patterns that fit: frozen memory
snapshot in the prompt, a TODO plan tool, SKILL.md skills (catalog in
prompt, body on demand — two starter skills in `qwenputer/skills/`),
SUB delegation to a nested agent task, and transcript search. All
verified live on Robin with the 1.7B; see `UPGRADES.md`.

## v1.3 — LFM3-1.7B (second architecture)

The same kernel now also runs LiquidAI's LFM3-1.7B (hybrid conv/attention)
via `.asmbin` v3 + a dedicated asm forward pass. Raw greedy output is
token-identical to the official weights (32/32, both test prompts) on
Robin under KVM at 18.5 tok/s decode. Exporter: `scripts/export_lfm3.py`.
Details and honest limits (agent driving is weaker than Qwen3-1.7B):
UPGRADES.md, "Shipped in v1.3".
