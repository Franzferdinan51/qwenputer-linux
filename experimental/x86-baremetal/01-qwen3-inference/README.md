# Phase 1: Qwen3-0.6B Inference in a 14 KB Pure x86-64 ELF

A complete inference engine for **Qwen/Qwen3-0.6B** written in **100% pure x86-64 NASM assembly** for Linux.

- **14,712 bytes** stripped static ELF executable (`FAST_Q8=1`, default) or **14,008 bytes** (`FAST_Q8=0`, legacy FP32-activation mode).
- **Zero runtime dependencies:** no `gcc`, no `libc`, no `libm`, no dynamic linker, no `.so` files.
- **Full Qwen3 architecture:** GQA (`16Q / 8KV`, `head_dim = 128`), per-head QK-Norm (`RMSNorm` on Q and K before RoPE), split-half NeoX RoPE (`rope_theta = 1e6`), SwiGLU MLP, tied embeddings / LM head, persistent multi-turn KV-cache (`--chat`), greedy and temperature/top-p sampling.
- **Built-in BPE tokenizer in assembly:** Fibonacci-hashed open-addressing merge table (`524,288` slots, 100% token-exact match with HuggingFace `AutoTokenizer`), UTF-8 prompt encoding (English, Cyrillic, code, numbers), Qwen3 chat template (default no-think and `--think` reasoning mode), and raw `write()` streaming to stdout.
- **`FAST_Q8=1` INT8xINT8 AVX2 + F16C engine + zero-libc thread pool:** quantizes activations to Q8_0 once per matmul, runs a 1-row sequential 4-block (`128B`) `vpabsb`+`vpsignb`+`vpmaddubsw`+`vpmaddwd` kernel with monotonic next-row prefetch, auto-detects `.asmbin` v2 FP16 scales (`vcvtph2ps`) and legacy v1 FP32 scales, runs parallel GQA attention over a contiguous head-major KV-cache, and synchronizes workers via a hybrid spin-then-futex lock-free thread pool (`0%` idle CPU in `--chat`). **126.8 tok/s** peak decode (**20.7 tok/s** 1T) on AMD EPYC 7642 — **+18% to +36% faster than `llama.cpp` CPU (`Q8_0`) across `1..40T`**, bit-exact across all thread counts.

---

## Prerequisites

- Linux x86-64 CPU with **AVX2 + FMA + F16C** (Intel Haswell / AMD Excavator or newer; no AVX-512 or VNNI required).
- Build tools:
  ```bash
  sudo apt install nasm binutils
  ```
- Offline weight export (one-time, Python only; never needed at runtime):
  ```bash
  pip install numpy huggingface_hub
  ```

---

## Step-by-Step Guide

### 1. Download `Qwen/Qwen3-0.6B`

Using the Hugging Face CLI (`hf` or `huggingface-cli`):

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B \
  config.json model.safetensors tokenizer.json \
  --local-dir models/qwen3-0.6b-bf16
```

*(If you have the older CLI installed, replace `hf download` with `huggingface-cli download`.)*

### 2. Export Weights to `.asmbin` (Q8_0)

Convert the BF16 safetensors and `tokenizer.json` into a single 64-byte aligned, directly `mmap()`-able `.asmbin` file:

```bash
# Compact v2 format with IEEE-754 FP16 block scales (default, 613.66 MiB — saves ~35.5 MiB/tok RAM bandwidth)
python3 scripts/export_weights.py \
  --model-dir models/qwen3-0.6b-bf16 \
  --out models/qwen3-0.6b-q8_0.asmbin \
  --scale-dtype fp16

# Legacy v1 format with FP32 block scales (649.18 MiB)
python3 scripts/export_weights.py \
  --model-dir models/qwen3-0.6b-bf16 \
  --out models/qwen3-0.6b-q8_0-fp32s.asmbin \
  --scale-dtype fp32
```

Both formats are auto-detected at load time (`model_init`) by `HDR_VERSION` / `HDR_FLAGS` in both `FAST_Q8=1` and `FAST_Q8=0` builds.

### 3. Build the 14 KB Pure x86-64 Binary

From the repository root, choose the compile-time `FAST_Q8` mode (`FAST_Q8=1` is the default):

```bash
# Default fast path: INT8xINT8 AVX2+F16C matmul + parallel GQA attention + contiguous KV-cache (14,712 B)
make inference test-math FAST_Q8=1
make check

# Legacy reference path: exact Q8_0 x FP32 activations (14,008 B)
make inference test-math FAST_Q8=0
```

Expected ELF inspection output (`FAST_Q8=1`):

```text
-rwxr-xr-x 1 user user 14712 build/asm-llm
build/asm-llm: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, stripped
There is no dynamic section in this file.
```

### 4. Run Inference

```bash
# Default Qwen3 chat (direct no-think answer, greedy, 16 threads)
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Explain how a CPU cache works in 3 concise sentences." -n 128 -t 16

# Interactive multi-turn chat preserving KV-cache across turns (/clear, /think, /nothink, /quit)
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin --chat -t 16

# Enable Qwen3 <think> reasoning mode
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "What is 17 * 23?" -n 512 -t 16 --think

# Temperature + top-p sampling
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Write a one-sentence motto for a systems programmer." \
  -n 64 -t 16 --temp 0.6 --topp 0.95 --seed 1337

# Russian / raw UTF-8 prompt and output
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Объясни в двух предложениях, как работает стек вызовов в процессоре." -n 128 -t 16
```

---

## CLI Reference

| Flag | Default | Description |
|---|---:|---|
| `-m <path>` | *(required)* | Path to `.asmbin` weight file |
| `-p <text>` | built-in | User prompt (UTF-8) |
| `-n <int>` | `128` | Maximum generated tokens |
| `-t <int>` | `32` | Worker thread count (`clone()` + `futex()` pool) |
| `-c <int>` | `2048` | KV-cache context length |
| `--chat` | off | Interactive multi-turn chat via raw Linux `read(0)`/`write(1)` syscalls; preserves KV-cache across turns, supports `/clear` (`/new`, `/reset`), `/think`, `/nothink`, `/quit` (`/exit`, Ctrl-D), and prints `ctx=X/2048` |
| `--think` | off | Enable Qwen3 `<think>...</think>` reasoning mode |
| `--raw` | off | Skip Qwen3 chat template, encode prompt bytes directly |
| `--tokens <csv>` | none | Feed comma-separated token IDs directly |
| `--temp <float>` | `0.0` | Sampling temperature (`0.0` = greedy argmax) |
| `--topp <float>` | `0.95` | Nucleus (top-p) sampling threshold |
| `--seed <uint>` | `42` | PRNG seed for sampling |
| `--dump-logits <path>` | none | Dump final prefill logits (`[vocab_size]` FP32) to file |
| `--print-tokens` | off | Print `PROMPT_TOKENS:` and `GEN_TOKENS:` CSV lines to stderr |
| `-h`, `--help` | - | Print usage and exit |

---

## `.asmbin` Binary Format (v2 FP16 Scales & v1 FP32 Scales)

Designed for zero-copy `mmap()` with **64-byte alignment** for every section, matrix, and row:

- **Header (128 bytes):** magic `"ASML"`, version `2` (FP16 scales, `HDR_FLAG_FP16_SCALES = 4`) or `1` (legacy FP32 scales), model dimensions (`dim=1024`, `hidden_dim=3072`, `n_layers=28`, `n_heads=16`, `n_kv_heads=8`, `head_dim=128`, `vocab_size=151936`, `rms_norm_eps=1e-6`, `rope_theta=1e6`, EOS tokens `151645`/`151643`), hash table mask, flags, and 64-byte aligned section offsets.
- **Tokenizer section (~9.5 MiB):** `byte_to_tok[256]`, `tok_offsets[vocab_size+1]`, raw UTF-8 `tok_bytes` pool (`976,309` B), and a pre-built `524,288`-slot open-addressing merge table (`151,387` merges, load factor `0.289`, avg probe `1.20`).
- **Tied Embeddings / LM Head (`[151936, 1024]` Q8_0):** verified bit-identical in BF16 (`max_diff = 0.0`), stored once (`157.65 MiB` in v2 / `166.92 MiB` in v1).
- **28 Transformer Layers (`stride = 16,720,896 B` v2 / `17,703,936 B` v1):** 4 FP32 norm vectors (`input_layernorm`, `q_norm`, `k_norm`, `post_attention_layernorm`) followed by 7 per-row split Q8_0 matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Contiguous layout allows fused QKV (`[4096, 1024]`) and gate+up (`[6144, 1024]`) matmuls in a single thread-pool barrier.
- **Per-row split Q8_0:** each row of `K` weights stores `[K int8 in -127..127][K/32 scales]`. For `K in {1024, 2048, 3072}`, row strides are `{1088, 2176, 3264}` B in v2 (`--scale-dtype fp16`) and `{1152, 2304, 3456}` B in v1 (`--scale-dtype fp32`) — all exact multiples of 64 bytes.
- **Total size:** `643,469,504` bytes (`613.66 MiB`) in v2 (`680,718,528` B / `649.18 MiB` in v1) vs `1.40 GiB` BF16 safetensors.

---

## Verification

### 1. Pure-Assembly Math & SIMD Unit Tests

```bash
python3 scripts/test_math.py build/test_math
```

| Kernel | Metric | Measured Error | Target |
|---|---|---:|---:|
| `fast_expf_avx2` | Max relative error (`[-120, +80]`) | `1.394e-07` | `< 1e-6` |
| `fast_sincosf` (sin) | Max abs error (`[-4096, +12345]`) | `2.940e-08` | `< 1e-6` |
| `fast_sincosf` (cos) | Max abs error (`[-4096, +12345]`) | `2.928e-08` | `< 1e-6` |
| `init_rope_freqs` (FP64) | Max abs error (`theta=1e6`, 64 freqs) | `4.366e-14` | `< 1e-12` |
| `rmsnorm_avx2` | Max abs error (`n=128`) | `2.056e-07` | `< 1e-6` |
| `silu_mul_avx2` | Max rel error (`[-150, +150]`, overflow-free) | `6.774e-08` | `< 1e-6` |
| `matmul_q8_0_range` (`FAST_Q8=1`, v1/v2) | Max abs error vs exact Q8_0xQ8_0 ref | `9.537e-07` / `5.960e-07` | `< 1e-5` |
| `matmul_q8_0_range` (`FAST_Q8=0`, v1/v2) | Max abs error vs exact Q8_0xFP32 ref | `9.537e-07` / `9.537e-07` | `< 1e-5` |

### 2. End-to-End Logits & 64-Token Greedy Parity

```bash
python3 scripts/verify_ref.py \
  --model models/qwen3-0.6b-q8_0.asmbin \
  --ref-dir models/qwen3-0.6b-bf16 \
  --compare-asm build/asm-llm \
  --threads 16
```

Prompt: `"Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."` (`34` prompt tokens, `64` generated tokens, `.asmbin` v2 FP16 scales):

| Comparison | Tokenizer Match | `max_abs_diff` | `mean_abs_diff` | `cosine_similarity` | Greedy Match |
|---|---|---:|---:|---:|---|
| **`asm-llm` (`FAST_Q8=1`, INT8xINT8) vs PyTorch FP32** | **Exact (34/34)** | `0.603448` | `0.099726` | **`0.9996517`** | **64/64 tokens (100%)** |
| **`asm-llm` (`FAST_Q8=1`, INT8xINT8) vs Python v2 Q8_0 ref** | **Exact (34/34)** | `0.583087` | `0.121621` | **`0.9996510`** | **64/64 tokens (100%)** |
| **`asm-llm` (`FAST_Q8=0`, Q8_0xFP32) vs Python v2 Q8_0 ref** | **Exact (34/34)** | **`0.0000153`** | **`0.0000024`** | **`1.000000000`** | **64/64 tokens (100%)** |
| Python `.asmbin` v2 (FP16 scales) vs PyTorch FP32 | **Exact (34/34)** | `0.452851` | `0.080212` | `0.9998394` | **64/64 tokens (100%)** |

All configurations (PyTorch FP32, Python Q8_0 reference, `FAST_Q8=0`, and `FAST_Q8=1`) produce the exact same 64 greedy tokens, and outputs are bit-for-bit identical across `1..40` threads.

---

## Benchmarks: `llm.asm` vs `llama.cpp` CPU (`Q8_0`, AMD EPYC 7642)

Head-to-head decode comparison on **AMD EPYC 7642** (47C online, 256 MiB L3, DDR4, CPU-only `-ngl 0`), `Qwen3-0.6B Q8_0`, `34` prompt tokens / `64` generated tokens, `3`-run mean (greedy, bit-for-bit identical across all thread counts):

| Threads | `asm-llm` `FAST_Q8=0` (v1 FP32s) | `llama.cpp` `llama-bench -ngl 0` (`Q8_0`) | **`asm-llm` `FAST_Q8=1` (v2 FP16s)** | **Speedup vs `FAST_Q8=0`** | **vs `llama.cpp`** | Determinism |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 7.18 tok/s | 16.63 ± 0.04 tok/s | **20.70 tok/s** | **2.88x** | **1.24x (+24.5%)** | Exact (64/64) |
| 4 | 28.12 tok/s | 51.08 ± 0.31 tok/s | **69.24 tok/s** | **2.46x** | **1.36x (+35.6%)** | Exact (64/64) |
| 8 | 53.09 tok/s | 80.79 ± 0.68 tok/s | **103.64 tok/s** | **1.95x** | **1.28x (+28.3%)** | Exact (64/64) |
| 16 | 77.50 tok/s | 103.03 ± 0.09 tok/s | **121.86 tok/s** | **1.57x** | **1.18x (+18.3%)** | Exact (64/64) |
| 24 | 91.88 tok/s | 104.11 ± 0.18 tok/s | **123.72 tok/s** | **1.35x** | **1.19x (+18.8%)** | Exact (64/64) |
| 32 | 96.63 tok/s | 103.82 ± 0.64 tok/s | **122.51 tok/s** | **1.27x** | **1.18x (+18.0%)** | Exact (64/64) |
| **40** | **98.50 tok/s** | **105.08 ± 0.08 tok/s** | **125.86 tok/s** *(peak 126.81)* | **1.28x** | **1.20x (+19.8%)** | **Exact (64/64)** |

*(Prefill note: `llama-bench` `pp34` uses batched GEMM (`56..1113 tok/s`), while `asm-llm` runs token-by-token GEMV (`26.7..160.8 tok/s` prefill at `1..40T`).)*

---

## Live Generation Samples (`build/asm-llm`, `-t 32`)

**1. Technical explanation:**
> *Prompt:* `Explain how a CPU cache works in 3 concise sentences.`  
> *Output (154.28 tok/s prefill, 103.40 tok/s decode):*  
> A CPU cache is a small amount of memory used to store frequently accessed data and instructions. It reduces the need for more memory by storing data that is accessed again quickly. This improves performance by minimizing the number of times data needs to be retrieved from main memory.

**2. Logic / arithmetic (direct mode):**
> *Prompt:* `If Alice has 3 apples and Bob gives her 4 more, then she eats 2, how many apples does Alice have?`  
> *Output (147.98 tok/s prefill, 90.06 tok/s decode):*  
> Let's break it down step by step:  
> - Alice starts with **3 apples**.  
> - Bob gives her **4 more apples**, so she now has: `3 + 4 = 7 apples`  
> - Then she eats **2 apples**, so the final number of apples she has is: `7 - 2 = 5`  
> **Answer:** Alice has **5 apples**.

**3. Russian UTF-8 (native UTF-8 BPE encode and stdout decode in assembly):**
> *Prompt:* `Объясни в двух предложениях, как работает стек вызовов в процессоре.`  
> *Output (149.49 tok/s prefill, 96.09 tok/s decode):*  
> Стек вызовов — это линейная последовательность вызовов, в которой каждый вызов ссылается на предыдущий, создавая цепочку выполнения. Он позволяет процессору эффективно обрабатывать задачи, обеспечивая стабильность и оптимизацию выполнения.

**4. Reasoning mode (`--think`):**
> *Prompt:* `What is 17 * 23?`  
> Breaks down `(10 + 7) * 23 = 230 + 161 = 391` and cross-checks `17 * 20 + 17 * 3 = 340 + 51 = 391` inside `<think>...</think>`.

**5. Sampling (`--temp 0.6 --topp 0.95 --seed 1337`):**
> *Prompt:* `Write a one-sentence motto for a systems programmer.`  
> *Output (146.74 tok/s prefill, 102.09 tok/s decode):*  
> A systems programmer thrives in the intersection of complexity and precision, building the foundation of innovation.
