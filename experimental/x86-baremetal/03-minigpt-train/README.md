# Phase 3: From-Scratch Mini-GPT Training & Inference in a 32 KB Pure x86-64 ELF

A complete Transformer training and autoregressive inference engine written in **100% pure x86-64 NASM assembly**. **Zero Python, zero C, zero libc, zero libm, and zero runtime dependencies** are needed to train a language model from scratch and generate text from it.

- **32,752 bytes (`32.0 KiB`)** stripped static Linux x86-64 ELF (`.text` = 27,954 B, `.rodata` = 4,173 B, `.bss` = 173,568 B `NOBITS`, zero dynamic libraries).
- **Modern Llama/GPT FP32 architecture:** auto-discovered character vocabulary, pre-norm RMSNorm, split-half RoPE (FP64-precomputed cos/sin tables), Causal Multi-Head Self-Attention, SwiGLU MLP, untied `tok_emb` and `lm_head`, and numerically stable Cross-Entropy Loss.
- **Hand-written `6x16` AVX2+FMA FP32 SGEMM:** 12 YMM accumulators, zero stack traffic in the inner `k`-loop, supporting `NN`, `TN`, and `NT` layouts with disjoint tile parallelism over our zero-libc `clone()` + `futex()` thread pool (**bit-exact** across `1T..24T`).
- **Full analytical backward pass + decoupled AdamW:** verified per-tensor against PyTorch CPU across all **21 parameter tensors** (`worst max_abs_diff = 3.35e-8`, `min cosine_sim = 1.0000000000`), global L2 gradient clipping, linear warmup + cosine LR decay, and `.gptckpt` binary checkpoints.
- **Flicker-free 80x24 pure-assembly live Braille TUI:** auto-detects interactive TTY via raw `ioctl(1, TCGETS, &termios)` and renders a `140x36` sub-pixel UTF-8 Braille loss curve, progress bar, live metrics, and Shakespeare sample box in a single buffered `write()` syscall.
- **27,662 train tok/s** peak on 24 AMD EPYC 7642 threads: `val_loss` drops from **`4.1906 -> 1.5123`** on TinyShakespeare in **4 minutes** (`245.7 s`).

---

## Prerequisites

- Linux x86-64 CPU with **AVX2 + FMA**.
- Build & dataset fetch (zero Python needed for training or inference):
  ```bash
  sudo apt install nasm binutils curl
  ```
- Optional, only if you want to run the PyTorch CPU gradient verification harness:
  ```bash
  pip install numpy torch
  ```

---

## Step-by-Step Guide

### 1. Download TinyShakespeare

```bash
mkdir -p data checkpoints
curl -fsSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/tinyshakespeare.txt
```

### 2. Build & Inspect the 32 KB Binary

From the repository root:

```bash
make train
make train-check
```

Expected ELF inspection output:

```text
-rwxr-xr-x 1 user user 32752 build/asm-gpt-train
build/asm-gpt-train: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, stripped
There is no dynamic section in this file.
```

### 3. Quick 35-Second Live Braille TUI Demo (400 Steps)

Run directly in a terminal (`ioctl` automatically enables the live 80x24 Braille TUI; add `--tui` to force or `--plain` for plain logs):

```bash
./build/asm-gpt-train \
  --data data/tinyshakespeare.txt \
  --layers 6 --dim 192 --heads 6 --hidden 512 \
  --batch 16 --seq 128 --steps 400 \
  --lr 0.002 --wd 0.15 --grad-clip 1.0 --warmup 40 --seed 42 \
  -t 16 --eval-every 40 --sample-every 100 --sample-tokens 200 \
  --prompt "ROMEO:\n" --save checkpoints/demo.ckpt
```

### 4. Full 4-Minute Convergence Run (1,500 Steps, 2.68M Params)

```bash
./build/asm-gpt-train \
  --data data/tinyshakespeare.txt \
  --layers 6 --dim 192 --heads 6 --hidden 512 \
  --batch 16 --seq 256 --steps 1500 \
  --lr 0.002 --wd 0.15 --grad-clip 1.0 --warmup 100 --seed 42 \
  -t 16 --eval-every 100 --sample-every 300 --sample-tokens 260 \
  --prompt "ROMEO:\n" --save checkpoints/shakespeare.ckpt
```

### 5. Generate & Chat with Your Trained Checkpoint

Single generation from a saved `.ckpt`:

```bash
./build/asm-gpt-train \
  --load checkpoints/shakespeare.ckpt --sample-only \
  --prompt "KING RICHARD:\n" \
  --sample-tokens 320 --temp 0.75 --topp 0.92 --seed 1337 -t 16
```

Interactive Shakespeare prompt loop in bash:

```bash
seed=100
while read -rp "Prompt> " line; do
  [ -z "$line" ] && break
  seed=$((seed + 1))
  ./build/asm-gpt-train \
    --load checkpoints/shakespeare.ckpt --sample-only \
    --prompt "${line}\n" \
    --sample-tokens 240 --temp 0.75 --topp 0.92 --seed "$seed" -t 16
done
```

### 6. Optional: Verify Gradients & AdamW Against PyTorch CPU

```bash
CUDA_VISIBLE_DEVICES="" python3 scripts/verify_train_ref.py \
  --bin build/asm-gpt-train \
  --data data/tinyshakespeare.txt
```

---

## Live Pure-Assembly 80x24 Braille TUI

Captured from a real PTY (exact 80x24 cells, rendered via a single buffered `write()` syscall per update, restores cursor cleanly on exit):

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│ asm-gpt-train │ 100% Pure x86-64 Assembly (Zero libc, AVX2+FMA)              │
│ L=6 D=192 H=6 FF=512 V=65 │ params=2681664 │ B=16 T=128 │ threads=20         │
├──────────────────────────────────────────────────────────────────────────────┤
│ Loss Curve (0..400)    ⣿⣿⣿ train   ◆ val    best_val=1.8399                  │
│ 4.34 ┤◆   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·  │
│      │⡇                                                                      │
│ 3.61 ┤⢱   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·  │
│      │ ⢱                                                                     │
│ 2.87 ┤· ⠣⣀·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·  │
│      │    ⠑⠢⠤⣀◆                                                              │
│ 2.14 ┤·   ·   ·⠉⠉⠑⠒⠒⠢⠤⠤◆⠤⣀⣀⣀⡀⣀⣀◆  ·   · ◆ ·   ·   ·   ·   ·   ·   ·   ·   ·  │
│      │                      ⠈  ⠉⠉⠉⠊⠉⠊⠒⠊⠒⠊⠒⠔⠒⠢⠤⠤⠔⠢◆⠤⠤⠤⢄⣀⣀⢄◆⣀⣀⣀⣀⡀⣀⣀⢀◆⡀ ⡀⢀   ⣀◆ │
│ 1.40 ┤·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ⠈  ⠁⠁⠈⠉⠈⠁⠉⠉⠉·⠈ │
│      └step 0 ───────────────────────── 200 ────────────────────────── 400    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Step 400/400 [████████████████████████████] 100% │ elapsed 35.2s             │
│ loss=1.6604  val=1.8399  best=1.8399  gnorm=0.53  lr=0.000200  25470 tok/s   │
├─ Live Sample (step 400) ─────────────────────────────────────────────────────┤
│ ROMEO:                                                                       │
│ Nay, to lond, in if as my kears, some with death.                            │
│                                                                              │
│ DUKE VINCENTIO:                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## CLI Reference

| Flag | Default | Description |
|---|---:|---|
| `--data <path>` | *(required to train)* | Input UTF-8/ASCII text corpus (auto-split 90% train / 10% val) |
| `--layers <int>` | `6` | Number of Transformer blocks (`L`) |
| `--dim <int>` | `192` | Model hidden dimension (`D`) |
| `--heads <int>` | `6` | Attention heads (`H`, must divide `D`) |
| `--hidden <int>` | `512` | SwiGLU MLP intermediate width (`FF`) |
| `--batch <int>` | `16` | Batch size (`B`) |
| `--seq <int>` | `128` | Context / sequence length (`T`) |
| `--steps <int>` | `1500` | Total optimization steps |
| `--lr <float>` | `0.001` | Peak learning rate (linear warmup + cosine decay to 10%) |
| `--wd <float>` | `0.1` | Decoupled AdamW weight decay (2D matrices only, 0.0 on 1D norms) |
| `--grad-clip <float>` | `1.0` | Global L2 gradient norm clipping threshold (`0` disables) |
| `--warmup <int>` | `min(100, steps/10)` | Linear LR warmup steps |
| `--seed <uint>` | `42` | Deterministic PRNG seed (`xorshift64*` + Box-Muller init) |
| `-t <int>` | `16` | Worker thread count |
| `--tui` / `--plain` | auto | Force live 80x24 Braille TUI or plain stdout logs (`ioctl` auto-detects TTY) |
| `--log-every <int>` | `50` | Plain-mode logging interval |
| `--eval-every <int>` | `100` | Validation evaluation interval (`0` disables) |
| `--eval-batches <int>` | `20` | Number of fixed validation batches per eval |
| `--sample-every <int>` | `250` | Periodic text sampling interval (`0` disables) |
| `--sample-tokens <int>` | `200` | Characters generated per sample |
| `--prompt <text>` | `"ROMEO:\n"` | Sampling prompt (`\n` escape supported) |
| `--temp <float>` | `0.8` | Sampling temperature (`0.0` = greedy) |
| `--topp <float>` | `0.95` | Top-p nucleus sampling threshold |
| `--save <path>` | none | Save binary `.gptckpt` checkpoint after training |
| `--load <path>` | none | Load saved `.gptckpt` checkpoint (for resume or `--sample-only`) |
| `--sample-only` | off | Skip training and generate from `--load` checkpoint |
| `--dump-debug <dir>` | none | Dump initial weights, batch, step-0 grads, losses, LRs, and final weights for verification |

---

## Verification vs PyTorch FP32 CPU (`scripts/verify_train_ref.py`)

Test model: `L=2, D=64, H=4, FF=128, V=65` (`90,560` params), `B=2, T=16`, `10` consecutive AdamW steps on a fixed batch with `CUDA_VISIBLE_DEVICES=""`:

| Verification Gate | Measured (`asm-gpt-train` vs PyTorch CPU) | Target | Status |
|---|---:|---:|---|
| Forward Cross-Entropy Loss (`step 0`) | `abs_diff = 4.768e-07` (`4.2066970` vs `4.2066965`) | `< 1.0e-5` | **PASS** |
| Analytical Gradients (all **21 tensors**: `tok_emb`, `L0/L1` `ln1/wq/wk/wv/wo/ln2/wgate/wup/wdown`, `final_ln`, `lm_head`) | worst `max_abs_diff = 3.353e-08`, min `cosine_sim = 1.0000000000` | `< 1.0e-4`, `> 0.99999` | **PASS** |
| 10-Step AdamW Trajectory (`clip=1.0, lr=1e-3, wd=0.1, warmup=3`) | `max_lr_diff = 0.0`, `max_loss_diff = 4.768e-07` (steps 1-8 exact `0.0`), `final_w max_diff = 6.153e-06` (`cos = 1.0000000000`) | `< 1.0e-4` | **PASS** |
| 10-Step AdamW w/ Active Grad-Clip (`clip=0.05, lr=3e-3`) | `max_loss_diff = 4.768e-07`, `final_w max_diff = 1.865e-06` (`cos = 1.0000000000`) | `< 1.0e-4` | **PASS** |
| Multi-Thread Determinism (`-t 1` vs `-t 8` vs `-t 16`) | `loss_diff = 0.0`, `grad_diff = 0.0`, `final_w_diff = 0.0` | `< 1.0e-5` | **BIT-EXACT** |

---

## TinyShakespeare Training Curve (`2.68M` Params, `B=16, T=256`, `-t 24`, `245.72 s`)

| Step | Train Loss | Val Loss | Grad Norm | LR | Step Time | Throughput (`train tok/s`) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (init) | 4.1837 *(step 1)* | **4.1906** | 10.32 | 0.000020 | 199.11 ms | 20,571 |
| 100 | 2.1719 | 2.1748 | 0.77 | 0.002000 | 148.90 ms | 27,508 |
| 200 | 1.7754 | 1.9262 | 0.50 | 0.001978 | 151.84 ms | 26,976 |
| 300 | 1.5971 | 1.8123 | 0.44 | 0.001912 | 149.15 ms | 27,462 |
| 500 | 1.4997 | 1.6882 | 0.40 | 0.001663 | 149.75 ms | 27,352 |
| 800 | 1.3770 | 1.5929 | 0.41 | 0.001102 | 150.71 ms | 27,178 |
| 1000 | 1.2975 | 1.5537 | 0.40 | 0.000711 | 150.26 ms | 27,260 |
| 1200 | 1.2928 | 1.5310 | 0.43 | 0.000398 | 149.07 ms | 27,477 |
| 1400 | 1.2322 | 1.5174 | 0.43 | 0.000223 | 149.78 ms | 27,346 |
| **1500 (final)** | **1.2955** *(min 1.2087)* | **1.5123** | 0.47 | 0.000200 | **149.41 ms** | **27,414** *(peak 27,662)* |

---

## Text Generation Evolution (`--prompt "ROMEO:\n"`, `--temp 0.8 --topp 0.95`)

- **Step 0 (`val_loss = 4.1906`, random init):**
  > `ROMEO:`  
  > `!IeTXZpKVp`  
  > `i3u&J3`  
  > `rBbfhCcNUs`  
  > `!RUl$tV-.U&3vRt,fWzutkFXEer.;;a;wa,x'.XV-Vm,sf.XjB.qI.zvLGp.$S ?wyORAR$:Tf!L3'XFX`
- **Step 300 (`val_loss = 1.8123`, English words and verse rhythm emerge):**
  > `ROMEO:`  
  > `Your bad the tall thy find for against and good`  
  > `When thy say what head of my truposed manious`  
  > `Wellike fair to dayes sain, these hither work;`
- **Step 600 (`val_loss = 1.6768`, speaker turns and character names):**
  > `ROMEO:`  
  > `But thou hast your most far was your stick.`  
  > `SICINIUS:`  
  > `Go is no meet to the earth / Is am there too all authors.`  
  > `First Murderer:`  
  > `No with thy tongue may to an the duke,`
- **Step 1200 (`val_loss = 1.5310`, multi-character Romeo & Juliet dialogue):**
  > `ROMEO:`  
  > `If now not this father, let them to make thee.`  
  > `BENVOLIO:`  
  > `Why, sir, what I am a most learned by you.`  
  > `JULIET:`  
  > `I will not malice to fine and there's all.`
- **Checkpoint `--load checkpoints/shakespeare.ckpt --sample-only --prompt "KING RICHARD:\n"` (`--temp 0.75 --topp 0.92`):**
  > `KING RICHARD:`  
  > `Hark, come, sir, thou shalt be the face.`  
  > `KING RICHARD II:`  
  > `Why, will I am the noble suitors and home:`  
  > `Where had you before I go with her king.`  
  > `KING EDWARD IV:`  
  > `Why, ay, my lord?`  
  > `PRINCE:`  
  > `What's the love are so desire is my mother?`

---

## CPU Thread Scaling (AMD EPYC 7642, `L=6, D=192, H=6, FF=512`, `B=16, T=256`, `4,096 tok/step`)

Measured with `--eval-every 0 --sample-every 0`:

| Threads | Avg Step (ms) | Steady Step (ms) | Avg Throughput (`train tok/s`) | Steady (`train tok/s`) | Speedup vs 1T | Determinism |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2289.51 | 2262.70 | 1,789.03 | 1,810.23 | 1.00x | Baseline |
| 4 | 593.62 | 588.39 | 6,900.08 | 6,961.37 | 3.86x | **Bit-exact** |
| 8 | 304.70 | 300.71 | 13,442.92 | 13,621.23 | 7.51x | **Bit-exact** |
| 16 | 204.95 | 197.00 | 19,985.09 | 20,791.95 | 11.17x | **Bit-exact** |
| **24** | **157.33** | **152.56** *(peak 148.07)* | **26,034.64** | **26,848.36** *(peak 27,662)* | **14.55x** | **Bit-exact** |
