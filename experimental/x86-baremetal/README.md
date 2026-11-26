# llm.asm

**Three complete LLM projects written in 100% pure x86-64 NASM assembly.**
Zero C, zero C++, zero Rust, zero libc, zero libm, zero runtime dependencies.

Assembled directly with `nasm` and linked with `ld`. Every math kernel (`expf`, `sincosf`, `RMSNorm`, `RoPE`, `SiLU`, `Softmax`), AVX2+FMA+F16C GEMM/GEMV engine, BPE tokenizer, `clone()`/`futex()` thread pool, bare-metal SMP kernel, analytical backward pass, AdamW optimizer, and live Unicode Braille TUI is hand-written x86-64 assembly (`17,252` lines total across all three projects).

## The Trilogy

| Project | Binary | Size | Deps | Key Numbers |
|---|---|---:|---:|---|
| **[01-qwen3-inference/](01-qwen3-inference/)** | `build/asm-llm` | **14,712 B** *(14,008 B `FAST_Q8=0`)* | **0** | Static Linux ELF running `Qwen/Qwen3-0.6B` (Q8_0, built-in BPE tokenizer, GQA + QK-Norm + RoPE + SwiGLU, single-shot & `--chat` multi-turn KV-cache REPL). **`FAST_Q8=1` (default):** **126.8 tok/s** decode at 40T (**20.7 tok/s** at 1T), **+18% to +36% faster than `llama.cpp` CPU (`Q8_0`) across 1..40T**, **`0.99965`** cosine & **64/64** exact greedy tokens vs PyTorch FP32. |
| **[02-baremetal-os/](02-baremetal-os/)** | `build/asm-llm-os.elf` | **24,252 B** *(23,612 B `FAST_Q8=0`)* | **0** | Bootable bare-metal `Qwen3-0.6B` OS with no Linux or Windows. Multiboot 1 -> Long Mode -> AVX2/FMA/F16C -> ACPI/APIC SMP -> VGA + PS/2 + COM1 multi-turn KV-cache chat (`/clear`, `/think`, `/bench`). **74–94 ms** boot, **134.9 tok/s** KVM EPYC (**23.7 tok/s** 1 vCPU, **136.6 tok/s** peak), **55.4 tok/s** live on Proxmox Xeon E5-2670 v3 with **0 GB** disk. |
| **[03-minigpt-train/](03-minigpt-train/)** | `build/asm-gpt-train` | **32,944 B** | **0** | From-scratch FP32 Mini-GPT trainer and sampler. Register-blocked `6x16` AVX2+FMA SGEMM, full analytical backward pass (**`1.0000000000`** cosine across all 21 gradient tensors vs PyTorch CPU), decoupled AdamW, **27,662 train tok/s**, live 80x24 Braille TUI, `val_loss 4.19 -> 1.51` on TinyShakespeare in 4 min. |

---

## Head-to-Head CPU Decode Benchmark: `llm.asm` vs `llama.cpp` (`Qwen3-0.6B Q8_0`)

Measured on the same machine (**AMD EPYC 7642**, 47C online, 256 MiB L3, DDR4, CPU-only `-ngl 0`), same model (`Qwen3-0.6B Q8_0`: `613.66 MiB` `.asmbin` v2 FP16-scale vs `604.15 MiB` GGUF `Q8_0`), `34` prompt tokens / `64` generated tokens, `3`-run average:

| Threads | `llm.asm` (`FAST_Q8=0`, FP32 act) | `llama.cpp` (`Q8_0`, `-ngl 0`) | **`llm.asm` (`FAST_Q8=1`, 14.7 KB)** | **vs `llama.cpp`** | **Bare-Metal OS KVM (`FAST_Q8=1`)** |
|---:|---:|---:|---:|---:|---:|
| **1** | 7.18 tok/s | 16.63 tok/s | **20.70 tok/s** | **1.24x (+24.5%)** | **23.67 tok/s** |
| **4** | 28.12 tok/s | 51.08 tok/s | **69.24 tok/s** | **1.36x (+35.6%)** | **78.89 tok/s** |
| **8** | 53.09 tok/s | 80.79 tok/s | **103.64 tok/s** | **1.28x (+28.3%)** | **118.08 tok/s** |
| **16** | 77.50 tok/s | 103.03 tok/s | **121.86 tok/s** | **1.18x (+18.3%)** | **134.94 tok/s** |
| **24** | 91.88 tok/s | 104.11 tok/s | **123.72 tok/s** | **1.19x (+18.8%)** | **132.90 tok/s** |
| **32** | 96.63 tok/s | 103.82 tok/s | **122.51 tok/s** | **1.18x (+18.0%)** | **133.75 tok/s** |
| **40** | 98.50 tok/s | 105.08 tok/s | **125.86 tok/s** *(peak 126.81)* | **1.20x (+19.8%)** | — |

*(Note: `llama.cpp` uses batched GEMM for prompt prefill `pp34`, whereas `llm.asm` runs token-by-token GEMV. For autoregressive decode `tg64`, `llm.asm` (`FAST_Q8=1`) combines INT8xINT8 AVX2 `vpmaddubsw`/`vpmaddwd`, `.asmbin` v2 FP16 block scales (`F16C vcvtph2ps`), monotonic next-row prefetch, contiguous head-major KV-cache, and parallel GQA attention — beating `llama.cpp` by +18%..+36% across 1..40 threads while producing bit-exact deterministic outputs across all thread counts. The bare-metal OS gains another ~10% from 2 MiB huge-page identity mapping.)*

---

## Live Output & Visual Proof

### Phase 2: Booting Qwen3-0.6B Directly on Bare Metal (QEMU KVM, 16 vCPUs)

![Bare-Metal Qwen3-0.6B OS VGA Console](assets/baremetal-vga.png)

### Phase 2: Running Live on a Proxmox VE VM (0 GB Disk, Intel Xeon E5-2670 v3, 12 Cores)

![Running on Real Proxmox VE VM (0 GB Disk, Intel Xeon KVM)](assets/proxmox-vm250-vga.png)

### Phase 3: Pure-Assembly Live 80x24 UTF-8 Braille TUI (`build/asm-gpt-train`)

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

## Quick Start

Clone the repo and build everything with `make all`:

```bash
git clone https://github.com/alesha-pro/llm.asm.git
cd llm.asm
make all
```

### 1. Run Qwen3-0.6B Inference in 14 KB (`01-qwen3-inference/`)

```bash
# Export compact .asmbin v2 (FP16 block scales, 613.66 MiB; or --scale-dtype fp32 for legacy v1)
python3 scripts/export_weights.py --model-dir models/qwen3-0.6b-bf16 --out models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16
# Build with FAST_Q8=1 (default: INT8xINT8 AVX2 + parallel GQA) or FAST_Q8=0 (exact FP32 activations)
make inference FAST_Q8=1
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin -p "Explain how a CPU cache works in 3 concise sentences." -n 128 -t 16
# Or start an interactive multi-turn chat preserving the KV-cache across turns:
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin --chat -t 16
```

Full step-by-step setup, model download, CLI flags, and benchmarks: **[01-qwen3-inference/README.md](01-qwen3-inference/README.md)**.

### 2. Boot Bare-Metal Qwen3-0.6B OS in 24 KB (`02-baremetal-os/`)

```bash
make os FAST_Q8=1
make os-run MODEL=models/qwen3-0.6b-q8_0.asmbin SMP=16
# Or build a bootable BIOS+UEFI ISO for QEMU / Proxmox VE:
make os-iso MODEL=models/qwen3-0.6b-q8_0.asmbin
```

Full QEMU KVM, bootable ISO, and 0 GB disk Proxmox VE deployment guide: **[02-baremetal-os/README.md](02-baremetal-os/README.md)**.

### 3. Train Mini-GPT from Scratch in 32 KB (`03-minigpt-train/`)

Zero Python required to train or run inference:

```bash
mkdir -p data checkpoints && curl -fsSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
make train
./build/asm-gpt-train --data data/tinyshakespeare.txt --steps 400 --seq 128 -t 16 --save checkpoints/shakespeare.ckpt
```

Full training, live Braille TUI, checkpoint chat loop, and PyTorch gradient check guide: **[03-minigpt-train/README.md](03-minigpt-train/README.md)**.

---

## Repository Layout

```text
llm.asm/
├── LICENSE                        # MIT License
├── Makefile                       # Single build file: inference, os, os-iso, train, all, check
├── README.md                      # You are here
├── assets/
│   ├── baremetal-vga.png          # QEMU KVM 80x25 VGA text-mode capture (EPYC 7642, 16 cores)
│   └── proxmox-vm250-vga.png      # Real Proxmox VE 0 GB disk VM capture (Xeon E5-2670 v3, 12 cores)
├── shared/                        # Pure x86-64 NASM core reused across phases (3,857 lines)
│   ├── constants.inc              # Linux syscalls, .asmbin v1/v2 header, runtime context offsets
│   ├── syscalls.asm               # Raw Linux syscalls + string/number formatting helpers
│   ├── math.asm                   # AVX2/SSE2 expf, sincosf, FP64 RoPE, RMSNorm, Softmax, SiLU, argmax, top-p
│   ├── matmul.asm                 # FAST_Q8=1 INT8xINT8 AVX2+F16C & FAST_Q8=0 Q8_0xFP32 matmul, dequant, dot, saxpy
│   ├── threads.asm                # Zero-libc Linux clone() + hybrid spin/futex thread pool (matmul + GQA attn)
│   ├── tokenizer.asm              # Fibonacci-hashed BPE encoder, UTF-8 decoder, Qwen3 chat template
│   └── transformer.asm            # Full 28-layer Qwen3 forward pass (v1/v2 auto-detect, contiguous KV, parallel GQA)
├── 01-qwen3-inference/            # Phase 1: 14 KB static Linux ELF inference engine + --chat (1,251 lines)
│   ├── README.md
│   ├── main.asm
│   └── test_math.asm
├── 02-baremetal-os/               # Phase 2: 24 KB bare-metal bootable LLM OS (4,167 lines)
│   ├── README.md
│   ├── boot.asm                   # Multiboot 1 -> 4-level paging -> Long Mode -> AVX2/FMA -> IDT
│   ├── drivers.asm                # COM1 115200, 80x25 Color VGA, PS/2 keyboard, PIT+RDTSC, ACPI poweroff
│   ├── kernel.asm                 # Kernel entrypoint, CPUID banner, cmdline parser, multi-turn chat REPL
│   ├── linker.ld                  # Bare-metal linker script at 0x00100000
│   ├── os_constants.inc           # Hardware, Multiboot 1, Local APIC, and SMP constants
│   ├── runtime.asm                # Multiboot mmap bump allocator + bare-metal formatting runtime
│   └── smp.asm                    # ACPI MADT + 16->32->64 AP trampoline + INIT-SIPI-SIPI spin-barrier
├── 03-minigpt-train/              # Phase 3: 32 KB Mini-GPT trainer + inference engine (7,977 lines)
│   ├── README.md
│   ├── train_constants.inc        # Training context offsets, SGEMM flags, .gptckpt format
│   ├── sgemm.asm                  # Thread pool + register-blocked 6x16 AVX2+FMA FP32 SGEMM (NN/TN/NT)
│   ├── kernels.asm                # Forward/backward RMSNorm, RoPE, Causal MHA, SwiGLU, CE, AdamW, PRNG
│   ├── model.asm                  # Arena layout, deterministic init, full fwd/bwd pass, sampler
│   ├── tui.asm                    # ioctl(TCGETS) TTY detect + flicker-free 80x24 ANSI/Braille TUI
│   └── main.asm                   # CLI, auto char-vocab, 90/10 split, .gptckpt I/O, train/sample loop
└── scripts/
    ├── export_weights.py          # Offline HuggingFace Qwen3-0.6B -> .asmbin (Q8_0) converter
    ├── mkiso.sh                   # Bootable BIOS+UEFI hybrid GRUB Multiboot 1 ISO builder
    ├── test_math.py               # Automated SIMD & math kernel unit test driver
    ├── verify_ref.py              # Independent NumPy Q8_0 + PyTorch FP32 parity harness
    ├── verify_os.py               # Phase 2 parity, SMP determinism, KVM benchmark & VGA capture suite
    └── verify_train_ref.py        # Phase 3 per-tensor PyTorch CPU grad-check & AdamW trajectory suite
```

---

## System Requirements

- **CPU:** x86-64 with **AVX2 + FMA + F16C** (Intel Haswell / AMD Excavator or newer).
- **Build tools:** `nasm` and GNU `binutils` (`ld`, `objcopy`).
  ```bash
  sudo apt install nasm binutils
  ```
- **Phase 1 & 3 runtime:** Any Linux x86-64 kernel. Zero shared libraries (`readelf -d` reports no dynamic section).
- **Phase 2 runtime (optional):** `qemu-system-x86_64` with KVM, or any x86-64 hypervisor/bare-metal machine (Proxmox VE, VirtualBox, real hardware). Building bootable ISOs uses `grub-mkrescue` (`grub-pc-bin`, `grub-efi-amd64-bin`, `mtools`, `xorriso`).
- **Offline tooling (optional):** `python3` + `numpy` only for one-time `Qwen3-0.6B` `.asmbin` export and reference verification (`torch` + `transformers` only if running PyTorch parity checks).

---

## Architecture & Verification Summary

Every phase is built on the same engineering rules: no runtime libraries, deterministic parallel partitioning, and numerical verification against independent Python/PyTorch references.

- **Shared assembly core (`shared/`):** Phase 2 compiles the exact same `math.asm`, `matmul.asm`, `tokenizer.asm`, and `transformer.asm` with `-DBARE_METAL` (swapping Linux `mmap`/`clone`/`write` for physical Multiboot modules, APIC spin-barriers, and VGA+COM1 drivers). Phase 3 reuses `syscalls.asm` and `math.asm`.
- **Phase 1 verification:** All math/SIMD kernels pass unit tests within `1e-7..1e-6` (`init_rope_freqs` FP64 `< 5e-14`) for both `FAST_Q8=1` and `FAST_Q8=0` across v1 FP32 and v2 FP16 scale formats. `FAST_Q8=1` matches PyTorch FP32 `Qwen3-0.6B` with `cosine = 0.9996517` and **64/64 exact greedy tokens** (`FAST_Q8=0` matches the Python Q8_0 reference with `cosine = 1.000000000`). Bit-exact deterministic from `1T` to `40T`.
- **Phase 2 verification:** **100% token-exact and byte-for-byte identical** to Phase 1 across 5 test prompts (English, Russian UTF-8, arithmetic, `--think` mode), multi-turn KV-cache recall + `/clear`, and across `1..32` KVM vCPUs (`134.9 tok/s` at 16 vCPUs). Verified on both QEMU KVM (AMD EPYC 7642) and a real Proxmox VE VM (Intel Xeon E5-2670 v3, 0 GB disk, BIOS GRUB ISO).
- **Phase 3 verification:** Verified against PyTorch FP32 CPU on all **21 parameter tensors** (`worst max_abs_diff = 3.35e-8`, `min cosine_sim = 1.0000000000`), 10-step AdamW + cosine LR + grad-clip trajectory (`max_loss_diff = 4.77e-7`), and bit-exact determinism across `1T`, `8T`, and `16T`.

See each project's README for full tables, memory layouts, and reproduction steps:
- [01-qwen3-inference/README.md](01-qwen3-inference/README.md)
- [02-baremetal-os/README.md](02-baremetal-os/README.md)
- [03-minigpt-train/README.md](03-minigpt-train/README.md)

---

## License

MIT License. See [LICENSE](LICENSE).
