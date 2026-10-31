# The upstream catalogue — everything divagr has that we don't (yet)

Swept 2026-10-31 at Duckets' direction ("what you kept is good but
there's more"): all four branches of `divagr18/qwenputer` (main,
`npu`, `rv-store`, `vx`), the standalone `divagr18/QwenPU` repo,
all of divagr1925's Hugging Face checkpoints, the divagr.com blog
series, and the docs/research trees inside the repos.

**Baseline fact:** our fork is at parity with upstream `main` —
`core.py`, `spec.py`, `calc.py`, `bridge.py` verified byte-identical
to main's current HEAD, and everything main carries (the Linux
machine's base, calc v2 with exact rationals, batched execution,
MicroPython port, the q35 store v1, the skills plan, the
hidden-reasoning research) is already in our tree. **All of the
"more" lives on the three side branches and in QwenPU.** None of
it is in the stock checkpoint or in our grown-map recompile.

## 1. The in-weights object store, v2 — branch `rv-store` ⭐

The filesystem, finished far past what the README admits. Status
doc on the branch: phases F0–F5 **done, GPU gate reached**.

- Five native tools next to `calc`: `put / append / get / list /
  delete`. The chip layers capture the call as Qwen writes it,
  store the text in DeltaNet heads, and print the
  `<tool_response>` themselves. Nothing runs on the host.
- Content-addressed directory: a name's first 6 tokens hashed by a
  fixed GF(2) map to one of 1024 slots, linear probing,
  tombstones on delete. Lookup ≈ 2–3 chip tokens, never a scan.
- Capacity as built: **512 files, 53,248 tokens of text** (512
  chunks × 104 tokens, 14 tokens packed per row).
- **Persistence is built and GPU-verified**: `save_store` /
  `load_store` — save the store's heads at the end of a session,
  load them at the start of the next, files still there. This is
  the recursive-memory-in-weights piece.
- A merged checkpoint exists (`rv-q4store`, 22.3 GB, hidden 4608,
  12 chip layers) but was **not uploaded** — adopting means
  compiling it ourselves: `core.build(text=T, store=Config())`,
  the same compiler operation as our grown-map build.
- Benchmarks (from the main README): multi-session recall 83.3
  vs base 7.8; facts changed in a later session 90.0 vs 3.3 —
  on par with a host-run Python memory tool (78.9 / 83.3).

Store v1 exists on the CHIP-8 (q35) line in main — protocol
reference (`chipformer/q35/store.py`, `docs/q35-store.md`).

## 2. The NPU — branch `npu`, checkpoint `Qwenputer-npu`

Three instructions beside the CPU, aimed at neural-net math:

- `VDOT`: 8 int8 activations × the next 8 int4 weights, summed
  into a register. `WB` sets the weight base. `SMAQA`: four
  signed byte products added (RISC-V P-extension style).
- Weight memory (WROM): 261,120 int4 weights in 17 DeltaNet
  heads, loaded from the program — model weights are *data* the
  guest program carries, so any small net that fits can run.
- Proof workload: Karpathy's stories260K (a 260K-param Llama)
  as a C program on the in-weights CPU — 225,000 trace tokens
  per story token vs 2,430,000 in plain C (≈10.8× cheaper).
- `fusedexec.py` (see §4) runs it ≈5,500 tok/s on an L4 → ≈41 s
  per story token. `npucheck.py` verifies the instructions
  teacher-forced on a merged checkpoint's own layers.
- Checkpoint published 2026-10-30. The branch also carries the
  stories260K guest (`experimental/riscv/guests/stories260k`).

## 3. The vector unit — branch `vx` + repo `QwenPU` + checkpoint

The NPU idea taken all the way: a small transformer accelerator
compiled into the model's early layers (L1–L3), with its own
memories (16 vector registers × 64 int32, 384 int8 activation
lanes, K/V caches of 32 positions, a WROM of 4,152 row-chunks of
64 int4 weights + row scales).

- Ops: `VMAT` (int8·int4 matmul rows, or argmax), `VQ`
  (quantize), `VEW` (residual add, SiLU-gated product, RoPE),
  `VMATR` (attention scores; probabilities × V), `VEXP`
  (softmax numerators), `VRED` (sum / sum of squares),
  `VGET`/`VPUT` (CPU ↔ unit transfers). SiLU and exp are
  fixed-point function units, exact like the adders.
- Verified hard: random op programs exact against the reference;
  the stories program exact through the circuit IR for 110k
  steps and **free-running on the GPU executor for all
  3,184,920 steps**; first 30k steps also exact on the
  checkpoint's own layers (`vxcheck`).
- Speed: ≈3,300 tok/s on an L4 → a story token in ≈16 s (vs
  the NPU's 41 s).
- Cost: a **different, wider checkpoint** — hidden 2560 → 5120,
  13 chip layers, 26.7 GB bf16. Does not fit a 16 GB card
  whole, and a tune of stock Qwen3.5-4B does **not** transplant
  into it (different hidden size — it would need its own tune).
- The QwenPU repo also carries the full stories260K build chain
  with int4 quantization-aware export (`guests/stories260k/`:
  `export_vx.py`, `qat.py`, `gen_vd2.py`, `storyrun.py`).

## 4. The fused executor — branches `npu` + `vx` (not on main)

`fusedexec.py`: ≈**5,500 trace tok/s on an L4 vs Executor2's
≈3,200** — a ~1.7× speedup of every circuit run, executor-side
only, no recompile. Read 2026-10-31: its `FreeRun3` takes the
same compiled-core object as Executor2 (`FreeRun3(c)`,
`core.snapshot(c, m)`, `load`/`run`) — it is Executor2's kernels
fused, with the token decode moved onto the GPU, **CUDA-only**
(torch + Triton). So the trial is: on a CUDA box, run our stock
graph through it and check the first 3,000 tokens against the
reference exactly, then the grown graph. That box is Batman —
the trial waits for it (WSL reboot still outstanding). The Mac
(fastexec, CPU) cannot host it.

## 5. A `python()` native tool — the author's stated next step

From the blog's "Limits and what's next": Qwen writing Python
and running it on the MicroPython **already inside the weights**,
through a native `python()` tool that works the same way `calc`
does. Not built on any branch we found — it's a pattern to copy
(the calc/store dispatch shows exactly how a native tool is
wired), and the MicroPython port itself is already in our tree
(`env/mpy/`, `env/build_mpy.sh`).

## 6. Skills as compiled firmware — `docs/q35-skills.md` (in main)

A costed plan (design + measured token estimates, not built) for
growing the calc firmware's built-in skill set: signed division,
`mod`, `gcd`, cutting division's ≈300-token BIOS overhead. The
idea that transfers: **skills don't have to be prompts — a
procedure used every day can be compiled into the weights**,
beside the CPU, where it costs a few tokens instead of context.
Mirrors our agent-side skills from the other direction.

## 7. The CHIP-8 line — quirks as machine state (in main)

The v1 computer (`chipformer` + `Qwenputer-CHIP8` checkpoints,
`games/` ROM library in our tree). One transferable trick:
`spec.QUIRKS` — six flag *cells in machine state*, set at boot,
read by the circuit at every instruction, so **one set of
weights runs every machine profile** (octo / chip48 / cosmac).
Our stock-vs-grown situation is currently two compiles; a
boot-set flag cell could make memory-map size a profile instead.

## 8. The research line — hidden reasoning + in-weight monitors

- `research/hidden-reasoning/` (in main, in our tree) + the blog
  post: under answer-only training, a hybrid model *can* learn
  to track state silently in its GDN layers; whether it
  generalises past trained lengths comes down to the write
  strength β. Qwen3.5-4B (β < 1) did **not** generalise in
  their runs (0 of 2); OLMo-Hybrid (β up to 2) did (5 of 5).
  This is the literal substrate of "recursive thought": the
  same state our CPU uses as RAM is a channel the brain itself
  could learn to compute in — and a reason a future tune's
  β projections deserve a deliberate decision, not a default.
- The second follow-up (blog, via its summary): the same
  compiler builds **in-weight monitoring tools the model can't
  tamper with** — compiled checkers/answer keys living beside
  the language layers. For us: a compiled verifier next to the
  CPU is the same trick as the store — e.g. an in-weights
  checker that an agent answer must pass. Concept-stage.

## 9. Checkpoint shelf (divagr1925, Hugging Face)

| Checkpoint | What | Date |
|---|---|---|
| Qwen3.5-4B-Qwenputer | the stock computer (ours) | 2026-10-19 |
| Qwen3.5-4B-Qwenputer-Filesystem | store v1-era build | 2026-10-20 |
| Qwenputer-npu | CPU + VDOT/WB/SMAQA | 2026-10-30 |
| QwenPU | CPU + vector unit, hidden 5120 | 2026-10-31 |
| Qwen3.5-4B-Qwenputer-CHIP8 | the v1 CHIP-8 computer | 2026-10-19 |
| chipformer | standalone CHIP-8 compiled transformer | 2026-10-15 |

(Plus his off-line work — factorio SFT/GRPO models,
antiatropos-qlora — not part of the computer line.)

## Staged plan — APPROVED by Duckets 2026-10-31, minus QwenPU

Duckets (2026-10-31 18:13 ET): "Okay we want that but will skip
the QwenPU for the moment." So: items 1–4 and 6 are the plan,
in this order, starting after the in-flight framework/dashboard
work lands; item 5 (QwenPU) is parked by his word. The rule
stands: runnable, not overdone.

1. **Store v2** — the only branch that pays for the *agent*
   roadmap (persistent in-weights memory the brain can use
   across sessions = the recursive memory ask). Same compiler
   op as the grown map; merge it into the grown build, prove
   token-for-token, keep the 32 KB stock build as the control.
2. **fusedexec check** — an hour's reading; if it eats our
   graph, every circuit run gets ~1.7× faster for free.
3. **Quirks-style profile flag** — fold stock/grown into one
   build the next time the compiler is open anyway.
4. **NPU** — only when a real job needs NN math on the
   in-weights CPU (per Duckets' 2026-10-31 conditional OK).
   Prefer the author's published checkpoint for experiments
   before recompiling anything.
5. **QwenPU / vector unit** — the trophy option: strongest
   per-pass power, but a different brain (own tune, 26.7 GB,
   no Batman fit). Park unless the direction changes.
6. **`python()` tool + firmware skills** — patterns to copy
   when the circuit agent gets its next build round; the
   MicroPython half already exists.
