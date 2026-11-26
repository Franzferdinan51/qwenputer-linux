# Qwenputer-X86 — upgrade log & roadmap

## Shipped in v1.2 (2026-11-25, night) — agent framework v3

Mined [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
(read-only) for the patterns that fit a small model in a bare-metal
kernel, and folded them into `qp_agent.py`:

1. **Frozen memory snapshot** (Hermes' core memory design: MEMORY.md
   enters the system prompt as a snapshot at session start) — notes
   ride in the instruction line as `name=value` pairs under a char
   budget, degrading to names-only past it.
2. **TODO tool** — the model keeps a plan list host-side
   (`TODO: add/done/start/clear`), every op returns the full list with
   Hermes' `[ ]`/`[>]`/`[x]` markers.
3. **Skills** — `SKILL.md` files with Hermes frontmatter
   (`name:`/`description:`); the catalog rides in the prompt, `SKILL:
   <name>` loads the body on demand (progressive disclosure). Ships
   with `exact-arithmetic` and `note-taking` skills.
4. **Delegation** — `SUB: <subtask>` runs a nested agent task on the
   same machine and returns its answer as the tool result (depth 1).
5. **Session search** — `--search <term>` over transcripts.

Verified on Robin + 1.7B (logs `boot-logs/qp-x86-robin-v3-*`):
snapshot recall answered 97406784 from the prompt alone; skill test —
SKILL loaded, then CALC 777*888 → SAY **689976**; delegation test —
TODO added, SUB ran its own CALC (99*99) and the parent answered
**9801**. One live bug found and fixed on the way: multi-line tool
results (skill bodies) are flattened before going back into the
single-line REPL.

## Shipped in v1.1 (2026-11-25, later)

1. **Qwen3-1.7B** — the kernel reads its geometry from the `.asmbin`
   header at boot, so the bigger model needed *no kernel changes*: same
   27,132-byte binary, 1.7B bound as the Multiboot module (export:
   1,838,448,832 B). Verified on Robin under KVM: 16/16 boot checks,
   coherent chat, **20.1 tok/s decode / 23.4 tok/s prefill** (8 cores,
   Ryzen 9 4900H). Needs `-m 4G` (the 0.6B runs in 2G).
2. **Sharded-weight exporter** — upstream's exporter assumed a single
   `model.safetensors`; Qwen3-1.7B+ ship sharded. `TensorStore` now
   reads the index (or unions the shards when the index is absent).
   Regression-proof: re-exporting the 0.6B reproduces the known-good
   SHA256 byte-for-byte.
3. **Agent framework v2** (`qp_agent.py`) — the Hermes pattern, pulled
   in properly now that x86 has room: a six-tool registry (CALC/PUT/
   GET/DEL/LIST/INFO), **persistent memory** (host JSON synced into the
   kernel FS at boot and back out before halt — the session-only RAM FS
   gains cross-session memory), a scheduler with raw-line *and* agent
   jobs (one-shot and repeating), and timestamped session transcripts.
   Loop discipline learned live: the typed-line echo is stripped before
   parsing (it contains the instruction text, `SAY:` included), and a
   tool line that precedes `SAY:` in a reply always runs first — small
   models write the whole plan at once, hallucinated results and all;
   the kernel's real result replaces them.
   **Proven end-to-end on Robin + 1.7B:** session 1 — CALC 123456*789 →
   kernel's exact 97406784 → PUT → GET → SAY 97406784, note saved;
   session 2 (fresh boot) — note synced in, GET cold, SAY 97406784.
   Logs: `boot-logs/qp-x86-robin-17b-agent{1,2}.log`.

## Verified hosts (2026-11-25)

Same kernel (27,132 bytes, built natively on each host), same model file —
the `.asmbin` export is deterministic: SHA256
`14dc92c18dfd4182be6627558f716b9e7c5ebf9bc657b0ff700304e5a6502796` on all
three machines. 16/16 serial boot checks pass on every host.

| Host | Accel | SMP | Prefill | Decode | Boot → 1st tok |
|---|---|---|---|---|---|
| Muse VM (4 vCPU) | QEMU TCG | 4/4 | — | 0.31 tok/s (parity smoke) | ~100 s |
| Mac Studio (Apple M5 Max) | QEMU TCG | 8/8 | 5.49 tok/s | 4.28 tok/s | 4.7 s |
| Robin (Ryzen 9 4900H) | **KVM** (Docker, `--device /dev/kvm`) | 8/8 | 72.9 tok/s | **56.2 tok/s** | 1.5 s |
| Robin, **Qwen3-1.7B** | **KVM** (same setup) | 8/8 | 23.4 tok/s | **20.1 tok/s** | 1.9 s |

Robin's KVM decode matches upstream's published live figure (55.4 tok/s,
Xeon E5-2670 v3) — with the Qwenputer additions compiled in. Chat parity
vs the pristine upstream kernel (token-identical output) was verified on
the VM before the Mac/Robin runs. Logs: `boot-logs/qp-x86-*.out/.log` in
the repo root. Host runners: `qwenputer/run_mac_test.sh`,
`qwenputer/run_robin_test.sh`, `qwenputer/Dockerfile.qptest`.

## Shipped in v1.0 (2026-11-25)

Baseline: upstream llm.asm @ `48feccb` imported whole (shared core, Phase 1
inference, Phase 2 bare-metal OS, Phase 3 trainer, scripts), MIT license
preserved, builds verified at upstream's exact published sizes before any
change was made.

On top of that baseline:

1. **Exact calc in the kernel** (`02-baremetal-os/qwenputer.asm`) — a
   recursive-descent signed-64-bit evaluator. This is Qwenputer's founding
   trick (the RISC-V CPU exists so arithmetic is *exact*, not generated)
   ported to where it is cheapest: the host CPU itself.
2. **A filesystem in the kernel** — 32-slot RAM FS with `/put /get /del
   /list`, mirroring the in-weights filesystem model's tool surface
   (`put/get/list/delete`) so prompts and habits transfer between tracks.
3. **Identity** — `qwenputer>` prompt, Qwenputer-X86 banner, `/help` and
   `/qhelp` covering both command sets, `/info` status.
4. **Scheduler** (host bridge) — the RISC-V track's `agentshell` idea
   (foreground supervisor firing due jobs into the machine) as
   `qp_agent.Scheduler` over a JSON-lines jobs file.
5. **Agent loop** (host bridge) — `CALC:/PUT:/GET:` tool lines from the
   model executed by the kernel and fed back; the RISC-V `hermes-agent`
   gadget pattern, x86 edition.
6. **Tests beyond upstream's** — a Python oracle for calc semantics
   (fuzz-verified), and a serial boot test that exercises every new kernel
   command in the real OS, not a simulation.

## The two tracks, lined up

| | RISC-V track (existing) | X86 track (this) |
|---|---|---|
| Computer lives | *inside* Qwen3.5-4B's weights (DeltaNet state as RAM) | *beside* Qwen3-0.6B, on the real CPU |
| OS | Linux 6.1 on the enhanced reference machine | Bare-metal asm kernel, no Linux at all |
| Speed | ~3,200 trace tok/s (Triton, L4) for the in-weights CPU | 134.9 tok/s upstream (KVM, 16 vCPU); **56.2 tok/s measured here (KVM, Ryzen 9 4900H, 8 cores)** |
| Exact calc | In-weights RISC-V program | In-kernel evaluator |
| Filesystem | In-weights (512 files, persistent across sessions via saved heads) | In-kernel RAM FS (32 files, session only) |
| Agent | Guest `hermes-agent` + `ask` coprocessor | Host bridge agent loop + scheduler |
| Status | Linux circuit port into weights still pending | Bootable today; verified on VM, Mac Studio and Robin (see Verified hosts) |

Neither replaces the other: the RISC-V track is the research claim (a
computer *in* a model), the x86 track is the usable machine (a model *as*
the computer). Features land on whichever track proves them cheaper, then
cross over — calc and the FS crossed this way.

## Roadmap (in order)

1. **Persistence** — save the RAM FS + KV-cache to a second Multiboot
   module / virtio-blk sector range on `/halt`, restore at boot. Design:
   FS image is 9,248 B; a raw 1 MiB append region suffices for FS + notes.
2. **Bigger models** — Qwen3-1.7B geometry in the shared transformer
   (config from the .asmbin header instead of hard-wired 0.6B constants).
3. **Framebuffer** — VESA/GOP linear framebuffer alongside VGA text, the
   display path the RISC-V Doom work already proved the shape of; then a
   bare-metal Doom port sharing cores with inference (SMP partitioning:
   N-1 cores inference, 1 core app).
4. **In-kernel scheduler** — move the jobs tick from the host bridge into
   the kernel (PIT interrupt already exists), so scheduled prompts fire
   with no host attached.
5. **Qwen3.5 hybrid** — DeltaNet layers in the asm transformer; at that
   point the x86 track could host the RISC-V circuit's *output* the same
   way LM Studio hosts it on the reference side.
