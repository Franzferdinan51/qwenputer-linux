# Qwenputer-X86 — upgrade log & roadmap

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
| Speed | ~3,200 trace tok/s (Triton, L4) for the in-weights CPU | 134.9 tok/s chat decode (KVM, 16 vCPU) |
| Exact calc | In-weights RISC-V program | In-kernel evaluator |
| Filesystem | In-weights (512 files, persistent across sessions via saved heads) | In-kernel RAM FS (32 files, session only) |
| Agent | Guest `hermes-agent` + `ask` coprocessor | Host bridge agent loop + scheduler |
| Status | Linux circuit port into weights still pending | Bootable today; chat is upstream-verified |

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
