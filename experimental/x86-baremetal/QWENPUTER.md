# Qwenputer-X86 lives here

This directory is upstream [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)
(MIT, commit `48feccb`) imported whole — shared asm core, Phase 1 inference,
Phase 2 bare-metal OS, Phase 3 trainer — plus the Qwenputer computer layer.

**Start with [`qwenputer/README.md`](qwenputer/README.md)** for what was added
(`/calc`, RAM filesystem, `/info`, host agent bridge + scheduler), how to
build and test it, and the honest status. Roadmap: `qwenputer/UPGRADES.md`.

Local changes vs upstream, for easy diffing:
- `02-baremetal-os/qwenputer.asm` — new module (calc + FS + info/help)
- `02-baremetal-os/kernel.asm` — REPL hook (`qp_handle_line`), prompt/banner/help text
- `Makefile` — `qwenputer.o` added to the OS objects
- `qwenputer/` — docs + host tools + tests (new)
- `assets/` (two PNG screenshots) omitted from this copy — binary files;
  see them in the upstream repo.
