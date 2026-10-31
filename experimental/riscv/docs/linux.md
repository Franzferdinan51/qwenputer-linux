# Linux on Qwenputer — design

Goal: boot a real Linux kernel on the CPU that lives in Qwenputer's weights,
enhanced enough to be Linux-class, and keep every piece verifiable.

## The machine spec is not invented — it is mini-rv32ima's

[mini-rv32ima](https://github.com/cnlohr/mini-rv32ima) (MIT, CNLohr) is a proven
~one-file RV32 emulator that boots nommu Linux. We match its machine exactly,
so its known-good kernel + initramfs images are our images:

| Piece | Spec |
|---|---|
| ISA | RV32IMA + Zicsr + Zifencei (fences are no-ops). `misa` = 0x40401101 |
| Privilege | **M + U only. The kernel runs in M-mode** (`CONFIG_RISCV_M_MODE`). No S-mode, no OpenSBI, no PLIC |
| RAM | at `0x80000000`, 64 MiB (the DTB declares 64 MiB − 16 KiB; the top 16 KiB holds the DTB) |
| Kernel entry | `pc = 0x80000000`, `a0 = 0` (hart), `a1 = ` DTB address |
| UART | 8250 subset at `0x10000000`: write byte = putchar, read = RX, LSR at +5 |
| CLINT | at `0x11000000`: `msip` +0x0000, `mtimecmp` +0x4000, `mtime` +0xBFF8. Timebase 1 MHz |
| SYSCON | at `0x11100000`: write 0x5555 = poweroff, 0x7777 = reboot |
| Qwen device | at `0x10010000` (added by this fork, not in mini-rv32ima): DATA +0, STATUS +4, CTRL +8 — see "The Qwen coprocessor" below |
| Rootfs | uncompressed initramfs linked into the kernel Image. No disk, no virtio |
| Kernel | 6.1.14 prebuilt (also 5.18 / 6.5.12 / 6.8-rc1) from `cnlohr/mini-rv32ima-images`, Buildroot, uClibc, static busybox. Image = 3,476,752 bytes |
| Boot cost | kernel init completes in **25–28M instructions** (measured on the reference C emulator). At Qwenputer's current 3,700 instr/s that is ~2 hours; at the persistent-kernel target (~30k instr/s) ~15 minutes. A post-boot state snapshot makes later boots a state load |

## How much was actually missing?

Measured against this spec, upstream Qwenputer's gap is small and precise:

- **A extension** — upstream is RV32IM. Linux needs atomics.
- **Real writable CSRs** — upstream CSRs are read-only stubs folded into the
  ROM; any CSR write is illegal. Linux writes `mtvec/mstatus/mepc/...` in its
  first instructions.
- **Traps that return** — upstream faults halt. Linux lives on traps:
  timer interrupts, `ecall` from userspace, misaligned fixups, `mret`.
- **A machine around the core** — RAM at the Linux address, a UART, a timer.
  In the hybrid design the host/executor provides these (see the main README
  discussion); in the reference machine they are implemented directly.

Everything else — the RV32IM + Zba core, the DeltaNet-as-RAM mechanism, the
executor — already matches.

## The Qwen coprocessor — a terminal that talks to the weights

The CPU in the weights is a grind engine: exact, tireless, slow. The
language model wrapped around it is the opposite. Per-instruction help
from the language layers can never pay (one text token costs ~400 chip
steps, and the host predicts the next instruction exactly for free) —
but a *big semantic job* inverts the trade: ~100 tokens of summarising
or parsing can replace millions of CPU instructions. So the model is
attached the way coprocessors always have been: as a device the guest
writes a job to and reads a result from.

**Register map** (MMIO window at `0x10010000`, no interrupts, polled):

| Offset | Read | Write |
|---|---|---|
| DATA +0 | pop one answer byte (0 = queue empty; answers are NUL-terminated) | append one prompt byte |
| STATUS +4 | bit 0: answer bytes waiting | — |
| CTRL +8 | — | 1 = submit prompt, 2 = clear both queues |

Submission is synchronous: the machine pauses while the model thinks.
That is the honest shape of the device — at circuit speeds the wait
dominates anyway, and a snapshot of a paused machine is still exact.

**Host side.** `LinuxMachine.on_qwen_prompt` is a plain `str -> str`
callable; `chipformer/rv/qwenbackend.py` provides `echo` (deterministic,
offline, proves the plumbing) and `lmstudio` (asks whatever model is
*already loaded* in LM Studio — auto-detected, never triggers a load).
The endgame backend is the Qwenputer checkpoint's own language layers:
the same model whose chip layers run the guest, woken from its frozen
state only for these jobs. It drops in behind the same callable with no
guest or device change.

**Guest side — two channels, one backend.** The prebuilt kernel knows
nothing about the window: its 8250 driver is built for 1 port (no spare
UART for a second device), and — verified in the guest 2026-10-31 —
busybox has no `devmem` applet and the kernel has no `/dev/mem`
(`CONFIG_DEVMEM` off), so the MMIO window cannot be poked from this
rootfs's userspace at all. `env/linux/ask.sh` therefore tries the MMIO
window first (devmem, else `/dev/mem` + `dd` — the register map above
stays the contract, and a rootfs rebuild with a real `/dev/qwen`
driver lights it up unchanged), and otherwise falls back to the
**console line protocol**: the guest prints a line starting with
`QWEN:` on its console, the host answers it and types the answer back
as console input — one line, flattened — and the script's `read`
collects it. The model sits on the terminal line itself, the way a
terminal server would. Usage from the guest shell:
`ask "what is the capital of France"`.

**Proof — verified 2026-10-31.** `chipformer/rv/qwentest.py` boots the
image, logs in, installs `/tmp/ask` over the serial console, and
verifies the full round trip with the host recording the exact prompt
it received. Echo backend: PASS (90.5M steps, transcript at the
repo root in `boot-logs/qwen-echo.out`). LM Studio backend (the model
already loaded there, auto-detected): the guest asked
`what is the capital of France` and printed the model's answer,
`The capital of France is **Paris**.` — PASS (91.0M steps,
`boot-logs/qwen-lm.out`). Getting console input to work at all took a
machine fix along the way: the UART's IIR read as a constant "no
interrupt", but with irq=0 the 8250 driver's backup-timer poll reads
IIR to decide whether to drain the receiver — so getty never consumed
a single byte. IIR now reports RDA while input is queued
(`tests/test_rv_linux.py` has the regression test).

## What is in this fork, in build order

1. **`chipformer/rv/linux.py` — the Linux-capable reference machine.**
   Upstream `ref.py`'s `Rv32` semantics extended to the spec above:
   A extension (AMO + LR/SC), a real CSR file, M/U privilege, trap entry and
   `mret`, CLINT timer interrupts, 8250 UART, SYSCON, 64 MiB of RAM.
   Unknown CSRs read 0 and ignore writes, and are *counted* — that counter is
   the trap-and-emulate seam, made visible.
2. **`tests/test_rv_linux.py`** — unit tests: AMO/LR/SC semantics, CSR
   read/write/set/clear, trap → handler → `mret` round trip, timer interrupt
   delivery, UART in/out, `ecall` causes from U and M.
3. **`chipformer/rv/linuxboot.py`** — boots the prebuilt kernel Image on the
   reference machine: loads Image + DTB, runs, streams the UART to your
   terminal, feeds your keystrokes back. This is the boot proof.
4. **The Qwen coprocessor** — `linux.py` (the MMIO window),
   `chipformer/rv/qwenbackend.py` (host backends), `env/linux/ask.sh`
   (guest tool), `chipformer/rv/qwentest.py` (the round-trip proof).
   Numbered after the circuit port because it changes nothing about it:
   the window is a machine device, and which backend answers — echo,
   LM Studio, or the checkpoint's own layers — is a host-side choice.
5. **The circuit port (next phase, spec'd here).** Upstream `core.py` compiles
   the RV32IM circuit from the same semantics `ref.py` defines. The port adds,
   in the compiler's IR: the AMO datapath (one read-modify-write record), the
   CSR file as DeltaNet heads with trap entry/exit records, and a resumable
   fault exit so the host monitor can service MMIO/timer and resume — the
   hybrid route described in the README. The reference machine in step 1 is
   the exact target the circuit is checked against, the same way upstream
   checks its circuit against `ref.py` today.

## Running the boot proof

```sh
# get the proven image (MIT, from cnlohr/mini-rv32ima-images)
curl -L -o /tmp/linux-6.1.14.zip \
  https://github.com/cnlohr/mini-rv32ima-images/raw/main/linux-6.1.14-rv32nommu-cnl-1.zip
unzip -o /tmp/linux-6.1.14.zip -d /tmp/rvlinux   # contains Image

cd experimental/riscv
python3 -m chipformer.rv.linuxboot /tmp/rvlinux/Image \
    --dtb chipformer/rv/env/linux/sixtyfourmb.dtb
```

The DTB (`env/linux/sixtyfourmb.dts` / `.dtb`) is from mini-rv32ima (MIT,
CNLohr) — it describes exactly the machine above.

## Honesty box

- **Verified 2026-10-30:** the prebuilt Linux 6.1.14 image boots on this
  fork's reference machine through the end of kernel init
  (`Run /init as init process`) and reaches the **Buildroot login prompt**
  (90.2M steps, clean log, no trap dumps).
- **Verified 2026-10-31:** interactive login works (root, no password)
  and the shell is fully usable, after the UART IIR fix described in
  "The Qwen coprocessor" above; and a guest shell command asked a real
  model a question through that device and printed its answer. The
  terminal talks to the weights' host-side brain today; the weights
  themselves answer when the circuit port (step 5) lands, behind the
  same `str -> str` backend callable.
- Linux booting on the **reference machine** = the design is right and the
  kernel agrees. That is what step 3 proves.
- Linux booting **purely in the weights** additionally needs the step-4
  circuit port. Nothing here claims that is done until the circuit replays
  the same boot trace exactly, which is the standard upstream already uses
  for every other program it runs.
