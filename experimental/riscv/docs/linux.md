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
4. **The circuit port (next phase, spec'd here).** Upstream `core.py` compiles
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

- Linux booting on the **reference machine** = the design is right and the
  kernel agrees. That is what step 3 proves.
- Linux booting **purely in the weights** additionally needs the step-4
  circuit port. Nothing here claims that is done until the circuit replays
  the same boot trace exactly, which is the standard upstream already uses
  for every other program it runs.
