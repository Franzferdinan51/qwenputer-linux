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

**The agent harness — `hermes-agent` (2026-10-31).** On top of `ask`
sits a tiny agent loop in the shape of a gadget client: the model is
the brain, a pre-configured shell script is the hands.
`env/linux/agent.sh` is installed in the guest as both
`/tmp/hermes-agent` and `/tmp/agent`; it bakes in a system prompt and
one tool (`run`), then loops over the QWEN: channel — one line each
way, because the device flattens every answer: the model replies
`TOOL run <busybox command>` (the harness runs it and sends the
output back as the next prompt's RESULT) or `SAY: <answer>`. The
tool is a whole Linux system, not a calculator: the prompt teaches
the model that files persist for the session and to keep a notebook
in `/root/nb` — facts and intermediate results written down and read
back in later steps or later tasks — alongside worked examples
(files, exact shell arithmetic, listing a directory). Repeated
commands are not re-run, and empty model replies get a nudge
instead of ending the loop; nine rounds max, and if the cap is hit
the harness reports the last tool result as the answer, labeled as
such. The
full Hermes Agent cannot live in here — it is a big Python framework
with a gateway and plugins — but the gadget pattern ports, and its
endgame backend is the same seat: the checkpoint's own language
layers. **Proof — verified 2026-10-31** (`chipformer/rv/agenttest.py`,
LM Studio backend): given "write the word banana into /tmp/hello.txt
then read it back and tell me what it says", the model's tool calls
wrote the file; the test then verified the contents from `od`, not
the model's word — PASS (`boot-logs/agent-lm.out`). The loop also
surfaced a second UART bug, the transmit twin of the IIR fix above:
IIR never reported THRE, which is the only kick that restarts a
blocked transmitter on this irq=0 port, so any guest write past the
inline chunk (~130 bytes — the agent's own prompt) slept forever.
IIR now reports THRE when no input is queued; same regression test.

**The bridge — the model can use `hermes-agent` (2026-10-31).** The
harness above is something a *user* starts. The bridge lets the
*model in the seat* start it: `qwenbackend.agent_bridge_backend`
wraps any backend and offers every plain question a choice — answer
it, or reply `AGENT: <task>`. On that reply the wrapper returns a
`[[RUN-AGENT: <task>]]` marker instead of an answer; the guest's
`ask` sees the marker, runs `/tmp/hermes-agent` right there in the
same partition (same files — a file the user made is a file the
agent sees), and asks again with the agent's report folded in as
`HERMES REPORT:`. Prompts from inside the harness carry its baked-in
system prompt and always go straight to the base model, so
delegation is exactly one level deep. **Proof — verified 2026-10-31**
(`chipformer/rv/bridgetest.py`, scripted seat): the full path —
marker, in-partition agent run, real file work, report, final
answer — passed with the file's contents verified from `od`
(`boot-logs/bridge-scripted.out`). Seating a real model
(`bridgetest --live`, or `liveterm --bridge`), whether it chooses
to delegate is the model's call; the plumbing no longer is.

**The bare-metal harness (2026-10-31).** The Linux harness needs a
whole OS under it. `env/progs/agent.c` is the same gadget pattern as
a single bare-metal program for the stock circuit map — the form
that can be snapshotted into the weights next to C and MicroPython:
a baked-in system prompt, exact tools (`store` / `recall` / `mul`),
and the model's seat reached through a **mailbox in RAM** — the
guest writes a prompt into `mb_out`, sets `mb_req`, and spins; the
runner (`chipformer/rv/bareagent.py`) services it with the same
backend callable `ask` uses. Verified on the reference machine
against LM Studio: the guest's tool calls stored a word and the
Rv32 `mul` tool returned the exact 17-digit product
123456789 × 987654321 = 121932631112635269 — the multiplications a
language model cannot do in its head are the ones its own CPU does
exactly. The harness mechanics (mailbox, dedupe, nudge, round cap,
labeled fallback) ran clean in every run; the seated 9B model's
follow-through — several empty replies per run, and never quite
assembling the final `SAY:` inside the cap — is the limit, as with
the Linux harness. Build: `env/build_agent.sh` (xPack newlib route
or Debian picolibc route); transcript in `boot-logs/bare-lm.out`.

### The preconfigured brain: the built-in weights

The harness is **preconfigured to think with the built-in weights —
Qwen 3.5 4B, the model whose own weights host this machine's CPU**
(the `builtin` backend in `qwenbackend.py`, and the default seat in
`liveterm.py` / `agenttest.py` / `bridgetest.py`). It resolves the
family in LM Studio's model list and pins it — a pristine/base build
wins over a tagged fine-tune when both are present — and it refuses,
loudly, when no Qwen 3.5 4B is available, rather than silently thinking
with a different model. The older `lmstudio` backend (first loaded
model, auto-detected) remains for experiments. Two practical notes:
today's LM Studio build here is a fine-tune of Qwen 3.5 4B (the
pristine base takes precedence automatically when one is added), and
Qwen 3.5's thinking wants headroom — `QWEN_MAX_TOKENS=1024` for agent
work; at 512 its first delegation reply once came back empty. With
the built-in weights seated, the live proofs all pass — the agent
round trip (`boot-logs/agent-builtin.out`) and, at 1024 tokens, the
delegation bridge end to end: the model chose `AGENT:`, hermes-agent
did the file work in the same partition, and the final answer folded
its report in (`boot-logs/bridge-builtin.out`). In the circuit
endgame this same backend slot is the checkpoint's own language
layers; nothing on the guest side changes.

### Running it fast: the native core and snapshots

The reference machine is the oracle, not the vehicle: at ~0.5M
instructions/s every boot is a minute and every agent test ten.
This fork now has the Doom port's medicine, transplanted:

- **`chipformer/rv/fastlinux.c` — a native core** mirroring
  `linux.py` function for function (same traps, CSRs, devices, Qwen
  window + console protocol, same deterministic 1-tick-per-retired-
  instruction time), speaking a small fd 3/4 host protocol
  (RUN / FEED / HASH + QWEN / QWENM prompt events) so the Python
  drivers and backends are unchanged behind `fastmachine.py`.
  **Parity is proven, not hoped**: at 8M and at 60M steps, the native
  core and the Python machine retire identical step counts, produce
  byte-identical console output, and hash identically over
  pc + privilege + registers + steps + mtime + all 64 MiB of RAM
  (`chipformer/rv/paritycheck.py`). Speed on the Mac: **~198M
  steps/s vs ~0.5M** — a boot in ~0.3 s, a full agent round trip in
  seconds of machine time; the model's reply latency is all that
  remains. Drivers take `--fast [--fast-bin PATH]`.
- **`chipformer/rv/snapshot.py` — warm shells**: prime once
  (boot → login → install, ~150M steps), save the whole machine,
  and later runs restore straight to the ready shell
  (`--snapshot shell.snap` in the tests; `liveterm --snapshot`).
  The snapshot agent test passes against the live model.

### The dashboard

`chipformer/rv/dashboard.py` is a stdlib-only host server (default
port 8952) and the human interface to the whole stack: Overview,
Chat, Terminal, Desktop, Apps and Settings tabs in one page. Chat
streams (SSE) from the seat resolved by the builtin rule; Settings
can pin the endpoint, model, temperature, token budget and system
prompt, persisted to `dash-settings.json` beside the rig. Overview
reads the circuit runs' own logs, so verdicts shown are the runs'
verdicts. The Terminal tab embeds the live terminal untouched; the
Desktop tab marks the framebuffer spike the Doom live view
prototyped (guest draws in its memory, host shows it, input goes
back).

### Running it on the circuit itself

Everything above runs the guest on a host-side machine (reference or
native) with a model in the seat. The other seat for the bare-metal
harness is the compiled circuit — the chip layers themselves:

- **`chipformer/rv/circuitagent.py`** drives `env/progs/agent.c`
  through the `fastexec` Executor: `core.snapshot` seeds the
  circuit's tables from the ELF, then each generated token is one
  instruction the circuit computes. The host's hands are the ones
  the circuit gives it — the RAM tables' state rows in the executor
  *are* the guest's memory (a store in the circuit is a row scatter
  into the same tensors), so the mailbox is serviced by reading and
  writing those rows directly; the guest binary and protocol do not
  change. Verified before any free run: `fastexec.check` (executor
  == IR evaluator, exact) and a 600-token free run matching the
  reference trace token for token (`chipformer/rv/circdiag.py`).
- **Measured on the stock circuit** (Mac CPU, ~269 tok/s): the
  guest booted in-circuit, composed its prompts, and had them
  serviced through the circuit's memory at tokens 14,336 / 79,360 /
  160,768, firing its tools from the replies
  (`boot-logs/circuit-echo.out`).
- **With the built-in weights in the seat: PASS** — 82,122 tokens,
  454 s, three mailbox rounds, final answer carrying the stored
  word and the exact product, verified in Python by the runner
  (`boot-logs/circuit-stock-builtin.out`). Three findings came with
  it, all measured against the live model, not guessed:
  (1) the 9-digit operand pair (123456789 × 987654321) sends this
  brain into an endless reasoning loop — it tries to work the
  product out in its head and never emits content at any budget up
  to 8,192 tokens; the task was bisected live and the operands are
  now 123456 × 654321 (an 11-digit product, still beyond in-head
  reliability). The 6-digit pair still needs a ~4,096-token budget:
  its thinking runs long even when it answers.
  (2) This brain simulates ahead — it writes the whole plan,
  including invented RESULTs and wrong products, into one reply.
  The guest now enforces one call per reply (the call is cut at the
  next simulated call / RESULT / SAY) and its repeat guard compares
  the cut call, so a replayed first step is caught and nudged
  instead of re-run. Both guards fired in the PASS run.
  (3) Honest caveat: in the PASS run the `mul` tool itself never
  fired — nudged past its replay loop, the model SAID the product
  its simulation had produced, and that simulation happened to be
  exact (it had invented four wrong products in earlier attempts).
  The circuit, mailbox, guards, and answer are proven; this brain's
  *delegation discipline* is not — it would rather simulate than
  delegate, and its simulated arithmetic is usually wrong. That is
  a harness/brain problem, not a circuit problem.
- **The grown map applies here too.** The agent rebuilds for the
  2 MB circuit (`env/rom-big.ld`, `AGENT_LD` + `AGENT_MB` knobs in
  `env/build_agent.sh`): RAM 32 KB → 2 MB, and the mailbox widens
  896 → 8,192 bytes so long model answers fit whole (the size is
  recorded in the `.sym.json` and both runners honor it). The
  grown graph + this ELF pass the same `fastexec.check` gate. The
  grown-map compiler itself (spec/core changes) lives on the
  `circuit-grown-map` branch, where the Doom port proved it.

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
   On top of it: `env/linux/agent.sh` (the `hermes-agent` harness) +
   `chipformer/rv/agenttest.py`, the delegation bridge
   (`agent_bridge_backend` in `qwenbackend.py`, the marker path in
   `ask.sh`) + `chipformer/rv/bridgetest.py`, and the bare-metal
   harness (`env/progs/agent.c`, `env/build_agent.sh`,
   `chipformer/rv/bareagent.py`). The seat is preconfigured:
   `builtin` in `qwenbackend.py` pins Qwen 3.5 4B — the built-in
   weights — as the harness's default brain.
4b. **Speed** — `chipformer/rv/fastlinux.c` (the native core) +
   `chipformer/rv/fastmachine.py` (its Python face) +
   `chipformer/rv/paritycheck.py` (the oracle-parity proof), and
   `chipformer/rv/snapshot.py` (primed warm shells). The native core
   is verified against step 1's machine, not instead of it.
4c. **The circuit seat** — `chipformer/rv/circuitagent.py` (the
   bare-metal harness executed by the compiled circuit, mailbox
   serviced through the executor's memory state) +
   `chipformer/rv/circdiag.py` (the executor-vs-reference gate) +
   `env/rom-big.ld` and the `AGENT_LD`/`AGENT_MB` build knobs (the
   grown-map agent: 2 MB RAM, 8 KB mailbox).
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
