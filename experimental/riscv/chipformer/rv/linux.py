"""A Linux-capable RV32 machine: upstream ``ref.Rv32`` semantics extended to the
mini-rv32ima machine spec (see docs/linux.md).

Adds, over RV32IM + Zba:
  * the A extension (AMO word ops + LR/SC with a reservation),
  * a real, writable CSR file (the 11 CSRs Linux uses here, plus the U-mode
    counter shadows; anything else reads 0, ignores writes, and is counted in
    ``csr_fallback`` — the trap-and-emulate seam, made visible),
  * M/U privilege, trap entry (exceptions + timer/software interrupts),
    ``mret`` and ``wfi``,
  * the machine around the core: 64 MiB of RAM at 0x80000000, an 8250 UART at
    0x10000000, a CLINT at 0x11000000 (msip / mtimecmp / mtime, 1 MHz), and a
    SYSCON poweroff/reboot register at 0x11100000.
  * a Qwen coprocessor window at 0x10010000 (the /dev/qwen seam): the guest
    writes prompt bytes to DATA (+0), writes 1 to CTRL (+8) to submit, polls
    STATUS (+4, bit 0 = answer waiting), and reads the NUL-terminated answer
    back from DATA. The host side answers through ``on_qwen_prompt`` — today
    a backend module (echo / LM Studio), in the endgame the Qwenputer
    checkpoint's own language layers, frozen while the CPU runs and woken
    only for these big semantic jobs. See docs/linux.md.

Deterministic time: ``mtime`` advances ``ticks_per_step`` per retired
instruction (default 1, i.e. the core runs at an implied 1 MIPS against the
1 MHz timebase). No wall clock anywhere — a run is replayable exactly.
"""

from dataclasses import dataclass, field

from chipformer.rv import ref
from chipformer.rv.ref import M32, alu, decode, sx, taken

RAM_BASE = 0x80000000
UART_BASE = 0x10000000
QWEN_BASE = 0x10010000
CLINT_BASE = 0x11000000
SYSCON_BASE = 0x11100000
MMIO_LO, MMIO_HI = 0x10000000, 0x12000000

MSTATUS_MIE = 1 << 3
MSTATUS_MPIE = 1 << 7
MSTATUS_MPP = 3 << 11
MIP_MSIP = 1 << 3
MIP_MTIP = 1 << 7
MIP_MEIP = 1 << 11

MISA = 0x40401101  # the value mini-rv32ima reports; the kernel is built for it

# trap causes
C_INST_MISALIGNED, C_INST_FAULT, C_ILLEGAL, C_EBREAK = 0, 1, 2, 3
C_LOAD_MISALIGNED, C_LOAD_FAULT, C_STORE_MISALIGNED, C_STORE_FAULT = 4, 5, 6, 7
C_ECALL_U, C_ECALL_M = 8, 11
IRQ_SOFTWARE, IRQ_TIMER = 3, 7  # interrupt causes (mcause bit 31 set)

AMO5 = {0x00: "AMOADD", 0x01: "AMOSWAP", 0x02: "LR", 0x03: "SC", 0x04: "AMOXOR",
        0x08: "AMOOR", 0x0C: "AMOAND", 0x10: "AMOMIN", 0x14: "AMOMAX",
        0x18: "AMOMINU", 0x1C: "AMOMAXU"}

CSR_RO_ZERO = {0xF11, 0xF12, 0xF13, 0xF14}  # mvendorid, marchid, mimpid, mhartid


class _Trap(Exception):
    def __init__(self, cause: int, tval: int = 0, interrupt: bool = False):
        self.cause, self.tval, self.interrupt = cause, tval, interrupt


@dataclass
class LinuxMachine:
    """RV32IMA + Zicsr, M/U privilege, the mini-rv32ima memory map."""

    ram_size: int = 64 * 1024 * 1024
    ticks_per_step: int = 1
    ram: bytearray = field(default_factory=bytearray)
    x: list[int] = field(default_factory=lambda: [0] * 32)
    pc: int = RAM_BASE
    priv: int = 3  # 3 = M, 0 = U
    out: bytearray = field(default_factory=bytearray)
    inp: bytearray = field(default_factory=bytearray)  # queued UART RX bytes
    halted: bool = False
    exit_reason: str | None = None
    steps: int = 0

    # machine state
    csr: dict[int, int] = field(default_factory=dict)
    mtime: int = 0
    mtimecmp: int = 0xFFFFFFFFFFFFFFFF
    msip: int = 0
    reservation: int | None = None
    uart_regs: dict[int, int] = field(default_factory=dict)
    qwen_prompt: bytearray = field(default_factory=bytearray)
    qwen_response: bytearray = field(default_factory=bytearray)
    qwen_line: bytearray = field(default_factory=bytearray)
    on_qwen_prompt = None  # optional callback(str) -> str behind the Qwen device
    csr_fallback: dict[int, int] = field(default_factory=dict)  # unknown CSR -> touches
    on_output = None  # optional callback(bytes) for streamed UART output

    def __post_init__(self):
        if not self.ram:
            self.ram = bytearray(self.ram_size)

    # ---- loading -------------------------------------------------------------

    def load_image(self, data: bytes, addr: int = RAM_BASE) -> None:
        off = addr - RAM_BASE
        if off < 0 or off + len(data) > self.ram_size:
            raise ValueError(f"image of {len(data)} bytes at {addr:#x} "
                             f"does not fit the {self.ram_size} bytes of RAM")
        self.ram[off:off + len(data)] = data

    def boot(self, dtb: bytes, dtb_off: int = 0x3FFC000) -> None:
        """Kernel entry state: pc at the Image, a0 = hart, a1 = DTB pointer."""
        self.load_image(dtb, RAM_BASE + dtb_off)
        self.x = [0] * 32
        self.x[10], self.x[11] = 0, RAM_BASE + dtb_off
        self.pc, self.priv = RAM_BASE, 3

    def feed_input(self, data: bytes) -> None:
        self.inp += data

    # ---- memory ---------------------------------------------------------------

    def _in_ram(self, addr: int, n: int) -> bool:
        return RAM_BASE <= addr and addr + n <= RAM_BASE + self.ram_size

    def read_mem(self, addr: int, n: int) -> int:
        if self._in_ram(addr, n):
            return int.from_bytes(self.ram[addr - RAM_BASE:addr - RAM_BASE + n], "little")
        if MMIO_LO <= addr < MMIO_HI:
            return self._mmio_read(addr, n)
        raise _Trap(C_LOAD_FAULT, addr)

    def write_mem(self, addr: int, v: int, n: int) -> None:
        v &= (1 << 8 * n) - 1
        if self._in_ram(addr, n):
            self.ram[addr - RAM_BASE:addr - RAM_BASE + n] = v.to_bytes(n, "little")
            return
        if MMIO_LO <= addr < MMIO_HI:
            self._mmio_write(addr, v, n)
            return
        raise _Trap(C_STORE_FAULT, addr)

    # ---- MMIO -----------------------------------------------------------------

    def _uart_read(self, off: int) -> int:
        if off == 0:
            if self.uart_regs.get(3, 0) & 0x80:  # DLAB: divisor latch, not data
                return self.uart_regs.get(0, 0)
            return self.inp.pop(0) if self.inp else 0
        if off == 2:
            # IIR reports the port state even with no IRQ line wired
            # (this port is irq=0): the 8250 driver's backup-timer poll
            # reads IIR and only services the source it names. RDA (0x04)
            # when input is queued - a constant NO_INT here starved guest
            # input completely (getty consumed zero bytes in 100M steps).
            # Otherwise THRE (0x02): the transmitter is always empty in
            # this model, and THRE is what lets the poll kick the TX path
            # - without it, a guest write longer than the inline chunk
            # (~130 bytes) sleeps forever waiting for a wakeup.
            return 0x04 if self.inp else 0x02
        if off == 5:
            return 0x60 | (1 if self.inp else 0)  # LSR: THR empty, + RX ready
        return self.uart_regs.get(off, 0)

    def _uart_write(self, off: int, v: int) -> None:
        if off == 0 and not (self.uart_regs.get(3, 0) & 0x80):
            self.out.append(v)
            if self.on_output:
                self.on_output(bytes([v]))
            self._qwen_line_feed(v)
            return
        self.uart_regs[off] = v

    def _qwen_line_feed(self, v: int) -> None:
        """The console face of the Qwen device: a guest program prints a
        line starting with ``QWEN:`` and the answer is typed back as
        console input, one line, for the program's ``read`` to collect.
        The prebuilt kernel has no /dev/mem and no spare UART, so this is
        the channel its shell can actually use; the MMIO window remains
        the device contract for kernels that can reach it. In the
        endgame this is the terminal-server pattern: the model sits on
        the console line itself."""
        if v == 0x0A:
            line = bytes(self.qwen_line)
            self.qwen_line.clear()
            if line.startswith(b"QWEN:") and self.on_qwen_prompt is not None:
                prompt = line[5:].decode("utf-8", "replace").strip()
                try:
                    answer = str(self.on_qwen_prompt(prompt))
                except Exception as e:  # a dead backend must not hang the guest
                    answer = f"[qwen backend error: {e}]"
                flat = " ".join(answer.split())  # one line, for `read`
                # the QWENANS: prefix marks this line as machine-typed:
                # a human typing at the same console while a session
                # runs would otherwise be read as the model's reply
                # (seen live 2026-10-31, it ended a session with the
                # user's own command as its "answer"). Guest readers
                # strip the prefix and skip unmarked lines.
                self.inp += b"QWENANS:" + flat.encode("utf-8", "replace") + b"\n"
        elif v != 0x0D:
            self.qwen_line.append(v)
            if len(self.qwen_line) > 4096:
                self.qwen_line.clear()

    # ---- Qwen coprocessor window -------------------------------------------
    # DATA (+0): write appends a prompt byte; read pops an answer byte
    # (0 when the queue is empty — answers are NUL-terminated, so 0 is also
    # the end marker). STATUS (+4): bit 0 = answer bytes waiting.
    # CTRL (+8): write 1 = submit the prompt, 2 = clear both queues.
    # Submission is synchronous: the machine pauses while the model thinks,
    # which is the honest shape of this device — the CPU is meant to wait
    # on its coprocessor, and at circuit speeds the wait dominates anyway.

    def _qwen_read(self, off: int) -> int:
        if off == 0:
            return self.qwen_response.pop(0) if self.qwen_response else 0
        if off == 4:
            return 1 if self.qwen_response else 0
        return 0

    def _qwen_write(self, off: int, v: int) -> None:
        if off == 0:
            self.qwen_prompt.append(v)
        elif off == 8:
            if v == 1:
                self._qwen_submit()
            elif v == 2:
                self.qwen_prompt.clear()
                self.qwen_response.clear()

    def _qwen_submit(self) -> None:
        prompt = bytes(self.qwen_prompt).decode("utf-8", "replace")
        self.qwen_prompt.clear()
        if self.on_qwen_prompt is None:
            answer = "[qwen: no model attached]"
        else:
            try:
                answer = str(self.on_qwen_prompt(prompt))
            except Exception as e:  # a dead backend must not hang the guest
                answer = f"[qwen backend error: {e}]"
        self.qwen_response += answer.encode("utf-8", "replace") + b"\0"

    def _mmio_read(self, addr: int, n: int) -> int:
        if QWEN_BASE <= addr < QWEN_BASE + 0x10:
            return sum(self._qwen_read(addr - QWEN_BASE + i) << 8 * i for i in range(n))
        if UART_BASE <= addr < UART_BASE + 0x100:
            return sum(self._uart_read(addr - UART_BASE + i) << 8 * i for i in range(n))
        if addr == CLINT_BASE:
            return self.msip & 1
        if addr in (CLINT_BASE + 0x4000, CLINT_BASE + 0x4004):
            return (self.mtimecmp >> (32 if addr & 4 else 0)) & M32
        if addr in (CLINT_BASE + 0xBFF8, CLINT_BASE + 0xBFFC):
            return (self.mtime >> (32 if addr & 4 else 0)) & M32
        return 0

    def _mmio_write(self, addr: int, v: int, n: int) -> None:
        if QWEN_BASE <= addr < QWEN_BASE + 0x10:
            for i in range(n):
                self._qwen_write(addr - QWEN_BASE + i, v >> 8 * i & 0xFF)
            return
        if UART_BASE <= addr < UART_BASE + 0x100:
            for i in range(n):
                self._uart_write(addr - UART_BASE + i, v >> 8 * i & 0xFF)
            return
        if addr == CLINT_BASE:
            self.msip = v & 1
            return
        if addr in (CLINT_BASE + 0x4000, CLINT_BASE + 0x4004):
            sh = 32 if addr & 4 else 0
            self.mtimecmp = (self.mtimecmp & ~(M32 << sh) | (v & M32) << sh) & 0xFFFFFFFFFFFFFFFF
            return
        if addr in (CLINT_BASE + 0xBFF8, CLINT_BASE + 0xBFFC):
            sh = 32 if addr & 4 else 0
            self.mtime = (self.mtime & ~(M32 << sh) | (v & M32) << sh) & 0xFFFFFFFFFFFFFFFF
            return
        if addr == SYSCON_BASE:
            if v == 0x5555:
                self.halted, self.exit_reason = True, "poweroff"
            elif v == 0x7777:
                self.halted, self.exit_reason = True, "reboot"
            return

    # ---- CSRs -------------------------------------------------------------------

    def mip_value(self) -> int:
        v = 0
        if self.msip:
            v |= MIP_MSIP
        if self.mtime >= self.mtimecmp:
            v |= MIP_MTIP
        return v

    def read_csr(self, csr: int) -> int:
        if csr == 0x301:
            return MISA
        if csr == 0x344:
            return self.mip_value()
        if csr in (0xC00, 0xC02):  # cycle / instret
            return self.steps & M32
        if csr in (0xC80, 0xC82):  # cycleh / instreth
            return (self.steps >> 32) & M32
        if csr == 0xC01:  # time
            return self.mtime & M32
        if csr == 0xC81:  # timeh
            return (self.mtime >> 32) & M32
        if csr in CSR_RO_ZERO:
            return 0
        if csr in self.csr:
            return self.csr[csr]
        self.csr_fallback[csr] = self.csr_fallback.get(csr, 0) + 1
        return 0

    def write_csr(self, csr: int, v: int) -> None:
        v &= M32
        if csr in (0x301, 0x344) or csr in CSR_RO_ZERO or 0xC00 <= csr <= 0xC02 \
                or 0xC80 <= csr <= 0xC82:
            return  # read-only views (mip is computed; counters are ours)
        if csr == 0x305:  # mtvec: base aligned, low 2 bits are the mode
            self.csr[csr] = v
            return
        if csr in (0x300, 0x302, 0x303, 0x304, 0x306, 0x340, 0x341, 0x342, 0x343,
                   0x3A0, 0x3B0, 0x3B1, 0x3B2, 0x3B3):
            self.csr[csr] = v
            return
        self.csr_fallback[csr] = self.csr_fallback.get(csr, 0) + 1

    # ---- traps ------------------------------------------------------------------

    def _take_trap(self, cause: int, tval: int, interrupt: bool, epc: int) -> None:
        self.csr[0x341] = epc & M32  # mepc
        self.csr[0x342] = (cause | (1 << 31 if interrupt else 0)) & M32  # mcause
        self.csr[0x343] = tval & M32  # mtval
        ms = self.csr.get(0x300, 0)
        ms = (ms & ~MSTATUS_MPP) | (self.priv << 11)  # MPP = old privilege
        ms = ms & ~MSTATUS_MPIE | (MSTATUS_MPIE if ms & MSTATUS_MIE else 0)
        self.csr[0x300] = ms & ~MSTATUS_MIE  # interrupts off in the handler
        self.priv = 3
        mtvec = self.csr.get(0x305, 0)
        base = mtvec & ~3
        self.pc = (base + 4 * cause) & M32 if (mtvec & 3) == 1 and interrupt else base

    def _pending_interrupt(self) -> int | None:
        mie = self.csr.get(0x304, 0)
        pend = self.mip_value() & mie
        if not pend:
            return None
        if not (self.priv < 3 or self.csr.get(0x300, 0) & MSTATUS_MIE):
            return None
        for cause, bit in ((IRQ_TIMER, MIP_MTIP), (IRQ_SOFTWARE, MIP_MSIP)):
            if pend & bit:
                return cause
        return None

    # ---- main step ----------------------------------------------------------------

    def step(self) -> None:
        if self.halted:
            return
        irq = self._pending_interrupt()
        if irq is not None:
            self._take_trap(irq, 0, True, self.pc)
            return
        pc = self.pc
        if pc & 3:
            self._take_trap(C_INST_MISALIGNED, pc, False, pc)
            return
        if not self._in_ram(pc, 4):
            self._take_trap(C_INST_FAULT, pc, False, pc)
            return
        w = int.from_bytes(self.ram[pc - RAM_BASE:pc - RAM_BASE + 4], "little")
        try:
            self._execute(w, pc)
        except _Trap as t:
            self._take_trap(t.cause, t.tval, t.interrupt, pc)
            return
        self.steps += 1
        self.mtime += self.ticks_per_step
        self.x[0] = 0

    def run(self, max_steps: int) -> int:
        """Run up to max_steps instructions; returns how many actually ran."""
        start = self.steps
        for _ in range(max_steps):
            if self.halted:
                break
            self.step()
        return self.steps - start

    # ---- instruction execution ------------------------------------------------------

    def _execute(self, w: int, pc: int) -> None:
        x = self.x
        nxt = (pc + 4) & M32
        if w == 0x30200073:  # MRET: MIE := MPIE, MPIE := 1, priv := MPP, MPP := U
            ms = self.csr.get(0x300, 0)
            new_priv = (ms >> 11) & 3
            ms = (ms & ~MSTATUS_MIE) | (MSTATUS_MIE if ms & MSTATUS_MPIE else 0)
            ms = ms | MSTATUS_MPIE
            ms = ms & ~MSTATUS_MPP  # MPP back to U, the least privileged supported
            self.csr[0x300] = ms
            self.priv = new_priv if new_priv in (0, 3) else 0
            self.pc = self.csr.get(0x341, 0) & M32
            return
        if w == 0x10500073:  # WFI: sleep until the timer fires
            if self.csr.get(0x304, 0) & MIP_MTIP and self.mtime < self.mtimecmp:
                self.mtime = self.mtimecmp
            self.pc = nxt
            return
        if w == 0x10200073:  # SRET: there is no S-mode
            raise _Trap(C_ILLEGAL, w)
        opc = w & 0x7F
        if opc == 0x2F:  # A extension, word ops only
            f3, f5 = (w >> 12) & 7, (w >> 27) & 0x1F
            name = AMO5.get(f5)
            if f3 != 2 or name is None:
                raise _Trap(C_ILLEGAL, w)
            rd, rs1, rs2 = (w >> 7) & 31, (w >> 15) & 31, (w >> 20) & 31
            ea = x[rs1] & M32
            if ea & 3:
                raise _Trap(C_STORE_MISALIGNED, ea)
            old = self.read_mem(ea, 4)
            src = x[rs2]
            if name == "LR":
                self.reservation = ea
                if rd:
                    x[rd] = old
            elif name == "SC":
                ok = self.reservation == ea
                if ok:
                    self.write_mem(ea, src, 4)
                self.reservation = None
                if rd:
                    x[rd] = 0 if ok else 1
            else:
                if name == "AMOADD":
                    nv = (old + src) & M32
                elif name == "AMOSWAP":
                    nv = src
                elif name == "AMOXOR":
                    nv = old ^ src
                elif name == "AMOOR":
                    nv = old | src
                elif name == "AMOAND":
                    nv = old & src
                elif name == "AMOMIN":
                    nv = old if sx(old, 32) < sx(src, 32) else src
                elif name == "AMOMAX":
                    nv = old if sx(old, 32) > sx(src, 32) else src
                elif name == "AMOMINU":
                    nv = min(old, src)
                else:  # AMOMAXU
                    nv = max(old, src)
                self.write_mem(ea, nv & M32, 4)
                if rd:
                    x[rd] = old
            self.pc = nxt
            return

        ins = decode(w)
        op = ins.op
        a, b = x[ins.rs1], x[ins.rs2]
        res = None
        if op == "LUI":
            res = ins.imm & M32
        elif op == "AUIPC":
            res = (pc + ins.imm) & M32
        elif op in ("JAL", "JALR"):
            res = nxt
            nxt = (pc + ins.imm) & M32 if op == "JAL" else (a + ins.imm) & M32 & ~1
        elif op in ref.BRANCH_OPS:
            if taken(op, a, b):
                nxt = (pc + ins.imm) & M32
        elif op in ref.LOAD_OPS:
            ea = (a + ins.imm) & M32
            n = {"LB": 1, "LBU": 1, "LH": 2, "LHU": 2, "LW": 4}[op]
            if ea % n:
                raise _Trap(C_LOAD_MISALIGNED, ea)
            v = self.read_mem(ea, n)
            res = (sx(v, 8 * n) if op in ("LB", "LH") else v) & M32
        elif op in ref.STORE_OPS:
            ea = (a + ins.imm) & M32
            n = {"SB": 1, "SH": 2, "SW": 4}[op]
            if ea % n:
                raise _Trap(C_STORE_MISALIGNED, ea)
            self.write_mem(ea, b, n)
        elif op == "ECALL":
            raise _Trap(C_ECALL_M if self.priv == 3 else C_ECALL_U, 0)
        elif op == "EBREAK":
            raise _Trap(C_EBREAK, pc)
        elif op == "ILLEGAL" or op in ref.CUSTOM:
            raise _Trap(C_ILLEGAL, w)
        elif op.startswith("CSR"):
            csr = ins.imm
            old = self.read_csr(csr)
            if op.endswith("I"):
                operand, is_imm = ins.rs1, True
            else:
                operand, is_imm = a, False
            base = op[:-1] if is_imm else op  # CSRRWI -> CSRRW, etc.
            if base == "CSRRW":
                self.write_csr(csr, operand)
            elif base == "CSRRS":
                if operand or is_imm and ins.rs1:
                    self.write_csr(csr, old | operand)
            elif base == "CSRRC":
                if operand or is_imm and ins.rs1:
                    self.write_csr(csr, old & ~operand & M32)
            res = old
        elif op == "FENCE":
            pass
        elif op in ref.IMM_OPS:
            res = alu(op, a, ins.imm & M32)
        else:
            res = alu(op, a, b)
        if res is not None and ins.rd:
            x[ins.rd] = res & M32
        self.pc = nxt & M32
