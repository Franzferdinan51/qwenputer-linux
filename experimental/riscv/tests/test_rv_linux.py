"""Tests for chipformer.rv.linux.LinuxMachine — the Linux-capable reference machine.

Programs are hand-encoded words placed in RAM at 0x80000000 (the assembler's
sections target the circuit's ROM map, not this machine's, so tests encode
with asm's encoders directly).
"""

from chipformer.rv.asm import enc_i, enc_j, enc_r, enc_s, enc_u
from chipformer.rv.linux import QWEN_BASE, RAM_BASE, UART_BASE, LinuxMachine

A0, A1, T0, T1, T2, S0, S1 = 10, 11, 5, 6, 7, 8, 9
ECALL, EBREAK, MRET, WFI = 0x00000073, 0x00100073, 0x30200073, 0x10500073


def addi(rd, rs1, imm):
    return enc_i(imm, rs1, 0, rd, 0x13)


def lw(rd, rs1, off=0):
    return enc_i(off, rs1, 2, rd, 0x03)


def lb(rd, rs1, off=0):
    return enc_i(off, rs1, 0, rd, 0x03)


def sw(rs2, rs1, off=0):
    return enc_s(off, rs2, rs1, 2)


def sb(rs2, rs1, off=0):
    return enc_s(off, rs2, rs1, 0)


def lui(rd, imm20):
    return enc_u(imm20, rd, 0x37)


def csrrw(rd, csr, rs1):
    return enc_i(csr, rs1, 1, rd, 0x73)


def csrrs(rd, csr, rs1):
    return enc_i(csr, rs1, 2, rd, 0x73)


def csrrc(rd, csr, rs1):
    return enc_i(csr, rs1, 3, rd, 0x73)


def machine(words, at=RAM_BASE):
    m = LinuxMachine()
    for i, w in enumerate(words):
        m.ram[at - RAM_BASE + 4 * i:at - RAM_BASE + 4 * i + 4] = w.to_bytes(4, "little")
    m.pc = at
    return m


def test_amoadd_and_swap():
    addr = RAM_BASE + 0x1000
    m = machine([
        lui(T0, 0x80001),          # t0 = addr (0x80000000 + 0x1000)
        addi(T1, 0, 1),
        enc_r(0x00, T1, T0, 2, A0, 0x2F),   # amoadd.w a0, t1, (t0)
        enc_r(0x04, T1, T0, 2, A1, 0x2F),   # amoswap.w a1, t1, (t0)
    ])
    m.ram[addr - RAM_BASE:addr - RAM_BASE + 4] = (41).to_bytes(4, "little")
    m.run(4)
    assert m.x[A0] == 41                       # amoadd returns the old value
    assert int.from_bytes(m.ram[addr - RAM_BASE:addr - RAM_BASE + 4], "little") == 1
    assert m.x[A1] == 42                       # swap returns the pre-swap value


def test_lr_sc():
    addr = RAM_BASE + 0x1000
    m = machine([
        lui(T0, 0x80001),
        enc_r(0x08, 0, T0, 2, A0, 0x2F),    # lr.w a0, (t0)
        addi(T1, 0, 7),
        enc_r(0x0C, T1, T0, 2, A1, 0x2F),   # sc.w a1, t1, (t0)
        enc_r(0x0C, T1, T0, 2, A1 + 1, 0x2F),  # sc.w a2, t1, (t0): no reservation now
    ])
    m.ram[addr - RAM_BASE:addr - RAM_BASE + 4] = (99).to_bytes(4, "little")
    m.run(5)
    assert m.x[A0] == 99
    assert m.x[A1] == 0                        # first sc succeeded
    assert int.from_bytes(m.ram[addr - RAM_BASE:addr - RAM_BASE + 4], "little") == 7
    assert m.x[A1 + 1] == 1                    # second sc failed, memory untouched


def test_csr_read_write_set_clear():
    m = machine([
        addi(T1, 0, 0x5A),
        csrrw(0, 0x340, T1),        # mscratch = 0x5a
        csrrs(A0, 0x340, 0),        # a0 = mscratch (pure read: rs1 = x0)
        addi(T1, 0, 0x0F),
        csrrs(0, 0x340, T1),        # mscratch |= 0x0f
        addi(T1, 0, 0x03),
        csrrc(0, 0x340, T1),        # mscratch &= ~0x03
        csrrs(A1, 0x340, 0),
    ])
    m.run(8)
    assert m.x[A0] == 0x5A
    assert m.x[A1] == 0x5C
    assert m.read_csr(0x340) == 0x5C


def _trap_program():
    """mtvec at +0x40; main: set mtvec, ecall, then a0 = 42. Handler records
    mcause in s0 and mepc in s1, skips the ecall, mret."""
    main = [
        lui(T0, 0x80000),           # t0 = RAM_BASE
        addi(T0, T0, 0x40),         # t0 = handler address
        csrrw(0, 0x305, T0),        # mtvec = t0
        ECALL,
        addi(A0, 0, 42),
        enc_j(0, 0),                # halt loop
    ]
    handler = [
        csrrs(S0, 0x342, 0),        # s0 = mcause
        csrrs(S1, 0x341, 0),        # s1 = mepc
        addi(S1, S1, 4),
        csrrw(0, 0x341, S1),        # mepc += 4
        MRET,
    ]
    words = main + [0] * (0x40 // 4 - len(main)) + handler
    return words


def test_ecall_from_m_traps_and_mret_returns():
    m = machine(_trap_program())
    m.run(20)
    assert m.x[S0] == 11            # ecall from M-mode
    assert m.x[A0] == 42            # execution resumed after the ecall
    assert m.priv == 3


def test_ecall_from_u_traps_with_cause_8():
    words = _trap_program()
    # main: set mtvec, then mret into U at +0x20 (an ecall sits there)
    main = [
        lui(T0, 0x80000),
        addi(T0, T0, 0x40),
        csrrw(0, 0x305, T0),        # mtvec
        lui(T1, 0x80000),
        addi(T1, T1, 0x20),
        csrrw(0, 0x341, T1),        # mepc = U entry
        csrrw(0, 0x300, 0),         # mstatus = 0 -> MPP = U
        MRET,
    ]
    u_code = [ECALL, enc_j(0, 0)]   # after mret, park in a loop
    words = main + [0] * (0x20 // 4 - len(main)) + u_code
    handler = [
        csrrs(S0, 0x342, 0),
        csrrs(S1, 0x341, 0),
        addi(S1, S1, 4),
        csrrw(0, 0x341, S1),
        MRET,
    ]
    words = words + [0] * (0x40 // 4 - len(words)) + handler
    m = machine(words)
    m.run(20)
    assert m.x[S0] == 8             # ecall from U-mode
    assert m.x[S1] == RAM_BASE + 0x24   # handler read mepc (0x20) and advanced it past the ecall


def test_timer_interrupt_and_wfi():
    # handler at +0x80: bump a counter at RAM+0x200, push mtimecmp far out, mret
    handler = [
        lui(T0, 0x80000),
        lw(T1, T0, 0x200),
        addi(T1, T1, 1),
        sw(T1, T0, 0x200),
        lui(T0, 0x11004),           # CLINT mtimecmp lo
        addi(T1, 0, -1),
        sw(T1, T0, 0),
        lui(T0, 0x11004),
        sw(T1, T0, 4),              # mtimecmp hi = 0xffffffff too via t1 = -1
        MRET,
    ]
    main = [
        lui(T0, 0x80000),
        addi(T0, T0, 0x80),
        csrrw(0, 0x305, T0),        # mtvec = handler
        addi(T1, 0, 0x80),
        csrrs(0, 0x304, T1),        # mie.MTIE = 1 (bit 7)
        addi(T1, 0, 8),
        csrrs(0, 0x300, T1),        # mstatus.MIE = 1 (bit 3)
        lui(T0, 0x11004),
        addi(T1, 0, 50),
        sw(T1, T0, 0),              # mtimecmp lo = 50
        sw(0, T0, 4),               # mtimecmp hi = 0
        WFI,                        # sleep until the timer fires
        addi(A0, 0, 1),             # reached after the interrupt + mret
        enc_j(0, 0),                # halt loop (jal x0, 0)
    ]
    words = main + [0] * (0x80 // 4 - len(main)) + handler
    m = machine(words)
    m.run(60)
    ctr = int.from_bytes(m.ram[0x200:0x204], "little")
    assert ctr == 1                 # the timer interrupt fired exactly once
    assert m.x[A0] == 1             # wfi woke and mret resumed main
    assert m.read_csr(0x300) & 8    # mret restored mstatus.MIE (regression: it once set MPIE's bit)


def test_uart_out_lsr_and_rx():
    m = machine([
        lui(T0, 0x10000),           # UART base
        addi(T1, 0, 0x41),          # 'A'
        sb(T1, T0, 0),              # THR = 'A'
        lb(A0, T0, 5),              # LSR: 0x60 with empty RX
        lb(A1, T0, 0),              # RBR: the fed byte
    ])
    m.feed_input(b"Z")
    m.run(4)
    assert bytes(m.out) == b"A"
    assert m.x[A0] == 0x61          # LSR: THR empty + RX ready (input was queued)
    m.run(1)
    assert m.x[A1] == ord("Z")


def test_syscon_poweroff():
    m = machine([
        lui(T0, 0x11100),           # SYSCON
        addi(T1, 0, 0x555),
        enc_i(4, T1, 1, T1, 0x13),  # slli t1, t1, 4
        addi(T1, T1, 5),            # t1 = 0x5555
        sw(T1, T0, 0),
        addi(A0, 0, 1),             # never reached
    ])
    m.run(6)
    assert m.halted and m.exit_reason == "poweroff"
    assert m.x[A0] == 0


def test_misaligned_load_traps():
    m = machine([
        lui(T0, 0x80000),
        addi(T0, T0, 0x40),
        csrrw(0, 0x305, T0),        # mtvec = +0x40 (zeros = illegal, but we only check state)
        lui(T1, 0x80001),
        lw(A0, T1, 2),              # misaligned lw
    ])
    m.run(5)
    assert m.read_csr(0x342) & 0x7FFFFFFF == 4       # load address misaligned
    assert m.read_csr(0x343) == RAM_BASE + 0x1002    # mtval = the bad address
    assert m.read_csr(0x341) == RAM_BASE + 16        # mepc = the lw itself


def test_subword_store_masks_to_width():
    # a register holding 0xFFFFFFFF stored as a byte/half must not overflow
    # and must write only its low bits (this crashed the first kernel boot)
    m = machine([
        lui(T0, 0x80001),
        addi(T1, 0, -1),
        sb(T1, T0, 0),
        enc_s(2, T1, T0, 1),        # sh t1, 2(t0)
        lw(A0, T0, 0),
    ])
    m.run(5)
    assert m.x[A0] == 0xFFFF00FF


def test_qwen_device_mmio_roundtrip():
    m = LinuxMachine()
    seen = []
    m.on_qwen_prompt = lambda p: seen.append(p) or f"qwen-echo: {p}"
    for ch in b"hi":
        m.write_mem(QWEN_BASE, ch, 1)            # DATA: prompt bytes
    m.write_mem(QWEN_BASE + 8, 1, 1)             # CTRL: submit
    assert m.read_mem(QWEN_BASE + 4, 1) == 1     # STATUS: answer waiting
    got = bytearray()
    while (b := m.read_mem(QWEN_BASE, 1)) != 0:  # DATA: answer, NUL-terminated
        got.append(b)
    assert bytes(got) == b"qwen-echo: hi"
    assert m.read_mem(QWEN_BASE + 4, 1) == 0     # drained
    m.write_mem(QWEN_BASE, ord("x"), 1)
    m.write_mem(QWEN_BASE + 8, 2, 1)             # CTRL: clear
    m.write_mem(QWEN_BASE + 8, 1, 1)             # submit the (now empty) prompt
    assert seen == ["hi", ""]                    # the 'x' never reached the model
    assert m.read_mem(QWEN_BASE, 1) == ord("q")  # "qwen-echo: " still answers


def test_qwen_device_no_backend_still_answers():
    m = LinuxMachine()
    m.write_mem(QWEN_BASE, ord("?"), 1)
    m.write_mem(QWEN_BASE + 8, 1, 1)
    assert m.read_mem(QWEN_BASE, 1) == ord("[")  # "[qwen: no model attached]"


def test_qwen_device_guest_program():
    # the whole exchange through real load/store instructions
    m = machine([
        lui(T0, 0x10010),        # t0 = 0x10010000, the Qwen window
        addi(T1, 0, 65),         # 'A'
        sb(T1, T0, 0),           # DATA <- 'A'
        addi(T1, 0, 1),
        sb(T1, T0, 8),           # CTRL <- submit
        lb(A0, T0, 4),           # STATUS
        lb(A1, T0, 0),           # first answer byte
    ])
    m.on_qwen_prompt = lambda p: f"qwen-echo: {p}"
    m.run(7)
    assert m.x[A0] == 1
    assert m.x[A1] == ord("q")   # "qwen-echo: A"


def test_uart_iir_reflects_rx_state():
    # irq=0 port: the guest's backup-timer poll reads IIR to decide which
    # side to service. It must show RDA while input is queued - a constant
    # NO_INT starved console input after boot (found 2026-10-31: getty
    # never consumed a single byte in 100M steps) - and THRE otherwise:
    # THRE is the only kick that restarts a blocked transmitter, and
    # without it guest writes past the inline chunk (~130 bytes) slept
    # forever (found 2026-10-31: the agent harness's 700-char prompt
    # stalled mid-line).
    m = LinuxMachine()
    assert m.read_mem(UART_BASE + 2, 1) == 0x02  # THRE, queue empty
    m.feed_input(b"root\n")
    assert m.read_mem(UART_BASE + 2, 1) == 0x04  # RDA, bytes waiting
    for _ in range(5):
        m.read_mem(UART_BASE, 1)
    assert m.read_mem(UART_BASE + 2, 1) == 0x02  # drained: back to THRE


def test_qwen_console_line_protocol():
    # the guest prints "QWEN:hi" on its console; the answer is typed back
    words = [lui(T0, 0x10000)]  # UART base
    for ch in b"QWEN:hi\n":
        words += [addi(T1, 0, ch), sb(T1, T0, 0)]
    m = machine(words)
    seen = []
    m.on_qwen_prompt = lambda p: seen.append(p) or f"qwen-echo: {p}"
    m.run(len(words))
    assert seen == ["hi"]
    # the QWENANS: prefix marks the line machine-typed (guest readers
    # strip it; see linux._qwen_line_feed and env/linux/ask.sh)
    assert bytes(m.inp) == b"QWENANS:qwen-echo: hi\n"  # queued as console input
    assert b"QWEN:hi\n" in bytes(m.out)        # the request line stays visible
