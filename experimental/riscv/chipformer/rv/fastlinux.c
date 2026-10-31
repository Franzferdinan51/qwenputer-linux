// fastlinux.c — a fast native core for the Qwenputer Linux machine.
//
// chipformer/rv/linux.py is the semantics oracle (itself checked
// against the circuit's ref.py lineage), but CPython retires only
// ~0.5M instructions/s: a Linux boot is a minute, an agent test ten.
// This interpreter mirrors linux.py function for function — RV32IMA +
// Zba + Zicsr, M/U privilege, traps, CLINT, 8250 UART, SYSCON, and the
// Qwen coprocessor (MMIO window + QWEN: console protocol) — and runs
// two to three orders of magnitude faster. Same trick as the Doom
// port's fastemu, transplanted from the bare machine to the Linux
// machine: the enhancements travel, the oracle stays Python.
//
// Determinism is identical to the oracle: mtime advances 1 tick per
// retired instruction (traps and interrupts do not retire), no wall
// clock anywhere — a run replays exactly, and the HASH command below
// produces the same FNV-1a state digest the Python side computes.
//
// Host protocol (the model's seat stays in Python):
//   stdin/stdout (fd 0/1): the guest console, raw bytes.
//   fd 3 (host -> core): line commands
//       RUN <n>            run up to n steps, then answer DONE
//       FEED <len>\n<bytes>  queue console input
//       HASH               answer with the state digest
//       QUIT
//   fd 4 (core -> host):
//       READY                       after the image is loaded
//       DONE <steps> <halted> <reason: none|poweroff|reboot>
//       HASH <16 hex>
//       QWEN <len>\n<bytes>   a console-protocol prompt: the core blocks
//                              until the host answers on fd 3 with
//                              ANS <len>\n<bytes> (answer is flattened
//                              to one line by the core, like linux.py)
//       QWENM <len>\n<bytes>  an MMIO submission: same answer form, used
//                              raw (+ NUL) as the device response
//   With FASTLINUX_HOST=0 there is no seat: console prompts are
//   dropped and MMIO submissions answer "[qwen: no model attached]",
//   mirroring on_qwen_prompt = None.
//
// Build:  cc -O3 -o fastlinux fastlinux.c
// Usage:  fastlinux <Image> <dtb>   (control on fds 3 and 4)

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RAM_BASE   0x80000000u
#define RAM_SIZE   (64u * 1024 * 1024)
#define UART_BASE  0x10000000u
#define QWEN_BASE  0x10010000u
#define CLINT_BASE 0x11000000u
#define SYSCON_BASE 0x11100000u
#define MMIO_LO    0x10000000u
#define MMIO_HI    0x12000000u
#define M32        0xFFFFFFFFu
#define MISA       0x40401101u

static uint8_t *ram;
static uint32_t x[32], pc;
static int priv = 3;                       // 3 = M, 0 = U
static uint64_t steps, mtime, mtimecmp = 0xFFFFFFFFFFFFFFFFull;
static uint32_t msip;
static int64_t reservation = -1;
static uint32_t csr[4096];
static uint64_t csr_fallback;
static uint8_t uart_regs[256];
static int halted;
static const char *exit_reason = "none";
static int host_attached = 1;

// byte queues (growable, head-indexed)
typedef struct { uint8_t *b; size_t len, cap, head; } Queue;
static Queue inp, qwen_resp;
static uint8_t *qwen_prompt; static size_t qp_len, qp_cap;
static uint8_t *qwen_line; static size_t ql_len, ql_cap;

static void qpush(Queue *q, uint8_t v) {
    if (q->head && q->head == q->len) q->head = q->len = 0;
    if (q->len == q->cap) {
        if (q->head) {
            memmove(q->b, q->b + q->head, q->len - q->head);
            q->len -= q->head; q->head = 0;
        } else {
            q->cap = q->cap ? 2 * q->cap : 4096;
            q->b = realloc(q->b, q->cap);
            if (!q->b) { perror("realloc"); exit(1); }
        }
    }
    q->b[q->len++] = v;
}
static int qpop(Queue *q) {
    return q->head < q->len ? q->b[q->head++] : -1;
}
static size_t qlen(const Queue *q) { return q->len - q->head; }
static void bufpush(uint8_t **b, size_t *len, size_t *cap, uint8_t v) {
    if (*len == *cap) {
        *cap = *cap ? 2 * *cap : 512;
        *b = realloc(*b, *cap);
        if (!*b) { perror("realloc"); exit(1); }
    }
    (*b)[(*len)++] = v;
}

// ---- console output: buffered, flushed before any fd-4 traffic ----
static uint8_t outbuf[65536];
static size_t outlen;
static void out_flush(void) {
    if (outlen) { (void)!write(1, outbuf, outlen); outlen = 0; }
}
static void out_byte(uint8_t v) {
    outbuf[outlen++] = v;
    if (outlen == sizeof outbuf) out_flush();
}

// ---- host channel ----
static FILE *ctl_in;                       // fd 3
static int ctl_out = 4;                    // fd 4
static void ctl_write(const char *s, size_t n) {
    out_flush();
    while (n) {
        ssize_t w = write(ctl_out, s, n);
        if (w <= 0) exit(0);
        s += w; n -= (size_t)w;
    }
}
static void ctl_line(const char *s) { ctl_write(s, strlen(s)); }

// read one ANS frame from fd 3: "ANS <len>\n" + len raw bytes.
// returns a malloc'd buffer (len_out set), or NULL on EOF.
static uint8_t *ctl_read_ans(size_t *len_out) {
    char line[64];
    if (!fgets(line, sizeof line, ctl_in)) return NULL;
    unsigned n;
    if (sscanf(line, "ANS %u", &n) != 1) return NULL;
    uint8_t *b = malloc(n ? n : 1);
    if (n && fread(b, 1, n, ctl_in) != n) { free(b); return NULL; }
    *len_out = n;
    return b;
}

// send a prompt to the seat and wait for its answer (blocking, like the
// device itself: the CPU waits on its coprocessor)
static uint8_t *ask_host(const char *kind, const uint8_t *prompt, size_t plen,
                         size_t *alen) {
    char hdr[64];
    int hn = snprintf(hdr, sizeof hdr, "%s %zu\n", kind, plen);
    ctl_write(hdr, (size_t)hn);
    if (plen) ctl_write((const char *)prompt, plen);
    uint8_t *a = ctl_read_ans(alen);
    if (!a) { *alen = 0; return (uint8_t *)""; }
    return a;
}

// ---- traps ----
static int trapped; static uint32_t trap_cause, trap_tval;
static void raise_trap(uint32_t cause, uint32_t tval) {
    trapped = 1; trap_cause = cause; trap_tval = tval;
}
#define C_INST_MISALIGNED 0
#define C_INST_FAULT 1
#define C_ILLEGAL 2
#define C_EBREAK 3
#define C_LOAD_MISALIGNED 4
#define C_LOAD_FAULT 5
#define C_STORE_MISALIGNED 6
#define C_STORE_FAULT 7
#define C_ECALL_U 8
#define C_ECALL_M 11

static void take_trap(uint32_t cause, uint32_t tval, int interrupt, uint32_t epc) {
    csr[0x341] = epc;
    csr[0x342] = cause | (interrupt ? 0x80000000u : 0);
    csr[0x343] = tval;
    uint32_t ms = csr[0x300];
    ms = (ms & ~(3u << 11)) | ((uint32_t)priv << 11);
    ms = (ms & ~(1u << 7)) | ((ms & (1u << 3)) ? (1u << 7) : 0);
    csr[0x300] = ms & ~(1u << 3);
    priv = 3;
    uint32_t mtvec = csr[0x305];
    uint32_t base = mtvec & ~3u;
    pc = ((mtvec & 3) == 1 && interrupt) ? base + 4 * cause : base;
}

static uint32_t mip_value(void) {
    uint32_t v = 0;
    if (msip) v |= 1u << 3;
    if (mtime >= mtimecmp) v |= 1u << 7;
    return v;
}

static int pending_interrupt(void) {
    uint32_t pend = mip_value() & csr[0x304];
    if (!pend) return -1;
    if (!(priv < 3 || (csr[0x300] & (1u << 3)))) return -1;
    if (pend & (1u << 7)) return 7;        // timer first, like the oracle
    if (pend & (1u << 3)) return 3;
    return -1;
}

// ---- CSRs ----
static uint32_t read_csr(uint32_t c) {
    switch (c) {
    case 0x301: return MISA;
    case 0x344: return mip_value();
    case 0xC00: case 0xC02: return (uint32_t)steps;
    case 0xC80: case 0xC82: return (uint32_t)(steps >> 32);
    case 0xC01: return (uint32_t)mtime;
    case 0xC81: return (uint32_t)(mtime >> 32);
    case 0xF11: case 0xF12: case 0xF13: case 0xF14: return 0;
    default: break;
    }
    if (c < 4096 && csr[c]) return csr[c];
    // a stored 0 and an unknown CSR both read 0; only the unknown count
    // differs (host-side diagnostics in the oracle, invisible to guests)
    static const uint16_t known[] = {0x300, 0x302, 0x303, 0x304, 0x305, 0x306,
        0x340, 0x341, 0x342, 0x343, 0x3A0, 0x3B0, 0x3B1, 0x3B2, 0x3B3};
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (csr[c] || c == known[i]) return csr[c];
    csr_fallback++;
    return 0;
}

static void write_csr(uint32_t c, uint32_t v) {
    if (c == 0x301 || c == 0x344 || c == 0xF11 || c == 0xF12 || c == 0xF13 ||
        c == 0xF14 || (c >= 0xC00 && c <= 0xC02) || (c >= 0xC80 && c <= 0xC82))
        return;                            // read-only views
    switch (c) {
    case 0x300: case 0x302: case 0x303: case 0x304: case 0x305: case 0x306:
    case 0x340: case 0x341: case 0x342: case 0x343:
    case 0x3A0: case 0x3B0: case 0x3B1: case 0x3B2: case 0x3B3:
        csr[c] = v;
        return;
    default:
        csr_fallback++;
    }
}

// ---- devices ----
static void qwen_submit_mmio(void);
static void qwen_line_feed(uint8_t v);

static int qwen_read(int off) {
    if (off == 0) { int b = qpop(&qwen_resp); return b < 0 ? 0 : b; }
    if (off == 4) return qlen(&qwen_resp) ? 1 : 0;
    return 0;
}
static void qwen_write(int off, uint8_t v) {
    if (off == 0) bufpush(&qwen_prompt, &qp_len, &qp_cap, v);
    else if (off == 8) {
        if (v == 1) qwen_submit_mmio();
        else if (v == 2) { qp_len = 0; qwen_resp.head = qwen_resp.len = 0; }
    }
}
static void qwen_submit_mmio(void) {
    if (!host_attached) {
        static const char msg[] = "[qwen: no model attached]";
        for (const char *p = msg; *p; p++) qpush(&qwen_resp, (uint8_t)*p);
        qpush(&qwen_resp, 0);
        qp_len = 0;
        return;
    }
    size_t alen;
    uint8_t *a = ask_host("QWENM", qwen_prompt, qp_len, &alen);
    qp_len = 0;
    for (size_t i = 0; i < alen; i++) qpush(&qwen_resp, a[i]);
    qpush(&qwen_resp, 0);
    if (alen) free(a);
}

static int uart_read(int off) {
    if (off == 0) {
        if (uart_regs[3] & 0x80) return uart_regs[0];
        int b = qpop(&inp);
        return b < 0 ? 0 : b;
    }
    if (off == 2) return qlen(&inp) ? 0x04 : 0x02;   // IIR: RDA / THRE
    if (off == 5) return 0x60 | (qlen(&inp) ? 1 : 0);
    return uart_regs[off];
}
static void uart_write(int off, uint8_t v) {
    if (off == 0 && !(uart_regs[3] & 0x80)) {
        out_byte(v);
        qwen_line_feed(v);
        return;
    }
    uart_regs[off] = v;
}

static void qwen_line_feed(uint8_t v) {
    if (v == 0x0A) {
        if (ql_len >= 5 && !memcmp(qwen_line, "QWEN:", 5) && host_attached) {
            // strip, like the oracle's .strip() on the decoded prompt
            const uint8_t *p = qwen_line + 5;
            size_t n = ql_len - 5;
            while (n && (p[0] == ' ' || p[0] == '\t' || p[0] == '\r')) { p++; n--; }
            while (n && (p[n-1] == ' ' || p[n-1] == '\t' || p[n-1] == '\r')) n--;
            size_t alen;
            uint8_t *a = ask_host("QWEN", p, n, &alen);
            // flatten to one line: whitespace runs -> single spaces
            int pending_space = 0, started = 0;
            for (size_t i = 0; i < alen; i++) {
                uint8_t c = a[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                    c == '\v' || c == '\f') { pending_space = started; continue; }
                if (pending_space) qpush(&inp, ' ');
                pending_space = 0; started = 1;
                qpush(&inp, c);
            }
            qpush(&inp, '\n');
            if (alen) free(a);
        }
        ql_len = 0;
    } else if (v != 0x0D) {
        bufpush(&qwen_line, &ql_len, &ql_cap, v);
        if (ql_len > 4096) ql_len = 0;
    }
}

static uint32_t mmio_read(uint32_t addr, int n) {
    if (addr >= QWEN_BASE && addr < QWEN_BASE + 0x10) {
        uint32_t v = 0;
        for (int i = 0; i < n; i++) v |= (uint32_t)qwen_read(addr - QWEN_BASE + i) << 8 * i;
        return v;
    }
    if (addr >= UART_BASE && addr < UART_BASE + 0x100) {
        uint32_t v = 0;
        for (int i = 0; i < n; i++) v |= (uint32_t)uart_read(addr - UART_BASE + i) << 8 * i;
        return v;
    }
    if (addr == CLINT_BASE) return msip & 1;
    if (addr == CLINT_BASE + 0x4000) return (uint32_t)mtimecmp;
    if (addr == CLINT_BASE + 0x4004) return (uint32_t)(mtimecmp >> 32);
    if (addr == CLINT_BASE + 0xBFF8) return (uint32_t)mtime;
    if (addr == CLINT_BASE + 0xBFFC) return (uint32_t)(mtime >> 32);
    return 0;
}
static void mmio_write(uint32_t addr, uint32_t v, int n) {
    if (addr >= QWEN_BASE && addr < QWEN_BASE + 0x10) {
        for (int i = 0; i < n; i++) qwen_write(addr - QWEN_BASE + i, v >> 8 * i & 0xFF);
        return;
    }
    if (addr >= UART_BASE && addr < UART_BASE + 0x100) {
        for (int i = 0; i < n; i++) uart_write(addr - UART_BASE + i, v >> 8 * i & 0xFF);
        return;
    }
    if (addr == CLINT_BASE) { msip = v & 1; return; }
    if (addr == CLINT_BASE + 0x4000 || addr == CLINT_BASE + 0x4004) {
        int sh = (addr & 4) ? 32 : 0;
        mtimecmp = (mtimecmp & ~(0xFFFFFFFFull << sh)) | ((uint64_t)v << sh);
        return;
    }
    if (addr == CLINT_BASE + 0xBFF8 || addr == CLINT_BASE + 0xBFFC) {
        int sh = (addr & 4) ? 32 : 0;
        mtime = (mtime & ~(0xFFFFFFFFull << sh)) | ((uint64_t)v << sh);
        return;
    }
    if (addr == SYSCON_BASE) {
        if (v == 0x5555) { halted = 1; exit_reason = "poweroff"; }
        else if (v == 0x7777) { halted = 1; exit_reason = "reboot"; }
    }
}

static int in_ram(uint32_t addr, int n) {
    return addr >= RAM_BASE && (uint64_t)addr + n <= (uint64_t)RAM_BASE + RAM_SIZE;
}
static uint32_t read_mem(uint32_t addr, int n) {
    if (in_ram(addr, n)) {
        uint32_t v = 0;
        for (int i = 0; i < n; i++) v |= (uint32_t)ram[addr - RAM_BASE + i] << 8 * i;
        return v;
    }
    if (addr >= MMIO_LO && addr < MMIO_HI) return mmio_read(addr, n);
    raise_trap(C_LOAD_FAULT, addr);
    return 0;
}
static void write_mem(uint32_t addr, uint32_t v, int n) {
    if (in_ram(addr, n)) {
        for (int i = 0; i < n; i++) ram[addr - RAM_BASE + i] = v >> 8 * i & 0xFF;
        return;
    }
    if (addr >= MMIO_LO && addr < MMIO_HI) { mmio_write(addr, v, n); return; }
    raise_trap(C_STORE_FAULT, addr);
}

// ---- execute (mirrors linux.py _execute + ref.decode/alu) ----
static int32_t sx32(uint32_t v, int bits) {
    uint32_t m = (1u << bits) - 1;
    v &= m;
    return (int32_t)(v << (32 - bits)) >> (32 - bits);
}

static void execute(uint32_t w, uint32_t at) {
    uint32_t nxt = at + 4;
    if (w == 0x30200073u) {                // MRET
        uint32_t ms = csr[0x300];
        int np = (ms >> 11) & 3;
        ms = (ms & ~(1u << 3)) | ((ms & (1u << 7)) ? (1u << 3) : 0);
        ms |= 1u << 7;
        ms &= ~(3u << 11);
        csr[0x300] = ms;
        priv = (np == 0 || np == 3) ? np : 0;
        pc = csr[0x341];
        return;
    }
    if (w == 0x10500073u) {                // WFI
        if ((csr[0x304] & (1u << 7)) && mtime < mtimecmp) mtime = mtimecmp;
        pc = nxt;
        return;
    }
    if (w == 0x10200073u) { raise_trap(C_ILLEGAL, w); return; }  // SRET

    uint32_t opc = w & 0x7F, rd = w >> 7 & 31, f3 = w >> 12 & 7;
    uint32_t rs1 = w >> 15 & 31, rs2 = w >> 20 & 31, f7 = w >> 25;
    int32_t i_imm = sx32(w >> 20, 12);
    int32_t s_imm = sx32((w >> 25) << 5 | (w >> 7 & 31), 12);
    int32_t b_imm = sx32((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 |
                         (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13);
    int32_t j_imm = sx32((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 |
                         (w >> 20 & 1) << 11 | (w >> 21 & 0x3FF) << 1, 21);
    uint32_t a = x[rs1], b = x[rs2];
    uint32_t res = 0; int has_res = 0;

    if (opc == 0x2F) {                     // A extension, word ops
        static const int8_t amo_ok[32] = {1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,
                                          1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0};
        // funct5 (w >> 27, aq/rl ignored like the oracle): 0 add, 1 swap,
        // 2 lr, 3 sc, 4 xor, 8 or, 12 and, 16 min, 20 max, 24 minu, 28 maxu
        uint32_t f5 = w >> 27;
        if (f3 != 2 || !amo_ok[f5]) { raise_trap(C_ILLEGAL, w); return; }
        uint32_t ea = a;
        if (ea & 3) { raise_trap(C_STORE_MISALIGNED, ea); return; }
        uint32_t old = read_mem(ea, 4);
        if (trapped) return;
        if (f5 == 2) {                      // LR
            reservation = ea;
            if (rd) x[rd] = old;
        } else if (f5 == 3) {              // SC
            int ok = reservation == (int64_t)ea;
            if (ok) { write_mem(ea, b, 4); if (trapped) return; }
            reservation = -1;
            if (rd) x[rd] = ok ? 0 : 1;
        } else {
            uint32_t nv;
            switch (f5) {
            case 0:  nv = old + b; break;
            case 1:  nv = b; break;
            case 4:  nv = old ^ b; break;
            case 8:  nv = old | b; break;
            case 12: nv = old & b; break;
            case 16: nv = (int32_t)old < (int32_t)b ? old : b; break;
            case 20: nv = (int32_t)old > (int32_t)b ? old : b; break;
            case 24: nv = old < b ? old : b; break;
            default: nv = old > b ? old : b; break;   // 28 maxu
            }
            write_mem(ea, nv, 4);
            if (trapped) return;
            if (rd) x[rd] = old;
        }
        pc = nxt;
        return;
    }

    switch (opc) {
    case 0x37: res = (uint32_t)sx32(w & 0xFFFFF000u, 32); has_res = 1; break;  // LUI
    case 0x17: res = at + (uint32_t)sx32(w & 0xFFFFF000u, 32); has_res = 1; break; // AUIPC
    case 0x6F: res = nxt; nxt = at + (uint32_t)j_imm; has_res = 1; break;      // JAL
    case 0x67:                                                            // JALR
        if (f3 != 0) { raise_trap(C_ILLEGAL, w); return; }
        res = nxt; nxt = (a + (uint32_t)i_imm) & ~1u; has_res = 1;
        break;
    case 0x63: {                                                          // branches
        if (f3 == 2 || f3 == 3) { raise_trap(C_ILLEGAL, w); return; }
        int take;
        switch (f3) {
        case 0: take = a == b; break;
        case 1: take = a != b; break;
        case 4: take = (int32_t)a < (int32_t)b; break;
        case 5: take = (int32_t)a >= (int32_t)b; break;
        case 6: take = a < b; break;
        default: take = a >= b; break;
        }
        if (take) nxt = at + (uint32_t)b_imm;
        break;
    }
    case 0x03: {                                                          // loads
        if (f3 == 3 || f3 == 6 || f3 == 7) { raise_trap(C_ILLEGAL, w); return; }
        int n = (f3 == 0 || f3 == 4) ? 1 : (f3 == 1 || f3 == 5) ? 2 : 4;
        uint32_t ea = a + (uint32_t)i_imm;
        if (ea % (uint32_t)n) { raise_trap(C_LOAD_MISALIGNED, ea); return; }
        uint32_t v = read_mem(ea, n);
        if (trapped) return;
        res = (f3 == 0) ? (uint32_t)sx32(v, 8) : (f3 == 1) ? (uint32_t)sx32(v, 16) : v;
        has_res = 1;
        break;
    }
    case 0x23: {                                                          // stores
        if (f3 > 2) { raise_trap(C_ILLEGAL, w); return; }
        int n = f3 == 0 ? 1 : f3 == 1 ? 2 : 4;
        uint32_t ea = a + (uint32_t)s_imm;
        if (ea % (uint32_t)n) { raise_trap(C_STORE_MISALIGNED, ea); return; }
        write_mem(ea, b, n);
        if (trapped) return;
        break;
    }
    case 0x13: {                                                          // OP-IMM
        switch (f3) {
        case 0: res = a + (uint32_t)i_imm; break;
        case 1:
            if (f7 != 0) { raise_trap(C_ILLEGAL, w); return; }
            res = a << (rs2 & 31); break;
        case 2: res = (int32_t)a < i_imm; break;
        case 3: res = a < (uint32_t)i_imm; break;
        case 4: res = a ^ (uint32_t)i_imm; break;
        case 5:
            if (f7 == 0) res = a >> (rs2 & 31);
            else if (f7 == 0x20) res = (uint32_t)((int32_t)a >> (rs2 & 31));
            else { raise_trap(C_ILLEGAL, w); return; }
            break;
        case 6: res = a | (uint32_t)i_imm; break;
        default: res = a & (uint32_t)i_imm; break;
        }
        has_res = 1;
        break;
    }
    case 0x33: {                                                          // OP
        if (f7 == 0) {
            switch (f3) {
            case 0: res = a + b; break;
            case 1: res = a << (b & 31); break;
            case 2: res = (int32_t)a < (int32_t)b; break;
            case 3: res = a < b; break;
            case 4: res = a ^ b; break;
            case 5: res = a >> (b & 31); break;
            case 6: res = a | b; break;
            default: res = a & b; break;
            }
        } else if (f7 == 0x20 && f3 == 0) res = a - b;
        else if (f7 == 0x20 && f3 == 5) res = (uint32_t)((int32_t)a >> (b & 31));
        else if (f7 == 1) {
            switch (f3) {
            case 0: res = a * b; break;
            case 1: res = (uint32_t)((int64_t)(int32_t)a * (int32_t)b >> 32); break;
            case 2: res = (uint32_t)((int64_t)(int32_t)a * (uint64_t)b >> 32); break;
            case 3: res = (uint32_t)((uint64_t)a * b >> 32); break;
            case 4: {
                int32_t ia = (int32_t)a, ib = (int32_t)b;
                res = ib == 0 ? 0xFFFFFFFFu : (ia == INT32_MIN && ib == -1)
                      ? 0x80000000u : (uint32_t)(ia / ib);
                break;
            }
            case 5: res = b == 0 ? 0xFFFFFFFFu : a / b; break;
            case 6: {
                int32_t ia = (int32_t)a, ib = (int32_t)b;
                res = ib == 0 ? a : (ia == INT32_MIN && ib == -1) ? 0 : (uint32_t)(ia % ib);
                break;
            }
            case 7: res = b == 0 ? a : a % b; break;
            }
        } else if (f7 == 0x10 && f3 == 2) res = (a << 1) + b;
        else if (f7 == 0x10 && f3 == 4) res = (a << 2) + b;
        else if (f7 == 0x10 && f3 == 6) res = (a << 3) + b;
        else { raise_trap(C_ILLEGAL, w); return; }
        has_res = 1;
        break;
    }
    case 0x0F: break;                                                     // FENCE: nop
    case 0x73: {
        if (w == 0x00000073u) { raise_trap(priv == 3 ? C_ECALL_M : C_ECALL_U, 0); return; }
        if (w == 0x00100073u) { raise_trap(C_EBREAK, at); return; }
        if (f3 == 1 || f3 == 2 || f3 == 3 || f3 == 5 || f3 == 6 || f3 == 7) {
            uint32_t c = w >> 20;
            uint32_t old = read_csr(c);
            int is_imm = f3 >= 5;
            uint32_t operand = is_imm ? rs1 : a;
            if (f3 == 1 || f3 == 5) write_csr(c, operand);                // CSRRW(I)
            else if (f3 == 2 || f3 == 6) {                                // CSRRS(I)
                if (operand) write_csr(c, old | operand);
            } else {                                                      // CSRRC(I)
                if (operand) write_csr(c, old & ~operand);
            }
            res = old; has_res = 1;
            break;
        }
        raise_trap(C_ILLEGAL, w);
        return;
    }
    default: raise_trap(C_ILLEGAL, w); return;                            // incl. CUSTOM 0x0B/0x2B
    }
    if (has_res && rd) x[rd] = res;
    pc = nxt;
}

static void step(void) {
    if (halted) return;
    int irq = pending_interrupt();
    if (irq >= 0) { take_trap((uint32_t)irq, 0, 1, pc); return; }
    uint32_t at = pc;
    if (at & 3) { take_trap(C_INST_MISALIGNED, at, 0, at); return; }
    if (!in_ram(at, 4)) { take_trap(C_INST_FAULT, at, 0, at); return; }
    uint32_t w = (uint32_t)ram[at - RAM_BASE] | (uint32_t)ram[at - RAM_BASE + 1] << 8 |
                 (uint32_t)ram[at - RAM_BASE + 2] << 16 | (uint32_t)ram[at - RAM_BASE + 3] << 24;
    trapped = 0;
    execute(w, at);
    if (trapped) { take_trap(trap_cause, trap_tval, 0, at); return; }
    steps++;
    mtime += 1;
    x[0] = 0;
}

// ---- state digest (must match the Python parity script byte for byte) ----
static uint64_t fnv_state(void) {
    uint64_t h = 1469598103934665603ull;
#define MIXB(bv) do { h ^= (uint8_t)(bv); h *= 1099511628211ull; } while (0)
#define MIX32(v) do { for (int k = 0; k < 4; k++) MIXB((v) >> 8 * k); } while (0)
#define MIX64(v) do { for (int k = 0; k < 8; k++) MIXB((v) >> 8 * k); } while (0)
    MIX32(pc); MIX32((uint32_t)priv);
    for (int i = 0; i < 32; i++) MIX32(x[i]);
    MIX64(steps); MIX64(mtime);
    for (uint32_t i = 0; i < RAM_SIZE; i++) MIXB(ram[i]);
    return h;
#undef MIXB
#undef MIX32
#undef MIX64
}

// ---- image loading + host loop ----
static void load_file(const char *path, uint8_t *dst) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fread(dst, 1, (size_t)sz, f) != (size_t)sz) { perror("fread"); exit(1); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: fastlinux <Image> <dtb>\n");
        return 1;
    }
    const char *e = getenv("FASTLINUX_HOST");
    if (e && e[0] == '0') host_attached = 0;
    ram = calloc(1, RAM_SIZE);
    if (!ram) { perror("calloc"); return 1; }
    load_file(argv[1], ram);
    load_file(argv[2], ram + 0x3FFC000);
    x[11] = RAM_BASE + 0x3FFC000;
    pc = RAM_BASE;
    priv = 3;

    const char *ci = getenv("FASTLINUX_CTL_IN");
    const char *co = getenv("FASTLINUX_CTL_OUT");
    int in_fd = ci ? atoi(ci) : 3;
    if (co) ctl_out = atoi(co);
    ctl_in = fdopen(in_fd, "r");
    if (!ctl_in) { fprintf(stderr, "fastlinux: host command fd required\n"); return 1; }
    ctl_line("READY\n");

    char line[128];
    while (fgets(line, sizeof line, ctl_in)) {
        if (!strncmp(line, "RUN ", 4)) {
            uint64_t n = strtoull(line + 4, NULL, 10);
            for (uint64_t i = 0; i < n && !halted; i++) step();
            char rep[96];
            snprintf(rep, sizeof rep, "DONE %llu %d %s\n",
                     (unsigned long long)steps, halted, exit_reason);
            ctl_write(rep, strlen(rep));
        } else if (!strncmp(line, "FEED ", 5)) {
            unsigned n = strtoul(line + 5, NULL, 10);
            for (unsigned i = 0; i < n; i++) {
                int c = fgetc(ctl_in);
                if (c == EOF) return 0;
                qpush(&inp, (uint8_t)c);
            }
        } else if (!strncmp(line, "HASH", 4)) {
            char rep[40];
            snprintf(rep, sizeof rep, "HASH %016llx\n",
                     (unsigned long long)fnv_state());
            ctl_write(rep, strlen(rep));
        } else if (!strncmp(line, "QUIT", 4)) {
            return 0;
        }
        // ANS lines are consumed inside ask_host, never here
    }
    return 0;
}
