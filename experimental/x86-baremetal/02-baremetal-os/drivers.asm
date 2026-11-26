; ============================================================================
; os/drivers.asm — Pure x86-64 Bare-Metal Hardware Drivers
;
;   1. COM1 Serial UART (0x3F8, 115200 8N1) — raw UTF-8 TX & RX
;   2. VGA Text Mode (0xB8000, 80x25, color attributes, HW cursor, scrolling,
;      clean UTF-8 decoder with Cyrillic transliteration on VGA)
;   3. PS/2 Keyboard Controller (0x60/0x64 polling, scancode set 1 -> ASCII)
;   4. PIT Channel 2 + RDTSC calibrated nanosecond timer & microsecond delay
;   5. QEMU / Bochs / ACPI clean poweroff
; ============================================================================

bits 64
default rel

%include "os_constants.inc"

section .text

global serial_init
global serial_putc
global serial_try_getc
global vga_clear
global vga_set_color
global vga_putc_utf8
global console_write
global console_write_cstr
global console_putc
global console_readline
global ps2_try_getc
global pit_delay_us
global timer_init
global timer_ns
global qemu_shutdown
global g_tsc_hz
global g_tsc_mhz

extern g_boot_tsc

; ============================================================================
; 1. COM1 Serial Driver (0x3F8, 115200 8N1)
; ============================================================================
serial_init:
    mov     dx, COM1_IER
    xor     al, al
    out     dx, al                      ; Disable interrupts

    mov     dx, COM1_LCR
    mov     al, 0x80                    ; Enable DLAB
    out     dx, al

    mov     dx, COM1_PORT
    mov     al, 0x01                    ; Divisor low = 1 (115200 baud)
    out     dx, al
    mov     dx, COM1_IER
    xor     al, al                      ; Divisor high = 0
    out     dx, al

    mov     dx, COM1_LCR
    mov     al, 0x03                    ; 8 bits, no parity, 1 stop bit, DLAB=0
    out     dx, al

    mov     dx, COM1_FCR
    mov     al, 0xC7                    ; Enable FIFO, clear TX/RX, 14-byte thresh
    out     dx, al

    mov     dx, COM1_MCR
    mov     al, 0x03                    ; DTR + RTS
    out     dx, al
    ret

; void serial_putc(uint8 al) — preserves rdi, rsi, rdx? No, clobbers dx, r8b, al.
; Actually let's follow clean reg discipline: input in dil OR al; provide serial_putc(al)
; Wait: boot.asm calls `mov al, 10; call serial_putc`, so take byte in AL and preserve rdi/rsi/rcx!
serial_putc:
    mov     r8b, al
    mov     dx, COM1_LSR
.wait:
    in      al, dx
    test    al, 0x20                    ; THRE (Transmit Holding Register Empty)
    jz      .wait
    mov     dx, COM1_PORT
    mov     al, r8b
    out     dx, al
    ret

; int32 serial_try_getc(void) -> eax = byte (0..255) or -1 if empty
serial_try_getc:
    mov     dx, COM1_LSR
    in      al, dx
    test    al, 0x01                    ; DR (Data Ready)
    jz      .none
    mov     dx, COM1_PORT
    xor     eax, eax
    in      al, dx
    ret
.none:
    mov     eax, -1
    ret

; ============================================================================
; 2. VGA Text Mode Driver (0xB8000, 80x25)
; ============================================================================

; void vga_clear(uint8 attr in dil)
vga_clear:
    mov     [rel vga_attr], dil
    mov     dword [rel vga_row], 0
    mov     dword [rel vga_col], 0
    mov     dword [rel utf8_rem], 0
    mov     dword [rel utf8_cp], 0

    movzx   eax, dil
    shl     eax, 8
    or      eax, 0x20                   ; space + attr
    mov     edx, eax
    shl     eax, 16
    or      eax, edx                    ; 2 cells in eax
    mov     rdi, VGA_MEM
    mov     ecx, (VGA_COLS * VGA_ROWS) / 2
    rep     stosd
    call    vga_update_cursor
    ret

; void vga_set_color(uint8 attr in dil)
vga_set_color:
    mov     [rel vga_attr], dil
    ret

; Internal: update hardware VGA cursor via CRTC 0x3D4/0x3D5
vga_update_cursor:
    mov     eax, [rel vga_row]
    imul    eax, eax, VGA_COLS
    add     eax, [rel vga_col]
    mov     ecx, eax

    mov     dx, VGA_CRTC_IDX
    mov     al, 0x0F                    ; Cursor Location Low
    out     dx, al
    mov     dx, VGA_CRTC_DATA
    mov     al, cl
    out     dx, al

    mov     dx, VGA_CRTC_IDX
    mov     al, 0x0E                    ; Cursor Location High
    out     dx, al
    mov     dx, VGA_CRTC_DATA
    mov     al, ch
    out     dx, al
    ret

; Internal: scroll VGA screen up by 1 line
vga_scroll:
    push    rdi
    push    rsi
    push    rcx
    mov     rdi, VGA_MEM
    mov     rsi, VGA_MEM + (VGA_COLS * 2)
    mov     ecx, ((VGA_ROWS - 1) * VGA_COLS) / 2
    rep     movsd

    ; Clear bottom row with current attribute
    movzx   eax, byte [rel vga_attr]
    shl     eax, 8
    or      eax, 0x20
    mov     edx, eax
    shl     eax, 16
    or      eax, edx
    mov     ecx, VGA_COLS / 2
    rep     stosd
    mov     dword [rel vga_row], VGA_ROWS - 1
    pop     rcx
    pop     rsi
    pop     rdi
    ret

; Internal: emit a single ASCII byte (al) to VGA text buffer
vga_emit_ascii:
    cmp     al, 10                      ; '\n'
    je      .newline
    cmp     al, 13                      ; '\r'
    je      .cr
    cmp     al, 8                       ; '\b'
    je      .bs
    cmp     al, 9                       ; '\t'
    je      .tab

    ; Printable ASCII 0x20..0x7E
    cmp     al, 0x20
    jb      .ret
    cmp     al, 0x7E
    ja      .ret

    mov     ecx, [rel vga_row]
    imul    ecx, ecx, VGA_COLS
    add     ecx, [rel vga_col]
    mov     ah, [rel vga_attr]
    mov     rdx, VGA_MEM
    mov     [rdx + rcx*2], ax

    mov     eax, [rel vga_col]
    inc     eax
    cmp     eax, VGA_COLS
    jb      .store_col
    xor     eax, eax
    mov     [rel vga_col], eax
    jmp     .advance_row
.store_col:
    mov     [rel vga_col], eax
    ret

.cr:
    mov     dword [rel vga_col], 0
    ret

.bs:
    mov     eax, [rel vga_col]
    test    eax, eax
    jz      .ret
    dec     eax
    mov     [rel vga_col], eax
    mov     ecx, [rel vga_row]
    imul    ecx, ecx, VGA_COLS
    add     ecx, eax
    mov     ah, [rel vga_attr]
    mov     al, ' '
    mov     rdx, VGA_MEM
    mov     [rdx + rcx*2], ax
    ret

.tab:
    mov     al, ' '
    call    vga_emit_ascii
    mov     eax, [rel vga_col]
    test    eax, 3
    jnz     .tab
    ret

.newline:
    mov     dword [rel vga_col], 0
.advance_row:
    mov     eax, [rel vga_row]
    inc     eax
    cmp     eax, VGA_ROWS
    jb      .store_row
    call    vga_scroll
    ret
.store_row:
    mov     [rel vga_row], eax
.ret:
    ret

; void vga_putc_utf8(uint8 byte in dil)
; Streams UTF-8 bytes cleanly onto VGA text mode:
;   - ASCII passes straight through
;   - Cyrillic (U+0410..U+044F, U+0401, U+0451) transliterates cleanly to ASCII
;   - Common Unicode dashes/quotes map to '-', '\'', '"'
;   - Other non-ASCII renders as '?' (COM1 serial ALWAYS gets raw UTF-8!)
vga_putc_utf8:
    movzx   eax, dil
    cmp     eax, 0x80
    jb      .ascii_fast

    ; Continuation byte 10xxxxxx?
    mov     ecx, eax
    and     ecx, 0xC0
    cmp     ecx, 0x80
    je      .cont

    ; Leading byte of multi-byte UTF-8 sequence
    cmp     eax, 0xE0
    jb      .lead2
    cmp     eax, 0xF0
    jb      .lead3
    ; 4-byte sequence
    and     eax, 0x07
    mov     [rel utf8_cp], eax
    mov     dword [rel utf8_rem], 3
    ret
.lead3:
    and     eax, 0x0F
    mov     [rel utf8_cp], eax
    mov     dword [rel utf8_rem], 2
    ret
.lead2:
    and     eax, 0x1F
    mov     [rel utf8_cp], eax
    mov     dword [rel utf8_rem], 1
    ret

.cont:
    mov     ecx, [rel utf8_rem]
    test    ecx, ecx
    jz      .ret
    mov     edx, [rel utf8_cp]
    shl     edx, 6
    and     eax, 0x3F
    or      edx, eax
    mov     [rel utf8_cp], edx
    dec     ecx
    mov     [rel utf8_rem], ecx
    jnz     .ret

    ; Complete Unicode codepoint in edx!
    ; Map Cyrillic U+0410..U+044F -> single ASCII translit char
    cmp     edx, 0x0401                 ; 'Ё'
    je      .emit_E
    cmp     edx, 0x0451                 ; 'ё'
    je      .emit_e
    cmp     edx, 0x0410
    jb      .chk_punct
    cmp     edx, 0x044F
    ja      .chk_punct
    sub     edx, 0x0410
    lea     rax, [rel cyr_ascii_tbl]
    movzx   eax, byte [rax + rdx]
    jmp     vga_emit_ascii

.chk_punct:
    cmp     edx, 0x2013                 ; en-dash
    je      .emit_dash
    cmp     edx, 0x2014                 ; em-dash
    je      .emit_dash
    cmp     edx, 0x2018                 ; left single quote
    je      .emit_sq
    cmp     edx, 0x2019                 ; right single quote
    je      .emit_sq
    cmp     edx, 0x201C                 ; left double quote
    je      .emit_dq
    cmp     edx, 0x201D                 ; right double quote
    je      .emit_dq
    cmp     edx, 0x2022                 ; bullet
    je      .emit_star
    mov     al, '?'
    jmp     vga_emit_ascii

.emit_E:    mov al, 'E'
            jmp vga_emit_ascii
.emit_e:    mov al, 'e'
            jmp vga_emit_ascii
.emit_dash: mov al, '-'
            jmp vga_emit_ascii
.emit_sq:   mov al, "'"
            jmp vga_emit_ascii
.emit_dq:   mov al, '"'
            jmp vga_emit_ascii
.emit_star: mov al, '*'
            jmp vga_emit_ascii

.ascii_fast:
    mov     dword [rel utf8_rem], 0
    jmp     vga_emit_ascii
.ret:
    ret

; ============================================================================
; Unified Console Output (COM1 Raw UTF-8 + VGA Text Mode)
; ============================================================================

; void console_putc(uint8 ch in dil)
console_putc:
    push    rbx
    mov     bl, dil
    mov     al, bl
    call    serial_putc
    movzx   edi, bl
    call    vga_putc_utf8
    call    vga_update_cursor
    pop     rbx
    ret

; void console_write(const void *buf in rdi, uint64 len in rsi)
; Batches VGA CRTC hardware cursor update to once per write (4x fewer KVM exits).
console_write:
    push    rbx
    push    r12
    push    r13
    mov     r12, rdi
    mov     r13, rsi
    xor     ebx, ebx
.lp:
    cmp     rbx, r13
    jae     .done
    movzx   eax, byte [r12 + rbx]
    push    rax
    call    serial_putc
    pop     rdi
    call    vga_putc_utf8
    inc     rbx
    jmp     .lp
.done:
    call    vga_update_cursor
    pop     r13
    pop     r12
    pop     rbx
    ret

; void console_write_cstr(const char *s in rdi)
console_write_cstr:
    push    rbx
    mov     rbx, rdi
.lp:
    movzx   eax, byte [rbx]
    test    al, al
    jz      .done
    push    rax
    call    serial_putc
    pop     rdi
    call    vga_putc_utf8
    inc     rbx
    jmp     .lp
.done:
    call    vga_update_cursor
    pop     rbx
    ret

; ============================================================================
; 3. PS/2 Keyboard Driver (0x60 / 0x64 polling, Scancode Set 1 -> ASCII)
; ============================================================================

; int32 ps2_try_getc(void) -> eax = ASCII (0..127) or -1 if no key
ps2_try_getc:
    in      al, PS2_STATUS
    test    al, 0x01                    ; Output buffer full?
    jz      .none
    test    al, 0x20                    ; Mouse data? Skip
    jnz     .drain_none

    xor     eax, eax
    in      al, PS2_DATA

    ; Ignore 0xE0 / 0xE1 extended prefix
    cmp     al, 0xE0
    je      .none
    cmp     al, 0xE1
    je      .none

    ; Track Left/Right Shift make & break
    cmp     al, 0x2A                    ; LShift make
    je      .shift_on
    cmp     al, 0x36                    ; RShift make
    je      .shift_on
    cmp     al, 0xAA                    ; LShift break
    je      .shift_off
    cmp     al, 0xB6                    ; RShift break
    je      .shift_off

    ; Ignore other key releases (bit 7 set)
    test    al, 0x80
    jnz     .none

    ; Translate scancode -> ASCII
    cmp     byte [rel ps2_shift], 0
    jne     .use_shift
    lea     rdx, [rel ps2_tbl_norm]
    jmp     .lookup
.use_shift:
    lea     rdx, [rel ps2_tbl_shift]
.lookup:
    movzx   eax, byte [rdx + rax]
    test    eax, eax
    jz      .none
    ret

.shift_on:
    mov     byte [rel ps2_shift], 1
    jmp     .none
.shift_off:
    mov     byte [rel ps2_shift], 0
    jmp     .none
.drain_none:
    in      al, PS2_DATA
.none:
    mov     eax, -1
    ret

; ============================================================================
; Interactive Line Reader (polls COM1 Serial AND PS/2 Keyboard simultaneously)
; int64 console_readline(char *buf in rdi, uint64 max_len in rsi)
; Returns length >= 0 (NUL-terminated), or -1 on Ctrl-D (EOF).
; ============================================================================
console_readline:
    push    rbx
    push    r12
    push    r13
    mov     r12, rdi                    ; buf
    lea     r13, [rsi - 1]              ; max chars (leave room for NUL)
    xor     ebx, ebx                    ; len = 0

.poll:
    call    serial_try_getc
    cmp     eax, -1
    jne     .got_char
    call    ps2_try_getc
    cmp     eax, -1
    jne     .got_char
    pause
    jmp     .poll

.got_char:
    ; Ignore trailing '\n' right after '\r'
    cmp     eax, 10
    jne     .not_lf_after_cr
    cmp     byte [rel last_was_cr], 0
    je      .enter
    mov     byte [rel last_was_cr], 0
    jmp     .poll

.not_lf_after_cr:
    mov     byte [rel last_was_cr], 0

    cmp     eax, 13                     ; '\r'
    je      .got_cr
    cmp     eax, 4                      ; Ctrl-D (EOF)
    je      .got_eof
    cmp     eax, 8                      ; Backspace
    je      .got_bs
    cmp     eax, 127                    ; DEL (Backspace over serial)
    je      .got_bs

    ; Accept printable ASCII & UTF-8 bytes >= 0x20
    cmp     eax, 0x20
    jb      .poll
    cmp     rbx, r13
    jae     .poll

    mov     [r12 + rbx], al
    inc     rbx
    mov     edi, eax
    call    console_putc
    jmp     .poll

.got_bs:
    test    rbx, rbx
    jz      .poll
    dec     rbx
    ; Echo "\b \b" to COM1 and '\b' to VGA
    mov     al, 8
    call    serial_putc
    mov     al, ' '
    call    serial_putc
    mov     al, 8
    call    serial_putc
    mov     dil, 8
    call    vga_putc_utf8
    call    vga_update_cursor
    jmp     .poll

.got_cr:
    mov     byte [rel last_was_cr], 1
.enter:
    mov     byte [r12 + rbx], 0
    mov     edi, 10
    call    console_putc
    mov     rax, rbx
    pop     r13
    pop     r12
    pop     rbx
    ret

.got_eof:
    mov     byte [r12 + rbx], 0
    mov     rax, -1
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; 4. PIT Channel 2 Delay & RDTSC Calibrated Nanosecond Timer
; ============================================================================

; Internal: wait for exact N PIT ticks (1..65535 in edi) using PIT Channel 2
pit_wait_ticks:
    mov     ecx, edi
    ; Gate CH2 ON (bit 0 = 1), Speaker OFF (bit 1 = 0)
    in      al, PIT_PORT_B
    and     al, 0xFC                    ; clear bits 0,1
    out     PIT_PORT_B, al
    or      al, 0x01                    ; enable CH2 gate
    out     PIT_PORT_B, al

    ; Program CH2: mode 0 (interrupt on terminal count), lobyte/hibyte
    mov     al, 0xB0
    out     PIT_CMD, al
    mov     al, cl
    out     PIT_CH2_DATA, al
    mov     al, ch
    out     PIT_CH2_DATA, al

    ; Poll OUT2 (bit 5 of port 0x61) until high
.wait:
    in      al, PIT_PORT_B
    test    al, 0x20
    jz      .wait
    ret

; void pit_delay_us(uint64 us in rdi)
; Accurate microsecond delay using PIT Channel 2 (supports arbitrary durations).
pit_delay_us:
    push    rbx
    ; Total PIT ticks = us * 1193182 / 1000000
    mov     rax, rdi
    mov     rcx, PIT_FREQ_HZ
    mul     rcx
    mov     rcx, 1000000
    div     rcx
    test    rax, rax
    jnz     .have_ticks
    mov     rax, 1
.have_ticks:
    mov     rbx, rax
.chunk_lp:
    test    rbx, rbx
    jz      .done
    mov     rdi, rbx
    cmp     rdi, 50000
    jbe     .run_chunk
    mov     rdi, 50000
.run_chunk:
    sub     rbx, rdi
    call    pit_wait_ticks
    jmp     .chunk_lp
.done:
    pop     rbx
    ret

; void timer_init(void)
; Calibrates RDTSC frequency against PIT Channel 2 using a 20 ms window
; (23,864 PIT ticks = 20.0003 ms) for accurate bare-metal tok/s timing.
timer_init:
    push    rbx
    push    r12

    ; Warm up PIT channel 2 briefly (1 ms)
    mov     edi, 1193
    call    pit_wait_ticks

    ; Measure RDTSC across 23,864 PIT ticks (~20.000 ms)
    lfence
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    mov     r12, rax                    ; t0

    mov     edi, 23864
    call    pit_wait_ticks

    lfence
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    sub     rax, r12                    ; delta_tsc

    ; tsc_hz = (delta_tsc * 1,193,182) / 23864
    mov     rcx, PIT_FREQ_HZ
    mul     rcx                         ; rdx:rax = delta_tsc * PIT_FREQ_HZ
    mov     rcx, 23864
    div     rcx                         ; rax = tsc_hz

    ; Sanity floor (>= 100 MHz) to prevent division by zero in weird emulators
    cmp     rax, 100000000
    jae     .hz_ok
    mov     rax, 2000000000             ; fallback 2.0 GHz
.hz_ok:
    mov     [rel g_tsc_hz], rax

    ; tsc_mhz = (tsc_hz + 500000) / 1000000
    add     rax, 500000
    xor     edx, edx
    mov     rcx, 1000000
    div     rcx
    mov     [rel g_tsc_mhz], rax

    pop     r12
    pop     rbx
    ret

; uint64 timer_ns(void) -> rax = nanoseconds since boot (_start)
timer_ns:
    lfence
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    sub     rax, [rel g_boot_tsc]
    jnc     .pos
    xor     eax, eax
.pos:
    mov     rcx, 1000000000
    mul     rcx                         ; rdx:rax = delta_tsc * 1e9
    div     qword [rel g_tsc_hz]        ; rax = ns
    ret

; ============================================================================
; 5. Clean QEMU / Bochs / VirtualBox Poweroff
; ============================================================================
qemu_shutdown:
    ; QEMU PIIX4 / i440fx / q35 ACPI S5 shutdown
    mov     dx, 0x604
    mov     ax, 0x2000
    out     dx, ax

    ; Older QEMU / Bochs ACPI shutdown
    mov     dx, 0xB004
    mov     ax, 0x2000
    out     dx, ax

    ; VirtualBox ACPI shutdown
    mov     dx, 0x4004
    mov     ax, 0x3400
    out     dx, ax

    ; QEMU isa-debug-exit device (if enabled)
    mov     dx, 0x501
    xor     al, al
    out     dx, al

.halt:
    cli
    hlt
    jmp     .halt

; ============================================================================
; Read-only Lookup Tables
; ============================================================================
section .rodata

; Cyrillic U+0410..U+044F (А..Я, а..я) -> single-char ASCII translit for VGA
cyr_ascii_tbl:
    ; А Б В Г Д Е Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я
    db 'A','B','V','G','D','E','Z','Z','I','Y','K','L','M','N','O','P'
    db 'R','S','T','U','F','H','C','C','S','S',"'",'Y',"'",'E','U','A'
    ; а б в г д е ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я
    db 'a','b','v','g','d','e','z','z','i','y','k','l','m','n','o','p'
    db 'r','s','t','u','f','h','c','c','s','s',"'",'y',"'",'e','u','a'

; PS/2 Scancode Set 1 -> ASCII (unshifted, 128 bytes)
ps2_tbl_norm:
    db 0,   27,  '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 8,   9
    db 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 10,  0,   'a', 's'
    db 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '`', 0,   '\', 'z', 'x', 'c', 'v'
    db 'b', 'n', 'm', ',', '.', '/', 0,   '*', 0,   ' ', 0,   0,   0,   0,   0,   0
    times 64 db 0

; PS/2 Scancode Set 1 -> ASCII (shifted, 128 bytes)
ps2_tbl_shift:
    db 0,   27,  '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', 8,   9
    db 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', 10,  0,   'A', 'S'
    db 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"', '~', 0,   '|', 'Z', 'X', 'C', 'V'
    db 'B', 'N', 'M', '<', '>', '?', 0,   '*', 0,   ' ', 0,   0,   0,   0,   0,   0
    times 64 db 0

; ============================================================================
; Driver State (.bss)
; ============================================================================
section .bss
alignb 8
g_tsc_hz:       resq 1
g_tsc_mhz:      resq 1
vga_row:        resd 1
vga_col:        resd 1
utf8_cp:        resd 1
utf8_rem:       resd 1
vga_attr:       resb 1
ps2_shift:      resb 1
last_was_cr:    resb 1
