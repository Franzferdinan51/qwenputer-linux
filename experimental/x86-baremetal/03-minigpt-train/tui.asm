; ============================================================================
; tui.asm — Pure x86-64 Flicker-Free Live ANSI/UTF-8 TUI Dashboard (Zero libc)
;
; Features:
;   - Raw Linux ioctl(1, TCGETS=0x5401) TTY auto-detection
;   - Exact 80x24 buffered frame renderer (single write() syscall per frame + \x1b[H)
;   - High-resolution 70x9 UTF-8 Braille (140x36 sub-pixel) live loss curve
;     with red->yellow->green gradient, auto-scaled Y-axis & cyan val ◆ markers
;   - Progress bar, live metrics (loss, val, best, gnorm, lr, tok/s, elapsed)
;   - 4-line live Shakespeare sample preview panel (captured from gpt_sample)
;   - Clean cursor hide (\x1b[?25l) on start & restore (\x1b[?25h) on exit
; ============================================================================

bits 64
default rel

%include "train_constants.inc"

%define SYS_ioctl           16
%define TCGETS              0x5401

%define PLOT_COLS           70
%define PLOT_ROWS           9
%define PLOT_PX_W           140         ; 70 * 2
%define PLOT_PX_H           36          ; 9 * 4
%define MAX_TRAIN_HIST      16384
%define MAX_VAL_HIST        1024
%define MAX_SAMPLE_CAP      1024

extern g_trn
extern sys_write_all

section .text

global g_sample_capture
global tui_isatty_stdout
global tui_init
global tui_record_train
global tui_record_val
global tui_sample_begin
global tui_sample_push_byte
global tui_sample_end
global tui_render
global tui_finish

; ============================================================================
; uint64 tui_isatty_stdout(void) -> rax (1 if stdout is TTY, 0 otherwise)
; Uses raw Linux syscall ioctl(1, TCGETS=0x5401, &termios_buf).
; ============================================================================
tui_isatty_stdout:
    sub     rsp, 72                     ; struct termios is 60B on x86-64 Linux
    mov     eax, SYS_ioctl
    mov     edi, 1                      ; stdout
    mov     esi, TCGETS
    mov     rdx, rsp
    syscall
    xor     ecx, ecx
    test    rax, rax
    sete    cl
    mov     eax, ecx
    add     rsp, 72
    ret

; ============================================================================
; void tui_init(uint64 threads)
; Clears history & sample buffer, hides cursor and clears screen once.
; ============================================================================
tui_init:
    mov     [rel tui_threads], rdi
    mov     qword [rel tui_n_train], 0
    mov     qword [rel tui_cur_step1], 0
    mov     qword [rel tui_val_cnt], 0
    mov     qword [rel tui_has_val], 0
    mov     qword [rel tui_sample_len], 0
    mov     qword [rel tui_sample_step], 0
    mov     qword [rel tui_step_ns], 0
    mov     qword [rel tui_elapsed_ns], 0
    mov     qword [rel g_sample_capture], 0

    mov     eax, 0x4479C000             ; 999.0f
    mov     [rel tui_best_val], eax
    mov     dword [rel tui_last_loss], 0
    mov     dword [rel tui_last_val], 0
    mov     dword [rel tui_last_gnorm], 0
    mov     dword [rel tui_last_lr], 0

    ; Emit "\x1b[?25l\x1b[2J\x1b[H" (hide cursor, clear screen once, cursor home)
    mov     edi, 1
    lea     rsi, [rel ansi_enter]
    mov     edx, ansi_enter_len
    jmp     sys_write_all

; ============================================================================
; void tui_record_train(uint64 step0, float loss, float gnorm, float lr,
;                       uint64 step_ns, uint64 elapsed_ns)
; rdi=step0, xmm0=loss, xmm1=gnorm, xmm2=lr, rsi=step_ns, rdx=elapsed_ns
; ============================================================================
tui_record_train:
    vmovss  [rel tui_last_loss], xmm0
    vmovss  [rel tui_last_gnorm], xmm1
    vmovss  [rel tui_last_lr], xmm2
    mov     [rel tui_step_ns], rsi
    mov     [rel tui_elapsed_ns], rdx
    lea     rax, [rdi + 1]
    mov     [rel tui_cur_step1], rax
    cmp     rdi, MAX_TRAIN_HIST
    jae     .ret
    lea     rcx, [rel tui_train_hist]
    vmovss  [rcx + rdi*4], xmm0
    cmp     rax, [rel tui_n_train]
    jbe     .ret
    mov     [rel tui_n_train], rax
.ret:
    ret

; ============================================================================
; void tui_record_val(uint64 step_num, float val_loss)
; rdi=step_num, xmm0=val_loss
; ============================================================================
tui_record_val:
    vmovss  [rel tui_last_val], xmm0
    mov     qword [rel tui_has_val], 1
    vmovss  xmm1, [rel tui_best_val]
    vucomiss xmm0, xmm1
    jae     .append
    vmovss  [rel tui_best_val], xmm0
.append:
    mov     rax, [rel tui_val_cnt]
    cmp     rax, MAX_VAL_HIST
    jae     .ret
    lea     rcx, [rel tui_val_step]
    mov     [rcx + rax*8], rdi
    lea     rcx, [rel tui_val_hist]
    vmovss  [rcx + rax*4], xmm0
    inc     rax
    mov     [rel tui_val_cnt], rax
.ret:
    ret

; ============================================================================
; Sample capture hooks called around gpt_sample in TUI mode
; ============================================================================
tui_sample_begin:
    mov     [rel tui_sample_step], rdi
    mov     qword [rel tui_sample_len], 0
    mov     qword [rel g_sample_capture], 1
    ret

; void tui_sample_push_byte(uint32 ch)
tui_sample_push_byte:
    mov     rax, [rel tui_sample_len]
    cmp     rax, MAX_SAMPLE_CAP - 1
    jae     .ret
    lea     rcx, [rel tui_sample_buf]
    mov     [rcx + rax], dil
    inc     rax
    mov     [rel tui_sample_len], rax
.ret:
    ret

tui_sample_end:
    mov     qword [rel g_sample_capture], 0
    mov     rax, [rel tui_sample_len]
    lea     rcx, [rel tui_sample_buf]
    mov     byte [rcx + rax], 0
    ret

; ============================================================================
; void tui_finish(void)
; Renders final frame, moves cursor to row 25 col 1 & shows cursor cleanly.
; ============================================================================
tui_finish:
    call    tui_render
    mov     edi, 1
    lea     rsi, [rel ansi_leave]
    mov     edx, ansi_leave_len
    jmp     sys_write_all

; ============================================================================
; BUFFER BUILDER HELPERS
; Convention inside tui_render:
;   r12 = current write ptr in tui_out_buf
;   r13 = current visible column on row (0..80)
;   All helpers preserve rbx, rbp, r14, r15.
; ============================================================================

; tb_ansi(rsi = cstr): copy raw bytes, do NOT change r13
tb_ansi:
.lp:
    mov     al, [rsi]
    test    al, al
    jz      .done
    mov     [r12], al
    inc     r12
    inc     rsi
    jmp     .lp
.done:
    ret

; tb_ascii(rsi = cstr): copy pure ASCII bytes, increment r13 per byte
tb_ascii:
.lp:
    mov     al, [rsi]
    test    al, al
    jz      .done
    mov     [r12], al
    inc     r12
    inc     r13
    inc     rsi
    jmp     .lp
.done:
    ret

; tb_utf8(rsi = cstr, rdx = visible_width)
tb_utf8:
    add     r13, rdx
    jmp     tb_ansi

; tb_repeat_utf8(rsi = single-col utf8 cstr, rcx = count)
tb_repeat_utf8:
    push    rbx
    mov     rbx, rsi
.lp:
    test    rcx, rcx
    jz      .done
    mov     rsi, rbx
    call    tb_ansi
    inc     r13
    dec     rcx
    jmp     .lp
.done:
    pop     rbx
    ret

; tb_u64(rdi = val)
tb_u64:
    sub     rsp, 32
    mov     rax, rdi
    lea     rcx, [rsp + 31]
    mov     r8, rcx
    mov     r9d, 10
    test    rax, rax
    jnz     .lp
    dec     rcx
    mov     byte [rcx], '0'
    jmp     .emit
.lp:
    xor     edx, edx
    div     r9
    add     dl, '0'
    dec     rcx
    mov     [rcx], dl
    test    rax, rax
    jnz     .lp
.emit:
    cmp     rcx, r8
    jae     .done
    mov     al, [rcx]
    mov     [r12], al
    inc     r12
    inc     r13
    inc     rcx
    jmp     .emit
.done:
    add     rsp, 32
    ret

; tb_f64_ndp(xmm0 = val, rdi = digits in 1..6)
tb_f64_ndp:
    push    rbx
    sub     rsp, 32                     ; 8 + 32 + 8 = 48 (16B aligned)
    mov     ebx, edi                    ; digits
    ; Compute scale = 10^digits
    mov     rcx, 1
    mov     eax, ebx
.pow:
    imul    rcx, rcx, 10
    dec     eax
    jnz     .pow

    vcvtsi2sd xmm1, xmm1, rcx
    vmulsd  xmm0, xmm0, xmm1
    vaddsd  xmm0, xmm0, [rel td_half]
    vcvttsd2si rax, xmm0
    test    rax, rax
    jns     .pos
    xor     eax, eax
.pos:
    xor     edx, edx
    div     rcx                         ; rax = int, rdx = frac
    mov     [rsp + 16], rdx             ; save frac
    mov     rdi, rax
    call    tb_u64

    mov     byte [r12], '.'
    inc     r12
    inc     r13

    mov     rax, [rsp + 16]             ; frac
    mov     ecx, ebx                    ; digits
    mov     r8, 10
.dig:
    xor     edx, edx
    div     r8
    add     dl, '0'
    mov     [rsp + rcx - 1], dl
    dec     ecx
    jnz     .dig

    xor     ecx, ecx
.out:
    cmp     ecx, ebx
    jae     .done
    mov     al, [rsp + rcx]
    mov     [r12], al
    inc     r12
    inc     r13
    inc     ecx
    jmp     .out
.done:
    add     rsp, 32
    pop     rbx
    ret

; tb_row_start: reset r13=0, emit cyan "│ " (r13 -> 2)
tb_row_start:
    xor     r13d, r13d
    lea     rsi, [rel ansi_lborder]
    mov     edx, 2
    jmp     tb_utf8

; tb_row_end: pad spaces up to col 79, emit cyan "│\n" (r13 -> 80)
tb_row_end:
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
.pad:
    cmp     r13, 79
    jae     .rborder
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    jmp     .pad
.rborder:
    lea     rsi, [rel ansi_rborder_nl]
    call    tb_ansi
    inc     r13
    ret

; tb_hline(rsi = left_glyph, rdx = right_glyph, ecx = add_nl)
tb_hline:
    push    rbx
    push    r14
    mov     rbx, rdx                    ; right_glyph
    mov     r14d, ecx                   ; add_nl
    xor     r13d, r13d
    push    rsi
    lea     rsi, [rel ansi_cyan]
    call    tb_ansi
    pop     rsi
    call    tb_ansi
    inc     r13
    lea     rsi, [rel u_horiz]
    mov     ecx, 78
    call    tb_repeat_utf8
    mov     rsi, rbx
    call    tb_ansi
    inc     r13
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    test    r14d, r14d
    jz      .done
    mov     byte [r12], 10
    inc     r12
.done:
    pop     r14
    pop     rbx
    ret

; ============================================================================
; BRAILLE PIXEL PLOTTER: set_braille_px(edi = px in 0..139, esi = py in 0..35)
; Clamps safely to [0..139] x [0..35]. Preserves all callee-saved regs + r8..r11!
; ============================================================================
set_braille_px:
    cmp     edi, 0
    jge     .px_lo
    xor     edi, edi
.px_lo:
    cmp     edi, PLOT_PX_W - 1
    jle     .px_hi
    mov     edi, PLOT_PX_W - 1
.px_hi:
    cmp     esi, 0
    jge     .py_lo
    xor     esi, esi
.py_lo:
    cmp     esi, PLOT_PX_H - 1
    jle     .py_hi
    mov     esi, PLOT_PX_H - 1
.py_hi:
    mov     eax, edi
    shr     eax, 1                      ; cx = px / 2
    mov     ecx, edi
    and     ecx, 1                      ; dx = px & 1
    shl     ecx, 2                      ; dx * 4

    mov     edx, esi
    shr     edx, 2                      ; cy = py / 4
    and     esi, 3                      ; dy = py & 3
    add     ecx, esi                    ; dx*4 + dy

    imul    edx, edx, PLOT_COLS
    add     edx, eax                    ; cy * 70 + cx
    lea     rax, [rel braille_bits]
    movzx   ecx, byte [rax + rcx]
    lea     rax, [rel tui_braille]
    or      [rax + rdx], cl
    ret

; ============================================================================
; void tui_render(void)
; Builds complete 80x24 ANSI/UTF-8 frame into tui_out_buf and writes in 1 syscall!
; ============================================================================
tui_render:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 104                    ; 6*8 + 104 + 8 = 160 (16B aligned)
    ; Locals:
    ;   [rsp + 0]  = y_min (f64)
    ;   [rsp + 8]  = y_max (f64)
    ;   [rsp + 16] = y_span (f64)
    ;   [rsp + 24] = cy (row 0..8)
    ;   [rsp + 32] = cx (col 0..69)
    ;   [rsp + 40] = has_prev (u64)
    ;   [rsp + 48] = prev_px (i64)
    ;   [rsp + 56] = prev_py (i64)
    ;   [rsp + 64] = cur_px (i64)
    ;   [rsp + 72] = cur_py (i64)
    ;   [rsp + 80] = sample scan ptr

    lea     rbx, [rel g_trn]

    ; ------------------------------------------------------------------------
    ; PHASE 1: Rasterize train_loss curve & val_loss markers onto 70x9 canvas
    ; ------------------------------------------------------------------------
    ; Zero tui_braille (630B), tui_val_mark (630B), bin_sum (140*8B), bin_cnt (140*4B)
    lea     rdi, [rel tui_braille]
    xor     eax, eax
    mov     ecx, PLOT_COLS * PLOT_ROWS
.zb:
    mov     byte [rdi + rcx - 1], 0
    dec     ecx
    jnz     .zb

    lea     rdi, [rel tui_val_mark]
    mov     ecx, PLOT_COLS * PLOT_ROWS
.zv:
    mov     byte [rdi + rcx - 1], 0
    dec     ecx
    jnz     .zv

    lea     rdi, [rel tui_bin_sum]
    mov     ecx, PLOT_PX_W
.zbs:
    mov     qword [rdi + rcx*8 - 8], 0
    dec     ecx
    jnz     .zbs

    lea     rdi, [rel tui_bin_cnt]
    mov     ecx, PLOT_PX_W
.zbc:
    mov     dword [rdi + rcx*4 - 4], 0
    dec     ecx
    jnz     .zbc

    ; Find [lo, hi] across all train_hist and val_hist
    vmovsd  xmm6, [rel td_big]          ; lo = +1e9
    vmovsd  xmm7, [rel td_neg_big]      ; hi = -1e9
    xor     r8d, r8d                    ; any_pt = 0

    mov     rcx, [rel tui_n_train]
    lea     rdi, [rel tui_train_hist]
    xor     eax, eax
.mm_tr:
    cmp     rax, rcx
    jae     .mm_va_init
    vcvtss2sd xmm0, xmm0, [rdi + rax*4]
    vminsd  xmm6, xmm6, xmm0
    vmaxsd  xmm7, xmm7, xmm0
    mov     r8d, 1
    inc     rax
    jmp     .mm_tr

.mm_va_init:
    mov     rcx, [rel tui_val_cnt]
    lea     rdi, [rel tui_val_hist]
    xor     eax, eax
.mm_va:
    cmp     rax, rcx
    jae     .mm_done
    vcvtss2sd xmm0, xmm0, [rdi + rax*4]
    vminsd  xmm6, xmm6, xmm0
    vmaxsd  xmm7, xmm7, xmm0
    mov     r8d, 1
    inc     rax
    jmp     .mm_va

.mm_done:
    test    r8d, r8d
    jnz     .mm_margin
    vxorpd  xmm6, xmm6, xmm6            ; default 0.0 .. 5.0
    vmovsd  xmm7, [rel td_five]
    jmp     .mm_store
.mm_margin:
    vsubsd  xmm6, xmm6, [rel td_margin] ; lo - 0.15
    vxorpd  xmm0, xmm0, xmm0
    vmaxsd  xmm6, xmm6, xmm0            ; clamp >= 0.0
    vaddsd  xmm7, xmm7, [rel td_margin] ; hi + 0.15
    vsubsd  xmm0, xmm7, xmm6
    vucomisd xmm0, [rel td_min_span]
    jae     .mm_store
    vaddsd  xmm7, xmm6, [rel td_min_span]
.mm_store:
    vmovsd  [rsp + 0], xmm6             ; y_min
    vmovsd  [rsp + 8], xmm7             ; y_max
    vsubsd  xmm0, xmm7, xmm6
    vmovsd  [rsp + 16], xmm0            ; y_span

    ; Bin train_hist[0..n_train-1] into 140 horizontal sub-pixels
    mov     r14, [rbx + TRN_STEPS]
    cmp     r14, 1
    jbe     .denom_1
    dec     r14                         ; denom = max(1, total_steps - 1)
    jmp     .bin_start
.denom_1:
    mov     r14, 1
.bin_start:
    mov     r15, [rel tui_n_train]
    lea     r8,  [rel tui_train_hist]
    lea     r9,  [rel tui_bin_sum]
    lea     r10, [rel tui_bin_cnt]
    xor     ecx, ecx                    ; k = 0
.bin_lp:
    cmp     rcx, r15
    jae     .plot_bins
    mov     rax, rcx
    imul    rax, rax, PLOT_PX_W - 1     ; k * 139
    xor     edx, edx
    div     r14                         ; px
    cmp     rax, PLOT_PX_W - 1
    jbe     .px_ok
    mov     rax, PLOT_PX_W - 1
.px_ok:
    vcvtss2sd xmm0, xmm0, [r8 + rcx*4]
    vaddsd  xmm0, xmm0, [r9 + rax*8]
    vmovsd  [r9 + rax*8], xmm0
    inc     dword [r10 + rax*4]
    inc     rcx
    jmp     .bin_lp

.plot_bins:
    mov     qword [rsp + 40], 0         ; has_prev = 0
    xor     r14d, r14d                  ; px = 0
.pb_lp:
    cmp     r14, PLOT_PX_W
    jae     .plot_vals
    lea     r10, [rel tui_bin_cnt]
    mov     eax, [r10 + r14*4]
    test    eax, eax
    jz      .pb_next

    ; avg = bin_sum[px] / bin_cnt[px]
    lea     r9, [rel tui_bin_sum]
    vmovsd  xmm0, [r9 + r14*8]
    vcvtsi2sd xmm1, xmm1, eax
    vdivsd  xmm0, xmm0, xmm1
    ; py = round((y_max - avg) / y_span * 35.0)
    vmovsd  xmm1, [rsp + 8]
    vsubsd  xmm1, xmm1, xmm0
    vdivsd  xmm1, xmm1, [rsp + 16]
    vmulsd  xmm1, xmm1, [rel td_35]
    vaddsd  xmm1, xmm1, [rel td_half]
    vcvttsd2si r15, xmm1                ; cur_py
    cmp     r15, 0
    jge     .py_c1
    xor     r15d, r15d
.py_c1:
    cmp     r15, PLOT_PX_H - 1
    jle     .py_c2
    mov     r15, PLOT_PX_H - 1
.py_c2:
    mov     [rsp + 64], r14             ; cur_px
    mov     [rsp + 72], r15             ; cur_py

    ; Plot current pixel
    mov     edi, r14d
    mov     esi, r15d
    call    set_braille_px

    ; Connect from (prev_px, prev_py) to (cur_px, cur_py) with linear interp
    cmp     qword [rsp + 40], 0
    je      .pb_save_prev

    mov     r8,  [rsp + 64]
    sub     r8,  [rsp + 48]             ; dx >= 1
    mov     r9,  [rsp + 72]
    sub     r9,  [rsp + 56]             ; dy (signed)
    mov     r10, r9
    test    r10, r10
    jns     .abs_ok
    neg     r10                         ; |dy|
.abs_ok:
    mov     r11, r8                     ; steps = max(dx, |dy|)
    cmp     r10, r11
    jle     .st_ok
    mov     r11, r10
.st_ok:
    cmp     r11, 1
    jle     .pb_save_prev

    mov     rcx, 1                      ; s = 1 .. steps - 1
.line_lp:
    cmp     rcx, r11
    jae     .pb_save_prev
    ; ix = prev_px + (dx * s) / steps
    mov     rax, r8
    imul    rax, rcx
    cqo
    idiv    r11
    add     rax, [rsp + 48]
    mov     edi, eax
    ; iy = prev_py + (dy * s) / steps
    mov     rax, r9
    imul    rax, rcx
    cqo
    idiv    r11
    add     rax, [rsp + 56]
    mov     esi, eax
    push    rcx
    push    r11
    call    set_braille_px
    pop     r11
    pop     rcx
    inc     rcx
    jmp     .line_lp

.pb_save_prev:
    mov     rax, [rsp + 64]
    mov     [rsp + 48], rax
    mov     rax, [rsp + 72]
    mov     [rsp + 56], rax
    mov     qword [rsp + 40], 1

.pb_next:
    inc     r14
    jmp     .pb_lp

    ; Plot val_loss diamond markers onto 70x9 cell grid
.plot_vals:
    mov     r14, [rbx + TRN_STEPS]
    cmp     r14, 1
    jae     .vd_ok
    mov     r14, 1
.vd_ok:
    mov     r15, [rel tui_val_cnt]
    lea     r8,  [rel tui_val_step]
    lea     r9,  [rel tui_val_hist]
    lea     r10, [rel tui_val_mark]
    xor     ecx, ecx
.pv_lp:
    cmp     rcx, r15
    jae     .render_start
    mov     rax, [r8 + rcx*8]           ; v_step
    imul    rax, rax, PLOT_COLS - 1     ; * 69
    xor     edx, edx
    div     r14                         ; cx
    cmp     rax, PLOT_COLS - 1
    jbe     .vcx_ok
    mov     rax, PLOT_COLS - 1
.vcx_ok:
    mov     r11, rax                    ; cx

    vcvtss2sd xmm0, xmm0, [r9 + rcx*4]  ; v_loss
    vmovsd  xmm1, [rsp + 8]
    vsubsd  xmm1, xmm1, xmm0
    vdivsd  xmm1, xmm1, [rsp + 16]
    vmulsd  xmm1, xmm1, [rel td_8]
    vaddsd  xmm1, xmm1, [rel td_half]
    vcvttsd2si rax, xmm1                ; cy in 0..8
    cmp     rax, 0
    jge     .vcy_1
    xor     eax, eax
.vcy_1:
    cmp     rax, PLOT_ROWS - 1
    jle     .vcy_2
    mov     rax, PLOT_ROWS - 1
.vcy_2:
    imul    rax, rax, PLOT_COLS
    add     rax, r11
    mov     byte [r10 + rax], 1
    inc     rcx
    jmp     .pv_lp

    ; ------------------------------------------------------------------------
    ; PHASE 2: Render exact 80x24 ANSI/UTF-8 Frame into tui_out_buf
    ; ------------------------------------------------------------------------
.render_start:
    lea     r12, [rel tui_out_buf]

    ; Cursor home (zero flicker!)
    lea     rsi, [rel ansi_home]
    call    tb_ansi

    ; === ROW 1: Top border ===
    lea     rsi, [rel u_tl]
    lea     rdx, [rel u_tr]
    mov     ecx, 1
    call    tb_hline

    ; === ROW 2: Title ===
    call    tb_row_start
    lea     rsi, [rel s_title_1]
    mov     edx, 13                     ; "asm-gpt-train"
    call    tb_utf8
    lea     rsi, [rel s_title_2]
    mov     edx, 28                     ; " │ 100% Pure x86-64 Assembly"
    call    tb_utf8
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
    lea     rsi, [rel s_title_3_txt]
    call    tb_ascii
    call    tb_row_end

    ; === ROW 3: Architecture Config ===
    call    tb_row_start
    lea     rsi, [rel s_cfg_l]
    call    tb_ascii
    mov     rdi, [rbx + TRN_LAYERS]
    call    tb_u64
    lea     rsi, [rel s_cfg_d]
    call    tb_ascii
    mov     rdi, [rbx + TRN_DIM]
    call    tb_u64
    lea     rsi, [rel s_cfg_h]
    call    tb_ascii
    mov     rdi, [rbx + TRN_HEADS]
    call    tb_u64
    lea     rsi, [rel s_cfg_ff]
    call    tb_ascii
    mov     rdi, [rbx + TRN_HIDDEN]
    call    tb_u64
    lea     rsi, [rel s_cfg_v]
    call    tb_ascii
    mov     rdi, [rbx + TRN_VOCAB]
    call    tb_u64
    lea     rsi, [rel s_sep]
    mov     edx, 3
    call    tb_utf8
    lea     rsi, [rel s_cfg_p]
    call    tb_ascii
    mov     rdi, [rbx + TRN_N_PARAMS]
    call    tb_u64
    lea     rsi, [rel s_sep]
    mov     edx, 3
    call    tb_utf8
    lea     rsi, [rel s_cfg_b]
    call    tb_ascii
    mov     rdi, [rbx + TRN_BATCH]
    call    tb_u64
    lea     rsi, [rel s_cfg_t]
    call    tb_ascii
    mov     rdi, [rbx + TRN_SEQ]
    call    tb_u64
    lea     rsi, [rel s_sep]
    mov     edx, 3
    call    tb_utf8
    lea     rsi, [rel s_cfg_thr]
    call    tb_ascii
    mov     rdi, [rel tui_threads]
    call    tb_u64
    call    tb_row_end

    ; === ROW 4: Divider ===
    lea     rsi, [rel u_ml]
    lea     rdx, [rel u_mr]
    mov     ecx, 1
    call    tb_hline

    ; === ROW 5: Chart Header & Legend ===
    call    tb_row_start
    lea     rsi, [rel ansi_bold_w]
    call    tb_ansi
    lea     rsi, [rel s_chart_hdr]
    call    tb_ascii
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    mov     rdi, [rbx + TRN_STEPS]
    call    tb_u64
    lea     rsi, [rel s_chart_hdr2]
    call    tb_ascii

    ; Legend: red/yellow/green braille + " train   " + cyan diamond + " val"
    lea     rsi, [rel ansi_red]
    call    tb_ansi
    lea     rsi, [rel u_braille_full]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_yel]
    call    tb_ansi
    lea     rsi, [rel u_braille_full]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_grn]
    call    tb_ansi
    lea     rsi, [rel u_braille_full]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    lea     rsi, [rel s_leg_tr]
    call    tb_ascii
    lea     rsi, [rel ansi_bcyan]
    call    tb_ansi
    lea     rsi, [rel u_diamond]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    lea     rsi, [rel s_leg_va]
    call    tb_ascii

    cmp     qword [rel tui_has_val], 0
    je      .r5_end
    lea     rsi, [rel s_leg_best]
    call    tb_ascii
    vcvtss2sd xmm0, xmm0, [rel tui_best_val]
    mov     edi, 4
    call    tb_f64_ndp
.r5_end:
    call    tb_row_end

    ; === ROWS 6..14: 9 Braille Chart Rows (cy = 0..8) ===
    mov     qword [rsp + 24], 0
.cy_lp:
    mov     r14, [rsp + 24]
    cmp     r14, PLOT_ROWS
    jae     .row15

    call    tb_row_start                ; r13 = 2

    ; Even rows (0,2,4,6,8): print Y-axis tick label "X.XX ┤" (r13 -> 8)
    ; Odd rows (1,3,5,7): print "     │" (r13 -> 8)
    test    r14, 1
    jnz     .y_odd
    ; y_val = y_max - cy * y_span / 8.0
    vcvtsi2sd xmm0, xmm0, r14
    vmulsd  xmm0, xmm0, [rsp + 16]
    vdivsd  xmm0, xmm0, [rel td_8]
    vmovsd  xmm1, [rsp + 8]
    vsubsd  xmm0, xmm1, xmm0
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
    mov     edi, 2
    call    tb_f64_ndp
.y_pad_e:
    cmp     r13, 7
    jae     .y_tick_e
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    jmp     .y_pad_e
.y_tick_e:
    lea     rsi, [rel u_rtick]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    jmp     .plot_row_cells

.y_odd:
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
.y_pad_o:
    cmp     r13, 7
    jae     .y_tick_o
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    jmp     .y_pad_o
.y_tick_o:
    lea     rsi, [rel u_vbar]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi

.plot_row_cells:
    ; Emit 70 plot cells for row r14 (r13: 8 -> 78)
    mov     qword [rsp + 32], 0
.cx_lp:
    mov     r15, [rsp + 32]
    cmp     r15, PLOT_COLS
    jae     .cy_end

    mov     rax, r14
    imul    rax, rax, PLOT_COLS
    add     rax, r15                    ; idx = cy*70 + cx

    lea     rcx, [rel tui_val_mark]
    cmp     byte [rcx + rax], 0
    je      .chk_braille

    ; Val marker: bold bright cyan ◆
    lea     rsi, [rel ansi_bcyan]
    call    tb_ansi
    lea     rsi, [rel u_diamond]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    jmp     .cx_next

.chk_braille:
    lea     rcx, [rel tui_braille]
    movzx   ebp, byte [rcx + rax]
    test    ebp, ebp
    jz      .empty_cell

    ; Choose color by row height: 0..2 red, 3..5 yellow, 6..8 green
    cmp     r14, 2
    jbe     .col_red
    cmp     r14, 5
    jbe     .col_yel
    lea     rsi, [rel ansi_grn]
    jmp     .emit_br
.col_red:
    lea     rsi, [rel ansi_red]
    jmp     .emit_br
.col_yel:
    lea     rsi, [rel ansi_yel]
.emit_br:
    call    tb_ansi
    ; Encode U+2800 + mask into 3 UTF-8 bytes: E2, A0|(m>>6), 80|(m&3F)
    mov     byte [r12], 0xE2
    mov     eax, ebp
    shr     eax, 6
    or      eax, 0xA0
    mov     [r12 + 1], al
    mov     eax, ebp
    and     eax, 0x3F
    or      eax, 0x80
    mov     [r12 + 2], al
    add     r12, 3
    inc     r13
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    jmp     .cx_next

.empty_cell:
    ; Subtle horizontal grid dots on even rows every 4 cols
    test    r14, 1
    jnz     .emit_sp
    test    r15, 3
    jnz     .emit_sp
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
    lea     rsi, [rel u_dot]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    jmp     .cx_next
.emit_sp:
    mov     byte [r12], ' '
    inc     r12
    inc     r13

.cx_next:
    inc     qword [rsp + 32]
    jmp     .cx_lp

.cy_end:
    call    tb_row_end
    inc     qword [rsp + 24]
    jmp     .cy_lp

    ; === ROW 15: X-axis with step ticks (0, mid, total_steps) ===
.row15:
    call    tb_row_start                ; r13 = 2
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
.r15_p0:
    cmp     r13, 7
    jae     .r15_corner
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    jmp     .r15_p0
.r15_corner:
    lea     rsi, [rel u_bl_axis]
    mov     edx, 1
    call    tb_utf8                     ; r13 = 8
    lea     rsi, [rel s_x0]
    call    tb_ascii                    ; "step 0" -> r13 = 14
.r15_h1:
    cmp     r13, 40
    jae     .r15_mid
    lea     rsi, [rel u_horiz]
    mov     edx, 1
    call    tb_utf8
    jmp     .r15_h1
.r15_mid:
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    mov     rdi, [rbx + TRN_STEPS]
    shr     rdi, 1                      ; steps / 2
    call    tb_u64
    mov     byte [r12], ' '
    inc     r12
    inc     r13
.r15_h2:
    cmp     r13, 71
    jae     .r15_end_lbl
    lea     rsi, [rel u_horiz]
    mov     edx, 1
    call    tb_utf8
    jmp     .r15_h2
.r15_end_lbl:
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    mov     rdi, [rbx + TRN_STEPS]
    call    tb_u64
    call    tb_row_end

    ; === ROW 16: Divider ===
    lea     rsi, [rel u_ml]
    lea     rdx, [rel u_mr]
    mov     ecx, 1
    call    tb_hline

    ; === ROW 17: Step + Progress Bar (28 cols) + % + Elapsed ===
    call    tb_row_start
    lea     rsi, [rel ansi_bold_w]
    call    tb_ansi
    lea     rsi, [rel s_step_lbl]
    call    tb_ascii
    mov     rdi, [rel tui_cur_step1]
    call    tb_u64
    mov     byte [r12], '/'
    inc     r12
    inc     r13
    mov     rdi, [rbx + TRN_STEPS]
    call    tb_u64
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    lea     rsi, [rel s_bar_open]
    call    tb_ascii

    ; Compute filled = (cur_step1 * 28) / max(1, steps), pct = (cur_step1 * 100) / max(1, steps)
    mov     rcx, [rbx + TRN_STEPS]
    test    rcx, rcx
    jnz     .st_nz
    mov     rcx, 1
.st_nz:
    mov     rax, [rel tui_cur_step1]
    imul    rax, rax, 28
    xor     edx, edx
    div     rcx
    cmp     rax, 28
    jbe     .fl_ok
    mov     rax, 28
.fl_ok:
    mov     r14, rax                    ; filled
    mov     r15, 28
    sub     r15, r14                    ; empty

    lea     rsi, [rel ansi_grn]
    call    tb_ansi
    lea     rsi, [rel u_block_full]
    mov     rcx, r14
    call    tb_repeat_utf8
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
    lea     rsi, [rel u_block_empty]
    mov     rcx, r15
    call    tb_repeat_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    lea     rsi, [rel s_bar_close]
    call    tb_ascii

    mov     rcx, [rbx + TRN_STEPS]
    test    rcx, rcx
    jnz     .pct_nz
    mov     rcx, 1
.pct_nz:
    mov     rax, [rel tui_cur_step1]
    imul    rax, rax, 100
    xor     edx, edx
    div     rcx
    mov     rdi, rax
    call    tb_u64
    lea     rsi, [rel s_pct]
    call    tb_ascii
    lea     rsi, [rel s_sep]
    mov     edx, 3
    call    tb_utf8
    lea     rsi, [rel s_elapsed]
    call    tb_ascii
    mov     rax, [rel tui_elapsed_ns]
    vcvtsi2sd xmm0, xmm0, rax
    vdivsd  xmm0, xmm0, [rel td_1e9]
    mov     edi, 1
    call    tb_f64_ndp
    mov     byte [r12], 's'
    inc     r12
    inc     r13
    call    tb_row_end

    ; === ROW 18: Metrics (loss, val, best, gnorm, lr, tok/s) ===
    call    tb_row_start
    lea     rsi, [rel s_m_loss]
    call    tb_ascii
    lea     rsi, [rel ansi_bold_y]
    call    tb_ansi
    vcvtss2sd xmm0, xmm0, [rel tui_last_loss]
    mov     edi, 4
    call    tb_f64_ndp
    lea     rsi, [rel ansi_reset]
    call    tb_ansi

    lea     rsi, [rel s_m_val]
    call    tb_ascii
    cmp     qword [rel tui_has_val], 0
    je      .m_no_val
    lea     rsi, [rel ansi_bcyan]
    call    tb_ansi
    vcvtss2sd xmm0, xmm0, [rel tui_last_val]
    mov     edi, 4
    call    tb_f64_ndp
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    lea     rsi, [rel s_m_best]
    call    tb_ascii
    lea     rsi, [rel ansi_grn]
    call    tb_ansi
    vcvtss2sd xmm0, xmm0, [rel tui_best_val]
    mov     edi, 4
    call    tb_f64_ndp
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    jmp     .m_gnorm
.m_no_val:
    lea     rsi, [rel s_dash]
    call    tb_ascii
    lea     rsi, [rel s_m_best]
    call    tb_ascii
    lea     rsi, [rel s_dash]
    call    tb_ascii

.m_gnorm:
    lea     rsi, [rel s_m_gnorm]
    call    tb_ascii
    vcvtss2sd xmm0, xmm0, [rel tui_last_gnorm]
    mov     edi, 2
    call    tb_f64_ndp

    lea     rsi, [rel s_m_lr]
    call    tb_ascii
    vcvtss2sd xmm0, xmm0, [rel tui_last_lr]
    mov     edi, 6
    call    tb_f64_ndp

    lea     rsi, [rel s_sp2]
    call    tb_ascii
    mov     rax, [rel tui_step_ns]
    test    rax, rax
    jz      .tps_zero
    vcvtsi2sd xmm0, xmm0, [rbx + TRN_BT]
    vmulsd  xmm0, xmm0, [rel td_1e9]
    vcvtsi2sd xmm1, xmm1, rax
    vdivsd  xmm0, xmm0, xmm1
    vcvttsd2si rdi, xmm0
    jmp     .tps_emit
.tps_zero:
    xor     edi, edi
.tps_emit:
    lea     rsi, [rel ansi_bold_g]
    call    tb_ansi
    call    tb_u64
    lea     rsi, [rel s_tps]
    call    tb_ascii
    call    tb_row_end

    ; === ROW 19: Live Sample Section Divider ===
    xor     r13d, r13d
    lea     rsi, [rel ansi_cyan]
    call    tb_ansi
    lea     rsi, [rel u_ml]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel u_horiz]
    mov     edx, 1
    call    tb_utf8
    mov     byte [r12], ' '
    inc     r12
    inc     r13
    lea     rsi, [rel ansi_bold_w]
    call    tb_ansi
    lea     rsi, [rel s_samp_hdr]
    call    tb_ascii
    mov     rdi, [rel tui_sample_step]
    call    tb_u64
    mov     byte [r12], ')'
    inc     r12
    inc     r13
    lea     rsi, [rel ansi_cyan]
    call    tb_ansi
    mov     byte [r12], ' '
    inc     r12
    inc     r13
.r19_h:
    cmp     r13, 79
    jae     .r19_end
    lea     rsi, [rel u_horiz]
    mov     edx, 1
    call    tb_utf8
    jmp     .r19_h
.r19_end:
    lea     rsi, [rel u_mr]
    mov     edx, 1
    call    tb_utf8
    lea     rsi, [rel ansi_reset]
    call    tb_ansi
    mov     byte [r12], 10
    inc     r12

    ; === ROWS 20..23: 4 Lines of Live Generated Text Sample ===
    lea     r14, [rel tui_sample_buf]
    mov     r15, [rel tui_sample_len]
    lea     rbp, [r14 + r15]            ; end ptr
    ; Skip any leading newlines so line 1 isn't empty
.skip_nl:
    cmp     r14, rbp
    jae     .samp_rows
    cmp     byte [r14], 10
    jne     .samp_rows
    inc     r14
    jmp     .skip_nl

.samp_rows:
    mov     [rsp + 80], r14
    mov     qword [rsp + 24], 0         ; line = 0..3
.srow_lp:
    cmp     qword [rsp + 24], 4
    jae     .row24
    call    tb_row_start                ; r13 = 2

    cmp     qword [rel tui_sample_len], 0
    jne     .srow_has_txt
    cmp     qword [rsp + 24], 0
    jne     .srow_end
    lea     rsi, [rel ansi_dim]
    call    tb_ansi
    lea     rsi, [rel s_no_samp]
    call    tb_ascii
    jmp     .srow_end

.srow_has_txt:
    lea     rsi, [rel ansi_white]
    call    tb_ansi
    mov     r14, [rsp + 80]
.schar_lp:
    cmp     r14, rbp
    jae     .srow_save
    cmp     r13, 78                     ; max 76 visible chars per line
    jae     .srow_save
    movzx   eax, byte [r14]
    inc     r14
    cmp     al, 10                      ; '\n' -> end this line!
    je      .srow_save
    cmp     al, 32
    jb      .schar_lp                   ; skip control chars (e.g. '\r', '\t')
    cmp     al, 126
    ja      .schar_lp
    mov     [r12], al
    inc     r12
    inc     r13
    jmp     .schar_lp

.srow_save:
    mov     [rsp + 80], r14
.srow_end:
    call    tb_row_end
    inc     qword [rsp + 24]
    jmp     .srow_lp

    ; === ROW 24: Bottom Border (NO trailing newline to prevent 80x24 scroll!) ===
.row24:
    lea     rsi, [rel u_bl]
    lea     rdx, [rel u_br]
    xor     ecx, ecx                    ; add_nl = 0
    call    tb_hline

    ; Flush entire frame in ONE single write() syscall!
    lea     rsi, [rel tui_out_buf]
    mov     rdx, r12
    sub     rdx, rsi
    mov     edi, 1
    call    sys_write_all

    add     rsp, 104
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; RODATA & BSS
; ============================================================================
section .rodata
align 16
td_half:      dq 0.5
td_five:      dq 5.0
td_8:         dq 8.0
td_35:        dq 35.0
td_margin:    dq 0.15
td_min_span:  dq 0.5
td_big:       dq 1.0e9
td_neg_big:   dq -1.0e9
td_1e9:       dq 1000000000.0

; Braille bit lookup: [dx*4 + dy] for dx in {0,1}, dy in {0,1,2,3}
braille_bits: db 0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80

; ANSI control sequences
ansi_enter:   db 27, "[?25l", 27, "[2J", 27, "[H"
ansi_enter_len equ $ - ansi_enter
ansi_leave:   db 27, "[25;1H", 27, "[?25h", 10
ansi_leave_len equ $ - ansi_leave
ansi_home:    db 27, "[H", 0
ansi_reset:   db 27, "[0m", 0
ansi_cyan:    db 27, "[36m", 0
ansi_bcyan:   db 27, "[1;96m", 0
ansi_red:     db 27, "[91m", 0
ansi_yel:     db 27, "[93m", 0
ansi_grn:     db 27, "[92m", 0
ansi_bold_g:  db 27, "[1;92m", 0
ansi_bold_y:  db 27, "[1;93m", 0
ansi_bold_w:  db 27, "[1;97m", 0
ansi_white:   db 27, "[97m", 0
ansi_dim:     db 27, "[90m", 0

ansi_lborder:    db 27, "[36m", 0xE2, 0x94, 0x82, 27, "[0m ", 0
ansi_rborder_nl: db 27, "[36m", 0xE2, 0x94, 0x82, 27, "[0m", 10, 0

; UTF-8 box-drawing & chart glyphs
u_tl:           db 0xE2, 0x95, 0xAD, 0          ; ╭
u_tr:           db 0xE2, 0x95, 0xAE, 0          ; ╮
u_bl:           db 0xE2, 0x95, 0xB0, 0          ; ╰
u_br:           db 0xE2, 0x95, 0xAF, 0          ; ╯
u_ml:           db 0xE2, 0x94, 0x9C, 0          ; ├
u_mr:           db 0xE2, 0x94, 0xA4, 0          ; ┤
u_horiz:        db 0xE2, 0x94, 0x80, 0          ; ─
u_vbar:         db 0xE2, 0x94, 0x82, 0          ; │
u_rtick:        db 0xE2, 0x94, 0xA4, 0          ; ┤
u_bl_axis:      db 0xE2, 0x94, 0x94, 0          ; └
u_diamond:      db 0xE2, 0x97, 0x86, 0          ; ◆
u_dot:          db 0xC2, 0xB7, 0                ; ·
u_braille_full: db 0xE2, 0xA3, 0xBF, 0          ; ⣿
u_block_full:   db 0xE2, 0x96, 0x88, 0          ; █
u_block_empty:  db 0xE2, 0x96, 0x91, 0          ; ░

; Labels
s_title_1:      db 27, "[1;96masm-gpt-train", 27, "[0m", 0
s_title_2:      db " ", 0xE2, 0x94, 0x82, " ", 27, "[1;97m100% Pure x86-64 Assembly", 27, "[0m", 0
s_title_3_txt:  db " (Zero libc, AVX2+FMA)", 0
s_sep:          db " ", 0xE2, 0x94, 0x82, " ", 0
s_cfg_l:        db "L=", 0
s_cfg_d:        db " D=", 0
s_cfg_h:        db " H=", 0
s_cfg_ff:       db " FF=", 0
s_cfg_v:        db " V=", 0
s_cfg_p:        db "params=", 0
s_cfg_b:        db "B=", 0
s_cfg_t:        db " T=", 0
s_cfg_thr:      db "threads=", 0
s_chart_hdr:    db "Loss Curve (0..", 0
s_chart_hdr2:   db ")    ", 0
s_leg_tr:       db " train   ", 0
s_leg_va:       db " val", 0
s_leg_best:     db "    best_val=", 0
s_x0:           db "step 0 ", 0
s_step_lbl:     db "Step ", 0
s_bar_open:     db " [", 0
s_bar_close:    db "] ", 0
s_pct:          db "%", 0
s_elapsed:      db "elapsed ", 0
s_m_loss:       db "loss=", 0
s_m_val:        db "  val=", 0
s_m_best:       db "  best=", 0
s_m_gnorm:      db "  gnorm=", 0
s_m_lr:         db "  lr=", 0
s_sp2:          db "  ", 0
s_tps:          db " tok/s", 0
s_dash:         db "----", 0
s_samp_hdr:     db "Live Sample (step ", 0
s_no_samp:      db "(awaiting first sample...)", 0

section .bss
alignb 8
g_sample_capture: resq 1
tui_threads:      resq 1
tui_n_train:      resq 1
tui_cur_step1:    resq 1
tui_val_cnt:      resq 1
tui_has_val:      resq 1
tui_sample_len:   resq 1
tui_sample_step:  resq 1
tui_step_ns:      resq 1
tui_elapsed_ns:   resq 1
tui_last_loss:    resd 1
tui_last_val:     resd 1
tui_best_val:     resd 1
tui_last_gnorm:   resd 1
tui_last_lr:      resd 1

alignb 64
tui_train_hist:   resd MAX_TRAIN_HIST
tui_val_step:     resq MAX_VAL_HIST
tui_val_hist:     resd MAX_VAL_HIST
tui_bin_sum:      resq PLOT_PX_W
tui_bin_cnt:      resd PLOT_PX_W
tui_braille:      resb PLOT_COLS * PLOT_ROWS
tui_val_mark:     resb PLOT_COLS * PLOT_ROWS
tui_sample_buf:   resb MAX_SAMPLE_CAP
tui_out_buf:      resb 32768
