; ============================================================================
; test_math.asm — Zero-libc standalone binary to unit-test pure-asm math
; primitives (fast_expf_avx2, fast_sincosf, init_rope_freqs, rmsnorm_avx2,
; silu_mul_avx2, quantize_q8_0_avx2, matmul_q8_0_range) against Python.
;
; Protocol: reads raw float32 inputs from stdin, writes raw float32 outputs
; to stdout:
;   stdin : [64 floats for exp] [64 floats for sincos] [128 floats x + 128 floats w for rmsnorm]
;           [64 floats gate + 64 floats up for silu_mul]
;           [64 floats x_vec + 8x64 Q8_0 v1 fp32-scale rows (8*72=576B)
;            + 8x64 Q8_0 v2 fp16-scale rows (8*68=544B)]
;   stdout: [64 exp_out] [64 sin_out] [64 cos_out] [64 rope_inv_freq f64 (512B)]
;           [128 rmsnorm_out] [64 silu_out] [8 matmul_v1_out] [8 matmul_v2_out]
; ============================================================================

bits 64
default rel

%include "constants.inc"

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%define IN_TOTAL_BYTES (2304 + 576 + 544)

extern sys_write_all
extern sys_exit_group
extern fast_expf_avx2
extern fast_sincosf
extern init_rope_freqs
extern rmsnorm_avx2
extern silu_mul_avx2
extern matmul_set_fp16
%if FAST_Q8
extern quantize_x_q8_0
%endif
extern matmul_q8_0_range

section .text
global _start

; read_all(fd=0, buf=rdi, len=rsi)
read_all:
    push    rbx
    push    r12
    mov     rbx, rdi
    mov     r12, rsi
.lp:
    test    r12, r12
    jz      .done
    xor     edi, edi            ; stdin
    mov     rsi, rbx
    mov     rdx, r12
    mov     eax, SYS_read
    syscall
    cmp     rax, -4
    je      .lp
    test    rax, rax
    jle     .done
    add     rbx, rax
    sub     r12, rax
    jmp     .lp
.done:
    pop     r12
    pop     rbx
    ret

_start:
    and     rsp, -16

    ; Read all test inputs in one chunk
    lea     rdi, [rel in_exp]
    mov     rsi, IN_TOTAL_BYTES
    call    read_all

    ; 1. fast_expf_avx2 on 64 floats (8 iterations of 8 floats)
    xor     ebx, ebx
.exp_lp:
    cmp     rbx, 256
    jae     .do_sincos
    lea     rax, [rel in_exp]
    vmovups ymm0, [rax + rbx]
    call    fast_expf_avx2
    lea     rax, [rel out_exp]
    vmovups [rax + rbx], ymm0
    add     rbx, 32
    jmp     .exp_lp

    ; 2. fast_sincosf on 64 floats
.do_sincos:
    xor     ebx, ebx
.sc_lp:
    cmp     rbx, 64
    jae     .do_rope
    lea     rax, [rel in_sc]
    vmovss  xmm0, [rax + rbx*4]
    call    fast_sincosf        ; xmm0=sin, xmm1=cos
    lea     rax, [rel out_sin]
    vmovss  [rax + rbx*4], xmm0
    lea     rax, [rel out_cos]
    vmovss  [rax + rbx*4], xmm1
    inc     rbx
    jmp     .sc_lp

    ; 3. init_rope_freqs(out_rope, 128, 1000000.0f)
.do_rope:
    lea     rdi, [rel out_rope]
    mov     rsi, 128
    vmovss  xmm0, [rel f_theta]
    call    init_rope_freqs

    ; 4. rmsnorm_avx2(out_rms, in_rms_x, in_rms_w, 128, 1e-6f)
    lea     rdi, [rel out_rms]
    lea     rsi, [rel in_rms_x]
    lea     rdx, [rel in_rms_w]
    mov     rcx, 128
    vmovss  xmm0, [rel f_eps]
    call    rmsnorm_avx2

    ; 5. silu_mul_avx2(in_gate, in_up, 64) -> copy to out_silu
    lea     rdi, [rel in_gate]
    lea     rsi, [rel in_up]
    mov     rdx, 64
    call    silu_mul_avx2

    ; 6. matmul_q8_0_range (8 rows x 64 cols): v1 fp32 scales + v2 fp16 scales
%if FAST_Q8
    lea     rdi, [rel in_mm_x]
    mov     rsi, 64
    call    quantize_x_q8_0
%endif
    xor     edi, edi
    call    matmul_set_fp16
    lea     rdi, [rel in_mm_w]
    lea     rsi, [rel in_mm_x]
    lea     rdx, [rel out_mm]
    mov     rcx, 64             ; K
    xor     r8d, r8d            ; r0 = 0
    mov     r9,  8              ; r1 = 8
    call    matmul_q8_0_range

    mov     edi, 1
    call    matmul_set_fp16
    lea     rdi, [rel in_mm_w16]
    lea     rsi, [rel in_mm_x]
    lea     rdx, [rel out_mm16]
    mov     rcx, 64             ; K
    xor     r8d, r8d            ; r0 = 0
    mov     r9,  8              ; r1 = 8
    call    matmul_q8_0_range

    ; Write all outputs to stdout
    mov     edi, 1
    lea     rsi, [rel out_exp]
    mov     edx, 256            ; 64 f32
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_sin]
    mov     edx, 256            ; 64 f32
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_cos]
    mov     edx, 256            ; 64 f32
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_rope]
    mov     edx, 512            ; 64 f64
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_rms]
    mov     edx, 512            ; 128 f32
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel in_gate]
    mov     edx, 256            ; 64 f32
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_mm]
    mov     edx, 32             ; 8 f32 (v1 fp32 scales)
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel out_mm16]
    mov     edx, 32             ; 8 f32 (v2 fp16 scales)
    call    sys_write_all

    xor     edi, edi
    call    sys_exit_group

section .rodata
align 4
f_theta: dd 1000000.0
f_eps:   dd 1.0e-6

section .bss
alignb 64
; Contiguous input block (read in one read_all)
in_exp:   resb 256              ; 64 f32
in_sc:    resb 256              ; 64 f32
in_rms_x: resb 512              ; 128 f32
in_rms_w: resb 512              ; 128 f32
in_gate:  resb 256              ; 64 f32
in_up:    resb 256              ; 64 f32
in_mm_x:  resb 256              ; 64 f32
in_mm_w:  resb 576              ; 8 * (64 + 8) v1 fp32 scales
in_mm_w16: resb 544             ; 8 * (64 + 4) v2 fp16 scales

alignb 64
out_exp:  resb 256
out_sin:  resb 256
out_cos:  resb 256
out_rope: resb 512
out_rms:  resb 512
out_mm:   resb 32
out_mm16: resb 32
