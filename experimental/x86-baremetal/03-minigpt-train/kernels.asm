; ============================================================================
; kernels.asm — Pure AVX2+FMA Forward/Backward Neural Kernels, PRNG & AdamW
; Zero libc, zero libm. Reuses math.asm primitives (fast_expf_avx2, log_f64,
; fast_sincos_f64, softmax_avx2) and sgemm.asm thread pool (tp_run).
; ============================================================================

bits 64
default rel

%include "train_constants.inc"

extern fast_expf_avx2
extern fast_expf_scalar
extern fast_sincos_f64
extern log_f64
extern softmax_avx2
extern tp_run
extern tp_get_threads

section .text

global zero_f32_avx2
global vec_add_f32_avx2
global vec_add3_f32_avx2
global tp_rmsnorm_fwd
global tp_rmsnorm_bwd
global tp_rope_batch
global tp_mha_fwd
global tp_mha_bwd
global tp_swiglu_fwd
global tp_swiglu_bwd
global tp_cross_entropy_fwd_bwd
global tp_grad_norm
global tp_vec_scale
global tp_adamw_range
global rng_next_u64
global init_normal_f32
global init_ones_f32
global compute_lr

; ============================================================================
; 1. SIMPLE VECTOR HELPERS (arbitrary n)
; ============================================================================

; void zero_f32_avx2(float *dst, uint64 n)
zero_f32_avx2:
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.lp32:
    lea     rcx, [rax + 32]
    cmp     rcx, rsi
    ja      .lp8
    vmovups [rdi + rax*4],      ymm0
    vmovups [rdi + rax*4 + 32], ymm0
    vmovups [rdi + rax*4 + 64], ymm0
    vmovups [rdi + rax*4 + 96], ymm0
    add     rax, 32
    jmp     .lp32
.lp8:
    lea     rcx, [rax + 8]
    cmp     rcx, rsi
    ja      .lp1
    vmovups [rdi + rax*4], ymm0
    add     rax, 8
    jmp     .lp8
.lp1:
    cmp     rax, rsi
    jae     .done
    mov     dword [rdi + rax*4], 0
    inc     rax
    jmp     .lp1
.done:
    vzeroupper
    ret

; void vec_add_f32_avx2(float *dst, const float *src, uint64 n)
; dst[i] += src[i]
vec_add_f32_avx2:
    xor     eax, eax
.lp8:
    lea     rcx, [rax + 8]
    cmp     rcx, rdx
    ja      .lp1
    vmovups ymm0, [rdi + rax*4]
    vaddps  ymm0, ymm0, [rsi + rax*4]
    vmovups [rdi + rax*4], ymm0
    add     rax, 8
    jmp     .lp8
.lp1:
    cmp     rax, rdx
    jae     .done
    vmovss  xmm0, [rdi + rax*4]
    vaddss  xmm0, xmm0, [rsi + rax*4]
    vmovss  [rdi + rax*4], xmm0
    inc     rax
    jmp     .lp1
.done:
    vzeroupper
    ret

; void vec_add3_f32_avx2(float *dst, const float *a, const float *b, uint64 n)
; dst[i] = a[i] + b[i]
vec_add3_f32_avx2:
    xor     eax, eax
.lp8:
    lea     r8, [rax + 8]
    cmp     r8, rcx
    ja      .lp1
    vmovups ymm0, [rsi + rax*4]
    vaddps  ymm0, ymm0, [rdx + rax*4]
    vmovups [rdi + rax*4], ymm0
    add     rax, 8
    jmp     .lp8
.lp1:
    cmp     rax, rcx
    jae     .done
    vmovss  xmm0, [rsi + rax*4]
    vaddss  xmm0, xmm0, [rdx + rax*4]
    vmovss  [rdi + rax*4], xmm0
    inc     rax
    jmp     .lp1
.done:
    vzeroupper
    ret

; Internal helper: float dot_8w(const float *a, const float *b, uint64 n) -> xmm0
; n must be a multiple of 8 (e.g., head_dim = 16, 32, 64).
dot_8w:
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
    shl     rdx, 2
.lp:
    cmp     rax, rdx
    jae     .red
    vmovups ymm1, [rdi + rax]
    vfmadd231ps ymm0, ymm1, [rsi + rax]
    add     rax, 32
    jmp     .lp
.red:
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    ret

; Internal helper: void saxpy_8w(float *dst, const float *src, uint64 n, float alpha)
; rdi=dst, rsi=src, rdx=n (multiple of 8), xmm0=alpha
saxpy_8w:
    vbroadcastss ymm0, xmm0
    xor     eax, eax
    shl     rdx, 2
.lp:
    cmp     rax, rdx
    jae     .done
    vmovups ymm1, [rdi + rax]
    vfmadd231ps ymm1, ymm0, [rsi + rax]
    vmovups [rdi + rax], ymm1
    add     rax, 32
    jmp     .lp
.done:
    ret

; ============================================================================
; 2. RMSNORM FORWARD & BACKWARD (parallel & bit-for-bit deterministic)
; ============================================================================

; void tp_rmsnorm_fwd(float *out, float *inv_rms, const float *x, const float *w,
;                     uint64 N, uint64 D, float eps)
; rdi=out, rsi=inv_rms, rdx=x, rcx=w, r8=N, r9=D, xmm0=eps
tp_rmsnorm_fwd:
    sub     rsp, 72                     ; 56B job + align (72+8=80)
    mov     [rsp + 0],  rdi             ; out
    mov     [rsp + 8],  rsi             ; inv_rms
    mov     [rsp + 16], rdx             ; x
    mov     [rsp + 24], rcx             ; w
    mov     [rsp + 32], r8              ; N
    mov     [rsp + 40], r9              ; D
    vmovss  [rsp + 48], xmm0            ; eps
    lea     rdi, [rel rmsnorm_fwd_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 72
    ret

rmsnorm_fwd_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbp, rdx                    ; job
    mov     r10, [rbp + 32]             ; N
    mov     r11, [rbp + 40]             ; D (mult of 8)

    ; n0 = (N * tid) / n_threads, n1 = (N * (tid + 1)) / n_threads
    mov     rax, r10
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; n = n0

    lea     rax, [rdi + 1]
    imul    rax, r10
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; n1

    mov     r8,  [rbp + 0]              ; out
    mov     r9,  [rbp + 8]              ; inv_rms
    mov     r14, [rbp + 16]             ; x
    mov     r15, [rbp + 24]             ; w
    mov     rbx, r11
    shl     rbx, 2                      ; D_bytes = D * 4
    vcvtsi2ss xmm6, xmm6, r11           ; (float)D
    vmovss  xmm7, [rel kf_one]
    vdivss  xmm6, xmm7, xmm6            ; inv_D = 1.0f / D
    vmovss  xmm5, [rbp + 48]            ; eps

.tok_lp:
    cmp     r12, r13
    jae     .done
    mov     rax, r12
    imul    rax, rbx                    ; n * D * 4
    lea     rdi, [r14 + rax]            ; x_row
    lea     rsi, [r8 + rax]             ; out_row

    ; Sum x^2
    vxorps  ymm0, ymm0, ymm0
    xor     ecx, ecx
.ss_lp:
    cmp     rcx, rbx
    jae     .ss_red
    vmovups ymm1, [rdi + rcx]
    vfmadd231ps ymm0, ymm1, ymm1
    add     rcx, 32
    jmp     .ss_lp
.ss_red:
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    vmulss  xmm0, xmm0, xmm6            ; mean(x^2)
    vaddss  xmm0, xmm0, xmm5            ; + eps
    vsqrtss xmm0, xmm0, xmm0            ; rms
    vdivss  xmm0, xmm7, xmm0            ; inv_rms = 1 / rms
    vmovss  [r9 + r12*4], xmm0
    vbroadcastss ymm2, xmm0

    xor     ecx, ecx
.sc_lp:
    cmp     rcx, rbx
    jae     .tok_next
    vmovups ymm0, [rdi + rcx]
    vmulps  ymm0, ymm0, ymm2
    vmulps  ymm0, ymm0, [r15 + rcx]
    vmovups [rsi + rcx], ymm0
    add     rcx, 32
    jmp     .sc_lp
.tok_next:
    inc     r12
    jmp     .tok_lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; void tp_rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x,
;                     const float *inv_rms, const float *w, uint64 N, uint64 D)
; rdi=dx, rsi=dw, rdx=dy, rcx=x, r8=inv_rms, r9=w, [rsp+8]=N, [rsp+16]=D
tp_rmsnorm_bwd:
    mov     r10, [rsp + 8]              ; N
    mov     r11, [rsp + 16]             ; D
    sub     rsp, 72                     ; 64B job + 8B align
    mov     [rsp + 0],  rdi             ; dx
    mov     [rsp + 8],  rsi             ; dw
    mov     [rsp + 16], rdx             ; dy
    mov     [rsp + 24], rcx             ; x
    mov     [rsp + 32], r8              ; inv_rms
    mov     [rsp + 40], r9              ; w
    mov     [rsp + 48], r10             ; N
    mov     [rsp + 56], r11             ; D

    ; Phase A: compute dx (parallel over tokens n)
    lea     rdi, [rel rmsnorm_bwd_dx_worker]
    mov     rsi, rsp
    call    tp_run

    ; Phase B: accumulate dw += sum_n dy*x*inv_rms (parallel over 8-float chunks of d)
    lea     rdi, [rel rmsnorm_bwd_dw_worker]
    mov     rsi, rsp
    call    tp_run

    add     rsp, 72
    ret

rmsnorm_bwd_dx_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbp, rdx                    ; job
    mov     r10, [rbp + 48]             ; N
    mov     r11, [rbp + 56]             ; D

    mov     rax, r10
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; n0

    lea     rax, [rdi + 1]
    imul    rax, r10
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; n1

    mov     rbx, r11
    shl     rbx, 2                      ; D_bytes
    vcvtsi2ss xmm6, xmm6, r11
    vmovss  xmm7, [rel kf_one]
    vdivss  xmm6, xmm7, xmm6            ; inv_D

    mov     r8,  [rbp + 0]              ; dx
    mov     r9,  [rbp + 16]             ; dy
    mov     r14, [rbp + 24]             ; x
    mov     r15, [rbp + 40]             ; w

.tok_lp:
    cmp     r12, r13
    jae     .done
    mov     rax, r12
    imul    rax, rbx                    ; n * D * 4
    lea     rdi, [r14 + rax]            ; x_row
    lea     rsi, [r9  + rax]            ; dy_row
    lea     rdx, [r8  + rax]            ; dx_row

    ; dot = sum_d (dy[d] * w[d] * x[d])
    vxorps  ymm0, ymm0, ymm0
    xor     ecx, ecx
.dot_lp:
    cmp     rcx, rbx
    jae     .dot_red
    vmovups ymm1, [rsi + rcx]           ; dy
    vmulps  ymm1, ymm1, [r15 + rcx]     ; dy * w
    vfmadd231ps ymm0, ymm1, [rdi + rcx] ; += (dy * w) * x
    add     rcx, 32
    jmp     .dot_lp
.dot_red:
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0            ; xmm0 = dot

    mov     rax, [rbp + 32]             ; inv_rms ptr
    vmovss  xmm2, [rax + r12*4]         ; r = inv_rms[n]
    vmulss  xmm3, xmm2, xmm2            ; r^2
    vmulss  xmm0, xmm0, xmm3            ; dot * r^2
    vmulss  xmm0, xmm0, xmm6            ; c = dot * r^2 / D
    vbroadcastss ymm2, xmm2             ; r
    vbroadcastss ymm3, xmm0             ; c

    xor     ecx, ecx
.dx_lp:
    cmp     rcx, rbx
    jae     .tok_next
    vmovups ymm0, [rsi + rcx]           ; dy
    vmulps  ymm0, ymm0, [r15 + rcx]     ; dy * w
    vfnmadd231ps ymm0, ymm3, [rdi + rcx]; dy * w - c * x
    vmulps  ymm0, ymm0, ymm2            ; r * (dy * w - c * x)
    vmovups [rdx + rcx], ymm0
    add     rcx, 32
    jmp     .dx_lp
.tok_next:
    inc     r12
    jmp     .tok_lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; Phase B: parallel over 8-float channel blocks c = 0 .. (D/8)-1
; Deterministic sequential accumulation over n = 0 .. N-1!
rmsnorm_bwd_dw_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbp, rdx                    ; job
    mov     r10, [rbp + 48]             ; N
    mov     r11, [rbp + 56]             ; D
    mov     rcx, r11
    shr     rcx, 3                      ; n_blocks = D / 8

    mov     rax, rcx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; b0

    lea     rax, [rdi + 1]
    imul    rax, rcx
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; b1

    mov     rbx, r11
    shl     rbx, 2                      ; D_bytes
    mov     r8,  [rbp + 8]              ; dw
    mov     r9,  [rbp + 16]             ; dy
    mov     r14, [rbp + 24]             ; x
    mov     r15, [rbp + 32]             ; inv_rms

.blk_lp:
    cmp     r12, r13
    jae     .done
    mov     rax, r12
    shl     rax, 5                      ; byte_off = blk * 32
    vmovups ymm0, [r8 + rax]            ; accumulate onto existing dw!
    lea     rdi, [r9  + rax]            ; dy_ptr
    lea     rsi, [r14 + rax]            ; x_ptr

    xor     ecx, ecx                    ; n = 0
.n_lp:
    cmp     rcx, r10
    jae     .blk_store
    vbroadcastss ymm1, [r15 + rcx*4]    ; inv_rms[n]
    vmovups ymm2, [rdi]                 ; dy[n, d..d+7]
    vmulps  ymm2, ymm2, [rsi]           ; dy * x
    vfmadd231ps ymm0, ymm2, ymm1        ; += dy * x * inv_rms[n]
    add     rdi, rbx
    add     rsi, rbx
    inc     rcx
    jmp     .n_lp

.blk_store:
    vmovups [r8 + rax], ymm0
    inc     r12
    jmp     .blk_lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; 3. SPLIT-HALF RoPE FORWARD & BACKWARD (batched [B, T, H, hd])
; ============================================================================

; void tp_rope_batch(float *x, const float *cos_tab, const float *sin_tab,
;                    uint64 B, uint64 T, uint64 H, uint64 hd, uint64 is_bwd)
; rdi=x, rsi=cos_tab, rdx=sin_tab, rcx=B, r8=T, r9=H, [rsp+8]=hd, [rsp+16]=is_bwd
tp_rope_batch:
    mov     r10, [rsp + 8]              ; hd
    mov     r11, [rsp + 16]             ; is_bwd
    sub     rsp, 72
    mov     [rsp + 0],  rdi             ; x
    mov     [rsp + 8],  rsi             ; cos_tab
    mov     [rsp + 16], rdx             ; sin_tab
    mov     [rsp + 24], rcx             ; B
    mov     [rsp + 32], r8              ; T
    mov     [rsp + 40], r9              ; H
    mov     [rsp + 48], r10             ; hd
    mov     [rsp + 56], r11             ; is_bwd
    lea     rdi, [rel rope_batch_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 72
    ret

rope_batch_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbp, rdx                    ; job
    mov     r10, [rbp + 24]             ; B
    imul    r10, [rbp + 32]             ; BT = B * T

    mov     rax, r10
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; n = n0

    lea     rax, [rdi + 1]
    imul    rax, r10
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; n1

    mov     r8,  [rbp + 48]             ; hd
    mov     r9,  r8
    shr     r9,  1
    shl     r9,  2                      ; half_bytes = (hd/2) * 4
    lea     r10, [r9 + r9]              ; head_bytes = hd * 4
    mov     r11, [rbp + 40]             ; H
    mov     r14, r11
    imul    r14, r10                    ; tok_bytes = H * hd * 4 = D * 4
    mov     r15, [rbp + 56]             ; is_bwd

.tok_lp:
    cmp     r12, r13
    jae     .done
    ; t = n % T
    mov     rax, r12
    xor     edx, edx
    div     qword [rbp + 32]            ; rdx = t
    mov     rax, rdx
    imul    rax, r9                     ; t * half_bytes
    mov     rsi, [rbp + 8]
    add     rsi, rax                    ; cos_t
    mov     rdx, [rbp + 16]
    add     rdx, rax                    ; sin_t

    mov     rdi, r12
    imul    rdi, r14
    add     rdi, [rbp + 0]              ; x_tok = x + n * D * 4

    mov     rcx, r11                    ; H heads
.head_lp:
    test    rcx, rcx
    jz      .tok_next
    lea     rbx, [rdi + r9]             ; x2_ptr
    xor     eax, eax
    test    r15, r15
    jnz     .dim_bwd

.dim_fwd:
    cmp     rax, r9
    jae     .head_next
    vmovups ymm0, [rdi + rax]           ; x1
    vmovups ymm1, [rbx + rax]           ; x2
    vmovups ymm2, [rsi + rax]           ; cos
    vmovups ymm3, [rdx + rax]           ; sin
    vmulps  ymm4, ymm0, ymm2            ; x1 * cos
    vfnmadd231ps ymm4, ymm1, ymm3       ; x1 * cos - x2 * sin
    vmulps  ymm5, ymm1, ymm2            ; x2 * cos
    vfmadd231ps ymm5, ymm0, ymm3        ; x2 * cos + x1 * sin
    vmovups [rdi + rax], ymm4
    vmovups [rbx + rax], ymm5
    add     rax, 32
    jmp     .dim_fwd

.dim_bwd:
    cmp     rax, r9
    jae     .head_next
    vmovups ymm0, [rdi + rax]           ; dy1
    vmovups ymm1, [rbx + rax]           ; dy2
    vmovups ymm2, [rsi + rax]           ; cos
    vmovups ymm3, [rdx + rax]           ; sin
    vmulps  ymm4, ymm0, ymm2            ; dy1 * cos
    vfmadd231ps ymm4, ymm1, ymm3        ; dy1 * cos + dy2 * sin
    vmulps  ymm5, ymm1, ymm2            ; dy2 * cos
    vfnmadd231ps ymm5, ymm0, ymm3       ; dy2 * cos - dy1 * sin
    vmovups [rdi + rax], ymm4
    vmovups [rbx + rax], ymm5
    add     rax, 32
    jmp     .dim_bwd

.head_next:
    add     rdi, r10
    dec     rcx
    jmp     .head_lp

.tok_next:
    inc     r12
    jmp     .tok_lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; 4. CAUSAL MULTI-HEAD SELF-ATTENTION FORWARD & BACKWARD
;    Parallel over (b, h) in 0 .. B*H - 1 (zero races, 100% deterministic!)
; ============================================================================

; void tp_mha_fwd(const float *Q, const float *K, const float *V,
;                 float *P, float *out,
;                 uint64 B, uint64 T, uint64 H, uint64 hd, float scale)
; rdi=Q, rsi=K, rdx=V, rcx=P, r8=out, r9=B,
; [rsp+8]=T, [rsp+16]=H, [rsp+24]=hd, xmm0=scale
tp_mha_fwd:
    mov     r10, [rsp + 8]              ; T
    mov     r11, [rsp + 16]             ; H
    mov     rax, [rsp + 24]             ; hd
    sub     rsp, 88                     ; 80B job + align
    mov     [rsp + 0],  rdi             ; Q
    mov     [rsp + 8],  rsi             ; K
    mov     [rsp + 16], rdx             ; V
    mov     [rsp + 24], rcx             ; P
    mov     [rsp + 32], r8              ; out
    mov     [rsp + 40], r9              ; B
    mov     [rsp + 48], r10             ; T
    mov     [rsp + 56], r11             ; H
    mov     [rsp + 64], rax             ; hd
    vmovss  [rsp + 72], xmm0            ; scale
    lea     rdi, [rel mha_fwd_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 88
    ret

mha_fwd_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 88                     ; 6 pushes(48) + 88 + 8 = 144 (16B aligned)
    ; Locals:
    ;   [rsp + 0]  = job
    ;   [rsp + 8]  = bh_cur
    ;   [rsp + 16] = bh_end
    ;   [rsp + 24] = D_bytes (H * hd * 4)
    ;   [rsp + 32] = T_bytes (T * 4)
    ;   [rsp + 40] = q_bh
    ;   [rsp + 48] = k_bh
    ;   [rsp + 56] = v_bh
    ;   [rsp + 64] = out_bh
    ;   [rsp + 72] = p_bh
    ;   [rsp + 80] = i_cur
    mov     [rsp + 0], rdx
    mov     rbp, rdx

    mov     rcx, [rbp + 40]             ; B
    imul    rcx, [rbp + 56]             ; total_bh = B * H

    mov     rax, rcx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     [rsp + 8], rax              ; bh0

    lea     rax, [rdi + 1]
    imul    rax, rcx
    xor     edx, edx
    div     rsi
    mov     [rsp + 16], rax             ; bh1

    mov     rax, [rbp + 56]             ; H
    imul    rax, [rbp + 64]             ; D = H * hd
    shl     rax, 2
    mov     [rsp + 24], rax             ; D_bytes
    mov     rax, [rbp + 48]             ; T
    shl     rax, 2
    mov     [rsp + 32], rax             ; T_bytes

.bh_lp:
    mov     rax, [rsp + 8]
    cmp     rax, [rsp + 16]
    jae     .done
    mov     rbp, [rsp + 0]

    ; b = bh / H, h = bh % H
    xor     edx, edx
    div     qword [rbp + 56]            ; rax = b, rdx = h
    ; base_off = (b * T * D + h * hd) * 4 = b * T * D_bytes + h * hd * 4
    imul    rax, [rbp + 48]
    imul    rax, [rsp + 24]             ; b * T * D_bytes
    imul    rdx, [rbp + 64]
    shl     rdx, 2                      ; h * hd * 4
    add     rax, rdx                    ; head byte offset in Q/K/V/out

    mov     rcx, [rbp + 0]
    add     rcx, rax
    mov     [rsp + 40], rcx             ; q_bh
    mov     rcx, [rbp + 8]
    add     rcx, rax
    mov     [rsp + 48], rcx             ; k_bh
    mov     rcx, [rbp + 16]
    add     rcx, rax
    mov     [rsp + 56], rcx             ; v_bh
    mov     rcx, [rbp + 32]
    add     rcx, rax
    mov     [rsp + 64], rcx             ; out_bh

    ; p_bh = P + bh * T * T * 4
    mov     rax, [rsp + 8]
    imul    rax, [rbp + 48]
    imul    rax, [rsp + 32]
    add     rax, [rbp + 24]
    mov     [rsp + 72], rax             ; p_bh

    mov     qword [rsp + 80], 0         ; i = 0
.i_lp:
    mov     r12, [rsp + 80]
    mov     rbp, [rsp + 0]
    cmp     r12, [rbp + 48]             ; i < T
    jae     .bh_next

    ; q_i = q_bh + i * D_bytes
    mov     rax, r12
    imul    rax, [rsp + 24]
    mov     r13, [rsp + 40]
    add     r13, rax                    ; r13 = q_i
    mov     r14, [rsp + 64]
    add     r14, rax                    ; r14 = out_i

    ; p_row = p_bh + i * T_bytes
    mov     rax, r12
    imul    rax, [rsp + 32]
    mov     r15, [rsp + 72]
    add     r15, rax                    ; r15 = p_row

    ; Compute scores[0..i]: p_row[j] = dot(q_i, k_j) * scale
    mov     rbx, [rsp + 48]             ; k_j = k_bh
    xor     r8d, r8d                    ; j = 0
.j_sc_lp:
    cmp     r8, r12
    ja      .do_sm
    mov     rdi, r13                    ; q_i
    mov     rsi, rbx                    ; k_j
    mov     rdx, [rbp + 64]             ; hd
    call    dot_8w                      ; xmm0 = dot (preserves r8, rbx, r12..r15, rbp!)
    vmulss  xmm0, xmm0, [rbp + 72]      ; * scale
    vmovss  [r15 + r8*4], xmm0
    add     rbx, [rsp + 24]             ; k_j += D_bytes
    inc     r8
    jmp     .j_sc_lp

.do_sm:
    mov     rdi, r15                    ; p_row
    lea     rsi, [r12 + 1]              ; len = i + 1
    call    softmax_avx2

    ; Zero out_i (hd floats, mult of 8)
    mov     rbp, [rsp + 0]
    mov     rcx, [rbp + 64]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.zero_o:
    cmp     rax, rcx
    jae     .v_accum
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .zero_o

.v_accum:
    mov     rbx, [rsp + 56]             ; v_j = v_bh
    xor     r8d, r8d                    ; j = 0
.j_v_lp:
    cmp     r8, r12
    ja      .i_next
    vmovss  xmm0, [r15 + r8*4]          ; p_row[j]
    mov     rdi, r14                    ; out_i
    mov     rsi, rbx                    ; v_j
    mov     rdx, [rbp + 64]             ; hd
    call    saxpy_8w                    ; preserves r8, rbx, r12..r15, rbp!
    add     rbx, [rsp + 24]             ; v_j += D_bytes
    inc     r8
    jmp     .j_v_lp

.i_next:
    inc     qword [rsp + 80]
    jmp     .i_lp

.bh_next:
    inc     qword [rsp + 8]
    jmp     .bh_lp

.done:
    vzeroupper
    add     rsp, 88
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; void tp_mha_bwd(const float *d_out, const float *Q, const float *K,
;                 const float *V, const float *P,
;                 float *dQ, float *dK, float *dV,
;                 uint64 B, uint64 T, uint64 H, uint64 hd, float scale)
; Stack/reg args (System V):
;   rdi=d_out, rsi=Q, rdx=K, rcx=V, r8=P, r9=dQ
;   [rsp+8]=dK, [rsp+16]=dV, [rsp+24]=B, [rsp+32]=T, [rsp+40]=H, [rsp+48]=hd
;   xmm0=scale
tp_mha_bwd:
    ; Load stack args before sub rsp
    mov     rax, [rsp + 8]              ; dK
    mov     r10, [rsp + 16]             ; dV
    mov     r11, [rsp + 24]             ; B
    push    rbx
    push    r12
    push    r13
    ; 3 pushes = 24B -> original [rsp+32] is at [rsp+56], +40 at +64, +48 at +72
    mov     rbx, [rsp + 56]             ; T
    mov     r12, [rsp + 64]             ; H
    mov     r13, [rsp + 72]             ; hd
    sub     rsp, 112                    ; 3 pushes(24) + 112 + 8 = 144 (16B aligned)

    mov     [rsp + 0],  rdi             ; d_out
    mov     [rsp + 8],  rsi             ; Q
    mov     [rsp + 16], rdx             ; K
    mov     [rsp + 24], rcx             ; V
    mov     [rsp + 32], r8              ; P
    mov     [rsp + 40], r9              ; dQ
    mov     [rsp + 48], rax             ; dK
    mov     [rsp + 56], r10             ; dV
    mov     [rsp + 64], r11             ; B
    mov     [rsp + 72], rbx             ; T
    mov     [rsp + 80], r12             ; H
    mov     [rsp + 88], r13             ; hd
    vmovss  [rsp + 96], xmm0            ; scale

    lea     rdi, [rel mha_bwd_worker]
    mov     rsi, rsp
    call    tp_run

    add     rsp, 112
    pop     r13
    pop     r12
    pop     rbx
    ret

mha_bwd_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    ; Reserve fixed locals (128B) + dynamic dP buffer on stack (T * 4, aligned to 32B)
    mov     rbp, rsp                    ; frame pointer to restore rsp cleanly!
    sub     rsp, 128                    ; 6 pushes(48) + 128 + 8 = 184 -> round below to 16B!
    mov     rax, [rdx + 72]             ; T
    shl     rax, 2                      ; T * 4
    add     rax, 31
    and     rax, -32
    sub     rsp, rax
    and     rsp, -32                    ; 32B aligned!
    ; [rsp] is dP[0..T-1] buffer!
    ; Locals relative to rbp:
    ;   [rbp - 8]   = job
    ;   [rbp - 16]  = bh_cur
    ;   [rbp - 24]  = bh_end
    ;   [rbp - 32]  = D_bytes
    ;   [rbp - 40]  = T_bytes
    ;   [rbp - 48]  = do_bh
    ;   [rbp - 56]  = q_bh
    ;   [rbp - 64]  = k_bh
    ;   [rbp - 72]  = v_bh
    ;   [rbp - 80]  = p_bh
    ;   [rbp - 88]  = dq_bh
    ;   [rbp - 96]  = dk_bh
    ;   [rbp - 104] = dv_bh
    ;   [rbp - 112] = i_cur
    ;   [rbp - 120] = dP_ptr (= rsp)
    mov     [rbp - 8], rdx
    mov     [rbp - 120], rsp

    mov     r10, rdx                    ; job
    mov     rcx, [r10 + 64]             ; B
    imul    rcx, [r10 + 80]             ; total_bh = B * H

    mov     rax, rcx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     [rbp - 16], rax             ; bh0

    lea     rax, [rdi + 1]
    imul    rax, rcx
    xor     edx, edx
    div     rsi
    mov     [rbp - 24], rax             ; bh1

    mov     rax, [r10 + 80]             ; H
    imul    rax, [r10 + 88]             ; D = H * hd
    shl     rax, 2
    mov     [rbp - 32], rax             ; D_bytes
    mov     rax, [r10 + 72]             ; T
    shl     rax, 2
    mov     [rbp - 40], rax             ; T_bytes

.bh_lp:
    mov     rax, [rbp - 16]
    cmp     rax, [rbp - 24]
    jae     .done
    mov     r10, [rbp - 8]              ; job

    xor     edx, edx
    div     qword [r10 + 80]            ; rax = b, rdx = h
    imul    rax, [r10 + 72]
    imul    rax, [rbp - 32]             ; b * T * D_bytes
    imul    rdx, [r10 + 88]
    shl     rdx, 2                      ; h * hd * 4
    add     rax, rdx                    ; head offset

    mov     rcx, [r10 + 0]
    add     rcx, rax
    mov     [rbp - 48], rcx             ; do_bh
    mov     rcx, [r10 + 8]
    add     rcx, rax
    mov     [rbp - 56], rcx             ; q_bh
    mov     rcx, [r10 + 16]
    add     rcx, rax
    mov     [rbp - 64], rcx             ; k_bh
    mov     rcx, [r10 + 24]
    add     rcx, rax
    mov     [rbp - 72], rcx             ; v_bh
    mov     rcx, [r10 + 40]
    add     rcx, rax
    mov     [rbp - 88], rcx             ; dq_bh
    mov     rcx, [r10 + 48]
    add     rcx, rax
    mov     [rbp - 96], rcx             ; dk_bh
    mov     rcx, [r10 + 56]
    add     rcx, rax
    mov     [rbp - 104], rcx            ; dv_bh

    ; p_bh = P + bh * T * T_bytes
    mov     rax, [rbp - 16]
    imul    rax, [r10 + 72]
    imul    rax, [rbp - 40]
    add     rax, [r10 + 32]
    mov     [rbp - 80], rax             ; p_bh

    ; Zero dK_bh and dV_bh for all j = 0 .. T-1
    mov     rdi, [rbp - 96]             ; dk_j
    mov     rsi, [rbp - 104]            ; dv_j
    mov     rcx, [r10 + 72]             ; T
    mov     r8,  [r10 + 88]
    shl     r8,  2                      ; hd_bytes
    vxorps  ymm0, ymm0, ymm0
.z_kv_lp:
    test    rcx, rcx
    jz      .start_i
    xor     eax, eax
.z_kv_d:
    cmp     rax, r8
    jae     .z_kv_next
    vmovups [rdi + rax], ymm0
    vmovups [rsi + rax], ymm0
    add     rax, 32
    jmp     .z_kv_d
.z_kv_next:
    add     rdi, [rbp - 32]
    add     rsi, [rbp - 32]
    dec     rcx
    jmp     .z_kv_lp

.start_i:
    mov     qword [rbp - 112], 0        ; i = 0
.i_lp:
    mov     r12, [rbp - 112]
    mov     r10, [rbp - 8]
    cmp     r12, [r10 + 72]             ; i < T
    jae     .bh_next

    ; row byte offset = i * D_bytes
    mov     rax, r12
    imul    rax, [rbp - 32]
    mov     r13, [rbp - 48]
    add     r13, rax                    ; r13 = do_i
    mov     rax, r12
    imul    rax, [rbp - 40]
    mov     r15, [rbp - 80]
    add     r15, rax                    ; r15 = p_row
    mov     r14, [rbp - 120]            ; r14 = dP scratch

    ; Pass 1: dP[j] = dot(do_i, v_j) and dV_j += p_row[j] * do_i for j = 0..i
    mov     rbx, [rbp - 72]             ; v_j
    mov     r9,  [rbp - 104]            ; dv_j
    xor     r8d, r8d                    ; j = 0
.p1_lp:
    cmp     r8, r12
    ja      .p2_sm
    mov     rdi, r13                    ; do_i
    mov     rsi, rbx                    ; v_j
    mov     rdx, [r10 + 88]             ; hd
    call    dot_8w                      ; xmm0 = dP[j]
    vmovss  [r14 + r8*4], xmm0

    vmovss  xmm0, [r15 + r8*4]          ; p_row[j]
    mov     rdi, r9                     ; dv_j
    mov     rsi, r13                    ; do_i
    mov     rdx, [r10 + 88]             ; hd
    call    saxpy_8w

    add     rbx, [rbp - 32]
    add     r9,  [rbp - 32]
    inc     r8
    jmp     .p1_lp

    ; Pass 2: Softmax backward: dot_p_dp = sum_{j=0..i} P[j]*dP[j],
    ;         dS[j] = P[j] * (dP[j] - dot_p_dp) * scale -> stored in dP[j]
.p2_sm:
    vxorps  xmm2, xmm2, xmm2            ; dot_p_dp
    xor     r8d, r8d
.p2_dot:
    cmp     r8, r12
    ja      .p2_ds
    vmovss  xmm0, [r15 + r8*4]
    vfmadd231ss xmm2, xmm0, [r14 + r8*4]
    inc     r8
    jmp     .p2_dot

.p2_ds:
    vmovss  xmm3, [r10 + 96]            ; scale
    xor     r8d, r8d
.p2_ds_lp:
    cmp     r8, r12
    ja      .p3_qk
    vmovss  xmm0, [r14 + r8*4]          ; dP[j]
    vsubss  xmm0, xmm0, xmm2            ; dP[j] - dot_p_dp
    vmulss  xmm0, xmm0, [r15 + r8*4]    ; * P[j]
    vmulss  xmm0, xmm0, xmm3            ; * scale
    vmovss  [r14 + r8*4], xmm0          ; dS[j]
    inc     r8
    jmp     .p2_ds_lp

    ; Pass 3: dQ_i = sum_j dS[j] * K_j, dK_j += dS[j] * Q_i
.p3_qk:
    mov     rax, r12
    imul    rax, [rbp - 32]
    mov     r13, [rbp - 88]
    add     r13, rax                    ; r13 = dq_i
    mov     r15, [rbp - 56]
    add     r15, rax                    ; r15 = q_i

    ; Zero dq_i
    mov     rcx, [r10 + 88]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.z_dq:
    cmp     rax, rcx
    jae     .p3_start
    vmovups [r13 + rax], ymm0
    add     rax, 32
    jmp     .z_dq

.p3_start:
    mov     rbx, [rbp - 64]             ; k_j
    mov     r9,  [rbp - 96]             ; dk_j
    xor     r8d, r8d                    ; j = 0
.p3_lp:
    cmp     r8, r12
    ja      .i_next
    vmovss  xmm0, [r14 + r8*4]          ; dS[j]
    mov     rdi, r13                    ; dq_i
    mov     rsi, rbx                    ; k_j
    mov     rdx, [r10 + 88]             ; hd
    call    saxpy_8w

    vmovss  xmm0, [r14 + r8*4]          ; dS[j]
    mov     rdi, r9                     ; dk_j
    mov     rsi, r15                    ; q_i
    mov     rdx, [r10 + 88]             ; hd
    call    saxpy_8w

    add     rbx, [rbp - 32]
    add     r9,  [rbp - 32]
    inc     r8
    jmp     .p3_lp

.i_next:
    inc     qword [rbp - 112]
    jmp     .i_lp

.bh_next:
    inc     qword [rbp - 16]
    jmp     .bh_lp

.done:
    vzeroupper
    mov     rsp, rbp
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; 5. SwiGLU FORWARD & BACKWARD (overflow-free, cancellation-free AVX2+FMA)
; ============================================================================

; void tp_swiglu_fwd(float *act, const float *gate, const float *up, uint64 n)
; rdi=act, rsi=gate, rdx=up, rcx=n (multiple of 8)
tp_swiglu_fwd:
    sub     rsp, 40
    mov     [rsp + 0],  rdi
    mov     [rsp + 8],  rsi
    mov     [rsp + 16], rdx
    mov     [rsp + 24], rcx
    lea     rdi, [rel swiglu_fwd_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 40
    ret

swiglu_fwd_worker:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r8,  [rdx + 0]              ; act
    mov     r9,  [rdx + 8]              ; gate
    mov     r10, [rdx + 16]             ; up
    mov     rcx, [rdx + 24]
    shr     rcx, 3                      ; n_blocks = n / 8

    mov     rax, rcx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    shl     rax, 5                      ; off0 = b0 * 32
    mov     r12, rax

    lea     rax, [rdi + 1]
    imul    rax, rcx
    xor     edx, edx
    div     rsi
    shl     rax, 5                      ; off1 = b1 * 32
    mov     r13, rax

    vpbroadcastd ymm8,  [rel ki_abs_mask]
    vpbroadcastd ymm9,  [rel ki_sign_mask]
    vbroadcastss ymm10, [rel kf_one]

.lp:
    cmp     r12, r13
    jae     .done
    vmovups ymm6, [r9  + r12]           ; g = gate
    vmovups ymm7, [r10 + r12]           ; u = up
    vandps  ymm0, ymm6, ymm8            ; |g|
    vxorps  ymm0, ymm0, ymm9            ; -|g|
    call    fast_expf_avx2              ; ymm0 = e = exp(-|g|) (preserves ymm6..ymm15!)
    vblendvps ymm1, ymm10, ymm0, ymm6   ; num = (g < 0) ? e : 1.0
    vaddps  ymm0, ymm0, ymm10           ; 1.0 + e
    vmulps  ymm1, ymm1, ymm6            ; g * num
    vdivps  ymm1, ymm1, ymm0            ; silu(g)
    vmulps  ymm1, ymm1, ymm7            ; silu(g) * u
    vmovups [r8 + r12], ymm1
    add     r12, 32
    jmp     .lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; void tp_swiglu_bwd(float *d_gate, float *d_up, const float *d_act,
;                    const float *gate, const float *up, uint64 n)
; rdi=d_gate, rsi=d_up, rdx=d_act, rcx=gate, r8=up, r9=n (multiple of 8)
tp_swiglu_bwd:
    sub     rsp, 56
    mov     [rsp + 0],  rdi
    mov     [rsp + 8],  rsi
    mov     [rsp + 16], rdx
    mov     [rsp + 24], rcx
    mov     [rsp + 32], r8
    mov     [rsp + 40], r9
    lea     rdi, [rel swiglu_bwd_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 56
    ret

swiglu_bwd_worker:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r8,  [rdx + 0]              ; d_gate
    mov     r9,  [rdx + 8]              ; d_up
    mov     r10, [rdx + 16]             ; d_act
    mov     r14, [rdx + 24]             ; gate
    mov     r15, [rdx + 32]             ; up
    mov     rcx, [rdx + 40]
    shr     rcx, 3                      ; n / 8

    mov     rax, rcx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    shl     rax, 5
    mov     r12, rax                    ; off0

    lea     rax, [rdi + 1]
    imul    rax, rcx
    xor     edx, edx
    div     rsi
    shl     rax, 5
    mov     r13, rax                    ; off1

    vpbroadcastd ymm8,  [rel ki_abs_mask]
    vpbroadcastd ymm9,  [rel ki_sign_mask]
    vbroadcastss ymm10, [rel kf_one]

.lp:
    cmp     r12, r13
    jae     .done
    vmovups ymm6,  [r14 + r12]          ; g
    vmovups ymm7,  [r15 + r12]          ; u
    vmovups ymm11, [r10 + r12]          ; d_act

    vandps  ymm0, ymm6, ymm8            ; |g|
    vxorps  ymm0, ymm0, ymm9            ; -|g|
    call    fast_expf_avx2              ; ymm0 = e = exp(-|g|)

    vaddps  ymm1, ymm0, ymm10           ; 1 + e
    vdivps  ymm1, ymm10, ymm1           ; inv = 1 / (1 + e)

    ; sig = ((g < 0) ? e : 1.0) * inv
    vblendvps ymm2, ymm10, ymm0, ymm6
    vmulps  ymm2, ymm2, ymm1            ; ymm2 = sig

    ; oms = 1 - sig = ((g < 0) ? 1.0 : e) * inv  (cancellation-free!)
    vblendvps ymm3, ymm0, ymm10, ymm6
    vmulps  ymm3, ymm3, ymm1            ; ymm3 = 1 - sig

    ; d_up = d_act * g * sig
    vmulps  ymm4, ymm6, ymm2            ; silu(g) = g * sig
    vmulps  ymm4, ymm4, ymm11           ; d_act * silu(g)
    vmovups [r9 + r12], ymm4

    ; d_gate = d_act * u * sig * (1.0 + g * oms)
    vfmadd213ps ymm3, ymm6, ymm10       ; 1.0 + g * (1 - sig)
    vmulps  ymm3, ymm3, ymm2            ; d_silu = sig * (1 + g*(1-sig))
    vmulps  ymm3, ymm3, ymm7            ; u * d_silu
    vmulps  ymm3, ymm3, ymm11           ; d_act * u * d_silu
    vmovups [r8 + r12], ymm3

    add     r12, 32
    jmp     .lp
.done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; 6. CROSS-ENTROPY LOSS FORWARD + BACKWARD (parallel & deterministic)
; ============================================================================

; float tp_cross_entropy_fwd_bwd(float *logits_inout, float *losses_out,
;                                const uint32 *targets, uint64 N, uint64 V,
;                                uint64 write_dlogits) -> xmm0 (mean loss f32)
; rdi=logits, rsi=losses, rdx=targets, rcx=N, r8=V, r9=write_dlogits
tp_cross_entropy_fwd_bwd:
    push    rbx
    push    r12
    sub     rsp, 56                     ; 48B job + 8B align (2*8+56+8=80)
    mov     [rsp + 0],  rdi             ; logits
    mov     [rsp + 8],  rsi             ; losses
    mov     [rsp + 16], rdx             ; targets
    mov     [rsp + 24], rcx             ; N
    mov     [rsp + 32], r8              ; V
    mov     [rsp + 40], r9              ; write_dlogits

    mov     rbx, rsi                    ; losses_out
    mov     r12, rcx                    ; N

    lea     rdi, [rel ce_worker]
    mov     rsi, rsp
    call    tp_run

    ; Deterministic sequential FP64 mean over losses_out[0..N-1]
    vxorpd  xmm0, xmm0, xmm0
    xor     ecx, ecx
.sum_lp:
    cmp     rcx, r12
    jae     .sum_done
    vcvtss2sd xmm1, xmm1, [rbx + rcx*4]
    vaddsd  xmm0, xmm0, xmm1
    inc     rcx
    jmp     .sum_lp
.sum_done:
    vcvtsi2sd xmm1, xmm1, r12
    vdivsd  xmm0, xmm0, xmm1
    vcvtsd2ss xmm0, xmm0, xmm0

    add     rsp, 56
    pop     r12
    pop     rbx
    ret

ce_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                     ; 6*8+40+8 = 96 (16B aligned)
    ; [rsp + 0]  = job
    ; [rsp + 8]  = V_bytes
    ; [rsp + 16] = inv_N (f32)
    ; [rsp + 24] = target_logit (f32)
    ; [rsp + 28] = target_id (u32)
    mov     [rsp + 0], rdx
    mov     rbp, rdx

    mov     r10, [rbp + 24]             ; N
    mov     r11, [rbp + 32]             ; V

    mov     rax, r10
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; n0

    lea     rax, [rdi + 1]
    imul    rax, r10
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; n1

    mov     rax, r11
    shl     rax, 2
    mov     [rsp + 8], rax              ; V_bytes

    vcvtsi2ss xmm0, xmm0, r10
    vmovss  xmm1, [rel kf_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 16], xmm1            ; inv_N = 1.0f / N

.tok_lp:
    cmp     r12, r13
    jae     .done
    mov     rbp, [rsp + 0]
    mov     r14, [rbp + 32]             ; V
    mov     rax, r12
    imul    rax, [rsp + 8]              ; n * V * 4
    mov     r15, [rbp + 0]
    add     r15, rax                    ; row = &logits[n, 0]

    ; Save target_id and target_logit before overwriting row
    mov     rax, [rbp + 16]
    mov     eax, [rax + r12*4]          ; target_id
    mov     [rsp + 28], eax
    vmovss  xmm0, [r15 + rax*4]
    vmovss  [rsp + 24], xmm0            ; target_logit

    ; Pass 1: max logit -> xmm6 (preserved across fast_expf_avx2!)
    vmovss  xmm6, [r15]
    mov     rcx, 1
.max_lp:
    cmp     rcx, r14
    jae     .max_done
    vmaxss  xmm6, xmm6, [r15 + rcx*4]
    inc     rcx
    jmp     .max_lp
.max_done:
    vbroadcastss ymm6, xmm6
    vxorps  ymm7, ymm7, ymm7            ; sum_exp accumulator
    xor     ecx, ecx
.exp8_lp:
    lea     rax, [rcx + 8]
    cmp     rax, r14
    ja      .exp_red
    vmovups ymm0, [r15 + rcx*4]
    vsubps  ymm0, ymm0, ymm6
    call    fast_expf_avx2
    vmovups [r15 + rcx*4], ymm0
    vaddps  ymm7, ymm7, ymm0
    add     rcx, 8
    jmp     .exp8_lp
.exp_red:
    vextractf128 xmm8, ymm7, 1
    vaddps  xmm7, xmm7, xmm8
    vhaddps xmm7, xmm7, xmm7
    vhaddps xmm7, xmm7, xmm7
.exp1_lp:
    cmp     rcx, r14
    jae     .exp_done
    vmovss  xmm0, [r15 + rcx*4]
    vsubss  xmm0, xmm0, xmm6
    call    fast_expf_scalar
    vmovss  [r15 + rcx*4], xmm0
    vaddss  xmm7, xmm7, xmm0
    inc     rcx
    jmp     .exp1_lp
.exp_done:
    ; loss_n = max_val + log_f64(sum_exp) - target_logit
    ; Save max_val (xmm6) and sum_exp (xmm7) across log_f64 (clobbers xmm0..xmm4, rax, rcx, rdx)
    vcvtss2sd xmm0, xmm0, xmm7
    vmovss  [rsp + 32], xmm6            ; save max
    vmovss  [rsp + 36], xmm7            ; save sum_exp
    call    log_f64                     ; xmm0 = ln(sum_exp) in f64
    vcvtss2sd xmm1, xmm1, [rsp + 32]    ; max in f64
    vaddsd  xmm0, xmm0, xmm1
    vcvtss2sd xmm1, xmm1, [rsp + 24]    ; target_logit in f64
    vsubsd  xmm0, xmm0, xmm1            ; loss_n in f64
    vcvtsd2ss xmm0, xmm0, xmm0
    mov     rbp, [rsp + 0]
    mov     rax, [rbp + 8]
    vmovss  [rax + r12*4], xmm0         ; losses_out[n]

    ; Write d_logits = (p - y) / N if requested
    cmp     qword [rbp + 40], 0
    je      .tok_next

    vmovss  xmm1, [rsp + 16]            ; inv_N
    vdivss  xmm0, xmm1, [rsp + 36]      ; scale = inv_N / sum_exp
    vbroadcastss ymm2, xmm0
    xor     ecx, ecx
.sc8_lp:
    lea     rax, [rcx + 8]
    cmp     rax, r14
    ja      .sc1_lp
    vmovups ymm0, [r15 + rcx*4]
    vmulps  ymm0, ymm0, ymm2
    vmovups [r15 + rcx*4], ymm0
    add     rcx, 8
    jmp     .sc8_lp
.sc1_lp:
    cmp     rcx, r14
    jae     .sub_tgt
    vmovss  xmm0, [r15 + rcx*4]
    vmulss  xmm0, xmm0, xmm2
    vmovss  [r15 + rcx*4], xmm0
    inc     rcx
    jmp     .sc1_lp
.sub_tgt:
    mov     eax, [rsp + 28]             ; target_id
    vmovss  xmm0, [r15 + rax*4]
    vsubss  xmm0, xmm0, [rsp + 16]      ; -= inv_N
    vmovss  [r15 + rax*4], xmm0

.tok_next:
    inc     r12
    jmp     .tok_lp
.done:
    vzeroupper
    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; 7. GLOBAL GRADIENT L2 NORM, SCALE, AND ADAMW OPTIMIZER
; ============================================================================

; float tp_grad_norm(const float *grads, uint64 n_params) -> xmm0 (L2 norm f32)
; Accumulates sum-of-squares in FP64 per thread -> precision ~1e-15,
; immune to float32 reduction drift across thread counts!
tp_grad_norm:
    sub     rsp, 24
    mov     [rsp + 0], rdi              ; grads
    mov     [rsp + 8], rsi              ; n_params
    lea     rdi, [rel grad_norm_worker]
    mov     rsi, rsp
    call    tp_run

    call    tp_get_threads
    mov     rcx, rax
    lea     rdi, [rel gn_partials]
    vxorpd  xmm0, xmm0, xmm0
    xor     eax, eax
.red:
    cmp     rax, rcx
    jae     .sqrt
    mov     rdx, rax
    shl     rdx, 6                      ; tid * 64B cache line
    vaddsd  xmm0, xmm0, [rdi + rdx]
    inc     rax
    jmp     .red
.sqrt:
    vsqrtsd xmm0, xmm0, xmm0
    vcvtsd2ss xmm0, xmm0, xmm0
    add     rsp, 24
    ret

grad_norm_worker:
    mov     r8,  [rdx + 0]              ; grads
    mov     r9,  [rdx + 8]              ; n_params

    mov     rax, r9
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r10, rax                    ; i0

    lea     rax, [rdi + 1]
    imul    rax, r9
    xor     edx, edx
    div     rsi
    mov     r11, rax                    ; i1

    vxorpd  ymm0, ymm0, ymm0            ; 4x f64 sum
    vxorpd  ymm1, ymm1, ymm1            ; 4x f64 sum
.lp8:
    lea     rcx, [r10 + 8]
    cmp     rcx, r11
    ja      .red_v
    vcvtps2pd ymm2, [r8 + r10*4]
    vcvtps2pd ymm3, [r8 + r10*4 + 16]
    vfmadd231pd ymm0, ymm2, ymm2
    vfmadd231pd ymm1, ymm3, ymm3
    add     r10, 8
    jmp     .lp8
.red_v:
    vaddpd  ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
.lp1:
    cmp     r10, r11
    jae     .store
    vcvtss2sd xmm1, xmm1, [r8 + r10*4]
    vfmadd231sd xmm0, xmm1, xmm1
    inc     r10
    jmp     .lp1
.store:
    lea     rax, [rel gn_partials]
    shl     rdi, 6
    vmovsd  [rax + rdi], xmm0
    vzeroupper
    ret

; void tp_vec_scale(float *x, uint64 n, float scale)
; rdi=x, rsi=n, xmm0=scale
tp_vec_scale:
    sub     rsp, 24
    mov     [rsp + 0],  rdi
    mov     [rsp + 8],  rsi
    vmovss  [rsp + 16], xmm0
    lea     rdi, [rel vec_scale_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 24
    ret

vec_scale_worker:
    mov     r8, [rdx + 0]
    mov     r9, [rdx + 8]
    vbroadcastss ymm2, [rdx + 16]

    mov     rax, r9
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r10, rax

    lea     rax, [rdi + 1]
    imul    rax, r9
    xor     edx, edx
    div     rsi
    mov     r11, rax

.lp8:
    lea     rcx, [r10 + 8]
    cmp     rcx, r11
    ja      .lp1
    vmovups ymm0, [r8 + r10*4]
    vmulps  ymm0, ymm0, ymm2
    vmovups [r8 + r10*4], ymm0
    add     r10, 8
    jmp     .lp8
.lp1:
    cmp     r10, r11
    jae     .done
    vmovss  xmm0, [r8 + r10*4]
    vmulss  xmm0, xmm0, xmm2
    vmovss  [r8 + r10*4], xmm0
    inc     r10
    jmp     .lp1
.done:
    vzeroupper
    ret

; void tp_adamw_range(float *p, const float *g, float *m, float *v, uint64 n,
;                     float lr, float wd, float inv_bc1, float inv_bc2)
; rdi=p, rsi=g, rdx=m, rcx=v, r8=n, xmm0=lr, xmm1=wd, xmm2=inv_bc1, xmm3=inv_bc2
tp_adamw_range:
    sub     rsp, 72
    mov     [rsp + 0],  rdi             ; p
    mov     [rsp + 8],  rsi             ; g
    mov     [rsp + 16], rdx             ; m
    mov     [rsp + 24], rcx             ; v
    mov     [rsp + 32], r8              ; n
    ; Precompute decay_mult = 1.0f - lr * wd, step_size = lr * inv_bc1
    vmulss  xmm4, xmm0, xmm1            ; lr * wd
    vmovss  xmm5, [rel kf_one]
    vsubss  xmm5, xmm5, xmm4            ; decay_mult = 1.0 - lr * wd
    vmulss  xmm6, xmm0, xmm2            ; step_size = lr * inv_bc1
    vmovss  [rsp + 40], xmm5            ; decay_mult
    vmovss  [rsp + 44], xmm6            ; step_size
    vmovss  [rsp + 48], xmm3            ; inv_bc2
    lea     rdi, [rel adamw_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 72
    ret

adamw_worker:
    push    rbx
    push    r12
    push    r13

    mov     r8,  [rdx + 0]              ; p
    mov     r9,  [rdx + 8]              ; g
    mov     r10, [rdx + 16]             ; m
    mov     r11, [rdx + 24]             ; v
    mov     rbx, [rdx + 32]             ; n

    vbroadcastss ymm8,  [rdx + 40]      ; decay_mult = 1 - lr*wd
    vbroadcastss ymm9,  [rdx + 44]      ; step_size = lr * inv_bc1
    vbroadcastss ymm10, [rdx + 48]      ; inv_bc2
    vbroadcastss ymm11, [rel kf_beta1]  ; 0.9
    vbroadcastss ymm12, [rel kf_1mb1]   ; 0.1
    vbroadcastss ymm13, [rel kf_beta2]  ; 0.999
    vbroadcastss ymm14, [rel kf_1mb2]   ; 0.001
    vbroadcastss ymm15, [rel kf_adam_eps]; 1e-8

    mov     rax, rbx
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; i0

    lea     rax, [rdi + 1]
    imul    rax, rbx
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; i1

.lp8:
    lea     rcx, [r12 + 8]
    cmp     rcx, r13
    ja      .lp1
    vmovups ymm0, [r8  + r12*4]         ; p
    vmovups ymm1, [r9  + r12*4]         ; g
    vmovups ymm2, [r10 + r12*4]         ; m
    vmovups ymm3, [r11 + r12*4]         ; v

    ; p *= decay_mult
    vmulps  ymm0, ymm0, ymm8
    ; m = beta1 * m + (1 - beta1) * g
    vmulps  ymm2, ymm2, ymm11
    vfmadd231ps ymm2, ymm12, ymm1
    vmovups [r10 + r12*4], ymm2
    ; v = beta2 * v + (1 - beta2) * g * g
    vmulps  ymm4, ymm1, ymm1            ; g^2
    vmulps  ymm3, ymm3, ymm13
    vfmadd231ps ymm3, ymm14, ymm4
    vmovups [r11 + r12*4], ymm3
    ; denom = sqrt(v * inv_bc2) + eps
    vmulps  ymm5, ymm3, ymm10
    vsqrtps ymm5, ymm5
    vaddps  ymm5, ymm5, ymm15
    ; p -= step_size * (m / denom)
    vdivps  ymm6, ymm2, ymm5
    vfnmadd231ps ymm0, ymm9, ymm6
    vmovups [r8 + r12*4], ymm0

    add     r12, 8
    jmp     .lp8

.lp1:
    cmp     r12, r13
    jae     .done
    vmovss  xmm0, [r8  + r12*4]
    vmovss  xmm1, [r9  + r12*4]
    vmovss  xmm2, [r10 + r12*4]
    vmovss  xmm3, [r11 + r12*4]

    vmulss  xmm0, xmm0, xmm8
    vmulss  xmm2, xmm2, xmm11
    vfmadd231ss xmm2, xmm12, xmm1
    vmovss  [r10 + r12*4], xmm2
    vmulss  xmm4, xmm1, xmm1
    vmulss  xmm3, xmm3, xmm13
    vfmadd231ss xmm3, xmm14, xmm4
    vmovss  [r11 + r12*4], xmm3
    vmulss  xmm5, xmm3, xmm10
    vsqrtss xmm5, xmm5, xmm5
    vaddss  xmm5, xmm5, xmm15
    vdivss  xmm6, xmm2, xmm5
    vfnmadd231ss xmm0, xmm9, xmm6
    vmovss  [r8 + r12*4], xmm0

    inc     r12
    jmp     .lp1
.done:
    vzeroupper
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; 8. PRNG (xorshift64*), BOX-MULLER INIT, AND COSINE LR SCHEDULE
; ============================================================================

; uint64 rng_next_u64(uint64 *state) -> rax
rng_next_u64:
    mov     rax, [rdi]
    mov     rdx, rax
    shr     rdx, 12
    xor     rax, rdx
    mov     rdx, rax
    shl     rdx, 25
    xor     rax, rdx
    mov     rdx, rax
    shr     rdx, 27
    xor     rax, rdx
    mov     [rdi], rax
    mov     rdx, 0x2545F4914F6CDD1D
    imul    rax, rdx
    ret

; void init_normal_f32(float *dst, uint64 n, float std, uint64 *rng_state)
; rdi=dst, rsi=n, xmm0=std, rdx=rng_state
init_normal_f32:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 24                     ; 4*8+24+8 = 64 (16B aligned)
    mov     r12, rdi                    ; dst
    mov     r13, rsi                    ; n
    mov     r14, rdx                    ; rng_state
    vcvtss2sd xmm0, xmm0, xmm0
    vmovsd  [rsp + 0], xmm0             ; std (f64)
    xor     ebx, ebx                    ; i = 0
.lp:
    cmp     rbx, r13
    jae     .done

    ; u1 = ((rng() >> 11) + 1) * 2^-53  in (0, 1]
    mov     rdi, r14
    call    rng_next_u64
    shr     rax, 11
    inc     rax
    vcvtsi2sd xmm0, xmm0, rax
    vmulsd  xmm0, xmm0, [rel kd_inv_2p53]
    call    log_f64                     ; ln(u1) <= 0
    vmulsd  xmm0, xmm0, [rel kd_neg2]   ; -2 * ln(u1) >= 0
    vsqrtsd xmm0, xmm0, xmm0            ; r
    vmulsd  xmm0, xmm0, [rsp + 0]       ; r * std
    vmovsd  [rsp + 8], xmm0             ; save r*std

    ; u2 = (rng() >> 11) * 2^-53  in [0, 1)
    mov     rdi, r14
    call    rng_next_u64
    shr     rax, 11
    vcvtsi2sd xmm0, xmm0, rax
    vmulsd  xmm0, xmm0, [rel kd_inv_2p53]
    vmulsd  xmm0, xmm0, [rel kd_2pi]    ; angle = 2*pi*u2
    call    fast_sincos_f64             ; xmm0=sin, xmm1=cos

    vmovsd  xmm2, [rsp + 8]             ; r * std
    vmulsd  xmm1, xmm1, xmm2            ; z0 = r * std * cos
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [r12 + rbx*4], xmm1
    inc     rbx
    cmp     rbx, r13
    jae     .done

    vmulsd  xmm0, xmm0, xmm2            ; z1 = r * std * sin
    vcvtsd2ss xmm0, xmm0, xmm0
    vmovss  [r12 + rbx*4], xmm0
    inc     rbx
    jmp     .lp
.done:
    add     rsp, 24
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; void init_ones_f32(float *dst, uint64 n)
init_ones_f32:
    vbroadcastss ymm0, [rel kf_one]
    xor     eax, eax
.lp8:
    lea     rcx, [rax + 8]
    cmp     rcx, rsi
    ja      .lp1
    vmovups [rdi + rax*4], ymm0
    add     rax, 8
    jmp     .lp8
.lp1:
    cmp     rax, rsi
    jae     .done
    vmovss  [rdi + rax*4], xmm0
    inc     rax
    jmp     .lp1
.done:
    vzeroupper
    ret

; float compute_lr(uint64 step, uint64 total_steps, uint64 warmup_steps, float max_lr)
; rdi=step, rsi=total_steps, rdx=warmup_steps, xmm0=max_lr -> xmm0=lr
compute_lr:
    sub     rsp, 24                     ; 24 + 8 = 32 (16B aligned)
    vcvtss2sd xmm0, xmm0, xmm0
    vmovsd  [rsp + 0], xmm0             ; max_lr (f64)

    ; Linear warmup: if warmup_steps > 0 && step < warmup_steps:
    ;   lr = max_lr * (step + 1) / warmup_steps
    test    rdx, rdx
    jz      .cosine
    cmp     rdi, rdx
    jae     .cosine
    lea     rax, [rdi + 1]
    vcvtsi2sd xmm1, xmm1, rax
    vcvtsi2sd xmm2, xmm2, rdx
    vdivsd  xmm1, xmm1, xmm2
    vmulsd  xmm0, xmm0, xmm1
    vcvtsd2ss xmm0, xmm0, xmm0
    add     rsp, 24
    ret

.cosine:
    ; If total_steps <= warmup_steps || total_steps <= 1: return max_lr
    cmp     rsi, rdx
    jbe     .ret_max
    cmp     rsi, 1
    jbe     .ret_max

    ; progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    mov     rax, rdi
    sub     rax, rdx                    ; step - warmup
    mov     rcx, rsi
    sub     rcx, rdx                    ; decay_steps = total - warmup
    vcvtsi2sd xmm0, xmm0, rax
    vcvtsi2sd xmm1, xmm1, rcx
    vdivsd  xmm0, xmm0, xmm1
    vminsd  xmm0, xmm0, [rel kd_one]
    vmulsd  xmm0, xmm0, [rel kd_pi]     ; pi * progress
    call    fast_sincos_f64             ; xmm1 = cos(pi * progress)

    ; coeff = 0.5 * (1.0 + cos)
    vaddsd  xmm1, xmm1, [rel kd_one]
    vmulsd  xmm1, xmm1, [rel kd_half]
    ; lr = min_lr + (max_lr - min_lr) * coeff  (with min_lr = 0.1 * max_lr)
    ;    = max_lr * (0.1 + 0.9 * coeff)
    vmulsd  xmm1, xmm1, [rel kd_0p9]
    vaddsd  xmm1, xmm1, [rel kd_0p1]
    vmulsd  xmm0, xmm1, [rsp + 0]
    vcvtsd2ss xmm0, xmm0, xmm0
    add     rsp, 24
    ret

.ret_max:
    vcvtsd2ss xmm0, xmm0, [rsp + 0]
    add     rsp, 24
    ret

; ============================================================================
; RODATA & BSS
; ============================================================================
section .rodata
align 16
kf_one:       dd 1.0
kf_beta1:     dd 0.9
kf_1mb1:      dd 0.1
kf_beta2:     dd 0.999
kf_1mb2:      dd 0.001
kf_adam_eps:  dd 1.0e-8
ki_abs_mask:  dd 0x7FFFFFFF
ki_sign_mask: dd 0x80000000

align 16
kd_one:       dq 1.0
kd_half:      dq 0.5
kd_0p1:       dq 0.1
kd_0p9:       dq 0.9
kd_neg2:      dq -2.0
kd_pi:        dq 3.14159265358979323846
kd_2pi:       dq 6.28318530717958647692
kd_inv_2p53:  dq 1.1102230246251565e-16

section .bss
alignb 64
gn_partials:  resb MAX_THREADS * 64
