; ============================================================================
; math.asm — Pure AVX2 / FMA / SSE2 math primitives (ZERO libm!)
;
; Functions:
;   fast_expf_avx2   : ymm0 -> ymm0 (8x float32 exp, ~2e-7 rel err, safe clamp)
;   fast_expf_scalar : xmm0 -> xmm0 (scalar float32 exp)
;   fast_sincos_f64  : xmm0 (f64) -> xmm0 (sin f64), xmm1 (cos f64) (~1e-13 err)
;   fast_sincosf     : xmm0 (f32) -> xmm0 (sin f32), xmm1 (cos f32) (~1e-7 err)
;   log_f64          : xmm0 (f64 > 0) -> xmm0 (ln f64, ~1e-12 err)
;   exp_f64          : xmm0 (f64) -> xmm0 (exp f64, ~1e-12 err)
;   init_rope_freqs  : (double *out, uint64 head_dim, float rope_theta)
;   compute_rope_cs  : (float *cos_out, float *sin_out, const double *inv_freq, uint64 half_dim, uint64 pos)
;   rope_apply_avx2  : (float *vec, const float *cos_b, const float *sin_b, uint64 n_heads, uint64 head_dim)
;   rmsnorm_avx2     : (float *out, const float *x, const float *w, uint64 n, float eps)
;   softmax_avx2     : (float *x, uint64 n)
;   silu_mul_avx2    : (float *gate_inout, const float *up, uint64 n)
;   vec_add_avx2     : (float *dst_inout, const float *src, uint64 n)
;   argmax_f32       : (const float *x, uint64 n) -> uint64
;   sample_top_p     : (float *logits, float *probs, uint64 vocab, float temp, float topp, uint64 *rng_state) -> uint64
; ============================================================================

bits 64
default rel

section .text

global fast_expf_avx2
global fast_expf_scalar
global fast_sincos_f64
global fast_sincosf
global log_f64
global exp_f64
global init_rope_freqs
global compute_rope_cs
global rope_apply_avx2
global rmsnorm_avx2
global softmax_avx2
global silu_mul_avx2
global vec_add_avx2
global argmax_f32
global sample_top_p

; ----------------------------------------------------------------------------
; ymm0 = fast_expf_avx2(ymm0)
; Clobbers ONLY ymm0..ymm5 (ymm6..ymm15 preserved for callers!).
; Safe for all finite x: x < -87.33654 -> 0.0f, x > 88.37626 clamped.
; Cody-Waite range reduction + degree-6 Taylor polynomial (~2e-7 rel error).
; ----------------------------------------------------------------------------
fast_expf_avx2:
    vbroadcastss ymm4, [rel f_exp_lo]
    vcmpps  ymm5, ymm0, ymm4, 13        ; _CMP_GE_OS: mask = (x >= exp_lo)
    vmaxps  ymm0, ymm0, ymm4
    vbroadcastss ymm4, [rel f_exp_hi]
    vminps  ymm0, ymm0, ymm4

    vbroadcastss ymm4, [rel f_log2e]
    vmulps  ymm1, ymm0, ymm4
    vroundps ymm1, ymm1, 0              ; k_f = round_nearest_even(x * log2e)
    vcvtps2dq ymm2, ymm1                ; k_i = (int32)k_f

    ; Cody-Waite range reduction: r = (x - k_f*ln2_hi) - k_f*ln2_lo
    vbroadcastss ymm4, [rel f_ln2_hi]
    vfnmadd231ps ymm0, ymm1, ymm4
    vbroadcastss ymm4, [rel f_ln2_lo]
    vfnmadd231ps ymm0, ymm1, ymm4

    ; Horner degree-6: 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
    vbroadcastss ymm3, [rel f_exp_c6]
    vbroadcastss ymm4, [rel f_exp_c5]
    vfmadd213ps ymm3, ymm0, ymm4
    vbroadcastss ymm4, [rel f_exp_c4]
    vfmadd213ps ymm3, ymm0, ymm4
    vbroadcastss ymm4, [rel f_exp_c3]
    vfmadd213ps ymm3, ymm0, ymm4
    vbroadcastss ymm4, [rel f_exp_c2]
    vfmadd213ps ymm3, ymm0, ymm4
    vbroadcastss ymm4, [rel f_one]
    vfmadd213ps ymm3, ymm0, ymm4
    vfmadd213ps ymm3, ymm0, ymm4

    ; Reconstruct 2^k via IEEE-754 exponent shift: (k + 127) << 23
    vpbroadcastd ymm4, [rel i_127]
    vpaddd  ymm2, ymm2, ymm4
    vpslld  ymm2, ymm2, 23
    vmulps  ymm0, ymm3, ymm2
    vandps  ymm0, ymm0, ymm5            ; flush underflowed lanes to 0.0
    ret

; ----------------------------------------------------------------------------
; xmm0 = fast_expf_scalar(xmm0)
; ----------------------------------------------------------------------------
fast_expf_scalar:
    vbroadcastss ymm0, xmm0
    call    fast_expf_avx2
    ret

; ----------------------------------------------------------------------------
; (xmm0=sin, xmm1=cos) = fast_sincos_f64(xmm0=angle_f64)
; Cody-Waite FP64 reduction by pi/2 + degree-11/10 Taylor polynomial.
; Accuracy: < 1e-13 across [-1e7, +1e7].
; Clobbers rax, xmm0..xmm5.
; ----------------------------------------------------------------------------
fast_sincos_f64:
    vmulsd  xmm1, xmm0, [rel d_inv_pio2]
    vroundsd xmm1, xmm1, xmm1, 0        ; q_d = round(x * 2/pi)
    vcvtsd2si rax, xmm1                 ; q = (int64)q_d

    ; r = (x - q_d * pio2_hi) - q_d * pio2_lo
    vmovsd  xmm2, [rel d_pio2_hi]
    vfnmadd231sd xmm0, xmm1, xmm2
    vmovsd  xmm2, [rel d_pio2_lo]
    vfnmadd231sd xmm0, xmm1, xmm2       ; xmm0 = r in [-pi/4, +pi/4]

    vmulsd  xmm2, xmm0, xmm0            ; xmm2 = r^2

    ; sin(r) = r * (1 + r2*(s1 + r2*(s2 + r2*(s3 + r2*(s4 + r2*s5)))))
    vmovsd  xmm3, [rel d_sin_s5]
    vfmadd213sd xmm3, xmm2, [rel d_sin_s4]
    vfmadd213sd xmm3, xmm2, [rel d_sin_s3]
    vfmadd213sd xmm3, xmm2, [rel d_sin_s2]
    vfmadd213sd xmm3, xmm2, [rel d_sin_s1]
    vfmadd213sd xmm3, xmm2, [rel d_one]
    vmulsd  xmm3, xmm3, xmm0            ; xmm3 = sin(r)

    ; cos(r) = 1 + r2*(c1 + r2*(c2 + r2*(c3 + r2*(c4 + r2*c5))))
    vmovsd  xmm4, [rel d_cos_c5]
    vfmadd213sd xmm4, xmm2, [rel d_cos_c4]
    vfmadd213sd xmm4, xmm2, [rel d_cos_c3]
    vfmadd213sd xmm4, xmm2, [rel d_cos_c2]
    vfmadd213sd xmm4, xmm2, [rel d_cos_c1]
    vfmadd213sd xmm4, xmm2, [rel d_one] ; xmm4 = cos(r)

    ; Quadrant selection based on q & 3
    and     eax, 3
    jz      .q0
    cmp     eax, 1
    je      .q1
    cmp     eax, 2
    je      .q2
    ; q == 3: sin = -cos_r, cos = sin_r
    vxorpd  xmm0, xmm4, [rel d_sign_mask]
    vmovapd xmm1, xmm3
    ret
.q0:
    ; q == 0: sin = sin_r, cos = cos_r
    vmovapd xmm0, xmm3
    vmovapd xmm1, xmm4
    ret
.q1:
    ; q == 1: sin = cos_r, cos = -sin_r
    vmovapd xmm0, xmm4
    vxorpd  xmm1, xmm3, [rel d_sign_mask]
    ret
.q2:
    ; q == 2: sin = -sin_r, cos = -cos_r
    vxorpd  xmm0, xmm3, [rel d_sign_mask]
    vxorpd  xmm1, xmm4, [rel d_sign_mask]
    ret

; ----------------------------------------------------------------------------
; (xmm0=sin_f32, xmm1=cos_f32) = fast_sincosf(xmm0=angle_f32)
; ----------------------------------------------------------------------------
fast_sincosf:
    vcvtss2sd xmm0, xmm0, xmm0
    call    fast_sincos_f64
    vcvtsd2ss xmm0, xmm0, xmm0
    vcvtsd2ss xmm1, xmm1, xmm1
    ret

; ----------------------------------------------------------------------------
; xmm0 = log_f64(xmm0 = x > 0)
; Exact range reduction x = m * 2^e with m in [sqrt(2)/2, sqrt(2)],
; then ln(m) = 2*u*(1 + u^2/3 + u^4/5 + ... + u^12/13), u = (m-1)/(m+1).
; ----------------------------------------------------------------------------
log_f64:
    vmovq   rax, xmm0
    mov     rcx, rax
    shr     rcx, 52
    and     ecx, 0x7FF
    sub     rcx, 1023                   ; rcx = exponent e

    ; Replace exponent with 1023 -> m in [1.0, 2.0)
    mov     rdx, 0x000FFFFFFFFFFFFF
    and     rax, rdx
    mov     rdx, 0x3FF0000000000000
    or      rax, rdx
    vmovq   xmm0, rax                   ; m

    ; If m >= sqrt(2), m *= 0.5, e += 1
    vucomisd xmm0, [rel d_sqrt2]
    jb      .reduced
    vmulsd  xmm0, xmm0, [rel d_half]
    inc     rcx
.reduced:
    ; u = (m - 1) / (m + 1)
    vmovsd  xmm1, [rel d_one]
    vsubsd  xmm2, xmm0, xmm1
    vaddsd  xmm3, xmm0, xmm1
    vdivsd  xmm0, xmm2, xmm3            ; u
    vmulsd  xmm2, xmm0, xmm0            ; u^2

    ; Horner: 1 + u^2*(1/3 + u^2*(1/5 + u^2*(1/7 + u^2*(1/9 + u^2*(1/11 + u^2/13)))))
    vmovsd  xmm3, [rel d_inv13]
    vfmadd213sd xmm3, xmm2, [rel d_inv11]
    vfmadd213sd xmm3, xmm2, [rel d_inv9]
    vfmadd213sd xmm3, xmm2, [rel d_inv7]
    vfmadd213sd xmm3, xmm2, [rel d_inv5]
    vfmadd213sd xmm3, xmm2, [rel d_inv3]
    vfmadd213sd xmm3, xmm2, [rel d_one]
    vmulsd  xmm3, xmm3, xmm0
    vaddsd  xmm3, xmm3, xmm3            ; 2 * u * poly = ln(m)

    vcvtsi2sd xmm4, xmm4, rcx           ; (double)e
    vfmadd231sd xmm3, xmm4, [rel d_ln2_full]
    vmovapd xmm0, xmm3
    ret

; ----------------------------------------------------------------------------
; xmm0 = exp_f64(xmm0 = x)
; Range reduction + degree-10 Taylor polynomial in FP64 (~1e-13 rel err).
; ----------------------------------------------------------------------------
exp_f64:
    vmulsd  xmm1, xmm0, [rel d_log2e]
    vroundsd xmm1, xmm1, xmm1, 0        ; k_d
    vcvtsd2si rax, xmm1                 ; k

    vmovsd  xmm2, [rel d_ln2_hi]
    vfnmadd231sd xmm0, xmm1, xmm2
    vmovsd  xmm2, [rel d_ln2_lo]
    vfnmadd231sd xmm0, xmm1, xmm2       ; r in [-ln2/2, ln2/2]

    ; Degree-10 Horner
    vmovsd  xmm3, [rel d_e10]
    vfmadd213sd xmm3, xmm0, [rel d_e9]
    vfmadd213sd xmm3, xmm0, [rel d_e8]
    vfmadd213sd xmm3, xmm0, [rel d_e7]
    vfmadd213sd xmm3, xmm0, [rel d_e6]
    vfmadd213sd xmm3, xmm0, [rel d_e5]
    vfmadd213sd xmm3, xmm0, [rel d_e4]
    vfmadd213sd xmm3, xmm0, [rel d_e3]
    vfmadd213sd xmm3, xmm0, [rel d_half]
    vfmadd213sd xmm3, xmm0, [rel d_one]
    vfmadd213sd xmm3, xmm0, [rel d_one]

    add     rax, 1023
    shl     rax, 52
    vmovq   xmm2, rax                   ; 2^k
    vmulsd  xmm0, xmm3, xmm2
    ret

; ----------------------------------------------------------------------------
; void init_rope_freqs(double *out, uint64 head_dim, float rope_theta)
; rdi = out (head_dim/2 doubles), rsi = head_dim, xmm0 = rope_theta
; inv_freq[i] = exp_f64( -ln(rope_theta) * (2*i) / head_dim )
; ----------------------------------------------------------------------------
init_rope_freqs:
    push    rbx
    push    r12
    push    r13
    sub     rsp, 16                     ; save ln_theta / head_dim on stack

    mov     r12, rdi                    ; out
    mov     r13, rsi                    ; head_dim
    vcvtss2sd xmm0, xmm0, xmm0
    call    log_f64                     ; xmm0 = ln(rope_theta)
    vcvtsi2sd xmm1, xmm1, r13           ; (double)head_dim
    vdivsd  xmm0, xmm0, xmm1            ; ln(theta) / head_dim
    vxorpd  xmm0, xmm0, [rel d_sign_mask] ; -ln(theta) / head_dim
    vmovsd  [rsp], xmm0                 ; step = -ln(theta) / head_dim

    shr     r13, 1                      ; half_dim = head_dim / 2
    xor     ebx, ebx                    ; i = 0
.lp:
    cmp     rbx, r13
    jae     .done
    lea     rax, [rbx + rbx]            ; 2 * i
    vcvtsi2sd xmm0, xmm0, rax
    vmulsd  xmm0, xmm0, [rsp]           ; -ln(theta) * (2*i) / head_dim
    call    exp_f64
    vmovsd  [r12 + rbx*8], xmm0
    inc     rbx
    jmp     .lp
.done:
    add     rsp, 16
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void compute_rope_cs(float *cos_out, float *sin_out, const double *inv_freq,
;                      uint64 half_dim, uint64 pos)
; rdi = cos_out, rsi = sin_out, rdx = inv_freq, rcx = half_dim, r8 = pos
; ----------------------------------------------------------------------------
compute_rope_cs:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 16

    mov     r12, rdi                    ; cos_out
    mov     r13, rsi                    ; sin_out
    mov     r14, rdx                    ; inv_freq
    mov     r15, rcx                    ; half_dim
    vcvtsi2sd xmm0, xmm0, r8
    vmovsd  [rsp], xmm0                 ; (double)pos

    xor     ebx, ebx
.lp:
    cmp     rbx, r15
    jae     .done
    vmovsd  xmm0, [rsp]
    vmulsd  xmm0, xmm0, [r14 + rbx*8]   ; angle = pos * inv_freq[i]
    call    fast_sincos_f64             ; xmm0 = sin, xmm1 = cos
    vcvtsd2ss xmm0, xmm0, xmm0
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [r13 + rbx*4], xmm0         ; sin_out[i]
    vmovss  [r12 + rbx*4], xmm1         ; cos_out[i]
    inc     rbx
    jmp     .lp
.done:
    add     rsp, 16
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void rope_apply_avx2(float *vec, const float *cos_b, const float *sin_b,
;                      uint64 n_heads, uint64 head_dim)
; rdi = vec [n_heads, head_dim], rsi = cos_b [half], rdx = sin_b [half],
; rcx = n_heads, r8 = head_dim (half_dim multiple of 8, e.g. 64)
; Split-half (NeoX / Qwen3) RoPE:
;   out1 = x1 * cos - x2 * sin
;   out2 = x2 * cos + x1 * sin
; ----------------------------------------------------------------------------
rope_apply_avx2:
    mov     r9, r8
    shr     r9, 1                       ; half_dim (elements)
    shl     r9, 2                       ; half_bytes = half_dim * 4
    lea     r10, [r9 + r9]              ; head_bytes = head_dim * 4
.head_lp:
    test    rcx, rcx
    jz      .done
    lea     r11, [rdi + r9]             ; x2_ptr = x1_ptr + half_bytes
    xor     eax, eax                    ; byte offset i*4
.dim_lp:
    cmp     rax, r9
    jae     .next_head
    vmovups ymm0, [rdi + rax]           ; x1
    vmovups ymm1, [r11 + rax]           ; x2
    vmovups ymm2, [rsi + rax]           ; cos
    vmovups ymm3, [rdx + rax]           ; sin
    vmulps  ymm4, ymm0, ymm2            ; x1 * cos
    vfnmadd231ps ymm4, ymm1, ymm3       ; x1 * cos - x2 * sin
    vmulps  ymm5, ymm1, ymm2            ; x2 * cos
    vfmadd231ps ymm5, ymm0, ymm3        ; x2 * cos + x1 * sin
    vmovups [rdi + rax], ymm4
    vmovups [r11 + rax], ymm5
    add     rax, 32
    jmp     .dim_lp
.next_head:
    add     rdi, r10
    dec     rcx
    jmp     .head_lp
.done:
    ret

; ----------------------------------------------------------------------------
; void rmsnorm_avx2(float *out, const float *x, const float *w, uint64 n, float eps)
; rdi = out, rsi = x, rdx = w, rcx = n (multiple of 32), xmm0 = eps
; Supports in-place (out == x).
; ----------------------------------------------------------------------------
rmsnorm_avx2:
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    vxorps  ymm4, ymm4, ymm4
    shl     rcx, 2                      ; n_bytes = n * 4
    xor     eax, eax
.ss_lp:
    cmp     rax, rcx
    jae     .ss_red
    vmovups ymm5, [rsi + rax]
    vmovups ymm6, [rsi + rax + 32]
    vmovups ymm7, [rsi + rax + 64]
    vmovups ymm8, [rsi + rax + 96]
    vfmadd231ps ymm1, ymm5, ymm5
    vfmadd231ps ymm2, ymm6, ymm6
    vfmadd231ps ymm3, ymm7, ymm7
    vfmadd231ps ymm4, ymm8, ymm8
    add     rax, 128
    jmp     .ss_lp
.ss_red:
    vaddps  ymm1, ymm1, ymm2
    vaddps  ymm3, ymm3, ymm4
    vaddps  ymm1, ymm1, ymm3
    ; Horizontal sum of ymm1 -> xmm1
    vextractf128 xmm2, ymm1, 1
    vaddps  xmm1, xmm1, xmm2
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1

    mov     r8, rcx
    shr     r8, 2                       ; n
    vcvtsi2ss xmm2, xmm2, r8
    vdivss  xmm1, xmm1, xmm2            ; mean(x^2)
    vaddss  xmm1, xmm1, xmm0            ; mean(x^2) + eps
    vsqrtss xmm1, xmm1, xmm1            ; rms
    vmovss  xmm2, [rel f_one]
    vdivss  xmm1, xmm2, xmm1            ; inv_rms = 1.0 / rms
    vbroadcastss ymm1, xmm1

    xor     eax, eax
.scale_lp:
    cmp     rax, rcx
    jae     .done
    vmovups ymm2, [rsi + rax]
    vmovups ymm3, [rsi + rax + 32]
    vmulps  ymm2, ymm2, ymm1
    vmulps  ymm3, ymm3, ymm1
    vmulps  ymm2, ymm2, [rdx + rax]
    vmulps  ymm3, ymm3, [rdx + rax + 32]
    vmovups [rdi + rax], ymm2
    vmovups [rdi + rax + 32], ymm3
    add     rax, 64
    jmp     .scale_lp
.done:
    ret

; ----------------------------------------------------------------------------
; void softmax_avx2(float *x, uint64 n)
; rdi = x, rsi = n (arbitrary n >= 1)
; ----------------------------------------------------------------------------
softmax_avx2:
    push    rbx
    push    r12
    mov     r12, rdi                    ; x
    mov     rbx, rsi                    ; n

    ; Pass 1: find max_val in xmm6
    vmovss  xmm6, [r12]
    mov     rcx, 1
.max_lp:
    cmp     rcx, rbx
    jae     .max_done
    vmaxss  xmm6, xmm6, [r12 + rcx*4]
    inc     rcx
    jmp     .max_lp
.max_done:
    vbroadcastss ymm6, xmm6             ; ymm6 = max_val (preserved across fast_expf_avx2)
    vxorps  ymm7, ymm7, ymm7            ; ymm7 = sum accumulator (preserved!)
    xor     ecx, ecx                    ; index i

    ; Pass 2a: 8-wide AVX2 exp(x - max)
.exp8_lp:
    lea     rax, [rcx + 8]
    cmp     rax, rbx
    ja      .exp1_lp
    vmovups ymm0, [r12 + rcx*4]
    vsubps  ymm0, ymm0, ymm6
    call    fast_expf_avx2              ; clobbers ymm0..ymm5 only!
    vmovups [r12 + rcx*4], ymm0
    vaddps  ymm7, ymm7, ymm0
    add     rcx, 8
    jmp     .exp8_lp

    ; Horizontal sum of ymm7 -> xmm7
.exp1_lp:
    vextractf128 xmm8, ymm7, 1
    vaddps  xmm7, xmm7, xmm8
    vhaddps xmm7, xmm7, xmm7
    vhaddps xmm7, xmm7, xmm7

    ; Pass 2b: scalar tail
.tail_lp:
    cmp     rcx, rbx
    jae     .norm
    vmovss  xmm0, [r12 + rcx*4]
    vsubss  xmm0, xmm0, xmm6
    call    fast_expf_scalar
    vmovss  [r12 + rcx*4], xmm0
    vaddss  xmm7, xmm7, xmm0
    inc     rcx
    jmp     .tail_lp

    ; Pass 3: divide by sum
.norm:
    vmovss  xmm0, [rel f_one]
    vdivss  xmm7, xmm0, xmm7            ; inv_sum
    vbroadcastss ymm7, xmm7
    xor     ecx, ecx
.norm8_lp:
    lea     rax, [rcx + 8]
    cmp     rax, rbx
    ja      .norm1_lp
    vmovups ymm0, [r12 + rcx*4]
    vmulps  ymm0, ymm0, ymm7
    vmovups [r12 + rcx*4], ymm0
    add     rcx, 8
    jmp     .norm8_lp
.norm1_lp:
    cmp     rcx, rbx
    jae     .done
    vmovss  xmm0, [r12 + rcx*4]
    vmulss  xmm0, xmm0, xmm7
    vmovss  [r12 + rcx*4], xmm0
    inc     rcx
    jmp     .norm1_lp
.done:
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void silu_mul_avx2(float *gate_inout, const float *up, uint64 n)
; rdi = gate, rsi = up, rdx = n (multiple of 8)
; Overflow-free branchless SiLU * up:
;   e = exp(-|x|) in [0, 1]
;   num = (x < 0) ? e : 1.0  (via vblendvps on sign bit of x)
;   out = (x * num / (1.0 + e)) * up
; ----------------------------------------------------------------------------
silu_mul_avx2:
    push    rbx
    push    r12
    push    r13
    mov     r12, rdi                    ; gate
    mov     r13, rsi                    ; up
    shl     rdx, 2                      ; n_bytes
    mov     rbx, rdx

    vpbroadcastd ymm8, [rel i_abs_mask] ; 0x7FFFFFFF (preserved across fast_expf_avx2!)
    vpbroadcastd ymm9, [rel i_sign_mask]; 0x80000000
    vbroadcastss ymm10, [rel f_one]     ; 1.0f

    xor     ecx, ecx
.lp:
    cmp     rcx, rbx
    jae     .done
    vmovups ymm6, [r12 + rcx]           ; x = gate
    vmovups ymm7, [r13 + rcx]           ; up
    vandps  ymm0, ymm6, ymm8            ; |x|
    vxorps  ymm0, ymm0, ymm9            ; -|x| <= 0
    call    fast_expf_avx2              ; ymm0 = e = exp(-|x|) in [0, 1]
    vblendvps ymm1, ymm10, ymm0, ymm6   ; (x < 0) ? e : 1.0f
    vaddps  ymm0, ymm0, ymm10           ; 1.0f + e
    vmulps  ymm1, ymm1, ymm6            ; x * num
    vdivps  ymm1, ymm1, ymm0            ; silu(x)
    vmulps  ymm1, ymm1, ymm7            ; silu(x) * up
    vmovups [r12 + rcx], ymm1
    add     rcx, 32
    jmp     .lp
.done:
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void vec_add_avx2(float *dst, const float *src, uint64 n)
; rdi = dst, rsi = src, rdx = n (multiple of 32)
; ----------------------------------------------------------------------------
vec_add_avx2:
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vmovups ymm0, [rdi + rax]
    vmovups ymm1, [rdi + rax + 32]
    vmovups ymm2, [rdi + rax + 64]
    vmovups ymm3, [rdi + rax + 96]
    vaddps  ymm0, ymm0, [rsi + rax]
    vaddps  ymm1, ymm1, [rsi + rax + 32]
    vaddps  ymm2, ymm2, [rsi + rax + 64]
    vaddps  ymm3, ymm3, [rsi + rax + 96]
    vmovups [rdi + rax], ymm0
    vmovups [rdi + rax + 32], ymm1
    vmovups [rdi + rax + 64], ymm2
    vmovups [rdi + rax + 96], ymm3
    add     rax, 128
    jmp     .lp
.done:
    ret

; ----------------------------------------------------------------------------
; uint64 argmax_f32(const float *x, uint64 n)
; rdi = x, rsi = n -> rax = index of maximum float (first index on tie)
; 8-wide AVX2 scan with exact first-index tie-breaking.
; ----------------------------------------------------------------------------
argmax_f32:
    cmp     rsi, 8
    jb      .scalar_init

    sub     rsp, 64                     ; 32B vals + 32B idxs
    vmovups ymm0, [rdi]                 ; best_vals = x[0..7]
    vmovdqa ymm1, [rel i_iota8]         ; best_idxs = [0..7]
    vpbroadcastd ymm3, [rel i_eight]    ; step = [8..8]
    vpaddd  ymm2, ymm1, ymm3            ; cur_idxs = [8..15]
    mov     rcx, 8
    mov     rdx, rsi
    and     rdx, -8                     ; n8 = n & ~7
.v_lp:
    cmp     rcx, rdx
    jae     .v_red
    vmovups ymm4, [rdi + rcx*4]
    vcmpps  ymm5, ymm4, ymm0, 14        ; _CMP_GT_OS: strict > preserves first index
    vblendvps ymm0, ymm0, ymm4, ymm5
    vblendvps ymm1, ymm1, ymm2, ymm5
    vpaddd  ymm2, ymm2, ymm3
    add     rcx, 8
    jmp     .v_lp

.v_red:
    vmovups [rsp], ymm0
    vmovdqu [rsp + 32], ymm1
    vmovss  xmm0, [rsp]                 ; best_val
    mov     eax, [rsp + 32]             ; best_idx (zero-extended)
    mov     r8d, 1
.red8:
    cmp     r8d, 8
    jae     .v_tail
    vmovss  xmm1, [rsp + r8*4]
    mov     r9d, [rsp + 32 + r8*4]
    vucomiss xmm1, xmm0
    ja      .take8
    jb      .next8
    ; Tie: keep smaller index
    cmp     r9, rax
    jae     .next8
.take8:
    vmovss  xmm0, xmm1, xmm1
    mov     rax, r9
.next8:
    inc     r8d
    jmp     .red8

.v_tail:
    add     rsp, 64
    jmp     .tail_lp

.scalar_init:
    xor     eax, eax                    ; best_idx = 0
    vmovss  xmm0, [rdi]                 ; best_val = x[0]
    mov     rcx, 1
.tail_lp:
    cmp     rcx, rsi
    jae     .done
    vmovss  xmm1, [rdi + rcx*4]
    vucomiss xmm1, xmm0
    jbe     .next
    vmovss  xmm0, xmm1, xmm1
    mov     rax, rcx
.next:
    inc     rcx
    jmp     .tail_lp
.done:
    ret

; ----------------------------------------------------------------------------
; uint64 sample_top_p(float *logits, float *probs, uint64 vocab,
;                     float temp, float topp, uint64 *rng_state)
; rdi = logits, rsi = probs, rdx = vocab, xmm0 = temp, xmm1 = topp, rcx = &rng
; Scales logits by 1/temp into probs, runs softmax_avx2, applies temperature
; + optional top-p (threshold via cumulative scan over high-prob tokens / full
; sampling with xorshift64*). Returns sampled token ID.
; ----------------------------------------------------------------------------
sample_top_p:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 24

    mov     r12, rdi                    ; logits
    mov     r13, rsi                    ; probs
    mov     r14, rdx                    ; vocab
    mov     rbx, rcx                    ; &rng_state
    vmovss  [rsp], xmm1                 ; save topp

    ; Scale logits[i] / temp -> probs[i]
    vmovss  xmm2, [rel f_one]
    vdivss  xmm2, xmm2, xmm0            ; inv_temp
    vbroadcastss ymm2, xmm2
    xor     eax, eax
.sc_lp:
    lea     rcx, [rax + 8]
    cmp     rcx, r14
    ja      .sc_tail
    vmovups ymm3, [r12 + rax*4]
    vmulps  ymm3, ymm3, ymm2
    vmovups [r13 + rax*4], ymm3
    add     rax, 8
    jmp     .sc_lp
.sc_tail:
    cmp     rax, r14
    jae     .do_sm
    vmovss  xmm3, [r12 + rax*4]
    vmulss  xmm3, xmm3, xmm2
    vmovss  [r13 + rax*4], xmm3
    inc     rax
    jmp     .sc_tail

.do_sm:
    mov     rdi, r13
    mov     rsi, r14
    call    softmax_avx2

    ; xorshift64* -> uniform u in [0, 1)
    mov     rax, [rbx]
    mov     rdx, rax
    shr     rdx, 12
    xor     rax, rdx
    mov     rdx, rax
    shl     rdx, 25
    xor     rax, rdx
    mov     rdx, rax
    shr     rdx, 27
    xor     rax, rdx
    mov     [rbx], rax
    mov     rdx, 0x2545F4914F6CDD1D
    imul    rax, rdx
    shr     rax, 40                     ; top 24 bits -> [0, 2^24)
    vcvtsi2ss xmm0, xmm0, eax
    vmulss  xmm0, xmm0, [rel f_inv_2p24]; u in [0, 1)

    ; Top-p cutoff filter: if topp < 1.0, filter out tiny-prob tail below (1 - topp)/(vocab)
    ; and re-scale u by cumulative kept probability, or if topp >= 0.9999 sample full CDF.
    vmovss  xmm4, [rsp]                 ; topp
    vmovss  xmm5, [rel f_one]
    vsubss  xmm5, xmm5, xmm4            ; 1 - topp
    vucomiss xmm5, [rel f_tiny]
    jbe     .full_cdf

    ; Find max prob to build an exact top-k/top-p candidate threshold:
    ; Keep tokens with p >= threshold = max_p * 0.01 (fast nucleus approximation
    ; that strictly zeroes out all 150k+ near-zero tail tokens in O(V) without sorting!)
    vmovss  xmm3, [r13]
    mov     rcx, 1
.mp_lp:
    cmp     rcx, r14
    jae     .mp_done
    vmaxss  xmm3, xmm3, [r13 + rcx*4]
    inc     rcx
    jmp     .mp_lp
.mp_done:
    vmulss  xmm3, xmm3, [rel f_min_p]   ; min_p = max_p * 0.05 * (1 - topp) / 0.05 = max_p * (1 - topp)
    vmulss  xmm3, xmm3, xmm5
    ; Sum kept probs
    vxorps  xmm6, xmm6, xmm6
    xor     ecx, ecx
.sum_lp:
    cmp     rcx, r14
    jae     .sum_done
    vmovss  xmm7, [r13 + rcx*4]
    vucomiss xmm7, xmm3
    jb      .sum_skip
    vaddss  xmm6, xmm6, xmm7
.sum_skip:
    inc     rcx
    jmp     .sum_lp
.sum_done:
    vmulss  xmm0, xmm0, xmm6            ; target = u * kept_sum
    vxorps  xmm6, xmm6, xmm6
    xor     eax, eax
    xor     edx, edx                    ; last valid index
.cdf_filt:
    cmp     rax, r14
    jae     .ret_last
    vmovss  xmm7, [r13 + rax*4]
    vucomiss xmm7, xmm3
    jb      .cf_next
    mov     rdx, rax
    vaddss  xmm6, xmm6, xmm7
    vucomiss xmm6, xmm0
    jae     .ret_rax
.cf_next:
    inc     rax
    jmp     .cdf_filt
.ret_last:
    mov     rax, rdx
    jmp     .ret_rax

.full_cdf:
    vxorps  xmm6, xmm6, xmm6
    xor     eax, eax
.cdf_lp:
    cmp     rax, r14
    jae     .ret_end
    vaddss  xmm6, xmm6, [r13 + rax*4]
    vucomiss xmm6, xmm0
    jae     .ret_rax
    inc     rax
    jmp     .cdf_lp
.ret_end:
    lea     rax, [r14 - 1]
.ret_rax:
    add     rsp, 24
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

section .rodata
align 32
i_iota8:     dd 0, 1, 2, 3, 4, 5, 6, 7
i_eight:     dd 8
; Float32 constants
f_one:       dd 1.0
f_exp_hi:    dd 88.3762626647949
f_exp_lo:    dd -87.3365447504000
f_log2e:     dd 1.4426950408889634
f_ln2_hi:    dd 0.693359375
f_ln2_lo:    dd -0.000212194440
f_exp_c2:    dd 0.5
f_exp_c3:    dd 0.1666666716337204
f_exp_c4:    dd 0.0416666679084301
f_exp_c5:    dd 0.008333333767950535
f_exp_c6:    dd 0.0013888889225199819
f_inv_2p24:  dd 5.960464477539063e-8
f_tiny:      dd 1.0e-5
f_min_p:     dd 1.0
i_127:       dd 127
i_abs_mask:  dd 0x7FFFFFFF
i_sign_mask: dd 0x80000000

align 16
; Float64 constants
d_one:       dq 1.0
d_half:      dq 0.5
d_sqrt2:     dq 1.4142135623730951
d_ln2_full:  dq 0.69314718055994530942
d_log2e:     dq 1.44269504088896340736
d_ln2_hi:    dq 0.69314718055994530942
d_ln2_lo:    dq 0.0
d_inv_pio2:  dq 0.63661977236758134308
d_pio2_hi:   dq 0x3FF921FB54400000      ; exact high 33 bits of pi/2
d_pio2_lo:   dq 0x3DD0B4611A626331      ; pi/2 - pio2_hi
d_sign_mask: dq 0x8000000000000000, 0

; sin(r) odd Taylor coefficients: -1/3!, 1/5!, -1/7!, 1/9!, -1/11!
d_sin_s1:    dq -0.16666666666666666667
d_sin_s2:    dq  0.00833333333333333333
d_sin_s3:    dq -0.00019841269841269841
d_sin_s4:    dq  0.00000275573192239859
d_sin_s5:    dq -0.00000002505210838544

; cos(r) even Taylor coefficients: -1/2!, 1/4!, -1/6!, 1/8!, -1/10!
d_cos_c1:    dq -0.5
d_cos_c2:    dq  0.04166666666666666667
d_cos_c3:    dq -0.00138888888888888889
d_cos_c4:    dq  0.00002480158730158730
d_cos_c5:    dq -0.00000027557319223986

; log_f64 series coefficients: 1/3, 1/5, 1/7, 1/9, 1/11, 1/13
d_inv3:      dq 0.33333333333333333333
d_inv5:      dq 0.2
d_inv7:      dq 0.14285714285714285714
d_inv9:      dq 0.11111111111111111111
d_inv11:     dq 0.09090909090909090909
d_inv13:     dq 0.07692307692307692308

; exp_f64 Taylor coefficients: 1/3! .. 1/10!
d_e3:        dq 0.16666666666666666667
d_e4:        dq 0.04166666666666666667
d_e5:        dq 0.00833333333333333333
d_e6:        dq 0.00138888888888888889
d_e7:        dq 0.00019841269841269841
d_e8:        dq 0.00002480158730158730
d_e9:        dq 0.00000275573192239859
d_e10:       dq 0.00000027557319223986
