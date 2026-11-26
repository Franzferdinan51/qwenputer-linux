; ============================================================================
; transformer_lfm3.asm — LFM3 forward pass for the multi-arch .asmbin v3
;
; LiquidAI LFM3-1.7B architecture (from modeling_lfm3.py, read directly):
;   - 24 layers: 13 short-conv, 6 full-attention, 5 sliding-attention
;     (window 256), dispatched per layer from the arch table
;   - Sandwich RMSNorms: operator/ffn pre-norms + post norms
;   - Conv block: in_proj -> [B, C, x]; h = B*x; causal depthwise conv
;     (kernel 7, rolling state); y = C*conv(h); out_proj
;   - Attention: QK RMSNorm, RoPE (full layers: theta 1e6, partial 0.5
;     zero-tail via zeroed inv_freq entries; sliding: theta 1e4),
;     per-head sigmoid gate from gate_proj applied before o_proj
;   - MLP: SwiGLU w2(silu(w1(x)) * w3(x))
;   - Final norm = embedding_norm weight; tied embeddings
;
; Correctness-first: projections use the shared thread-pool Q8 matmul,
; attention/conv are scalar loops over the same head-major KV layout
; the Qwen3 path uses. Shares g_ctx and all math primitives.
; ============================================================================

bits 64
default rel

%include "constants.inc"

; LFM3 per-layer weight layout (see scripts/export_lfm3.py)
%define LF_N_OP        0
%define LF_N_POST_OP   8192
%define LF_N_FFN       16384
%define LF_N_POST_FFN  24576
%define LF_N_QNORM     32768
%define LF_N_KNORM     33280
%define LF_SLOTS       33792

extern g_ctx
extern rmsnorm_avx2
extern tp_matmul_q8_0
extern dequant_row_q8_0
extern compute_rope_cs
extern rope_apply_avx2
extern softmax_avx2
extern silu_mul_avx2
extern vec_add_avx2
extern dot_f32_avx2
extern saxpy_f32_avx2
extern fast_expf_scalar

section .text
global transformer_forward_lfm3

; ----------------------------------------------------------------------------
; void transformer_forward_lfm3(uint32 token_id, uint64 pos, int logits)
; Same contract as transformer_forward.
; ----------------------------------------------------------------------------
transformer_forward_lfm3:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 168
    ; locals:
    ; [rsp+0]   pos            [rsp+8]   compute_logits
    ; [rsp+16]  layer l        [rsp+24]  conv ordinal
    ; [rsp+32]  layer base     [rsp+40]  att_scale f32 | gqa_ratio u32 @44
    ; [rsp+48]  slot A         [rsp+56]  slot B
    ; [rsp+64]  slot C         [rsp+72]  slot D
    ; [rsp+80]  slot E         [rsp+88]  slot F (w1)
    ; [rsp+96]  slot G (w3)    [rsp+104] slot H (w2)
    ; [rsp+112] layer type     [rsp+120] layer kv base bytes / conv_k
    ; [rsp+128] t_start        [rsp+136] cos ptr (selected) / sin @144
    ; [rsp+152] token_id (rdi is clobbered by the conv-state reset)
    mov     [rsp + 0], rsi
    mov     [rsp + 8], rdx
    mov     [rsp + 152], rdi
    lea     rbx, [rel g_ctx]

    ; att_scale = 1/sqrt(head_dim), gqa_ratio = n_heads / n_kv_heads
    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel lf_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 40], xmm1
    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    mov     [rsp + 44], eax

    ; pos == 0: fresh sequence -> zero all conv states
    cmp     qword [rsp + 0], 0
    jne     .no_reset
    mov     rdi, [rbx + CTX_BUF_CONV_STATE]
    mov     rcx, [rbx + CTX_N_LAYERS]
    mov     eax, [rbx + CTX_CONV_KERNEL]    ; u32 field: load 32-bit!
    imul    rcx, rax
    imul    rcx, [rbx + CTX_DIM]
    vxorps  ymm0, ymm0, ymm0
.zero_lp:
    vmovups [rdi], ymm0
    add     rdi, 32
    sub     rcx, 8
    jnz     .zero_lp
.no_reset:

    ; Embedding lookup (same as Qwen3 path)
    mov     rax, [rbx + CTX_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     rdx, rax
    shr     rdx, cl
    add     rdx, rax
    mov     rdi, [rsp + 152]
    imul    rdi, rdx
    add     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, rax
    call    dequant_row_q8_0

    ; RoPE tables for both attention kinds at this position
    mov     rdi, [rbx + CTX_BUF_COS]
    mov     rsi, [rbx + CTX_BUF_SIN]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1
    mov     r8,  [rsp + 0]
    call    compute_rope_cs
    mov     rdi, [rbx + CTX_BUF_COS2]
    mov     rsi, [rbx + CTX_BUF_SIN2]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ2]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1
    mov     r8,  [rsp + 0]
    call    compute_rope_cs

    mov     qword [rsp + 16], 0         ; l = 0
    mov     qword [rsp + 24], 0         ; conv ordinal = 0

.layer_lp:
    mov     rax, [rsp + 16]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .layers_done

    ; layer base + type from the arch table
    mov     rcx, [rbx + CTX_ARCH_PTR]
    movzx   edx, byte [rcx + rax]       ; type (ARCH_OFF_TYPES = 0)
    mov     [rsp + 112], rdx
    mov     rdx, [rcx + ARCH_OFF_LAYERS + rax*8]
    add     rdx, [rbx + CTX_LAYERS_PTR]
    mov     [rsp + 32], rdx

    ; slot pointers (row stride rs(K) = K + (K >> scale_shift))
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    mov     r12, [rbx + CTX_DIM]
    mov     r13, r12
    shr     r13, cl
    add     r12, r13                    ; r12 = rs(dim)
    lea     rax, [rdx + LF_SLOTS]
    mov     [rsp + 48], rax             ; A
    cmp     qword [rsp + 112], 0
    je      .slots_conv
    ; attention: A = q_dim * rs(dim)
    mov     rcx, [rbx + CTX_Q_DIM]
    imul    rcx, r12
    add     rax, rcx
    mov     [rsp + 56], rax             ; B = k_proj
    mov     rcx, [rbx + CTX_KV_DIM]
    imul    rcx, r12
    add     rax, rcx
    mov     [rsp + 64], rax             ; C = v_proj
    add     rax, rcx
    mov     [rsp + 72], rax             ; D = o_proj (K = q_dim)
    ; rs(q_dim):
    mov     rdx, [rbx + CTX_Q_DIM]
    mov     ecx, [rbx + CTX_SCALE_SHIFT]
    shr     rdx, cl
    add     rdx, [rbx + CTX_Q_DIM]      ; rdx = rs(q_dim)
    imul    rdx, [rbx + CTX_DIM]
    add     rax, rdx
    mov     [rsp + 80], rax             ; E = gate_proj
    mov     rcx, [rbx + CTX_N_HEADS]
    imul    rcx, r12
    add     rax, rcx
    jmp     .slots_mlp
.slots_conv:
    mov     rcx, [rbx + CTX_DIM]
    lea     rcx, [rcx + rcx*2]
    imul    rcx, r12                    ; A size = 3*dim * rs(dim)
    add     rax, rcx
    mov     [rsp + 56], rax             ; B = conv weight fp32
    mov     rcx, [rbx + CTX_DIM]
    mov     edx, [rbx + CTX_CONV_KERNEL]    ; u32 field: load 32-bit!
    imul    rcx, rdx
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 64], rax             ; C = out_proj
    mov     rcx, [rbx + CTX_DIM]
    imul    rcx, r12
    add     rax, rcx
.slots_mlp:
    mov     [rsp + 88], rax             ; F = w1
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    imul    rcx, r12
    add     rax, rcx
    mov     [rsp + 96], rax             ; G = w3
    add     rax, rcx
    mov     [rsp + 104], rax            ; H = w2

    ; xb = rmsnorm(x, operator_norm)
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_OP
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    cmp     qword [rsp + 112], 0
    je      .conv_block
    jmp     .attn_block

; ---------------------------------------------------------------------------
.conv_block:
    ; bcx = in_proj(xb)  [B | C | x]
    mov     rdi, [rsp + 48]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_BCX]
    mov     rcx, [rbx + CTX_DIM]
    lea     rcx, [rcx + rcx*2]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

    ; h[i] = bcx[2*dim + i] * bcx[i]        (x * B)
    mov     rdi, [rbx + CTX_BUF_BCX]
    mov     rsi, [rbx + CTX_BUF_H]
    mov     rcx, [rbx + CTX_DIM]
    mov     rdx, rcx
    shl     rdx, 2
    lea     r8,  [rdi + rdx*2]          ; &bcx[2*dim]
    xor     eax, eax
.h_lp:
    vmovups ymm0, [rdi + rax]
    vmovups ymm1, [r8 + rax]
    vmulps  ymm0, ymm0, ymm1
    vmovups [rsi + rax], ymm0
    add     rax, 32
    sub     rcx, 8
    jnz     .h_lp

    ; conv state: shift window left by one vector, append h
    mov     eax, [rbx + CTX_CONV_KERNEL]    ; u32 field: load 32-bit!
    mov     [rsp + 120], rax            ; stash conv_k for this block
    mov     rax, [rsp + 24]             ; ordinal
    imul    rax, [rsp + 120]
    imul    rax, [rbx + CTX_DIM]
    shl     rax, 2
    add     rax, [rbx + CTX_BUF_CONV_STATE]
    mov     rdi, rax                    ; state base
    mov     rcx, [rsp + 120]
    dec     rcx
    imul    rcx, [rbx + CTX_DIM]        ; floats to shift = (k-1)*dim
    mov     rdx, [rbx + CTX_DIM]
    shl     rdx, 2                      ; byte offset of one vector
    lea     rsi, [rdi + rdx]
.shift_lp:
    vmovups ymm0, [rsi]
    vmovups [rdi], ymm0
    add     rdi, 32
    add     rsi, 32
    sub     rcx, 8
    jnz     .shift_lp
    ; append h at slot k-1
    mov     rsi, [rbx + CTX_BUF_H]
    mov     rcx, [rbx + CTX_DIM]
.append_lp:
    vmovups ymm0, [rsi]
    vmovups [rdi], ymm0
    add     rsi, 32
    add     rdi, 32
    sub     rcx, 8
    jnz     .append_lp

    ; conv_out[i] = sum_j w[i*K + j] * state[j*dim + i]   -> buf_xb
    mov     r12, [rsp + 56]             ; conv weight [dim][K] fp32
    mov     rax, [rsp + 24]
    imul    rax, [rsp + 120]
    imul    rax, [rbx + CTX_DIM]
    shl     rax, 2
    add     rax, [rbx + CTX_BUF_CONV_STATE]
    mov     r13, rax
    mov     r14, [rbx + CTX_BUF_XB]     ; out
    mov     r15, [rbx + CTX_DIM]        ; i count
    xor     ecx, ecx                    ; i = 0
.conv_i_lp:
    cmp     rcx, r15
    jae     .conv_done
    mov     rax, rcx
    imul    rax, [rsp + 120]
    lea     r8,  [r12 + rax*4]          ; &w[i*K]
    vxorps  xmm0, xmm0, xmm0            ; acc
    xor     edx, edx                    ; j = 0
.conv_j_lp:
    cmp     rdx, [rsp + 120]
    jae     .conv_j_done
    mov     rax, rdx
    imul    rax, [rbx + CTX_DIM]
    add     rax, rcx
    vmovss  xmm1, [r13 + rax*4]         ; state[j*dim + i]
    vmulss  xmm1, xmm1, [r8 + rdx*4]
    vaddss  xmm0, xmm0, xmm1
    inc     rdx
    jmp     .conv_j_lp
.conv_j_done:
    vmovss  [r14 + rcx*4], xmm0
    inc     rcx
    jmp     .conv_i_lp
.conv_done:

    ; y[i] = bcx[dim + i] * conv_out[i]     (C * out) -> buf_h
    mov     rdi, [rbx + CTX_BUF_BCX]
    mov     rdx, [rbx + CTX_DIM]
    shl     rdx, 2
    add     rdi, rdx                    ; &bcx[dim] = C
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     r8,  [rbx + CTX_BUF_H]
    mov     rcx, [rbx + CTX_DIM]
    xor     eax, eax
.y_lp:
    vmovups ymm0, [rdi + rax]
    vmovups ymm1, [rsi + rax]
    vmulps  ymm0, ymm0, ymm1
    vmovups [r8 + rax], ymm0
    add     rax, 32
    sub     rcx, 8
    jnz     .y_lp

    ; att_out = out_proj(y); post_operator_norm in place; x += att_out
    mov     rdi, [rsp + 64]
    mov     rsi, [rbx + CTX_BUF_H]
    mov     rdx, [rbx + CTX_BUF_ATT_OUT]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rbx + CTX_BUF_ATT_OUT]
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_POST_OP
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2
    inc     qword [rsp + 24]            ; ordinal++
    jmp     .mlp_block

; ---------------------------------------------------------------------------
.attn_block:
    ; q, k, v projections (separate matmuls; xb still needed for gate)
    mov     rdi, [rsp + 48]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_Q]
    mov     rcx, [rbx + CTX_Q_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rsp + 56]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_K]
    mov     rcx, [rbx + CTX_KV_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rsp + 64]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_V]
    mov     rcx, [rbx + CTX_KV_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

    ; QK RMSNorm per head (in place)
    mov     r12, [rbx + CTX_BUF_Q]
    mov     r13, [rbx + CTX_N_HEADS]
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2
.qn_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_QNORM
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .qn_lp
    mov     r12, [rbx + CTX_BUF_K]
    mov     r13, [rbx + CTX_N_KV_HEADS]
.kn_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_KNORM
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .kn_lp

    ; RoPE table selection: full (1) -> COS/SIN, sliding (2) -> COS2/SIN2
    mov     rax, [rbx + CTX_BUF_COS]
    mov     rcx, [rbx + CTX_BUF_SIN]
    cmp     qword [rsp + 112], 2
    jne     .rope_sel_done
    mov     rax, [rbx + CTX_BUF_COS2]
    mov     rcx, [rbx + CTX_BUF_SIN2]
.rope_sel_done:
    mov     [rsp + 136], rax
    mov     [rsp + 144], rcx
    mov     rdi, [rbx + CTX_BUF_Q]
    mov     rsi, rax
    mov     rdx, rcx
    mov     rcx, [rbx + CTX_N_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2
    mov     rdi, [rbx + CTX_BUF_K]
    mov     rsi, [rsp + 136]
    mov     rdx, [rsp + 144]
    mov     rcx, [rbx + CTX_N_KV_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

    ; Store K, V head-major: cache[l][kv_h][pos][head_dim]
    mov     rax, [rsp + 16]
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    mov     [rsp + 120], rax            ; layer kv base bytes
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes
    mov     r15, [rbx + CTX_MAX_SEQ]
    imul    r15, r14                    ; kv_head_stride
    mov     rcx, [rsp + 0]
    imul    rcx, r14
    add     rax, rcx
    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     r8,  [rbx + CTX_BUF_V]
    mov     r12, [rbx + CTX_N_KV_HEADS]
.kv_lp:
    xor     r9d, r9d
.kv_cp:
    cmp     r9, r14
    jae     .kv_next
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_cp
.kv_next:
    add     rsi, r14
    add     r8,  r14
    add     rdi, r15
    add     rdx, r15
    dec     r12
    jnz     .kv_lp

    ; t_start: sliding layers attend only the last `window` positions
    mov     qword [rsp + 128], 0
    cmp     qword [rsp + 112], 2
    jne     .tstart_done
    mov     rax, [rsp + 0]
    mov     ecx, [rbx + CTX_SLIDING_WIN]
    dec     ecx
    sub     rax, rcx
    js      .tstart_done
    mov     [rsp + 128], rax
.tstart_done:

    ; Attention per head (scalar)
    xor     r12d, r12d                  ; h = 0
.head_lp:
    cmp     r12, [rbx + CTX_N_HEADS]
    jae     .att_done
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 44]            ; kv_h = h / gqa_ratio
    mov     r13, rax
    imul    r13, r15                    ; kv_h * kv_head_stride (r15 kept)
    add     r13, [rsp + 120]            ; + layer base
    ; q_head
    mov     r14, r12
    imul    r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2
    add     r14, [rbx + CTX_BUF_Q]
    ; score buffer for this head
    mov     r11, r12
    imul    r11, [rbx + CTX_MAX_SEQ]
    shl     r11, 2
    add     r11, [rbx + CTX_BUF_ATT_SCR]
    ; scores over t in [t_start, pos]
    mov     rsi, [rbx + CTX_BUF_K_CACHE]
    add     rsi, r13
    mov     rcx, [rsp + 128]
    imul    rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rsi, rcx                    ; k ptr at t_start
    mov     rcx, [rsp + 128]            ; t
.score_lp:
    cmp     rcx, [rsp + 0]
    ja      .scores_done
    push    rcx
    push    r11
    push    rsi
    push    rax                         ; 4th push keeps call alignment
    mov     rdi, r14
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    dot_f32_avx2
    pop     rax
    pop     rsi
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 40]
    mov     rax, rcx
    sub     rax, [rsp + 128]
    vmovss  [r11 + rax*4], xmm0
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    add     rsi, rax
    inc     rcx
    jmp     .score_lp
.scores_done:
    mov     rdi, r11
    mov     rsi, [rsp + 0]
    sub     rsi, [rsp + 128]
    inc     rsi
    call    softmax_avx2
    ; att_head = sum_t prob[t] * v[t]
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.az_lp:
    cmp     rax, rcx
    jae     .az_done
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .az_lp
.az_done:
    mov     rsi, [rbx + CTX_BUF_V_CACHE]
    add     rsi, r13
    mov     rcx, [rsp + 128]
    imul    rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rsi, rcx
    mov     rcx, [rsp + 128]
.val_lp:
    cmp     rcx, [rsp + 0]
    ja      .head_next
    mov     rax, rcx
    sub     rax, [rsp + 128]
    vmovss  xmm0, [r11 + rax*4]
    push    rcx
    push    r11
    push    rsi
    push    rax                         ; 4th push keeps call alignment
    mov     rdi, r14
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    saxpy_f32_avx2
    pop     rax
    pop     rsi
    pop     r11
    pop     rcx
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    add     rsi, rax
    inc     rcx
    jmp     .val_lp
.head_next:
    inc     r12
    jmp     .head_lp
.att_done:

    ; gate = sigmoid(gate_proj(xb)); att_out[h] *= gate[h]
    mov     rdi, [rsp + 80]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_GATE16]
    mov     rcx, [rbx + CTX_N_HEADS]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    xor     r12d, r12d
.gate_lp:
    cmp     r12, [rbx + CTX_N_HEADS]
    jae     .gate_done
    mov     rax, [rbx + CTX_BUF_GATE16]
    vmovss  xmm0, [rax + r12*4]
    vxorps  xmm1, xmm1, xmm1
    vsubss  xmm0, xmm1, xmm0            ; -x
    call    fast_expf_scalar            ; exp(-x)
    vmovss  xmm1, [rel lf_one]
    vaddss  xmm0, xmm0, xmm1
    vdivss  xmm1, xmm1, xmm0            ; sigmoid
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     rdx, [rbx + CTX_BUF_ATT_OUT]
    add     rdx, rax
    vbroadcastss ymm1, xmm1
    mov     rcx, [rbx + CTX_HEAD_DIM]
    xor     eax, eax
.gate_mul_lp:
    vmovups ymm0, [rdx + rax]
    vmulps  ymm0, ymm0, ymm1
    vmovups [rdx + rax], ymm0
    add     rax, 32
    sub     rcx, 8
    jnz     .gate_mul_lp
    inc     r12
    jmp     .gate_lp
.gate_done:

    ; xb = o_proj(att_out); post_operator_norm in place; x += xb
    mov     rdi, [rsp + 72]
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_Q_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_POST_OP
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

; ---------------------------------------------------------------------------
.mlp_block:
    ; xb = rmsnorm(x, ffn_norm)
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_FFN
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    ; gate = w1(xb), up = w3(xb)
    mov     rdi, [rsp + 88]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_GATE]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rsp + 96]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_UP]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
    ; gate = silu(gate) * up
    mov     rdi, [rbx + CTX_BUF_GATE]
    mov     rsi, [rbx + CTX_BUF_UP]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2
    ; xb = w2(gate); post_ffn_norm in place; x += xb
    mov     rdi, [rsp + 104]
    mov     rsi, [rbx + CTX_BUF_GATE]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_HIDDEN_DIM]
    call    tp_matmul_q8_0
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rsp + 32]
    add     rdx, LF_N_POST_FFN
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    inc     qword [rsp + 16]
    jmp     .layer_lp

.layers_done:
    cmp     qword [rsp + 8], 0
    je      .done
    ; Final norm (embedding_norm weight) + tied LM head
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    mov     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0
.done:
    add     rsp, 168
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

section .rodata
align 4
lf_one: dd 1.0
