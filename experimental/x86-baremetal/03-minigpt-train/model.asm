; ============================================================================
; model.asm — Pure x86-64 Mini-GPT Model Layout, Init, Forward, Backward,
;             AdamW Step & Autoregressive Sampling (Zero libc/libm)
;
; Architecture (FP32, modern Llama/GPT style):
;   - Token Embedding [V, D]
;   - L Transformer Layers:
;       RMSNorm -> Q/K/V -> Split-Half RoPE -> Causal MHA -> O-proj + Residual
;       RMSNorm -> SwiGLU (gate/up/down) + Residual
;   - Final RMSNorm -> Untied LM Head [V, D] -> Cross-Entropy Loss
;
; Contiguous flat parameter array order (for exact PyTorch verification):
;   [tok_emb (V*D)]
;   for l in 0..L-1:
;     [ln1 (D), wq (D*D), wk (D*D), wv (D*D), wo (D*D),
;      ln2 (D), wgate (FF*D), wup (FF*D), wdown (D*FF)]
;   [final_ln (D)]
;   [lm_head (V*D)]
; ============================================================================

bits 64
default rel

%include "train_constants.inc"

extern sys_mmap_anon
extern sys_munmap
extern sys_write_all

extern init_rope_freqs
extern compute_rope_cs
extern argmax_f32
extern sample_top_p

extern tp_sgemm
extern tp_sgemm_nt

extern zero_f32_avx2
extern vec_add_f32_avx2
extern vec_add3_f32_avx2
extern tp_rmsnorm_fwd
extern tp_rmsnorm_bwd
extern tp_rope_batch
extern tp_mha_fwd
extern tp_mha_bwd
extern tp_swiglu_fwd
extern tp_swiglu_bwd
extern tp_cross_entropy_fwd_bwd
extern tp_grad_norm
extern tp_vec_scale
extern tp_adamw_range
extern init_normal_f32
extern init_ones_f32
extern g_sample_capture
extern tui_sample_push_byte

section .text

global g_trn
global gpt_init_layout
global gpt_init_weights
global gpt_free
global gpt_forward
global gpt_backward
global gpt_adamw_step
global gpt_sample

%macro ALIGN64 1
    add     %1, 63
    and     %1, -64
%endmacro

; ============================================================================
; int64 gpt_init_layout(void)
; Uses g_trn.{VOCAB, DIM, LAYERS, HEADS, HIDDEN, SEQ, BATCH, SAMPLE_ONLY}
; Computes parameter offsets, allocates parameter & activation arenas,
; precomputes RoPE cos/sin tables. Returns 0 on success, <0 on error.
; ============================================================================
gpt_init_layout:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 24                     ; 6*8 + 24 + 8 = 80 (16B aligned)

    lea     rbx, [rel g_trn]

    ; head_dim = dim / heads
    mov     rax, [rbx + TRN_DIM]
    xor     edx, edx
    div     qword [rbx + TRN_HEADS]
    mov     [rbx + TRN_HEAD_DIM], rax

    ; att_scale = 1.0f / sqrt(head_dim)
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel mf_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rbx + TRN_ATT_SCALE], xmm1

    ; BT = B * T
    mov     rax, [rbx + TRN_BATCH]
    imul    rax, [rbx + TRN_SEQ]
    mov     [rbx + TRN_BT], rax

    ; Compute parameter sub-offsets (in FLOAT counts)
    mov     r12, [rbx + TRN_DIM]        ; D
    mov     r13, [rbx + TRN_HIDDEN]     ; FF
    mov     r14, r12
    imul    r14, r12                    ; D*D
    mov     r15, r12
    imul    r15, r13                    ; D*FF

    ; Within a single layer:
    mov     qword [rbx + TRN_LOFF_LN1], 0
    mov     rax, r12                    ; D
    mov     [rbx + TRN_LOFF_WQ], rax
    add     rax, r14
    mov     [rbx + TRN_LOFF_WK], rax
    add     rax, r14
    mov     [rbx + TRN_LOFF_WV], rax
    add     rax, r14
    mov     [rbx + TRN_LOFF_WO], rax
    add     rax, r14
    mov     [rbx + TRN_LOFF_LN2], rax
    add     rax, r12
    mov     [rbx + TRN_LOFF_WGATE], rax
    add     rax, r15
    mov     [rbx + TRN_LOFF_WUP], rax
    add     rax, r15
    mov     [rbx + TRN_LOFF_WDOWN], rax
    add     rax, r15
    mov     [rbx + TRN_LAYER_PARAMS], rax ; 2*D + 4*D*D + 3*D*FF

    ; Global parameter offsets:
    mov     rcx, [rbx + TRN_VOCAB]
    imul    rcx, r12                    ; V * D
    mov     [rbx + TRN_POFF_LAYERS], rcx

    imul    rax, [rbx + TRN_LAYERS]     ; L * layer_params
    add     rax, rcx                    ; V*D + L*layer_params
    mov     [rbx + TRN_POFF_FINAL_LN], rax
    add     rax, r12                    ; + D
    mov     [rbx + TRN_POFF_LM_HEAD], rax
    add     rax, rcx                    ; + V*D
    mov     [rbx + TRN_N_PARAMS], rax

    ; Allocate parameter arena:
    ;   Train mode: params, grads, opt_m, opt_v (4 * aligned(n_params * 4))
    ;   Sample-only mode: params only (1 * aligned(n_params * 4))
    shl     rax, 2                      ; n_params * 4
    ALIGN64 rax
    mov     r14, rax                    ; single param block size
    cmp     qword [rbx + TRN_SAMPLE_ONLY], 0
    jne     .param_sample_only
    shl     rax, 2                      ; * 4
.param_sample_only:
    mov     [rbx + TRN_PARAM_ARENA_SZ], rax
    mov     rdi, rax
    call    sys_mmap_anon
    test    rax, rax
    js      .err
    mov     [rbx + TRN_PARAM_ARENA], rax
    mov     [rbx + TRN_PARAMS], rax
    cmp     qword [rbx + TRN_SAMPLE_ONLY], 0
    jne     .param_done
    add     rax, r14
    mov     [rbx + TRN_GRADS], rax
    add     rax, r14
    mov     [rbx + TRN_OPT_M], rax
    add     rax, r14
    mov     [rbx + TRN_OPT_V], rax
.param_done:

    ; Initialize AdamW FP64 powers b1_pow = 1.0, b2_pow = 1.0
    vmovsd  xmm0, [rel md_one]
    vmovsd  [rbx + TRN_B1_POW], xmm0
    vmovsd  [rbx + TRN_B2_POW], xmm0

    ; ------------------------------------------------------------------------
    ; Compute per-layer activation block offsets (inside LACT)
    ; ------------------------------------------------------------------------
    mov     r8,  [rbx + TRN_BT]         ; BT
    mov     r9,  r8
    imul    r9,  r12
    shl     r9,  2                      ; BT * D * 4
    mov     r10, r8
    imul    r10, r13
    shl     r10, 2                      ; BT * FF * 4
    mov     r11, r8
    shl     r11, 2                      ; BT * 4

    xor     rbp, rbp
    %macro ALLOC_LOFF 2
        ALIGN64 rbp
        mov     [rbx + %1], rbp
        add     rbp, %2
    %endmacro

    ALLOC_LOFF TRN_OFF_LN1_OUT, r9
    ALLOC_LOFF TRN_OFF_LN1_INV, r11
    ALLOC_LOFF TRN_OFF_Q,       r9
    ALLOC_LOFF TRN_OFF_K,       r9
    ALLOC_LOFF TRN_OFF_V,       r9

    ; Probs: B * H * T * T * 4
    mov     rax, [rbx + TRN_BATCH]
    imul    rax, [rbx + TRN_HEADS]
    imul    rax, [rbx + TRN_SEQ]
    imul    rax, [rbx + TRN_SEQ]
    shl     rax, 2
    ALLOC_LOFF TRN_OFF_PROBS,   rax

    ALLOC_LOFF TRN_OFF_ATT_OUT, r9
    ALLOC_LOFF TRN_OFF_X_MID,   r9
    ALLOC_LOFF TRN_OFF_LN2_OUT, r9
    ALLOC_LOFF TRN_OFF_LN2_INV, r11
    ALLOC_LOFF TRN_OFF_GATE,    r10
    ALLOC_LOFF TRN_OFF_UP,      r10
    ALLOC_LOFF TRN_OFF_ACT,     r10
    ALIGN64 rbp
    mov     [rbx + TRN_LACT_STRIDE], rbp

    ; ------------------------------------------------------------------------
    ; Compute global activation & scratch arena layout
    ; ------------------------------------------------------------------------
    xor     rbp, rbp
    %macro ALLOC_ABUF 2
        ALIGN64 rbp
        mov     [rbx + %1], rbp
        add     rbp, %2
    %endmacro

    ; layer_x: (L + 1) * BT * D * 4
    mov     rax, [rbx + TRN_LAYERS]
    inc     rax
    imul    rax, r9
    ALLOC_ABUF TRN_BUF_LAYER_X, rax

    ; LACT: L * lact_stride (in sample_only, only 1 layer needed? Keep L layers, tiny for B=1!)
    mov     rax, [rbx + TRN_LAYERS]
    imul    rax, [rbx + TRN_LACT_STRIDE]
    ALLOC_ABUF TRN_BUF_LACT, rax

    ALLOC_ABUF TRN_BUF_FINAL_OUT, r9
    ALLOC_ABUF TRN_BUF_FINAL_INV, r11

    ; logits: BT * V * 4
    mov     rax, r8
    imul    rax, [rbx + TRN_VOCAB]
    shl     rax, 2
    ALLOC_ABUF TRN_BUF_LOGITS, rax
    ALLOC_ABUF TRN_BUF_LOSSES, r11

    ; buf_wt: max(D*D, D*FF, V*D) * 4 for W^T transpose
    mov     rax, r12
    imul    rax, r12                    ; D*D
    mov     rcx, r12
    imul    rcx, r13                    ; D*FF
    cmp     rcx, rax
    jbe     .wt1
    mov     rax, rcx
.wt1:
    mov     rcx, [rbx + TRN_VOCAB]
    imul    rcx, r12                    ; V*D
    cmp     rcx, rax
    jbe     .wt2
    mov     rax, rcx
.wt2:
    shl     rax, 2
    ALLOC_ABUF TRN_BUF_WT, rax

    ; Backward scratch buffers (also used for o_out / down_out in forward -> reuse BUF_DTMP!)
    ALLOC_ABUF TRN_BUF_DX,    r9
    ALLOC_ABUF TRN_BUF_DTMP,  r9
    ALLOC_ABUF TRN_BUF_DQ,    r9
    ALLOC_ABUF TRN_BUF_DK,    r9
    ALLOC_ABUF TRN_BUF_DV,    r9
    ALLOC_ABUF TRN_BUF_DGATE, r10
    ALLOC_ABUF TRN_BUF_DUP,   r10
    ALLOC_ABUF TRN_BUF_DACT,  r10

    ; RoPE buffers: inv_freq [hd/2 * 8], cos [T * hd/2 * 4], sin [T * hd/2 * 4]
    mov     rax, [rbx + TRN_HEAD_DIM]
    shl     rax, 2                      ; (hd/2) * 8
    ALLOC_ABUF TRN_BUF_INV_FREQ, rax
    mov     rax, [rbx + TRN_HEAD_DIM]
    shr     rax, 1
    imul    rax, [rbx + TRN_SEQ]
    shl     rax, 2                      ; T * (hd/2) * 4
    ALLOC_ABUF TRN_BUF_ROPE_COS, rax
    ALLOC_ABUF TRN_BUF_ROPE_SIN, rax

    ; Batch x/y & sampling probs
    ALLOC_ABUF TRN_BUF_BATCH_X, r11
    ALLOC_ABUF TRN_BUF_BATCH_Y, r11
    mov     rax, [rbx + TRN_VOCAB]
    shl     rax, 2
    ALLOC_ABUF TRN_BUF_SAMPLE_P, rax

    ALIGN64 rbp
    mov     [rbx + TRN_ACT_ARENA_SZ], rbp

    mov     rdi, rbp
    call    sys_mmap_anon
    test    rax, rax
    js      .err
    mov     [rbx + TRN_ACT_ARENA], rax

    ; Relocate all activation buffer offsets -> absolute pointers:
    ;   BUF_LAYER_X (224), BUF_LACT (232)
    ;   BUF_FINAL_OUT (352) .. BUF_SAMPLE_P (496) -> 19 pointers!
    add     [rbx + TRN_BUF_LAYER_X], rax
    add     [rbx + TRN_BUF_LACT], rax
    lea     rcx, [rbx + TRN_BUF_FINAL_OUT]
    mov     edx, 19
.reloc_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc_lp

    ; Precompute RoPE inv_freq and cos/sin tables for t = 0 .. T-1
    mov     rdi, [rbx + TRN_BUF_INV_FREQ]
    mov     rsi, [rbx + TRN_HEAD_DIM]
    vmovss  xmm0, [rbx + TRN_ROPE_THETA]
    call    init_rope_freqs

    mov     r12, [rbx + TRN_HEAD_DIM]
    shr     r12, 1                      ; half_dim
    mov     r13, r12
    shl     r13, 2                      ; half_bytes
    xor     r14d, r14d                  ; t = 0
.rope_lp:
    cmp     r14, [rbx + TRN_SEQ]
    jae     .ok
    mov     rax, r14
    imul    rax, r13
    mov     rdi, [rbx + TRN_BUF_ROPE_COS]
    add     rdi, rax
    mov     rsi, [rbx + TRN_BUF_ROPE_SIN]
    add     rsi, rax
    mov     rdx, [rbx + TRN_BUF_INV_FREQ]
    mov     rcx, r12
    mov     r8,  r14
    call    compute_rope_cs
    inc     r14
    jmp     .rope_lp

.ok:
    xor     eax, eax
    jmp     .ret
.err:
    mov     rax, -1
.ret:
    add     rsp, 24
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; void gpt_init_weights(void)
; Deterministic Box-Muller init: std=0.02, residual projections (wo, wdown)
; scaled by 1/sqrt(2*L), RMSNorm weights = 1.0.
; ============================================================================
gpt_init_weights:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 24

    lea     rbx, [rel g_trn]
    mov     r12, [rbx + TRN_DIM]
    mov     r13, [rbx + TRN_HIDDEN]
    mov     r14, r12
    imul    r14, r12                    ; D*D
    mov     r15, r12
    imul    r15, r13                    ; D*FF

    ; std_res = 0.02 / sqrt(2 * L)
    mov     rax, [rbx + TRN_LAYERS]
    add     rax, rax
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel mf_init_std]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 0], xmm1             ; std_res

    lea     rbp, [rbx + TRN_RNG]

    ; 1. tok_emb [V*D]
    mov     rdi, [rbx + TRN_PARAMS]
    mov     rsi, [rbx + TRN_VOCAB]
    imul    rsi, r12
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    ; 2. Layers 0 .. L-1
    mov     qword [rsp + 8], 0
.layer_lp:
    mov     rax, [rsp + 8]
    cmp     rax, [rbx + TRN_LAYERS]
    jae     .final_init
    imul    rax, [rbx + TRN_LAYER_PARAMS]
    add     rax, [rbx + TRN_POFF_LAYERS]
    shl     rax, 2
    add     rax, [rbx + TRN_PARAMS]
    mov     [rsp + 16], rax             ; layer_base ptr

    ; ln1 [D] = 1.0
    mov     rdi, rax
    mov     rsi, r12
    call    init_ones_f32

    ; wq, wk, wv [D*D each]
    mov     rax, [rbx + TRN_LOFF_WQ]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r14
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    mov     rax, [rbx + TRN_LOFF_WK]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r14
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    mov     rax, [rbx + TRN_LOFF_WV]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r14
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    ; wo [D*D] with std_res
    mov     rax, [rbx + TRN_LOFF_WO]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r14
    vmovss  xmm0, [rsp + 0]
    mov     rdx, rbp
    call    init_normal_f32

    ; ln2 [D] = 1.0
    mov     rax, [rbx + TRN_LOFF_LN2]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r12
    call    init_ones_f32

    ; wgate, wup [FF*D each]
    mov     rax, [rbx + TRN_LOFF_WGATE]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r15
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    mov     rax, [rbx + TRN_LOFF_WUP]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r15
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    ; wdown [D*FF] with std_res
    mov     rax, [rbx + TRN_LOFF_WDOWN]
    shl     rax, 2
    add     rax, [rsp + 16]
    mov     rdi, rax
    mov     rsi, r15
    vmovss  xmm0, [rsp + 0]
    mov     rdx, rbp
    call    init_normal_f32

    inc     qword [rsp + 8]
    jmp     .layer_lp

.final_init:
    ; 3. final_ln [D] = 1.0
    mov     rax, [rbx + TRN_POFF_FINAL_LN]
    shl     rax, 2
    add     rax, [rbx + TRN_PARAMS]
    mov     rdi, rax
    mov     rsi, r12
    call    init_ones_f32

    ; 4. lm_head [V*D]
    mov     rax, [rbx + TRN_POFF_LM_HEAD]
    shl     rax, 2
    add     rax, [rbx + TRN_PARAMS]
    mov     rdi, rax
    mov     rsi, [rbx + TRN_VOCAB]
    imul    rsi, r12
    vmovss  xmm0, [rel mf_init_std]
    mov     rdx, rbp
    call    init_normal_f32

    add     rsp, 24
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; void gpt_free(void)
; ============================================================================
gpt_free:
    push    rbx
    lea     rbx, [rel g_trn]

    mov     rdi, [rbx + TRN_ACT_ARENA]
    test    rdi, rdi
    jz      .f_param
    mov     rsi, [rbx + TRN_ACT_ARENA_SZ]
    call    sys_munmap
.f_param:
    mov     rdi, [rbx + TRN_PARAM_ARENA]
    test    rdi, rdi
    jz      .f_toks
    mov     rsi, [rbx + TRN_PARAM_ARENA_SZ]
    call    sys_munmap
.f_toks:
    mov     rdi, [rbx + TRN_TOK_IDS]
    test    rdi, rdi
    jz      .f_data
    mov     rsi, [rbx + TRN_DATA_SIZE]
    call    sys_munmap
.f_data:
    mov     rdi, [rbx + TRN_DATA_MMAP]
    test    rdi, rdi
    jz      .done
    mov     rsi, [rbx + TRN_DATA_SIZE]
    call    sys_munmap
.done:
    pop     rbx
    ret

; ============================================================================
; float gpt_forward(const uint32 *x_toks, const uint32 *y_toks,
;                   uint64 cur_B, uint64 cur_T,
;                   uint64 compute_loss, uint64 write_dlogits) -> xmm0 (loss)
; rdi=x_toks, rsi=y_toks, rdx=cur_B, rcx=cur_T, r8=compute_loss, r9=write_dlogits
; ============================================================================
gpt_forward:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136                    ; 6*8 + 136 + 8 = 192 (16B aligned)
    ; Locals (keep [rsp+0..31] free for outgoing stack arguments!):
    ;   [rsp + 32] = x_toks
    ;   [rsp + 40] = y_toks
    ;   [rsp + 48] = cur_B
    ;   [rsp + 56] = cur_T
    ;   [rsp + 64] = cur_BT (cur_B * cur_T)
    ;   [rsp + 72] = compute_loss
    ;   [rsp + 80] = write_dlogits
    ;   [rsp + 88] = layer_x_stride (max_BT * D * 4)
    ;   [rsp + 96] = layer idx l
    ;   [rsp + 104]= cur_layer_w (float* base of layer l params)
    ;   [rsp + 112]= cur_layer_a (void*  base of layer l LACT)
    ;   [rsp + 120]= cur_x_in    (layer_x[l])
    ;   [rsp + 128]= cur_x_out   (layer_x[l+1])
    mov     [rsp + 32], rdi
    mov     [rsp + 40], rsi
    mov     [rsp + 48], rdx
    mov     [rsp + 56], rcx
    mov     rax, rdx
    imul    rax, rcx
    mov     [rsp + 64], rax             ; cur_BT
    mov     [rsp + 72], r8
    mov     [rsp + 80], r9

    lea     rbx, [rel g_trn]
    mov     r12, [rbx + TRN_DIM]        ; D
    mov     r13, [rbx + TRN_HIDDEN]     ; FF

    mov     rax, [rbx + TRN_BT]
    imul    rax, r12
    shl     rax, 2
    mov     [rsp + 88], rax             ; layer_x_stride

    ; 1. Embedding lookup: layer_x[0][n, :] = tok_emb[x_toks[n], :]
    mov     r8,  [rsp + 32]             ; x_toks
    mov     r9,  [rbx + TRN_PARAMS]     ; tok_emb
    mov     r10, [rbx + TRN_BUF_LAYER_X]; dst = layer_x[0]
    mov     r11, [rsp + 64]             ; cur_BT
    mov     r14, r12
    shl     r14, 2                      ; D_bytes
    xor     r15d, r15d                  ; n = 0
.emb_lp:
    cmp     r15, r11
    jae     .emb_done
    mov     eax, [r8 + r15*4]           ; tid
    imul    rax, r14
    lea     rsi, [r9 + rax]             ; src = tok_emb[tid]
    xor     ecx, ecx
.emb_cp:
    cmp     rcx, r14
    jae     .emb_next
    vmovups ymm0, [rsi + rcx]
    vmovups [r10 + rcx], ymm0
    add     rcx, 32
    jmp     .emb_cp
.emb_next:
    add     r10, r14
    inc     r15
    jmp     .emb_lp
.emb_done:
    vzeroupper

    ; 2. Transformer layers l = 0 .. L-1
    mov     qword [rsp + 96], 0
.layer_lp:
    mov     rax, [rsp + 96]
    cmp     rax, [rbx + TRN_LAYERS]
    jae     .layers_done

    ; cur_layer_w = params + (POFF_LAYERS + l * LAYER_PARAMS) * 4
    mov     rcx, rax
    imul    rcx, [rbx + TRN_LAYER_PARAMS]
    add     rcx, [rbx + TRN_POFF_LAYERS]
    shl     rcx, 2
    add     rcx, [rbx + TRN_PARAMS]
    mov     [rsp + 104], rcx

    ; cur_layer_a = LACT + l * LACT_STRIDE
    mov     rcx, rax
    imul    rcx, [rbx + TRN_LACT_STRIDE]
    add     rcx, [rbx + TRN_BUF_LACT]
    mov     [rsp + 112], rcx

    ; cur_x_in = layer_x[l], cur_x_out = layer_x[l+1]
    imul    rax, [rsp + 88]
    add     rax, [rbx + TRN_BUF_LAYER_X]
    mov     [rsp + 120], rax
    add     rax, [rsp + 88]
    mov     [rsp + 128], rax

    ; Helper: get layer weight ptr -> reg = cur_layer_w + LOFF * 4
    ;         get layer act ptr    -> reg = cur_layer_a + OFF

    ; 2a. ln1: ln1_out, ln1_inv = rmsnorm(x_in, W_ln1)
    mov     r14, [rsp + 112]            ; cur_layer_a
    mov     r15, [rsp + 104]            ; cur_layer_w
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN1_OUT]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN1_INV]
    mov     rdx, [rsp + 120]            ; x_in
    mov     rcx, r15                    ; W_ln1 (LOFF=0)
    mov     r8,  [rsp + 64]             ; cur_BT
    mov     r9,  r12                    ; D
    vmovss  xmm0, [rbx + TRN_RMS_EPS]
    call    tp_rmsnorm_fwd

    ; 2b. Q = ln1_out @ W_q^T
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN1_OUT]
    mov     rsi, [rbx + TRN_LOFF_WQ]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_Q]
    mov     rcx, [rsp + 64]             ; M = cur_BT
    mov     r8,  r12                    ; N = D
    mov     r9,  r12                    ; K = D
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    ; 2c. K = ln1_out @ W_k^T
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN1_OUT]
    mov     rsi, [rbx + TRN_LOFF_WK]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_K]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    ; 2d. V = ln1_out @ W_v^T
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN1_OUT]
    mov     rsi, [rbx + TRN_LOFF_WV]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_V]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    ; 2e. Split-half RoPE on Q and K (in-place)
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_Q]
    mov     rsi, [rbx + TRN_BUF_ROPE_COS]
    mov     rdx, [rbx + TRN_BUF_ROPE_SIN]
    mov     rcx, [rsp + 48]             ; cur_B
    mov     r8,  [rsp + 56]             ; cur_T
    mov     r9,  [rbx + TRN_HEADS]
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 0], rax
    mov     qword [rsp + 8], 0          ; is_bwd = 0
    call    tp_rope_batch

    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_K]
    mov     rsi, [rbx + TRN_BUF_ROPE_COS]
    mov     rdx, [rbx + TRN_BUF_ROPE_SIN]
    mov     rcx, [rsp + 48]
    mov     r8,  [rsp + 56]
    mov     r9,  [rbx + TRN_HEADS]
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 0], rax
    mov     qword [rsp + 8], 0
    call    tp_rope_batch

    ; 2f. Causal Multi-Head Self-Attention
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_Q]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_K]
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_V]
    mov     rcx, r14
    add     rcx, [rbx + TRN_OFF_PROBS]
    mov     r8,  r14
    add     r8,  [rbx + TRN_OFF_ATT_OUT]
    mov     r9,  [rsp + 48]             ; cur_B
    mov     rax, [rsp + 56]
    mov     [rsp + 0], rax              ; cur_T
    mov     rax, [rbx + TRN_HEADS]
    mov     [rsp + 8], rax              ; H
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 16], rax             ; hd
    vmovss  xmm0, [rbx + TRN_ATT_SCALE]
    call    tp_mha_fwd

    ; 2g. O-proj + residual: BUF_DTMP = att_out @ W_o^T; x_mid = x_in + BUF_DTMP
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_ATT_OUT]
    mov     rsi, [rbx + TRN_LOFF_WO]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_X_MID]
    mov     rsi, [rsp + 120]            ; x_in
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    imul    rcx, r12                    ; cur_BT * D
    call    vec_add3_f32_avx2

    ; 2h. ln2: ln2_out, ln2_inv = rmsnorm(x_mid, W_ln2)
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN2_OUT]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN2_INV]
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_X_MID]
    mov     rcx, [rbx + TRN_LOFF_LN2]
    shl     rcx, 2
    add     rcx, r15
    mov     r8,  [rsp + 64]
    mov     r9,  r12
    vmovss  xmm0, [rbx + TRN_RMS_EPS]
    call    tp_rmsnorm_fwd

    ; 2i. gate = ln2_out @ W_gate^T, up = ln2_out @ W_up^T
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN2_OUT]
    mov     rsi, [rbx + TRN_LOFF_WGATE]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_GATE]
    mov     rcx, [rsp + 64]             ; M = cur_BT
    mov     r8,  r13                    ; N = FF
    mov     r9,  r12                    ; K = D
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_LN2_OUT]
    mov     rsi, [rbx + TRN_LOFF_WUP]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_UP]
    mov     rcx, [rsp + 64]
    mov     r8,  r13
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    ; 2j. SwiGLU: act = silu(gate) * up
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_ACT]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_GATE]
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_UP]
    mov     rcx, [rsp + 64]
    imul    rcx, r13                    ; cur_BT * FF
    call    tp_swiglu_fwd

    ; 2k. down_proj + residual: BUF_DTMP = act @ W_down^T; x_out = x_mid + BUF_DTMP
    mov     rdi, r14
    add     rdi, [rbx + TRN_OFF_ACT]
    mov     rsi, [rbx + TRN_LOFF_WDOWN]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]             ; M = cur_BT
    mov     r8,  r12                    ; N = D
    mov     r9,  r13                    ; K = FF
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    mov     rdi, [rsp + 128]            ; layer_x[l+1]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_X_MID]
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    imul    rcx, r12
    call    vec_add3_f32_avx2

    inc     qword [rsp + 96]
    jmp     .layer_lp

.layers_done:
    ; 3. Final RMSNorm: final_out, final_inv = rmsnorm(layer_x[L], W_final_ln)
    mov     rax, [rbx + TRN_LAYERS]
    imul    rax, [rsp + 88]
    add     rax, [rbx + TRN_BUF_LAYER_X] ; layer_x[L]

    mov     rdi, [rbx + TRN_BUF_FINAL_OUT]
    mov     rsi, [rbx + TRN_BUF_FINAL_INV]
    mov     rdx, rax
    mov     rcx, [rbx + TRN_POFF_FINAL_LN]
    shl     rcx, 2
    add     rcx, [rbx + TRN_PARAMS]
    mov     r8,  [rsp + 64]             ; cur_BT
    mov     r9,  r12                    ; D
    vmovss  xmm0, [rbx + TRN_RMS_EPS]
    call    tp_rmsnorm_fwd

    ; 4. LM head: logits = final_out @ W_lm_head^T
    mov     rdi, [rbx + TRN_BUF_FINAL_OUT]
    mov     rsi, [rbx + TRN_POFF_LM_HEAD]
    shl     rsi, 2
    add     rsi, [rbx + TRN_PARAMS]
    mov     rdx, [rbx + TRN_BUF_LOGITS]
    mov     rcx, [rsp + 64]             ; M = cur_BT
    mov     r8,  [rbx + TRN_VOCAB]      ; N = V
    mov     r9,  r12                    ; K = D
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    mov     rax, [rbx + TRN_BUF_WT]
    mov     [rsp + 8], rax
    call    tp_sgemm_nt

    ; 5. Cross-entropy loss (+ optional d_logits)
    vxorps  xmm0, xmm0, xmm0
    cmp     qword [rsp + 72], 0
    je      .ret
    mov     rdi, [rbx + TRN_BUF_LOGITS]
    mov     rsi, [rbx + TRN_BUF_LOSSES]
    mov     rdx, [rsp + 40]             ; y_toks
    mov     rcx, [rsp + 64]             ; cur_BT
    mov     r8,  [rbx + TRN_VOCAB]
    mov     r9,  [rsp + 80]             ; write_dlogits
    call    tp_cross_entropy_fwd_bwd

.ret:
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; void gpt_backward(const uint32 *x_toks)
; Exact analytical backward pass matching PyTorch autograd.
; Assumes BUF_LOGITS holds d_logits from gpt_forward(..., write_dlogits=1).
; ============================================================================
gpt_backward:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136                    ; 6*8 + 136 + 8 = 192 (16B aligned)
    ; Outgoing stack args: [rsp + 0..47]
    ; Locals:
    ;   [rsp + 56] = x_toks
    ;   [rsp + 64] = BT
    ;   [rsp + 72] = layer_x_stride (BT * D * 4)
    ;   [rsp + 80] = l (signed, L-1 down to 0)
    ;   [rsp + 88] = cur_w  (params base of layer l)
    ;   [rsp + 96] = cur_dw (grads  base of layer l)
    ;   [rsp + 104]= cur_a  (LACT   base of layer l)
    ;   [rsp + 112]= cur_x_in (layer_x[l])
    mov     [rsp + 56], rdi
    lea     rbx, [rel g_trn]
    mov     r12, [rbx + TRN_DIM]        ; D
    mov     r13, [rbx + TRN_HIDDEN]     ; FF
    mov     rax, [rbx + TRN_BT]
    mov     [rsp + 64], rax
    imul    rax, r12
    shl     rax, 2
    mov     [rsp + 72], rax             ; layer_x_stride

    ; 0. Zero all parameter gradients
    mov     rdi, [rbx + TRN_GRADS]
    mov     rsi, [rbx + TRN_N_PARAMS]
    call    zero_f32_avx2

    ; 1. LM head backward:
    ;    dW_lm_head += d_logits^T @ final_out  (M=V, N=D, K=BT)
    mov     rdi, [rbx + TRN_BUF_LOGITS]
    mov     rsi, [rbx + TRN_BUF_FINAL_OUT]
    mov     rdx, [rbx + TRN_POFF_LM_HEAD]
    shl     rdx, 2
    add     rdx, [rbx + TRN_GRADS]
    mov     rcx, [rbx + TRN_VOCAB]      ; M = V
    mov     r8,  r12                    ; N = D
    mov     r9,  [rsp + 64]             ; K = BT
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ;    BUF_DTMP = d_logits @ W_lm_head  (M=BT, N=D, K=V)
    mov     rdi, [rbx + TRN_BUF_LOGITS]
    mov     rsi, [rbx + TRN_POFF_LM_HEAD]
    shl     rsi, 2
    add     rsi, [rbx + TRN_PARAMS]
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]             ; M = BT
    mov     r8,  r12                    ; N = D
    mov     r9,  [rbx + TRN_VOCAB]      ; K = V
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    call    tp_sgemm

    ; 2. Final RMSNorm backward -> BUF_DX, dW_final_ln
    mov     rax, [rbx + TRN_LAYERS]
    imul    rax, [rsp + 72]
    add     rax, [rbx + TRN_BUF_LAYER_X]; layer_x[L]
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, [rbx + TRN_POFF_FINAL_LN]
    shl     rsi, 2
    mov     r9,  rsi
    add     rsi, [rbx + TRN_GRADS]      ; dW_final_ln
    add     r9,  [rbx + TRN_PARAMS]     ; W_final_ln
    mov     rdx, [rbx + TRN_BUF_DTMP]   ; dy
    mov     rcx, rax                    ; x = layer_x[L]
    mov     r8,  [rbx + TRN_BUF_FINAL_INV]
    mov     rax, [rsp + 64]
    mov     [rsp + 0], rax              ; N = BT
    mov     [rsp + 8], r12              ; D
    call    tp_rmsnorm_bwd

    ; 3. Reverse layer loop: l = L - 1 down to 0
    mov     rax, [rbx + TRN_LAYERS]
    dec     rax
    mov     [rsp + 80], rax

.bwd_layer_lp:
    mov     rax, [rsp + 80]
    test    rax, rax
    js      .emb_bwd

    ; Compute cur_w, cur_dw, cur_a, cur_x_in
    mov     rcx, rax
    imul    rcx, [rbx + TRN_LAYER_PARAMS]
    add     rcx, [rbx + TRN_POFF_LAYERS]
    shl     rcx, 2
    mov     rdx, rcx
    add     rcx, [rbx + TRN_PARAMS]
    add     rdx, [rbx + TRN_GRADS]
    mov     [rsp + 88], rcx             ; cur_w
    mov     [rsp + 96], rdx             ; cur_dw

    mov     rcx, rax
    imul    rcx, [rbx + TRN_LACT_STRIDE]
    add     rcx, [rbx + TRN_BUF_LACT]
    mov     [rsp + 104], rcx            ; cur_a

    imul    rax, [rsp + 72]
    add     rax, [rbx + TRN_BUF_LAYER_X]
    mov     [rsp + 112], rax            ; layer_x[l]

    mov     r14, [rsp + 104]            ; cur_a
    mov     r15, [rsp + 88]             ; cur_w
    mov     rbp, [rsp + 96]             ; cur_dw

    ; --- 3a. MLP Backward (BUF_DX = d(layer_x[l+1])) ---
    ; dW_down += BUF_DX^T @ act  (M=D, N=FF, K=BT)
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_ACT]
    mov     rdx, [rbx + TRN_LOFF_WDOWN]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r12                    ; M = D
    mov     r8,  r13                    ; N = FF
    mov     r9,  [rsp + 64]             ; K = BT
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ; BUF_DACT = BUF_DX @ W_down  (M=BT, N=FF, K=D)
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, [rbx + TRN_LOFF_WDOWN]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DACT]
    mov     rcx, [rsp + 64]             ; M = BT
    mov     r8,  r13                    ; N = FF
    mov     r9,  r12                    ; K = D
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    call    tp_sgemm

    ; SwiGLU backward -> BUF_DGATE, BUF_DUP
    mov     rdi, [rbx + TRN_BUF_DGATE]
    mov     rsi, [rbx + TRN_BUF_DUP]
    mov     rdx, [rbx + TRN_BUF_DACT]
    mov     rcx, r14
    add     rcx, [rbx + TRN_OFF_GATE]
    mov     r8,  r14
    add     r8,  [rbx + TRN_OFF_UP]
    mov     r9,  [rsp + 64]
    imul    r9,  r13                    ; BT * FF
    call    tp_swiglu_bwd

    ; dW_gate += BUF_DGATE^T @ ln2_out  (M=FF, N=D, K=BT)
    mov     rdi, [rbx + TRN_BUF_DGATE]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN2_OUT]
    mov     rdx, [rbx + TRN_LOFF_WGATE]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r13                    ; M = FF
    mov     r8,  r12                    ; N = D
    mov     r9,  [rsp + 64]             ; K = BT
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ; dW_up += BUF_DUP^T @ ln2_out  (M=FF, N=D, K=BT)
    mov     rdi, [rbx + TRN_BUF_DUP]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN2_OUT]
    mov     rdx, [rbx + TRN_LOFF_WUP]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r13
    mov     r8,  r12
    mov     r9,  [rsp + 64]
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ; BUF_DTMP = BUF_DGATE @ W_gate + BUF_DUP @ W_up  (M=BT, N=D, K=FF)
    mov     rdi, [rbx + TRN_BUF_DGATE]
    mov     rsi, [rbx + TRN_LOFF_WGATE]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r13
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    call    tp_sgemm

    mov     rdi, [rbx + TRN_BUF_DUP]
    mov     rsi, [rbx + TRN_LOFF_WUP]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r13
    mov     qword [rsp + 0], SGEMM_BETA_ONE
    call    tp_sgemm

    ; RMSNorm2 backward: BUF_DQ, dW_ln2 += rmsnorm_bwd(BUF_DTMP, x_mid, ln2_inv, W_ln2)
    mov     rdi, [rbx + TRN_BUF_DQ]
    mov     rsi, [rbx + TRN_LOFF_LN2]
    shl     rsi, 2
    mov     r9,  rsi
    add     rsi, rbp                    ; dW_ln2
    add     r9,  r15                    ; W_ln2
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, r14
    add     rcx, [rbx + TRN_OFF_X_MID]
    mov     r8,  r14
    add     r8,  [rbx + TRN_OFF_LN2_INV]
    mov     rax, [rsp + 64]
    mov     [rsp + 0], rax
    mov     [rsp + 8], r12
    call    tp_rmsnorm_bwd

    ; Residual add: BUF_DX += BUF_DQ  -> now BUF_DX = d(x_mid)
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, [rbx + TRN_BUF_DQ]
    mov     rdx, [rsp + 64]
    imul    rdx, r12
    call    vec_add_f32_avx2

    ; --- 3b. Attention Backward (BUF_DX = d(x_mid)) ---
    ; dW_o += BUF_DX^T @ att_out  (M=D, N=D, K=BT)
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_ATT_OUT]
    mov     rdx, [rbx + TRN_LOFF_WO]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r12
    mov     r8,  r12
    mov     r9,  [rsp + 64]
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ; BUF_DTMP = BUF_DX @ W_o  (M=BT, N=D, K=D)
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, [rbx + TRN_LOFF_WO]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    call    tp_sgemm

    ; Causal MHA backward -> BUF_DQ, BUF_DK, BUF_DV
    mov     rdi, [rbx + TRN_BUF_DTMP]   ; d_out
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_Q]
    mov     rdx, r14
    add     rdx, [rbx + TRN_OFF_K]
    mov     rcx, r14
    add     rcx, [rbx + TRN_OFF_V]
    mov     r8,  r14
    add     r8,  [rbx + TRN_OFF_PROBS]
    mov     r9,  [rbx + TRN_BUF_DQ]
    mov     rax, [rbx + TRN_BUF_DK]
    mov     [rsp + 0],  rax             ; dK
    mov     rax, [rbx + TRN_BUF_DV]
    mov     [rsp + 8],  rax             ; dV
    mov     rax, [rbx + TRN_BATCH]
    mov     [rsp + 16], rax             ; B
    mov     rax, [rbx + TRN_SEQ]
    mov     [rsp + 24], rax             ; T
    mov     rax, [rbx + TRN_HEADS]
    mov     [rsp + 32], rax             ; H
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 40], rax             ; hd
    vmovss  xmm0, [rbx + TRN_ATT_SCALE]
    call    tp_mha_bwd

    ; RoPE backward on dQ and dK (in-place, angle = -theta)
    mov     rdi, [rbx + TRN_BUF_DQ]
    mov     rsi, [rbx + TRN_BUF_ROPE_COS]
    mov     rdx, [rbx + TRN_BUF_ROPE_SIN]
    mov     rcx, [rbx + TRN_BATCH]
    mov     r8,  [rbx + TRN_SEQ]
    mov     r9,  [rbx + TRN_HEADS]
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 0], rax
    mov     qword [rsp + 8], 1          ; is_bwd = 1
    call    tp_rope_batch

    mov     rdi, [rbx + TRN_BUF_DK]
    mov     rsi, [rbx + TRN_BUF_ROPE_COS]
    mov     rdx, [rbx + TRN_BUF_ROPE_SIN]
    mov     rcx, [rbx + TRN_BATCH]
    mov     r8,  [rbx + TRN_SEQ]
    mov     r9,  [rbx + TRN_HEADS]
    mov     rax, [rbx + TRN_HEAD_DIM]
    mov     [rsp + 0], rax
    mov     qword [rsp + 8], 1
    call    tp_rope_batch

    ; dW_q, dW_k, dW_v += d{Q,K,V}^T @ ln1_out  (M=D, N=D, K=BT)
    mov     rdi, [rbx + TRN_BUF_DQ]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN1_OUT]
    mov     rdx, [rbx + TRN_LOFF_WQ]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r12
    mov     r8,  r12
    mov     r9,  [rsp + 64]
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    mov     rdi, [rbx + TRN_BUF_DK]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN1_OUT]
    mov     rdx, [rbx + TRN_LOFF_WK]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r12
    mov     r8,  r12
    mov     r9,  [rsp + 64]
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    mov     rdi, [rbx + TRN_BUF_DV]
    mov     rsi, r14
    add     rsi, [rbx + TRN_OFF_LN1_OUT]
    mov     rdx, [rbx + TRN_LOFF_WV]
    shl     rdx, 2
    add     rdx, rbp
    mov     rcx, r12
    mov     r8,  r12
    mov     r9,  [rsp + 64]
    mov     qword [rsp + 0], SGEMM_TRANS_A | SGEMM_BETA_ONE
    call    tp_sgemm

    ; BUF_DTMP = dQ @ W_q + dK @ W_k + dV @ W_v  (M=BT, N=D, K=D)
    mov     rdi, [rbx + TRN_BUF_DQ]
    mov     rsi, [rbx + TRN_LOFF_WQ]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ZERO
    call    tp_sgemm

    mov     rdi, [rbx + TRN_BUF_DK]
    mov     rsi, [rbx + TRN_LOFF_WK]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ONE
    call    tp_sgemm

    mov     rdi, [rbx + TRN_BUF_DV]
    mov     rsi, [rbx + TRN_LOFF_WV]
    shl     rsi, 2
    add     rsi, r15
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 64]
    mov     r8,  r12
    mov     r9,  r12
    mov     qword [rsp + 0], SGEMM_BETA_ONE
    call    tp_sgemm

    ; RMSNorm1 backward: BUF_DQ, dW_ln1 += rmsnorm_bwd(BUF_DTMP, layer_x[l], ln1_inv, W_ln1)
    mov     rdi, [rbx + TRN_BUF_DQ]
    mov     rsi, rbp                    ; dW_ln1 (LOFF=0)
    mov     r9,  r15                    ; W_ln1  (LOFF=0)
    mov     rdx, [rbx + TRN_BUF_DTMP]
    mov     rcx, [rsp + 112]            ; layer_x[l]
    mov     r8,  r14
    add     r8,  [rbx + TRN_OFF_LN1_INV]
    mov     rax, [rsp + 64]
    mov     [rsp + 0], rax
    mov     [rsp + 8], r12
    call    tp_rmsnorm_bwd

    ; Residual add: BUF_DX += BUF_DQ -> now BUF_DX = d(layer_x[l])
    mov     rdi, [rbx + TRN_BUF_DX]
    mov     rsi, [rbx + TRN_BUF_DQ]
    mov     rdx, [rsp + 64]
    imul    rdx, r12
    call    vec_add_f32_avx2

    dec     qword [rsp + 80]
    jmp     .bwd_layer_lp

.emb_bwd:
    ; 4. Embedding scatter-add: d_tok_emb[x_toks[n]] += BUF_DX[n]
    ; Sequential over n = 0 .. BT-1 -> race-free & bit-for-bit deterministic!
    mov     r8,  [rsp + 56]             ; x_toks
    mov     r9,  [rbx + TRN_GRADS]      ; d_tok_emb (offset 0)
    mov     r10, [rbx + TRN_BUF_DX]     ; src
    mov     r11, [rsp + 64]             ; BT
    mov     r14, r12
    shl     r14, 2                      ; D_bytes
    xor     r15d, r15d                  ; n = 0
.sc_lp:
    cmp     r15, r11
    jae     .done
    mov     eax, [r8 + r15*4]           ; tid
    imul    rax, r14
    lea     rdi, [r9 + rax]             ; dst = d_tok_emb[tid]
    xor     ecx, ecx
.sc_add:
    cmp     rcx, r14
    jae     .sc_next
    vmovups ymm0, [rdi + rcx]
    vaddps  ymm0, ymm0, [r10 + rcx]
    vmovups [rdi + rcx], ymm0
    add     rcx, 32
    jmp     .sc_add
.sc_next:
    add     r10, r14
    inc     r15
    jmp     .sc_lp

.done:
    vzeroupper
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; void gpt_adamw_step(float lr)
; xmm0 = lr
; 1. Global L2 grad clipping (matching PyTorch clip_grad_norm_: clip / (norm + 1e-6))
; 2. Selective decoupled AdamW (wd on 2D weights, 0.0 on 1D RMSNorm weights)
; ============================================================================
gpt_adamw_step:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                     ; 6*8 + 40 + 8 = 96 (16B aligned)
    ; [rsp + 0]  = lr (f32)
    ; [rsp + 4]  = inv_bc1 (f32)
    ; [rsp + 8]  = inv_bc2 (f32)
    ; [rsp + 16] = layer idx l
    ; [rsp + 24] = cur_byte_off
    vmovss  [rsp + 0], xmm0
    lea     rbx, [rel g_trn]
    vmovss  [rbx + TRN_LAST_LR], xmm0

    ; 1. Global L2 norm & clipping
    mov     rdi, [rbx + TRN_GRADS]
    mov     rsi, [rbx + TRN_N_PARAMS]
    call    tp_grad_norm                ; xmm0 = total_norm
    vmovss  [rbx + TRN_LAST_GNORM], xmm0

    vmovss  xmm1, [rbx + TRN_GRAD_CLIP]
    vxorps  xmm2, xmm2, xmm2
    vucomiss xmm1, xmm2
    jbe     .no_clip
    vaddss  xmm0, xmm0, [rel mf_clip_eps]; total_norm + 1e-6
    vdivss  xmm1, xmm1, xmm0            ; clip_coef = max_norm / (total_norm + 1e-6)
    vmovss  xmm2, [rel mf_one]
    vucomiss xmm1, xmm2
    jae     .no_clip
    mov     rdi, [rbx + TRN_GRADS]
    mov     rsi, [rbx + TRN_N_PARAMS]
    vmovss  xmm0, xmm1, xmm1
    call    tp_vec_scale
.no_clip:

    ; 2. Advance FP64 beta1^t, beta2^t -> inv_bc1, inv_bc2
    vmovsd  xmm0, [rbx + TRN_B1_POW]
    vmulsd  xmm0, xmm0, [rel md_beta1]
    vmovsd  [rbx + TRN_B1_POW], xmm0
    vmovsd  xmm2, [rel md_one]
    vsubsd  xmm1, xmm2, xmm0            ; 1 - beta1^t
    vdivsd  xmm1, xmm2, xmm1            ; inv_bc1
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [rsp + 4], xmm1

    vmovsd  xmm0, [rbx + TRN_B2_POW]
    vmulsd  xmm0, xmm0, [rel md_beta2]
    vmovsd  [rbx + TRN_B2_POW], xmm0
    vsubsd  xmm1, xmm2, xmm0            ; 1 - beta2^t
    vdivsd  xmm1, xmm2, xmm1            ; inv_bc2
    vcvtsd2ss xmm1, xmm1, xmm1
    vmovss  [rsp + 8], xmm1

    ; Helper macro to call tp_adamw_range at float offset [rsp + 24], count %1, wd %2
    %macro ADAMW_SPAN 2
        ; %1 = count reg, %2 = wd_reg_load
        mov     rax, [rsp + 24]
        shl     rax, 2                  ; byte offset
        mov     rdi, [rbx + TRN_PARAMS]
        add     rdi, rax
        mov     rsi, [rbx + TRN_GRADS]
        add     rsi, rax
        mov     rdx, [rbx + TRN_OPT_M]
        add     rdx, rax
        mov     rcx, [rbx + TRN_OPT_V]
        add     rcx, rax
        mov     r8,  %1
        vmovss  xmm0, [rsp + 0]
        %2
        vmovss  xmm2, [rsp + 4]
        vmovss  xmm3, [rsp + 8]
        call    tp_adamw_range
        add     [rsp + 24], %1
    %endmacro

    mov     r12, [rbx + TRN_DIM]        ; D
    mov     r13, [rbx + TRN_HIDDEN]     ; FF
    mov     r14, r12
    imul    r14, r12
    shl     r14, 2                      ; 4 * D * D (wq, wk, wv, wo)
    mov     r15, r12
    imul    r15, r13
    imul    r15, r15, 3                 ; 3 * D * FF (wgate, wup, wdown)

    mov     qword [rsp + 24], 0         ; float offset = 0

    ; tok_emb [V * D] (2D -> decay)
    mov     rbp, [rbx + TRN_VOCAB]
    imul    rbp, r12
    ADAMW_SPAN rbp, {vmovss xmm1, [rbx + TRN_WD]}

    ; Layers 0 .. L-1
    mov     qword [rsp + 16], 0
.lp:
    mov     rax, [rsp + 16]
    cmp     rax, [rbx + TRN_LAYERS]
    jae     .tail_spans

    ; ln1 [D] (1D -> wd=0)
    ADAMW_SPAN r12, {vxorps xmm1, xmm1, xmm1}
    ; wq, wk, wv, wo [4*D*D] (2D -> decay)
    ADAMW_SPAN r14, {vmovss xmm1, [rbx + TRN_WD]}
    ; ln2 [D] (1D -> wd=0)
    ADAMW_SPAN r12, {vxorps xmm1, xmm1, xmm1}
    ; wgate, wup, wdown [3*D*FF] (2D -> decay)
    ADAMW_SPAN r15, {vmovss xmm1, [rbx + TRN_WD]}

    inc     qword [rsp + 16]
    jmp     .lp

.tail_spans:
    ; final_ln [D] (1D -> wd=0)
    ADAMW_SPAN r12, {vxorps xmm1, xmm1, xmm1}
    ; lm_head [V * D] (2D -> decay)
    ADAMW_SPAN rbp, {vmovss xmm1, [rbx + TRN_WD]}

    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; void gpt_sample(const char *prompt_cstr, uint64 max_gen,
;                 float temp, float topp, uint64 *sample_rng)
; rdi=prompt, rsi=max_gen, xmm0=temp, xmm1=topp, rdx=sample_rng
; Decodes \n escapes in prompt, echoes prompt to stdout, generates max_gen
; characters autoregressively with sliding window T, streaming to stdout.
; ============================================================================
gpt_sample:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40                     ; 6*8+40+8 = 96 (16B aligned)
    ; [rsp + 0]  = max_gen
    ; [rsp + 8]  = sample_rng
    ; [rsp + 16] = temp (f32)
    ; [rsp + 20] = topp (f32)
    ; [rsp + 24] = byte scratch (1B)
    mov     [rsp + 0],  rsi
    mov     [rsp + 8],  rdx
    vmovss  [rsp + 16], xmm0
    vmovss  [rsp + 20], xmm1

    lea     rbx, [rel g_trn]
    lea     r12, [rel sample_tok_buf]   ; uint32[4096]
    xor     r13d, r13d                  ; tok_len = 0

    ; Encode prompt_cstr (supporting literal "\n" -> 0x0A) and echo to stdout
    mov     r14, rdi
    test    r14, r14
    jz      .chk_empty
.p_lp:
    movzx   eax, byte [r14]
    test    al, al
    jz      .chk_empty
    cmp     al, '\'
    jne     .p_emit
    cmp     byte [r14 + 1], 'n'
    jne     .p_emit
    mov     al, 10                      ; '\n'
    inc     r14
.p_emit:
    inc     r14
    ; Echo byte to stdout (or capture into TUI sample buffer when active)
    movzx   r15d, al                    ; save raw byte
    cmp     qword [rel g_sample_capture], 0
    je      .p_stdout
    mov     edi, r15d
    call    tui_sample_push_byte
    jmp     .p_map
.p_stdout:
    mov     [rsp + 24], al
    mov     edi, 1
    lea     rsi, [rsp + 24]
    mov     edx, 1
    call    sys_write_all
.p_map:

    ; Map byte -> token ID via byte_to_id[byte]
    movzx   eax, byte [rbx + TRN_BYTE_TO_ID + r15]
    cmp     eax, 0xFF
    jne     .p_id_ok
    xor     eax, eax                    ; fallback to token 0
.p_id_ok:
    cmp     r13, 4096
    jae     .p_lp
    mov     [r12 + r13*4], eax
    inc     r13
    jmp     .p_lp

.chk_empty:
    test    r13, r13
    jnz     .gen_start
    mov     dword [r12], 0              ; seed with token 0 ('\n')
    mov     r13, 1

.gen_start:
    xor     r14d, r14d                  ; gen_i = 0
.gen_lp:
    cmp     r14, [rsp + 0]
    jae     .done

    ; cur_T = min(tok_len, seq_len), start = tok_len - cur_T
    mov     rcx, r13
    cmp     rcx, [rbx + TRN_SEQ]
    jbe     .t_ok
    mov     rcx, [rbx + TRN_SEQ]
.t_ok:
    mov     r15, rcx                    ; save cur_T
    mov     rax, r13
    sub     rax, rcx                    ; start
    lea     rdi, [r12 + rax*4]          ; x_toks = &buf[start]
    xor     esi, esi                    ; y_toks = NULL
    mov     edx, 1                      ; B = 1
    ; rcx = cur_T
    xor     r8d, r8d                    ; compute_loss = 0
    xor     r9d, r9d                    ; write_dlogits = 0
    call    gpt_forward

    ; last_logits = BUF_LOGITS + (cur_T - 1) * V * 4
    lea     rax, [r15 - 1]
    imul    rax, [rbx + TRN_VOCAB]
    shl     rax, 2
    add     rax, [rbx + TRN_BUF_LOGITS]

    ; Sample or greedy argmax
    vmovss  xmm0, [rsp + 16]            ; temp
    vxorps  xmm2, xmm2, xmm2
    vucomiss xmm0, xmm2
    ja      .do_sample
    mov     rdi, rax
    mov     rsi, [rbx + TRN_VOCAB]
    call    argmax_f32
    jmp     .got_tok
.do_sample:
    mov     rdi, rax                    ; logits
    mov     rsi, [rbx + TRN_BUF_SAMPLE_P]; probs
    mov     rdx, [rbx + TRN_VOCAB]      ; vocab
    vmovss  xmm1, [rsp + 20]            ; topp
    mov     rcx, [rsp + 8]              ; &sample_rng
    call    sample_top_p

.got_tok:
    mov     rbp, rax                    ; next_tok
    ; Append to sample_tok_buf (slide left by 1 if full at 4096)
    cmp     r13, 4096
    jb      .append
    mov     rcx, 0
.shift_lp:
    cmp     rcx, 4095
    jae     .shift_done
    mov     eax, [r12 + rcx*4 + 4]
    mov     [r12 + rcx*4], eax
    inc     rcx
    jmp     .shift_lp
.shift_done:
    mov     r13, 4095
.append:
    mov     [r12 + r13*4], ebp
    inc     r13

    ; Decode token -> byte and stream to stdout (or TUI sample buffer)
    movzx   eax, byte [rbx + TRN_ID_TO_BYTE + rbp]
    cmp     qword [rel g_sample_capture], 0
    je      .g_stdout
    mov     edi, eax
    call    tui_sample_push_byte
    jmp     .g_next
.g_stdout:
    mov     [rsp + 24], al
    mov     edi, 1
    lea     rsi, [rsp + 24]
    mov     edx, 1
    call    sys_write_all
.g_next:

    inc     r14
    jmp     .gen_lp

.done:
    add     rsp, 40
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
mf_one:      dd 1.0
mf_init_std: dd 0.02
mf_clip_eps: dd 1.0e-6

align 16
md_one:      dq 1.0
md_beta1:    dq 0.9
md_beta2:    dq 0.999

section .bss
alignb 64
g_trn:          resb TRN_SIZE

alignb 64
sample_tok_buf: resd 4096
