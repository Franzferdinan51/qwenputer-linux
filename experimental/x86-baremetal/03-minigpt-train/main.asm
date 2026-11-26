; ============================================================================
; main.asm — Phase 3 Pure x86-64 Mini-GPT Training & Inference CLI (_start)
;
; Zero libc, zero libm, zero dynamic libraries.
; Features:
;   - Auto char-level vocabulary discovery (sorted unique bytes -> id_to_byte/byte_to_id)
;   - Automatic 90%/10% train/val split with deterministic validation eval
;   - Full AdamW training loop with linear warmup + cosine LR decay & grad clip
;   - Live train/val loss, grad norm, lr, step ms, tok/s & periodic text sampling
;   - Binary checkpoint save/load (.gptckpt) + --sample-only --prompt "..." mode
;   - --dump-debug <dir> for automated per-tensor PyTorch CPU gradient & AdamW check
; ============================================================================

bits 64
default rel

%include "train_constants.inc"

extern g_trn
extern sys_openat_ro
extern sys_openat_wr
extern sys_close
extern sys_fstat
extern sys_mmap_file
extern sys_mmap_anon
extern sys_munmap
extern sys_write_all
extern sys_clock_ns
extern sys_exit_group
extern str_len
extern str_eq
extern parse_u64
extern write_cstr
extern write_u64
extern write_f64_2dp

extern tp_init
extern tp_shutdown
extern rng_next_u64
extern compute_lr
extern gpt_init_layout
extern gpt_init_weights
extern gpt_free
extern gpt_forward
extern gpt_backward
extern gpt_adamw_step
extern gpt_sample
extern tui_isatty_stdout
extern tui_init
extern tui_record_train
extern tui_record_val
extern tui_sample_begin
extern tui_sample_end
extern tui_render
extern tui_finish

section .text

global _start

; ============================================================================
; ENTRYPOINT (_start)
; ============================================================================
_start:
    mov     r12, [rsp]                  ; argc
    lea     r13, [rsp + 8]              ; argv
    and     rsp, -16

    ; Initialize default configuration in g_trn and CLI opts
    lea     rbp, [rel g_trn]
    mov     qword [rbp + TRN_DIM], 192
    mov     qword [rbp + TRN_LAYERS], 6
    mov     qword [rbp + TRN_HEADS], 6
    mov     qword [rbp + TRN_HIDDEN], 512
    mov     qword [rbp + TRN_SEQ], 128
    mov     qword [rbp + TRN_BATCH], 16
    mov     qword [rbp + TRN_STEPS], 1500
    mov     qword [rbp + TRN_WARMUP], 0xFFFFFFFFFFFFFFFF ; sentinel = auto
    mov     qword [rbp + TRN_SAMPLE_ONLY], 0
    mov     qword [rbp + TRN_RNG], 42

    mov     eax, 0x3727C5AC             ; 1.0e-5f (rms_eps)
    mov     [rbp + TRN_RMS_EPS], eax
    mov     eax, 0x461C4000             ; 10000.0f (rope_theta)
    mov     [rbp + TRN_ROPE_THETA], eax
    mov     eax, 0x3F800000             ; 1.0f (grad_clip)
    mov     [rbp + TRN_GRAD_CLIP], eax
    mov     eax, 0x3A83126F             ; 0.001f (lr)
    mov     [rbp + TRN_LR_MAX], eax
    mov     eax, 0x3DCCCCCD             ; 0.1f (wd)
    mov     [rbp + TRN_WD], eax

    mov     qword [rel opt_data], 0
    mov     qword [rel opt_save], 0
    mov     qword [rel opt_load], 0
    mov     qword [rel opt_dump_debug], 0
    lea     rax, [rel def_prompt]
    mov     [rel opt_prompt], rax
    mov     qword [rel opt_threads], 16
    mov     qword [rel opt_eval_every], 100
    mov     qword [rel opt_eval_batches], 20
    mov     qword [rel opt_log_every], 50
    mov     qword [rel opt_sample_every], 250
    mov     qword [rel opt_sample_tokens], 200
    mov     qword [rel opt_seed], 42
    mov     qword [rel opt_tui_mode], 0 ; 0=auto (ioctl TCGETS), 1=--tui, 2=--plain
    mov     qword [rel tui_active], 0
    mov     eax, 0x3F4CCCCD             ; 0.8f (temp)
    mov     [rel opt_temp], eax
    mov     eax, 0x3F733333             ; 0.95f (topp)
    mov     [rel opt_topp], eax

    ; Parse argv[1 .. argc-1]
    mov     rbx, 1
.arg_lp:
    cmp     rbx, r12
    jae     .args_done
    mov     r14, [r13 + rbx*8]

    %macro MATCH_VAL_STR 2
        mov     rdi, r14
        lea     rsi, [rel %1]
        call    str_eq
        test    eax, eax
        jz      %%next
        inc     rbx
        cmp     rbx, r12
        jae     .usage
        mov     rax, [r13 + rbx*8]
        mov     %2, rax
        jmp     .arg_next
    %%next:
    %endmacro

    %macro MATCH_VAL_U64 2
        mov     rdi, r14
        lea     rsi, [rel %1]
        call    str_eq
        test    eax, eax
        jz      %%next
        inc     rbx
        cmp     rbx, r12
        jae     .usage
        mov     rdi, [r13 + rbx*8]
        call    parse_u64
        mov     %2, rax
        jmp     .arg_next
    %%next:
    %endmacro

    %macro MATCH_VAL_F32 2
        mov     rdi, r14
        lea     rsi, [rel %1]
        call    str_eq
        test    eax, eax
        jz      %%next
        inc     rbx
        cmp     rbx, r12
        jae     .usage
        mov     rdi, [r13 + rbx*8]
        call    parse_f32_sci
        vmovss  %2, xmm0
        jmp     .arg_next
    %%next:
    %endmacro

    MATCH_VAL_STR flg_data,          [rel opt_data]
    MATCH_VAL_STR flg_save,          [rel opt_save]
    MATCH_VAL_STR flg_load,          [rel opt_load]
    MATCH_VAL_STR flg_prompt,        [rel opt_prompt]
    MATCH_VAL_STR flg_dump_debug,    [rel opt_dump_debug]

    MATCH_VAL_U64 flg_steps,         [rbp + TRN_STEPS]
    MATCH_VAL_U64 flg_batch,         [rbp + TRN_BATCH]
    MATCH_VAL_U64 flg_seq,           [rbp + TRN_SEQ]
    MATCH_VAL_U64 flg_dim,           [rbp + TRN_DIM]
    MATCH_VAL_U64 flg_layers,        [rbp + TRN_LAYERS]
    MATCH_VAL_U64 flg_heads,         [rbp + TRN_HEADS]
    MATCH_VAL_U64 flg_hidden,        [rbp + TRN_HIDDEN]
    MATCH_VAL_U64 flg_warmup,        [rbp + TRN_WARMUP]
    MATCH_VAL_U64 flg_t,             [rel opt_threads]
    MATCH_VAL_U64 flg_seed,          [rel opt_seed]
    MATCH_VAL_U64 flg_eval_every,    [rel opt_eval_every]
    MATCH_VAL_U64 flg_eval_batches,  [rel opt_eval_batches]
    MATCH_VAL_U64 flg_log_every,     [rel opt_log_every]
    MATCH_VAL_U64 flg_sample_every,  [rel opt_sample_every]
    MATCH_VAL_U64 flg_sample_tokens, [rel opt_sample_tokens]

    MATCH_VAL_F32 flg_lr,            [rbp + TRN_LR_MAX]
    MATCH_VAL_F32 flg_wd,            [rbp + TRN_WD]
    MATCH_VAL_F32 flg_grad_clip,     [rbp + TRN_GRAD_CLIP]
    MATCH_VAL_F32 flg_temp,          [rel opt_temp]
    MATCH_VAL_F32 flg_topp,          [rel opt_topp]

    ; --sample-only
    mov     rdi, r14
    lea     rsi, [rel flg_sample_only]
    call    str_eq
    test    eax, eax
    jz      .chk_tui
    mov     qword [rbp + TRN_SAMPLE_ONLY], 1
    jmp     .arg_next

.chk_tui:
    mov     rdi, r14
    lea     rsi, [rel flg_tui]
    call    str_eq
    test    eax, eax
    jz      .chk_plain
    mov     qword [rel opt_tui_mode], 1
    jmp     .arg_next

.chk_plain:
    mov     rdi, r14
    lea     rsi, [rel flg_plain]
    call    str_eq
    test    eax, eax
    jz      .chk_help
    mov     qword [rel opt_tui_mode], 2
    jmp     .arg_next

.chk_help:
    mov     rdi, r14
    lea     rsi, [rel flg_h]
    call    str_eq
    test    eax, eax
    jnz     .usage
    mov     rdi, r14
    lea     rsi, [rel flg_help]
    call    str_eq
    test    eax, eax
    jnz     .usage

.arg_next:
    inc     rbx
    jmp     .arg_lp

.args_done:
    ; Set RNG seed (must be non-zero for xorshift64*)
    mov     rax, [rel opt_seed]
    test    rax, rax
    jnz     .seed_ok
    mov     rax, 1
.seed_ok:
    mov     [rbp + TRN_RNG], rax

    ; Resolve auto warmup if unset: min(100, steps / 10)
    mov     rax, [rbp + TRN_WARMUP]
    cmp     rax, 0xFFFFFFFFFFFFFFFF
    jne     .warmup_ok
    mov     rax, [rbp + TRN_STEPS]
    xor     edx, edx
    mov     rcx, 10
    div     rcx
    cmp     rax, 100
    jbe     .store_wu
    mov     rax, 100
.store_wu:
    mov     [rbp + TRN_WARMUP], rax
.warmup_ok:

    ; Initialize thread pool
    mov     rdi, [rel opt_threads]
    call    tp_init

    ; ========================================================================
    ; SAMPLE-ONLY MODE (--load <ckpt> --sample-only [--prompt "..."])
    ; ========================================================================
    cmp     qword [rbp + TRN_SAMPLE_ONLY], 0
    je      .train_mode

    mov     rdi, [rel opt_load]
    test    rdi, rdi
    jz      .err_need_load
    mov     qword [rbp + TRN_BATCH], 1
    call    load_checkpoint
    test    rax, rax
    js      .err_ckpt

    mov     rdi, [rel opt_prompt]
    mov     rsi, [rel opt_sample_tokens]
    vmovss  xmm0, [rel opt_temp]
    vmovss  xmm1, [rel opt_topp]
    lea     rdx, [rbp + TRN_RNG]
    call    gpt_sample

    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

    call    tp_shutdown
    call    gpt_free
    xor     edi, edi
    call    sys_exit_group

    ; ========================================================================
    ; TRAINING MODE
    ; ========================================================================
.train_mode:
    mov     rdi, [rel opt_data]
    test    rdi, rdi
    jz      .usage

    ; If --load provided, load checkpoint header + layout + weights first, then dataset
    mov     rax, [rel opt_load]
    test    rax, rax
    jz      .fresh_init

    mov     rdi, rax
    call    load_checkpoint
    test    rax, rax
    js      .err_ckpt

    mov     rdi, [rel opt_data]
    mov     esi, 1                      ; keep loaded vocab
    call    load_dataset
    test    rax, rax
    js      .err_data
    jmp     .ready_train

.fresh_init:
    mov     rdi, [rel opt_data]
    xor     esi, esi                    ; discover vocab from dataset
    call    load_dataset
    test    rax, rax
    js      .err_data

    call    gpt_init_layout
    test    rax, rax
    js      .err_oom

    call    gpt_init_weights

.ready_train:
    ; Decide whether live ANSI TUI is active:
    ;   --dump-debug -> always plain
    ;   --tui        -> force TUI on
    ;   --plain      -> force plain
    ;   default      -> auto-detect TTY via ioctl(1, TCGETS)
    mov     qword [rel tui_active], 0
    cmp     qword [rel opt_dump_debug], 0
    jne     .tui_decided
    mov     rax, [rel opt_tui_mode]
    cmp     rax, 1
    je      .tui_on
    cmp     rax, 2
    je      .tui_decided
    call    tui_isatty_stdout
    test    rax, rax
    jz      .tui_decided
.tui_on:
    mov     qword [rel tui_active], 1
.tui_decided:

    cmp     qword [rel tui_active], 0
    je      .plain_banner
    mov     rdi, [rel opt_threads]
    call    tui_init
    call    tui_render
    jmp     .chk_dump_init
.plain_banner:
    call    print_banner

.chk_dump_init:
    ; Optional: dump debug metadata + initial weights + fixed batch
    cmp     qword [rel opt_dump_debug], 0
    je      .pre_eval

    mov     rdi, [rel opt_dump_debug]
    call    sys_mkdir

    ; Sample the fixed batch used across all steps in --dump-debug mode
    xor     edi, edi                    ; is_val = 0
    lea     rsi, [rbp + TRN_RNG]
    call    sample_batch

    call    dump_debug_init

.pre_eval:
    ; Step 0 pre-training val eval + sample (only when NOT in --dump-debug)
    cmp     qword [rel opt_dump_debug], 0
    jne     .start_loop

    cmp     qword [rel opt_eval_every], 0
    je      .pre_sample
    mov     rdi, [rel opt_eval_batches]
    call    eval_val_loss               ; xmm0 = val_loss
    vmovss  [rel val_loss_tmp], xmm0
    cmp     qword [rel tui_active], 0
    je      .plain_init_val
    xor     edi, edi                    ; step 0
    vmovss  xmm0, [rel val_loss_tmp]
    call    tui_record_val
    call    tui_render
    jmp     .pre_sample
.plain_init_val:
    mov     edi, 1
    lea     rsi, [rel msg_init_val]
    call    write_cstr
    vcvtss2sd xmm0, xmm0, [rel val_loss_tmp]
    mov     edi, 1
    call    write_f64_4dp
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

.pre_sample:
    cmp     qword [rel opt_sample_every], 0
    je      .start_loop
    xor     edi, edi                    ; step 0
    call    print_sample_block

    ; ========================================================================
    ; MAIN TRAINING LOOP: step = 0 .. steps - 1
    ; ========================================================================
.start_loop:
    call    sys_clock_ns
    mov     [rel tm_train_start], rax

    xor     r12d, r12d                  ; step = 0
.train_lp:
    cmp     r12, [rbp + TRN_STEPS]
    jae     .train_done
    mov     [rbp + TRN_CUR_STEP], r12

    call    sys_clock_ns
    mov     [rel tm_step_start], rax

    ; Compute LR for this step
    mov     rdi, r12
    mov     rsi, [rbp + TRN_STEPS]
    mov     rdx, [rbp + TRN_WARMUP]
    vmovss  xmm0, [rbp + TRN_LR_MAX]
    call    compute_lr
    vmovss  [rel cur_lr], xmm0

    ; Sample batch (unless --dump-debug keeps fixed batch)
    cmp     qword [rel opt_dump_debug], 0
    jne     .do_fwd
    xor     edi, edi                    ; is_val = 0
    lea     rsi, [rbp + TRN_RNG]
    call    sample_batch

.do_fwd:
    mov     rdi, [rbp + TRN_BUF_BATCH_X]
    mov     rsi, [rbp + TRN_BUF_BATCH_Y]
    mov     rdx, [rbp + TRN_BATCH]
    mov     rcx, [rbp + TRN_SEQ]
    mov     r8d, 1                      ; compute_loss = 1
    mov     r9d, 1                      ; write_dlogits = 1
    call    gpt_forward
    vmovss  [rbp + TRN_LAST_LOSS], xmm0

    ; Backward pass
    mov     rdi, [rbp + TRN_BUF_BATCH_X]
    call    gpt_backward

    ; In --dump-debug mode: record loss/lr and dump step 0 raw gradients BEFORE clip/AdamW
    cmp     qword [rel opt_dump_debug], 0
    je      .do_opt
    cmp     r12, 4096
    jae     .chk_dbg_g0
    lea     rax, [rel dbg_losses]
    mov     ecx, [rbp + TRN_LAST_LOSS]
    mov     [rax + r12*4], ecx
    lea     rax, [rel dbg_lrs]
    mov     ecx, [rel cur_lr]
    mov     [rax + r12*4], ecx
.chk_dbg_g0:
    test    r12, r12
    jnz     .do_opt
    call    dump_debug_step0_grads

.do_opt:
    vmovss  xmm0, [rel cur_lr]
    call    gpt_adamw_step

    call    sys_clock_ns
    mov     [rel tm_step_end], rax

    ; Determine if we should log, eval, or sample on this step
    lea     r13, [r12 + 1]              ; step1 = step + 1
    xor     r14d, r14d                  ; do_eval = 0
    xor     r15d, r15d                  ; do_log = 0

    ; Always log step 0 and final step
    test    r12, r12
    jz      .set_log
    cmp     r13, [rbp + TRN_STEPS]
    je      .set_log
    mov     rcx, [rel opt_log_every]
    test    rcx, rcx
    jz      .chk_eval
    mov     rax, r13
    xor     edx, edx
    div     rcx
    test    rdx, rdx
    jnz     .chk_eval
.set_log:
    mov     r15d, 1

.chk_eval:
    cmp     qword [rel opt_dump_debug], 0
    jne     .maybe_log
    mov     rcx, [rel opt_eval_every]
    test    rcx, rcx
    jz      .maybe_log
    cmp     r13, [rbp + TRN_STEPS]
    je      .set_eval
    mov     rax, r13
    xor     edx, edx
    div     rcx
    test    rdx, rdx
    jnz     .maybe_log
.set_eval:
    mov     r14d, 1
    mov     r15d, 1

.maybe_log:
    ; Record metrics into TUI on every step when TUI is active
    cmp     qword [rel tui_active], 0
    je      .plain_log_path

    ; Evaluate val_loss if scheduled on this step
    test    r14d, r14d
    jz      .tui_rec_step
    mov     rdi, [rel opt_eval_batches]
    call    eval_val_loss
    vmovss  [rel val_loss_tmp], xmm0
    mov     rdi, r13                    ; step1
    vmovss  xmm0, [rel val_loss_tmp]
    call    tui_record_val

.tui_rec_step:
    mov     rdi, r12                    ; step0
    vmovss  xmm0, [rbp + TRN_LAST_LOSS]
    vmovss  xmm1, [rbp + TRN_LAST_GNORM]
    vmovss  xmm2, [rbp + TRN_LAST_LR]
    mov     rsi, [rel tm_step_end]
    sub     rsi, [rel tm_step_start]    ; step_ns
    mov     rdx, [rel tm_step_end]
    sub     rdx, [rel tm_train_start]   ; elapsed_ns
    call    tui_record_train
    call    tui_render
    jmp     .chk_sample

.plain_log_path:
    test    r15d, r15d
    jz      .chk_sample

    ; Optionally evaluate val_loss
    test    r14d, r14d
    jz      .print_step
    mov     rdi, [rel opt_eval_batches]
    call    eval_val_loss
    vmovss  [rel val_loss_tmp], xmm0

.print_step:
    mov     rdi, r13                    ; display 1-based step (or 1..steps)
    mov     esi, r14d                   ; has_val
    call    print_step_line

.chk_sample:
    cmp     qword [rel opt_dump_debug], 0
    jne     .next_step
    mov     rcx, [rel opt_sample_every]
    test    rcx, rcx
    jz      .next_step
    cmp     r13, [rbp + TRN_STEPS]
    je      .do_sample_blk
    mov     rax, r13
    xor     edx, edx
    div     rcx
    test    rdx, rdx
    jnz     .next_step
.do_sample_blk:
    mov     rdi, r13
    call    print_sample_block

.next_step:
    inc     r12
    jmp     .train_lp

.train_done:
    call    sys_clock_ns
    mov     [rel tm_train_end], rax

    ; If TUI is active, render final frame & restore cursor below row 24
    cmp     qword [rel tui_active], 0
    je      .chk_dbg_final
    call    tui_finish

.chk_dbg_final:
    ; Dump final debug artifacts if requested
    cmp     qword [rel opt_dump_debug], 0
    je      .chk_save
    call    dump_debug_final

.chk_save:
    ; Save checkpoint if --save given
    mov     rdi, [rel opt_save]
    test    rdi, rdi
    jz      .print_summary
    call    save_checkpoint
    test    rax, rax
    js      .err_save
    mov     edi, 1
    lea     rsi, [rel msg_saved]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel opt_save]
    call    write_cstr
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

.print_summary:
    ; Print total wall time & average throughput (steps * BT / wall_s)
    mov     rax, [rel tm_train_end]
    sub     rax, [rel tm_train_start]
    test    rax, rax
    jnz     .tm_ok
    mov     rax, 1
.tm_ok:
    vcvtsi2sd xmm0, xmm0, rax           ; dt_ns
    vmovsd  [rel tm_tmp], xmm0

    mov     edi, 1
    lea     rsi, [rel msg_done_1]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbp + TRN_STEPS]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_done_2]
    call    write_cstr
    vmovsd  xmm0, [rel tm_tmp]
    vdivsd  xmm0, xmm0, [rel md_1e9]    ; seconds
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_done_3]
    call    write_cstr

    ; avg step ms = dt_ns / (steps * 1e6)
    vmovsd  xmm0, [rel tm_tmp]
    vcvtsi2sd xmm1, xmm1, [rbp + TRN_STEPS]
    vmulsd  xmm1, xmm1, [rel md_1e6]
    vdivsd  xmm0, xmm0, xmm1
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_done_4]
    call    write_cstr

    ; avg train tok/s = (steps * BT * 1e9) / dt_ns
    mov     rax, [rbp + TRN_STEPS]
    imul    rax, [rbp + TRN_BT]
    vcvtsi2sd xmm0, xmm0, rax
    vmulsd  xmm0, xmm0, [rel md_1e9]
    vdivsd  xmm0, xmm0, [rel tm_tmp]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_done_5]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel opt_threads]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_close_nl]
    call    write_cstr

    call    tp_shutdown
    call    gpt_free
    xor     edi, edi
    call    sys_exit_group

.usage:
    mov     edi, 2
    lea     rsi, [rel msg_usage]
    call    write_cstr
    mov     edi, 1
    call    sys_exit_group

.err_need_load:
    mov     edi, 2
    lea     rsi, [rel msg_err_need_load]
    call    write_cstr
    mov     edi, 2
    call    sys_exit_group

.err_data:
    mov     edi, 2
    lea     rsi, [rel msg_err_data]
    call    write_cstr
    mov     edi, 3
    call    sys_exit_group

.err_ckpt:
    mov     edi, 2
    lea     rsi, [rel msg_err_ckpt]
    call    write_cstr
    mov     edi, 4
    call    sys_exit_group

.err_save:
    mov     edi, 2
    lea     rsi, [rel msg_err_save]
    call    write_cstr
    mov     edi, 5
    call    sys_exit_group

.err_oom:
    mov     edi, 2
    lea     rsi, [rel msg_err_oom]
    call    write_cstr
    mov     edi, 6
    call    sys_exit_group

; ============================================================================
; DATASET LOADER, VOCAB DISCOVERY, BATCH SAMPLER & VAL EVAL
; ============================================================================

; int64 load_dataset(const char *path, int keep_vocab)
load_dataset:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, STAT_SIZE + 280        ; 144 + 256(present) + 24 = 424 (6*8+424+8=480, 16B aligned)
    mov     r12d, esi                   ; keep_vocab
    lea     rbx, [rel g_trn]

    call    sys_openat_ro
    test    rax, rax
    js      .err
    mov     r13, rax                    ; fd

    mov     rdi, r13
    mov     rsi, rsp                    ; statbuf
    call    sys_fstat
    test    rax, rax
    js      .err_close
    mov     r14, [rsp + STAT_ST_SIZE]   ; data_size
    cmp     r14, 16
    jb      .err_close

    mov     rdi, r14
    mov     rsi, r13
    call    sys_mmap_file
    mov     r15, rax                    ; data_mmap
    mov     rdi, r13
    call    sys_close
    test    r15, r15
    js      .err

    mov     [rbx + TRN_DATA_MMAP], r15
    mov     [rbx + TRN_DATA_SIZE], r14

    test    r12d, r12d
    jnz     .encode

    ; Discover sorted unique bytes in dataset
    lea     rdi, [rsp + STAT_SIZE]      ; present[256]
    xor     eax, eax
    mov     ecx, 32
.clr_pres:
    mov     [rdi + rcx*8 - 8], rax
    dec     ecx
    jnz     .clr_pres

    xor     rcx, rcx
.scan_lp:
    cmp     rcx, r14
    jae     .build_vocab
    movzx   eax, byte [r15 + rcx]
    mov     byte [rdi + rax], 1
    inc     rcx
    jmp     .scan_lp

.build_vocab:
    ; Fill byte_to_id[0..255] = 0xFF
    xor     ecx, ecx
.fill_ff:
    cmp     ecx, 256
    jae     .assign_ids
    mov     byte [rbx + TRN_BYTE_TO_ID + rcx], 0xFF
    inc     ecx
    jmp     .fill_ff

.assign_ids:
    xor     ecx, ecx                    ; byte b = 0..255
    xor     edx, edx                    ; vocab_id v = 0
.id_lp:
    cmp     ecx, 256
    jae     .vocab_done
    cmp     byte [rdi + rcx], 0
    je      .id_next
    mov     [rbx + TRN_ID_TO_BYTE + rdx], cl
    mov     [rbx + TRN_BYTE_TO_ID + rcx], dl
    inc     edx
.id_next:
    inc     ecx
    jmp     .id_lp
.vocab_done:
    mov     [rbx + TRN_VOCAB], rdx

.encode:
    ; Allocate encoded token array (uint8[data_size])
    mov     rdi, r14
    call    sys_mmap_anon
    test    rax, rax
    js      .err
    mov     [rbx + TRN_TOK_IDS], rax

    xor     rcx, rcx
.enc_lp:
    cmp     rcx, r14
    jae     .split
    movzx   edx, byte [r15 + rcx]
    movzx   edx, byte [rbx + TRN_BYTE_TO_ID + rdx]
    cmp     dl, 0xFF
    jne     .enc_st
    xor     edx, edx
.enc_st:
    mov     [rax + rcx], dl
    inc     rcx
    jmp     .enc_lp

.split:
    ; 90% train / 10% val
    mov     rax, r14
    imul    rax, rax, 9
    xor     edx, edx
    mov     rcx, 10
    div     rcx
    mov     [rbx + TRN_TRAIN_LEN], rax
    mov     rcx, r14
    sub     rcx, rax
    mov     [rbx + TRN_VAL_LEN], rcx

    xor     eax, eax
    jmp     .ret
.err_close:
    mov     rdi, r13
    call    sys_close
.err:
    mov     rax, -1
.ret:
    add     rsp, STAT_SIZE + 280
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; void sample_batch(uint64 is_val, uint64 *rng_ptr)
; rdi = is_val (0=train, 1=val), rsi = rng_ptr
sample_batch:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                      ; 6*8+8+8=64 (16B aligned)

    lea     rbx, [rel g_trn]
    mov     rbp, rsi                    ; rng_ptr
    mov     r12, [rbx + TRN_TOK_IDS]
    mov     r13, [rbx + TRN_TRAIN_LEN]
    test    rdi, rdi
    jz      .src_ok
    add     r12, r13
    mov     r13, [rbx + TRN_VAL_LEN]
.src_ok:
    mov     r14, [rbx + TRN_SEQ]        ; T
    sub     r13, r14
    dec     r13                         ; max_start = len - T - 1
    cmp     r13, 1
    jge     .ms_ok
    mov     r13, 1
.ms_ok:
    mov     r8,  [rbx + TRN_BUF_BATCH_X]
    mov     r9,  [rbx + TRN_BUF_BATCH_Y]
    xor     r15d, r15d                  ; b = 0
.b_lp:
    cmp     r15, [rbx + TRN_BATCH]
    jae     .done
    mov     rdi, rbp
    call    rng_next_u64
    xor     edx, edx
    div     r13                         ; rdx = start in [0, max_start)
    lea     rsi, [r12 + rdx]            ; src ptr
    xor     ecx, ecx
.t_lp:
    cmp     rcx, r14
    jae     .b_next
    movzx   eax, byte [rsi + rcx]
    mov     [r8 + rcx*4], eax
    movzx   eax, byte [rsi + rcx + 1]
    mov     [r9 + rcx*4], eax
    inc     rcx
    jmp     .t_lp
.b_next:
    lea     rax, [r14 * 4]
    add     r8, rax
    add     r9, rax
    inc     r15
    jmp     .b_lp
.done:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; float eval_val_loss(uint64 n_batches) -> xmm0
; Deterministic validation evaluation (fixed local RNG seed every call!).
eval_val_loss:
    push    rbx
    push    rbp
    push    r12
    push    r13
    sub     rsp, 24                     ; 4*8+24+8 = 64 (16B aligned)
    ; [rsp + 0] = val_rng
    ; [rsp + 8] = sum_loss (f64)
    mov     r12, rdi                    ; n_batches
    test    r12, r12
    jnz     .nb_ok
    mov     r12, 1
.nb_ok:
    mov     rax, 0x123456789ABCDEF0
    mov     [rsp + 0], rax
    vxorpd  xmm0, xmm0, xmm0
    vmovsd  [rsp + 8], xmm0

    lea     rbx, [rel g_trn]
    xor     r13d, r13d
.lp:
    cmp     r13, r12
    jae     .done
    mov     edi, 1                      ; is_val = 1
    lea     rsi, [rsp + 0]              ; &val_rng
    call    sample_batch

    mov     rdi, [rbx + TRN_BUF_BATCH_X]
    mov     rsi, [rbx + TRN_BUF_BATCH_Y]
    mov     rdx, [rbx + TRN_BATCH]
    mov     rcx, [rbx + TRN_SEQ]
    mov     r8d, 1                      ; compute_loss = 1
    xor     r9d, r9d                    ; write_dlogits = 0
    call    gpt_forward
    vcvtss2sd xmm0, xmm0, xmm0
    vaddsd  xmm0, xmm0, [rsp + 8]
    vmovsd  [rsp + 8], xmm0
    inc     r13
    jmp     .lp
.done:
    vmovsd  xmm0, [rsp + 8]
    vcvtsi2sd xmm1, xmm1, r12
    vdivsd  xmm0, xmm0, xmm1
    vcvtsd2ss xmm0, xmm0, xmm0
    add     rsp, 24
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; CHECKPOINT SAVE / LOAD (.gptckpt)
; ============================================================================

; int64 save_checkpoint(const char *path)
save_checkpoint:
    push    rbx
    push    r12
    sub     rsp, CKPT_HDR_SIZE + 8      ; 72 + 16 + 8 = 96 (16B aligned)
    lea     rbx, [rel g_trn]

    call    sys_openat_wr
    test    rax, rax
    js      .err
    mov     r12, rax                    ; fd

    ; Zero 64B header
    xor     eax, eax
    mov     ecx, 8
.zh:
    mov     [rsp + rcx*8 - 8], rax
    dec     ecx
    jnz     .zh

    mov     dword [rsp + CHDR_MAGIC],      CKPT_MAGIC
    mov     dword [rsp + CHDR_VERSION],    CKPT_VERSION
    mov     eax, [rbx + TRN_VOCAB]
    mov     [rsp + CHDR_VOCAB], eax
    mov     eax, [rbx + TRN_DIM]
    mov     [rsp + CHDR_DIM], eax
    mov     eax, [rbx + TRN_LAYERS]
    mov     [rsp + CHDR_LAYERS], eax
    mov     eax, [rbx + TRN_HEADS]
    mov     [rsp + CHDR_HEADS], eax
    mov     eax, [rbx + TRN_HIDDEN]
    mov     [rsp + CHDR_HIDDEN], eax
    mov     eax, [rbx + TRN_SEQ]
    mov     [rsp + CHDR_SEQ], eax
    mov     rax, [rbx + TRN_CUR_STEP]
    inc     rax
    mov     [rsp + CHDR_STEP], rax
    mov     rax, [rbx + TRN_N_PARAMS]
    mov     [rsp + CHDR_N_PARAMS], rax
    mov     eax, [rbx + TRN_RMS_EPS]
    mov     [rsp + CHDR_RMS_EPS], eax
    mov     eax, [rbx + TRN_ROPE_THETA]
    mov     [rsp + CHDR_ROPE_THETA], eax

    ; Write 64B header
    mov     edi, r12d
    mov     rsi, rsp
    mov     edx, CKPT_HDR_SIZE
    call    sys_write_all
    test    rax, rax
    js      .err_close

    ; Write 512B vocab tables (id_to_byte + byte_to_id are contiguous in g_trn!)
    mov     edi, r12d
    lea     rsi, [rbx + TRN_ID_TO_BYTE]
    mov     edx, CKPT_VOCAB_SIZE_B
    call    sys_write_all
    test    rax, rax
    js      .err_close

    ; Write n_params * 4 bytes of FP32 weights
    mov     edi, r12d
    mov     rsi, [rbx + TRN_PARAMS]
    mov     rdx, [rbx + TRN_N_PARAMS]
    shl     rdx, 2
    call    sys_write_all
    test    rax, rax
    js      .err_close

    mov     rdi, r12
    call    sys_close
    xor     eax, eax
    jmp     .ret

.err_close:
    mov     rdi, r12
    call    sys_close
.err:
    mov     rax, -1
.ret:
    add     rsp, CKPT_HDR_SIZE + 8
    pop     r12
    pop     rbx
    ret

; int64 load_checkpoint(const char *path)
load_checkpoint:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, STAT_SIZE + 8          ; 4*8 + 152 + 8 = 192 (16B aligned)
    lea     rbx, [rel g_trn]

    call    sys_openat_ro
    test    rax, rax
    js      .err
    mov     r12, rax                    ; fd

    mov     rdi, r12
    mov     rsi, rsp
    call    sys_fstat
    test    rax, rax
    js      .err_close
    mov     r13, [rsp + STAT_ST_SIZE]   ; file size
    cmp     r13, CKPT_HDR_SIZE + CKPT_VOCAB_SIZE_B
    jb      .err_close

    mov     rdi, r13
    mov     rsi, r12
    call    sys_mmap_file
    mov     r14, rax                    ; base
    mov     rdi, r12
    call    sys_close
    test    r14, r14
    js      .err

    ; Verify header
    cmp     dword [r14 + CHDR_MAGIC], CKPT_MAGIC
    jne     .err_unmap
    cmp     dword [r14 + CHDR_VERSION], CKPT_VERSION
    jne     .err_unmap

    mov     eax, [r14 + CHDR_VOCAB]
    mov     [rbx + TRN_VOCAB], rax
    mov     eax, [r14 + CHDR_DIM]
    mov     [rbx + TRN_DIM], rax
    mov     eax, [r14 + CHDR_LAYERS]
    mov     [rbx + TRN_LAYERS], rax
    mov     eax, [r14 + CHDR_HEADS]
    mov     [rbx + TRN_HEADS], rax
    mov     eax, [r14 + CHDR_HIDDEN]
    mov     [rbx + TRN_HIDDEN], rax
    mov     eax, [r14 + CHDR_SEQ]
    mov     [rbx + TRN_SEQ], rax
    mov     eax, [r14 + CHDR_RMS_EPS]
    mov     [rbx + TRN_RMS_EPS], eax
    mov     eax, [r14 + CHDR_ROPE_THETA]
    mov     [rbx + TRN_ROPE_THETA], eax

    ; Copy 512B vocab tables
    lea     rdi, [rbx + TRN_ID_TO_BYTE]
    lea     rsi, [r14 + CKPT_HDR_SIZE]
    mov     ecx, 64
.cp_voc:
    mov     rax, [rsi + rcx*8 - 8]
    mov     [rdi + rcx*8 - 8], rax
    dec     ecx
    jnz     .cp_voc

    ; Initialize model layout matching checkpoint dims
    call    gpt_init_layout
    test    rax, rax
    js      .err_unmap

    ; Verify n_params & file size
    mov     rax, [rbx + TRN_N_PARAMS]
    cmp     rax, [r14 + CHDR_N_PARAMS]
    jne     .err_unmap
    shl     rax, 2                      ; weight bytes
    lea     rcx, [rax + CKPT_HDR_SIZE + CKPT_VOCAB_SIZE_B]
    cmp     r13, rcx
    jb      .err_unmap

    ; Copy weights into g_trn.params
    mov     rdi, [rbx + TRN_PARAMS]
    lea     rsi, [r14 + CKPT_HDR_SIZE + CKPT_VOCAB_SIZE_B]
    xor     rcx, rcx
.cp_w:
    cmp     rcx, rax
    jae     .cp_done
    mov     edx, [rsi + rcx]
    mov     [rdi + rcx], edx
    add     rcx, 4
    jmp     .cp_w
.cp_done:

    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    xor     eax, eax
    jmp     .ret

.err_unmap:
    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    jmp     .err
.err_close:
    mov     rdi, r12
    call    sys_close
.err:
    mov     rax, -1
.ret:
    add     rsp, STAT_SIZE + 8
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; DEBUG DUMP HELPERS (--dump-debug <dir> for PyTorch CPU Verification)
; ============================================================================

; void dump_debug_init(void) — writes meta.bin, init_weights.bin, batch_x.bin, batch_y.bin
dump_debug_init:
    push    rbx
    sub     rsp, 576                    ; 64B meta + 512B pathbuf (1*8+576+8=592)
    lea     rbx, [rel g_trn]

    ; Pack 64B meta struct at [rsp + 0]
    mov     eax, [rbx + TRN_VOCAB]
    mov     [rsp + 0], eax
    mov     eax, [rbx + TRN_DIM]
    mov     [rsp + 4], eax
    mov     eax, [rbx + TRN_LAYERS]
    mov     [rsp + 8], eax
    mov     eax, [rbx + TRN_HEADS]
    mov     [rsp + 12], eax
    mov     eax, [rbx + TRN_HIDDEN]
    mov     [rsp + 16], eax
    mov     eax, [rbx + TRN_SEQ]
    mov     [rsp + 20], eax
    mov     eax, [rbx + TRN_BATCH]
    mov     [rsp + 24], eax
    mov     eax, [rbx + TRN_STEPS]
    mov     [rsp + 28], eax
    mov     rax, [rbx + TRN_N_PARAMS]
    mov     [rsp + 32], rax
    mov     eax, [rbx + TRN_LR_MAX]
    mov     [rsp + 40], eax
    mov     eax, [rbx + TRN_WD]
    mov     [rsp + 44], eax
    mov     eax, [rbx + TRN_GRAD_CLIP]
    mov     [rsp + 48], eax
    mov     eax, [rbx + TRN_RMS_EPS]
    mov     [rsp + 52], eax
    mov     eax, [rbx + TRN_ROPE_THETA]
    mov     [rsp + 56], eax
    mov     eax, [rbx + TRN_WARMUP]
    mov     [rsp + 60], eax

    lea     rdi, [rsp + 64]
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_meta]
    call    join_path
    lea     rdi, [rsp + 64]
    mov     rsi, rsp
    mov     edx, 64
    call    write_file_raw

    lea     rdi, [rsp + 64]
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_init_w]
    call    join_path
    lea     rdi, [rsp + 64]
    mov     rsi, [rbx + TRN_PARAMS]
    mov     rdx, [rbx + TRN_N_PARAMS]
    shl     rdx, 2
    call    write_file_raw

    lea     rdi, [rsp + 64]
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_bx]
    call    join_path
    lea     rdi, [rsp + 64]
    mov     rsi, [rbx + TRN_BUF_BATCH_X]
    mov     rdx, [rbx + TRN_BT]
    shl     rdx, 2
    call    write_file_raw

    lea     rdi, [rsp + 64]
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_by]
    call    join_path
    lea     rdi, [rsp + 64]
    mov     rsi, [rbx + TRN_BUF_BATCH_Y]
    mov     rdx, [rbx + TRN_BT]
    shl     rdx, 2
    call    write_file_raw

    add     rsp, 576
    pop     rbx
    ret

dump_debug_step0_grads:
    push    rbx
    sub     rsp, 512                    ; 8+512+8 = 528 (16B aligned)
    lea     rbx, [rel g_trn]
    mov     rdi, rsp
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_g0]
    call    join_path
    mov     rdi, rsp
    mov     rsi, [rbx + TRN_GRADS]
    mov     rdx, [rbx + TRN_N_PARAMS]
    shl     rdx, 2
    call    write_file_raw
    add     rsp, 512
    pop     rbx
    ret

dump_debug_final:
    push    rbx
    sub     rsp, 512
    lea     rbx, [rel g_trn]

    mov     rdi, rsp
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_losses]
    call    join_path
    mov     rdi, rsp
    lea     rsi, [rel dbg_losses]
    mov     rdx, [rbx + TRN_STEPS]
    cmp     rdx, 4096
    jbe     .dl_ok
    mov     rdx, 4096
.dl_ok:
    shl     rdx, 2
    call    write_file_raw

    mov     rdi, rsp
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_lrs]
    call    join_path
    mov     rdi, rsp
    lea     rsi, [rel dbg_lrs]
    mov     rdx, [rbx + TRN_STEPS]
    cmp     rdx, 4096
    jbe     .dr_ok
    mov     rdx, 4096
.dr_ok:
    shl     rdx, 2
    call    write_file_raw

    mov     rdi, rsp
    mov     rsi, [rel opt_dump_debug]
    lea     rdx, [rel dbg_f_final_w]
    call    join_path
    mov     rdi, rsp
    mov     rsi, [rbx + TRN_PARAMS]
    mov     rdx, [rbx + TRN_N_PARAMS]
    shl     rdx, 2
    call    write_file_raw

    add     rsp, 512
    pop     rbx
    ret

; ============================================================================
; LOGGING, FORMATTING & TINY SYSCALL UTILITIES
; ============================================================================

print_banner:
    push    rbx
    lea     rbx, [rel g_trn]

    mov     edi, 1
    lea     rsi, [rel msg_hdr_1]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_LAYERS]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_d]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_DIM]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_h]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_HEADS]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_ff]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_HIDDEN]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_v]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_VOCAB]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_p]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_N_PARAMS]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

    mov     edi, 1
    lea     rsi, [rel msg_hdr_2]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_BATCH]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_t]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_SEQ]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_steps]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_STEPS]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_thr]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel opt_threads]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_tr]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_TRAIN_LEN]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_hdr_va]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_VAL_LEN]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

    pop     rbx
    ret

; void print_step_line(uint64 step1, int has_val)
print_step_line:
    push    rbx
    push    r12
    push    r13
    mov     r12, rdi                    ; step1
    mov     r13d, esi                   ; has_val
    lea     rbx, [rel g_trn]

    mov     edi, 1
    lea     rsi, [rel msg_step]
    call    write_cstr
    mov     edi, 1
    mov     rsi, r12
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_slash]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbx + TRN_STEPS]
    call    write_u64

    mov     edi, 1
    lea     rsi, [rel msg_loss]
    call    write_cstr
    vcvtss2sd xmm0, xmm0, [rbx + TRN_LAST_LOSS]
    mov     edi, 1
    call    write_f64_4dp

    test    r13d, r13d
    jz      .no_val
    mov     edi, 1
    lea     rsi, [rel msg_val]
    call    write_cstr
    vcvtss2sd xmm0, xmm0, [rel val_loss_tmp]
    mov     edi, 1
    call    write_f64_4dp
.no_val:

    mov     edi, 1
    lea     rsi, [rel msg_gnorm]
    call    write_cstr
    vcvtss2sd xmm0, xmm0, [rbx + TRN_LAST_GNORM]
    mov     edi, 1
    call    write_f64_2dp

    mov     edi, 1
    lea     rsi, [rel msg_lr]
    call    write_cstr
    vcvtss2sd xmm0, xmm0, [rbx + TRN_LAST_LR]
    mov     edi, 1
    call    write_f64_6dp

    ; Step time (ms) and tok/s
    mov     rax, [rel tm_step_end]
    sub     rax, [rel tm_step_start]
    test    rax, rax
    jnz     .st_ok
    mov     rax, 1
.st_ok:
    vcvtsi2sd xmm0, xmm0, rax
    vmovsd  [rel tm_tmp], xmm0

    mov     edi, 1
    lea     rsi, [rel msg_ms_sep]
    call    write_cstr
    vmovsd  xmm0, [rel tm_tmp]
    vdivsd  xmm0, xmm0, [rel md_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_ms_tps]
    call    write_cstr
    vcvtsi2sd xmm0, xmm0, [rbx + TRN_BT]
    vmulsd  xmm0, xmm0, [rel md_1e9]
    vdivsd  xmm0, xmm0, [rel tm_tmp]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_tps_nl]
    call    write_cstr

    pop     r13
    pop     r12
    pop     rbx
    ret

; void print_sample_block(uint64 step_num)
print_sample_block:
    push    rbx
    sub     rsp, 16                     ; 8 + 16 + 8 = 32 (16B aligned)
    mov     rbx, rdi
    ; Fixed seed per sample block so evolution is clean & reproducible
    mov     rax, 0x9E3779B97F4A7C15
    xor     rax, rbx
    or      rax, 1
    mov     [rsp + 0], rax

    cmp     qword [rel tui_active], 0
    je      .plain_sample

    ; TUI mode: capture sample into TUI Live Sample buffer & re-render!
    mov     rdi, rbx
    call    tui_sample_begin
    mov     rdi, [rel opt_prompt]
    mov     rsi, [rel opt_sample_tokens]
    vmovss  xmm0, [rel opt_temp]
    vmovss  xmm1, [rel opt_topp]
    mov     rdx, rsp
    call    gpt_sample
    call    tui_sample_end
    call    tui_render
    jmp     .done

.plain_sample:
    mov     edi, 1
    lea     rsi, [rel msg_samp_hdr]
    call    write_cstr
    mov     edi, 1
    mov     rsi, rbx
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_samp_hdr2]
    call    write_cstr

    mov     rdi, [rel opt_prompt]
    mov     rsi, [rel opt_sample_tokens]
    vmovss  xmm0, [rel opt_temp]
    vmovss  xmm1, [rel opt_topp]
    mov     rdx, rsp
    call    gpt_sample

    mov     edi, 1
    lea     rsi, [rel msg_samp_end]
    call    write_cstr

.done:
    add     rsp, 16
    pop     rbx
    ret

; int64 sys_mkdir(const char *path)
sys_mkdir:
    mov     esi, 0o755
    mov     eax, SYS_mkdir
    syscall
    ret

; void write_file_raw(const char *path, const void *buf, uint64 len)
write_file_raw:
    push    rbx
    push    r12
    push    r13
    mov     r12, rsi
    mov     r13, rdx
    call    sys_openat_wr
    test    rax, rax
    js      .done
    mov     rbx, rax
    mov     edi, ebx
    mov     rsi, r12
    mov     rdx, r13
    call    sys_write_all
    mov     rdi, rbx
    call    sys_close
.done:
    pop     r13
    pop     r12
    pop     rbx
    ret

; void join_path(char *out, const char *dir, const char *file)
join_path:
.cp_d:
    mov     al, [rsi]
    test    al, al
    jz      .sep
    mov     [rdi], al
    inc     rdi
    inc     rsi
    jmp     .cp_d
.sep:
    mov     byte [rdi], '/'
    inc     rdi
.cp_f:
    mov     al, [rdx]
    mov     [rdi], al
    inc     rdi
    inc     rdx
    test    al, al
    jnz     .cp_f
    ret

; float parse_f32_sci(const char *s) -> xmm0
; Supports optional '-', integer, optional '.frac', optional 'e'/'E'[+-]exp.
parse_f32_sci:
    vxorpd  xmm0, xmm0, xmm0
    vmovsd  xmm1, [rel md_ten]
    xor     ecx, ecx                    ; neg flag
    cmp     byte [rdi], '-'
    jne     .int_lp
    mov     ecx, 1
    inc     rdi
.int_lp:
    movzx   eax, byte [rdi]
    sub     eax, '0'
    cmp     eax, 9
    ja      .chk_dot
    vmulsd  xmm0, xmm0, xmm1
    vcvtsi2sd xmm2, xmm2, eax
    vaddsd  xmm0, xmm0, xmm2
    inc     rdi
    jmp     .int_lp
.chk_dot:
    cmp     byte [rdi], '.'
    jne     .chk_exp
    inc     rdi
    vmovsd  xmm3, [rel md_tenth]
.frac_lp:
    movzx   eax, byte [rdi]
    sub     eax, '0'
    cmp     eax, 9
    ja      .chk_exp
    vcvtsi2sd xmm2, xmm2, eax
    vmulsd  xmm2, xmm2, xmm3
    vaddsd  xmm0, xmm0, xmm2
    vmulsd  xmm3, xmm3, [rel md_tenth]
    inc     rdi
    jmp     .frac_lp
.chk_exp:
    movzx   eax, byte [rdi]
    cmp     al, 'e'
    je      .do_exp
    cmp     al, 'E'
    jne     .apply_sign
.do_exp:
    inc     rdi
    xor     r8d, r8d                    ; exp_neg = 0
    cmp     byte [rdi], '-'
    jne     .chk_plus
    mov     r8d, 1
    inc     rdi
    jmp     .exp_num
.chk_plus:
    cmp     byte [rdi], '+'
    jne     .exp_num
    inc     rdi
.exp_num:
    xor     r9d, r9d
.exp_lp:
    movzx   eax, byte [rdi]
    sub     eax, '0'
    cmp     eax, 9
    ja      .exp_apply
    imul    r9d, r9d, 10
    add     r9d, eax
    inc     rdi
    jmp     .exp_lp
.exp_apply:
    test    r9d, r9d
    jz      .apply_sign
    vmovsd  xmm4, [rel md_ten]
    test    r8d, r8d
    jz      .exp_mul
    vmovsd  xmm4, [rel md_tenth]
.exp_mul:
    vmulsd  xmm0, xmm0, xmm4
    dec     r9d
    jnz     .exp_mul
.apply_sign:
    test    ecx, ecx
    jz      .to_f32
    vxorpd  xmm0, xmm0, [rel md_signbit]
.to_f32:
    vcvtsd2ss xmm0, xmm0, xmm0
    ret

; void write_f64_4dp(int fd, double val)
write_f64_4dp:
    mov     rsi, 10000
    mov     edx, 4
    jmp     write_f64_ndp

; void write_f64_6dp(int fd, double val)
write_f64_6dp:
    mov     rsi, 1000000
    mov     edx, 6
    ; fallthrough

; void write_f64_ndp(int fd, uint64 scale, uint32 digits, double val)
write_f64_ndp:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 24                     ; 4*8+24+8 = 64 (16B aligned)
    mov     ebx, edi                    ; fd
    mov     r12, rsi                    ; scale (10^digits)
    mov     r13d, edx                   ; digits

    vcvtsi2sd xmm1, xmm1, r12
    vmulsd  xmm0, xmm0, xmm1
    vaddsd  xmm0, xmm0, [rel md_half]
    vcvttsd2si rax, xmm0
    test    rax, rax
    jns     .pos
    xor     eax, eax
.pos:
    xor     edx, edx
    div     r12                         ; rax = int, rdx = frac
    mov     r14, rdx
    mov     edi, ebx
    mov     rsi, rax
    call    write_u64

    mov     byte [rsp], '.'
    mov     rax, r14
    mov     ecx, r13d
    mov     r8, 10
.dig_lp:
    xor     edx, edx
    div     r8
    add     dl, '0'
    mov     [rsp + rcx], dl
    dec     ecx
    jnz     .dig_lp

    mov     edi, ebx
    mov     rsi, rsp
    lea     edx, [r13d + 1]
    call    sys_write_all

    add     rsp, 24
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; RODATA & BSS
; ============================================================================
section .rodata
align 16
md_ten:     dq 10.0
md_tenth:   dq 0.1
md_half:    dq 0.5
md_1e6:     dq 1000000.0
md_1e9:     dq 1000000000.0
md_signbit: dq 0x8000000000000000, 0

flg_data:          db "--data", 0
flg_steps:         db "--steps", 0
flg_batch:         db "--batch", 0
flg_seq:           db "--seq", 0
flg_dim:           db "--dim", 0
flg_layers:        db "--layers", 0
flg_heads:         db "--heads", 0
flg_hidden:        db "--hidden", 0
flg_lr:            db "--lr", 0
flg_wd:            db "--wd", 0
flg_grad_clip:     db "--grad-clip", 0
flg_warmup:        db "--warmup", 0
flg_seed:          db "--seed", 0
flg_t:             db "-t", 0
flg_eval_every:    db "--eval-every", 0
flg_eval_batches:  db "--eval-batches", 0
flg_log_every:     db "--log-every", 0
flg_sample_every:  db "--sample-every", 0
flg_sample_tokens: db "--sample-tokens", 0
flg_temp:          db "--temp", 0
flg_topp:          db "--topp", 0
flg_save:          db "--save", 0
flg_load:          db "--load", 0
flg_sample_only:   db "--sample-only", 0
flg_tui:           db "--tui", 0
flg_plain:         db "--plain", 0
flg_prompt:        db "--prompt", 0
flg_dump_debug:    db "--dump-debug", 0
flg_h:             db "-h", 0
flg_help:          db "--help", 0

def_prompt:        db 10, 0
dbg_f_meta:        db "meta.bin", 0
dbg_f_init_w:      db "init_weights.bin", 0
dbg_f_bx:          db "batch_x.bin", 0
dbg_f_by:          db "batch_y.bin", 0
dbg_f_g0:          db "step0_grads.bin", 0
dbg_f_losses:      db "losses.bin", 0
dbg_f_lrs:         db "lrs.bin", 0
dbg_f_final_w:     db "final_weights.bin", 0

msg_nl:            db 10
msg_hdr_1:         db "[asm-gpt-train] Pure x86-64 Mini-GPT (FP32 AVX2+FMA, zero libc) | L=", 0
msg_hdr_d:         db " D=", 0
msg_hdr_h:         db " H=", 0
msg_hdr_ff:        db " FF=", 0
msg_hdr_v:         db " V=", 0
msg_hdr_p:         db " params=", 0
msg_hdr_2:         db "[asm-gpt-train] B=", 0
msg_hdr_t:         db " T=", 0
msg_hdr_steps:     db " steps=", 0
msg_hdr_thr:       db " threads=", 0
msg_hdr_tr:        db " train_tok=", 0
msg_hdr_va:        db " val_tok=", 0
msg_init_val:      db "[asm-gpt-train] step 0 (init) | val_loss=", 0
msg_step:          db "step ", 0
msg_slash:         db "/", 0
msg_loss:          db " | train_loss=", 0
msg_val:           db " | val_loss=", 0
msg_gnorm:         db " | gnorm=", 0
msg_lr:            db " | lr=", 0
msg_ms_sep:        db " | ", 0
msg_ms_tps:        db " ms (", 0
msg_tps_nl:        db " tok/s)", 10, 0
msg_samp_hdr:      db "--- sample @ step ", 0
msg_samp_hdr2:     db " ---", 10, 0
msg_samp_end:      db 10, "--- end sample ---", 10, 0
msg_saved:         db "[asm-gpt-train] saved checkpoint: ", 0
msg_done_1:        db "[asm-gpt-train] done: ", 0
msg_done_2:        db " steps in ", 0
msg_done_3:        db " s (", 0
msg_done_4:        db " ms/step, ", 0
msg_done_5:        db " train tok/s, threads=", 0
msg_close_nl:      db ")", 10, 0

msg_err_need_load: db "error: --sample-only requires --load <ckpt>", 10, 0
msg_err_data:      db "error: failed to load dataset", 10, 0
msg_err_ckpt:      db "error: failed to load checkpoint", 10, 0
msg_err_save:      db "error: failed to save checkpoint", 10, 0
msg_err_oom:       db "error: mmap arena allocation failed", 10, 0

msg_usage:
    db "asm-gpt-train (100% pure x86-64 AVX2/FMA Mini-GPT trainer & sampler, zero libc)", 10
    db "Usage: asm-gpt-train --data <file> [options]", 10
    db '       asm-gpt-train --load <ckpt> --sample-only --prompt "ROMEO:\n" [options]', 10
    db "  --data <path>         Dataset text file (auto 90%/10% train/val split)", 10
    db "  --steps <N>           Training steps (default: 1500)", 10
    db "  --batch <B>           Batch size (default: 16)", 10
    db "  --seq <T>             Sequence length (default: 128)", 10
    db "  --dim <D>             Model dimension, mult of 16 (default: 192)", 10
    db "  --layers <L>          Transformer layers (default: 6)", 10
    db "  --heads <H>           Attention heads, head_dim mult of 16 (default: 6)", 10
    db "  --hidden <FF>         SwiGLU hidden dim, mult of 16 (default: 512)", 10
    db "  --lr <float>          Peak learning rate (default: 0.001)", 10
    db "  --wd <float>          AdamW weight decay on 2D weights (default: 0.1)", 10
    db "  --grad-clip <float>   Global L2 gradient clip (default: 1.0)", 10
    db "  --warmup <N>          Warmup steps (default: min(100, steps/10))", 10
    db "  --seed <uint>         PRNG seed (default: 42)", 10
    db "  -t <threads>          CPU thread count (default: 16)", 10
    db "  --eval-every <K>      Val loss eval interval (default: 100)", 10
    db "  --eval-batches <K>    Val eval batch count (default: 20)", 10
    db "  --log-every <K>       Step log interval (default: 50)", 10
    db "  --sample-every <K>    Live sample interval (default: 250)", 10
    db "  --sample-tokens <M>   Generated chars per sample (default: 200)", 10
    db "  --temp <float>        Sampling temperature, 0=greedy (default: 0.8)", 10
    db "  --topp <float>        Top-p threshold (default: 0.95)", 10
    db "  --save <path>         Save binary checkpoint (.gptckpt)", 10
    db "  --load <path>         Load binary checkpoint (.gptckpt)", 10
    db "  --sample-only         Generate from loaded checkpoint without training", 10
    db "  --prompt <str>        Sampling prompt (supports \n escapes)", 10
    db "  --tui                 Force live ANSI/UTF-8 TUI dashboard (auto on TTY)", 10
    db "  --plain               Force plain line-by-line log output (auto on pipe)", 10
    db "  --dump-debug <dir>    Dump weights/batch/grads/losses for PyTorch verify", 10, 0

section .bss
alignb 8
opt_data:          resq 1
opt_save:          resq 1
opt_load:          resq 1
opt_dump_debug:    resq 1
opt_prompt:        resq 1
opt_threads:       resq 1
opt_eval_every:    resq 1
opt_eval_batches:  resq 1
opt_log_every:     resq 1
opt_sample_every:  resq 1
opt_sample_tokens: resq 1
opt_seed:          resq 1
opt_tui_mode:      resq 1
tui_active:        resq 1
opt_temp:          resd 1
opt_topp:          resd 1
cur_lr:            resd 1
val_loss_tmp:      resd 1
tm_train_start:    resq 1
tm_train_end:      resq 1
tm_step_start:     resq 1
tm_step_end:       resq 1
tm_tmp:            resq 1

alignb 64
dbg_losses:        resd 4096
dbg_lrs:           resd 4096
