; ============================================================================
; main.asm — Pure x86-64 Linux ELF entrypoint (_start), CLI & Generation Loop
;
; Zero libc, zero libm, zero dynamic libraries.
; Build: nasm -f elf64 + ld --gc-sections -s
; ============================================================================

bits 64
default rel

%include "constants.inc"

extern g_ctx
extern sys_openat_wr
extern sys_close
extern sys_write_all
extern sys_clock_ns
extern sys_exit_group
extern str_len
extern str_eq
extern parse_u64
extern parse_f32
extern write_cstr
extern write_u64
extern write_f64_2dp

extern argmax_f32
extern sample_top_p
extern tp_init
extern tp_shutdown
extern tok_build_chat_prompt
extern tok_write_stdout
extern model_init
extern model_free
extern transformer_forward

section .text

global _start

_start:
    ; Linux x86-64 initial stack: [rsp] = argc, [rsp+8] = argv[0], ...
    mov     r12, [rsp]                  ; argc
    lea     r13, [rsp + 8]              ; argv
    and     rsp, -16                    ; 16-byte align stack for all calls

    ; Initialize default CLI options
    mov     qword [rel opt_model], 0
    lea     rax, [rel def_prompt]
    mov     [rel opt_prompt], rax
    mov     qword [rel opt_max_tok], 128
    mov     qword [rel opt_threads], 32
    mov     qword [rel opt_ctx_len], 2048
    mov     dword [rel opt_think], 0
    mov     dword [rel opt_raw], 0
    mov     dword [rel opt_chat], 0
    mov     dword [rel opt_has_prompt], 0
    mov     dword [rel opt_print_toks], 0
    mov     qword [rel g_chat_pos], 0
    mov     qword [rel opt_tokens_csv], 0
    mov     qword [rel opt_dump_logits], 0
    mov     dword [rel opt_temp], 0     ; 0.0f = greedy argmax
    mov     eax, 0x3F733333             ; 0.95f
    mov     [rel opt_topp], eax
    mov     qword [rel opt_rng], 42

    ; Parse argv[1 .. argc-1]
    mov     rbx, 1
.arg_lp:
    cmp     rbx, r12
    jae     .args_done

    mov     r14, [r13 + rbx*8]          ; arg = argv[rbx]

    ; -m <model_path>
    mov     rdi, r14
    lea     rsi, [rel flg_m]
    call    str_eq
    test    eax, eax
    jz      .chk_p
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rax, [r13 + rbx*8]
    mov     [rel opt_model], rax
    jmp     .arg_next

.chk_p:
    mov     rdi, r14
    lea     rsi, [rel flg_p]
    call    str_eq
    test    eax, eax
    jz      .chk_n
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rax, [r13 + rbx*8]
    mov     [rel opt_prompt], rax
    mov     dword [rel opt_has_prompt], 1
    jmp     .arg_next

.chk_n:
    mov     rdi, r14
    lea     rsi, [rel flg_n]
    call    str_eq
    test    eax, eax
    jz      .chk_t
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_u64
    mov     [rel opt_max_tok], rax
    jmp     .arg_next

.chk_t:
    mov     rdi, r14
    lea     rsi, [rel flg_t]
    call    str_eq
    test    eax, eax
    jz      .chk_c
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_u64
    mov     [rel opt_threads], rax
    jmp     .arg_next

.chk_c:
    mov     rdi, r14
    lea     rsi, [rel flg_c]
    call    str_eq
    test    eax, eax
    jz      .chk_think
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_u64
    mov     [rel opt_ctx_len], rax
    jmp     .arg_next

.chk_think:
    mov     rdi, r14
    lea     rsi, [rel flg_think]
    call    str_eq
    test    eax, eax
    jz      .chk_raw
    mov     dword [rel opt_think], 1
    jmp     .arg_next

.chk_raw:
    mov     rdi, r14
    lea     rsi, [rel flg_raw]
    call    str_eq
    test    eax, eax
    jz      .chk_tokens
    mov     dword [rel opt_raw], 1
    jmp     .arg_next

.chk_tokens:
    mov     rdi, r14
    lea     rsi, [rel flg_tokens]
    call    str_eq
    test    eax, eax
    jz      .chk_temp
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rax, [r13 + rbx*8]
    mov     [rel opt_tokens_csv], rax
    jmp     .arg_next

.chk_temp:
    mov     rdi, r14
    lea     rsi, [rel flg_temp]
    call    str_eq
    test    eax, eax
    jz      .chk_topp
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_f32
    vmovss  [rel opt_temp], xmm0
    jmp     .arg_next

.chk_topp:
    mov     rdi, r14
    lea     rsi, [rel flg_topp]
    call    str_eq
    test    eax, eax
    jz      .chk_seed
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_f32
    vmovss  [rel opt_topp], xmm0
    jmp     .arg_next

.chk_seed:
    mov     rdi, r14
    lea     rsi, [rel flg_seed]
    call    str_eq
    test    eax, eax
    jz      .chk_dump
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rdi, [r13 + rbx*8]
    call    parse_u64
    test    rax, rax
    jnz     .store_seed
    mov     rax, 1                      ; xorshift64* state must be non-zero
.store_seed:
    mov     [rel opt_rng], rax
    jmp     .arg_next

.chk_dump:
    mov     rdi, r14
    lea     rsi, [rel flg_dump]
    call    str_eq
    test    eax, eax
    jz      .chk_ptoks
    inc     rbx
    cmp     rbx, r12
    jae     .usage
    mov     rax, [r13 + rbx*8]
    mov     [rel opt_dump_logits], rax
    jmp     .arg_next

.chk_ptoks:
    mov     rdi, r14
    lea     rsi, [rel flg_ptoks]
    call    str_eq
    test    eax, eax
    jz      .chk_chat
    mov     dword [rel opt_print_toks], 1
    jmp     .arg_next

.chk_chat:
    mov     rdi, r14
    lea     rsi, [rel flg_chat]
    call    str_eq
    test    eax, eax
    jz      .chk_help
    mov     dword [rel opt_chat], 1
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
    mov     rdi, [rel opt_model]
    test    rdi, rdi
    jz      .usage

    ; Load & mmap model
    mov     rsi, [rel opt_ctx_len]
    call    model_init
    test    rax, rax
    js      .err_model

    ; Start thread pool
    mov     rdi, [rel opt_threads]
    call    tp_init

    lea     rbp, [rel g_ctx]
    cmp     dword [rel opt_chat], 0
    jne     .enter_chat

    ; ========================================================================
    ; SINGLE-SHOT MODE (100% identical to original behavior)
    ; ========================================================================
    mov     rax, [rel opt_tokens_csv]
    test    rax, rax
    jnz     .parse_csv_toks

    mov     rdi, [rel opt_prompt]
    mov     rsi, [rbp + CTX_BUF_PROMPT]
    mov     rdx, MAX_PROMPT_TOKENS
    mov     ecx, [rel opt_raw]
    mov     r8d, [rel opt_think]
    call    tok_build_chat_prompt
    mov     r12, rax                    ; r12 = n_prompt
    jmp     .have_prompt

.parse_csv_toks:
    mov     rsi, rax                    ; csv string ptr
    mov     rdi, [rbp + CTX_BUF_PROMPT]
    xor     r12d, r12d                  ; count
.csv_lp:
    cmp     byte [rsi], 0
    je      .have_prompt
    cmp     r12, MAX_PROMPT_TOKENS
    jae     .have_prompt
    xor     eax, eax
.csv_num:
    movzx   edx, byte [rsi]
    sub     edx, '0'
    cmp     edx, 9
    ja      .csv_sep
    imul    rax, rax, 10
    add     rax, rdx
    inc     rsi
    jmp     .csv_num
.csv_sep:
    mov     [rdi + r12*4], eax
    inc     r12
    cmp     byte [rsi], ','
    jne     .csv_chk_end
    inc     rsi
    jmp     .csv_lp
.csv_chk_end:
    cmp     byte [rsi], 0
    je      .have_prompt
    inc     rsi
    jmp     .csv_lp

.have_prompt:
    test    r12, r12
    jz      .err_prompt
    mov     rdi, r12
    call    run_tokens
    jmp     .shutdown

    ; ========================================================================
    ; INTERACTIVE MULTI-TURN CHAT MODE (--chat, zero-libc raw stdin/stdout)
    ; ========================================================================
.enter_chat:
    mov     edi, 2
    lea     rsi, [rel msg_chat_help]
    call    write_cstr

    ; If user also passed -p "...", run it as Turn 1 and keep its KV-cache
    cmp     dword [rel opt_has_prompt], 0
    je      .chat_loop
    mov     rdi, [rel opt_prompt]
    mov     rsi, [rbp + CTX_BUF_PROMPT]
    mov     rdx, MAX_PROMPT_TOKENS
    mov     ecx, [rel opt_raw]
    mov     r8d, [rel opt_think]
    call    tok_build_chat_prompt
    test    rax, rax
    jz      .chat_loop
    mov     rdi, rax
    call    run_tokens

.chat_loop:
    mov     edi, 2
    lea     rsi, [rel msg_chat_prompt]
    call    write_cstr

    lea     rdi, [rel g_line_buf]
    mov     rsi, 4096
    call    read_stdin_line
    test    rax, rax
    js      .shutdown                   ; EOF / Ctrl-D

    ; Trim leading spaces/tabs
    lea     r12, [rel g_line_buf]
.trim_lead:
    mov     al, [r12]
    cmp     al, ' '
    je      .adv_lead
    cmp     al, 9
    je      .adv_lead
    jmp     .trim_tail
.adv_lead:
    inc     r12
    jmp     .trim_lead

.trim_tail:
    mov     rdi, r12
    call    str_len
    test    rax, rax
    jz      .chat_loop                  ; empty line
    lea     rcx, [r12 + rax - 1]
.tt_lp:
    cmp     rcx, r12
    jb      .chk_chat_cmd
    mov     al, [rcx]
    cmp     al, ' '
    je      .tt_chop
    cmp     al, 9
    je      .tt_chop
    jmp     .chk_chat_cmd
.tt_chop:
    mov     byte [rcx], 0
    dec     rcx
    jmp     .tt_lp

.chk_chat_cmd:
    ; /quit, /exit, exit
    mov     rdi, r12
    lea     rsi, [rel cmd_quit]
    call    str_eq
    test    eax, eax
    jnz     .shutdown

    mov     rdi, r12
    lea     rsi, [rel cmd_exit_slash]
    call    str_eq
    test    eax, eax
    jnz     .shutdown

    mov     rdi, r12
    lea     rsi, [rel cmd_exit]
    call    str_eq
    test    eax, eax
    jnz     .shutdown

    ; /clear, /new, /reset
    mov     rdi, r12
    lea     rsi, [rel cmd_clear]
    call    str_eq
    test    eax, eax
    jnz     .do_clear

    mov     rdi, r12
    lea     rsi, [rel cmd_new]
    call    str_eq
    test    eax, eax
    jnz     .do_clear

    mov     rdi, r12
    lea     rsi, [rel cmd_reset]
    call    str_eq
    test    eax, eax
    jnz     .do_clear

    ; /think
    mov     rdi, r12
    lea     rsi, [rel cmd_think]
    call    str_eq
    test    eax, eax
    jz      .chk_cmd_nothink
    mov     dword [rel opt_think], 1
    mov     edi, 2
    lea     rsi, [rel msg_think_on]
    call    write_cstr
    jmp     .chat_loop

.chk_cmd_nothink:
    mov     rdi, r12
    lea     rsi, [rel cmd_nothink]
    call    str_eq
    test    eax, eax
    jz      .chat_infer
    mov     dword [rel opt_think], 0
    mov     edi, 2
    lea     rsi, [rel msg_think_off]
    call    write_cstr
    jmp     .chat_loop

.do_clear:
    mov     qword [rel g_chat_pos], 0
    mov     edi, 2
    lea     rsi, [rel msg_chat_cleared]
    call    write_cstr
    mov     edi, 2
    mov     rsi, [rbp + CTX_MAX_SEQ]
    call    write_u64
    mov     edi, 2
    lea     rsi, [rel msg_close_nl]
    call    write_cstr
    jmp     .chat_loop

.chat_infer:
    mov     rdi, r12
    mov     rsi, [rbp + CTX_BUF_PROMPT]
    mov     rdx, MAX_PROMPT_TOKENS
    mov     ecx, [rel opt_raw]
    mov     r8d, [rel opt_think]
    call    tok_build_chat_prompt
    test    rax, rax
    jz      .chat_loop
    mov     rdi, rax
    call    run_tokens
    jmp     .chat_loop

.shutdown:
    call    tp_shutdown
    call    model_free
    xor     edi, edi
    call    sys_exit_group

.usage:
    mov     edi, 2
    lea     rsi, [rel msg_usage]
    call    write_cstr
    mov     edi, 1
    call    sys_exit_group

.err_model:
    mov     edi, 2
    lea     rsi, [rel msg_err_model]
    call    write_cstr
    mov     edi, 2
    call    sys_exit_group

.err_prompt:
    mov     edi, 2
    lea     rsi, [rel msg_err_prompt]
    call    write_cstr
    mov     edi, 3
    call    sys_exit_group

; ============================================================================
; void run_tokens(uint64 n_prompt in rdi)
; Runs prefill (from g_chat_pos) + streaming decode + optional KV-cache close
; ============================================================================
run_tokens:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                      ; 6 pushes + 8 + ret(8) = 64 -> 16B aligned

    mov     r12, rdi                    ; r12 = n_prompt
    lea     rbp, [rel g_ctx]

    ; Multi-turn KV-cache bounds check (chat mode only)
    cmp     dword [rel opt_chat], 0
    je      .ctx_room_ok
    mov     rax, [rel g_chat_pos]
    test    rax, rax
    jz      .ctx_room_ok
    lea     rcx, [rax + r12 + 64]
    cmp     rcx, [rbp + CTX_MAX_SEQ]
    jb      .ctx_room_ok
    mov     qword [rel g_chat_pos], 0
    mov     edi, 2
    lea     rsi, [rel msg_ctx_full]
    call    write_cstr
.ctx_room_ok:
    mov     rax, [rbp + CTX_MAX_SEQ]
    cmp     r12, rax
    jbe     .prompt_len_ok
    mov     r12, rax
.prompt_len_ok:

    ; Print PROMPT_TOKENS if requested
    cmp     dword [rel opt_print_toks], 0
    je      .do_prefill
    mov     edi, 2
    lea     rsi, [rel msg_ptoks]
    call    write_cstr
    mov     rdi, [rbp + CTX_BUF_PROMPT]
    mov     rsi, r12
    call    write_tok_csv_stderr

    ; ========================================================================
    ; PREFILL PHASE
    ; ========================================================================
.do_prefill:
    call    sys_clock_ns
    mov     [rel tm_prefill_start], rax

    mov     r13, [rel g_chat_pos]       ; pos = g_chat_pos
    xor     ebx, ebx                    ; i = 0
    mov     r14, [rbp + CTX_BUF_PROMPT]
.prefill_lp:
    cmp     rbx, r12
    jae     .prefill_done
    mov     edi, [r14 + rbx*4]          ; token_id
    mov     rsi, r13                    ; pos
    lea     rax, [rbx + 1]
    xor     edx, edx
    cmp     rax, r12
    sete    dl                          ; compute_logits = (i == n_prompt - 1)
    call    transformer_forward
    inc     rbx
    inc     r13
    jmp     .prefill_lp

.prefill_done:
    call    sys_clock_ns
    mov     [rel tm_prefill_end], rax

    ; Optional: dump raw float32 logits after prefill for exact verification
    mov     rdi, [rel opt_dump_logits]
    test    rdi, rdi
    jz      .sample_first
    call    sys_openat_wr
    test    rax, rax
    js      .sample_first
    mov     r15, rax                    ; dump fd
    mov     edi, r15d
    mov     rsi, [rbp + CTX_BUF_LOGITS]
    mov     rdx, [rbp + CTX_VOCAB_SIZE]
    shl     rdx, 2
    call    sys_write_all
    mov     rdi, r15
    call    sys_close

    ; Sample first generated token from prefill logits
.sample_first:
    call    pick_next_token
    mov     r14, rax                    ; r14 = next_tok

    ; ========================================================================
    ; DECODE PHASE
    ; ========================================================================
    call    sys_clock_ns
    mov     [rel tm_decode_start], rax

    mov     dword [rsp], 0              ; last_tok_unfed = 0
    xor     r15d, r15d                  ; n_gen = 0
    mov     rbx, [rel opt_max_tok]
.decode_lp:
    cmp     r15, rbx
    jae     .decode_done
    ; Stop on EOS (<|im_end|> or <|endoftext|>)
    cmp     r14, [rbp + CTX_EOS_1]
    je      .decode_done
    cmp     r14, [rbp + CTX_EOS_2]
    je      .decode_done

    ; Record & stream token to stdout immediately
    mov     rax, [rbp + CTX_BUF_GEN]
    mov     [rax + r15*4], r14d
    inc     r15

    mov     rdi, r14
    call    tok_write_stdout

    ; Check if we reached max_tok or max_seq
    cmp     r15, rbx
    jae     .decode_done_unfed
    cmp     r13, [rbp + CTX_MAX_SEQ]
    jae     .decode_done

    ; Forward pass for generated token at position r13
    mov     rdi, r14
    mov     rsi, r13
    mov     edx, 1                      ; compute_logits = 1
    call    transformer_forward
    inc     r13

    call    pick_next_token
    mov     r14, rax
    jmp     .decode_lp

.decode_done_unfed:
    mov     dword [rsp], 1
.decode_done:
    call    sys_clock_ns
    mov     [rel tm_decode_end], rax

    ; Trailing newline on stdout
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all

    ; In --chat mode, close assistant turn in KV-cache with <|im_end|> + \n
    cmp     dword [rel opt_chat], 0
    je      .after_kv_close

    cmp     dword [rsp], 0
    je      .close_eot
    cmp     r13, [rbp + CTX_MAX_SEQ]
    jae     .save_chat_pos
    mov     rdi, r14
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13

.close_eot:
    cmp     dword [rel opt_raw], 0
    jne     .save_chat_pos
    lea     rax, [r13 + 2]
    cmp     rax, [rbp + CTX_MAX_SEQ]
    jae     .save_chat_pos
    mov     edi, TOK_IM_END
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
    mov     edi, TOK_NL
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
.save_chat_pos:
    mov     [rel g_chat_pos], r13
.after_kv_close:

    ; Print GEN_TOKENS if requested
    cmp     dword [rel opt_print_toks], 0
    je      .print_stats
    mov     edi, 2
    lea     rsi, [rel msg_gtoks]
    call    write_cstr
    mov     rdi, [rbp + CTX_BUF_GEN]
    mov     rsi, r15
    call    write_tok_csv_stderr

    ; Print performance summary to stderr
.print_stats:
    ; Prefill stats
    mov     edi, 2
    lea     rsi, [rel msg_prefill]
    call    write_cstr
    mov     edi, 2
    mov     rsi, r12
    call    write_u64
    mov     edi, 2
    lea     rsi, [rel msg_in]
    call    write_cstr

    mov     rax, [rel tm_prefill_end]
    sub     rax, [rel tm_prefill_start]
    test    rax, rax
    jnz     .pf_nz
    mov     rax, 1
.pf_nz:
    vcvtsi2sd xmm0, xmm0, rax           ; dt_ns
    vdivsd  xmm1, xmm0, [rel d_1e6]     ; ms
    vmovsd  [rel tm_tmp], xmm0
    mov     edi, 2
    vmovapd xmm0, xmm1
    call    write_f64_2dp
    mov     edi, 2
    lea     rsi, [rel msg_ms_open]
    call    write_cstr
    vcvtsi2sd xmm0, xmm0, r12
    vmulsd  xmm0, xmm0, [rel d_1e9]
    vdivsd  xmm0, xmm0, [rel tm_tmp]    ; tok/s
    mov     edi, 2
    call    write_f64_2dp
    mov     edi, 2
    lea     rsi, [rel msg_tps_end]
    call    write_cstr

    ; Decode stats
    mov     edi, 2
    lea     rsi, [rel msg_decode]
    call    write_cstr
    mov     edi, 2
    mov     rsi, r15
    call    write_u64
    mov     edi, 2
    lea     rsi, [rel msg_in]
    call    write_cstr

    mov     rax, [rel tm_decode_end]
    sub     rax, [rel tm_decode_start]
    test    rax, rax
    jnz     .dec_nz
    mov     rax, 1
.dec_nz:
    vcvtsi2sd xmm0, xmm0, rax           ; dt_ns
    vmovsd  [rel tm_tmp], xmm0
    vdivsd  xmm0, xmm0, [rel d_1e6]     ; ms
    mov     edi, 2
    call    write_f64_2dp
    mov     edi, 2
    lea     rsi, [rel msg_ms_open]
    call    write_cstr
    mov     rax, r15
    cmp     rax, 1
    jbe     .dec_cnt_ok
    dec     rax                         ; exact number of decode forward passes
.dec_cnt_ok:
    vcvtsi2sd xmm0, xmm0, rax
    vmulsd  xmm0, xmm0, [rel d_1e9]
    vdivsd  xmm0, xmm0, [rel tm_tmp]
    mov     edi, 2
    call    write_f64_2dp
    mov     edi, 2
    lea     rsi, [rel msg_tps_thr]
    call    write_cstr
    mov     edi, 2
    mov     rsi, [rel opt_threads]
    call    write_u64

    cmp     dword [rel opt_chat], 0
    je      .stats_close_single
    mov     edi, 2
    lea     rsi, [rel msg_ctx_lbl]
    call    write_cstr
    mov     edi, 2
    mov     rsi, r13
    call    write_u64
    mov     edi, 2
    lea     rsi, [rel msg_slash]
    call    write_cstr
    mov     edi, 2
    mov     rsi, [rbp + CTX_MAX_SEQ]
    call    write_u64
    mov     edi, 2
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all
    jmp     .rt_done

.stats_close_single:
    mov     edi, 2
    lea     rsi, [rel msg_close_nl]
    call    write_cstr

.rt_done:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; int64 read_stdin_line(char *buf in rdi, uint64 max_len in rsi)
; Raw Linux SYS_read(0, ...) line reader. Returns len >= 0, or -1 on EOF.
; ----------------------------------------------------------------------------
read_stdin_line:
    push    rbx
    push    r12
    push    r13
    sub     rsp, 16                     ; 3 pushes (24) + 16 + ret(8) = 48 -> 16B aligned
    mov     r12, rdi
    lea     r13, [rsi - 1]
    xor     ebx, ebx
.rd_lp:
    xor     edi, edi                    ; fd = 0 (stdin)
    mov     rsi, rsp
    mov     edx, 1
    mov     eax, SYS_read
    syscall
    cmp     rax, -4                     ; EINTR
    je      .rd_lp
    test    rax, rax
    jle     .rd_eof
    movzx   eax, byte [rsp]
    cmp     al, 10                      ; '\n'
    je      .rd_done
    cmp     al, 13                      ; '\r'
    je      .rd_lp
    cmp     rbx, r13
    jae     .rd_lp
    mov     [r12 + rbx], al
    inc     rbx
    jmp     .rd_lp
.rd_eof:
    test    rbx, rbx
    jnz     .rd_done
    mov     rax, -1
    jmp     .rd_ret
.rd_done:
    mov     byte [r12 + rbx], 0
    mov     rax, rbx
.rd_ret:
    add     rsp, 16
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; uint64 pick_next_token(void) -> rax
; Greedy argmax when opt_temp <= 0.0f, else temperature + top-p sampling.
; ----------------------------------------------------------------------------
pick_next_token:
    lea     rax, [rel g_ctx]
    vmovss  xmm0, [rel opt_temp]
    vxorps  xmm1, xmm1, xmm1
    vucomiss xmm0, xmm1
    ja      .sample
    mov     rdi, [rax + CTX_BUF_LOGITS]
    mov     rsi, [rax + CTX_VOCAB_SIZE]
    jmp     argmax_f32
.sample:
    mov     rdi, [rax + CTX_BUF_LOGITS]
    mov     rsi, [rax + CTX_BUF_PROBS]
    mov     rdx, [rax + CTX_VOCAB_SIZE]
    vmovss  xmm1, [rel opt_topp]
    lea     rcx, [rel opt_rng]
    jmp     sample_top_p

; ----------------------------------------------------------------------------
; void write_tok_csv_stderr(const uint32 *toks, uint64 count)
; ----------------------------------------------------------------------------
write_tok_csv_stderr:
    push    rbx
    push    r12
    push    r13
    mov     r12, rdi
    mov     r13, rsi
    xor     ebx, ebx
.lp:
    cmp     rbx, r13
    jae     .done
    test    rbx, rbx
    jz      .num
    mov     edi, 2
    lea     rsi, [rel msg_comma]
    mov     edx, 1
    call    sys_write_all
.num:
    mov     edi, 2
    mov     esi, [r12 + rbx*4]
    call    write_u64
    inc     rbx
    jmp     .lp
.done:
    mov     edi, 2
    lea     rsi, [rel msg_nl]
    mov     edx, 1
    call    sys_write_all
    pop     r13
    pop     r12
    pop     rbx
    ret

section .rodata
align 8
d_1e6:          dq 1000000.0
d_1e9:          dq 1000000000.0

flg_m:          db "-m", 0
flg_p:          db "-p", 0
flg_n:          db "-n", 0
flg_t:          db "-t", 0
flg_c:          db "-c", 0
flg_think:      db "--think", 0
flg_raw:        db "--raw", 0
flg_tokens:     db "--tokens", 0
flg_temp:       db "--temp", 0
flg_topp:       db "--topp", 0
flg_seed:       db "--seed", 0
flg_dump:       db "--dump-logits", 0
flg_ptoks:      db "--print-tokens", 0
flg_chat:       db "--chat", 0
flg_h:          db "-h", 0
flg_help:       db "--help", 0

cmd_quit:       db "/quit", 0
cmd_exit_slash: db "/exit", 0
cmd_exit:       db "exit", 0
cmd_clear:      db "/clear", 0
cmd_new:        db "/new", 0
cmd_reset:      db "/reset", 0
cmd_think:      db "/think", 0
cmd_nothink:    db "/nothink", 0

def_prompt:     db "Hello!", 0
msg_nl:         db 10
msg_comma:      db ","
msg_slash:      db "/", 0
msg_ptoks:      db "PROMPT_TOKENS:", 0
msg_gtoks:      db "GEN_TOKENS:", 0
msg_prefill:    db "[asm-llm] prefill: ", 0
msg_decode:     db "[asm-llm] decode : ", 0
msg_in:         db " tok in ", 0
msg_ms_open:    db " ms (", 0
msg_tps_end:    db " tok/s)", 10, 0
msg_tps_thr:    db " tok/s, threads=", 0
msg_ctx_lbl:    db ") ctx=", 0
msg_close_nl:   db ")", 10, 0
msg_chat_help:  db "Interactive Multi-Turn Chat (/clear, /think, /nothink, /quit)", 10, 0
msg_chat_prompt: db 10, "asm-llm> ", 0
msg_chat_cleared: db "[chat] history cleared (ctx=0/", 0
msg_ctx_full:   db "[chat] context full, resetting KV-cache", 10, 0
msg_think_on:   db "[mode] think=1", 10, 0
msg_think_off:  db "[mode] think=0", 10, 0
msg_err_model:  db "error: failed to load .asmbin model", 10, 0
msg_err_prompt: db "error: empty prompt", 10, 0
msg_usage:
    db "asm-llm (pure x86-64 AVX2/FMA Qwen3-0.6B Q8_0, zero libc)", 10
    db "Usage: asm-llm -m <model.asmbin> [options]", 10
    db "  -m <path>            Path to .asmbin model (required)", 10
    db "  -p <prompt>          Prompt text (default: Hello!)", 10
    db "  -n <max_tokens>      Max generated tokens (default: 128)", 10
    db "  -t <threads>         Thread count (default: 32)", 10
    db "  -c <ctx_len>         Max context / KV cache length (default: 2048)", 10
    db "  --chat               Interactive multi-turn chat (preserves KV-cache)", 10
    db "  --think              Enable Qwen3 <think> reasoning mode", 10
    db "  --raw                Do not wrap prompt in Qwen3 chat template", 10
    db "  --tokens <csv>       Pass exact comma-separated token IDs", 10
    db "  --temp <float>       Temperature (0.0 = greedy, default: 0.0)", 10
    db "  --topp <float>       Top-p threshold when temp > 0 (default: 0.95)", 10
    db "  --seed <uint>        PRNG seed for sampling (default: 42)", 10
    db "  --dump-logits <path> Dump float32 prefill logits to file", 10
    db "  --print-tokens       Print prompt & generated token IDs to stderr", 10, 0

section .bss
alignb 8
opt_model:        resq 1
opt_prompt:       resq 1
opt_max_tok:      resq 1
opt_threads:      resq 1
opt_ctx_len:      resq 1
opt_tokens_csv:   resq 1
opt_dump_logits:  resq 1
opt_rng:          resq 1
g_chat_pos:       resq 1
opt_temp:         resd 1
opt_topp:         resd 1
opt_think:        resd 1
opt_raw:          resd 1
opt_chat:         resd 1
opt_has_prompt:   resd 1
opt_print_toks:   resd 1
tm_prefill_start: resq 1
tm_prefill_end:   resq 1
tm_decode_start:  resq 1
tm_decode_end:    resq 1
tm_tmp:           resq 1
g_line_buf:       resb 4096
