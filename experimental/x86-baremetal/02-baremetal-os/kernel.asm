; ============================================================================
; os/kernel.asm — asm-llm OS v2.0 Bare-Metal Kernel, Boot Banner & REPL
;
; Boots directly on x86-64 hardware / QEMU KVM via Multiboot 1 (no OS, no libc):
;   - Initializes COM1 Serial + 80x25 Color VGA + PIT/RDTSC Timer + SMP APIC
;   - Parses Multiboot 1 cmdline & binds .asmbin Q8_0 model zero-copy from RAM
;   - Displays color system & model boot banner
;   - Runs batch cmdline prompt (with optional clean QEMU ACPI shutdown) or
;     interactive bare-metal REPL console accepting both Serial and PS/2 input
; ============================================================================

bits 64
default rel

%include "constants.inc"
%include "os_constants.inc"

extern g_ctx
extern g_mb_magic
extern g_mb_info
extern g_has_avx2_fma
extern g_tsc_mhz
extern g_total_ram_mb
extern g_arena_used_mb
extern g_smp_total_cpus
extern g_smp_online_cpus

extern serial_init
extern serial_putc
extern vga_clear
extern vga_set_color
extern console_write
extern console_write_cstr
extern console_readline
extern timer_init
extern timer_ns
extern qemu_shutdown

extern mem_init
extern smp_init
extern tp_shutdown
extern model_init
extern transformer_forward
extern tok_build_chat_prompt
extern tok_write_stdout
extern argmax_f32
extern sample_top_p

extern str_len
extern str_eq
extern parse_u64
extern parse_f32
extern sys_write_all
extern write_cstr
extern write_u64
extern write_f64_2dp
extern write_hex64_raw

extern qp_handle_line           ; Qwenputer-X86 extensions (qwenputer.asm)

section .text

global kernel_main

; ============================================================================
; void kernel_main(void)
; ============================================================================
kernel_main:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    ; 1. Early hardware init: COM1 Serial + Color VGA + PIT/RDTSC Timer
    call    serial_init
    mov     dil, VGA_ATTR_GREY
    call    vga_clear

    ; Check AVX2 + FMA before executing any AVX instructions
    cmp     dword [rel g_has_avx2_fma], 0
    je      .err_no_avx2

    call    timer_init
    call    read_cpu_brand

    ; 2. Validate Multiboot 1 magic & extract module + cmdline
    cmp     dword [rel g_mb_magic], MB_BOOTLOADER_MAGIC
    jne     .err_multiboot

    mov     r12d, [rel g_mb_info]       ; r12 = Multiboot Info struct (phys addr)

    ; Initialize default kernel options
    mov     byte [rel opt_prompt_buf], 0
    mov     dword [rel opt_has_prompt], 0
    mov     qword [rel opt_max_tok], 128
    mov     qword [rel opt_ctx_len], 2048
    mov     qword [rel opt_req_threads], 0 ; 0 = all online SMP cores
    mov     dword [rel opt_think], 0
    mov     dword [rel opt_raw], 0
    mov     dword [rel opt_single_turn], 0
    mov     dword [rel opt_shutdown], 0
    mov     dword [rel opt_print_toks], 0
    mov     qword [rel g_chat_pos], 0
    mov     dword [rel opt_temp], 0     ; 0.0f = greedy argmax
    mov     eax, 0x3F733333             ; 0.95f
    mov     [rel opt_topp], eax
    mov     qword [rel opt_rng], 42

    ; Copy & parse Multiboot cmdline if present (flags bit 2 = 0x04)
    mov     eax, [r12 + MBI_FLAGS]
    test    eax, 0x04
    jz      .no_cmdline
    mov     esi, [r12 + MBI_CMDLINE]
    test    esi, esi
    jz      .no_cmdline
    lea     rdi, [rel g_cmdline_buf]
    mov     ecx, CMDLINE_BUF_SIZE - 1
.copy_cmd:
    mov     al, [rsi]
    mov     [rdi], al
    test    al, al
    jz      .cmd_copied
    inc     rsi
    inc     rdi
    dec     ecx
    jnz     .copy_cmd
    mov     byte [rdi], 0
.cmd_copied:
    call    parse_cmdline
.no_cmdline:

    ; Locate .asmbin Multiboot module (flags bit 3 = 0x08, mods_count >= 1)
    mov     eax, [r12 + MBI_FLAGS]
    test    eax, 0x08
    jz      .err_no_module
    mov     eax, [r12 + MBI_MODS_COUNT]
    test    eax, eax
    jz      .err_no_module

    mov     eax, [r12 + MBI_MODS_ADDR]  ; ptr to first mb_module_t
    mov     r13d, [rax + MB_MOD_START]  ; r13 = mod_start
    mov     r14d, [rax + MB_MOD_END]    ; r14 = mod_end
    cmp     r14, r13
    jbe     .err_no_module
    mov     [rel g_mod_start], r13
    mov     [rel g_mod_end], r14
    mov     r15, r14
    sub     r15, r13                    ; r15 = mod_size
    mov     [rel g_mod_size], r15

    ; Verify ASML magic in physical RAM before anything else
    cmp     dword [r13 + HDR_MAGIC], ASMBIN_MAGIC
    jne     .err_bad_magic

    ; 3. Initialize bare-metal physical memory bump allocator
    mov     rdi, r12
    mov     rsi, r13
    mov     rdx, r14
    call    mem_init
    test    rax, rax
    js      .err_oom

    ; 4. Boot all SMP AP cores via Local APIC INIT-SIPI-SIPI + 16-bit trampoline
    mov     rdi, [rel opt_req_threads]
    call    smp_init
    mov     [rel g_active_cores], rax

    ; 5. Bind Qwen3-0.6B model zero-copy & allocate KV-cache / scratch arena
    mov     rdi, r13                    ; mod_start
    mov     rsi, r15                    ; mod_size
    mov     rdx, [rel opt_ctx_len]      ; max_seq_override
    call    model_init
    test    rax, rax
    js      .err_model

    ; Capture boot-ready timestamp (from _start to model+SMP ready)
    call    timer_ns
    mov     [rel tm_boot_ready], rax

    ; 6. Print colored Bare-Metal OS Boot Banner (80-column clean)
    call    print_boot_banner

    ; 7. Batch mode (if prompt= was supplied via kernel cmdline)
    cmp     dword [rel opt_has_prompt], 0
    je      .enter_repl

    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_prompt_lbl]
    call    write_cstr
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel opt_prompt_buf]
    call    write_cstr
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    call    write_cstr

    lea     rdi, [rel opt_prompt_buf]
    mov     esi, 1                      ; show boot-to-1st-token timing
    call    run_inference

    cmp     dword [rel opt_shutdown], 0
    je      .enter_repl

    call    tp_shutdown
    call    qemu_shutdown

    ; 8. Interactive Bare-Metal REPL Console (COM1 Serial + PS/2 Keyboard)
.enter_repl:
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_repl_help]
    call    write_cstr

.repl_loop:
    mov     dil, VGA_ATTR_GREEN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_repl_prompt]
    call    write_cstr
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color

    lea     rdi, [rel g_line_buf]
    mov     rsi, LINE_BUF_SIZE
    call    console_readline
    test    rax, rax
    js      .do_halt                    ; Ctrl-D EOF

    ; Trim leading spaces
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
    jz      .repl_loop                  ; empty line
    lea     rcx, [r12 + rax - 1]
.tt_lp:
    cmp     rcx, r12
    jb      .chk_cmd
    mov     al, [rcx]
    cmp     al, ' '
    je      .tt_chop
    cmp     al, 9
    je      .tt_chop
    jmp     .chk_cmd
.tt_chop:
    mov     byte [rcx], 0
    dec     rcx
    jmp     .tt_lp

.chk_cmd:
    ; /halt, /quit, exit
    mov     rdi, r12
    lea     rsi, [rel cmd_halt]
    call    str_eq
    test    eax, eax
    jnz     .do_halt

    mov     rdi, r12
    lea     rsi, [rel cmd_quit]
    call    str_eq
    test    eax, eax
    jnz     .do_halt

    mov     rdi, r12
    lea     rsi, [rel cmd_exit]
    call    str_eq
    test    eax, eax
    jnz     .do_halt

    ; /clear, /new, /reset — reset multi-turn KV-cache history
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
    jz      .chk_nothink
    mov     dword [rel opt_think], 1
    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_think_on]
    call    write_cstr
    jmp     .repl_loop

.do_clear:
    mov     qword [rel g_chat_pos], 0
    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_chat_cleared]
    call    write_cstr
    lea     rax, [rel g_ctx]
    mov     edi, 1
    mov     rsi, [rax + CTX_MAX_SEQ]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_close_nl]
    call    write_cstr
    jmp     .repl_loop

.chk_nothink:
    mov     rdi, r12
    lea     rsi, [rel cmd_nothink]
    call    str_eq
    test    eax, eax
    jz      .chk_bench
    mov     dword [rel opt_think], 0
    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_think_off]
    call    write_cstr
    jmp     .repl_loop

.chk_bench:
    mov     rdi, r12
    lea     rsi, [rel cmd_bench]
    call    str_eq
    test    eax, eax
    jz      .chk_help
    mov     r13, [rel opt_max_tok]
    mov     r14d, [rel opt_single_turn]
    mov     qword [rel opt_max_tok], 64
    mov     dword [rel opt_single_turn], 1
    mov     qword [rel g_chat_pos], 0
    lea     rdi, [rel bench_prompt]
    xor     esi, esi
    call    run_inference
    mov     [rel opt_max_tok], r13
    mov     [rel opt_single_turn], r14d
    mov     qword [rel g_chat_pos], 0
    jmp     .repl_loop

.chk_help:
    mov     rdi, r12
    lea     rsi, [rel cmd_help]
    call    str_eq
    test    eax, eax
    jz      .chk_qp

    ; (help body follows below)
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_repl_help]
    call    write_cstr
    jmp     .repl_loop

.chk_qp:
    ; Qwenputer-X86 commands (/calc, /put, /get, /del, /list, /info, /qhelp)
    mov     rdi, r12
    call    qp_handle_line
    test    eax, eax
    jnz     .repl_loop

.run_user_prompt:
    mov     rdi, r12
    xor     esi, esi
    call    run_inference
    jmp     .repl_loop

.do_halt:
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_poweroff]
    call    write_cstr
    call    tp_shutdown
    call    qemu_shutdown

; Error handlers
.err_no_avx2:
    lea     rdi, [rel msg_err_avx2]
    jmp     .fatal
.err_multiboot:
    lea     rdi, [rel msg_err_mb]
    jmp     .fatal
.err_no_module:
    lea     rdi, [rel msg_err_mod]
    jmp     .fatal
.err_bad_magic:
    lea     rdi, [rel msg_err_magic]
    jmp     .fatal
.err_oom:
    lea     rdi, [rel msg_err_oom]
    jmp     .fatal
.err_model:
    lea     rdi, [rel msg_err_model]
.fatal:
    push    rdi
    mov     dil, VGA_ATTR_RED
    call    vga_set_color
    pop     rdi
    call    console_write_cstr
    call    tp_shutdown
    call    qemu_shutdown

; ============================================================================
; void run_inference(const char *prompt in rdi, int show_boot_ttft in esi)
; Exact same Qwen3-0.6B BPE + prefill + streaming decode loop as Phase 1!
; ============================================================================
run_inference:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                      ; 6 pushes + 8 + ret(8) = 64 -> 16B aligned

    mov     [rel cur_show_ttft], esi
    lea     rbp, [rel g_ctx]

    ; Build prompt tokens via pure-asm BPE + Qwen3 chat template
    mov     rsi, [rbp + CTX_BUF_PROMPT]
    mov     rdx, MAX_PROMPT_TOKENS
    mov     ecx, [rel opt_raw]
    mov     r8d, [rel opt_think]
    call    tok_build_chat_prompt
    mov     r12, rax                    ; r12 = n_prompt
    test    r12, r12
    jz      .done

    ; Multi-turn KV-cache management
    cmp     dword [rel opt_single_turn], 0
    je      .chk_ctx_room
    mov     qword [rel g_chat_pos], 0
    jmp     .ctx_room_ok
.chk_ctx_room:
    mov     rax, [rel g_chat_pos]
    test    rax, rax
    jz      .ctx_room_ok
    lea     rcx, [rax + r12 + 64]
    cmp     rcx, [rbp + CTX_MAX_SEQ]
    jb      .ctx_room_ok
    mov     qword [rel g_chat_pos], 0
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_ctx_full]
    call    write_cstr
.ctx_room_ok:
    mov     rax, [rbp + CTX_MAX_SEQ]
    cmp     r12, rax
    jbe     .prompt_len_ok
    mov     r12, rax
.prompt_len_ok:

    ; Optional PROMPT_TOKENS output for automated exact verification
    cmp     dword [rel opt_print_toks], 0
    je      .do_prefill
    mov     edi, 1
    lea     rsi, [rel msg_ptoks]
    call    write_cstr
    mov     rdi, [rbp + CTX_BUF_PROMPT]
    mov     rsi, r12
    call    write_tok_csv

    ; --- PREFILL (incremental from pos = g_chat_pos, preserving KV-cache!) ---
.do_prefill:
    call    timer_ns
    mov     [rel tm_prefill_start], rax

    mov     r13, [rel g_chat_pos]       ; pos = g_chat_pos
    xor     ebx, ebx                    ; i = 0
    mov     r14, [rbp + CTX_BUF_PROMPT]
.prefill_lp:
    cmp     rbx, r12
    jae     .prefill_done
    mov     edi, [r14 + rbx*4]
    mov     rsi, r13
    lea     rax, [rbx + 1]
    xor     edx, edx
    cmp     rax, r12
    sete    dl                          ; compute_logits = (i == n_prompt - 1)
    call    transformer_forward
    inc     rbx
    inc     r13
    jmp     .prefill_lp

.prefill_done:
    call    timer_ns
    mov     [rel tm_prefill_end], rax

    ; Sample first generated token
    call    pick_next_token
    mov     r14, rax                    ; r14 = next_tok

    ; Capture exact boot-to-first-token timestamp
    call    timer_ns
    mov     [rel tm_first_tok], rax

    ; --- DECODE (streaming directly to COM1 UTF-8 + VGA bright white!) ---
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color

    call    timer_ns
    mov     [rel tm_decode_start], rax

    mov     dword [rsp], 0              ; last_tok_unfed = 0
    xor     r15d, r15d                  ; n_gen = 0
    mov     rbx, [rel opt_max_tok]
.decode_lp:
    cmp     r15, rbx
    jae     .decode_done
    cmp     r14, [rbp + CTX_EOS_1]
    je      .decode_done
    cmp     r14, [rbp + CTX_EOS_2]
    je      .decode_done

    mov     rax, [rbp + CTX_BUF_GEN]
    mov     [rax + r15*4], r14d
    inc     r15

    mov     rdi, r14
    call    tok_write_stdout

    cmp     r15, rbx
    jae     .decode_done_unfed
    cmp     r13, [rbp + CTX_MAX_SEQ]
    jae     .decode_done

    mov     rdi, r14
    mov     rsi, r13
    mov     edx, 1
    call    transformer_forward
    inc     r13

    call    pick_next_token
    mov     r14, rax
    jmp     .decode_lp

.decode_done_unfed:
    mov     dword [rsp], 1              ; r14 was emitted on max_tok hit, not yet forwarded
.decode_done:
    call    timer_ns
    mov     [rel tm_decode_end], rax

    mov     edi, 1
    lea     rsi, [rel msg_nl]
    call    write_cstr

    ; Close assistant turn in KV-cache for seamless Qwen3 multi-turn continuity
    cmp     dword [rel opt_single_turn], 0
    jne     .reset_chat_after

    ; If decode stopped at max_tokens, feed the last emitted token into KV-cache
    cmp     dword [rsp], 0
    je      .close_eot
    cmp     r13, [rbp + CTX_MAX_SEQ]
    jae     .save_chat_pos
    mov     rdi, r14
    mov     rsi, r13
    xor     edx, edx                    ; compute_logits = 0
    call    transformer_forward
    inc     r13

.close_eot:
    ; Feed <|im_end|> and \n so next turn's <|im_start|>user\n aligns exact
    ; (LFM3: im_end comes from the header, \n is token 207 in its vocab)
    cmp     dword [rel opt_raw], 0
    jne     .save_chat_pos
    lea     rax, [r13 + 2]
    cmp     rax, [rbp + CTX_MAX_SEQ]
    jae     .save_chat_pos
    mov     edi, TOK_IM_END
    cmp     dword [rbp + CTX_ARCH], 1
    jne     .eot_feed
    mov     edi, [rbp + CTX_EOS_1]
.eot_feed:
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
    mov     edi, TOK_NL
    cmp     dword [rbp + CTX_ARCH], 1
    jne     .nl_feed
    mov     edi, LFM3_TOK_NL
.nl_feed:
    mov     rsi, r13
    xor     edx, edx
    call    transformer_forward
    inc     r13
.save_chat_pos:
    mov     [rel g_chat_pos], r13
    jmp     .after_kv_close

.reset_chat_after:
    mov     qword [rel g_chat_pos], 0
.after_kv_close:

    ; Optional GEN_TOKENS output for automated exact verification
    cmp     dword [rel opt_print_toks], 0
    je      .print_stats
    mov     edi, 1
    lea     rsi, [rel msg_gtoks]
    call    write_cstr
    mov     rdi, [rbp + CTX_BUF_GEN]
    mov     rsi, r15
    call    write_tok_csv

    ; --- PERFORMANCE SUMMARY (green on VGA) ---
.print_stats:
    mov     dil, VGA_ATTR_GREEN
    call    vga_set_color

    ; Prefill line
    mov     edi, 1
    lea     rsi, [rel msg_prefill]
    call    write_cstr
    mov     edi, 1
    mov     rsi, r12
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_in]
    call    write_cstr

    mov     rax, [rel tm_prefill_end]
    sub     rax, [rel tm_prefill_start]
    test    rax, rax
    jnz     .pf_nz
    mov     rax, 1
.pf_nz:
    vcvtsi2sd xmm0, xmm0, rax
    vmovsd  [rel tm_tmp], xmm0
    vdivsd  xmm0, xmm0, [rel d_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_ms_open]
    call    write_cstr
    vcvtsi2sd xmm0, xmm0, r12
    vmulsd  xmm0, xmm0, [rel d_1e9]
    vdivsd  xmm0, xmm0, [rel tm_tmp]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_tps_end]
    call    write_cstr

    ; Decode line
    mov     edi, 1
    lea     rsi, [rel msg_decode]
    call    write_cstr
    mov     edi, 1
    mov     rsi, r15
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_in]
    call    write_cstr

    mov     rax, [rel tm_decode_end]
    sub     rax, [rel tm_decode_start]
    test    rax, rax
    jnz     .dec_nz
    mov     rax, 1
.dec_nz:
    vcvtsi2sd xmm0, xmm0, rax
    vmovsd  [rel tm_tmp], xmm0
    vdivsd  xmm0, xmm0, [rel d_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
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
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_tps_cores]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel g_active_cores]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_ctx_lbl]
    call    write_cstr
    mov     edi, 1
    mov     rsi, r13
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_slash]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rbp + CTX_MAX_SEQ]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    call    write_cstr

    ; Optional boot-to-first-token line (batch mode)
    cmp     dword [rel cur_show_ttft], 0
    je      .done
    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel msg_boot_ttft]
    call    write_cstr
    mov     rax, [rel tm_boot_ready]
    vcvtsi2sd xmm0, xmm0, rax
    vdivsd  xmm0, xmm0, [rel d_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_ttft_mid]
    call    write_cstr
    mov     rax, [rel tm_first_tok]
    vcvtsi2sd xmm0, xmm0, rax
    vdivsd  xmm0, xmm0, [rel d_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_ms_nl]
    call    write_cstr

.done:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; uint64 pick_next_token(void) -> rax
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
; void write_tok_csv(const uint32 *toks in rdi, uint64 count in rsi)
; ----------------------------------------------------------------------------
write_tok_csv:
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
    mov     edi, 1
    lea     rsi, [rel msg_comma]
    mov     edx, 1
    call    sys_write_all
.num:
    mov     edi, 1
    mov     esi, [r12 + rbx*4]
    call    write_u64
    inc     rbx
    jmp     .lp
.done:
    mov     edi, 1
    lea     rsi, [rel msg_nl]
    call    write_cstr
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; Boot Banner Printer (80-column clean, colored on VGA + crisp text on COM1)
; ============================================================================
print_boot_banner:
    push    rbx

    ; Start banner on a clean line on COM1 serial (keep VGA at row 0!)
    mov     al, 10
    call    serial_putc

    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_line]
    call    write_cstr

    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_title]
    call    write_cstr

    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_line]
    call    write_cstr

    ; CPU line
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_cpu]
    call    write_cstr
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel g_cpu_brand]
    call    write_cstr
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_avx_tsc]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel g_tsc_mhz]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel ban_mhz_nl]
    call    write_cstr

    ; SMP line
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_smp]
    call    write_cstr
    mov     dil, VGA_ATTR_GREEN
    call    vga_set_color
    mov     edi, 1
    mov     rsi, [rel g_active_cores]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_slash]
    call    write_cstr
    mov     edi, 1
    mov     esi, [rel g_smp_total_cpus]
    call    write_u64
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_smp_tail]
    call    write_cstr

    ; RAM + Boot line
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_ram]
    call    write_cstr
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color
    mov     edi, 1
    mov     rsi, [rel g_total_ram_mb]
    call    write_u64
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_ram_mid]
    call    write_cstr
    mov     edi, 1
    mov     rsi, [rel g_arena_used_mb]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel ban_boot_lbl]
    call    write_cstr
    mov     rax, [rel tm_boot_ready]
    vcvtsi2sd xmm0, xmm0, rax
    vdivsd  xmm0, xmm0, [rel d_1e6]
    mov     edi, 1
    call    write_f64_2dp
    mov     edi, 1
    lea     rsi, [rel msg_ms_nl]
    call    write_cstr

    ; Model line
    mov     dil, VGA_ATTR_YELLOW
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_model]
    call    write_cstr
    mov     dil, VGA_ATTR_WHITE
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_qwen]
    call    write_cstr
    mov     rax, [rel g_mod_size]
    add     rax, 524288
    shr     rax, 20                     ; MiB
    mov     edi, 1
    mov     rsi, rax
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel ban_mod_at]
    call    write_cstr
    mov     rdi, [rel g_mod_start]
    call    write_hex64_raw
    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_mod_ctx]
    call    write_cstr
    lea     rax, [rel g_ctx]
    mov     edi, 1
    mov     rsi, [rax + CTX_MAX_SEQ]
    call    write_u64
    mov     edi, 1
    lea     rsi, [rel msg_close_nl]
    call    write_cstr

    mov     dil, VGA_ATTR_CYAN
    call    vga_set_color
    mov     edi, 1
    lea     rsi, [rel ban_line]
    call    write_cstr

    mov     dil, VGA_ATTR_GREY
    call    vga_set_color
    pop     rbx
    ret

; ============================================================================
; CPUID Brand String Reader (leaves 0x80000002..0x80000004 -> g_cpu_brand)
; ============================================================================
read_cpu_brand:
    push    rbx
    push    r12
    mov     eax, 0x80000000
    cpuid
    cmp     eax, 0x80000004
    jb      .fallback

    lea     r12, [rel cpu_brand_raw]
    mov     eax, 0x80000002
    cpuid
    mov     [r12 + 0],  eax
    mov     [r12 + 4],  ebx
    mov     [r12 + 8],  ecx
    mov     [r12 + 12], edx
    mov     eax, 0x80000003
    cpuid
    mov     [r12 + 16], eax
    mov     [r12 + 20], ebx
    mov     [r12 + 24], ecx
    mov     [r12 + 28], edx
    mov     eax, 0x80000004
    cpuid
    mov     [r12 + 32], eax
    mov     [r12 + 36], ebx
    mov     [r12 + 40], ecx
    mov     [r12 + 44], edx
    mov     byte [r12 + 48], 0

    ; Trim leading & trailing spaces into g_cpu_brand (max 38 chars for 80-col VGA)
    mov     rsi, r12
.skip_sp:
    cmp     byte [rsi], ' '
    jne     .copy
    inc     rsi
    jmp     .skip_sp
.copy:
    lea     rdi, [rel g_cpu_brand]
    xor     ecx, ecx
.cpy_lp:
    mov     al, [rsi + rcx]
    test    al, al
    jz      .trim_end
    cmp     ecx, 38
    jae     .trim_end
    mov     [rdi + rcx], al
    inc     ecx
    jmp     .cpy_lp
.trim_end:
    mov     byte [rdi + rcx], 0
.te_lp:
    test    ecx, ecx
    jz      .done
    dec     ecx
    cmp     byte [rdi + rcx], ' '
    jne     .done
    mov     byte [rdi + rcx], 0
    jmp     .te_lp

.fallback:
    lea     rdi, [rel g_cpu_brand]
    lea     rsi, [rel def_cpu_brand]
.fb_lp:
    mov     al, [rsi]
    mov     [rdi], al
    inc     rsi
    inc     rdi
    test    al, al
    jnz     .fb_lp
.done:
    pop     r12
    pop     rbx
    ret

; ============================================================================
; Multiboot Kernel Cmdline Parser
; Supports:
;   prompt="text with spaces" (or prompt='...' or prompt=singleword)
;   max_tokens=N, ctx=N, threads=N, think=0|1, raw=0|1, shutdown=0|1,
;   print_tokens=0|1, temp=F, topp=F, seed=N
; ============================================================================
parse_cmdline:
    push    rbx
    push    r12
    push    r13
    lea     rbx, [rel g_cmdline_buf]

.tok_lp:
    ; Skip whitespace
    mov     al, [rbx]
    test    al, al
    jz      .done
    cmp     al, ' '
    je      .skip_ws
    cmp     al, 9
    je      .skip_ws
    jmp     .parse_key
.skip_ws:
    inc     rbx
    jmp     .tok_lp

.parse_key:
    ; Extract key up to '=' or whitespace or NUL
    lea     rdi, [rel cmd_key_buf]
    xor     ecx, ecx
.key_lp:
    mov     al, [rbx]
    test    al, al
    jz      .done
    cmp     al, ' '
    je      .tok_lp                     ; bare word without '=', ignore
    cmp     al, 9
    je      .tok_lp
    cmp     al, '='
    je      .key_eq
    cmp     ecx, 30
    jae     .key_skip
    mov     [rdi + rcx], al
    inc     ecx
.key_skip:
    inc     rbx
    jmp     .key_lp

.key_eq:
    mov     byte [rdi + rcx], 0
    inc     rbx                         ; skip '='

    ; Extract value (supports "..." or '...' quoting, or unquoted word)
    lea     r12, [rel cmd_val_buf]
    xor     ecx, ecx
    mov     dl, 0                       ; quote char (0 = none)
    mov     al, [rbx]
    cmp     al, '"'
    je      .start_quote
    cmp     al, "'"
    je      .start_quote
    jmp     .val_lp
.start_quote:
    mov     dl, al
    inc     rbx

.val_lp:
    mov     al, [rbx]
    test    al, al
    jz      .val_done
    test    dl, dl
    jz      .val_unquoted
    cmp     al, dl
    je      .end_quote
    ; Handle \" inside double quotes
    cmp     al, '\'
    jne     .val_store
    cmp     byte [rbx + 1], 0
    je      .val_store
    inc     rbx
    mov     al, [rbx]
    jmp     .val_store
.val_unquoted:
    cmp     al, ' '
    je      .val_done
    cmp     al, 9
    je      .val_done
.val_store:
    cmp     ecx, CMDLINE_BUF_SIZE - 1
    jae     .val_adv
    mov     [r12 + rcx], al
    inc     ecx
.val_adv:
    inc     rbx
    jmp     .val_lp

.end_quote:
    inc     rbx
.val_done:
    mov     byte [r12 + rcx], 0

    ; Match key against supported options
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_prompt]
    call    str_eq
    test    eax, eax
    jz      .chk_max_tok
    ; Copy cmd_val_buf -> opt_prompt_buf
    lea     rsi, [rel cmd_val_buf]
    lea     rdi, [rel opt_prompt_buf]
.cp_p:
    mov     al, [rsi]
    mov     [rdi], al
    inc     rsi
    inc     rdi
    test    al, al
    jnz     .cp_p
    mov     dword [rel opt_has_prompt], 1
    jmp     .tok_lp

.chk_max_tok:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_max_tokens]
    call    str_eq
    test    eax, eax
    jz      .chk_ctx
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_max_tok], rax
    jmp     .tok_lp

.chk_ctx:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_ctx]
    call    str_eq
    test    eax, eax
    jz      .chk_threads
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_ctx_len], rax
    jmp     .tok_lp

.chk_threads:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_threads]
    call    str_eq
    test    eax, eax
    jz      .chk_think
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_req_threads], rax
    jmp     .tok_lp

.chk_think:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_think]
    call    str_eq
    test    eax, eax
    jz      .chk_raw
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_think], eax
    jmp     .tok_lp

.chk_raw:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_raw]
    call    str_eq
    test    eax, eax
    jz      .chk_shutdown
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_raw], eax
    jmp     .tok_lp

.chk_shutdown:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_shutdown]
    call    str_eq
    test    eax, eax
    jz      .chk_ptoks
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_shutdown], eax
    jmp     .tok_lp

.chk_ptoks:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_print_tokens]
    call    str_eq
    test    eax, eax
    jz      .chk_temp
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_print_toks], eax
    jmp     .tok_lp

.chk_temp:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_temp]
    call    str_eq
    test    eax, eax
    jz      .chk_topp
    lea     rdi, [rel cmd_val_buf]
    call    parse_f32
    vmovss  [rel opt_temp], xmm0
    jmp     .tok_lp

.chk_topp:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_topp]
    call    str_eq
    test    eax, eax
    jz      .chk_seed
    lea     rdi, [rel cmd_val_buf]
    call    parse_f32
    vmovss  [rel opt_topp], xmm0
    jmp     .tok_lp

.chk_seed:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_seed]
    call    str_eq
    test    eax, eax
    jz      .chk_single_turn
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    test    rax, rax
    jnz     .st_seed
    mov     rax, 1
.st_seed:
    mov     [rel opt_rng], rax
    jmp     .tok_lp

.chk_single_turn:
    lea     rdi, [rel cmd_key_buf]
    lea     rsi, [rel key_single_turn]
    call    str_eq
    test    eax, eax
    jz      .tok_lp
    lea     rdi, [rel cmd_val_buf]
    call    parse_u64
    mov     [rel opt_single_turn], eax
    jmp     .tok_lp

.done:
    pop     r13
    pop     r12
    pop     rbx
    ret

; ============================================================================
; Read-Only Strings & Constants
; ============================================================================
section .rodata
align 8
d_1e6:          dq 1000000.0
d_1e9:          dq 1000000000.0

ban_line:       db "==============================================================================", 10, 0
ban_title:      db "  Qwenputer-X86 v1.0 | asm-llm OS v2.0 | Bare-Metal Qwen3-0.6B, Pure x86-64 Asm", 10, 0
ban_cpu:        db "  CPU   : ", 0
ban_avx_tsc:    db " (AVX2+FMA, ", 0
ban_mhz_nl:     db " MHz)", 10, 0
ban_smp:        db "  SMP   : ", 0
ban_smp_tail:   db " x86-64 cores online (APIC INIT-SIPI-SIPI, lock-free)", 10, 0
ban_ram:        db "  RAM   : ", 0
ban_ram_mid:    db " MiB | Arena: ", 0
ban_boot_lbl:   db " MiB | Boot: ", 0
ban_model:      db "  Model : ", 0
ban_qwen:       db "Qwen3-0.6B Q8_0 (", 0
ban_mod_at:     db " MiB @ 0x", 0
ban_mod_ctx:    db ", 28L, ctx=", 0

msg_prompt_lbl: db "Prompt: ", 0
msg_repl_help:  db "Interactive Bare-Metal Chat (/clear, /think, /nothink, /bench, /halt, /help)", 10, "Qwenputer-X86: /calc /put /get /del /list /info /qhelp", 10, 0
msg_repl_prompt: db 10, "qwenputer> ", 0
msg_chat_cleared: db "[chat] history cleared (ctx=0/", 0
msg_ctx_full:   db "[chat] context full, resetting KV-cache", 10, 0
msg_think_on:   db "[mode] think=1 (Qwen3 reasoning enabled)", 10, 0
msg_think_off:  db "[mode] think=0 (direct answer mode)", 10, 0
msg_poweroff:   db "Powering off bare-metal VM...", 10, 0

msg_nl:         db 10, 0
msg_comma:      db ","
msg_slash:      db "/", 0
msg_ptoks:      db "PROMPT_TOKENS:", 0
msg_gtoks:      db "GEN_TOKENS:", 0
msg_prefill:    db "[asm-llm-os] prefill: ", 0
msg_decode:     db "[asm-llm-os] decode : ", 0
msg_in:         db " tok in ", 0
msg_ms_open:    db " ms (", 0
msg_tps_end:    db " tok/s)", 10, 0
msg_tps_cores:  db " tok/s, cores=", 0
msg_ctx_lbl:    db ") ctx=", 0
msg_close_nl:   db ")", 10, 0
msg_boot_ttft:  db "[asm-llm-os] boot   : ready ", 0
msg_ttft_mid:   db " ms | boot->1st-tok ", 0
msg_ms_nl:      db " ms", 10, 0

def_cpu_brand:  db "x86-64 Processor", 0
bench_prompt:   db "Explain how a CPU cache works in 3 concise sentences.", 0

cmd_halt:       db "/halt", 0
cmd_quit:       db "/quit", 0
cmd_exit:       db "exit", 0
cmd_clear:      db "/clear", 0
cmd_new:        db "/new", 0
cmd_reset:      db "/reset", 0
cmd_think:      db "/think", 0
cmd_nothink:    db "/nothink", 0
cmd_bench:      db "/bench", 0
cmd_help:       db "/help", 0

key_prompt:       db "prompt", 0
key_max_tokens:   db "max_tokens", 0
key_ctx:          db "ctx", 0
key_threads:      db "threads", 0
key_think:        db "think", 0
key_raw:          db "raw", 0
key_shutdown:     db "shutdown", 0
key_print_tokens: db "print_tokens", 0
key_temp:         db "temp", 0
key_topp:         db "topp", 0
key_seed:         db "seed", 0
key_single_turn:  db "single_turn", 0

msg_err_avx2:   db "[FATAL] CPU lacks AVX2+FMA+XSAVE (use QEMU -cpu host)", 10, 0
msg_err_mb:     db "[FATAL] Invalid Multiboot 1 magic", 10, 0
msg_err_mod:    db "[FATAL] Missing .asmbin Multiboot module (pass -initrd model.asmbin)", 10, 0
msg_err_magic:  db "[FATAL] Invalid .asmbin magic (expected ASML)", 10, 0
msg_err_oom:    db "[FATAL] Not enough free physical RAM (use -m 2G)", 10, 0
msg_err_model:  db "[FATAL] model_init failed (arena OOM or bad header)", 10, 0

; ============================================================================
; Kernel State (.bss)
; ============================================================================
section .bss
alignb 8
g_mod_start:      resq 1
g_mod_end:        resq 1
g_mod_size:       resq 1
g_active_cores:   resq 1
g_chat_pos:       resq 1

opt_max_tok:      resq 1
opt_ctx_len:      resq 1
opt_req_threads:  resq 1
opt_rng:          resq 1
opt_temp:         resd 1
opt_topp:         resd 1
opt_think:        resd 1
opt_raw:          resd 1
opt_single_turn:  resd 1
opt_shutdown:     resd 1
opt_print_toks:   resd 1
opt_has_prompt:   resd 1
cur_show_ttft:    resd 1

tm_boot_ready:    resq 1
tm_prefill_start: resq 1
tm_prefill_end:   resq 1
tm_first_tok:     resq 1
tm_decode_start:  resq 1
tm_decode_end:    resq 1
tm_tmp:           resq 1

cpu_brand_raw:    resb 64
g_cpu_brand:      resb 64
g_cmdline_buf:    resb CMDLINE_BUF_SIZE
opt_prompt_buf:   resb CMDLINE_BUF_SIZE
cmd_key_buf:      resb 32
cmd_val_buf:      resb CMDLINE_BUF_SIZE
g_line_buf:       resb LINE_BUF_SIZE
