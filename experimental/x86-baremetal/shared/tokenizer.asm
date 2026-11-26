; ============================================================================
; tokenizer.asm — Pure x86-64 BPE Encoder, UTF-8 Decoder & Qwen3 Chat Template
;
; Uses the pre-built Fibonacci-hashed open-addressing merge table directly
; from the mmap'd .asmbin file (zero heap allocations, zero libc).
; ============================================================================

bits 64
default rel

%include "constants.inc"

extern g_ctx
extern sys_write_all
extern str_len

section .text

global tok_encode_bpe
global tok_build_chat_prompt
global tok_write_stdout

; ----------------------------------------------------------------------------
; uint64 tok_encode_bpe(const uint8 *text, uint64 len, uint32 *out_toks, uint64 max_toks)
; rdi = text, rsi = len, rdx = out_toks, rcx = max_toks -> rax = n_tokens
; ----------------------------------------------------------------------------
tok_encode_bpe:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    cmp     rsi, rcx
    jbe     .len_ok
    mov     rsi, rcx
.len_ok:
    test    rsi, rsi
    jz      .ret_zero

    lea     r12, [rel g_ctx]
    mov     r8,  [r12 + CTX_B2T_PTR]    ; uint32 byte_to_tok[256]
    mov     r13, [r12 + CTX_HT_PTR]     ; MergeEntry[524288]
    mov     r14, [r12 + CTX_HT_MASK]    ; 524287
    mov     r15, rdx                    ; out_toks
    mov     rbx, rsi                    ; n = len

    ; Step 1: map each raw UTF-8 byte -> single-byte base token ID
    xor     eax, eax
.map_lp:
    cmp     rax, rbx
    jae     .bpe_loop
    movzx   ecx, byte [rdi + rax]
    mov     ecx, [r8 + rcx*4]
    mov     [r15 + rax*4], ecx
    inc     rax
    jmp     .map_lp

    ; Step 2: Greedy lowest-rank BPE merge loop
.bpe_loop:
    cmp     rbx, 2
    jb      .done

    mov     ebp, HT_EMPTY_RANK          ; best_rank = 0xFFFFFFFF
    mov     r9, -1                      ; best_idx = -1
    xor     r10d, r10d                  ; best_merged = 0

    lea     r11, [rbx - 1]              ; limit = n - 1
    xor     ecx, ecx                    ; i = 0
    mov     rsi, FIB_MULT_64            ; Fibonacci multiplier
.scan_lp:
    cmp     rcx, r11
    jae     .scan_done

    ; key = ((uint64)toks[i] << 32) | toks[i + 1]
    mov     eax, [r15 + rcx*4]
    shl     rax, 32
    mov     edx, [r15 + rcx*4 + 4]
    or      rax, rdx                    ; rax = key

    ; slot = (key * FIB_MULT_64) >> 45
    mov     rdx, rax
    imul    rdx, rsi
    shr     rdx, HT_SHIFT

.probe_lp:
    mov     rdi, rdx
    shl     rdi, 4                      ; slot * 16 bytes
    mov     r8d, [r13 + rdi + HT_RANK]
    cmp     r8d, HT_EMPTY_RANK
    je      .scan_next                  ; empty slot -> pair not in merge table
    cmp     [r13 + rdi + HT_KEY], rax
    je      .found_pair
    inc     rdx
    and     rdx, r14                    ; (slot + 1) & ht_mask
    jmp     .probe_lp

.found_pair:
    cmp     r8d, ebp                    ; rank < best_rank?
    jae     .scan_next
    mov     ebp, r8d
    mov     r9, rcx
    mov     r10d, [r13 + rdi + HT_MERGED]

.scan_next:
    inc     rcx
    jmp     .scan_lp

.scan_done:
    cmp     r9, 0
    jl      .done                       ; no mergeable pair found!

    ; Apply best merge: toks[best_idx] = best_merged, shift tail left by 1
    mov     [r15 + r9*4], r10d
    lea     rcx, [r9 + 1]
    lea     rdx, [rbx - 1]
.shift_lp:
    cmp     rcx, rdx
    jae     .shift_done
    mov     eax, [r15 + rcx*4 + 4]
    mov     [r15 + rcx*4], eax
    inc     rcx
    jmp     .shift_lp
.shift_done:
    dec     rbx
    jmp     .bpe_loop

.ret_zero:
    xor     eax, eax
    jmp     .exit
.done:
    mov     rax, rbx
.exit:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; uint64 tok_build_chat_prompt(const char *prompt, uint32 *out_toks,
;                              uint64 max_toks, int raw_mode, int think_mode)
; rdi = prompt, rsi = out_toks, rdx = max_toks, rcx = raw_mode, r8 = think_mode
; Returns total number of tokens written to out_toks.
; ----------------------------------------------------------------------------
tok_build_chat_prompt:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15                         ; 5 pushes -> 16B aligned

    mov     r12, rdi                    ; prompt
    mov     r13, rsi                    ; out_toks
    mov     r14, rdx                    ; max_toks
    mov     r15, rcx                    ; raw_mode
    mov     rbx, r8                     ; think_mode

    mov     rdi, r12
    call    str_len                     ; rax = prompt byte length

    test    r15, r15
    jz      .chat_mode

    ; Raw mode: encode prompt directly without chat tags
    mov     rdi, r12
    mov     rsi, rax
    mov     rdx, r13
    mov     rcx, r14
    call    tok_encode_bpe
    jmp     .done

.chat_mode:
    ; Prefix: <|im_start|> user \n  (3 tokens)
    mov     dword [r13 + 0], TOK_IM_START
    mov     dword [r13 + 4], TOK_USER
    mov     dword [r13 + 8], TOK_NL

    ; Encode user message into out_toks + 3
    mov     rdi, r12
    mov     rsi, rax
    lea     rdx, [r13 + 12]
    lea     rcx, [r14 - 16]
    call    tok_encode_bpe
    lea     rcx, [rax + 3]              ; current count

    ; Suffix: <|im_end|> \n <|im_start|> assistant \n (5 tokens)
    mov     dword [r13 + rcx*4 + 0],  TOK_IM_END
    mov     dword [r13 + rcx*4 + 4],  TOK_NL
    mov     dword [r13 + rcx*4 + 8],  TOK_IM_START
    mov     dword [r13 + rcx*4 + 12], TOK_ASSISTANT
    mov     dword [r13 + rcx*4 + 16], TOK_NL
    add     rcx, 5

    ; If think_mode == 0, append empty think block: <think>\n\n</think>\n\n (4 tokens)
    test    rbx, rbx
    jnz     .ret_cnt
    mov     dword [r13 + rcx*4 + 0],  TOK_THINK_START
    mov     dword [r13 + rcx*4 + 4],  TOK_NL2
    mov     dword [r13 + rcx*4 + 8],  TOK_THINK_END
    mov     dword [r13 + rcx*4 + 12], TOK_NL2
    add     rcx, 4

.ret_cnt:
    mov     rax, rcx
.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void tok_write_stdout(uint32 token_id)
; rdi = token_id -> writes raw UTF-8 token bytes directly to stdout (fd=1).
; ----------------------------------------------------------------------------
tok_write_stdout:
    lea     rax, [rel g_ctx]
    cmp     rdi, [rax + CTX_VOCAB_SIZE]
    jae     .ret
    mov     rcx, [rax + CTX_TOK_OFFS_PTR]
    mov     edx, [rcx + rdi*4]          ; start offset
    mov     r8d, [rcx + rdi*4 + 4]      ; end offset
    sub     r8, rdx                     ; byte length
    jz      .ret
    mov     rsi, [rax + CTX_TOK_BYTES_PTR]
    add     rsi, rdx                    ; buf ptr
    mov     edi, 1                      ; stdout
    mov     rdx, r8                     ; len
    jmp     sys_write_all               ; tail call!
.ret:
    ret
