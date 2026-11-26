; ============================================================================
; qwenputer.asm — Qwenputer-X86 extensions to the asm-llm bare-metal OS
;
; Added by the qwenputer-linux project (Duckets / Juno), 2026-11-25.
; Upstream asm-llm (alesha-pro/llm.asm, MIT) ships chat only: the model can
; talk, but it cannot *do* anything exactly. Qwenputer's whole point — on the
; RISC-V in-weights side too — is a real computer next to the language model:
; exact arithmetic, a filesystem, and a notebook the model/user can rely on.
; This module ports that idea to the bare-metal x86 kernel:
;
;   /calc <expr>        exact signed 64-bit integer arithmetic:
;                       + - * / % , parentheses, unary minus, precedence.
;                       A language model guesses 123456789*987654321;
;                       this returns 121932631112635269, every time.
;   /put <name> <text>  store text in the RAM filesystem (32 files)
;   /get <name>         print a stored file
;   /del <name>         delete a stored file
;   /list , /files      list stored files with sizes
;   /info , /about      Qwenputer-X86 status (FS usage, feature list)
;   /qhelp              help for the Qwenputer commands
;
; ABI: int qp_handle_line(const char *line)
;   returns 1 if the line was a Qwenputer command (handled), else 0.
;   Called from kernel.asm's REPL before a line is sent to the model.
; Single-threaded REPL only; parser state is one global cursor.
; ============================================================================

bits 64
default rel

%include "os_constants.inc"

extern str_len
extern str_eq
extern write_cstr
extern write_u64
extern vga_set_color

global qp_handle_line

%define FS_SLOTS   32
%define FS_NAME_LEN 32
%define FS_DATA_LEN 256

section .text

; ---------------------------------------------------------------------------
; helpers: qp_puts(rsi = cstr) — print to console (dest 1 = serial+VGA)
; ---------------------------------------------------------------------------
qp_puts:
    mov     edi, 1
    jmp     write_cstr

qp_nl:
    lea     rsi, [rel s_nl]
    jmp     qp_puts

; qp_puti64(rsi = signed value)
qp_puti64:
    test    rsi, rsi
    jns     .pos
    push    rsi
    lea     rsi, [rel s_minus]
    mov     edi, 1
    call    write_cstr
    pop     rsi
    neg     rsi
.pos:
    mov     edi, 1
    jmp     write_u64

; ---------------------------------------------------------------------------
; qp_prefix(rdi = line, rsi = prefix) -> rax = ptr just past prefix if the
; line starts with prefix AND the next char is space/tab/NUL, else 0.
; ---------------------------------------------------------------------------
qp_prefix:
.lp:
    movzx   eax, byte [rsi]
    test    al, al
    jz      .matched
    cmp     al, byte [rdi]
    jne     .no
    inc     rdi
    inc     rsi
    jmp     .lp
.matched:
    movzx   eax, byte [rdi]
    cmp     al, 0
    je      .yes
    cmp     al, ' '
    je      .yes
    cmp     al, 9
    je      .yes
    xor     eax, eax
    ret
.yes:
    mov     rax, rdi
    ret
.no:
    xor     eax, eax
    ret

; skip spaces at [qp_ptr]
qp_skip:
    mov     rax, [rel qp_ptr]
.lp:
    cmp     byte [rax], ' '
    je      .adv
    cmp     byte [rax], 9
    je      .adv
    jmp     .done
.adv:
    inc     rax
    jmp     .lp
.done:
    mov     [rel qp_ptr], rax
    ret

; ---------------------------------------------------------------------------
; expression parser (recursive descent, signed 64-bit, wrap on overflow)
;   parse_expr   := parse_term { ('+'|'-') parse_term }
;   parse_term   := parse_factor { ('*'|'/'|'%') parse_factor }
;   parse_factor := ('-'|'+') parse_factor | '(' parse_expr ')' | number
; result in rax; qp_err set on any error (incl. divide by zero)
; ---------------------------------------------------------------------------
parse_expr:
    push    rbx
    call    parse_term
    mov     rbx, rax
.loop:
    call    qp_skip
    mov     rax, [rel qp_ptr]
    movzx   ecx, byte [rax]
    cmp     cl, '+'
    je      .add
    cmp     cl, '-'
    je      .sub
    mov     rax, rbx
    pop     rbx
    ret
.add:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_term
    add     rbx, rax
    jmp     .loop
.sub:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_term
    sub     rbx, rax
    jmp     .loop

parse_term:
    push    rbx
    call    parse_factor
    mov     rbx, rax
.loop:
    call    qp_skip
    mov     rax, [rel qp_ptr]
    movzx   ecx, byte [rax]
    cmp     cl, '*'
    je      .mul
    cmp     cl, '/'
    je      .div
    cmp     cl, '%'
    je      .mod
    mov     rax, rbx
    pop     rbx
    ret
.mul:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_factor
    imul    rbx, rax
    jmp     .loop
.div:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_factor
    test    rax, rax
    jz      .divzero
    cmp     rax, -1
    je      .div_neg1
    mov     rcx, rax
    mov     rax, rbx
    cqo
    idiv    rcx
    mov     rbx, rax
    jmp     .loop
.div_neg1:
    neg     rbx                     ; INT64_MIN / -1 wraps to INT64_MIN
    jmp     .loop
.mod:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_factor
    test    rax, rax
    jz      .divzero
    cmp     rax, -1
    je      .mod_neg1
    mov     rcx, rax
    mov     rax, rbx
    cqo
    idiv    rcx
    mov     rbx, rdx
    jmp     .loop
.mod_neg1:
    xor     ebx, ebx                ; x % -1 == 0 (incl. INT64_MIN)
    jmp     .loop
.divzero:
    mov     dword [rel qp_err], 1
    xor     ebx, ebx
    mov     rax, rbx
    pop     rbx
    ret

parse_factor:
    call    qp_skip
    mov     rax, [rel qp_ptr]
    movzx   ecx, byte [rax]
    cmp     cl, '-'
    je      .neg
    cmp     cl, '+'
    je      .pos
    cmp     cl, '('
    je      .paren
    cmp     cl, '0'
    jb      .bad
    cmp     cl, '9'
    ja      .bad
    ; number
    xor     eax, eax
.num_lp:
    mov     rdx, [rel qp_ptr]
    movzx   ecx, byte [rdx]
    sub     ecx, '0'
    cmp     ecx, 9
    ja      .num_done
    imul    rax, rax, 10
    add     rax, rcx
    inc     rdx
    mov     [rel qp_ptr], rdx
    jmp     .num_lp
.num_done:
    ret
.neg:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_factor
    neg     rax
    ret
.pos:
    inc     rax
    mov     [rel qp_ptr], rax
    jmp     parse_factor
.paren:
    inc     rax
    mov     [rel qp_ptr], rax
    call    parse_expr
    push    rax
    call    qp_skip
    mov     rax, [rel qp_ptr]
    cmp     byte [rax], ')'
    jne     .paren_err
    inc     rax
    mov     [rel qp_ptr], rax
    pop     rax
    ret
.paren_err:
    mov     dword [rel qp_err], 1
    pop     rax
    ret
.bad:
    mov     dword [rel qp_err], 1
    xor     eax, eax
    ret

; ---------------------------------------------------------------------------
; /calc handler: rdi = ptr just past "/calc"
; ---------------------------------------------------------------------------
do_calc:
    mov     [rel qp_ptr], rdi
    mov     dword [rel qp_err], 0
    call    qp_skip
    mov     rax, [rel qp_ptr]
    cmp     byte [rax], 0
    je      .usage
    call    parse_expr
    push    rax
    call    qp_skip
    mov     rax, [rel qp_ptr]
    cmp     byte [rax], 0
    jne     .badexpr
    cmp     dword [rel qp_err], 0
    jne     .badexpr
    lea     rsi, [rel s_calc]
    call    qp_puts
    pop     rsi
    call    qp_puti64
    call    qp_nl
    mov     eax, 1
    ret
.badexpr:
    pop     rax
    lea     rsi, [rel s_calc_err]
    call    qp_puts
    mov     eax, 1
    ret
.usage:
    lea     rsi, [rel s_calc_usage]
    call    qp_puts
    mov     eax, 1
    ret

; ---------------------------------------------------------------------------
; RAM filesystem
;   fs_parse_name(rdi = ptr) -> qp_name filled, rax = ptr past the name
; ---------------------------------------------------------------------------
fs_parse_name:
    push    rbx
    mov     rbx, rdi
    call    qp_skip_ws_rbx          ; advance rbx past spaces (inline below)
    lea     rsi, [rel qp_name]
    xor     ecx, ecx
.lp:
    movzx   eax, byte [rbx]
    test    al, al
    jz      .done
    cmp     al, ' '
    je      .done
    cmp     al, 9
    je      .done
    cmp     ecx, FS_NAME_LEN - 1
    jae     .done
    mov     [rsi + rcx], al
    inc     rbx
    inc     rcx
    jmp     .lp
.done:
    mov     byte [rsi + rcx], 0
    mov     rax, rbx
    pop     rbx
    ret

qp_skip_ws_rbx:
.lp:
    cmp     byte [rbx], ' '
    je      .adv
    cmp     byte [rbx], 9
    je      .adv
    ret
.adv:
    inc     rbx
    jmp     .lp

; fs_find() -> rax = slot index or -1 (uses qp_name)
fs_find:
    xor     ecx, ecx
.lp:
    cmp     ecx, FS_SLOTS
    jae     .notfound
    lea     rax, [rel fs_used]
    cmp     byte [rax + rcx], 0
    je      .next
    mov     eax, ecx
    imul    rax, rax, FS_NAME_LEN
    lea     rdi, [rel fs_names]
    add     rdi, rax
    lea     rsi, [rel qp_name]
    push    rcx
    call    str_eq
    pop     rcx
    test    eax, eax
    jnz     .found
.next:
    inc     ecx
    jmp     .lp
.found:
    mov     eax, ecx
    ret
.notfound:
    mov     rax, -1
    ret

; fs_find_or_alloc() -> rax = slot index or -1 when full
fs_find_or_alloc:
    call    fs_find
    test    rax, rax
    jns     .done
    xor     ecx, ecx
.lp:
    cmp     ecx, FS_SLOTS
    jae     .full
    lea     rax, [rel fs_used]
    cmp     byte [rax + rcx], 0
    je      .alloc
    inc     ecx
    jmp     .lp
.alloc:
    mov     eax, ecx
.done:
    ret
.full:
    mov     rax, -1
    ret

; /put: rdi = ptr past "/put"
do_put:
    push    rbx
    push    r12
    push    r13
    call    fs_parse_name           ; rax = ptr past name
    mov     rbx, rax
    lea     rax, [rel qp_name]
    cmp     byte [rax], 0
    je      .usage
    ; skip spaces before data
    mov     rdi, rbx
    call    qp_skip_ws_rbx2
    mov     rbx, rdi
    call    fs_find_or_alloc
    test    rax, rax
    js      .full
    mov     r12, rax                ; slot
    ; mark used + copy name
    lea     rax, [rel fs_used]
    mov     byte [rax + r12], 1
    mov     rax, r12
    imul    rax, rax, FS_NAME_LEN
    lea     rdi, [rel fs_names]
    add     rdi, rax
    lea     rsi, [rel qp_name]
    xor     ecx, ecx
.copy_name:
    movzx   eax, byte [rsi + rcx]
    mov     [rdi + rcx], al
    test    al, al
    jz      .name_done
    inc     rcx
    jmp     .copy_name
.name_done:
    ; copy data from rbx, up to FS_DATA_LEN-1
    mov     rax, r12
    imul    rax, rax, FS_DATA_LEN
    lea     rdi, [rel fs_data]
    add     rdi, rax
    xor     ecx, ecx
.copy_data:
    cmp     ecx, FS_DATA_LEN - 1
    jae     .data_done
    movzx   eax, byte [rbx + rcx]
    test    al, al
    jz      .data_done
    mov     [rdi + rcx], al
    inc     rcx
    jmp     .copy_data
.data_done:
    mov     byte [rdi + rcx], 0
    mov     r13, rcx                ; length survives write calls (syscall clobbers rcx)
    lea     rsi, [rel s_put_ok]
    call    qp_puts
    lea     rsi, [rel qp_name]
    call    qp_puts
    lea     rsi, [rel s_put_mid]
    call    qp_puts
    mov     rsi, r13
    mov     edi, 1
    call    write_u64
    lea     rsi, [rel s_put_end]
    call    qp_puts
    pop     r13
    pop     r12
    pop     rbx
    mov     eax, 1
    ret
.usage:
    lea     rsi, [rel s_put_usage]
    call    qp_puts
    pop     r13
    pop     r12
    pop     rbx
    mov     eax, 1
    ret
.full:
    lea     rsi, [rel s_fs_full]
    call    qp_puts
    pop     r13
    pop     r12
    pop     rbx
    mov     eax, 1
    ret

qp_skip_ws_rbx2:                    ; rdi in/out
.lp:
    cmp     byte [rdi], ' '
    je      .adv
    cmp     byte [rdi], 9
    je      .adv
    ret
.adv:
    inc     rdi
    jmp     .lp

; /get: rdi = ptr past "/get"
do_get:
    call    fs_parse_name
    call    fs_find
    test    rax, rax
    js      .notfound
    imul    rax, rax, FS_DATA_LEN
    lea     rsi, [rel fs_data]
    add     rsi, rax
    call    qp_puts
    call    qp_nl
    mov     eax, 1
    ret
.notfound:
    lea     rsi, [rel s_notfound]
    call    qp_puts
    mov     eax, 1
    ret

; /del: rdi = ptr past "/del"
do_del:
    call    fs_parse_name
    call    fs_find
    test    rax, rax
    js      .notfound
    lea     rcx, [rel fs_used]
    mov     byte [rcx + rax], 0
    imul    rax, rax, FS_NAME_LEN
    lea     rcx, [rel fs_names]
    mov     byte [rcx + rax], 0
    lea     rsi, [rel s_del_ok]
    call    qp_puts
    lea     rsi, [rel qp_name]
    call    qp_puts
    lea     rsi, [rel s_quote_nl]
    call    qp_puts
    mov     eax, 1
    ret
.notfound:
    lea     rsi, [rel s_notfound]
    call    qp_puts
    mov     eax, 1
    ret

; /list
do_list:
    push    rbx
    push    r12
    xor     ebx, ebx                ; slot
    xor     r12d, r12d              ; count
.lp:
    cmp     ebx, FS_SLOTS
    jae     .done
    lea     rax, [rel fs_used]
    cmp     byte [rax + rbx], 0
    je      .next
    inc     r12d
    lea     rsi, [rel s_list_pre]
    call    qp_puts
    mov     eax, ebx
    imul    rax, rax, FS_NAME_LEN
    lea     rsi, [rel fs_names]
    add     rsi, rax
    call    qp_puts
    lea     rsi, [rel s_list_mid]
    call    qp_puts
    mov     eax, ebx
    imul    rax, rax, FS_DATA_LEN
    lea     rdi, [rel fs_data]
    add     rdi, rax
    call    str_len
    mov     rsi, rax
    mov     edi, 1
    call    write_u64
    lea     rsi, [rel s_list_end]
    call    qp_puts
.next:
    inc     ebx
    jmp     .lp
.done:
    test    r12d, r12d
    jnz     .ret
    lea     rsi, [rel s_fs_empty]
    call    qp_puts
.ret:
    pop     r12
    pop     rbx
    mov     eax, 1
    ret

; fs_count() -> rax = used slots
fs_count:
    xor     eax, eax
    xor     ecx, ecx
.lp:
    cmp     ecx, FS_SLOTS
    jae     .done
    lea     rdx, [rel fs_used]
    cmp     byte [rdx + rcx], 0
    je      .next
    inc     rax
.next:
    inc     ecx
    jmp     .lp
.done:
    ret

; /info
do_info:
    lea     rsi, [rel s_info1]
    call    qp_puts
    call    fs_count
    mov     rsi, rax
    mov     edi, 1
    call    write_u64
    lea     rsi, [rel s_info2]
    call    qp_puts
    mov     eax, 1
    ret

; /qhelp
do_qhelp:
    lea     rsi, [rel s_qhelp]
    call    qp_puts
    mov     eax, 1
    ret

; ---------------------------------------------------------------------------
; int qp_handle_line(const char *line in rdi)
; ---------------------------------------------------------------------------
qp_handle_line:
    push    rbx
    mov     rbx, rdi

    lea     rsi, [rel c_calc]
    call    qp_prefix
    test    rax, rax
    jz      .t_put
    mov     rdi, rax
    call    do_calc
    pop     rbx
    ret

.t_put:
    mov     rdi, rbx
    lea     rsi, [rel c_put]
    call    qp_prefix
    test    rax, rax
    jz      .t_get
    mov     rdi, rax
    call    do_put
    pop     rbx
    ret

.t_get:
    mov     rdi, rbx
    lea     rsi, [rel c_get]
    call    qp_prefix
    test    rax, rax
    jz      .t_del
    mov     rdi, rax
    call    do_get
    pop     rbx
    ret

.t_del:
    mov     rdi, rbx
    lea     rsi, [rel c_del]
    call    qp_prefix
    test    rax, rax
    jz      .t_list
    mov     rdi, rax
    call    do_del
    pop     rbx
    ret

.t_list:
    mov     rdi, rbx
    lea     rsi, [rel c_list]
    call    str_eq
    test    eax, eax
    jnz     .yes_list
    mov     rdi, rbx
    lea     rsi, [rel c_files]
    call    str_eq
    test    eax, eax
    jz      .t_info
.yes_list:
    call    do_list
    pop     rbx
    ret

.t_info:
    mov     rdi, rbx
    lea     rsi, [rel c_info]
    call    str_eq
    test    eax, eax
    jnz     .yes_info
    mov     rdi, rbx
    lea     rsi, [rel c_about]
    call    str_eq
    test    eax, eax
    jnz     .yes_info
    mov     rdi, rbx
    lea     rsi, [rel c_qp]
    call    str_eq
    test    eax, eax
    jz      .t_qhelp
.yes_info:
    call    do_info
    pop     rbx
    ret

.t_qhelp:
    mov     rdi, rbx
    lea     rsi, [rel c_qhelp]
    call    str_eq
    test    eax, eax
    jz      .none
    call    do_qhelp
    pop     rbx
    ret

.none:
    xor     eax, eax
    pop     rbx
    ret

; ============================================================================
section .rodata
align 8
s_nl:         db 10, 0
s_minus:      db "-", 0
s_calc:       db "[calc] ", 0
s_calc_err:   db "[calc] error (bad expression or divide by zero)", 10, 0
s_calc_usage: db "[calc] usage: /calc <expr>  e.g. /calc 123456789*987654321", 10, 0
s_put_ok:     db "[fs] stored '", 0
s_put_mid:    db "' (", 0
s_put_end:    db " B)", 10, 0
s_put_usage:  db "[fs] usage: /put <name> <text>", 10, 0
s_fs_full:    db "[fs] filesystem full (32 files)", 10, 0
s_notfound:   db "[fs] not found", 10, 0
s_del_ok:     db "[fs] deleted '", 0
s_quote_nl:   db "'", 10, 0
s_list_pre:   db "  ", 0
s_list_mid:   db "  (", 0
s_list_end:   db " B)", 10, 0
s_fs_empty:   db "[fs] empty - /put <name> <text> to store something", 10, 0
s_info1:      db "[qwenputer] Qwenputer-X86 v1.0 | bare-metal Qwen3-0.6B + exact calc + RAM filesystem | files: ", 0
s_info2:      db "/32 | commands: /calc /put /get /del /list /info /qhelp", 10, 0
s_qhelp:      db "Qwenputer commands:", 10
              db "  /calc <expr>       exact 64-bit integer math (+ - * / %, parentheses)", 10
              db "  /put <name> <text> store text in the RAM filesystem", 10
              db "  /get <name>        print a stored file", 10
              db "  /del <name>        delete a stored file", 10
              db "  /list              list stored files", 10
              db "  /info              Qwenputer-X86 status", 10
              db "  (upstream chat: /clear /think /nothink /bench /halt /help)", 10, 0

c_calc:  db "/calc", 0
c_put:   db "/put", 0
c_get:   db "/get", 0
c_del:   db "/del", 0
c_list:  db "/list", 0
c_files: db "/files", 0
c_info:  db "/info", 0
c_about: db "/about", 0
c_qp:    db "/qp", 0
c_qhelp: db "/qhelp", 0

; ============================================================================
section .bss
alignb 8
qp_ptr:   resq 1
qp_err:   resd 1
qp_name:  resb 64
fs_used:  resb FS_SLOTS
fs_names: resb FS_SLOTS * FS_NAME_LEN
fs_data:  resb FS_SLOTS * FS_DATA_LEN
