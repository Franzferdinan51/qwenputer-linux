; ============================================================================
; syscalls.asm — Raw Linux x86-64 syscall wrappers & tiny formatting helpers
; Zero libc, zero external libraries.
; ============================================================================

bits 64
default rel

%include "constants.inc"

section .text

global sys_openat_ro
global sys_openat_wr
global sys_close
global sys_fstat
global sys_mmap_file
global sys_mmap_anon
global sys_munmap
global sys_write_all
global sys_clock_ns
global sys_exit_group
global str_len
global str_eq
global parse_u64
global parse_f32
global write_cstr
global write_u64
global write_f64_2dp

; ----------------------------------------------------------------------------
; int64 sys_openat_ro(const char *path)
; rdi = path -> rax = fd (or negative errno)
; ----------------------------------------------------------------------------
sys_openat_ro:
    mov     rsi, rdi
    mov     rdi, AT_FDCWD
    xor     edx, edx            ; O_RDONLY
    xor     r10d, r10d
    mov     eax, SYS_openat
    syscall
    ret

; ----------------------------------------------------------------------------
; int64 sys_openat_wr(const char *path)
; rdi = path -> rax = fd (or negative errno)
; ----------------------------------------------------------------------------
sys_openat_wr:
    mov     rsi, rdi
    mov     rdi, AT_FDCWD
    mov     edx, O_WRONLY_CREAT_TR
    mov     r10d, MODE_0644
    mov     eax, SYS_openat
    syscall
    ret

; ----------------------------------------------------------------------------
; int64 sys_close(int fd)
; ----------------------------------------------------------------------------
sys_close:
    mov     eax, SYS_close
    syscall
    ret

; ----------------------------------------------------------------------------
; int64 sys_fstat(int fd, void *statbuf)
; ----------------------------------------------------------------------------
sys_fstat:
    mov     eax, SYS_fstat
    syscall
    ret

; ----------------------------------------------------------------------------
; void *sys_mmap_file(uint64 size, int fd)
; rdi = size, rsi = fd -> rax = mapped addr (or negative errno)
; ----------------------------------------------------------------------------
sys_mmap_file:
    mov     r8, rsi             ; fd
    mov     rsi, rdi            ; length
    xor     edi, edi            ; addr = NULL
    mov     edx, PROT_READ
    mov     r10d, MAP_PRIVATE
    xor     r9d, r9d            ; offset = 0
    mov     eax, SYS_mmap
    syscall
    ret

; ----------------------------------------------------------------------------
; void *sys_mmap_anon(uint64 size)
; rdi = size -> rax = mapped addr (zeroed, page-aligned)
; ----------------------------------------------------------------------------
sys_mmap_anon:
    mov     rsi, rdi            ; length
    xor     edi, edi            ; addr = NULL
    mov     edx, PROT_RW
    mov     r10d, MAP_PRIV_ANON
    mov     r8, -1              ; fd = -1
    xor     r9d, r9d            ; offset = 0
    mov     eax, SYS_mmap
    syscall
    ret

; ----------------------------------------------------------------------------
; int64 sys_munmap(void *addr, uint64 size)
; ----------------------------------------------------------------------------
sys_munmap:
    mov     eax, SYS_munmap
    syscall
    ret

; ----------------------------------------------------------------------------
; int64 sys_write_all(int fd, const void *buf, uint64 len)
; Handles partial writes reliably. Returns 0 on success, <0 on error.
; ----------------------------------------------------------------------------
sys_write_all:
    push    rbx
    push    r12
    push    r13
    mov     ebx, edi            ; fd
    mov     r12, rsi            ; buf
    mov     r13, rdx            ; remaining
.loop:
    test    r13, r13
    jz      .done
    mov     edi, ebx
    mov     rsi, r12
    mov     rdx, r13
    mov     eax, SYS_write
    syscall
    cmp     rax, -4             ; EINTR
    je      .loop
    test    rax, rax
    jle     .err
    add     r12, rax
    sub     r13, rax
    jmp     .loop
.done:
    xor     eax, eax
    pop     r13
    pop     r12
    pop     rbx
    ret
.err:
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; uint64 sys_clock_ns(void)
; Returns CLOCK_MONOTONIC time in nanoseconds.
; ----------------------------------------------------------------------------
sys_clock_ns:
    sub     rsp, 24             ; struct timespec { int64 tv_sec; int64 tv_nsec; }
    mov     edi, CLOCK_MONOTONIC
    mov     rsi, rsp
    mov     eax, SYS_clock_gettime
    syscall
    mov     rax, [rsp]          ; tv_sec
    imul    rax, rax, 1000000000
    add     rax, [rsp + 8]      ; + tv_nsec
    add     rsp, 24
    ret

; ----------------------------------------------------------------------------
; void sys_exit_group(int code)
; ----------------------------------------------------------------------------
sys_exit_group:
    mov     eax, SYS_exit_group
    syscall
    hlt

; ----------------------------------------------------------------------------
; uint64 str_len(const char *s)
; ----------------------------------------------------------------------------
str_len:
    xor     eax, eax
.lp:
    cmp     byte [rdi + rax], 0
    je      .ret
    inc     rax
    jmp     .lp
.ret:
    ret

; ----------------------------------------------------------------------------
; int str_eq(const char *a, const char *b) -> 1 if equal, 0 otherwise
; ----------------------------------------------------------------------------
str_eq:
.lp:
    movzx   eax, byte [rdi]
    movzx   edx, byte [rsi]
    cmp     al, dl
    jne     .ne
    test    al, al
    jz      .eq
    inc     rdi
    inc     rsi
    jmp     .lp
.eq:
    mov     eax, 1
    ret
.ne:
    xor     eax, eax
    ret

; ----------------------------------------------------------------------------
; uint64 parse_u64(const char *s)
; Parses unsigned decimal integer until non-digit.
; ----------------------------------------------------------------------------
parse_u64:
    xor     eax, eax
.lp:
    movzx   edx, byte [rdi]
    sub     edx, '0'
    cmp     edx, 9
    ja      .done
    imul    rax, rax, 10
    add     rax, rdx
    inc     rdi
    jmp     .lp
.done:
    ret

; ----------------------------------------------------------------------------
; float parse_f32(const char *s) -> xmm0
; Parses simple decimal float like "0.0", "0.6", "0.95", "1.25".
; ----------------------------------------------------------------------------
parse_f32:
    vxorps  xmm0, xmm0, xmm0    ; integer + frac accumulator (in double)
    vmovsd  xmm1, [rel d_ten]
    xor     ecx, ecx            ; sign flag
    cmp     byte [rdi], '-'
    jne     .int_lp
    mov     ecx, 1
    inc     rdi
.int_lp:
    movzx   eax, byte [rdi]
    sub     eax, '0'
    cmp     eax, 9
    ja      .check_dot
    vmulsd  xmm0, xmm0, xmm1
    vcvtsi2sd xmm2, xmm2, eax
    vaddsd  xmm0, xmm0, xmm2
    inc     rdi
    jmp     .int_lp
.check_dot:
    cmp     byte [rdi], '.'
    jne     .finish
    inc     rdi
    vmovsd  xmm3, [rel d_tenth] ; weight = 0.1
.frac_lp:
    movzx   eax, byte [rdi]
    sub     eax, '0'
    cmp     eax, 9
    ja      .finish
    vcvtsi2sd xmm2, xmm2, eax
    vmulsd  xmm2, xmm2, xmm3
    vaddsd  xmm0, xmm0, xmm2
    vmulsd  xmm3, xmm3, [rel d_tenth]
    inc     rdi
    jmp     .frac_lp
.finish:
    test    ecx, ecx
    jz      .to_f32
    vxorpd  xmm0, xmm0, [rel d_signbit]
.to_f32:
    vcvtsd2ss xmm0, xmm0, xmm0
    ret

; ----------------------------------------------------------------------------
; void write_cstr(int fd, const char *s)
; ----------------------------------------------------------------------------
write_cstr:
    push    rbx
    push    r12
    mov     ebx, edi
    mov     r12, rsi
    mov     rdi, rsi
    call    str_len
    mov     edi, ebx
    mov     rsi, r12
    mov     rdx, rax
    call    sys_write_all
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void write_u64(int fd, uint64 val)
; ----------------------------------------------------------------------------
write_u64:
    push    rbx
    sub     rsp, 32
    mov     ebx, edi            ; fd
    mov     rax, rsi            ; val
    lea     rcx, [rsp + 31]     ; write backwards
    mov     r8, rcx             ; end ptr
    mov     r9d, 10
    test    rax, rax
    jnz     .lp
    dec     rcx
    mov     byte [rcx], '0'
    jmp     .emit
.lp:
    xor     edx, edx
    div     r9
    add     dl, '0'
    dec     rcx
    mov     [rcx], dl
    test    rax, rax
    jnz     .lp
.emit:
    mov     edi, ebx
    mov     rsi, rcx
    mov     rdx, r8
    sub     rdx, rcx
    call    sys_write_all
    add     rsp, 32
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void write_f64_2dp(int fd, double val)
; Formats non-negative double with 2 decimal places (e.g., "123.45").
; ----------------------------------------------------------------------------
write_f64_2dp:
    push    rbx
    push    r12
    sub     rsp, 24
    mov     ebx, edi
    vmulsd  xmm0, xmm0, [rel d_100]
    vaddsd  xmm0, xmm0, [rel d_half]
    vcvttsd2si rax, xmm0        ; scaled integer (val * 100)
    test    rax, rax
    jns     .pos
    xor     eax, eax
.pos:
    xor     edx, edx
    mov     rcx, 100
    div     rcx                 ; rax = int part, rdx = frac (0..99)
    mov     r12, rdx            ; save frac
    mov     edi, ebx
    mov     rsi, rax
    call    write_u64
    ; Write ".XX"
    mov     byte [rsp], '.'
    mov     rax, r12
    xor     edx, edx
    mov     rcx, 10
    div     rcx                 ; rax = tens, rdx = ones
    add     al, '0'
    add     dl, '0'
    mov     [rsp + 1], al
    mov     [rsp + 2], dl
    mov     edi, ebx
    mov     rsi, rsp
    mov     edx, 3
    call    sys_write_all
    add     rsp, 24
    pop     r12
    pop     rbx
    ret

section .rodata
align 16
d_ten:      dq 10.0
d_tenth:    dq 0.1
d_100:      dq 100.0
d_half:     dq 0.5
d_signbit:  dq 0x8000000000000000, 0
