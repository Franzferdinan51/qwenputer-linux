; ============================================================================
; os/runtime.asm — Bare-Metal Platform Layer (Bump Allocator + String/Fmt API)
;
; Drop-in bare-metal replacement for src/syscalls.asm:
;   - sys_write_all -> unified COM1 + VGA console output
;   - sys_mmap_anon -> 64-byte-aligned AVX2-zeroed bump allocator over the
;                      largest free physical RAM region from Multiboot mmap
;   - sys_clock_ns  -> PIT-calibrated RDTSC nanosecond timer
;   - String & number formatting/parsing utilities (identical ABI to Phase 1)
; ============================================================================

bits 64
default rel

%include "os_constants.inc"

extern _kernel_end
extern console_write
extern timer_ns

section .text

global mem_init
global sys_mmap_anon
global sys_munmap
global sys_write_all
global sys_clock_ns
global str_len
global str_eq
global parse_u64
global parse_f32
global write_cstr
global write_u64
global write_f64_2dp
global write_hex64_raw

global g_total_ram_mb
global g_heap_base
global g_heap_cur
global g_heap_end
global g_arena_used_mb

; ----------------------------------------------------------------------------
; int64 mem_init(const void *mbi in rdi, uint64 mod_start in rsi, uint64 mod_end in rdx)
; Finds the largest usable RAM region below 0xF0000000 (3.75 GiB, strictly
; inside our 4 GiB identity map and below Local APIC MMIO) that does NOT
; overlap [0, kernel_end) or [mod_start, mod_end).
; Returns 0 on success, -1 if no usable heap found.
; ----------------------------------------------------------------------------
mem_init:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    ; Align mod_start down to 4 KiB, mod_end up to 4 KiB
    and     rsi, -4096
    add     rdx, 4095
    and     rdx, -4096
    mov     r12, rsi                    ; mod_start_aligned
    mov     r13, rdx                    ; mod_end_aligned

    ; Minimum heap floor = max(0x00200000, align_up(_kernel_end, 4096))
    lea     rax, [rel _kernel_end]
    add     rax, 4095
    and     rax, -4096
    cmp     rax, 0x00200000
    jae     .floor_ok
    mov     rax, 0x00200000
.floor_ok:
    mov     r14, rax                    ; min_floor

    ; Track best free region [best_start, best_end) and total usable RAM bytes
    xor     r15d, r15d                  ; best_size = 0
    mov     qword [rel g_heap_base], 0
    mov     qword [rel g_heap_end], 0
    mov     qword [rel total_ram_bytes], 0

    ; Check Multiboot mmap flag (bit 6 = 0x40)
    mov     eax, [rdi + MBI_FLAGS]
    test    eax, 0x40
    jz      .fallback_mem_upper

    mov     ebx, [rdi + MBI_MMAP_ADDR]  ; cur ptr (32-bit phys addr)
    mov     ebp, [rdi + MBI_MMAP_LENGTH]
    add     rbp, rbx                    ; end ptr

.mmap_lp:
    cmp     rbx, rbp
    jae     .mmap_done

    ; Multiboot mmap entry:
    ;   [rbx + 0]  = uint32 size (size of entry excluding this 4-byte field)
    ;   [rbx + 4]  = uint64 base_addr
    ;   [rbx + 12] = uint64 length
    ;   [rbx + 20] = uint32 type (1 = usable RAM)
    mov     eax, [rbx + 20]
    cmp     eax, 1
    jne     .mmap_next

    mov     r8,  [rbx + 4]              ; region start
    mov     r9,  [rbx + 12]             ; region length
    add     [rel total_ram_bytes], r9
    lea     r9,  [r8 + r9]              ; region end = start + length

    ; Clamp to [min_floor, 0xF0000000)
    cmp     r8, r14
    jae     .s_ok
    mov     r8, r14
.s_ok:
    mov     rax, 0xF0000000
    cmp     r9, rax
    jbe     .e_ok
    mov     r9, rax
.e_ok:
    cmp     r9, r8
    jbe     .mmap_next

    ; Subtract [mod_start, mod_end) -> up to 2 candidate intervals:
    ;   Candidate 1 (below module): [r8, min(r9, mod_start))
    ;   Candidate 2 (above module): [max(r8, mod_end), r9)
    cmp     r13, r12                    ; no module?
    jbe     .try_whole

    ; Check if region overlaps module at all
    cmp     r9, r12
    jbe     .try_whole
    cmp     r8, r13
    jae     .try_whole

    ; Candidate 1: [r8, mod_start)
    mov     rax, r12
    cmp     rax, r8
    jbe     .try_above
    mov     rcx, rax
    sub     rcx, r8
    cmp     rcx, r15
    jbe     .try_above
    mov     r15, rcx
    mov     [rel g_heap_base], r8
    mov     [rel g_heap_end], rax

.try_above:
    ; Candidate 2: [mod_end, r9)
    cmp     r9, r13
    jbe     .mmap_next
    mov     rcx, r9
    sub     rcx, r13
    cmp     rcx, r15
    jbe     .mmap_next
    mov     r15, rcx
    mov     [rel g_heap_base], r13
    mov     [rel g_heap_end], r9
    jmp     .mmap_next

.try_whole:
    mov     rcx, r9
    sub     rcx, r8
    cmp     rcx, r15
    jbe     .mmap_next
    mov     r15, rcx
    mov     [rel g_heap_base], r8
    mov     [rel g_heap_end], r9

.mmap_next:
    mov     eax, [rbx + 0]              ; entry size
    add     rbx, rax
    add     rbx, 4
    jmp     .mmap_lp

.fallback_mem_upper:
    ; Use MBI_MEM_UPPER (KiB above 1 MiB)
    mov     eax, [rdi + MBI_MEM_UPPER]
    shl     rax, 10                     ; bytes above 1 MiB
    lea     r9, [rax + 0x00100000]
    mov     [rel total_ram_bytes], r9
    mov     rax, 0xF0000000
    cmp     r9, rax
    jbe     .fb_eok
    mov     r9, rax
.fb_eok:
    ; Test below & above module inside [r14, r9)
    mov     r8, r14
    cmp     r12, r8
    jbe     .fb_above
    mov     rax, r12
    cmp     rax, r9
    jbe     .fb_c1
    mov     rax, r9
.fb_c1:
    mov     rcx, rax
    sub     rcx, r8
    mov     r15, rcx
    mov     [rel g_heap_base], r8
    mov     [rel g_heap_end], rax
.fb_above:
    cmp     r9, r13
    jbe     .mmap_done
    mov     rcx, r9
    sub     rcx, r13
    cmp     rcx, r15
    jbe     .mmap_done
    mov     r15, rcx
    mov     [rel g_heap_base], r13
    mov     [rel g_heap_end], r9

.mmap_done:
    ; Align heap_base up to 4096 bytes
    mov     rax, [rel g_heap_base]
    add     rax, 4095
    and     rax, -4096
    mov     [rel g_heap_base], rax
    mov     [rel g_heap_cur], rax

    ; Compute g_total_ram_mb = (total_ram_bytes + 524288) >> 20
    mov     rax, [rel total_ram_bytes]
    add     rax, 524288
    shr     rax, 20
    mov     [rel g_total_ram_mb], rax

    cmp     r15, 0x01000000             ; need at least 16 MiB free
    jb      .err
    xor     eax, eax
    jmp     .ret
.err:
    mov     rax, -1
.ret:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void *sys_mmap_anon(uint64 size in rdi)
; Allocates 64-byte-aligned zeroed memory from the bare-metal physical arena.
; Returns pointer on success, -1 on OOM.
; ----------------------------------------------------------------------------
sys_mmap_anon:
    mov     rax, [rel g_heap_cur]
    add     rax, 63
    and     rax, -64                    ; aligned start ptr

    mov     rcx, rdi
    add     rcx, 63
    and     rcx, -64                    ; aligned size (multiple of 64 bytes)

    lea     rdx, [rax + rcx]
    cmp     rdx, [rel g_heap_end]
    ja      .oom

    mov     [rel g_heap_cur], rdx

    ; Track allocated MiB for banner display
    mov     r8, rdx
    sub     r8, [rel g_heap_base]
    add     r8, 524288
    shr     r8, 20
    mov     [rel g_arena_used_mb], r8

    ; Fast AVX2 zeroing (64 bytes per iteration; rax is 64B aligned, rcx is 64B mult)
    vxorps  ymm0, ymm0, ymm0
    mov     r8, rax
    mov     r9, rcx
.zero_lp:
    test    r9, r9
    jz      .zero_done
    vmovdqa [r8], ymm0
    vmovdqa [r8 + 32], ymm0
    add     r8, 64
    sub     r9, 64
    jmp     .zero_lp
.zero_done:
    vzeroupper
    ret

.oom:
    mov     rax, -1
    ret

; ----------------------------------------------------------------------------
; int64 sys_munmap(void *addr, uint64 size) -> no-op in bare-metal bump arena
; ----------------------------------------------------------------------------
sys_munmap:
    xor     eax, eax
    ret

; ----------------------------------------------------------------------------
; int64 sys_write_all(int fd, const void *buf, uint64 len)
; Routes stdout/stderr directly to our bare-metal console (COM1 + VGA).
; ----------------------------------------------------------------------------
sys_write_all:
    mov     rdi, rsi
    mov     rsi, rdx
    call    console_write
    xor     eax, eax
    ret

; ----------------------------------------------------------------------------
; uint64 sys_clock_ns(void)
; ----------------------------------------------------------------------------
sys_clock_ns:
    jmp     timer_ns

; ============================================================================
; String & Numeric Formatting / Parsing Helpers (Identical ABI to Phase 1)
; ============================================================================

str_len:
    xor     eax, eax
.lp:
    cmp     byte [rdi + rax], 0
    je      .ret
    inc     rax
    jmp     .lp
.ret:
    ret

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

parse_f32:
    vxorps  xmm0, xmm0, xmm0
    vmovsd  xmm1, [rel d_ten]
    xor     ecx, ecx
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
    vmovsd  xmm3, [rel d_tenth]
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

write_u64:
    push    rbx
    sub     rsp, 32
    mov     ebx, edi
    mov     rax, rsi
    lea     rcx, [rsp + 31]
    mov     r8, rcx
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

write_f64_2dp:
    push    rbx
    push    r12
    sub     rsp, 24
    mov     ebx, edi
    vmulsd  xmm0, xmm0, [rel d_100]
    vaddsd  xmm0, xmm0, [rel d_half]
    vcvttsd2si rax, xmm0
    test    rax, rax
    jns     .pos
    xor     eax, eax
.pos:
    xor     edx, edx
    mov     rcx, 100
    div     rcx
    mov     r12, rdx
    mov     edi, ebx
    mov     rsi, rax
    call    write_u64
    mov     byte [rsp], '.'
    mov     rax, r12
    xor     edx, edx
    mov     rcx, 10
    div     rcx
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

; void write_hex64_raw(uint64 val in rdi) — prints 8 or 16 hex digits (no 0x prefix)
write_hex64_raw:
    push    rbx
    sub     rsp, 24
    mov     rax, rdi
    lea     rbx, [rel hex_digits]
    ; Print compact: 8 digits if upper 32 bits == 0, else 16 digits
    mov     rcx, rax
    shr     rcx, 32
    test    ecx, ecx
    jz      .hex8
    mov     ecx, 16
    jmp     .gen
.hex8:
    mov     ecx, 8
.gen:
    mov     r8d, ecx
    lea     rdx, [rsp + rcx]
.lp:
    dec     rdx
    mov     r9, rax
    and     r9d, 0x0F
    mov     r9b, [rbx + r9]
    mov     [rdx], r9b
    shr     rax, 4
    dec     ecx
    jnz     .lp
    mov     rdi, rsp
    mov     rsi, r8
    call    console_write
    add     rsp, 24
    pop     rbx
    ret

section .rodata
align 16
d_ten:      dq 10.0
d_tenth:    dq 0.1
d_100:      dq 100.0
d_half:     dq 0.5
d_signbit:  dq 0x8000000000000000, 0
hex_digits: db "0123456789ABCDEF"

section .bss
alignb 8
total_ram_bytes: resq 1
g_total_ram_mb:  resq 1
g_heap_base:     resq 1
g_heap_cur:      resq 1
g_heap_end:      resq 1
g_arena_used_mb: resq 1
