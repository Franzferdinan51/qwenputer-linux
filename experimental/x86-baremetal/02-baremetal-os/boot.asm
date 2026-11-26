; ============================================================================
; os/boot.asm — Multiboot 1 Entry (32-bit) -> 64-bit Long Mode + AVX2/FMA + IDT
;
; Boot sequence:
;   1. Multiboot 1 header (0x1BADB002) within first 8 KiB of ELF.
;   2. 32-bit _start saves Multiboot magic/info, constructs 4-level page tables
;      identity-mapping 0..4 GiB with 2 MiB huge pages (UC on 0xFE000000..4GiB
;      for Local APIC MMIO), enables PAE + EFER.LME + CR0.PG, loads 64-bit GDT,
;      and far-jumps to 64-bit Long Mode.
;   3. 64-bit long_mode_start records initial RDTSC, installs minimal exception
;      IDT, verifies CPUID AVX2+FMA, enables SSE/AVX/OSXSAVE/XCR0=7/MXCSR, and
;      enters kernel_main.
; ============================================================================

%include "os_constants.inc"

extern _bss_start
extern _bss_end
extern kernel_main
extern serial_putc
extern console_write_cstr
extern write_hex64_raw

global _start
global enable_avx_fpu
global idt_load
global gdt64_ptr
global idt64_ptr
global pml4_table
global g_mb_magic
global g_mb_info
global g_boot_tsc
global g_has_avx2_fma

; ============================================================================
; Multiboot 1 Header (must reside in first 8192 bytes of ELF file)
; ============================================================================
section .multiboot align=4
    dd MB_HEADER_MAGIC
    dd MB_HEADER_FLAGS
    dd MB_HEADER_CHECKSUM

; ============================================================================
; 32-bit Protected Mode Bootstrap
; ============================================================================
section .text.boot
bits 32
default abs

_start:
    cli
    cld

    ; Preserve Multiboot magic (eax) and info ptr (ebx) across .bss zeroing
    mov     esi, eax
    mov     ebp, ebx

    ; Explicitly zero .bss so every Multiboot loader (QEMU, GRUB BIOS/UEFI) starts clean
    mov     edi, _bss_start
    mov     ecx, _bss_end
    sub     ecx, edi
    shr     ecx, 2
    xor     eax, eax
    rep     stosd

    ; Set up temporary 32-bit stack at top of BSP stack
    mov     esp, bsp_stack_top

    ; Save Multiboot 1 magic and info pointer
    mov     [g_mb_magic], esi
    mov     [g_mb_info], ebp

    ; Capture earliest possible RDTSC right at boot entry
    rdtsc
    mov     [g_boot_tsc], eax
    mov     [g_boot_tsc + 4], edx

    ; Zero page tables (6 pages = 24 KiB = 6144 dwords)
    mov     edi, pml4_table
    xor     eax, eax
    mov     ecx, (6 * 4096) / 4
    rep     stosd

    ; PML4[0] -> pdpt_table | Present | Writable (0x03)
    mov     eax, pdpt_table
    or      eax, 0x03
    mov     [pml4_table], eax

    ; PDPT[0..3] -> pd_tables[0..3] | Present | Writable (0x03)
    mov     eax, pd_tables
    or      eax, 0x03
    mov     [pdpt_table + 0], eax
    add     eax, 4096
    mov     [pdpt_table + 8], eax
    add     eax, 4096
    mov     [pdpt_table + 16], eax
    add     eax, 4096
    mov     [pdpt_table + 24], eax

    ; Fill all 4 * 512 = 2048 PD entries with 2 MiB huge pages (0..4 GiB)
    ; Entries 0..2031 (0x00000000..0xFDFFFFFF): flags 0x83 (P | RW | PS)
    ; Entries 2032..2047 (0xFE000000..0xFFFFFFFF): flags 0x9B (P | RW | PWT | PCD | PS)
    ;   -> Strong Uncacheable (UC) for Local APIC (0xFEE00000) & IOAPIC MMIO!
    mov     edi, pd_tables
    xor     ebx, ebx                    ; physical address = 0
    xor     ecx, ecx                    ; entry index = 0..2047
.fill_pd:
    mov     eax, ebx
    cmp     ecx, 2032
    jae     .mmio_flags
    or      eax, 0x83
    jmp     .store_pde
.mmio_flags:
    or      eax, 0x9B
.store_pde:
    mov     [edi], eax
    mov     dword [edi + 4], 0          ; upper 32 bits = 0 (all < 4 GiB)
    add     edi, 8
    add     ebx, 0x00200000             ; +2 MiB
    inc     ecx
    cmp     ecx, 2048
    jb      .fill_pd

    ; Load CR3 with PML4 physical address
    mov     eax, pml4_table
    mov     cr3, eax

    ; Enable PAE (CR4.PAE = bit 5)
    mov     eax, cr4
    or      eax, (1 << 5)
    mov     cr4, eax

    ; Enable Long Mode in IA32_EFER MSR (0xC0000080, bit 8 = LME)
    mov     ecx, 0xC0000080
    rdmsr
    or      eax, (1 << 8)
    wrmsr

    ; Enable Paging (CR0.PG = bit 31, CR0.PE = bit 0)
    mov     eax, cr0
    or      eax, 0x80000001
    mov     cr0, eax

    ; Load 64-bit GDT and far jump into 64-bit Long Mode
    lgdt    [gdt64_ptr]
    jmp     0x08:long_mode_start

; ============================================================================
; 64-bit Long Mode Entry
; ============================================================================
section .text
bits 64
default rel

long_mode_start:
    mov     ax, 0x10
    mov     ds, ax
    mov     es, ax
    mov     ss, ax
    mov     fs, ax
    mov     gs, ax

    lea     rsp, [rel bsp_stack_top]
    and     rsp, -16

    ; Clean RFLAGS (IF=0, DF=0)
    push    0
    popfq

    ; Build & load minimal 32-entry exception IDT (catches any CPU fault cleanly)
    call    idt_setup_and_load

    ; Check CPUID for XSAVE + AVX + FMA + AVX2
    call    check_cpu_features
    mov     [rel g_has_avx2_fma], eax
    test    eax, eax
    jz      .skip_avx
    call    enable_avx_fpu
.skip_avx:

    ; Enter bare-metal OS kernel
    call    kernel_main

.halt_forever:
    cli
    hlt
    jmp     .halt_forever

; ----------------------------------------------------------------------------
; uint32 check_cpu_features(void) -> eax = 1 if XSAVE+AVX+FMA+AVX2 supported
; ----------------------------------------------------------------------------
check_cpu_features:
    push    rbx
    mov     eax, 1
    cpuid
    ; ECX: bit 12 (FMA), bit 26 (XSAVE), bit 28 (AVX), bit 29 (F16C)
    mov     edx, (1 << 12) | (1 << 26) | (1 << 28) | (1 << 29)
    and     ecx, edx
    cmp     ecx, edx
    jne     .no

    ; Check max leaf >= 7
    xor     eax, eax
    cpuid
    cmp     eax, 7
    jb      .no

    mov     eax, 7
    xor     ecx, ecx
    cpuid
    ; EBX: bit 5 (AVX2)
    test    ebx, (1 << 5)
    jz      .no

    mov     eax, 1
    pop     rbx
    ret
.no:
    xor     eax, eax
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void enable_avx_fpu(void)
; Called by BSP and EVERY AP core before executing SSE/AVX2/FMA instructions.
; Clears CR0.EM, sets CR0.MP, sets CR4.OSFXSR|OSXMMEXCPT|OSXSAVE, sets XCR0=7,
; initializes x87 and MXCSR (0x1F80: round-to-nearest-even, exceptions masked).
; ----------------------------------------------------------------------------
enable_avx_fpu:
    mov     rax, cr0
    and     ax, 0xFFFB                  ; clear CR0.EM (bit 2)
    or      ax, 0x0002                  ; set CR0.MP (bit 1)
    mov     cr0, rax

    mov     rax, cr4
    or      eax, (1 << 9) | (1 << 10) | (1 << 18) ; OSFXSR | OSXMMEXCPT | OSXSAVE
    mov     cr4, rax

    xor     ecx, ecx                    ; XCR0
    xgetbv
    or      eax, 7                      ; x87 (bit 0) | SSE (bit 1) | AVX (bit 2)
    xsetbv

    fninit
    ldmxcsr [rel default_mxcsr]
    ret

; ----------------------------------------------------------------------------
; Minimal 64-bit Exception IDT (vectors 0..31) for clean fault diagnostics
; ----------------------------------------------------------------------------
idt_setup_and_load:
    lea     rdi, [rel idt64]
    lea     rsi, [rel isr_stub_table]
    xor     ecx, ecx
.fill:
    cmp     ecx, 32
    jae     .load
    mov     rax, [rsi + rcx*8]          ; handler address
    mov     [rdi + 0], ax               ; offset 15..0
    mov     word [rdi + 2], 0x08        ; 64-bit code selector
    mov     byte [rdi + 4], 0           ; IST = 0
    mov     byte [rdi + 5], 0x8E        ; Present | DPL=0 | Interrupt Gate
    shr     rax, 16
    mov     [rdi + 6], ax               ; offset 31..16
    shr     rax, 16
    mov     [rdi + 8], eax              ; offset 63..32
    mov     dword [rdi + 12], 0         ; reserved
    add     rdi, 16
    inc     ecx
    jmp     .fill
.load:
    lidt    [rel idt64_ptr]
    ret

idt_load:
    lidt    [rel idt64_ptr]
    ret

; Common CPU exception reporter (prints vector + RIP + error code to COM1 & halts)
isr_common:
    ; Stack: [rsp] = vec, [rsp+8] = err, [rsp+16] = rip, [rsp+24] = cs, [rsp+32] = rflags
    mov     r12, [rsp + 0]              ; vec
    mov     r13, [rsp + 8]              ; err
    mov     r14, [rsp + 16]             ; rip
    and     rsp, -16
    lea     rdi, [rel msg_exc_hdr]
    call    console_write_cstr
    mov     rdi, r12
    call    write_hex64_raw
    lea     rdi, [rel msg_exc_rip]
    call    console_write_cstr
    mov     rdi, r14
    call    write_hex64_raw
    lea     rdi, [rel msg_exc_err]
    call    console_write_cstr
    mov     rdi, r13
    call    write_hex64_raw
    lea     rdi, [rel msg_exc_cr2]
    call    console_write_cstr
    mov     rdi, cr2
    call    write_hex64_raw
    mov     al, 10
    call    serial_putc
.hang:
    cli
    hlt
    jmp     .hang

%macro ISR_NOERR 1
isr_stub_%1:
    push    qword 0
    push    qword %1
    jmp     isr_common
%endmacro

%macro ISR_ERR 1
isr_stub_%1:
    push    qword %1
    jmp     isr_common
%endmacro

ISR_NOERR 0
ISR_NOERR 1
ISR_NOERR 2
ISR_NOERR 3
ISR_NOERR 4
ISR_NOERR 5
ISR_NOERR 6
ISR_NOERR 7
ISR_ERR   8
ISR_NOERR 9
ISR_ERR   10
ISR_ERR   11
ISR_ERR   12
ISR_ERR   13
ISR_ERR   14
ISR_NOERR 15
ISR_NOERR 16
ISR_ERR   17
ISR_NOERR 18
ISR_NOERR 19
ISR_NOERR 20
ISR_ERR   21
ISR_NOERR 22
ISR_NOERR 23
ISR_NOERR 24
ISR_NOERR 25
ISR_NOERR 26
ISR_NOERR 27
ISR_NOERR 28
ISR_ERR   29
ISR_ERR   30
ISR_NOERR 31

; ============================================================================
; Read-only tables & GDT
; ============================================================================
section .rodata
align 16
gdt64:
    dq 0x0000000000000000               ; 0x00: Null descriptor
    dq 0x00AF9A000000FFFF               ; 0x08: 64-bit Code (L=1, P=1, DPL=0, RX)
    dq 0x00CF92000000FFFF               ; 0x10: 64-bit Data (P=1, DPL=0, RW)
gdt64_end:

align 8
gdt64_ptr:
    dw gdt64_end - gdt64 - 1
    dq gdt64

align 8
idt64_ptr:
    dw 32 * 16 - 1
    dq idt64

align 8
isr_stub_table:
%assign i 0
%rep 32
    dq isr_stub_%+i
%assign i i+1
%endrep

align 4
default_mxcsr:  dd 0x00001F80

msg_exc_hdr:    db 10, "[PANIC] CPU Exception vec=0x", 0
msg_exc_rip:    db " RIP=0x", 0
msg_exc_err:    db " ERR=0x", 0
msg_exc_cr2:    db " CR2=0x", 0

; ============================================================================
; Uninitialized Data (.bss — zero bytes on disk)
; ============================================================================
section .bss
alignb 4096
pml4_table:     resb 4096
pdpt_table:     resb 4096
pd_tables:      resb 4 * 4096           ; 4 x 512 x 2 MiB = 4 GiB identity map

alignb 16
idt64:          resb 32 * 16

alignb 16
bsp_stack:      resb BSP_STACK_SIZE
bsp_stack_top:

alignb 8
g_boot_tsc:     resq 1
g_mb_magic:     resd 1
g_mb_info:      resd 1
g_has_avx2_fma: resd 1
