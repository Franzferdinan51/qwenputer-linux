; ============================================================================
; os/os_constants.inc — Hardware ports, Multiboot 1, ACPI, APIC, VGA, SMP defs
; Pure x86-64 NASM assembly (zero C, zero libc, zero OS)
; ============================================================================

; --- Multiboot 1 ---
%define MB_HEADER_MAGIC     0x1BADB002
%define MB_HEADER_FLAGS     0x00000003      ; PAGE_ALIGN | MEMORY_INFO
%define MB_HEADER_CHECKSUM  (-(MB_HEADER_MAGIC + MB_HEADER_FLAGS))
%define MB_BOOTLOADER_MAGIC 0x2BADB002

; Multiboot 1 Info struct offsets
%define MBI_FLAGS           0
%define MBI_MEM_LOWER       4
%define MBI_MEM_UPPER       8
%define MBI_CMDLINE         16
%define MBI_MODS_COUNT      20
%define MBI_MODS_ADDR       24
%define MBI_MMAP_LENGTH     44
%define MBI_MMAP_ADDR       48

; Multiboot module struct (16 bytes)
%define MB_MOD_START        0
%define MB_MOD_END          4
%define MB_MOD_STRING       8

; --- Serial COM1 (0x3F8, 115200 8N1) ---
%define COM1_PORT           0x3F8
%define COM1_IER            0x3F9
%define COM1_FCR            0x3FA
%define COM1_LCR            0x3FB
%define COM1_MCR            0x3FC
%define COM1_LSR            0x3FD

; --- VGA Text Mode (80x25 @ 0xB8000) ---
%define VGA_MEM             0x000B8000
%define VGA_COLS            80
%define VGA_ROWS            25
%define VGA_CRTC_IDX        0x3D4
%define VGA_CRTC_DATA       0x3D5

; VGA Color Attributes (bg<<4 | fg)
%define VGA_ATTR_GREY       0x07
%define VGA_ATTR_DARK       0x08
%define VGA_ATTR_BLUE       0x09
%define VGA_ATTR_GREEN      0x0A
%define VGA_ATTR_CYAN       0x0B
%define VGA_ATTR_RED        0x0C
%define VGA_ATTR_MAGENTA    0x0D
%define VGA_ATTR_YELLOW     0x0E
%define VGA_ATTR_WHITE      0x0F

; --- PS/2 Keyboard ---
%define PS2_DATA            0x60
%define PS2_STATUS          0x64

; --- PIT Channel 2 (1,193,182 Hz) ---
%define PIT_CH2_DATA        0x42
%define PIT_CMD             0x43
%define PIT_PORT_B          0x61
%define PIT_FREQ_HZ         1193182

; --- Local APIC MMIO ---
%define APIC_DEFAULT_BASE   0xFEE00000
%define APIC_REG_ID         0x020
%define APIC_REG_SVR        0x0F0
%define APIC_REG_ESR        0x280
%define APIC_REG_ICR_LO     0x300
%define APIC_REG_ICR_HI     0x310

; --- SMP AP Trampoline physical addresses (low memory < 1 MiB) ---
%define TRAMP_BASE          0x00008000
%define TRAMP_VECTOR        0x08            ; 0x08 * 4096 = 0x8000
%define TRAMP_PML4          0x00008F80      ; uint32 pml4 phys addr
%define TRAMP_ENTRY64       0x00008F88      ; uint64 ap_entry64 addr

; --- SMP / Stack sizing ---
%define OS_MAX_CPUS         64
%define BSP_STACK_SIZE      65536
%define AP_STACK_SIZE       32768           ; 32 KiB per AP core (64 * 32K = 2 MiB in .bss)
%define CMDLINE_BUF_SIZE    1024
%define LINE_BUF_SIZE       1024
