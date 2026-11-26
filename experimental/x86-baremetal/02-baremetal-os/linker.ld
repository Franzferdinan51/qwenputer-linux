/* ============================================================================
 * os/linker.ld — Bare-Metal x86-64 Linker Script for asm-llm OS
 *
 * Linked at physical = virtual 1 MiB (0x00100000), 4 KiB page alignment so
 * the Multiboot 1 header sits safely within the first 8 KiB of the ELF file.
 * ============================================================================ */

ENTRY(_start)

PHDRS
{
    text PT_LOAD FLAGS(5);   /* PF_R | PF_X */
    data PT_LOAD FLAGS(6);   /* PF_R | PF_W */
}

SECTIONS
{
    . = 0x00100000;
    _kernel_start = .;

    .text ALIGN(4096) : AT(ADDR(.text))
    {
        KEEP(*(.multiboot))
        *(.text.boot)
        *(.text .text.*)
    } :text

    .rodata ALIGN(64) : AT(ADDR(.rodata))
    {
        *(.rodata .rodata.*)
    } :text

    .data ALIGN(4096) : AT(ADDR(.data))
    {
        *(.data .data.*)
    } :data

    .bss ALIGN(4096) : AT(ADDR(.bss))
    {
        _bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(4096);
        _bss_end = .;
    } :data

    . = ALIGN(4096);
    _kernel_end = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note .note.*)
    }
}
