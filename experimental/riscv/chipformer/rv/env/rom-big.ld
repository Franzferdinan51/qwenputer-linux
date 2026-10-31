/* The GROWN Qwenputer address map (the recompiled circuit from the
   qwenputer-circuit work; same layout the Doom port runs on):
     RAM  0x000000  2 MB   data / bss / heap / stack
     DROM 0x200000  6 MB   read-only data
     ROM  0x800000  256 KB code (the trace format's 16-bit anchor cap)
   crt0 expects the host/snapshot to set sp to the top of RAM (0x200000).
   Pair with a grown-map circuit build (spec.py RAM_WORDS = 1 << 19);
   binaries linked here do NOT run on the stock 32 KB circuit. */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x100, LENGTH = 0x1FFF00
  DROM (r) : ORIGIN = 0x200000, LENGTH = 6M
  ROM (rx) : ORIGIN = 0x800000, LENGTH = 256K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .rodata : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);  /* (.tbss takes no address space of its own) */
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM
  __heap_end = 0x1F0000;  /* 64 KB of stack above the heap */
  _end = __heap_start;
}
