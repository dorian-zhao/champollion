/* One read/write/execute segment, like the bare-metal environment riscv-tests expect.
   Used for tests that write to code or execute data (fence_i, rvc). */
PHDRS { image PT_LOAD FLAGS(7); }
SECTIONS {
    . = 0x10000;
    .text : { *(.text*) } :image
    .data : { *(.data*) *(.sdata*) } :image
    .bss : { *(.bss*) *(.sbss*) } :image
}
