/* Linker script for the user-mode ACT build, based on the upstream qemu DUT's
   config/qemu/qemu-RVI20U64/link.ld with one change (marked below): zig's lld does not
   honor location-counter advances between output sections, so `.text.rvmodel` is given
   an explicit expression address instead of `.`. */

/* Set these to match the DUT memory map and hart count. */
RAM_ORIGIN = 0x80000000;
RAM_LENGTH = 0x80000000;
TEST_BASE = RAM_ORIGIN;
NUM_HARTS = 1;

/* Most users should not need to modify anything below this line. */
STACK_SIZE = 0x20000;

OUTPUT_ARCH( "riscv" )
ENTRY(rvtest_entry_point)

MEMORY
{
  ram (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

PROVIDE(__stack_size = STACK_SIZE);
PROVIDE(__num_harts = NUM_HARTS);

SECTIONS
{
  ASSERT(TEST_BASE >= ORIGIN(ram) && TEST_BASE < ORIGIN(ram) + LENGTH(ram), "TEST_BASE is outside RAM; update TEST_BASE and RAM_ORIGIN/RAM_LENGTH in link.ld, and the matching memory.regions entry in sail.json")

  .text.init   TEST_BASE : { *(.text.init) } > ram
  .text.rvtest . : { *(.text.rvtest) *(.text.rvtest.*) } > ram

  . = ALIGN(0x4000);
  .rodata . : { *(.rodata) *(.rodata.*) *(.srodata) *(.srodata.*) } > ram
  .data   . : { *(.data) *(.data.*) *(.sdata) *(.sdata.*) } > ram

  . = ALIGN(16);
  .bss . : {
    __bss_start = .;
    *(.sbss) *(.sbss.*) *(.bss) *(.bss.*) *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > ram

  . = ALIGN(16);
  __stack_bottom = .;
  . += 0x20000;
  __stack_top = .;

  /* Upstream writes `. = ALIGN(0x1000);` above and addresses the section as `.`, which
     lld then places before those advances. Addressing the section explicitly instead
     also keeps it a page past the stack, so this R E segment never shares a page with
     the RW data: qemu-user maps segments page by page and a shared page would lose its
     write permission with the later mapping (signature stores then fault). */
  .text.rvmodel ALIGN(__stack_top, 0x1000) : { *(.text.rvmodel) *(.text.rvmodel.*) *(.text) *(.text.*) } > ram

  . = ALIGN(0x1000);
  _end = .;

  ASSERT(_end <= ORIGIN(ram) + LENGTH(ram), "ACT ELF exceeds RAM; increase RAM_LENGTH in link.ld and the matching memory.regions size in sail.json")
}
