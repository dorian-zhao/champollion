//! Static ELF loader (M2): map PT_LOAD segments, build the initial stack (argc, argv,
//! envp, auxv) and return the entry pc.

/// `e_machine` value for RISC-V.
pub const EM_RISCV: u16 = 243;
