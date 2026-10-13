//! Sanity checks on the committed RISC-V fixtures (rebuild with tests/fixtures/build.sh).

use champollion_rv::loader::EM_RISCV;

const HELLO: &[u8] = include_bytes!("../../../tests/fixtures/hello");

#[test]
fn hello_is_static_rv64_elf() {
    assert_eq!(&HELLO[..4], b"\x7fELF");
    assert_eq!(HELLO[4], 2, "ELFCLASS64");
    assert_eq!(HELLO[5], 1, "little-endian");
    let e_type = u16::from_le_bytes([HELLO[16], HELLO[17]]);
    let e_machine = u16::from_le_bytes([HELLO[18], HELLO[19]]);
    assert_eq!(e_type, 2, "ET_EXEC (static, non-PIE)");
    assert_eq!(e_machine, EM_RISCV);
}
