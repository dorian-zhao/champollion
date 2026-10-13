//! Checks the JIT's ARM64 encoder against an assembler: tests/fixtures/arm64.s, assembled
//! into arm64.bin by build.sh, lists the same instructions this test emits, in order.

use champollion_rv::jit::arm64::{Asm, AtomicOp, Cond, LR, Mem, SP, XZR};

#[test]
fn matches_assembler() {
    let mut a = Asm::new();
    a.add(0, 1, 2);
    a.add_w(3, 4, 5);
    a.add_lsl(6, 7, 8, 3);
    a.sub(9, 10, 11);
    a.sub_w(12, 13, 14);
    a.cmp(15, 16);
    a.and(17, 19, 20);
    a.orr(21, 22, 23);
    a.eor(24, 25, 26);
    a.mov(27, 28);
    a.mov_w(0, 1);
    a.mvn(2, 3);
    a.lslv(2, 3, 4);
    a.lsrv(5, 6, 7);
    a.asrv(8, 9, 10);
    a.lslv_w(11, 12, 13);
    a.lsrv_w(14, 15, 16);
    a.asrv_w(17, 19, 20);
    a.mul(0, 1, 2);
    a.mul_w(3, 4, 5);
    a.msub(6, 7, 8, 9);
    a.msub_w(10, 11, 12, 13);
    a.smulh(14, 15, 16);
    a.umulh(17, 19, 20);
    a.sdiv(21, 22, 23);
    a.udiv(24, 25, 26);
    a.sdiv_w(27, 28, 29);
    a.udiv_w(0, 1, 2);
    a.csel(3, 4, 5, Cond::Lt);
    a.cset(6, Cond::Eq);
    a.csinv(0, 1, XZR, Cond::Ne);
    a.csinv_w(2, 3, XZR, Cond::Ne);
    a.cset(7, Cond::Lo);
    a.add_imm(8, 9, 4095);
    a.add_imm_w(10, 11, 1);
    a.add_imm(SP, SP, 16);
    a.sub_imm(12, 13, 2048);
    a.sub_imm_w(14, 15, 7);
    a.cmp_imm(16, 0);
    a.cmp_imm_w(17, 5);
    a.and_imm(17, 19, 0x3f_ffff_ffff).unwrap();
    a.orr_imm(20, XZR, 0xff00).unwrap();
    a.eor_imm(21, 22, 0x5555_5555_5555_5555).unwrap();
    a.lsl_imm(0, 1, 12);
    a.lsr_imm(2, 3, 63);
    a.asr_imm(4, 5, 1);
    a.lsl_imm_w(6, 7, 31);
    a.lsr_imm_w(8, 9, 4);
    a.asr_imm_w(10, 11, 31);
    a.sxtw(12, 13);
    a.movz(14, 0x1234, 16);
    a.movk(15, 0xffff, 48);
    a.movn(16, 5, 0);
    a.ldr_reg(Mem::U8, 0, 1, 2);
    a.ldr_reg(Mem::U16, 3, 4, 5);
    a.ldr_reg(Mem::U32, 6, 7, 8);
    a.ldr_reg(Mem::U64, 9, 10, 11);
    a.ldr_reg(Mem::S8, 12, 13, 14);
    a.ldr_reg(Mem::S16, 15, 16, 17);
    a.ldr_reg(Mem::S32, 19, 20, 21);
    a.str_reg(Mem::U8, 22, 23, 24);
    a.str_reg(Mem::U16, 25, 26, 27);
    a.str_reg(Mem::U32, 28, 29, 0);
    a.str_reg(Mem::U64, 1, 2, 3);
    a.ldr_off(4, 5, 32760);
    a.str_off(6, 7, 8);
    a.ldr_off_w(8, 9, 4);
    a.str_off_w(10, 11, 16380);
    a.stp_pre(29, LR, SP, -16);
    a.ldp_post(29, LR, SP, 16);
    a.br(16);
    a.blr(17);
    a.ret();
    a.nop();
    a.brk(0x1234);
    a.dmb_ish();
    a.tst_imm(0, 7).unwrap();
    a.cmp_w(1, 2);
    a.atomic(AtomicOp::Add, true, 3, 4, 5);
    a.atomic(AtomicOp::Clr, false, 6, 7, 8);
    a.atomic(AtomicOp::Eor, true, 9, 10, 11);
    a.atomic(AtomicOp::Set, false, 12, 13, 14);
    a.atomic(AtomicOp::Smax, true, 15, 16, 17);
    a.atomic(AtomicOp::Smin, false, 19, 20, 21);
    a.atomic(AtomicOp::Umax, true, 22, 23, 24);
    a.atomic(AtomicOp::Umin, false, 25, 26, 27);
    a.atomic(AtomicOp::Swp, true, 0, XZR, 1);
    a.casal(true, 2, 3, 4);
    a.casal(false, 5, 6, 7);

    let source = include_str!("../../../tests/fixtures/arm64.s");
    let lines: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//") && !l.starts_with('.'))
        .collect();
    let words: Vec<u32> = include_bytes!("../../../tests/fixtures/arm64.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(words.len(), lines.len(), "arm64.bin is stale; run tests/fixtures/build.sh");
    assert_eq!(a.code.len(), words.len(), "emitted a different number of instructions");
    for ((got, want), line) in a.code.iter().zip(&words).zip(&lines) {
        assert_eq!(got, want, "{line}: got {got:#010x}, want {want:#010x}");
    }
}

#[test]
fn branches_match_assembler_offsets() {
    // b/bl/b.cond/cbz/cbnz/adr forwards and backwards, checked against known encodings.
    let mut a = Asm::new();
    a.b(4); // +4
    a.bl(0); // -1
    a.b_cond(Cond::Ge, 0); // -2
    a.cbnz(5, 5); // +2
    a.adr(16, 6); // +2 instructions = +8 bytes
    a.cbz(1, 0); // -5
    a.nop();
    assert_eq!(a.code[0], 0x1400_0004); // b .+16
    assert_eq!(a.code[1], 0x97ff_ffff); // bl .-4
    assert_eq!(a.code[2], 0x54ff_ffca); // b.ge .-8
    assert_eq!(a.code[3], 0xb500_0045); // cbnz x5, .+8
    assert_eq!(a.code[4], 0x1000_0050); // adr x16, .+8
    assert_eq!(a.code[5], 0xb4ff_ff61); // cbz x1, .-20
}
