//! Checks the JIT's ARM64 encoder against an assembler: tests/fixtures/arm64.s, assembled
//! into arm64.bin by build.sh, lists the same instructions this test emits, in order.

use champollion_rv::jit::arm64::{
    Asm, AtomicOp, Cond, FmaKind, Fp, FpOp, FpRound, LR, Mem, SP, XZR,
};

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
    a.sub_imm_lsl12(16, 27, 20);
    a.add_imm_lsl12(13, 14, 20);
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
    a.adrp(14, 1);
    a.adrp(0, -2);
    a.adrp(17, 0x4000);
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
    a.ldr_fp(Fp::D, 16, 28, 256);
    a.str_fp(Fp::D, 17, 28, 504);
    a.ldr_fp(Fp::S, 18, 1, 4);
    a.str_fp(Fp::S, 19, 2, 8);
    a.fmov_from_gpr(Fp::S, 16, 14);
    a.fmov_from_gpr(Fp::D, 17, 15);
    a.fmov_to_gpr(Fp::S, 14, 18);
    a.fmov_to_gpr(Fp::D, 15, 19);
    a.fp_arith(FpOp::Add, Fp::S, 16, 17, 18);
    a.fp_arith(FpOp::Sub, Fp::D, 16, 17, 18);
    a.fp_arith(FpOp::Mul, Fp::S, 19, 20, 21);
    a.fp_arith(FpOp::Div, Fp::D, 22, 23, 24);
    a.fsqrt(Fp::S, 16, 17);
    a.fsqrt(Fp::D, 18, 19);
    a.fma(FmaKind::Fmadd, Fp::S, 16, 17, 18, 19);
    a.fma(FmaKind::Fmsub, Fp::D, 20, 21, 22, 23);
    a.fma(FmaKind::Fnmadd, Fp::S, 24, 25, 26, 27);
    a.fma(FmaKind::Fnmsub, Fp::D, 28, 29, 30, 31);
    a.fcmp(false, Fp::S, 16, 17);
    a.fcmp(true, Fp::D, 18, 19);
    a.fcvt(Fp::D, 16, 17);
    a.fcvt(Fp::S, 18, 19);
    a.fcvt_to_int(FpRound::Nearest, true, false, Fp::S, 0, 16);
    a.fcvt_to_int(FpRound::Away, false, true, Fp::D, 1, 17);
    a.fcvt_to_int(FpRound::Up, true, true, Fp::S, 2, 18);
    a.fcvt_to_int(FpRound::Down, false, false, Fp::D, 3, 19);
    a.fcvt_to_int(FpRound::Zero, true, true, Fp::D, 4, 20);
    a.fcvt_to_int(FpRound::Zero, false, false, Fp::S, 5, 21);
    a.cvt_from_int(true, false, Fp::S, 16, 0);
    a.cvt_from_int(false, true, Fp::D, 17, 1);
    a.cvt_from_int(true, true, Fp::D, 18, 2);
    a.cvt_from_int(false, false, Fp::S, 19, 3);
    a.mrs_fpcr(0);
    a.msr_fpcr(1);
    a.mrs_fpsr(2);
    a.msr_fpsr(3);
    a.rbit_w(4, 5);
    a.cmn_imm_w(6, 1);
    a.and_imm_w(7, 8, 0x1f).unwrap();
    a.ldr_fp_reg(Fp::D, 20, 27, 17);
    a.str_fp_reg(Fp::D, 21, 27, 17);
    a.ldr_fp_reg(Fp::S, 22, 27, 17);
    a.str_fp_reg(Fp::S, 23, 27, 17);
    a.fmov_d(1, 2);
    a.fneg_d(3, 4);
    a.fabs_d(5, 6);
    a.stp_pre_d(8, 9, SP, -16);
    a.ldp_post_d(14, 15, SP, 16);
    for (mem, rt, offset) in [
        (Mem::U8, 0, 4095),
        (Mem::U16, 1, 8190),
        (Mem::U32, 2, 16380),
        (Mem::U64, 3, 32760),
        (Mem::S8, 4, 1),
        (Mem::S16, 5, 2),
        (Mem::S32, 6, 4),
        (Mem::U8, 7, -256),
        (Mem::U16, 8, -2),
        (Mem::U32, 9, -4),
        (Mem::U64, 10, -8),
        (Mem::S8, 11, -1),
        (Mem::S16, 12, -3),
        (Mem::S32, 13, 255),
        (Mem::U64, 19, 7),
    ] {
        assert!(a.ldr_imm(mem, rt, 17, offset));
    }
    for (mem, rt, offset) in [
        (Mem::U8, 20, 4095),
        (Mem::U16, 21, 2),
        (Mem::U32, 22, 4),
        (Mem::U64, 23, 8),
        (Mem::U8, 24, -1),
        (Mem::U16, 25, -2),
        (Mem::U32, 26, -4),
        (Mem::U64, 0, -256),
    ] {
        assert!(a.str_imm(mem, rt, 17, offset));
    }
    assert!(a.ldr_fp_imm(Fp::S, 1, 17, 4));
    assert!(a.ldr_fp_imm(Fp::D, 2, 17, 8));
    assert!(a.ldr_fp_imm(Fp::S, 3, 17, -4));
    assert!(a.ldr_fp_imm(Fp::D, 4, 17, -8));
    assert!(a.str_fp_imm(Fp::S, 5, 17, 16380));
    assert!(a.str_fp_imm(Fp::D, 6, 17, 32760));
    assert!(a.str_fp_imm(Fp::S, 7, 17, -4));
    assert!(a.str_fp_imm(Fp::D, 8, 17, 3));
    a.bic(0, 1, 2);
    a.orn(3, 4, 5);
    a.eon(6, 7, 8);
    a.clz(9, 10);
    a.clz_w(11, 12);
    a.rbit(13, 14);
    a.rev(15, 16);
    a.rorv(17, 19, 20);
    a.rorv_w(21, 22, 23);
    a.ror_imm(24, 25, 26);
    a.ror_imm_w(27, 28, 29);
    a.add_uxtw(0, 1, 2);
    a.add_uxtw_lsl(3, 4, 5, 1);
    a.add_uxtw_lsl(6, 7, 8, 2);
    a.add_uxtw_lsl(9, 10, 11, 3);
    a.cnt16b(16, 17);
    a.addv16b(18, 19);
    a.cmeq_zero16b(20, 21);
    a.mvn16b(22, 23);
    a.rbit16b(16, 16);
    a.rbit16b(3, 5);
    a.ldr_q(0, 1);
    assert!(a.ldr_q_off(2, 3, 32));
    a.str_q(4, 5);
    assert!(a.str_q_off(6, 7, 48));
    a.ldr_q_reg(25, 2, 3);
    a.str_q_reg(26, 2, 3);
    a.vec_add(8, 0, 1, 2);
    a.vec_add(16, 3, 4, 5);
    a.vec_add(32, 6, 7, 8);
    a.vec_add(64, 9, 10, 11);
    a.vec_sub(8, 0, 1, 2);
    a.vec_sub(16, 3, 4, 5);
    a.vec_sub(32, 6, 7, 8);
    a.vec_sub(64, 9, 10, 11);
    a.vec_and(12, 13, 14);
    a.vec_orr(15, 16, 17);
    a.vec_eor(18, 19, 20);
    a.vec_dup(21, 1, 1);
    a.vec_dup(22, 2, 2);
    a.vec_dup(23, 3, 4);
    a.vec_dup(24, 4, 8);
    a.cmeq16b(3, 4, 5);
    a.movi16b_one(5);
    a.ext16b(2, 0, 0, 8);
    a.vec_ushr_imm(8, 0, 1, 3);
    a.vec_ushr_imm(16, 2, 3, 5);
    a.vec_ushr_imm(32, 4, 5, 7);
    a.vec_ushr_imm(64, 6, 7, 9);
    a.vec_sshr_imm(8, 8, 9, 3);
    a.vec_sshr_imm(16, 10, 11, 5);
    a.vec_sshr_imm(32, 12, 13, 7);
    a.vec_sshr_imm(64, 14, 15, 9);
    a.vec_ushl(8, 16, 17, 18);
    a.vec_ushl(16, 19, 20, 21);
    a.vec_ushl(32, 22, 23, 24);
    a.vec_ushl(64, 25, 26, 27);
    a.vec_sshl(8, 0, 1, 2);
    a.vec_sshl(16, 3, 4, 5);
    a.vec_sshl(32, 6, 7, 8);
    a.vec_sshl(64, 9, 10, 11);
    a.vec_mul(8, 12, 13, 14);
    a.vec_mul(16, 15, 16, 17);
    a.vec_mul(32, 18, 19, 20);
    a.vec_ins_s0(14, 19);
    a.vec_fadd(32, 0, 1, 2);
    a.vec_fadd(64, 3, 4, 5);
    a.vec_fsub(32, 6, 7, 8);
    a.vec_fsub(64, 9, 10, 11);
    a.vec_fmul(32, 12, 13, 14);
    a.vec_fmul(64, 15, 16, 17);
    a.vec_fdiv(32, 18, 19, 20);
    a.vec_fdiv(64, 21, 22, 23);
    a.vec_fmla(32, 24, 25, 26);
    a.vec_fmla(64, 27, 28, 29);
    a.vec_fmls(32, 30, 31, 0);
    a.vec_fmls(64, 1, 2, 3);
    a.vec_fneg(32, 4, 5);
    a.vec_fneg(64, 6, 7);
    a.vec_dup_lane0(32, 8, 9);
    a.vec_dup_lane0(64, 10, 11);
    let len = a.code.len();
    assert!(!a.ldr_imm(Mem::U64, 0, 17, 32768));
    assert!(!a.str_imm(Mem::U8, 0, 17, -257));
    assert!(!a.ldr_fp_imm(Fp::D, 0, 17, 4100));
    assert_eq!(a.code.len(), len, "unencodable offsets emit nothing");

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
