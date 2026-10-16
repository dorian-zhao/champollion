//! Checks the decoder against encodings produced by a real assembler (LLVM, via `zig cc`).
//! Each `tests/fixtures/*.s` lists instructions one per line; the matching `.bin` is the
//! assembled `.text`.

use champollion_rv::decoder::inst_len;
use champollion_rv::decoder::{
    AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, FArithOp, FCmpOp, FWidth, FmaOp, IntType,
    LoadWidth, RM_DYN, SgnjKind, StoreWidth,
};
use champollion_rv::decoder::{
    VArithOp, VCmpOp, VMaskOp, VMemKind, VRedOp, VSel, VSetKind, VStoreKind, VWidth,
};
use champollion_rv::{Inst, decode};

// Register numbers by ABI name, for readability below.
const ZERO: u8 = 0;
const RA: u8 = 1;
const SP: u8 = 2;
const GP: u8 = 3;
const TP: u8 = 4;
const T0: u8 = 5;
const T1: u8 = 6;
const S0: u8 = 8;
const S1: u8 = 9;
const A0: u8 = 10;
const A1: u8 = 11;
const A2: u8 = 12;
const A3: u8 = 13;
const A4: u8 = 14;
const A5: u8 = 15;
const A6: u8 = 16;
const A7: u8 = 17;
const S2: u8 = 18;
const S3: u8 = 19;
const S4: u8 = 20;
const S5: u8 = 21;
const S6: u8 = 22;
const S7: u8 = 23;
const S8: u8 = 24;
const S9: u8 = 25;
const S10: u8 = 26;
const S11: u8 = 27;
const T4: u8 = 29;
const T5: u8 = 30;
const T6: u8 = 31;

fn rv64i_expected() -> Vec<(&'static str, Inst)> {
    use AluOp as A;
    use AluOp32 as W;
    use BranchCond as C;
    use Inst::*;
    use LoadWidth as L;
    use StoreWidth as S;
    vec![
        ("lui a0, 0x12345", Lui { rd: A0, imm: 0x1234_5000 }),
        ("lui t1, 0xfffff", Lui { rd: T1, imm: -0x1000 }),
        ("auipc t0, 1", Auipc { rd: T0, imm: 0x1000 }),
        ("auipc sp, 0x80000", Auipc { rd: SP, imm: i32::MIN }),
        ("jal ra, 2048", Jal { rd: RA, offset: 2048 }),
        ("jal zero, -1048576", Jal { rd: ZERO, offset: -1_048_576 }),
        ("jalr ra, 16(a0)", Jalr { rd: RA, rs1: A0, offset: 16 }),
        ("jalr zero, -1(ra)", Jalr { rd: ZERO, rs1: RA, offset: -1 }),
        ("beq a0, a1, -4096", Branch { cond: C::Eq, rs1: A0, rs2: A1, offset: -4096 }),
        ("bne a0, a1, 4094", Branch { cond: C::Ne, rs1: A0, rs2: A1, offset: 4094 }),
        ("blt t0, t1, 8", Branch { cond: C::Lt, rs1: T0, rs2: T1, offset: 8 }),
        ("bge t0, t1, -8", Branch { cond: C::Ge, rs1: T0, rs2: T1, offset: -8 }),
        ("bltu s0, s1, 16", Branch { cond: C::Ltu, rs1: S0, rs2: S1, offset: 16 }),
        ("bgeu s0, s1, -16", Branch { cond: C::Geu, rs1: S0, rs2: S1, offset: -16 }),
        ("lb a0, -2048(sp)", Load { width: L::B, rd: A0, rs1: SP, offset: -2048 }),
        ("lh a1, -1(sp)", Load { width: L::H, rd: A1, rs1: SP, offset: -1 }),
        ("lw a2, 0(sp)", Load { width: L::W, rd: A2, rs1: SP, offset: 0 }),
        ("ld a3, 8(sp)", Load { width: L::D, rd: A3, rs1: SP, offset: 8 }),
        ("lbu a4, 1(gp)", Load { width: L::Bu, rd: A4, rs1: GP, offset: 1 }),
        ("lhu a5, 2(tp)", Load { width: L::Hu, rd: A5, rs1: TP, offset: 2 }),
        ("lwu a6, 2047(s0)", Load { width: L::Wu, rd: A6, rs1: S0, offset: 2047 }),
        ("sb a0, -2048(sp)", Store { width: S::B, rs1: SP, rs2: A0, offset: -2048 }),
        ("sh a1, -1(sp)", Store { width: S::H, rs1: SP, rs2: A1, offset: -1 }),
        ("sw a2, 0(sp)", Store { width: S::W, rs1: SP, rs2: A2, offset: 0 }),
        ("sd a3, 2047(sp)", Store { width: S::D, rs1: SP, rs2: A3, offset: 2047 }),
        ("addi a0, a1, -2048", OpImm { op: A::Add, rd: A0, rs1: A1, imm: -2048 }),
        ("slti a0, a1, 2047", OpImm { op: A::Slt, rd: A0, rs1: A1, imm: 2047 }),
        ("sltiu a0, a1, -1", OpImm { op: A::Sltu, rd: A0, rs1: A1, imm: -1 }),
        ("xori a0, a1, -1", OpImm { op: A::Xor, rd: A0, rs1: A1, imm: -1 }),
        ("ori a0, a1, 0x7ff", OpImm { op: A::Or, rd: A0, rs1: A1, imm: 0x7ff }),
        ("andi a0, a1, 0xff", OpImm { op: A::And, rd: A0, rs1: A1, imm: 0xff }),
        ("slli a0, a1, 63", OpImm { op: A::Sll, rd: A0, rs1: A1, imm: 63 }),
        ("srli a0, a1, 1", OpImm { op: A::Srl, rd: A0, rs1: A1, imm: 1 }),
        ("srai a0, a1, 63", OpImm { op: A::Sra, rd: A0, rs1: A1, imm: 63 }),
        ("add s2, s3, s4", Op { op: A::Add, rd: S2, rs1: S3, rs2: S4 }),
        ("sub s2, s3, s4", Op { op: A::Sub, rd: S2, rs1: S3, rs2: S4 }),
        ("sll s2, s3, s4", Op { op: A::Sll, rd: S2, rs1: S3, rs2: S4 }),
        ("slt s2, s3, s4", Op { op: A::Slt, rd: S2, rs1: S3, rs2: S4 }),
        ("sltu s2, s3, s4", Op { op: A::Sltu, rd: S2, rs1: S3, rs2: S4 }),
        ("xor s2, s3, s4", Op { op: A::Xor, rd: S2, rs1: S3, rs2: S4 }),
        ("srl s2, s3, s4", Op { op: A::Srl, rd: S2, rs1: S3, rs2: S4 }),
        ("sra s2, s3, s4", Op { op: A::Sra, rd: S2, rs1: S3, rs2: S4 }),
        ("or s2, s3, s4", Op { op: A::Or, rd: S2, rs1: S3, rs2: S4 }),
        ("and t4, t5, t6", Op { op: A::And, rd: T4, rs1: T5, rs2: T6 }),
        ("addiw a0, a1, -1", OpImm32 { op: W::Add, rd: A0, rs1: A1, imm: -1 }),
        ("slliw a0, a1, 31", OpImm32 { op: W::Sll, rd: A0, rs1: A1, imm: 31 }),
        ("srliw a0, a1, 0", OpImm32 { op: W::Srl, rd: A0, rs1: A1, imm: 0 }),
        ("sraiw a0, a1, 31", OpImm32 { op: W::Sra, rd: A0, rs1: A1, imm: 31 }),
        ("addw a0, a1, a2", Op32 { op: W::Add, rd: A0, rs1: A1, rs2: A2 }),
        ("subw a0, a1, a2", Op32 { op: W::Sub, rd: A0, rs1: A1, rs2: A2 }),
        ("sllw a0, a1, a2", Op32 { op: W::Sll, rd: A0, rs1: A1, rs2: A2 }),
        ("srlw a0, a1, a2", Op32 { op: W::Srl, rd: A0, rs1: A1, rs2: A2 }),
        ("sraw a0, a1, a2", Op32 { op: W::Sra, rd: A0, rs1: A1, rs2: A2 }),
        ("fence", Fence { fm: 0, pred: 0b1111, succ: 0b1111 }),
        ("fence rw, w", Fence { fm: 0, pred: 0b0011, succ: 0b0001 }),
        ("fence.tso", Fence { fm: 0b1000, pred: 0b0011, succ: 0b0011 }),
        ("ecall", Ecall),
        ("ebreak", Ebreak),
    ]
}

/// Decode every word of an assembled fixture and compare against `expected`, which lists
/// the source lines of the `.s` file in order alongside the `Inst` each should decode to.
fn check(source: &str, binary: &[u8], expected: &[(&str, Inst)]) {
    let lines: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('.'))
        .collect();
    // Split the byte stream into instructions using the length encoded in the low bits.
    let mut words = Vec::new();
    let mut rest = binary;
    while !rest.is_empty() {
        let len = inst_len(u16::from_le_bytes([rest[0], rest[1]])) as usize;
        let mut w = [0u8; 4];
        w[..len].copy_from_slice(&rest[..len]);
        words.push(u32::from_le_bytes(w));
        rest = &rest[len..];
    }

    assert_eq!(lines.len(), words.len(), "fixture .bin is stale; run tests/fixtures/build.sh");
    assert_eq!(lines.len(), expected.len(), "fixture .s and the table are out of sync");

    for ((line, word), (asm, inst)) in lines.iter().zip(&words).zip(expected) {
        assert_eq!(line, asm, "fixture .s and the table are out of order");
        assert_eq!(decode(*word), Ok(*inst), "{asm} ({word:#010x})");
    }
}

#[test]
fn rv64i() {
    check(
        include_str!("../../../tests/fixtures/rv64i.s"),
        include_bytes!("../../../tests/fixtures/rv64i.bin"),
        &rv64i_expected(),
    );
}

#[test]
fn rv64m() {
    use AluOp as A;
    use AluOp32 as W;
    use Inst::*;
    let expected = [
        ("mul a0, a1, a2", Op { op: A::Mul, rd: A0, rs1: A1, rs2: A2 }),
        ("mulh a0, a1, a2", Op { op: A::Mulh, rd: A0, rs1: A1, rs2: A2 }),
        ("mulhsu a0, a1, a2", Op { op: A::Mulhsu, rd: A0, rs1: A1, rs2: A2 }),
        ("mulhu a0, a1, a2", Op { op: A::Mulhu, rd: A0, rs1: A1, rs2: A2 }),
        ("div s2, s3, s4", Op { op: A::Div, rd: S2, rs1: S3, rs2: S4 }),
        ("divu s2, s3, s4", Op { op: A::Divu, rd: S2, rs1: S3, rs2: S4 }),
        ("rem t4, t5, t6", Op { op: A::Rem, rd: T4, rs1: T5, rs2: T6 }),
        ("remu t4, t5, t6", Op { op: A::Remu, rd: T4, rs1: T5, rs2: T6 }),
        ("mulw a0, a1, a2", Op32 { op: W::Mul, rd: A0, rs1: A1, rs2: A2 }),
        ("divw a0, a1, a2", Op32 { op: W::Div, rd: A0, rs1: A1, rs2: A2 }),
        ("divuw a0, a1, a2", Op32 { op: W::Divu, rd: A0, rs1: A1, rs2: A2 }),
        ("remw a0, a1, a2", Op32 { op: W::Rem, rd: A0, rs1: A1, rs2: A2 }),
        ("remuw a0, a1, a2", Op32 { op: W::Remu, rd: A0, rs1: A1, rs2: A2 }),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64m.s"),
        include_bytes!("../../../tests/fixtures/rv64m.bin"),
        &expected,
    );
}

#[test]
fn rv64a() {
    use AmoOp as O;
    use AmoWidth::{D, W};
    use Inst::*;
    let amo = |op, width, aq, rl| Amo { op, width, rd: A0, rs1: A2, rs2: A1, aq, rl };
    let amo_d = |op, aq, rl| Amo { op, width: D, rd: T0, rs1: SP, rs2: T1, aq, rl };
    let expected = [
        ("lr.w a0, (a1)", Lr { width: W, rd: A0, rs1: A1, aq: false, rl: false }),
        ("lr.d.aqrl a0, (a1)", Lr { width: D, rd: A0, rs1: A1, aq: true, rl: true }),
        ("sc.w a2, a3, (a1)", Sc { width: W, rd: A2, rs1: A1, rs2: A3, aq: false, rl: false }),
        ("sc.d.rl a2, a3, (a1)", Sc { width: D, rd: A2, rs1: A1, rs2: A3, aq: false, rl: true }),
        ("amoswap.w a0, a1, (a2)", amo(O::Swap, W, false, false)),
        ("amoadd.w.aq a0, a1, (a2)", amo(O::Add, W, true, false)),
        ("amoxor.w a0, a1, (a2)", amo(O::Xor, W, false, false)),
        ("amoand.w a0, a1, (a2)", amo(O::And, W, false, false)),
        ("amoor.w a0, a1, (a2)", amo(O::Or, W, false, false)),
        ("amomin.w a0, a1, (a2)", amo(O::Min, W, false, false)),
        ("amomax.w a0, a1, (a2)", amo(O::Max, W, false, false)),
        ("amominu.w a0, a1, (a2)", amo(O::Minu, W, false, false)),
        ("amomaxu.w.rl a0, a1, (a2)", amo(O::Maxu, W, false, true)),
        ("amoswap.d t0, t1, (sp)", amo_d(O::Swap, false, false)),
        ("amoadd.d t0, t1, (sp)", amo_d(O::Add, false, false)),
        ("amoxor.d t0, t1, (sp)", amo_d(O::Xor, false, false)),
        ("amoand.d t0, t1, (sp)", amo_d(O::And, false, false)),
        ("amoor.d t0, t1, (sp)", amo_d(O::Or, false, false)),
        ("amomin.d t0, t1, (sp)", amo_d(O::Min, false, false)),
        ("amomax.d t0, t1, (sp)", amo_d(O::Max, false, false)),
        ("amominu.d t0, t1, (sp)", amo_d(O::Minu, false, false)),
        ("amomaxu.d.aqrl t0, t1, (sp)", amo_d(O::Maxu, true, true)),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64a.s"),
        include_bytes!("../../../tests/fixtures/rv64a.bin"),
        &expected,
    );
}

#[test]
fn rv64c() {
    use AluOp as A;
    use AluOp32 as W;
    use BranchCond as C;
    use Inst::*;
    use LoadWidth as L;
    use StoreWidth as S;
    let expected = [
        ("c.addi4spn s0, sp, 4", OpImm { op: A::Add, rd: S0, rs1: SP, imm: 4 }),
        ("c.addi4spn a5, sp, 1020", OpImm { op: A::Add, rd: A5, rs1: SP, imm: 1020 }),
        ("c.lw a0, 0(a1)", Load { width: L::W, rd: A0, rs1: A1, offset: 0 }),
        ("c.lw s1, 124(a5)", Load { width: L::W, rd: S1, rs1: A5, offset: 124 }),
        ("c.ld a0, 0(a1)", Load { width: L::D, rd: A0, rs1: A1, offset: 0 }),
        ("c.ld a2, 248(s0)", Load { width: L::D, rd: A2, rs1: S0, offset: 248 }),
        ("c.sw a0, 124(a1)", Store { width: S::W, rs1: A1, rs2: A0, offset: 124 }),
        ("c.sd a3, 248(a4)", Store { width: S::D, rs1: A4, rs2: A3, offset: 248 }),
        ("c.nop", OpImm { op: A::Add, rd: ZERO, rs1: ZERO, imm: 0 }),
        ("c.addi a0, -32", OpImm { op: A::Add, rd: A0, rs1: A0, imm: -32 }),
        ("c.addi t6, 31", OpImm { op: A::Add, rd: T6, rs1: T6, imm: 31 }),
        ("c.addiw a0, -1", OpImm32 { op: W::Add, rd: A0, rs1: A0, imm: -1 }),
        ("c.li ra, 31", OpImm { op: A::Add, rd: RA, rs1: ZERO, imm: 31 }),
        ("c.li a0, -32", OpImm { op: A::Add, rd: A0, rs1: ZERO, imm: -32 }),
        ("c.addi16sp sp, -512", OpImm { op: A::Add, rd: SP, rs1: SP, imm: -512 }),
        ("c.addi16sp sp, 496", OpImm { op: A::Add, rd: SP, rs1: SP, imm: 496 }),
        ("c.lui a0, 1", Lui { rd: A0, imm: 0x1000 }),
        ("c.lui t0, 0xfffff", Lui { rd: T0, imm: -0x1000 }),
        ("c.lui s1, 0x1f", Lui { rd: S1, imm: 0x1f000 }),
        ("c.srli a0, 1", OpImm { op: A::Srl, rd: A0, rs1: A0, imm: 1 }),
        ("c.srli a5, 63", OpImm { op: A::Srl, rd: A5, rs1: A5, imm: 63 }),
        ("c.srai s0, 63", OpImm { op: A::Sra, rd: S0, rs1: S0, imm: 63 }),
        ("c.andi a1, -32", OpImm { op: A::And, rd: A1, rs1: A1, imm: -32 }),
        ("c.andi a1, 31", OpImm { op: A::And, rd: A1, rs1: A1, imm: 31 }),
        ("c.sub s0, s1", Op { op: A::Sub, rd: S0, rs1: S0, rs2: S1 }),
        ("c.xor a0, a1", Op { op: A::Xor, rd: A0, rs1: A0, rs2: A1 }),
        ("c.or a2, a3", Op { op: A::Or, rd: A2, rs1: A2, rs2: A3 }),
        ("c.and a4, a5", Op { op: A::And, rd: A4, rs1: A4, rs2: A5 }),
        ("c.subw s0, a5", Op32 { op: W::Sub, rd: S0, rs1: S0, rs2: A5 }),
        ("c.addw a0, a1", Op32 { op: W::Add, rd: A0, rs1: A0, rs2: A1 }),
        ("c.j -2048", Jal { rd: ZERO, offset: -2048 }),
        ("c.j 2046", Jal { rd: ZERO, offset: 2046 }),
        ("c.beqz a0, -256", Branch { cond: C::Eq, rs1: A0, rs2: ZERO, offset: -256 }),
        ("c.bnez s1, 254", Branch { cond: C::Ne, rs1: S1, rs2: ZERO, offset: 254 }),
        ("c.slli a0, 1", OpImm { op: A::Sll, rd: A0, rs1: A0, imm: 1 }),
        ("c.slli t6, 63", OpImm { op: A::Sll, rd: T6, rs1: T6, imm: 63 }),
        ("c.lwsp ra, 252(sp)", Load { width: L::W, rd: RA, rs1: SP, offset: 252 }),
        ("c.lwsp a0, 0(sp)", Load { width: L::W, rd: A0, rs1: SP, offset: 0 }),
        ("c.ldsp ra, 504(sp)", Load { width: L::D, rd: RA, rs1: SP, offset: 504 }),
        ("c.ldsp a0, 8(sp)", Load { width: L::D, rd: A0, rs1: SP, offset: 8 }),
        ("c.jr ra", Jalr { rd: ZERO, rs1: RA, offset: 0 }),
        ("c.mv a0, a1", Op { op: A::Add, rd: A0, rs1: ZERO, rs2: A1 }),
        ("c.ebreak", Ebreak),
        ("c.jalr t0", Jalr { rd: RA, rs1: T0, offset: 0 }),
        ("c.add a0, t6", Op { op: A::Add, rd: A0, rs1: A0, rs2: T6 }),
        ("c.swsp ra, 252(sp)", Store { width: S::W, rs1: SP, rs2: RA, offset: 252 }),
        ("c.sdsp s0, 504(sp)", Store { width: S::D, rs1: SP, rs2: S0, offset: 504 }),
        // f registers: fa0 = f10, fs1 = f9, ft0 = f0, fs11 = f27
        ("c.fld fa0, 248(a1)", FLoad { width: FWidth::D, rd: 10, rs1: A1, offset: 248 }),
        ("c.fsd fs1, 0(a5)", FStore { width: FWidth::D, rs1: A5, rs2: 9, offset: 0 }),
        ("c.fldsp ft0, 504(sp)", FLoad { width: FWidth::D, rd: 0, rs1: SP, offset: 504 }),
        ("c.fsdsp fs11, 8(sp)", FStore { width: FWidth::D, rs1: SP, rs2: 27, offset: 8 }),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64c.s"),
        include_bytes!("../../../tests/fixtures/rv64c.bin"),
        &expected,
    );
}

#[test]
fn rv64fd() {
    use FWidth::{D, S};
    use Inst::*;
    use SgnjKind::*;
    // f register numbers by ABI name.
    const FT0: u8 = 0;
    const FT1: u8 = 1;
    const FT3: u8 = 3;
    const FT4: u8 = 4;
    const FT5: u8 = 5;
    const FS0: u8 = 8;
    const FA0: u8 = 10;
    const FA1: u8 = 11;
    const FA2: u8 = 12;
    const FA5: u8 = 15;
    const FA7: u8 = 17;
    const FS11: u8 = 27;
    const FT2: u8 = 2;
    const FS1: u8 = 9;
    const FS2: u8 = 18;
    const FS3: u8 = 19;
    const FA3: u8 = 13;
    const FA4: u8 = 14;

    const T2: u8 = 7;
    let expected = [
        ("flw fa0, -4(sp)", FLoad { width: S, rd: FA0, rs1: SP, offset: -4 }),
        ("fld fs11, 2040(a0)", FLoad { width: D, rd: FS11, rs1: A0, offset: 2040 }),
        ("fsw ft0, 0(a1)", FStore { width: S, rs1: A1, rs2: FT0, offset: 0 }),
        ("fsd fa7, -2048(s0)", FStore { width: D, rs1: S0, rs2: FA7, offset: -2048 }),
        ("fmv.x.w a0, fa1", FMvToInt { width: S, rd: A0, rs1: FA1 }),
        ("fmv.w.x ft1, t2", FMvFromInt { width: S, rd: FT1, rs1: T2 }),
        ("fmv.x.d a2, fs0", FMvToInt { width: D, rd: A2, rs1: FS0 }),
        ("fmv.d.x fa5, zero", FMvFromInt { width: D, rd: FA5, rs1: ZERO }),
        ("fsgnj.s fa0, fa1, fa2", FSgnj { kind: Sgnj, width: S, rd: FA0, rs1: FA1, rs2: FA2 }),
        ("fsgnjn.s fa0, fa1, fa2", FSgnj { kind: Sgnjn, width: S, rd: FA0, rs1: FA1, rs2: FA2 }),
        ("fsgnjx.s fa0, fa1, fa2", FSgnj { kind: Sgnjx, width: S, rd: FA0, rs1: FA1, rs2: FA2 }),
        ("fsgnj.d ft3, ft4, ft5", FSgnj { kind: Sgnj, width: D, rd: FT3, rs1: FT4, rs2: FT5 }),
        ("fsgnjn.d ft3, ft4, ft5", FSgnj { kind: Sgnjn, width: D, rd: FT3, rs1: FT4, rs2: FT5 }),
        ("fsgnjx.d ft3, ft4, ft5", FSgnj { kind: Sgnjx, width: D, rd: FT3, rs1: FT4, rs2: FT5 }),
        (
            "fadd.s fa0, fa1, fa2, rne",
            FArith { op: FArithOp::Add, width: S, rd: FA0, rs1: FA1, rs2: FA2, rm: 0 },
        ),
        (
            "fsub.d ft0, ft1, ft2, rtz",
            FArith { op: FArithOp::Sub, width: D, rd: FT0, rs1: FT1, rs2: FT2, rm: 1 },
        ),
        (
            "fmul.s fs0, fs1, fs2, rdn",
            FArith { op: FArithOp::Mul, width: S, rd: FS0, rs1: FS1, rs2: FS2, rm: 2 },
        ),
        (
            "fdiv.d fa3, fa4, fa5, rup",
            FArith { op: FArithOp::Div, width: D, rd: FA3, rs1: FA4, rs2: FA5, rm: 3 },
        ),
        (
            "fadd.d fa0, fa1, fa2, rmm",
            FArith { op: FArithOp::Add, width: D, rd: FA0, rs1: FA1, rs2: FA2, rm: 4 },
        ),
        (
            "fmul.d fa0, fa1, fa2, dyn",
            FArith { op: FArithOp::Mul, width: D, rd: FA0, rs1: FA1, rs2: FA2, rm: RM_DYN },
        ),
        ("fsqrt.s ft3, ft4", FSqrt { width: S, rd: FT3, rs1: FT4, rm: RM_DYN }),
        ("fsqrt.d ft3, ft4, rtz", FSqrt { width: D, rd: FT3, rs1: FT4, rm: 1 }),
        (
            "fmadd.s fa0, fa1, fa2, fa3",
            FFma { op: FmaOp::Madd, width: S, rd: FA0, rs1: FA1, rs2: FA2, rs3: FA3, rm: RM_DYN },
        ),
        (
            "fmsub.d fa0, fa1, fa2, fa3, rne",
            FFma { op: FmaOp::Msub, width: D, rd: FA0, rs1: FA1, rs2: FA2, rs3: FA3, rm: 0 },
        ),
        (
            "fnmsub.s ft0, ft1, ft2, ft3, rtz",
            FFma { op: FmaOp::Nmsub, width: S, rd: FT0, rs1: FT1, rs2: FT2, rs3: FT3, rm: 1 },
        ),
        (
            "fnmadd.d fs0, fs1, fs2, fs3",
            FFma { op: FmaOp::Nmadd, width: D, rd: FS0, rs1: FS1, rs2: FS2, rs3: FS3, rm: RM_DYN },
        ),
        ("fmin.s fa0, fa1, fa2", FMinMax { max: false, width: S, rd: FA0, rs1: FA1, rs2: FA2 }),
        ("fmax.d fa0, fa1, fa2", FMinMax { max: true, width: D, rd: FA0, rs1: FA1, rs2: FA2 }),
        ("feq.s a0, fa1, fa2", FCmp { op: FCmpOp::Eq, width: S, rd: A0, rs1: FA1, rs2: FA2 }),
        ("flt.d a1, ft0, ft1", FCmp { op: FCmpOp::Lt, width: D, rd: A1, rs1: FT0, rs2: FT1 }),
        ("fle.s a2, fs0, fs1", FCmp { op: FCmpOp::Le, width: S, rd: A2, rs1: FS0, rs2: FS1 }),
        ("fclass.d a3, fa0", FClass { width: D, rd: A3, rs1: FA0 }),
        ("fcvt.w.s a0, fa0, rtz", FCvtToInt { from: S, to: IntType::W, rd: A0, rs1: FA0, rm: 1 }),
        ("fcvt.wu.d a1, fa1, rne", FCvtToInt { from: D, to: IntType::Wu, rd: A1, rs1: FA1, rm: 0 }),
        ("fcvt.l.s a2, fa2", FCvtToInt { from: S, to: IntType::L, rd: A2, rs1: FA2, rm: RM_DYN }),
        ("fcvt.lu.d a3, fa3, rmm", FCvtToInt { from: D, to: IntType::Lu, rd: A3, rs1: FA3, rm: 4 }),
        ("fcvt.s.w fa0, a0", FCvtFromInt { from: IntType::W, to: S, rd: FA0, rs1: A0, rm: RM_DYN }),
        ("fcvt.d.wu fa1, a1", FCvtFromInt { from: IntType::Wu, to: D, rd: FA1, rs1: A1, rm: 0 }),
        ("fcvt.s.l fa2, a2, rdn", FCvtFromInt { from: IntType::L, to: S, rd: FA2, rs1: A2, rm: 2 }),
        (
            "fcvt.d.lu fa3, a3",
            FCvtFromInt { from: IntType::Lu, to: D, rd: FA3, rs1: A3, rm: RM_DYN },
        ),
        ("fcvt.s.d fa0, fa1", FCvtFloat { to: S, rd: FA0, rs1: FA1, rm: RM_DYN }),
        ("fcvt.d.s fa2, fa3", FCvtFloat { to: D, rd: FA2, rs1: FA3, rm: 0 }),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64fd.s"),
        include_bytes!("../../../tests/fixtures/rv64fd.bin"),
        &expected,
    );
}

#[test]
fn zicsr() {
    use Inst::Csr as I;
    use champollion_rv::decoder::Csr::*;
    use champollion_rv::decoder::CsrOp::*;
    let expected = [
        ("csrrw a0, fcsr, a1", I { op: Rw, csr: Fcsr, rd: A0, src: A1, imm: false }),
        ("csrrs t0, fflags, zero", I { op: Rs, csr: Fflags, rd: T0, src: ZERO, imm: false }),
        ("csrrc zero, frm, a2", I { op: Rc, csr: Frm, rd: ZERO, src: A2, imm: false }),
        ("csrrwi a0, frm, 3", I { op: Rw, csr: Frm, rd: A0, src: 3, imm: true }),
        ("csrrsi a1, fflags, 31", I { op: Rs, csr: Fflags, rd: A1, src: 31, imm: true }),
        ("csrrci a2, fcsr, 1", I { op: Rc, csr: Fcsr, rd: A2, src: 1, imm: true }),
        ("csrrs a3, cycle, zero", I { op: Rs, csr: Cycle, rd: A3, src: ZERO, imm: false }),
        ("csrrs a4, time, zero", I { op: Rs, csr: Time, rd: A4, src: ZERO, imm: false }),
        ("csrrs a5, instret, zero", I { op: Rs, csr: Instret, rd: A5, src: ZERO, imm: false }),
        ("csrrc s0, instret, zero", I { op: Rc, csr: Instret, rd: S0, src: ZERO, imm: false }),
    ];
    check(
        include_str!("../../../tests/fixtures/zicsr.s"),
        include_bytes!("../../../tests/fixtures/zicsr.bin"),
        &expected,
    );
}

#[test]
fn rv64zb() {
    use AluOp as A;
    use AluOp32 as W;
    use Inst::*;
    use LoadWidth as L;
    use StoreWidth as S;
    let expected = [
        // Zba
        ("sh1add a0, a1, a2", Op { op: A::Sh1Add, rd: A0, rs1: A1, rs2: A2 }),
        ("sh2add a3, a4, a5", Op { op: A::Sh2Add, rd: A3, rs1: A4, rs2: A5 }),
        ("sh3add a6, a7, s0", Op { op: A::Sh3Add, rd: A6, rs1: A7, rs2: S0 }),
        ("sh1add.uw s1, s2, s3", Op { op: A::Sh1AddUw, rd: S1, rs1: S2, rs2: S3 }),
        ("sh2add.uw s4, s5, s6", Op { op: A::Sh2AddUw, rd: S4, rs1: S5, rs2: S6 }),
        ("sh3add.uw s7, s8, s9", Op { op: A::Sh3AddUw, rd: S7, rs1: S8, rs2: S9 }),
        ("add.uw s10, s11, a0", Op { op: A::AddUw, rd: S10, rs1: S11, rs2: A0 }),
        ("slli.uw a1, a2, 40", OpImm { op: A::SllUw, rd: A1, rs1: A2, imm: 40 }),
        // Zbb register-register
        ("andn a0, a1, a2", Op { op: A::Andn, rd: A0, rs1: A1, rs2: A2 }),
        ("orn a3, a4, a5", Op { op: A::Orn, rd: A3, rs1: A4, rs2: A5 }),
        ("xnor a6, a7, s0", Op { op: A::Xnor, rd: A6, rs1: A7, rs2: S0 }),
        ("max s7, s8, s9", Op { op: A::Max, rd: S7, rs1: S8, rs2: S9 }),
        ("maxu s10, s11, a0", Op { op: A::Maxu, rd: S10, rs1: S11, rs2: A0 }),
        ("min a1, a2, a3", Op { op: A::Min, rd: A1, rs1: A2, rs2: A3 }),
        ("minu a4, a5, a6", Op { op: A::Minu, rd: A4, rs1: A5, rs2: A6 }),
        ("rol s9, s10, s11", Op { op: A::Rol, rd: S9, rs1: S10, rs2: S11 }),
        ("ror a0, a1, a2", Op { op: A::Ror, rd: A0, rs1: A1, rs2: A2 }),
        // Zbb unary and immediate
        ("clz s1, s2", OpImm { op: A::Clz, rd: S1, rs1: S2, imm: 0 }),
        ("ctz s3, s4", OpImm { op: A::Ctz, rd: S3, rs1: S4, imm: 0 }),
        ("cpop s5, s6", OpImm { op: A::Cpop, rd: S5, rs1: S6, imm: 0 }),
        ("sext.b a7, s0", OpImm { op: A::SextB, rd: A7, rs1: S0, imm: 0 }),
        ("sext.h s1, s2", OpImm { op: A::SextH, rd: S1, rs1: S2, imm: 0 }),
        ("zext.h s3, s4", Op { op: A::ZextH, rd: S3, rs1: S4, rs2: 0 }),
        ("rev8 s5, s6", OpImm { op: A::Rev8, rd: S5, rs1: S6, imm: 0 }),
        ("orc.b s7, s8", OpImm { op: A::OrcB, rd: S7, rs1: S8, imm: 0 }),
        ("rori a3, a4, 60", OpImm { op: A::Ror, rd: A3, rs1: A4, imm: 60 }),
        ("rolw a5, a6, a7", Op32 { op: W::Rol, rd: A5, rs1: A6, rs2: A7 }),
        ("rorw s0, s1, s2", Op32 { op: W::Ror, rd: S0, rs1: S1, rs2: S2 }),
        ("roriw s3, s4, 25", OpImm32 { op: W::Ror, rd: S3, rs1: S4, imm: 25 }),
        ("clzw a5, a6", OpImm32 { op: W::Clz, rd: A5, rs1: A6, imm: 0 }),
        ("ctzw s0, s1", OpImm32 { op: W::Ctz, rd: S0, rs1: S1, imm: 0 }),
        ("cpopw s2, s3", OpImm32 { op: W::Cpop, rd: S2, rs1: S3, imm: 0 }),
        // Zbs
        ("bclr s5, s6, s7", Op { op: A::Bclr, rd: S5, rs1: S6, rs2: S7 }),
        ("bext s8, s9, s10", Op { op: A::Bext, rd: S8, rs1: S9, rs2: S10 }),
        ("binv s11, a0, a1", Op { op: A::Binv, rd: S11, rs1: A0, rs2: A1 }),
        ("bset a2, a3, a4", Op { op: A::Bset, rd: A2, rs1: A3, rs2: A4 }),
        ("bclri a5, a6, 63", OpImm { op: A::Bclr, rd: A5, rs1: A6, imm: 63 }),
        ("bexti a7, s0, 20", OpImm { op: A::Bext, rd: A7, rs1: S0, imm: 20 }),
        ("binvi s1, s2, 33", OpImm { op: A::Binv, rd: S1, rs1: S2, imm: 33 }),
        ("bseti s3, s4, 42", OpImm { op: A::Bset, rd: S3, rs1: S4, imm: 42 }),
        // Zicond
        ("czero.eqz s5, s6, s7", Op { op: A::CzeroEqz, rd: S5, rs1: S6, rs2: S7 }),
        ("czero.nez s8, s9, s10", Op { op: A::CzeroNez, rd: S8, rs1: S9, rs2: S10 }),
        // Zcb
        ("c.lbu a0, 2(s1)", Load { width: L::Bu, rd: A0, rs1: S1, offset: 2 }),
        ("c.lhu a1, 2(s1)", Load { width: L::Hu, rd: A1, rs1: S1, offset: 2 }),
        ("c.sb a2, 3(s1)", Store { width: S::B, rs1: S1, rs2: A2, offset: 3 }),
        ("c.sh a3, 2(s1)", Store { width: S::H, rs1: S1, rs2: A3, offset: 2 }),
        ("c.zext.b a0", OpImm { op: A::And, rd: A0, rs1: A0, imm: 0xff }),
        ("c.sext.b a1", OpImm { op: A::SextB, rd: A1, rs1: A1, imm: 0 }),
        ("c.zext.h a2", OpImm { op: A::ZextH, rd: A2, rs1: A2, imm: 0 }),
        ("c.sext.h a3", OpImm { op: A::SextH, rd: A3, rs1: A3, imm: 0 }),
        ("c.zext.w a4", Op { op: A::AddUw, rd: A4, rs1: A4, rs2: 0 }),
        ("c.not a5", OpImm { op: A::Xor, rd: A5, rs1: A5, imm: -1 }),
        ("c.mul a0, a1", Op { op: A::Mul, rd: A0, rs1: A0, rs2: A1 }),
        // Zbkb
        ("brev8 a0, a1", OpImm { op: A::Brev8, rd: A0, rs1: A1, imm: 0 }),
        ("pack a2, a3, a4", Op { op: A::Pack, rd: A2, rs1: A3, rs2: A4 }),
        ("packh a5, a6, a7", Op { op: A::Packh, rd: A5, rs1: A6, rs2: A7 }),
        ("packw s0, s1, s2", Op { op: A::Packw, rd: S0, rs1: S1, rs2: S2 }),
        // Zbc
        ("clmul s3, s4, s5", Op { op: A::Clmul, rd: S3, rs1: S4, rs2: S5 }),
        ("clmulh s6, s7, s8", Op { op: A::Clmulh, rd: S6, rs1: S7, rs2: S8 }),
        ("clmulr s9, s10, s11", Op { op: A::Clmulr, rd: S9, rs1: S10, rs2: S11 }),
        // Zbkx
        ("xperm4 t4, t5, t6", Op { op: A::Xperm4, rd: T4, rs1: T5, rs2: T6 }),
        ("xperm8 a0, a1, a2", Op { op: A::Xperm8, rd: A0, rs1: A1, rs2: A2 }),
        // The sign-extending Zcb halfword load, and Zcmop (nops whose odd register
        // field selects the N).
        ("c.lh a1, 2(s1)", Load { width: L::H, rd: A1, rs1: S1, offset: 2 }),
        ("c.mop.1", OpImm { op: A::Add, rd: 0, rs1: 0, imm: 0 }),
        ("c.mop.15", OpImm { op: A::Add, rd: 0, rs1: 0, imm: 0 }),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64zb.s"),
        include_bytes!("../../../tests/fixtures/rv64zb.bin"),
        &expected,
    );
}

#[test]
fn rv64v() {
    use Inst::*;
    use VCmpOp as C;
    use VMemKind as M;
    use VSel::*;
    use VSetKind as K;
    use VStoreKind as SM;
    use VWidth::*;
    let expected = [
        (
            "vsetvli a0, a1, e8, m1, ta, ma",
            VSetvl { rd: A0, rs1: A1, rs2: 0x0c0, kind: K::RegisterAvl },
        ),
        (
            "vsetvli a0, a1, e32, mf2, ta, ma",
            VSetvl { rd: A0, rs1: A1, rs2: 0x0d7, kind: K::RegisterAvl },
        ),
        (
            "vsetvli a0, a1, e64, m1, ta, ma",
            VSetvl { rd: A0, rs1: A1, rs2: 0x0d8, kind: K::RegisterAvl },
        ),
        (
            "vsetivli a0, 16, e8, m1, tu, mu",
            VSetvl { rd: A0, rs1: 16, rs2: 0x000, kind: K::ImmediateAvl },
        ),
        (
            "vsetivli a0, 2, e8, mf8, ta, ma",
            VSetvl { rd: A0, rs1: 2, rs2: 0x0c5, kind: K::ImmediateAvl },
        ),
        ("vsetvli a0, a1, e8, m1", VSetvl { rd: A0, rs1: A1, rs2: 0x000, kind: K::RegisterAvl }),
        ("vsetvl a0, a1, a2", VSetvl { rd: A0, rs1: A1, rs2: 12, kind: K::RegisterVtype }),
        (
            "vle8.v v1, (a0)",
            VLoad { width: E8, kind: M::Unit, vd: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vle8ff.v v1, (a0)",
            VLoad {
                width: E8,
                kind: M::FaultOnlyFirst,
                vd: 1,
                rs1: A0,
                rs2: 0,
                mask: false,
                nf: 0,
            },
        ),
        (
            "vle16.v v1, (a0)",
            VLoad { width: E16, kind: M::Unit, vd: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vle32.v v1, (a0)",
            VLoad { width: E32, kind: M::Unit, vd: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vle64.v v1, (a0)",
            VLoad { width: E64, kind: M::Unit, vd: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vse8.v v1, (a0)",
            VStore { width: E8, kind: SM::Unit, vs3: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vse16.v v1, (a0)",
            VStore { width: E16, kind: SM::Unit, vs3: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vse32.v v1, (a0)",
            VStore { width: E32, kind: SM::Unit, vs3: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vse64.v v1, (a0)",
            VStore { width: E64, kind: SM::Unit, vs3: 1, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vle8.v v1, (a0), v0.t",
            VLoad { width: E8, kind: M::Unit, vd: 1, rs1: A0, rs2: 0, mask: true, nf: 0 },
        ),
        (
            "vlse64.v v1, (a0), a1",
            VLoad { width: E64, kind: M::Stride, vd: 1, rs1: A0, rs2: A1, mask: false, nf: 0 },
        ),
        (
            "vlm.v v0, (a0)",
            VLoad { width: E8, kind: M::Mask, vd: 0, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vl1re64.v v2, (a0)",
            VLoad {
                width: E64,
                kind: M::WholeRegister,
                vd: 2,
                rs1: A0,
                rs2: 0,
                mask: false,
                nf: 0,
            },
        ),
        (
            "vl1r.v v2, (a0)",
            VLoad { width: E8, kind: M::WholeRegister, vd: 2, rs1: A0, rs2: 0, mask: false, nf: 0 },
        ),
        (
            "vl2r.v v2, (a0)",
            VLoad { width: E8, kind: M::WholeRegister, vd: 2, rs1: A0, rs2: 0, mask: false, nf: 1 },
        ),
        (
            "vs1r.v v2, (a0)",
            VStore {
                width: E8,
                kind: SM::WholeRegister,
                vs3: 2,
                rs1: A0,
                rs2: 0,
                mask: false,
                nf: 0,
            },
        ),
        (
            "vs2r.v v2, (a0)",
            VStore {
                width: E8,
                kind: SM::WholeRegister,
                vs3: 2,
                rs1: A0,
                rs2: 0,
                mask: false,
                nf: 1,
            },
        ),
        ("vsext.vf8 v8, v10", VExt { sext: true, narrow: 8, vd: 8, vs2: 10 }),
        ("vzext.vf4 v8, v10", VExt { sext: false, narrow: 4, vd: 8, vs2: 10 }),
        ("vmv.v.i v1, 5", Vmv { vd: 1, src: I(5) }),
        ("vmv.v.x v1, a0", Vmv { vd: 1, src: X(10) }),
        ("vmv.s.x v1, a0", VmvSx { vd: 1, rs1: 10 }),
        ("vmv.x.s a0, v1", VmvXs { rd: 10, vs2: 1 }),
        ("vmv1r.v v2, v3", VmvNr { vd: 2, vs2: 3 }),
        ("vid.v v1", Vid { vd: 1 }),
        ("vmseq.vi v1, v2, 3", VCmp { op: C::Eq, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vmseq.vx v1, v2, a0", VCmp { op: C::Eq, vd: 1, vs2: 2, src: X(10), mask: false }),
        ("vmsne.vv v1, v2, v3", VCmp { op: C::Ne, vd: 1, vs2: 2, src: V(3), mask: false }),
        ("vmsne.vi v1, v2, 3", VCmp { op: C::Ne, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vmsleu.vi v1, v2, 3", VCmp { op: C::LeU, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vmsgtu.vi v1, v2, 3", VCmp { op: C::GtU, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vmor.mm v1, v2, v3", VMask { op: VMaskOp::Or, vd: 1, vs1: 3, vs2: 2 }),
        ("vfirst.m a0, v1", Vfirst { rd: 10, vs2: 1 }),
        ("vcpop.m a0, v1", Vcpop { rd: 10, vs2: 1 }),
        ("vadd.vv v1, v2, v3", VArith { op: VArithOp::Add, vd: 1, vs2: 2, src: V(3), mask: false }),
        (
            "vadd.vx v1, v2, a0",
            VArith { op: VArithOp::Add, vd: 1, vs2: 2, src: X(10), mask: false },
        ),
        ("vadd.vi v1, v2, 3", VArith { op: VArithOp::Add, vd: 1, vs2: 2, src: I(3), mask: false }),
        (
            "vadd.vv v1, v2, v3, v0.t",
            VArith { op: VArithOp::Add, vd: 1, vs2: 2, src: V(3), mask: true },
        ),
        ("vmacc.vv v1, v2, v3", VMacc { add: true, vd: 1, vs2: 3, src: V(2) }),
        ("vmacc.vx v1, a0, v3", VMacc { add: true, vd: 1, vs2: 3, src: X(10) }),
        ("vmadd.vv v1, v2, v3", VMacc { add: false, vd: 1, vs2: 3, src: V(2) }),
        ("vmadd.vx v1, a0, v3", VMacc { add: false, vd: 1, vs2: 3, src: X(10) }),
        ("vxor.vv v1, v2, v3", VArith { op: VArithOp::Xor, vd: 1, vs2: 2, src: V(3), mask: false }),
        ("vxor.vi v1, v2, 3", VArith { op: VArithOp::Xor, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vor.vv v1, v2, v3", VArith { op: VArithOp::Or, vd: 1, vs2: 2, src: V(3), mask: false }),
        ("vand.vv v1, v2, v3", VArith { op: VArithOp::And, vd: 1, vs2: 2, src: V(3), mask: false }),
        ("vand.vi v1, v2, 3", VArith { op: VArithOp::And, vd: 1, vs2: 2, src: I(3), mask: false }),
        (
            "vandn.vv v1, v2, v3",
            VArith { op: VArithOp::Andn, vd: 1, vs2: 2, src: V(3), mask: false },
        ),
        (
            "vmul.vx v1, v2, a0",
            VArith { op: VArithOp::Mul, vd: 1, vs2: 2, src: X(10), mask: false },
        ),
        (
            "vrsub.vi v1, v2, 3",
            VArith { op: VArithOp::Rsub, vd: 1, vs2: 2, src: I(3), mask: false },
        ),
        ("vsll.vi v1, v2, 3", VArith { op: VArithOp::Sll, vd: 1, vs2: 2, src: I(3), mask: false }),
        ("vnsrl.wi v1, v2, 3", Vnsrl { vd: 1, vs2: 2, imm: 3 }),
        (
            "vslidedown.vi v1, v2, 3",
            VArith { op: VArithOp::SlideDown, vd: 1, vs2: 2, src: I(3), mask: false },
        ),
        (
            "vslideup.vi v1, v2, 3",
            VArith { op: VArithOp::SlideUp, vd: 1, vs2: 2, src: I(3), mask: false },
        ),
        (
            "vslide1down.vx v1, v2, a0",
            VArith { op: VArithOp::Slide1Down, vd: 1, vs2: 2, src: X(10), mask: false },
        ),
        (
            "vrgather.vi v1, v2, 3",
            VArith { op: VArithOp::Gather, vd: 1, vs2: 2, src: I(3), mask: false },
        ),
        (
            "vrgather.vv v1, v2, v3",
            VArith { op: VArithOp::Gather, vd: 1, vs2: 2, src: V(3), mask: false },
        ),
        ("vmerge.vvm v1, v2, v3, v0", Vmerge { vd: 1, vs1: 3, vs2: 2 }),
        ("vredor.vs v1, v2, v3", VRed { op: VRedOp::Or, vd: 1, vs1: 3, vs2: 2 }),
        ("vredand.vs v1, v2, v3", VRed { op: VRedOp::And, vd: 1, vs1: 3, vs2: 2 }),
    ];
    check(
        include_str!("../../../tests/fixtures/rv64v.s"),
        include_bytes!("../../../tests/fixtures/rv64v.bin"),
        &expected,
    );
}
