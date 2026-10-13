//! Checks the decoder against encodings produced by a real assembler (LLVM, via `zig cc`).
//! Each `tests/fixtures/*.s` lists instructions one per line; the matching `.bin` is the
//! assembled `.text`.

use champollion_rv::decoder::inst_len;
use champollion_rv::decoder::{AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, LoadWidth, StoreWidth};
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
const S2: u8 = 18;
const S3: u8 = 19;
const S4: u8 = 20;
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
    ];
    check(
        include_str!("../../../tests/fixtures/rv64c.s"),
        include_bytes!("../../../tests/fixtures/rv64c.bin"),
        &expected,
    );
}
