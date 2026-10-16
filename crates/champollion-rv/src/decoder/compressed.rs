//! C extension: expands 16-bit instructions into the equivalent 32-bit [`Inst`].
//!
//! Every compressed instruction (RVC, and Zcb's byte/half loads, bit-manipulation, and
//! `c.mul`) is an alias for a base instruction, so the interpreter and JIT never need to
//! know an instruction was compressed (beyond its 2-byte length, which only matters for
//! the link value and the next pc).
//!
//! The double-precision forms (`c.fld`, `c.fsd`, `c.fldsp`, `c.fsdsp`) expand to D
//! loads and stores; RV64 has no single-precision compressed forms.

use super::{AluOp, AluOp32, BranchCond, DecodeError, FWidth, Inst, LoadWidth, StoreWidth};

const SP: u8 = 2;

/// Bits `hi..=lo` of `h`, shifted down to bit 0.
#[inline]
fn bits(h: u16, hi: u32, lo: u32) -> u32 {
    (h as u32 >> lo) & ((1 << (hi - lo + 1)) - 1)
}

/// Place bits `hi..=lo` of `h` at bit position `to` (and up).
#[inline]
fn field(h: u16, hi: u32, lo: u32, to: u32) -> u32 {
    bits(h, hi, lo) << to
}

/// Sign-extend the low `n` bits of `v`.
#[inline]
fn sext(v: u32, n: u32) -> i32 {
    ((v << (32 - n)) as i32) >> (32 - n)
}

/// Full register number from a 3-bit compressed register field (x8..x15).
#[inline]
fn creg(h: u16, lo: u32) -> u8 {
    8 + bits(h, lo + 2, lo) as u8
}

/// 6-bit immediate: imm[5] in bit 12, imm[4:0] in bits 6:2.
fn imm6(h: u16) -> u32 {
    field(h, 12, 12, 5) | bits(h, 6, 2)
}

/// Branch offset for `c.beqz`/`c.bnez`: offset[8|4:3] in 12:10, [7:6|2:1|5] in 6:2.
fn offset_b(h: u16) -> i32 {
    let v = field(h, 12, 12, 8)
        | field(h, 11, 10, 3)
        | field(h, 6, 5, 6)
        | field(h, 4, 3, 1)
        | field(h, 2, 2, 5);
    sext(v, 9)
}

/// Jump offset for `c.j`: offset[11|4|9:8|10|6|7|3:1|5] in bits 12:2.
fn offset_j(h: u16) -> i32 {
    let v = field(h, 12, 12, 11)
        | field(h, 11, 11, 4)
        | field(h, 10, 9, 8)
        | field(h, 8, 8, 10)
        | field(h, 7, 7, 6)
        | field(h, 6, 6, 7)
        | field(h, 5, 3, 1)
        | field(h, 2, 2, 5);
    sext(v, 12)
}

pub fn decode_compressed(h: u16) -> Result<Inst, DecodeError> {
    let illegal = Err(DecodeError::Illegal(h as u32));
    let rd = bits(h, 11, 7) as u8; // full rd/rs1 field (quadrants 1 and 2)
    let rs2 = bits(h, 6, 2) as u8; // full rs2 field (quadrant 2)
    let rd_ = creg(h, 2); // rd'/rs2' in bits 4:2
    let rs1_ = creg(h, 7); // rs1'/rd' in bits 9:7

    // Scaled unsigned offsets for loads/stores.
    let off_w = (field(h, 12, 10, 3) | field(h, 6, 6, 2) | field(h, 5, 5, 6)) as i32;
    let off_d = (field(h, 12, 10, 3) | field(h, 6, 5, 6)) as i32;

    let inst = match (h & 0b11, bits(h, 15, 13)) {
        // Quadrant 0
        (0b00, 0b000) => {
            // c.addi4spn: nzuimm[5:4|9:6|2|3] in bits 12:5
            let imm =
                field(h, 12, 11, 4) | field(h, 10, 7, 6) | field(h, 6, 6, 2) | field(h, 5, 5, 3);
            if imm == 0 {
                return illegal; // includes the all-zero instruction
            }
            Inst::OpImm { op: AluOp::Add, rd: rd_, rs1: SP, imm: imm as i32 }
        }
        (0b00, 0b010) => Inst::Load { width: LoadWidth::W, rd: rd_, rs1: rs1_, offset: off_w },
        (0b00, 0b001) => Inst::FLoad { width: FWidth::D, rd: rd_, rs1: rs1_, offset: off_d },
        (0b00, 0b011) => Inst::Load { width: LoadWidth::D, rd: rd_, rs1: rs1_, offset: off_d },
        (0b00, 0b100) => {
            // Zcb: c.lbu/c.lhu/c.lh/c.sb/c.sh. uimm[1] is bit 5, and bit 6 is uimm[0]
            // for the byte forms; the halfword loads repurpose it as the sign selector
            // (0 = c.lhu, 1 = c.lh), and the halfword store requires it clear.
            let uimm1 = bits(h, 5, 5) << 1;
            let uimm0 = bits(h, 6, 6);
            match bits(h, 11, 10) {
                0b00 => Inst::Load {
                    width: LoadWidth::Bu,
                    rd: rd_,
                    rs1: rs1_,
                    offset: (uimm1 | uimm0) as i32,
                },
                0b01 if uimm0 == 0 => {
                    Inst::Load { width: LoadWidth::Hu, rd: rd_, rs1: rs1_, offset: uimm1 as i32 }
                }
                0b01 => {
                    Inst::Load { width: LoadWidth::H, rd: rd_, rs1: rs1_, offset: uimm1 as i32 }
                }
                0b10 => Inst::Store {
                    width: StoreWidth::B,
                    rs1: rs1_,
                    rs2: rd_,
                    offset: (uimm1 | uimm0) as i32,
                },
                0b11 if uimm0 == 0 => {
                    Inst::Store { width: StoreWidth::H, rs1: rs1_, rs2: rd_, offset: uimm1 as i32 }
                }
                _ => return illegal, // c.sh with the reserved bit set
            }
        }
        (0b00, 0b101) => Inst::FStore { width: FWidth::D, rs1: rs1_, rs2: rd_, offset: off_d },
        (0b00, 0b110) => Inst::Store { width: StoreWidth::W, rs1: rs1_, rs2: rd_, offset: off_w },
        (0b00, 0b111) => Inst::Store { width: StoreWidth::D, rs1: rs1_, rs2: rd_, offset: off_d },

        // Quadrant 1
        (0b01, 0b000) => Inst::OpImm { op: AluOp::Add, rd, rs1: rd, imm: sext(imm6(h), 6) },
        (0b01, 0b001) if rd != 0 => {
            Inst::OpImm32 { op: AluOp32::Add, rd, rs1: rd, imm: sext(imm6(h), 6) }
        }
        (0b01, 0b010) => Inst::OpImm { op: AluOp::Add, rd, rs1: 0, imm: sext(imm6(h), 6) },
        (0b01, 0b011) if rd == SP => {
            // c.addi16sp: nzimm[9] in bit 12, nzimm[4|6|8:7|5] in bits 6:2
            let v = field(h, 12, 12, 9)
                | field(h, 6, 6, 4)
                | field(h, 5, 5, 6)
                | field(h, 4, 3, 7)
                | field(h, 2, 2, 5);
            if v == 0 {
                return illegal;
            }
            Inst::OpImm { op: AluOp::Add, rd: SP, rs1: SP, imm: sext(v, 10) }
        }
        (0b01, 0b011) => {
            // c.lui: nzimm[17] in bit 12, nzimm[16:12] in bits 6:2
            if imm6(h) == 0 {
                // Zcmop's c.mop.N are the reserved all-zero-immediate encodings with an
                // odd register: instructions that must not write anything (nops today).
                if rd % 2 == 1 && rd <= 15 {
                    return Ok(Inst::OpImm { op: AluOp::Add, rd: 0, rs1: 0, imm: 0 });
                }
                return illegal;
            }
            Inst::Lui { rd, imm: sext(imm6(h), 6) << 12 }
        }
        (0b01, 0b100) => {
            let (rd, rs1) = (rs1_, rs1_);
            match (bits(h, 11, 10), bits(h, 12, 12), bits(h, 6, 5)) {
                (0b00, _, _) => Inst::OpImm { op: AluOp::Srl, rd, rs1, imm: imm6(h) as i32 },
                (0b01, _, _) => Inst::OpImm { op: AluOp::Sra, rd, rs1, imm: imm6(h) as i32 },
                (0b10, _, _) => Inst::OpImm { op: AluOp::And, rd, rs1, imm: sext(imm6(h), 6) },
                (0b11, 0, 0b00) => Inst::Op { op: AluOp::Sub, rd, rs1, rs2: rd_ },
                (0b11, 0, 0b01) => Inst::Op { op: AluOp::Xor, rd, rs1, rs2: rd_ },
                (0b11, 0, 0b10) => Inst::Op { op: AluOp::Or, rd, rs1, rs2: rd_ },
                (0b11, 0, 0b11) => Inst::Op { op: AluOp::And, rd, rs1, rs2: rd_ },
                (0b11, 1, 0b00) => Inst::Op32 { op: AluOp32::Sub, rd, rs1, rs2: rd_ },
                (0b11, 1, 0b01) => Inst::Op32 { op: AluOp32::Add, rd, rs1, rs2: rd_ },
                // Zcb: c.mul takes the last CA slot (its operands are rd' and rs2');
                // c.zext.b/c.sext.b/c.zext.h/c.sext.h/c.zext.w/c.not fill the slot after
                // it, selected by bits 4:2, operating in place.
                (0b11, 1, 0b10) => Inst::Op { op: AluOp::Mul, rd, rs1, rs2: rd_ },
                (0b11, 1, 0b11) => match bits(h, 4, 2) {
                    0b000 => Inst::OpImm { op: AluOp::And, rd, rs1, imm: 0xff },
                    0b001 => Inst::OpImm { op: AluOp::SextB, rd, rs1, imm: 0 },
                    0b010 => Inst::OpImm { op: AluOp::ZextH, rd, rs1, imm: 0 },
                    0b011 => Inst::OpImm { op: AluOp::SextH, rd, rs1, imm: 0 },
                    0b100 => Inst::Op { op: AluOp::AddUw, rd, rs1, rs2: 0 },
                    0b101 => Inst::OpImm { op: AluOp::Xor, rd, rs1, imm: -1 },
                    _ => return illegal,
                },
                _ => return illegal,
            }
        }
        (0b01, 0b101) => Inst::Jal { rd: 0, offset: offset_j(h) },
        (0b01, 0b110) => {
            Inst::Branch { cond: BranchCond::Eq, rs1: rs1_, rs2: 0, offset: offset_b(h) }
        }
        (0b01, 0b111) => {
            Inst::Branch { cond: BranchCond::Ne, rs1: rs1_, rs2: 0, offset: offset_b(h) }
        }

        // Quadrant 2
        (0b10, 0b000) => Inst::OpImm { op: AluOp::Sll, rd, rs1: rd, imm: imm6(h) as i32 },
        (0b10, 0b010) if rd != 0 => {
            // c.lwsp: uimm[5] in bit 12, uimm[4:2|7:6] in bits 6:2
            let off = field(h, 12, 12, 5) | field(h, 6, 4, 2) | field(h, 3, 2, 6);
            Inst::Load { width: LoadWidth::W, rd, rs1: SP, offset: off as i32 }
        }
        (0b10, 0b001) => {
            // c.fldsp: same offset encoding as c.ldsp; rd = f0 is allowed
            let off = field(h, 12, 12, 5) | field(h, 6, 5, 3) | field(h, 4, 2, 6);
            Inst::FLoad { width: FWidth::D, rd, rs1: SP, offset: off as i32 }
        }
        (0b10, 0b011) if rd != 0 => {
            // c.ldsp: uimm[5] in bit 12, uimm[4:3|8:6] in bits 6:2
            let off = field(h, 12, 12, 5) | field(h, 6, 5, 3) | field(h, 4, 2, 6);
            Inst::Load { width: LoadWidth::D, rd, rs1: SP, offset: off as i32 }
        }
        (0b10, 0b100) => match (bits(h, 12, 12), rd, rs2) {
            (0, 0, 0) => return illegal,
            (0, rs1, 0) => Inst::Jalr { rd: 0, rs1, offset: 0 }, // c.jr
            (0, rd, rs2) => Inst::Op { op: AluOp::Add, rd, rs1: 0, rs2 }, // c.mv
            (1, 0, 0) => Inst::Ebreak,
            (1, rs1, 0) => Inst::Jalr { rd: 1, rs1, offset: 0 }, // c.jalr
            (_, rd, rs2) => Inst::Op { op: AluOp::Add, rd, rs1: rd, rs2 }, // c.add
        },
        (0b10, 0b110) => {
            // c.swsp: uimm[5:2|7:6] in bits 12:7
            let off = field(h, 12, 9, 2) | field(h, 8, 7, 6);
            Inst::Store { width: StoreWidth::W, rs1: SP, rs2, offset: off as i32 }
        }
        (0b10, 0b101) => {
            // c.fsdsp: same offset encoding as c.sdsp
            let off = field(h, 12, 10, 3) | field(h, 9, 7, 6);
            Inst::FStore { width: FWidth::D, rs1: SP, rs2, offset: off as i32 }
        }
        (0b10, 0b111) => {
            // c.sdsp: uimm[5:3|8:6] in bits 12:7
            let off = field(h, 12, 10, 3) | field(h, 9, 7, 6);
            Inst::Store { width: StoreWidth::D, rs1: SP, rs2, offset: off as i32 }
        }

        _ => return illegal,
    };
    Ok(inst)
}
