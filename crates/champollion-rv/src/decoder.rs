//! RISC-V instruction decoder.
//!
//! Currently covers the M1 starting set: `add`, `addi`, `beq`, `jal`. Immediates are stored
//! already sign-extended, and branch/jump offsets are byte offsets relative to the
//! instruction's own pc.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Add { rd: u8, rs1: u8, rs2: u8 },
    Addi { rd: u8, rs1: u8, imm: i64 },
    Beq { rs1: u8, rs2: u8, offset: i64 },
    Jal { rd: u8, offset: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// A 16-bit compressed instruction (C extension); not supported yet.
    Compressed(u16),
    Illegal(u32),
}

const OP_IMM: u32 = 0b001_0011;
const OP: u32 = 0b011_0011;
const BRANCH: u32 = 0b110_0011;
const JAL: u32 = 0b110_1111;

#[inline]
fn rd(w: u32) -> u8 {
    ((w >> 7) & 0x1f) as u8
}
#[inline]
fn rs1(w: u32) -> u8 {
    ((w >> 15) & 0x1f) as u8
}
#[inline]
fn rs2(w: u32) -> u8 {
    ((w >> 20) & 0x1f) as u8
}
#[inline]
fn funct3(w: u32) -> u32 {
    (w >> 12) & 0x7
}
#[inline]
fn funct7(w: u32) -> u32 {
    w >> 25
}

/// Sign-extend the low `bits` bits of `v`.
#[inline]
fn sext(v: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v as i64) << shift) >> shift
}

fn imm_i(w: u32) -> i64 {
    sext(w >> 20, 12)
}

fn imm_b(w: u32) -> i64 {
    let v = ((w >> 31) & 1) << 12
        | ((w >> 7) & 1) << 11
        | ((w >> 25) & 0x3f) << 5
        | ((w >> 8) & 0xf) << 1;
    sext(v, 13)
}

fn imm_j(w: u32) -> i64 {
    let v = ((w >> 31) & 1) << 20
        | ((w >> 12) & 0xff) << 12
        | ((w >> 20) & 1) << 11
        | ((w >> 21) & 0x3ff) << 1;
    sext(v, 21)
}

/// Length in bytes of the instruction whose low 16 bits are `lo`.
pub fn inst_len(lo: u16) -> u64 {
    if lo & 0b11 == 0b11 { 4 } else { 2 }
}

pub fn decode(w: u32) -> Result<Inst, DecodeError> {
    if w & 0b11 != 0b11 {
        return Err(DecodeError::Compressed(w as u16));
    }
    let illegal = Err(DecodeError::Illegal(w));
    let inst = match w & 0x7f {
        OP => match (funct3(w), funct7(w)) {
            (0b000, 0b000_0000) => Inst::Add {
                rd: rd(w),
                rs1: rs1(w),
                rs2: rs2(w),
            },
            _ => return illegal,
        },
        OP_IMM => match funct3(w) {
            0b000 => Inst::Addi {
                rd: rd(w),
                rs1: rs1(w),
                imm: imm_i(w),
            },
            _ => return illegal,
        },
        BRANCH => match funct3(w) {
            0b000 => Inst::Beq {
                rs1: rs1(w),
                rs2: rs2(w),
                offset: imm_b(w),
            },
            _ => return illegal,
        },
        JAL => Inst::Jal {
            rd: rd(w),
            offset: imm_j(w),
        },
        _ => return illegal,
    };
    Ok(inst)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // Minimal encoders so tests read like assembly. Offsets/immediates are taken as i32 and
    // truncated to their field width.

    pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
        rs2 << 20 | rs1 << 15 | rd << 7 | OP
    }

    pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32) & 0xfff) << 20 | rs1 << 15 | rd << 7 | OP_IMM
    }

    pub fn beq(rs1: u32, rs2: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 12) & 1) << 31
            | ((o >> 5) & 0x3f) << 25
            | rs2 << 20
            | rs1 << 15
            | ((o >> 1) & 0xf) << 8
            | ((o >> 11) & 1) << 7
            | BRANCH
    }

    pub fn jal(rd: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 20) & 1) << 31
            | ((o >> 1) & 0x3ff) << 21
            | ((o >> 11) & 1) << 20
            | ((o >> 12) & 0xff) << 12
            | rd << 7
            | JAL
    }

    #[test]
    fn decodes_known_encodings() {
        // Encodings cross-checked against GNU as / llvm-mc output.
        assert_eq!(
            decode(0x00b50533),
            Ok(Inst::Add {
                rd: 10,
                rs1: 10,
                rs2: 11
            })
        ); // add a0,a0,a1
        assert_eq!(
            decode(0xfff00293),
            Ok(Inst::Addi {
                rd: 5,
                rs1: 0,
                imm: -1
            })
        ); // li t0,-1
        assert_eq!(
            decode(0x00000013),
            Ok(Inst::Addi {
                rd: 0,
                rs1: 0,
                imm: 0
            })
        ); // nop
        assert_eq!(
            decode(0x00b50463),
            Ok(Inst::Beq {
                rs1: 10,
                rs2: 11,
                offset: 8
            })
        ); // beq a0,a1,.+8
        assert_eq!(decode(0x008000ef), Ok(Inst::Jal { rd: 1, offset: 8 })); // jal ra,.+8
        assert_eq!(decode(0x0000006f), Ok(Inst::Jal { rd: 0, offset: 0 })); // j .
    }

    #[test]
    fn immediate_extremes_roundtrip() {
        for imm in [-2048, -1, 0, 1, 2047] {
            assert_eq!(
                decode(addi(1, 2, imm)),
                Ok(Inst::Addi {
                    rd: 1,
                    rs1: 2,
                    imm: imm as i64
                })
            );
        }
        for off in [-4096, -2, 2, 4094] {
            assert_eq!(
                decode(beq(3, 4, off)),
                Ok(Inst::Beq {
                    rs1: 3,
                    rs2: 4,
                    offset: off as i64
                })
            );
        }
        for off in [-(1 << 20), -2, 2, (1 << 20) - 2] {
            assert_eq!(
                decode(jal(1, off)),
                Ok(Inst::Jal {
                    rd: 1,
                    offset: off as i64
                })
            );
        }
    }

    #[test]
    fn rejects_unknown_and_compressed() {
        assert_eq!(decode(0x0000_0001), Err(DecodeError::Compressed(0x0001))); // c.nop
        assert_eq!(decode(0x4000_0033), Err(DecodeError::Illegal(0x4000_0033))); // sub: not yet
        assert_eq!(decode(0xffff_ffff), Err(DecodeError::Illegal(0xffff_ffff)));
        assert_eq!(decode(0x0000_0000), Err(DecodeError::Compressed(0))); // defined illegal
    }
}
