//! RISC-V instruction decoder.
//!
//! Covers RV64I, M, and A. Immediates are stored already sign-extended (`lui`/`auipc` already
//! shifted left by 12), and branch/jump offsets are byte offsets relative to the
//! instruction's own pc. Every immediate fits in an `i32`, which keeps [`Inst`] at 8 bytes.

/// A decoded instruction.
///
/// - `OpImm`: for shifts, `imm` is the shift amount.
/// - `OpImm32` / `Op32` (`*w` ops): operate on the low 32 bits, sign-extend the result.
/// - `Fence`: `pred`/`succ` are IORW bit sets; `fm` is 0b1000 for `fence.tso`.
/// - `Lr` / `Sc` / `Amo`: `aq`/`rl` are the acquire/release ordering bits. `Sc` writes 0 to
///   `rd` on success, nonzero on failure; `Amo` writes the old memory value to `rd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Lui { rd: u8, imm: i32 },
    Auipc { rd: u8, imm: i32 },
    Jal { rd: u8, offset: i32 },
    Jalr { rd: u8, rs1: u8, offset: i32 },
    Branch { cond: BranchCond, rs1: u8, rs2: u8, offset: i32 },
    Load { width: LoadWidth, rd: u8, rs1: u8, offset: i32 },
    Store { width: StoreWidth, rs1: u8, rs2: u8, offset: i32 },
    OpImm { op: AluOp, rd: u8, rs1: u8, imm: i32 },
    Op { op: AluOp, rd: u8, rs1: u8, rs2: u8 },
    OpImm32 { op: AluOp32, rd: u8, rs1: u8, imm: i32 },
    Op32 { op: AluOp32, rd: u8, rs1: u8, rs2: u8 },
    Fence { fm: u8, pred: u8, succ: u8 },
    Ecall,
    Ebreak,
    Lr { width: AmoWidth, rd: u8, rs1: u8, aq: bool, rl: bool },
    Sc { width: AmoWidth, rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    Amo { op: AmoOp, width: AmoWidth, rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadWidth {
    B,
    H,
    W,
    D,
    Bu,
    Hu,
    Wu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreWidth {
    B,
    H,
    W,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
    // M extension; register-register only.
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp32 {
    Add,
    Sub,
    Sll,
    Srl,
    Sra,
    // M extension; register-register only.
    Mul,
    Div,
    Divu,
    Rem,
    Remu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoWidth {
    W,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoOp {
    Swap,
    Add,
    Xor,
    And,
    Or,
    Min,
    Max,
    Minu,
    Maxu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// A 16-bit compressed instruction (C extension); not supported yet.
    Compressed(u16),
    Illegal(u32),
}

const LOAD: u32 = 0b000_0011;
const MISC_MEM: u32 = 0b000_1111;
const OP_IMM: u32 = 0b001_0011;
const AUIPC: u32 = 0b001_0111;
const OP_IMM_32: u32 = 0b001_1011;
const STORE: u32 = 0b010_0011;
const AMO: u32 = 0b010_1111;
const OP: u32 = 0b011_0011;
const LUI: u32 = 0b011_0111;
const OP_32: u32 = 0b011_1011;
const BRANCH: u32 = 0b110_0011;
const JALR: u32 = 0b110_0111;
const JAL: u32 = 0b110_1111;
const SYSTEM: u32 = 0b111_0011;

const ECALL: u32 = 0x0000_0073;
const EBREAK: u32 = 0x0010_0073;

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
fn sext(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v << shift) as i32) >> shift
}

fn imm_i(w: u32) -> i32 {
    (w as i32) >> 20
}

fn imm_s(w: u32) -> i32 {
    ((w as i32) >> 25) << 5 | ((w >> 7) & 0x1f) as i32
}

fn imm_b(w: u32) -> i32 {
    let v = ((w >> 31) & 1) << 12
        | ((w >> 7) & 1) << 11
        | ((w >> 25) & 0x3f) << 5
        | ((w >> 8) & 0xf) << 1;
    sext(v, 13)
}

fn imm_u(w: u32) -> i32 {
    (w & 0xffff_f000) as i32
}

fn imm_j(w: u32) -> i32 {
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
    let (rd, rs1, rs2, f3, f7) = (rd(w), rs1(w), rs2(w), funct3(w), funct7(w));
    let inst = match w & 0x7f {
        LUI => Inst::Lui { rd, imm: imm_u(w) },
        AUIPC => Inst::Auipc { rd, imm: imm_u(w) },
        JAL => Inst::Jal { rd, offset: imm_j(w) },
        JALR if f3 == 0 => Inst::Jalr { rd, rs1, offset: imm_i(w) },
        BRANCH => {
            let cond = match f3 {
                0b000 => BranchCond::Eq,
                0b001 => BranchCond::Ne,
                0b100 => BranchCond::Lt,
                0b101 => BranchCond::Ge,
                0b110 => BranchCond::Ltu,
                0b111 => BranchCond::Geu,
                _ => return illegal,
            };
            Inst::Branch { cond, rs1, rs2, offset: imm_b(w) }
        }
        LOAD => {
            let width = match f3 {
                0b000 => LoadWidth::B,
                0b001 => LoadWidth::H,
                0b010 => LoadWidth::W,
                0b011 => LoadWidth::D,
                0b100 => LoadWidth::Bu,
                0b101 => LoadWidth::Hu,
                0b110 => LoadWidth::Wu,
                _ => return illegal,
            };
            Inst::Load { width, rd, rs1, offset: imm_i(w) }
        }
        STORE => {
            let width = match f3 {
                0b000 => StoreWidth::B,
                0b001 => StoreWidth::H,
                0b010 => StoreWidth::W,
                0b011 => StoreWidth::D,
                _ => return illegal,
            };
            Inst::Store { width, rs1, rs2, offset: imm_s(w) }
        }
        OP_IMM => {
            // RV64 shifts take a 6-bit shamt, leaving funct6 (bits 31:26) as the selector.
            let shamt = ((w >> 20) & 0x3f) as i32;
            let (op, imm) = match (f3, w >> 26) {
                (0b000, _) => (AluOp::Add, imm_i(w)),
                (0b010, _) => (AluOp::Slt, imm_i(w)),
                (0b011, _) => (AluOp::Sltu, imm_i(w)),
                (0b100, _) => (AluOp::Xor, imm_i(w)),
                (0b110, _) => (AluOp::Or, imm_i(w)),
                (0b111, _) => (AluOp::And, imm_i(w)),
                (0b001, 0b00_0000) => (AluOp::Sll, shamt),
                (0b101, 0b00_0000) => (AluOp::Srl, shamt),
                (0b101, 0b01_0000) => (AluOp::Sra, shamt),
                _ => return illegal,
            };
            Inst::OpImm { op, rd, rs1, imm }
        }
        OP => {
            let op = match (f3, f7) {
                (0b000, 0b000_0000) => AluOp::Add,
                (0b000, 0b010_0000) => AluOp::Sub,
                (0b001, 0b000_0000) => AluOp::Sll,
                (0b010, 0b000_0000) => AluOp::Slt,
                (0b011, 0b000_0000) => AluOp::Sltu,
                (0b100, 0b000_0000) => AluOp::Xor,
                (0b101, 0b000_0000) => AluOp::Srl,
                (0b101, 0b010_0000) => AluOp::Sra,
                (0b110, 0b000_0000) => AluOp::Or,
                (0b111, 0b000_0000) => AluOp::And,
                (0b000, 0b000_0001) => AluOp::Mul,
                (0b001, 0b000_0001) => AluOp::Mulh,
                (0b010, 0b000_0001) => AluOp::Mulhsu,
                (0b011, 0b000_0001) => AluOp::Mulhu,
                (0b100, 0b000_0001) => AluOp::Div,
                (0b101, 0b000_0001) => AluOp::Divu,
                (0b110, 0b000_0001) => AluOp::Rem,
                (0b111, 0b000_0001) => AluOp::Remu,
                _ => return illegal,
            };
            Inst::Op { op, rd, rs1, rs2 }
        }
        OP_IMM_32 => {
            // 32-bit shifts take a 5-bit shamt; funct7 is the full selector.
            let shamt = rs2 as i32;
            let (op, imm) = match (f3, f7) {
                (0b000, _) => (AluOp32::Add, imm_i(w)),
                (0b001, 0b000_0000) => (AluOp32::Sll, shamt),
                (0b101, 0b000_0000) => (AluOp32::Srl, shamt),
                (0b101, 0b010_0000) => (AluOp32::Sra, shamt),
                _ => return illegal,
            };
            Inst::OpImm32 { op, rd, rs1, imm }
        }
        OP_32 => {
            let op = match (f3, f7) {
                (0b000, 0b000_0000) => AluOp32::Add,
                (0b000, 0b010_0000) => AluOp32::Sub,
                (0b001, 0b000_0000) => AluOp32::Sll,
                (0b101, 0b000_0000) => AluOp32::Srl,
                (0b101, 0b010_0000) => AluOp32::Sra,
                (0b000, 0b000_0001) => AluOp32::Mul,
                (0b100, 0b000_0001) => AluOp32::Div,
                (0b101, 0b000_0001) => AluOp32::Divu,
                (0b110, 0b000_0001) => AluOp32::Rem,
                (0b111, 0b000_0001) => AluOp32::Remu,
                _ => return illegal,
            };
            Inst::Op32 { op, rd, rs1, rs2 }
        }
        // rd, rs1, and unknown fm values are reserved for future fence variants and must be
        // ignored, which makes every such encoding a (possibly stronger-than-needed) fence.
        MISC_MEM if f3 == 0 => Inst::Fence {
            fm: (w >> 28) as u8,
            pred: ((w >> 24) & 0xf) as u8,
            succ: ((w >> 20) & 0xf) as u8,
        },
        AMO => {
            let width = match f3 {
                0b010 => AmoWidth::W,
                0b011 => AmoWidth::D,
                _ => return illegal, // Zabha byte/halfword AMOs are not supported
            };
            let (aq, rl) = (w >> 26 & 1 != 0, w >> 25 & 1 != 0);
            let op = match w >> 27 {
                0b00010 if rs2 == 0 => return Ok(Inst::Lr { width, rd, rs1, aq, rl }),
                0b00011 => return Ok(Inst::Sc { width, rd, rs1, rs2, aq, rl }),
                0b00001 => AmoOp::Swap,
                0b00000 => AmoOp::Add,
                0b00100 => AmoOp::Xor,
                0b01100 => AmoOp::And,
                0b01000 => AmoOp::Or,
                0b10000 => AmoOp::Min,
                0b10100 => AmoOp::Max,
                0b11000 => AmoOp::Minu,
                0b11100 => AmoOp::Maxu,
                _ => return illegal,
            };
            Inst::Amo { op, width, rd, rs1, rs2, aq, rl }
        }
        SYSTEM if w == ECALL => Inst::Ecall,
        SYSTEM if w == EBREAK => Inst::Ebreak,
        _ => return illegal,
    };
    Ok(inst)
}

/// Instruction encoders for tests, so programs read like assembly. Immediates and offsets
/// are truncated to their field width; they are checked against real assembler output
/// indirectly, since `decode` is (see `tests/decode_rv64i.rs`) and these must round-trip.
#[cfg(test)]
pub(crate) mod enc {
    use super::*;

    pub fn r(op: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
    }

    pub fn i(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32) & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
    }

    pub fn s(op: u32, f3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let o = imm as u32;
        ((o >> 5) & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (o & 0x1f) << 7 | op
    }

    pub fn b(f3: u32, rs1: u32, rs2: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 12) & 1) << 31
            | ((o >> 5) & 0x3f) << 25
            | rs2 << 20
            | rs1 << 15
            | f3 << 12
            | ((o >> 1) & 0xf) << 8
            | ((o >> 11) & 1) << 7
            | BRANCH
    }

    pub fn u(op: u32, rd: u32, imm20: u32) -> u32 {
        (imm20 & 0xfffff) << 12 | rd << 7 | op
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

    pub fn lui(rd: u32, imm20: u32) -> u32 {
        u(LUI, rd, imm20)
    }
    pub fn auipc(rd: u32, imm20: u32) -> u32 {
        u(AUIPC, rd, imm20)
    }
    pub fn jalr(rd: u32, rs1: u32, off: i32) -> u32 {
        i(JALR, 0, rd, rs1, off)
    }

    pub fn beq(rs1: u32, rs2: u32, off: i32) -> u32 {
        b(0b000, rs1, rs2, off)
    }

    /// Load with funct3 `f3` (0=lb 1=lh 2=lw 3=ld 4=lbu 5=lhu 6=lwu).
    pub fn load(f3: u32, rd: u32, rs1: u32, off: i32) -> u32 {
        i(LOAD, f3, rd, rs1, off)
    }
    /// Store with funct3 `f3` (0=sb 1=sh 2=sw 3=sd).
    pub fn store(f3: u32, rs1: u32, rs2: u32, off: i32) -> u32 {
        s(STORE, f3, rs1, rs2, off)
    }

    pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM, 0b000, rd, rs1, imm)
    }
    pub fn op_imm(f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM, f3, rd, rs1, imm)
    }
    pub fn op(f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(OP, f3, f7, rd, rs1, rs2)
    }
    pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
        op(0b000, 0, rd, rs1, rs2)
    }
    pub fn op_imm_32(f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM_32, f3, rd, rs1, imm)
    }
    pub fn op_32(f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(OP_32, f3, f7, rd, rs1, rs2)
    }

    /// AMO with funct5 `f5` (2=lr 3=sc 1=swap 0=add ...) and funct3 `f3` (2=w 3=d).
    pub fn amo(f5: u32, f3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(AMO, f3, f5 << 2, rd, rs1, rs2)
    }

    pub const FENCE: u32 = 0x0ff0_000f;
    pub const ECALL: u32 = super::ECALL;
    pub const EBREAK: u32 = super::EBREAK;
}

#[cfg(test)]
mod tests {
    use super::enc::*;
    use super::*;

    #[test]
    fn inst_stays_small() {
        assert!(std::mem::size_of::<Inst>() <= 8);
    }

    #[test]
    fn immediate_extremes_roundtrip() {
        for imm in [-2048, -1, 0, 1, 2047] {
            assert_eq!(
                decode(addi(1, 2, imm)),
                Ok(Inst::OpImm { op: AluOp::Add, rd: 1, rs1: 2, imm })
            );
            assert_eq!(
                decode(store(3, 2, 1, imm)),
                Ok(Inst::Store { width: StoreWidth::D, rs1: 2, rs2: 1, offset: imm })
            );
        }
        for offset in [-4096, -2, 2, 4094] {
            assert_eq!(
                decode(beq(3, 4, offset)),
                Ok(Inst::Branch { cond: BranchCond::Eq, rs1: 3, rs2: 4, offset })
            );
        }
        for offset in [-(1 << 20), -2, 2, (1 << 20) - 2] {
            assert_eq!(decode(jal(1, offset)), Ok(Inst::Jal { rd: 1, offset }));
        }
        assert_eq!(decode(lui(1, 0x80000)), Ok(Inst::Lui { rd: 1, imm: i32::MIN }));
    }

    #[test]
    fn rejects_reserved_encodings() {
        let illegal = [
            0xffff_ffff,
            0x0000_607f,                        // unknown major opcode
            jalr(1, 2, 0) | 0b001 << 12,        // jalr with funct3 != 0
            b(0b010, 1, 2, 8),                  // branch funct3 010
            load(0b111, 1, 2, 0),               // ldu does not exist
            store(0b100, 1, 2, 0),              // sq is RV128
            op_imm(0b001, 1, 2, 0x400 | 3),     // slli with funct6 != 0
            op_imm(0b101, 1, 2, 0x200 | 3),     // srli/srai with bad funct6
            op(0b001, 0b010_0000, 1, 2, 3),     // "subl"
            op(0b000, 0b000_0011, 1, 2, 3),     // unassigned funct7
            op_32(0b001, 0b000_0001, 1, 2, 3),  // no mulhw
            op_imm_32(0b001, 1, 2, 32),         // slliw with shamt[5] set
            op_imm_32(0b101, 1, 2, 0x400 | 32), // sraiw with shamt[5] set
            op_32(0b010, 0, 1, 2, 3),           // no sltw
            0x0000_100f,                        // fence.i: Zifencei, not yet
            0x0020_0073,                        // uret-shaped SYSTEM encoding
            0x0000_1073,                        // csrrw: Zicsr, not yet
            amo(0b00010, 0b010, 1, 2, 3),       // lr.w with rs2 != 0
            amo(0b00000, 0b000, 1, 2, 3),       // amoadd.b: Zabha
            amo(0b00101, 0b011, 1, 2, 3),       // amocas.d: Zacas
            amo(0b00110, 0b010, 1, 2, 3),       // unassigned funct5
        ];
        for w in illegal {
            assert_eq!(decode(w), Err(DecodeError::Illegal(w)), "{w:#010x}");
        }
        assert_eq!(decode(0x0000_0001), Err(DecodeError::Compressed(0x0001))); // c.nop
        assert_eq!(decode(0x0000_0000), Err(DecodeError::Compressed(0))); // defined illegal
    }

    #[test]
    fn fence_ignores_reserved_fields() {
        // rd and rs1 set, which software must not do but hardware must tolerate.
        assert_eq!(
            decode(FENCE | 5 << 7 | 6 << 15),
            Ok(Inst::Fence { fm: 0, pred: 0xf, succ: 0xf })
        );
    }
}
