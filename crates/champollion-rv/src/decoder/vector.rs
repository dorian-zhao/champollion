//! The RVV subset needed by RVA23 userlands (glibc's string and memory routines and
//! compiler-vectorized loops): `vset*`, unit-stride/strided/whole-register/mask memory
//! operations, and the integer arithmetic, comparisons, moves, permutations, and
//! reductions those routines use.
//!
//! Everything outside that subset is illegal here, as RVA23 code is not expected to
//! reach it. `vmv1r.v` only handles `nf = 1`; segmented and indexed memory forms are
//! rejected.

use super::{
    DecodeError, Inst, VArithOp, VCmpOp, VMaskOp, VMemKind, VRedOp, VSel, VSetKind, VStoreKind,
    VWidth,
};

fn illegal(w: u32) -> DecodeError {
    DecodeError::Illegal(w)
}

/// `vsetvli`/`vsetivli`/`vsetvl` (funct3 = 111 in OP-V). The top two bits select the
/// form: 00 immediate, 10 the register form, 11 the `i` immediate form.
pub fn decode_vset(w: u32) -> Result<Inst, DecodeError> {
    let rd = ((w >> 7) & 31) as u8;
    let rs1 = ((w >> 15) & 31) as u8;
    match w >> 30 {
        // The 10-bit vtype sits in bits 29:20 for both immediate forms.
        0b00 => Ok(Inst::VSetvl {
            rd,
            rs1,
            rs2: ((w >> 20) & 0x3ff) as u16,
            kind: VSetKind::RegisterAvl,
        }),
        0b11 => Ok(Inst::VSetvl {
            rd,
            rs1,
            rs2: ((w >> 20) & 0x3ff) as u16,
            kind: VSetKind::ImmediateAvl,
        }),
        0b10 => Ok(Inst::VSetvl {
            rd,
            rs1,
            rs2: ((w >> 20) & 31) as u16,
            kind: VSetKind::RegisterVtype,
        }),
        _ => Err(illegal(w)),
    }
}

/// Vector loads and stores (opcodes 0x07 and 0x27). `nf`/`mop`/`lumop` classify the
/// access and funct3 selects the element width.
pub fn decode_mem(w: u32) -> Result<Inst, DecodeError> {
    let nf = ((w >> 29) & 7) as u8;
    let width = match (w >> 12) & 7 {
        0b000 => VWidth::E8,
        0b101 => VWidth::E16,
        0b110 => VWidth::E32,
        0b111 => VWidth::E64,
        _ => return Err(illegal(w)),
    };
    let mop = (w >> 26) & 3;
    let rs2 = ((w >> 20) & 31) as u8;
    let rs1 = ((w >> 15) & 31) as u8;
    let vd = ((w >> 7) & 31) as u8;
    let mask = (w >> 25) & 1 == 0;
    if w & 0x7f == 0x07 {
        let kind = match (mop, rs2) {
            (0b00, 0b00000) if nf == 0 => VMemKind::Unit,
            (0b00, 0b10000) if nf == 0 => VMemKind::FaultOnlyFirst,
            (0b00, 0b01000) if !mask && nf <= 2 => VMemKind::WholeRegister,
            (0b00, 0b01011) if !mask && nf == 0 && vd == 0 => VMemKind::Mask,
            (0b10, _) if nf == 0 => VMemKind::Stride,
            _ => return Err(illegal(w)),
        };
        // Only the strided form uses `rs2`; the classification fields read as zero.
        let rs2 = if kind == VMemKind::Stride { rs2 } else { 0 };
        Ok(Inst::VLoad { width, kind, vd, rs1, rs2, mask, nf })
    } else {
        let kind = match (mop, rs2) {
            (0b00, 0b00000) if nf == 0 => VStoreKind::Unit,
            (0b00, 0b01000) if !mask && nf <= 2 => VStoreKind::WholeRegister,
            (0b10, _) if nf == 0 => VStoreKind::Stride,
            _ => return Err(illegal(w)),
        };
        let rs2 = if kind == VStoreKind::Stride { rs2 } else { 0 };
        Ok(Inst::VStore { width, kind, vs3: vd, rs1, rs2, mask, nf })
    }
}

/// The OP-V integer operations (funct3 != 111). funct3 selects the operand kind
/// (000 vector, 011 immediate, 100 scalar, 010 mask-input, 110 mask-input scalar) and
/// funct6 the operation. Mask/merge bits follow the spec: `vm = 0` in the encoding
/// means "masked by v0", which is the inverted `mask` field here.
pub fn decode_op(w: u32) -> Result<Inst, DecodeError> {
    let f6 = (w >> 26) & 0x3f;
    let f3 = (w >> 12) & 7;
    let vm = (w >> 25) & 1;
    let vs2 = ((w >> 20) & 31) as u8;
    let vs1 = ((w >> 15) & 31) as u8;
    let vd = ((w >> 7) & 31) as u8;
    let mask = vm == 0;
    // Arithmetic table shared by the .vv/.vi/.vx kinds; the immediate forms sign-extend
    // a 5-bit value (spec: simm5), and `vsll`/`vxor`/... take it raw.
    let arith = |op: VArithOp, src: VSel| Ok(Inst::VArith { op, vd, vs2, src, mask });
    let cmp = |op: VCmpOp, src: VSel| Ok(Inst::VCmp { op, vd, vs2, src, mask });
    match f3 {
        0b000 => {
            let v = VSel::V(vs1);
            match f6 {
                0b000000 => arith(VArithOp::Add, v),
                0b000001 => arith(VArithOp::Andn, v),
                0b001001 => arith(VArithOp::And, v),
                0b001010 => arith(VArithOp::Or, v),
                0b001011 => arith(VArithOp::Xor, v),
                0b001100 => arith(VArithOp::Gather, v),
                0b100101 => arith(VArithOp::Sll, v),
                0b011000 => cmp(VCmpOp::Eq, v),
                0b011001 => cmp(VCmpOp::Ne, v),
                0b011100 => cmp(VCmpOp::LeU, v),
                0b011110 => cmp(VCmpOp::GtU, v),
                // vmerge (masked by v0) and, with vs2 = 0 and vm = 1, vmv.v.v.
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: v }),
                0b010111 if vm == 0 => Ok(Inst::Vmerge { vd, vs1, vs2 }),
                _ => Err(illegal(w)),
            }
        }
        0b011 => {
            let i = VSel::I(vs1);
            match f6 {
                0b000000 => arith(VArithOp::Add, i),
                0b000011 => arith(VArithOp::Rsub, i),
                0b001001 => arith(VArithOp::And, i),
                0b001010 => arith(VArithOp::Or, i),
                0b001011 => arith(VArithOp::Xor, i),
                0b001100 => arith(VArithOp::Gather, i),
                0b001110 => arith(VArithOp::SlideUp, i),
                0b001111 => arith(VArithOp::SlideDown, i),
                0b100101 => arith(VArithOp::Sll, i),
                0b101100 => Ok(Inst::Vnsrl { vd, vs2, imm: vs1 }),
                0b011000 => cmp(VCmpOp::Eq, i),
                0b011001 => cmp(VCmpOp::Ne, i),
                0b011100 => cmp(VCmpOp::LeU, i),
                0b011110 => cmp(VCmpOp::GtU, i),
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: i }),
                0b100111 if vm == 1 && vs1 == 0 => Ok(Inst::VmvNr { vd, vs2 }),
                _ => Err(illegal(w)),
            }
        }
        0b100 => {
            let x = VSel::X(vs1);
            match f6 {
                0b000000 => arith(VArithOp::Add, x),
                0b000011 => arith(VArithOp::Rsub, x),
                0b001001 => arith(VArithOp::And, x),
                0b001010 => arith(VArithOp::Or, x),
                0b001011 => arith(VArithOp::Xor, x),
                0b001110 => arith(VArithOp::SlideUp, x),
                0b001111 => arith(VArithOp::SlideDown, x),
                0b100101 => arith(VArithOp::Sll, x),
                0b011000 => cmp(VCmpOp::Eq, x),
                0b011001 => cmp(VCmpOp::Ne, x),
                0b011100 => cmp(VCmpOp::LeU, x),
                0b011110 => cmp(VCmpOp::GtU, x),
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: x }),
                _ => Err(illegal(w)),
            }
        }
        0b010 => match f6 {
            0b000001 => Ok(Inst::VRed { op: VRedOp::And, vd, vs1, vs2 }),
            0b000010 => Ok(Inst::VRed { op: VRedOp::Or, vd, vs1, vs2 }),
            0b011010 => Ok(Inst::VMask { op: VMaskOp::Or, vd, vs1, vs2 }),
            // vsext/vzext share funct6 010010: bit 15 selects sign/zero extension and
            // bits 17:16 the source width (01 = SEW/8, 10 = SEW/4, 11 = SEW/2).
            0b010010 if (w >> 16) & 3 != 0 => Ok(Inst::VExt {
                sext: (w >> 15) & 1 != 0,
                narrow: 8 >> (((w >> 16) & 3) - 1),
                vd,
                vs2,
            }),
            // vmadd.vv / vmacc.vv.
            0b101001 => Ok(Inst::VMacc { add: false, vd, vs2, src: VSel::V(vs1) }),
            0b101101 => Ok(Inst::VMacc { add: true, vd, vs2, src: VSel::V(vs1) }),
            // The vmv/cpop/vfirst family shares funct6 010000; bits 19:15 pick one.
            0b010000 => match vs1 {
                0b00000 => Ok(Inst::VmvXs { rd: vd, vs2 }),
                0b10000 => Ok(Inst::Vcpop { rd: vd, vs2 }),
                0b10001 => Ok(Inst::Vfirst { rd: vd, vs2 }),
                _ => Err(illegal(w)),
            },
            0b010100 if vs1 == 0b10001 => Ok(Inst::Vid { vd }),
            _ => Err(illegal(w)),
        },
        0b110 => match f6 {
            0b100101 => arith(VArithOp::Mul, VSel::X(vs1)),
            0b010000 => Ok(Inst::VmvSx { vd, rs1: vs1 }),
            0b001111 => arith(VArithOp::Slide1Down, VSel::X(vs1)),
            0b101001 => Ok(Inst::VMacc { add: false, vd, vs2, src: VSel::X(vs1) }),
            0b101101 => Ok(Inst::VMacc { add: true, vd, vs2, src: VSel::X(vs1) }),
            _ => Err(illegal(w)),
        },
        _ => Err(illegal(w)),
    }
}
