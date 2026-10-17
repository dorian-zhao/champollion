//! The RVV subset needed by RVA23 userlands (glibc's string and memory routines and
//! compiler-vectorized loops): `vset*`, unit-stride/strided/whole-register/mask memory
//! operations, and the integer arithmetic, comparisons, moves, permutations, and
//! reductions those routines use.
//!
//! Everything outside that subset is illegal here; segmented forms are rejected.
//! Zvbb's single-source bit operations and the Zvbb shifts decode too, since Ubuntu
//! 26.04's baseline (RVA23U64) compiles them into libc without a runtime check.

use super::{
    DecodeError, Inst, VArithOp, VCmpOp, VFArithOp, VFCmpOp, VFMaOp, VFRedOp, VFUnaryOp, VMaskOp,
    VMemKind, VRedOp, VSel, VSetKind, VStoreKind, VUnaryOp, VWidth,
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
        // Indexed-unordered loads are their own instruction: the width selects the
        // *index* element size, the loaded data is SEW wide.
        if mop == 0b01 && nf == 0 {
            return Ok(Inst::VIndexLoad { width, mask, vd, rs1, vs2: rs2 });
        }
        // Segments: `nf` counts fields beyond the first. The unit form keeps the
        // packed layout, the strided form takes the run-to-run distance from rs2.
        if nf > 0 {
            // The whole-register forms also set `nf`, so only the two segment layouts
            // are claimed here and anything else falls through below.
            match (mop, rs2) {
                (0b00, 0b00000) => {
                    return Ok(Inst::VSegLoad {
                        width,
                        stride: false,
                        mask,
                        fields: nf + 1,
                        vd,
                        rs1,
                        rs2,
                    });
                }
                (0b10, _) => {
                    return Ok(Inst::VSegLoad {
                        width,
                        stride: true,
                        mask,
                        fields: nf + 1,
                        vd,
                        rs1,
                        rs2,
                    });
                }
                _ => {}
            }
        }
        let kind = match (mop, rs2) {
            (0b00, 0b00000) if nf == 0 => VMemKind::Unit,
            (0b00, 0b10000) if nf == 0 => VMemKind::FaultOnlyFirst,
            (0b00, 0b01000) if !mask => VMemKind::WholeRegister,
            (0b00, 0b01011) if !mask && nf == 0 => VMemKind::Mask,
            (0b10, _) if nf == 0 => VMemKind::Stride,
            _ => return Err(illegal(w)),
        };
        // Only the strided form uses `rs2`; the classification fields read as zero.
        let rs2 = if kind == VMemKind::Stride { rs2 } else { 0 };
        Ok(Inst::VLoad { width, kind, vd, rs1, rs2, mask, nf })
    } else {
        // Indexed-unordered stores (`vsuxeiN.v`), the mirror of the load above.
        if mop == 0b01 && nf == 0 {
            return Ok(Inst::VIndexStore { width, mask, vs3: vd, rs1, vs2: rs2 });
        }
        if nf > 0 {
            match (mop, rs2) {
                (0b00, 0b00000) => {
                    return Ok(Inst::VSegStore {
                        width,
                        stride: false,
                        mask,
                        fields: nf + 1,
                        vs3: vd,
                        rs1,
                        rs2,
                    });
                }
                (0b10, _) => {
                    return Ok(Inst::VSegStore {
                        width,
                        stride: true,
                        mask,
                        fields: nf + 1,
                        vs3: vd,
                        rs1,
                        rs2,
                    });
                }
                _ => {}
            }
        }
        let kind = match (mop, rs2) {
            (0b00, 0b00000) if nf == 0 => VStoreKind::Unit,
            (0b00, 0b01000) if !mask => VStoreKind::WholeRegister,
            (0b00, 0b01011) if !mask && nf == 0 => VStoreKind::Mask,
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
                0b000010 => arith(VArithOp::Sub, v),
                0b001001 => arith(VArithOp::And, v),
                0b001010 => arith(VArithOp::Or, v),
                0b001011 => arith(VArithOp::Xor, v),
                0b001100 => arith(VArithOp::Gather, v),
                0b010100 => arith(VArithOp::Ror, v),
                0b010101 => arith(VArithOp::Rol, v),
                0b100101 => arith(VArithOp::Sll, v),
                0b101000 => arith(VArithOp::Srl, v),
                0b101001 => arith(VArithOp::Sra, v),
                0b101100 => Ok(Inst::VNarrow { sra: false, mask, vd, vs2, src: v }),
                0b101101 => Ok(Inst::VNarrow { sra: true, mask, vd, vs2, src: v }),
                0b011000 => cmp(VCmpOp::Eq, v),
                0b011001 => cmp(VCmpOp::Ne, v),
                0b011010 => cmp(VCmpOp::LtU, v),
                0b011011 => cmp(VCmpOp::Lt, v),
                0b011100 => cmp(VCmpOp::LeU, v),
                0b011101 => cmp(VCmpOp::Le, v),
                0b011110 => cmp(VCmpOp::GtU, v),
                0b011111 => cmp(VCmpOp::Gt, v),
                // vmerge (masked by v0) and, with vs2 = 0 and vm = 1, vmv.v.v.
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: v }),
                0b010111 if vm == 0 => Ok(Inst::Vmerge { vd, vs2, src: v }),
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
                0b101000 => arith(VArithOp::Srl, i),
                0b101001 => arith(VArithOp::Sra, i),
                0b101100 => Ok(Inst::VNarrow { sra: false, mask, vd, vs2, src: i }),
                0b101101 => Ok(Inst::VNarrow { sra: true, mask, vd, vs2, src: i }),
                0b011000 => cmp(VCmpOp::Eq, i),
                0b011001 => cmp(VCmpOp::Ne, i),
                0b011100 => cmp(VCmpOp::LeU, i),
                0b011101 => cmp(VCmpOp::Le, i),
                0b011110 => cmp(VCmpOp::GtU, i),
                0b011111 => cmp(VCmpOp::Gt, i),
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: i }),
                0b010111 if vm == 0 => Ok(Inst::Vmerge { vd, vs2, src: i }),
                // vmv<n>r.v moves 1, 2, 4, or 8 registers: the count code sits in
                // the vs1 field (0, 1, 3, 7), though the move ignores vs1 otherwise.
                0b100111 if vm == 1 && matches!(vs1, 0 | 1 | 3 | 7) => {
                    Ok(Inst::VmvNr { vd, vs2, nf: vs1 })
                }
                // Zvbb's rotate-immediate splits its 6-bit amount around the vm bit:
                // bit 5 rides in bit 26, the rest in the `vs1` field.
                0b010100 | 0b010101 if vm == 1 => Ok(Inst::VArith {
                    op: VArithOp::Ror,
                    vd,
                    vs2,
                    src: VSel::I(((w >> 26 & 1) as u8) << 5 | vs1),
                    mask: false,
                }),
                _ => Err(illegal(w)),
            }
        }
        0b100 => {
            let x = VSel::X(vs1);
            match f6 {
                0b000000 => arith(VArithOp::Add, x),
                0b000011 => arith(VArithOp::Rsub, x),
                0b000010 => arith(VArithOp::Sub, x),
                0b000001 => arith(VArithOp::Andn, x),
                0b001001 => arith(VArithOp::And, x),
                0b001010 => arith(VArithOp::Or, x),
                0b001011 => arith(VArithOp::Xor, x),
                0b001110 => arith(VArithOp::SlideUp, x),
                0b001111 => arith(VArithOp::SlideDown, x),
                0b010100 => arith(VArithOp::Ror, x),
                0b010101 => arith(VArithOp::Rol, x),
                0b100101 => arith(VArithOp::Sll, x),
                0b101000 => arith(VArithOp::Srl, x),
                0b101001 => arith(VArithOp::Sra, x),
                0b101100 => Ok(Inst::VNarrow { sra: false, mask, vd, vs2, src: x }),
                0b101101 => Ok(Inst::VNarrow { sra: true, mask, vd, vs2, src: x }),
                0b011000 => cmp(VCmpOp::Eq, x),
                0b011001 => cmp(VCmpOp::Ne, x),
                0b011010 => cmp(VCmpOp::LtU, x),
                0b011011 => cmp(VCmpOp::Lt, x),
                0b011100 => cmp(VCmpOp::LeU, x),
                0b011101 => cmp(VCmpOp::Le, x),
                0b011110 => cmp(VCmpOp::GtU, x),
                0b011111 => cmp(VCmpOp::Gt, x),
                0b010111 if vm == 1 && vs2 == 0 => Ok(Inst::Vmv { vd, src: x }),
                0b010111 if vm == 0 => Ok(Inst::Vmerge { vd, vs2, src: x }),
                _ => Err(illegal(w)),
            }
        }
        0b010 => match f6 {
            0b000000 => Ok(Inst::VRed { op: VRedOp::Sum, vd, vs1, vs2 }),
            0b000001 => Ok(Inst::VRed { op: VRedOp::And, vd, vs1, vs2 }),
            0b000010 => Ok(Inst::VRed { op: VRedOp::Or, vd, vs1, vs2 }),
            0b000011 => Ok(Inst::VRed { op: VRedOp::Xor, vd, vs1, vs2 }),
            // The mask-register logical ops share funct6 011000..011111. They are
            // always unmasked, so vm = 0 is reserved.
            0b011000 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Andn, vd, vs1, vs2 }),
            0b011001 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::And, vd, vs1, vs2 }),
            0b011010 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Or, vd, vs1, vs2 }),
            0b011011 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Xor, vd, vs1, vs2 }),
            0b011100 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Orn, vd, vs1, vs2 }),
            0b011101 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Nand, vd, vs1, vs2 }),
            0b011110 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Nor, vd, vs1, vs2 }),
            0b011111 if vm == 1 => Ok(Inst::VMask { op: VMaskOp::Xnor, vd, vs1, vs2 }),
            // Zvbb's single-source ops and vsext/vzext share funct6 010010; the vs1
            // field (which these have no use for) selects one. The extension ops keep
            // the low six codes, the bit operations start at 01000.
            0b010010 if (w >> 15) & 0x1f >= 0b01000 => {
                let op = match vs1 {
                    0b01000 => VUnaryOp::Brev8,
                    0b01001 => VUnaryOp::Rev8,
                    0b01010 => VUnaryOp::Brev,
                    0b01100 => VUnaryOp::Clz,
                    0b01101 => VUnaryOp::Ctz,
                    0b01110 => VUnaryOp::Cpop,
                    _ => return Err(illegal(w)),
                };
                Ok(Inst::VUnary { op, mask, vd, vs2 })
            }
            // vsext/vzext: bit 15 selects sign/zero extension and bits 17:16 the source
            // width (01 = SEW/8, 10 = SEW/4, 11 = SEW/2).
            0b010010 if (w >> 15) & 0x1f >= 0b00010 => Ok(Inst::VExt {
                sext: (w >> 15) & 1 != 0,
                narrow: 8 >> (((w >> 16) & 3) - 1),
                vd,
                vs2,
            }),
            // vmadd.vv / vmacc.vv.
            0b101001 => Ok(Inst::VMacc { add: false, vd, vs2, src: VSel::V(vs1) }),
            0b101101 => Ok(Inst::VMacc { add: true, vd, vs2, src: VSel::V(vs1) }),
            // vmul.vv / vwaddu.wv.
            0b100101 => arith(VArithOp::Mul, VSel::V(vs1)),
            0b110100 => Ok(Inst::VWiden { mask, vd, vs2, src: VSel::V(vs1) }),
            // vcompress.vm is never masked by v0 (it consumes v0 as its selector).
            0b010111 if vm == 1 => Ok(Inst::VCompress { vd, vs2, vs1 }),
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
            0b110100 => Ok(Inst::VWiden { mask, vd, vs2, src: VSel::X(vs1) }),
            0b010000 => Ok(Inst::VmvSx { vd, rs1: vs1 }),
            0b001111 => arith(VArithOp::Slide1Down, VSel::X(vs1)),
            0b101001 => Ok(Inst::VMacc { add: false, vd, vs2, src: VSel::X(vs1) }),
            0b101101 => Ok(Inst::VMacc { add: true, vd, vs2, src: VSel::X(vs1) }),
            _ => Err(illegal(w)),
        },
        0b001 => decode_fp(w, f6, vd, vs2, vs1, mask, false),
        0b101 => decode_fp(w, f6, vd, vs2, vs1, mask, true),
        _ => Err(illegal(w)),
    }
}

/// The OPFVV (`funct3` 001) and OPFVF (101) float groups. They share a `funct6` table;
/// `scalar` selects whether the third operand is an `f` register (`...vf`) or a vector
/// register (`...vv`), and a few encodings exist in only one of the two groups.
fn decode_fp(
    w: u32,
    f6: u32,
    vd: u8,
    vs2: u8,
    vs1: u8,
    mask: bool,
    scalar: bool,
) -> Result<Inst, DecodeError> {
    let src = if scalar { VSel::F(vs1) } else { VSel::V(vs1) };
    let arith = |op| Ok(Inst::VFArith { op, vd, vs2, src, mask });
    let fma = |op| Ok(Inst::VFMa { op, vd, vs2, src, mask });
    let cmp = |op| Ok(Inst::VFCmp { op, vd, vs2, src, mask });
    match f6 {
        0b000000 => arith(VFArithOp::Add),
        0b000010 => arith(VFArithOp::Sub),
        // vfrsub exists only as .vf (like vfrdiv).
        0b100111 if scalar => arith(VFArithOp::Rsub),
        0b000100 => arith(VFArithOp::Min),
        0b000110 => arith(VFArithOp::Max),
        0b001000 => arith(VFArithOp::Sgnj),
        0b001001 => arith(VFArithOp::Sgnjn),
        0b001010 => arith(VFArithOp::Sgnjx),
        0b100000 => arith(VFArithOp::Div),
        0b100100 => arith(VFArithOp::Mul),
        0b101000 => fma(VFMaOp::Madd),
        0b101001 => fma(VFMaOp::Nmadd),
        0b101010 => fma(VFMaOp::Msub),
        0b101011 => fma(VFMaOp::Nmsub),
        0b101100 => fma(VFMaOp::Macc),
        0b101101 => fma(VFMaOp::Nmacc),
        0b101110 => fma(VFMaOp::Msac),
        0b101111 => fma(VFMaOp::Nmsac),
        0b011000 => cmp(VFCmpOp::Eq),
        0b011001 => cmp(VFCmpOp::Le),
        0b011011 => cmp(VFCmpOp::Lt),
        0b011100 => cmp(VFCmpOp::Ne),
        0b011101 => cmp(VFCmpOp::Gt),
        0b011111 => cmp(VFCmpOp::Ge),
        // Reductions are OPFVV-only and always unmasked.
        0b000001 if !scalar && !mask => Ok(Inst::VFRed { op: VFRedOp::Usum, vd, vs1, vs2 }),
        0b000011 if !scalar && !mask => Ok(Inst::VFRed { op: VFRedOp::Osum, vd, vs1, vs2 }),
        0b000101 if !scalar && !mask => Ok(Inst::VFRed { op: VFRedOp::Min, vd, vs1, vs2 }),
        0b000111 if !scalar && !mask => Ok(Inst::VFRed { op: VFRedOp::Max, vd, vs1, vs2 }),
        // vfsqrt.v / vfclass.v (OPFVV, selector in the vs1 field).
        0b010011 if !scalar && vs1 == 0x00 => {
            Ok(Inst::VFUnary { op: VFUnaryOp::Sqrt, vd, vs2, mask })
        }
        0b010011 if !scalar && vs1 == 0x10 => {
            Ok(Inst::VFUnary { op: VFUnaryOp::Class, vd, vs2, mask })
        }
        // vfcvt.* (OPFVV, selector in the vs1 field).
        0b010010 if !scalar => match vs1 {
            0x00 => Ok(Inst::VFUnary { op: VFUnaryOp::XuF, vd, vs2, mask }),
            0x01 => Ok(Inst::VFUnary { op: VFUnaryOp::XF, vd, vs2, mask }),
            0x02 => Ok(Inst::VFUnary { op: VFUnaryOp::FXu, vd, vs2, mask }),
            0x03 => Ok(Inst::VFUnary { op: VFUnaryOp::FX, vd, vs2, mask }),
            0x06 => Ok(Inst::VFUnary { op: VFUnaryOp::RtzXuF, vd, vs2, mask }),
            0x07 => Ok(Inst::VFUnary { op: VFUnaryOp::RtzXF, vd, vs2, mask }),
            0x0c => Ok(Inst::VFHalfConvert { widening: true, vd, vs2, mask }),
            0x14 => Ok(Inst::VFHalfConvert { widening: false, vd, vs2, mask }),
            _ => Err(illegal(w)),
        },
        // vfmv.f.s: element 0 into f[rd].
        0b010000 if !scalar && vs1 == 0 => Ok(Inst::VFmvFS { rd: vd, vs2 }),
        _ if scalar => match f6 {
            // vfmv.s.f: f[rs1] into element 0.
            0b010000 if vs2 == 0 => Ok(Inst::VFmvSF { vd, rs1: vs1 }),
            // vfmv.v.f (vm = 1, vs2 = 0) and vfmerge.vfm (vm = 0) share funct6.
            0b010111 if !mask && vs2 == 0 => Ok(Inst::Vmv { vd, src: VSel::F(vs1) }),
            0b010111 if mask => Ok(Inst::Vmerge { vd, vs2, src: VSel::F(vs1) }),
            0b001110 => Ok(Inst::VFSlide1 { down: false, vd, vs2, rs1: vs1, mask }),
            0b001111 => Ok(Inst::VFSlide1 { down: true, vd, vs2, rs1: vs1, mask }),
            _ => Err(illegal(w)),
        },
        _ => Err(illegal(w)),
    }
}
