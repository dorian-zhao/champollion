//! Execution of the RVV subset (see `decoder/vector.rs`): `vtype`/`vl` handling, lane
//! and mask access, and the per-element operations. Memory access lives in the
//! interpreter so faults flow through the usual trap paths.
//!
//! Policies are implemented "undisturbed": elements outside the active set (beyond
//! `vl`, or masked off) keep their previous value. That is a legal implementation of
//! all four tail/mask policy combinations, and `vstart` stays zero because a
//! mid-instruction fault is reported at instruction granularity (state may be
//! partially written, which the subset's uses never observe: glibc's fault-only-first
//! scanning is handled explicitly and does not trap).

use crate::cpu::CpuState;
use crate::decoder::{VArithOp, VCmpOp, VRedOp, VSel, VUnaryOp};
use crate::memory::{MemFault, Memory};

/// VLEN in bytes. The minimum the vector spec allows, and enough for the RVA23
/// userlands: every `vset*` in their loaders runs at m1 or a fractional LMUL.
pub const VLEN_BYTES: u64 = 16;

/// `vl` for one `vset*`: validate `vtype`, set it with `vl`, and return the new `vl`.
pub fn vset(cpu: &mut CpuState, avl: u64, bits: u64) -> u64 {
    let (vtype, vl) = vset_compute(avl, bits);
    cpu.vtype = vtype;
    cpu.vl = vl;
    vl
}

/// `vset`'s arithmetic without the state write: the `vtype` and `vl` that setting `avl`
/// with this raw `vtype` produces. (The JIT translates its statically known `vsetivli`
/// through this, so both engines write bit-identical state.)
pub fn vset_compute(avl: u64, bits: u64) -> (u64, u64) {
    if bits & 7 == 4 {
        return (bits | 1 << 63, 0); // vlmul 100 is reserved
    }
    if (bits >> 3) & 7 > 3 {
        return (bits | 1 << 63, 0); // vsew 4..7 are reserved
    }
    (bits, avl.min(vlmax_of(bits)))
}

/// `vlmax` for a validated `vtype`.
fn vlmax_of(bits: u64) -> u64 {
    let sew_b = 1u64 << ((bits >> 3) & 7);
    let (num, den) = match bits & 7 {
        0 => (1, 1),
        1 => (2, 1),
        2 => (4, 1),
        3 => (8, 1),
        5 => (1, 8),
        6 => (1, 4),
        _ => (1, 2),
    };
    VLEN_BYTES * num / (sew_b * den)
}

/// Element width in bytes for the current `vtype`.
#[inline]
pub fn sew_bytes(cpu: &CpuState) -> u64 {
    1u64 << ((cpu.vtype >> 3) & 7)
}

/// Registers per field group at the current LMUL: 1/2/4/8 for m1..m8, and 1 for the
/// fractional LMULs (a segment's fields each occupy the same fractional slice).
#[inline]
pub fn lmul_regs(cpu: &CpuState) -> u64 {
    match cpu.vtype & 7 {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 1,
    }
}

/// `vlmax` for the current `vtype` (0 after a reserved `vset*`).
#[inline]
pub fn vlmax(cpu: &CpuState) -> u64 {
    if cpu.vtype >> 63 != 0 { 0 } else { vlmax_of(cpu.vtype) }
}

/// Read element `i` (of `sew_b` bytes) of vector register `reg`, zero-extended.
#[inline]
pub fn get_elem(cpu: &CpuState, reg: u8, i: u64, sew_b: u64) -> u64 {
    let mut off = i * sew_b;
    let mut r = reg as usize & 31;
    while off >= VLEN_BYTES {
        off -= VLEN_BYTES;
        r = (r + 1) & 31;
    }
    let b = &cpu.v[r][off as usize..(off + sew_b) as usize];
    let mut v = 0u64;
    for (k, &byte) in b.iter().enumerate() {
        v |= u64::from(byte) << (8 * k as u32);
    }
    v
}

/// Write the low `sew_b` bytes of `value` to element `i` of register `reg`.
#[inline]
pub fn set_elem(cpu: &mut CpuState, reg: u8, i: u64, sew_b: u64, value: u64) {
    let mut off = i * sew_b;
    let mut r = reg as usize & 31;
    while off >= VLEN_BYTES {
        off -= VLEN_BYTES;
        r = (r + 1) & 31;
    }
    let b = &mut cpu.v[r][off as usize..(off + sew_b) as usize];
    for (k, byte) in b.iter_mut().enumerate() {
        *byte = (value >> (8 * k as u32)) as u8;
    }
}

/// Bit `i` of a mask register.
#[inline]
pub fn mask_get(cpu: &CpuState, reg: u8, i: u64) -> bool {
    cpu.v[reg as usize & 31][(i / 8) as usize] >> (i % 8) & 1 != 0
}

/// Set bit `i` of a mask register.
#[inline]
pub fn mask_set(cpu: &mut CpuState, reg: u8, i: u64, v: bool) {
    let byte = &mut cpu.v[reg as usize & 31][(i / 8) as usize];
    let bit = 1u8 << (i % 8);
    if v {
        *byte |= bit;
    } else {
        *byte &= !bit;
    }
}

/// Sign-extended 5-bit immediate (`simm5` forms).
#[inline]
pub fn sext5(v: u8) -> u64 {
    ((v as i8) << 3 >> 3) as i64 as u64
}

/// Sign-extend the low `bytes` bytes of `raw`.
#[inline]
pub fn sign_extend(raw: u64, bytes: u64) -> u64 {
    let shift = 64 - 8 * bytes;
    ((raw << shift) as i64 >> shift) as u64
}

/// The value of the second source for element `i`, with immediate signedness applied
/// per operation family (arithmetic `.vi` forms sign-extend, shift/gather/permute forms
/// take the immediate as unsigned).
fn src_value(cpu: &CpuState, src: VSel, i: u64, sew_b: u64, imm_unsigned: bool) -> u64 {
    match src {
        VSel::V(reg) => get_elem(cpu, reg, i, sew_b),
        VSel::X(reg) => cpu.x(reg),
        VSel::I(imm) => {
            if imm_unsigned {
                u64::from(imm)
            } else {
                sext5(imm)
            }
        }
    }
}

/// Element `i` of a source register, through a snapshot when one was taken.
#[inline]
fn lens(
    cpu: &CpuState,
    table: &Option<[u8; 8 * VLEN_BYTES as usize]>,
    reg: u8,
    i: u64,
    sew_b: u64,
) -> u64 {
    match table {
        Some(bytes) => snap_get(bytes, i, sew_b),
        None => get_elem(cpu, reg, i, sew_b),
    }
}

/// Element `i` of a snapshot of one register group. Snapshot readers only ever look
/// at elements within the group (indices the active set can produce are < vlmax).
#[inline]
fn snap_get(table: &[u8; 8 * VLEN_BYTES as usize], i: u64, sew_b: u64) -> u64 {
    let off = (i * sew_b) as usize;
    let mut v = 0u64;
    for (k, &byte) in table[off..off + sew_b as usize].iter().enumerate() {
        v |= u64::from(byte) << (8 * k as u32);
    }
    v
}

/// A copy of the register group starting at `reg` (one register at m1 and below).
/// The slide and gather ops can read indices up to VLMAX while also writing their
/// destination in place, so they work off this snapshot.
fn snap_group(cpu: &CpuState, reg: u8) -> [u8; 8 * VLEN_BYTES as usize] {
    let mut bytes = [0u8; 8 * VLEN_BYTES as usize];
    for k in 0..lmul_regs(cpu) as usize {
        bytes[k * VLEN_BYTES as usize..][..VLEN_BYTES as usize]
            .copy_from_slice(&cpu.v[(reg as usize + k) & 31]);
    }
    bytes
}

/// Per-element arithmetic, truncated to the element width.
#[inline]
fn binop(op: VArithOp, a: u64, b: u64, sew_b: u64) -> u64 {
    let mask = if sew_b == 8 { u64::MAX } else { (1u64 << (8 * sew_b)) - 1 };
    match op {
        VArithOp::Add => a.wrapping_add(b) & mask,
        VArithOp::Sub => a.wrapping_sub(b) & mask,
        VArithOp::Rsub => b.wrapping_sub(a) & mask,
        VArithOp::And => a & b,
        VArithOp::Or => a | b,
        VArithOp::Xor => a ^ b,
        VArithOp::Andn => (a & !b) & mask,
        VArithOp::Sll => (a << (b % (8 * sew_b))) & mask,
        VArithOp::Srl => a >> (b % (8 * sew_b)),
        VArithOp::Sra => ((sign_extend(a & mask, sew_b) as i64) >> (b % (8 * sew_b))) as u64 & mask,
        VArithOp::Ror => {
            let bits = 8 * sew_b;
            let sh = b % bits;
            ((a >> sh) | (a << ((bits - sh) % bits))) & mask
        }
        VArithOp::Mul => a.wrapping_mul(b) & mask,
        // The remaining variants are handled element-wise with their own control flow.
        _ => unreachable!("controlled ops are not binary"),
    }
}

/// Execute one arithmetic/permute instruction over the active elements.
pub fn exec_arith(cpu: &mut CpuState, op: VArithOp, vd: u8, vs2: u8, src: VSel, mask: bool) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let vlmax = vlmax(cpu);
    // Slide and gather ops can overlap their destination with a source; take copies so
    // the reads all see the original elements.
    let snapshot = matches!(
        op,
        VArithOp::SlideUp | VArithOp::SlideDown | VArithOp::Slide1Down | VArithOp::Gather
    );
    let old_vs2 = if snapshot { Some(snap_group(cpu, vs2)) } else { None };
    let old_src = match (snapshot, src) {
        (true, VSel::V(reg)) => Some(snap_group(cpu, reg)),
        _ => None,
    };
    for i in 0..vl {
        if mask && !mask_get(cpu, 0, i) {
            continue; // undisturbed
        }
        let a = lens(cpu, &old_vs2, vs2, i, sew_b);
        let value = match op {
            VArithOp::SlideUp => {
                let off = u64::from(match src {
                    VSel::I(v) => v,
                    VSel::X(r) => cpu.x(r) as u8,
                    VSel::V(_) => unreachable!("vslideup has no vector form"),
                });
                if i < off {
                    continue; // undisturbed
                }
                lens(cpu, &old_vs2, vs2, i - off, sew_b)
            }
            VArithOp::SlideDown => {
                let off = u64::from(match src {
                    VSel::I(v) => v,
                    VSel::X(r) => cpu.x(r) as u8,
                    VSel::V(_) => unreachable!("vslidedown has no vector form"),
                });
                // The source read is bounded by VLMAX, not vl (spec: "0 <= i+OFFSET <
                // VLMAX: src[i] = vs2[i+OFFSET]"). Compilers lean on this: clang's
                // 128-bit extract is vsetivli vl=1 followed by vslidedown.vx 1, which
                // reads element 1 out of the undisturbed tail.
                if i + off < vlmax { lens(cpu, &old_vs2, vs2, i + off, sew_b) } else { 0 }
            }
            VArithOp::Slide1Down => {
                if i + 1 < vl {
                    lens(cpu, &old_vs2, vs2, i + 1, sew_b)
                } else if i + 1 == vl {
                    cpu.x(match src {
                        VSel::X(r) => r,
                        _ => unreachable!("vslide1down is vx-only"),
                    })
                } else {
                    0
                }
            }
            VArithOp::Gather => {
                let index = match src {
                    VSel::V(reg) => lens(cpu, &old_src, reg, i, sew_b),
                    VSel::I(imm) => u64::from(imm),
                    VSel::X(_) => unreachable!("vrgather has no vx form"),
                };
                if index < vlmax { lens(cpu, &old_vs2, vs2, index, sew_b) } else { 0 }
            }
            _ => {
                let unsigned = matches!(
                    op,
                    VArithOp::Sll
                        | VArithOp::Srl
                        | VArithOp::Sra
                        | VArithOp::Ror
                        | VArithOp::Gather
                        | VArithOp::SlideUp
                        | VArithOp::SlideDown
                );
                binop(op, a, src_value(cpu, src, i, sew_b, unsigned), sew_b)
            }
        };
        set_elem(cpu, vd, i, sew_b, value);
    }
}

/// Execute a mask-producing comparison: bits for masked-off elements are cleared.
pub fn exec_cmp(cpu: &mut CpuState, op: VCmpOp, vd: u8, vs2: u8, src: VSel, mask: bool) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        let b = src_value(cpu, src, i, sew_b, false);
        let a = get_elem(cpu, vs2, i, sew_b);
        let bit = !mask || mask_get(cpu, 0, i);
        let result = match op {
            VCmpOp::Eq => a == b,
            VCmpOp::Ne => a != b,
            VCmpOp::Lt => (sign_extend(a, sew_b) as i64) < (sign_extend(b, sew_b) as i64),
            VCmpOp::LtU => a < b,
            VCmpOp::Le => (sign_extend(a, sew_b) as i64) <= (sign_extend(b, sew_b) as i64),
            VCmpOp::LeU => a <= b,
            VCmpOp::Gt => (sign_extend(a, sew_b) as i64) > (sign_extend(b, sew_b) as i64),
            VCmpOp::GtU => a > b,
        };
        mask_set(cpu, vd, i, bit && result);
    }
}

/// `vmor.mm`.
pub fn exec_mask_or(cpu: &mut CpuState, vd: u8, vs1: u8, vs2: u8) {
    for i in 0..cpu.vl {
        mask_set(cpu, vd, i, mask_get(cpu, vs2, i) || mask_get(cpu, vs1, i));
    }
}

/// `vmv.v.*` / `vmv1r.v` / `vmv.s.x` / `vmerge.vvm`.
pub fn exec_mv(cpu: &mut CpuState, vd: u8, src: VSel) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        let v = src_value(cpu, src, i, sew_b, false);
        set_elem(cpu, vd, i, sew_b, v);
    }
}

pub fn exec_mv_nr(cpu: &mut CpuState, vd: u8, vs2: u8, nf: u8) {
    for k in 0..=nf {
        cpu.v[(vd as usize + k as usize) & 31] = cpu.v[(vs2 as usize + k as usize) & 31];
    }
}

pub fn exec_merge(cpu: &mut CpuState, vd: u8, vs2: u8, src: VSel) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        // Spec: vd[i] = v0.mask[i] ? src[i] : vs2[i] (the mask selects the source).
        let v = if mask_get(cpu, 0, i) {
            src_value(cpu, src, i, sew_b, false)
        } else {
            get_elem(cpu, vs2, i, sew_b)
        };
        set_elem(cpu, vd, i, sew_b, v);
    }
}

/// Single-width reductions (`vredsum`/`vredand`/`vredor`/`vredxor`): element 0 of `vd`
/// gets the SEW-wide result, seeded from element 0 of `vs1`. The spec is explicit that
/// "all operands and results ... have the same SEW width", so nothing wider than one
/// element may be written; the tail is left to the tail policy.
pub fn exec_red(cpu: &mut CpuState, op: VRedOp, vd: u8, vs1: u8, vs2: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let mut acc = get_elem(cpu, vs1, 0, sew_b);
    for i in 0..vl {
        let el = get_elem(cpu, vs2, i, sew_b);
        acc = match op {
            VRedOp::And => acc & el,
            VRedOp::Or => acc | el,
            VRedOp::Xor => acc ^ el,
            VRedOp::Sum => acc.wrapping_add(el),
        };
    }
    let keep = if sew_b >= 8 { u64::MAX } else { (1u64 << (8 * sew_b)) - 1 };
    set_elem(cpu, vd, 0, sew_b, acc & keep);
}

/// `vmv.x.s`: element 0 to a scalar register, sign-extended to XLEN (the spec's
/// widening on the way out of a reduction, which qemu-riscv64 confirms).
pub fn elem0(cpu: &CpuState, vs2: u8) -> u64 {
    let sew_b = sew_bytes(cpu);
    sign_extend(get_elem(cpu, vs2, 0, sew_b), sew_b)
}

/// `vmv.s.x`: scalar to element 0 (the rest is undisturbed).
pub fn set_elem0(cpu: &mut CpuState, vd: u8, value: u64) {
    let sew_b = sew_bytes(cpu);
    set_elem(cpu, vd, 0, sew_b, value);
}

/// `vcpop.m`: set mask bits below `vl`.
pub fn cpop(cpu: &CpuState, vs2: u8) -> u64 {
    (0..cpu.vl).filter(|&i| mask_get(cpu, vs2, i)).count() as u64
}

/// `vfirst.m`: index of the first set mask bit, or all ones.
pub fn vfirst(cpu: &CpuState, vs2: u8) -> u64 {
    (0..cpu.vl).find(|&i| mask_get(cpu, vs2, i)).unwrap_or(u64::MAX)
}

/// `vid.v`.
pub fn exec_vid(cpu: &mut CpuState, vd: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        set_elem(cpu, vd, i, sew_b, i);
    }
}

/// `vnsrl.{wi,wx,wv}`/`vnsra.*`: take 2×SEW-wide elements from `vs2`, shift, narrow to
/// SEW. The shift amount is masked to log2(2×SEW) bits, like every vector shift.
pub fn exec_narrow(cpu: &mut CpuState, sra: bool, mask: bool, vd: u8, vs2: u8, src: VSel) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let wide_b = sew_b * 2;
    for i in 0..vl {
        if mask && !mask_get(cpu, 0, i) {
            continue;
        }
        let sh = src_value(cpu, src, i, sew_b, true) % (8 * wide_b);
        let wide = get_elem(cpu, vs2, i, wide_b);
        let value =
            if sra { ((sign_extend(wide, wide_b) as i64) >> sh) as u64 } else { wide >> sh };
        set_elem(cpu, vd, i, sew_b, value);
    }
}

/// Zvbb's single-source bit operations.
pub fn exec_unary(cpu: &mut CpuState, op: VUnaryOp, mask: bool, vd: u8, vs2: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let bits = 8 * sew_b;
    for i in 0..vl {
        if mask && !mask_get(cpu, 0, i) {
            continue;
        }
        let e = get_elem(cpu, vs2, i, sew_b);
        let v = match op {
            VUnaryOp::Brev8 => {
                let mut out = 0u64;
                for k in 0..sew_b {
                    // Reverse within the byte, *then* widen: reversing after the
                    // widening would reverse all 64 bits instead.
                    out |= u64::from(((e >> (8 * k)) as u8).reverse_bits()) << (8 * k);
                }
                out
            }
            VUnaryOp::Rev8 => {
                let mut out = 0u64;
                for k in 0..sew_b {
                    out |= (e >> (8 * k) & 0xff) << (8 * (sew_b - 1 - k));
                }
                out
            }
            VUnaryOp::Brev => e.reverse_bits() >> (64 - bits),
            VUnaryOp::Clz => {
                if e == 0 {
                    bits
                } else {
                    u64::from(e.leading_zeros()) - (64 - bits)
                }
            }
            VUnaryOp::Ctz => {
                if e == 0 {
                    bits
                } else {
                    u64::from(e.trailing_zeros())
                }
            }
            VUnaryOp::Cpop => u64::from(e.count_ones()),
        };
        set_elem(cpu, vd, i, sew_b, v);
    }
}

/// `vcompress.vm`: pack the elements of `vs2` that the mask in `vs1` selects to the
/// front of `vd` (the spec is explicit that the mask operand is `vs1`, not v0).
pub fn exec_compress(cpu: &mut CpuState, vd: u8, vs2: u8, vs1: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let mut j = 0u64;
    for i in 0..vl {
        if mask_get(cpu, vs1, i) {
            let v = get_elem(cpu, vs2, i, sew_b);
            set_elem(cpu, vd, j, sew_b, v);
            j += 1;
        }
    }
}

/// `vwaddu.wv`/`vwaddu.wx`: `vs2` is read 2×SEW wide, the source is zero-extended from
/// SEW, and the sum is 2×SEW wide.
pub fn exec_widen(cpu: &mut CpuState, mask: bool, vd: u8, vs2: u8, src: VSel) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        if mask && !mask_get(cpu, 0, i) {
            continue;
        }
        let a = get_elem(cpu, vs2, i, sew_b * 2);
        let b = match src {
            VSel::V(reg) => get_elem(cpu, reg, i, sew_b),
            VSel::X(reg) => cpu.x(reg),
            VSel::I(_) => unreachable!("vwaddu has no immediate form"),
        };
        set_elem(cpu, vd, i, sew_b * 2, a.wrapping_add(b));
    }
}

/// Indexed load (`vluxeiN.v`): element `i` at `x[rs1] + zext(index[i])`; the index
/// width is the instruction's, the loaded data is SEW wide. Masked-off elements are
/// undisturbed and never accessed.
pub fn exec_index_load(
    cpu: &mut CpuState,
    mem: &impl Memory,
    width: u64,
    mask: bool,
    vd: u8,
    rs1: u8,
    vs2: u8,
) -> Result<(), MemFault> {
    let base = cpu.x(rs1);
    let sew_b = sew_bytes(cpu);
    for i in 0..cpu.vl {
        if mask && !mask_get(cpu, 0, i) {
            continue;
        }
        let off = get_elem(cpu, vs2, i, width);
        let addr = base.wrapping_add(off);
        let mut b = [0u8; 8];
        mem.read(addr, &mut b[..sew_b as usize])?;
        set_elem(cpu, vd, i, sew_b, u64::from_le_bytes(b));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::FlatMemory;

    /// A CPU with `e8`/`m1` (or another vtype) and a chosen `vl`.
    fn machine(vtype: u64, vl: u64) -> CpuState {
        let mut c = CpuState::new(0);
        c.vtype = vtype;
        c.vl = vl;
        c
    }

    fn elems(cpu: &CpuState, reg: u8, n: u64, sew_b: u64) -> Vec<u64> {
        (0..n).map(|i| get_elem(cpu, reg, i, sew_b)).collect()
    }

    fn load(cpu: &mut CpuState, reg: u8, values: &[u64], sew_b: u64) {
        for (i, v) in values.iter().enumerate() {
            set_elem(cpu, reg, i as u64, sew_b, *v);
        }
    }

    #[test]
    fn shifts_mask_the_amount_and_sign_extend() {
        let mut c = machine(0, 4); // e8, m1
        load(&mut c, 2, &[0x81, 0xff, 0x40, 0x01], 1);
        load(&mut c, 3, &[1, 7, 12, 0], 1);
        exec_arith(&mut c, VArithOp::Srl, 1, 2, VSel::V(3), false);
        // 12 is masked to log2(8) = 3 bits, so the last element shifts by 4.
        assert_eq!(elems(&c, 1, 4, 1), [0x40, 0x01, 0x04, 0x01]);
        exec_arith(&mut c, VArithOp::Sra, 1, 2, VSel::V(3), false);
        assert_eq!(elems(&c, 1, 4, 1), [0xc0, 0xff, 0x04, 0x01]);
        exec_arith(&mut c, VArithOp::Ror, 1, 2, VSel::V(3), false);
        assert_eq!(elems(&c, 1, 4, 1), [0xc0, 0xff, 0x04, 0x01]);
    }

    #[test]
    fn merge_picks_the_source_through_the_mask() {
        let mut c = machine(0, 4);
        load(&mut c, 2, &[0x11, 0x22, 0x33, 0x44], 1);
        mask_set(&mut c, 0, 1, true);
        mask_set(&mut c, 0, 2, true);
        // vmerge.vim sign-extends its immediate: 31 is -1.
        exec_merge(&mut c, 1, 2, VSel::I(31));
        assert_eq!(elems(&c, 1, 4, 1), [0x11, 0xff, 0xff, 0x44]);
    }

    #[test]
    fn compress_packs_the_selected_elements() {
        let mut c = machine(0, 4);
        load(&mut c, 2, &[0x11, 0x22, 0x33, 0x44], 1);
        load(&mut c, 1, &[0xee, 0xee, 0xee, 0xee], 1); // the old tail stays
        mask_set(&mut c, 3, 0, true);
        mask_set(&mut c, 3, 2, true);
        exec_compress(&mut c, 1, 2, 3);
        assert_eq!(elems(&c, 1, 4, 1), [0x11, 0x33, 0xee, 0xee]);
    }

    #[test]
    fn reduction_and_widening_use_the_spec_widths() {
        let mut c = machine(0, 2);
        load(&mut c, 2, &[0xff, 0x0f], 1);
        load(&mut c, 3, &[0x01], 1); // the seed is element 0 of vs1
        set_elem(&mut c, 1, 1, 1, 0x5a); // element 1 must stay untouched
        exec_red(&mut c, VRedOp::Xor, 1, 3, 2);
        // Single-width reductions write exactly SEW: "all operands and results ...
        // have the same SEW width"; only element 0 changes.
        assert_eq!(get_elem(&c, 1, 0, 1), 0x01 ^ 0xff ^ 0x0f);
        assert_eq!(get_elem(&c, 1, 1, 1), 0x5a);

        // vwaddu.wv at e8: vs2 is read 16 bits wide, the source is zero-extended.
        let mut c = machine(0, 2);
        set_elem(&mut c, 2, 0, 2, 0xff00);
        set_elem(&mut c, 2, 1, 2, 0x0001);
        load(&mut c, 3, &[0x01, 0xff], 1);
        exec_widen(&mut c, false, 1, 2, VSel::V(3));
        assert_eq!(elems(&c, 1, 2, 2), [0xff01, 0x0100]);
    }

    #[test]
    fn narrow_shifts_keep_the_low_bits() {
        let mut c = machine(0, 1);
        set_elem(&mut c, 2, 0, 2, 0xf123);
        exec_narrow(&mut c, false, false, 1, 2, VSel::I(12));
        assert_eq!(get_elem(&c, 1, 0, 1), 0x0f);
        exec_narrow(&mut c, true, false, 1, 2, VSel::I(12));
        assert_eq!(get_elem(&c, 1, 0, 1), 0xff);
        // Small shifts agree; the amount is masked to log2(2×SEW) bits.
        set_elem(&mut c, 2, 0, 2, 0x0780);
        exec_narrow(&mut c, false, false, 1, 2, VSel::I(3));
        assert_eq!(get_elem(&c, 1, 0, 1), 0xf0);
    }

    #[test]
    fn unary_bit_operations() {
        let mut c = machine(0, 2); // e8
        load(&mut c, 2, &[0x12, 0x00], 1);
        exec_unary(&mut c, VUnaryOp::Brev8, false, 1, 2);
        assert_eq!(elems(&c, 1, 2, 1), [0x48, 0x00]);
        exec_unary(&mut c, VUnaryOp::Clz, false, 1, 2);
        // Operates on vs2 (still 0x12 after the Brev8 write to v1): 3 leading zeros.
        assert_eq!(elems(&c, 1, 2, 1), [3, 8]);

        let mut c = machine(2 << 3, 1); // e32
        load(&mut c, 2, &[0x01020304], 4);
        exec_unary(&mut c, VUnaryOp::Rev8, false, 1, 2);
        assert_eq!(elems(&c, 1, 1, 4), [0x04030201]);
        exec_unary(&mut c, VUnaryOp::Brev, false, 1, 2);
        assert_eq!(elems(&c, 1, 1, 4), [0x20c0_4080]);
        load(&mut c, 2, &[0], 4);
        exec_unary(&mut c, VUnaryOp::Ctz, false, 1, 2);
        assert_eq!(elems(&c, 1, 1, 4), [32]);
    }

    #[test]
    fn compares_pick_signed_or_unsigned() {
        let mut c = machine(0, 2);
        load(&mut c, 2, &[0x81, 0x01], 1);
        load(&mut c, 3, &[0x02, 0x02], 1);
        exec_cmp(&mut c, VCmpOp::Lt, 1, 2, VSel::V(3), false);
        assert!(mask_get(&c, 1, 0), "0x81 is -127, so signed less-than");
        exec_cmp(&mut c, VCmpOp::LtU, 1, 2, VSel::V(3), false);
        assert!(!mask_get(&c, 1, 0), "and 0x81 is 129 unsigned");
    }

    #[test]
    fn indexed_loads_gather_through_the_index_vector() {
        const BASE: u64 = 0x1000;
        let mem = FlatMemory::new(BASE, 256);
        for (i, b) in [0xa0u8, 0xb1, 0xc2, 0xd3, 0xe4].iter().enumerate() {
            mem.write(BASE + 64 + i as u64, &[*b]).unwrap();
        }
        let mut c = machine(0, 3);
        load(&mut c, 2, &[0, 2, 4], 1); // e8 indices
        c.set_x(10, BASE + 64);
        exec_index_load(&mut c, &mem, 1, false, 1, 10, 2).unwrap();
        assert_eq!(elems(&c, 1, 3, 1), [0xa0, 0xc2, 0xe4]);
        // 16-bit indices address further, and masked-off elements keep their value.
        let mut c = machine(0, 2);
        load(&mut c, 1, &[0x77, 0x77], 1);
        set_elem(&mut c, 2, 0, 2, 1);
        set_elem(&mut c, 2, 1, 2, 3);
        c.set_x(10, BASE + 64);
        mask_set(&mut c, 0, 0, true);
        mask_set(&mut c, 0, 1, false);
        exec_index_load(&mut c, &mem, 2, true, 1, 10, 2).unwrap();
        assert_eq!(elems(&c, 1, 2, 1), [0xb1, 0x77]);
    }
}
