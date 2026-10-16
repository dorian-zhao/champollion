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
use crate::decoder::{VArithOp, VCmpOp, VRedOp, VSel};

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
fn lens(cpu: &CpuState, table: &Option<[u8; 16]>, reg: u8, i: u64, sew_b: u64) -> u64 {
    match table {
        Some(bytes) => snap_get(bytes, i, sew_b),
        None => get_elem(cpu, reg, i, sew_b),
    }
}

/// Element `i` of a snapshot of one register. Snapshot readers only ever look at
/// elements within the register (indices the active set can produce are < vlmax).
#[inline]
fn snap_get(table: &[u8; 16], i: u64, sew_b: u64) -> u64 {
    let off = (i * sew_b) as usize;
    let mut v = 0u64;
    for (k, &byte) in table[off..off + sew_b as usize].iter().enumerate() {
        v |= u64::from(byte) << (8 * k as u32);
    }
    v
}

/// Per-element arithmetic, truncated to the element width.
#[inline]
fn binop(op: VArithOp, a: u64, b: u64, sew_b: u64) -> u64 {
    let mask = if sew_b == 8 { u64::MAX } else { (1u64 << (8 * sew_b)) - 1 };
    match op {
        VArithOp::Add => a.wrapping_add(b) & mask,
        VArithOp::Rsub => b.wrapping_sub(a) & mask,
        VArithOp::And => a & b,
        VArithOp::Or => a | b,
        VArithOp::Xor => a ^ b,
        VArithOp::Andn => (a & !b) & mask,
        VArithOp::Sll => (a << (b % (8 * sew_b))) & mask,
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
    let old_vs2 = if snapshot { Some(cpu.v[vs2 as usize & 31]) } else { None };
    let old_src = match (snapshot, src) {
        (true, VSel::V(reg)) => Some(cpu.v[reg as usize & 31]),
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
                if i + off < vl { lens(cpu, &old_vs2, vs2, i + off, sew_b) } else { 0 }
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
                    VArithOp::Sll | VArithOp::Gather | VArithOp::SlideUp | VArithOp::SlideDown
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
            VCmpOp::LeU => a <= b,
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

pub fn exec_mv_nr(cpu: &mut CpuState, vd: u8, vs2: u8) {
    cpu.v[vd as usize & 31] = cpu.v[vs2 as usize & 31];
}

pub fn exec_merge(cpu: &mut CpuState, vd: u8, vs1: u8, vs2: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        // Spec: vd[i] = v0.mask[i] ? vs1[i] : vs2[i] (the mask selects vs1).
        let v = if mask_get(cpu, 0, i) {
            get_elem(cpu, vs1, i, sew_b)
        } else {
            get_elem(cpu, vs2, i, sew_b)
        };
        set_elem(cpu, vd, i, sew_b, v);
    }
}

/// `vredand.vs` / `vredor.vs`: the accumulator is 2×SEW wide and lands in element 0,
/// seeded from element 0 of `vs1` (zero-extended, as the spec requires for logical ops).
pub fn exec_red(cpu: &mut CpuState, op: VRedOp, vd: u8, vs1: u8, vs2: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    let mut acc = get_elem(cpu, vs1, 0, sew_b);
    for i in 0..vl {
        let el = get_elem(cpu, vs2, i, sew_b);
        acc = match op {
            VRedOp::And => acc & el,
            VRedOp::Or => acc | el,
        };
    }
    set_elem(cpu, vd, 0, sew_b * 2, acc);
}

/// `vmv.x.s`: element 0 to a scalar register.
pub fn elem0(cpu: &CpuState, vs2: u8) -> u64 {
    get_elem(cpu, vs2, 0, sew_bytes(cpu))
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

/// `vnsrl.wi`: take 2×SEW-wide elements from `vs2`, shift, narrow to SEW.
pub fn exec_nsrl(cpu: &mut CpuState, vd: u8, vs2: u8, imm: u8) {
    let vl = cpu.vl;
    let sew_b = sew_bytes(cpu);
    for i in 0..vl {
        let wide = get_elem(cpu, vs2, i, sew_b * 2) >> u32::from(imm);
        set_elem(cpu, vd, i, sew_b, wide);
    }
}
