//! Reference interpreter. Slow on purpose: it is the oracle the JIT is diffed against.

use crate::cpu::CpuState;
use crate::decoder::{
    self, AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, Csr, CsrOp, DecodeError, FWidth, Inst,
    LoadWidth, SgnjKind, StoreWidth, VMaskOp, VMemKind, VSel, VSetKind, VStoreKind,
};
use crate::fpu;
use crate::memory::{MemFault, Memory};
use crate::vector;

/// Why execution stopped. On every trap, `cpu.pc` still points at the trapping instruction
/// and no architectural state has been modified, like `sepc` on real hardware.
///
/// - `Ecall`: system call request. The handler reads `a7`/`a0..a5`, writes `a0`, and
///   resumes at `pc + 4` (`ecall` has no compressed form).
/// - `MisalignedAtomic`: `lr`/`sc`/AMO on an address not aligned to its width; Linux
///   delivers SIGBUS.
/// - `ReservedRoundingMode`: an FP instruction with `rm = dyn` while `frm` holds a reserved
///   value (5–7); an illegal instruction on real hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    FetchFault(MemFault),
    LoadFault { pc: u64, fault: MemFault },
    StoreFault { pc: u64, fault: MemFault },
    IllegalInstruction { pc: u64, err: DecodeError },
    Ecall { pc: u64 },
    Ebreak { pc: u64 },
    MisalignedAtomic { pc: u64, addr: u64 },
    ReservedRoundingMode { pc: u64 },
}

// With the C extension, IALIGN is 16 and every control-transfer target is even by
// construction (branch/jal offsets are multiples of 2, jalr clears bit 0), so instruction
// address misalignment can never occur and has no trap variant.

/// Fetch, decode and execute one instruction.
pub fn step(cpu: &mut CpuState, mem: &impl Memory) -> Result<(), Trap> {
    let pc = cpu.pc;
    let lo = mem.fetch_u16(pc).map_err(Trap::FetchFault)?;
    let len = decoder::inst_len(lo);
    let word = if len == 4 { mem.fetch_u32(pc).map_err(Trap::FetchFault)? } else { lo as u32 };
    let inst = decoder::decode(word).map_err(|err| Trap::IllegalInstruction { pc, err })?;
    execute(cpu, mem, inst, len)
}

/// Run until a trap. Returns the trap and the number of instructions retired before it.
pub fn run(cpu: &mut CpuState, mem: &impl Memory, max_steps: u64) -> (Option<Trap>, u64) {
    for n in 0..max_steps {
        if let Err(t) = step(cpu, mem) {
            return (Some(t), n);
        }
    }
    (None, max_steps)
}

/// Execute an already-decoded instruction of `len` bytes at `cpu.pc`.
pub fn execute(cpu: &mut CpuState, mem: &impl Memory, inst: Inst, len: u64) -> Result<(), Trap> {
    let pc = cpu.pc;
    let mut next = pc.wrapping_add(len);
    match inst {
        Inst::Lui { rd, imm } => cpu.set_x(rd, sext(imm)),
        Inst::Auipc { rd, imm } => cpu.set_x(rd, pc.wrapping_add(sext(imm))),
        Inst::Jal { rd, offset } => {
            cpu.set_x(rd, next);
            next = pc.wrapping_add(sext(offset));
        }
        Inst::Jalr { rd, rs1, offset } => {
            // Compute the target before writing rd: `jalr ra, 0(ra)` is common.
            let target = cpu.x(rs1).wrapping_add(sext(offset)) & !1;
            cpu.set_x(rd, next);
            next = target;
        }
        Inst::Branch { cond, rs1, rs2, offset } => {
            if branch_taken(cond, cpu.x(rs1), cpu.x(rs2)) {
                next = pc.wrapping_add(sext(offset));
            }
        }
        Inst::Load { width, rd, rs1, offset } => {
            let addr = cpu.x(rs1).wrapping_add(sext(offset));
            let v = load_value(mem, addr, width).map_err(|fault| Trap::LoadFault { pc, fault })?;
            cpu.set_x(rd, v);
        }
        Inst::Store { width, rs1, rs2, offset } => {
            let addr = cpu.x(rs1).wrapping_add(sext(offset));
            let bytes = cpu.x(rs2).to_le_bytes();
            let n = match width {
                StoreWidth::B => 1,
                StoreWidth::H => 2,
                StoreWidth::W => 4,
                StoreWidth::D => 8,
            };
            mem.write(addr, &bytes[..n]).map_err(|fault| Trap::StoreFault { pc, fault })?;
        }
        Inst::OpImm { op, rd, rs1, imm } => cpu.set_x(rd, alu(op, cpu.x(rs1), sext(imm))),
        Inst::Op { op, rd, rs1, rs2 } => cpu.set_x(rd, alu(op, cpu.x(rs1), cpu.x(rs2))),
        Inst::OpImm32 { op, rd, rs1, imm } => cpu.set_x(rd, alu32(op, cpu.x(rs1), sext(imm))),
        Inst::Op32 { op, rd, rs1, rs2 } => cpu.set_x(rd, alu32(op, cpu.x(rs1), cpu.x(rs2))),
        // The interpreter's own accesses are relaxed atomics, so a full host fence gives
        // every fence (including fence.tso) at least the ordering it asks for. The JIT will
        // emit the matching, cheaper `dmb` variant instead.
        Inst::Fence { .. } => std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst),
        // This interpreter fetches from memory on every step, so it has no stale code to
        // discard; the caching engine flushes its decoded blocks here.
        Inst::FenceI => {}
        Inst::Csr { op, csr, rd, src, imm } => {
            let operand = if imm { src as u64 } else { cpu.x(src) };
            let old = read_csr(cpu, csr);
            // The decoder already rejected writes to read-only CSRs; csrrs/csrrc with a
            // zero source do not write at all.
            let writes = op == CsrOp::Rw || src != 0;
            if writes {
                let new = match op {
                    CsrOp::Rw => operand,
                    CsrOp::Rs => old | operand,
                    CsrOp::Rc => old & !operand,
                };
                write_csr(cpu, csr, new);
            }
            cpu.set_x(rd, old);
        }
        Inst::Ecall => return Err(Trap::Ecall { pc }),
        Inst::Ebreak => return Err(Trap::Ebreak { pc }),
        // Atomics map onto host atomics (sequentially consistent, which covers every aq/rl
        // combination). lr/sc use a value-based reservation and compare-and-swap.
        Inst::Lr { width, rd, rs1, .. } => {
            let addr = atomic_addr(cpu, rs1, width)?;
            let v = load_value(mem, addr, amo_load_width(width))
                .map_err(|fault| Trap::LoadFault { pc, fault })?;
            cpu.set_x(rd, v);
            (cpu.reserved_addr, cpu.reserved_value) = (addr, v);
        }
        Inst::Sc { width, rd, rs1, rs2, .. } => {
            let addr = atomic_addr(cpu, rs1, width)?;
            let size = amo_bytes(width);
            let mask = if size == 4 { 0xffff_ffff } else { u64::MAX };
            let stored = cpu.reserved_addr == addr
                && mem
                    .compare_exchange(addr, size, cpu.reserved_value & mask, cpu.x(rs2) & mask)
                    .map_err(|fault| Trap::StoreFault { pc, fault })?;
            cpu.set_x(rd, if stored { 0 } else { 1 });
            cpu.reserved_addr = crate::cpu::NO_RESERVATION;
        }
        Inst::Amo { op, width, rd, rs1, rs2, .. } => {
            let addr = atomic_addr(cpu, rs1, width)?;
            let src = cpu.x(rs2);
            // AMOs need write permission, so any fault is a store fault. rd is written
            // last so a fault leaves it intact.
            let old = mem
                .atomic_update(addr, amo_bytes(width), &mut |old| amo_op(op, width, old, src))
                .map_err(|fault| Trap::StoreFault { pc, fault })?;
            let old = if width == AmoWidth::W { old as i32 as u64 } else { old };
            cpu.set_x(rd, old);
        }
        Inst::FLoad { width, rd, rs1, offset } => {
            let addr = cpu.x(rs1).wrapping_add(sext(offset));
            let v = match width {
                FWidth::S => load_value(mem, addr, LoadWidth::Wu).map(nan_box),
                FWidth::D => load_value(mem, addr, LoadWidth::D),
            }
            .map_err(|fault| Trap::LoadFault { pc, fault })?;
            cpu.f[rd as usize] = v;
        }
        Inst::FStore { width, rs1, rs2, offset } => {
            // fsw stores the low 32 bits as-is, without checking the NaN-boxing.
            let addr = cpu.x(rs1).wrapping_add(sext(offset));
            let bytes = cpu.f[rs2 as usize].to_le_bytes();
            let n = if width == FWidth::S { 4 } else { 8 };
            mem.write(addr, &bytes[..n]).map_err(|fault| Trap::StoreFault { pc, fault })?;
        }
        Inst::FMvToInt { width, rd, rs1 } => {
            let v = cpu.f[rs1 as usize];
            cpu.set_x(rd, if width == FWidth::S { v as i32 as u64 } else { v });
        }
        Inst::FMvFromInt { width, rd, rs1 } => {
            let v = cpu.x(rs1);
            cpu.f[rd as usize] = if width == FWidth::S { nan_box(v as u32 as u64) } else { v };
        }
        Inst::FArith { op, width, rd, rs1, rs2, rm } => {
            let rm = rounding(cpu, rm)?;
            let (r, fl) = fpu::binop(op, width, fget(cpu, width, rs1), fget(cpu, width, rs2), rm);
            fset(cpu, width, rd, r, fl);
        }
        Inst::FSqrt { width, rd, rs1, rm } => {
            let rm = rounding(cpu, rm)?;
            let (r, fl) = fpu::sqrt(width, fget(cpu, width, rs1), rm);
            fset(cpu, width, rd, r, fl);
        }
        Inst::FFma { op, width, rd, rs1, rs2, rs3, rm } => {
            let rm = rounding(cpu, rm)?;
            let (a, b, c) = (fget(cpu, width, rs1), fget(cpu, width, rs2), fget(cpu, width, rs3));
            let (r, fl) = fpu::fma(op, width, a, b, c, rm);
            fset(cpu, width, rd, r, fl);
        }
        Inst::FMinMax { max, width, rd, rs1, rs2 } => {
            let (r, fl) = fpu::min_max(max, width, fget(cpu, width, rs1), fget(cpu, width, rs2));
            fset(cpu, width, rd, r, fl);
        }
        Inst::FCmp { op, width, rd, rs1, rs2 } => {
            let (r, fl) = fpu::compare(op, width, fget(cpu, width, rs1), fget(cpu, width, rs2));
            cpu.fcsr |= fl as u32;
            cpu.set_x(rd, r as u64);
        }
        Inst::FClass { width, rd, rs1 } => {
            cpu.set_x(rd, fpu::classify(width, fget(cpu, width, rs1)))
        }
        Inst::FCvtToInt { from, to, rd, rs1, rm } => {
            let rm = rounding(cpu, rm)?;
            let (r, fl) = fpu::to_int(from, to, fget(cpu, from, rs1), rm);
            cpu.fcsr |= fl as u32;
            cpu.set_x(rd, r);
        }
        Inst::FCvtFromInt { from, to, rd, rs1, rm } => {
            let rm = rounding(cpu, rm)?;
            let (r, fl) = fpu::from_int(to, from, cpu.x(rs1), rm);
            fset(cpu, to, rd, r, fl);
        }
        Inst::FCvtFloat { to, rd, rs1, rm } => {
            let rm = rounding(cpu, rm)?;
            let from = if to == FWidth::S { FWidth::D } else { FWidth::S };
            let (r, fl) = fpu::convert(to, fget(cpu, from, rs1), rm);
            fset(cpu, to, rd, r, fl);
        }
        Inst::FSgnj { kind, width, rd, rs1, rs2 } => {
            cpu.f[rd as usize] = match width {
                FWidth::S => {
                    let a = unbox(cpu.f[rs1 as usize]) as u64;
                    let b = unbox(cpu.f[rs2 as usize]) as u64;
                    nan_box(sign_inject(kind, a, b, 31))
                }
                FWidth::D => sign_inject(kind, cpu.f[rs1 as usize], cpu.f[rs2 as usize], 63),
            };
        }
        Inst::VSetvl { rd, rs1, rs2, kind } => {
            // AVL: x0 means "largest available" for the register forms; `vsetivli`
            // carries it as a 5-bit immediate instead. The immediate forms carry the
            // vtype in the instruction, the register form reads it from `rs2`.
            let avl = match kind {
                VSetKind::ImmediateAvl => u64::from(rs1),
                VSetKind::RegisterAvl | VSetKind::RegisterVtype if rs1 == 0 => u64::MAX,
                VSetKind::RegisterAvl | VSetKind::RegisterVtype => cpu.x(rs1),
            };
            let bits = match kind {
                VSetKind::RegisterVtype => cpu.x(rs2 as u8) & 0x3ff,
                _ => u64::from(rs2),
            };
            let vl = vector::vset(cpu, avl, bits);
            cpu.set_x(rd, vl);
        }
        Inst::VLoad { width, kind, vd, rs1, rs2, mask, nf } => {
            let base = cpu.x(rs1);
            let sew_b = width.bytes();
            match kind {
                VMemKind::WholeRegister => {
                    // Whole-register ops move `1 + nf` registers (vl1r..vl3r).
                    let n = 1 + nf as usize;
                    let mut b = vec![0u8; n * vector::VLEN_BYTES as usize];
                    mem.read(base, &mut b).map_err(|fault| Trap::LoadFault { pc, fault })?;
                    for (k, chunk) in b.as_chunks::<16>().0.iter().enumerate() {
                        cpu.v[(vd as usize + k) % 32] = *chunk;
                    }
                }
                VMemKind::Mask => {
                    // The mask register is VLEN/8 bits, loaded little-endian.
                    let mut b = [0u8; (vector::VLEN_BYTES / 8) as usize];
                    mem.read(base, &mut b).map_err(|fault| Trap::LoadFault { pc, fault })?;
                    cpu.v[0] = [0; 16];
                    cpu.v[0][..b.len()].copy_from_slice(&b);
                }
                VMemKind::Unit | VMemKind::FaultOnlyFirst | VMemKind::Stride => {
                    for i in 0..cpu.vl {
                        if mask && !vector::mask_get(cpu, 0, i) {
                            continue; // masked-off elements are never accessed
                        }
                        let addr = if kind == VMemKind::Stride {
                            base.wrapping_add(cpu.x(rs2).wrapping_mul(i))
                        } else {
                            base.wrapping_add(i.wrapping_mul(sew_b))
                        };
                        let mut b = [0u8; 8];
                        match mem.read(addr, &mut b[..sew_b as usize]) {
                            Ok(()) => {
                                let v = u64::from_le_bytes(b);
                                vector::set_elem(cpu, vd, i, sew_b, v);
                            }
                            Err(fault) => {
                                if kind == VMemKind::FaultOnlyFirst {
                                    // The point of the ff form: stop at the first fault
                                    // and report the successful prefix in `vl`.
                                    cpu.vl = i;
                                    break;
                                }
                                return Err(Trap::LoadFault { pc, fault });
                            }
                        }
                    }
                }
            }
        }
        Inst::VStore { width, kind, vs3, rs1, rs2, mask, nf } => {
            let base = cpu.x(rs1);
            let sew_b = width.bytes();
            match kind {
                VStoreKind::WholeRegister => {
                    let n = 1 + nf as usize;
                    let mut b = vec![0u8; n * vector::VLEN_BYTES as usize];
                    for k in 0..n {
                        b[k * 16..(k + 1) * 16].copy_from_slice(&cpu.v[(vs3 as usize + k) % 32]);
                    }
                    mem.write(base, &b).map_err(|fault| Trap::StoreFault { pc, fault })?;
                }
                VStoreKind::Unit | VStoreKind::Stride => {
                    for i in 0..cpu.vl {
                        if mask && !vector::mask_get(cpu, 0, i) {
                            continue;
                        }
                        let addr = if kind == VStoreKind::Stride {
                            base.wrapping_add(cpu.x(rs2).wrapping_mul(i))
                        } else {
                            base.wrapping_add(i.wrapping_mul(sew_b))
                        };
                        let v = vector::get_elem(cpu, vs3, i, sew_b).to_le_bytes();
                        mem.write(addr, &v[..sew_b as usize])
                            .map_err(|fault| Trap::StoreFault { pc, fault })?;
                    }
                }
            }
        }
        Inst::VArith { op, vd, vs2, src, mask } => vector::exec_arith(cpu, op, vd, vs2, src, mask),
        Inst::VCmp { op, vd, vs2, src, mask } => vector::exec_cmp(cpu, op, vd, vs2, src, mask),
        Inst::VMask { op: VMaskOp::Or, vd, vs1, vs2 } => vector::exec_mask_or(cpu, vd, vs1, vs2),
        Inst::Vmv { vd, src } => vector::exec_mv(cpu, vd, src),
        Inst::VmvNr { vd, vs2 } => vector::exec_mv_nr(cpu, vd, vs2),
        Inst::Vmerge { vd, vs1, vs2 } => vector::exec_merge(cpu, vd, vs1, vs2),
        Inst::VRed { op, vd, vs1, vs2 } => vector::exec_red(cpu, op, vd, vs1, vs2),
        Inst::VmvXs { rd, vs2 } => {
            let v = vector::elem0(cpu, vs2);
            cpu.set_x(rd, v);
        }
        Inst::VmvSx { vd, rs1 } => {
            let v = cpu.x(rs1);
            vector::set_elem0(cpu, vd, v);
        }
        Inst::Vcpop { rd, vs2 } => {
            let v = vector::cpop(cpu, vs2);
            cpu.set_x(rd, v);
        }
        Inst::Vfirst { rd, vs2 } => {
            let v = vector::vfirst(cpu, vs2);
            cpu.set_x(rd, v);
        }
        Inst::Vid { vd } => vector::exec_vid(cpu, vd),
        Inst::VExt { sext, narrow, vd, vs2 } => {
            let sew_b = vector::sew_bytes(cpu);
            let src_b = sew_b / u64::from(narrow);
            for i in 0..cpu.vl {
                let raw = vector::get_elem(cpu, vs2, i, src_b);
                let v = if sext { vector::sign_extend(raw, src_b) } else { raw };
                vector::set_elem(cpu, vd, i, sew_b, v);
            }
        }
        Inst::VMacc { add, vd, vs2, src } => {
            let sew_b = vector::sew_bytes(cpu);
            let mask = if sew_b == 8 { u64::MAX } else { (1u64 << (8 * sew_b)) - 1 };
            for i in 0..cpu.vl {
                let s = match src {
                    VSel::V(reg) => vector::get_elem(cpu, reg, i, sew_b),
                    VSel::X(reg) => cpu.x(reg),
                    VSel::I(imm) => u64::from(imm),
                };
                let old = vector::get_elem(cpu, vd, i, sew_b);
                let el = vector::get_elem(cpu, vs2, i, sew_b);
                let v = if add {
                    // vmacc: vd += src × vs2.
                    s.wrapping_mul(el).wrapping_add(old) & mask
                } else {
                    // vmadd: vd = src × vd + vs2.
                    s.wrapping_mul(old).wrapping_add(el) & mask
                };
                vector::set_elem(cpu, vd, i, sew_b, v);
            }
        }
        Inst::Vnsrl { vd, vs2, imm } => vector::exec_nsrl(cpu, vd, vs2, imm),
    }
    cpu.pc = next;
    cpu.instret = cpu.instret.wrapping_add(1);
    Ok(())
}

/// `time` ticks at 10 MHz, the timebase QEMU's `virt` machine reports, counted from the
/// host's monotonic clock.
fn host_time() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: ts is a valid out-pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 10_000_000 + ts.tv_nsec as u64 / 100
}

fn read_csr(cpu: &CpuState, csr: Csr) -> u64 {
    let fcsr = cpu.fcsr as u64;
    match csr {
        Csr::Fflags => fcsr & 0x1f,
        Csr::Frm => (fcsr >> 5) & 0b111,
        Csr::Fcsr => fcsr & 0xff,
        // No cycle model: report retired instructions, which is monotonic and cheap.
        Csr::Cycle | Csr::Instret => cpu.instret,
        Csr::Time => host_time(),
        Csr::Vl => cpu.vl,
        Csr::Vtype => cpu.vtype,
        Csr::Vlenb => vector::VLEN_BYTES,
    }
}

fn write_csr(cpu: &mut CpuState, csr: Csr, v: u64) {
    let v = v as u32;
    cpu.fcsr = match csr {
        Csr::Fflags => (cpu.fcsr & !0x1f) | (v & 0x1f),
        Csr::Frm => (cpu.fcsr & !0xe0) | ((v & 0b111) << 5),
        Csr::Fcsr => v & 0xff,
        Csr::Cycle | Csr::Time | Csr::Instret => unreachable!("read-only, rejected by decode"),
        Csr::Vl | Csr::Vtype | Csr::Vlenb => unreachable!("read-only, rejected by decode"),
    };
}

/// Effective address of an atomic, which must be naturally aligned.
fn atomic_addr(cpu: &CpuState, rs1: u8, width: AmoWidth) -> Result<u64, Trap> {
    let addr = cpu.x(rs1);
    if !addr.is_multiple_of(amo_bytes(width) as u64) {
        return Err(Trap::MisalignedAtomic { pc: cpu.pc, addr });
    }
    Ok(addr)
}

fn amo_bytes(width: AmoWidth) -> usize {
    match width {
        AmoWidth::W => 4,
        AmoWidth::D => 8,
    }
}

/// Word atomics sign-extend the loaded value into `rd`.
fn amo_load_width(width: AmoWidth) -> LoadWidth {
    match width {
        AmoWidth::W => LoadWidth::W,
        AmoWidth::D => LoadWidth::D,
    }
}

/// New memory value for an AMO. `old` is the zero-extended memory value; for word ops only
/// the low 32 bits of the result are stored, and min/max compare 32-bit values.
fn amo_op(op: AmoOp, width: AmoWidth, old: u64, src: u64) -> u64 {
    let (a, b) = match width {
        AmoWidth::W => (old as i32 as i64, src as i32 as i64),
        AmoWidth::D => (old as i64, src as i64),
    };
    let (ua, ub) = match width {
        AmoWidth::W => (old as u32 as u64, src as u32 as u64),
        AmoWidth::D => (old, src),
    };
    match op {
        AmoOp::Swap => src,
        AmoOp::Add => old.wrapping_add(src),
        AmoOp::Xor => old ^ src,
        AmoOp::And => old & src,
        AmoOp::Or => old | src,
        AmoOp::Min => a.min(b) as u64,
        AmoOp::Max => a.max(b) as u64,
        AmoOp::Minu => ua.min(ub),
        AmoOp::Maxu => ua.max(ub),
    }
}

/// Resolve an instruction's `rm` field, reading `frm` for `dyn`.
#[inline]
fn rounding(cpu: &CpuState, rm: u8) -> Result<fpu::Rm, Trap> {
    let bits = if rm == decoder::RM_DYN { (cpu.fcsr >> 5 & 0b111) as u8 } else { rm };
    fpu::Rm::from_bits(bits).ok_or(Trap::ReservedRoundingMode { pc: cpu.pc })
}

/// Read an FP operand as raw bits: doubles as-is, singles unboxed into the low 32 bits.
#[inline]
fn fget(cpu: &CpuState, w: FWidth, r: u8) -> u64 {
    match w {
        FWidth::S => unbox(cpu.f[r as usize]) as u64,
        FWidth::D => cpu.f[r as usize],
    }
}

/// Write an FP result (NaN-boxing singles) and accrue its exception flags.
#[inline]
fn fset(cpu: &mut CpuState, w: FWidth, r: u8, v: u64, flags: fpu::Fflags) {
    cpu.f[r as usize] = match w {
        FWidth::S => nan_box(v),
        FWidth::D => v,
    };
    cpu.fcsr |= flags as u32;
}

/// NaN-box a single-precision value (in the low 32 bits) for an `f` register.
fn nan_box(v: u64) -> u64 {
    0xffff_ffff_0000_0000 | (v & 0xffff_ffff)
}

/// Single-precision operand of an `f` register: values that are not properly NaN-boxed
/// read as the canonical NaN.
fn unbox(v: u64) -> u32 {
    if v >> 32 == 0xffff_ffff { v as u32 } else { 0x7fc0_0000 }
}

/// Sign injection on raw bits; `sign` is the sign bit position (31 or 63).
fn sign_inject(kind: SgnjKind, a: u64, b: u64, sign: u32) -> u64 {
    let mask = 1u64 << sign;
    let s = match kind {
        SgnjKind::Sgnj => b & mask,
        SgnjKind::Sgnjn => !b & mask,
        SgnjKind::Sgnjx => (a ^ b) & mask,
    };
    (a & !mask) | s
}

#[inline]
fn sext(imm: i32) -> u64 {
    imm as i64 as u64
}

#[inline]
fn branch_taken(cond: BranchCond, a: u64, b: u64) -> bool {
    match cond {
        BranchCond::Eq => a == b,
        BranchCond::Ne => a != b,
        BranchCond::Lt => (a as i64) < (b as i64),
        BranchCond::Ge => (a as i64) >= (b as i64),
        BranchCond::Ltu => a < b,
        BranchCond::Geu => a >= b,
    }
}

/// Misaligned addresses are allowed: Linux guarantees user-mode misaligned access works
/// (in hardware or via kernel emulation), and arm64 handles it natively.
#[inline]
fn load_value(mem: &impl Memory, addr: u64, width: LoadWidth) -> Result<u64, MemFault> {
    let mut b = [0u8; 8];
    let n = match width {
        LoadWidth::B | LoadWidth::Bu => 1,
        LoadWidth::H | LoadWidth::Hu => 2,
        LoadWidth::W | LoadWidth::Wu => 4,
        LoadWidth::D => 8,
    };
    mem.read(addr, &mut b[..n])?;
    let v = u64::from_le_bytes(b);
    Ok(match width {
        LoadWidth::B => v as i8 as u64,
        LoadWidth::H => v as i16 as u64,
        LoadWidth::W => v as i32 as u64,
        LoadWidth::D | LoadWidth::Bu | LoadWidth::Hu | LoadWidth::Wu => v,
    })
}

/// 64-bit ALU. Shift amounts use the low 6 bits of `b`.
///
/// Division never traps (M extension): x / 0 = all ones, x % 0 = x, and the signed overflow
/// case MIN / -1 = MIN with remainder 0, which is exactly what `wrapping_div`/`wrapping_rem`
/// give.
#[inline]
fn alu(op: AluOp, a: u64, b: u64) -> u64 {
    let sh = (b & 63) as u32;
    match op {
        AluOp::Add => a.wrapping_add(b),
        AluOp::Sub => a.wrapping_sub(b),
        AluOp::Sll => a << sh,
        AluOp::Slt => ((a as i64) < (b as i64)) as u64,
        AluOp::Sltu => (a < b) as u64,
        AluOp::Xor => a ^ b,
        AluOp::Srl => a >> sh,
        AluOp::Sra => ((a as i64) >> sh) as u64,
        AluOp::Or => a | b,
        AluOp::And => a & b,
        AluOp::Mul => a.wrapping_mul(b),
        AluOp::Mulh => ((a as i64 as i128 * b as i64 as i128) >> 64) as u64,
        AluOp::Mulhsu => ((a as i64 as i128 * b as i128) >> 64) as u64,
        AluOp::Mulhu => ((a as u128 * b as u128) >> 64) as u64,
        AluOp::Div if b == 0 => u64::MAX,
        AluOp::Div => (a as i64).wrapping_div(b as i64) as u64,
        AluOp::Divu => a.checked_div(b).unwrap_or(u64::MAX),
        AluOp::Rem if b == 0 => a,
        AluOp::Rem => (a as i64).wrapping_rem(b as i64) as u64,
        AluOp::Remu => a.checked_rem(b).unwrap_or(a),
        // Zba. The word-sourced adds zero-extend the low word of `a` but produce a full
        // 64-bit sum, so they carry a 64-bit result here, not an `AluOp32` one.
        AluOp::Sh1Add => (a << 1).wrapping_add(b),
        AluOp::Sh2Add => (a << 2).wrapping_add(b),
        AluOp::Sh3Add => (a << 3).wrapping_add(b),
        AluOp::AddUw => (a as u32 as u64).wrapping_add(b),
        AluOp::Sh1AddUw => ((a as u32 as u64) << 1).wrapping_add(b),
        AluOp::Sh2AddUw => ((a as u32 as u64) << 2).wrapping_add(b),
        AluOp::Sh3AddUw => ((a as u32 as u64) << 3).wrapping_add(b),
        AluOp::SllUw => (a as u32 as u64) << sh,
        // Zbb.
        AluOp::Andn => a & !b,
        AluOp::Orn => a | !b,
        AluOp::Xnor => !(a ^ b),
        AluOp::Max => (a as i64).max(b as i64) as u64,
        AluOp::Maxu => a.max(b),
        AluOp::Min => (a as i64).min(b as i64) as u64,
        AluOp::Minu => a.min(b),
        AluOp::Rol => a.rotate_left(sh),
        AluOp::Ror => a.rotate_right(sh),
        AluOp::Clz => u64::from(a.leading_zeros()),
        AluOp::Ctz => u64::from(a.trailing_zeros()),
        AluOp::Cpop => u64::from(a.count_ones()),
        AluOp::SextB => a as i8 as i64 as u64,
        AluOp::SextH => a as i16 as i64 as u64,
        AluOp::ZextH => a as u16 as u64,
        AluOp::Rev8 => a.swap_bytes(),
        AluOp::Brev8 => u64::from_le_bytes(a.to_le_bytes().map(|b| b.reverse_bits())),
        AluOp::OrcB => {
            let mut r = 0u64;
            for i in 0..8 {
                if a >> (8 * i) & 0xff != 0 {
                    r |= 0xff << (8 * i);
                }
            }
            r
        }
        // Zbkb: the low half (XLEN/2, so a word here) of each source, packed; `packw`
        // sign-extends the halfword it inserts, and `packh` packs bytes.
        AluOp::Pack => (a & 0xffff_ffff) | (b << 32),
        AluOp::Packh => (a & 0xff) | ((b & 0xff) << 8),
        AluOp::Packw => ((a & 0xffff) | ((b & 0xffff) << 16)) as u32 as i32 as u64,
        // Zbc: the carry-less product is a sum (xor) of `a` shifted by each set bit of
        // `b`; clmul keeps the low 64 bits, clmulh the high ones, and clmulr the window
        // around bit 63 (bits 126..63).
        AluOp::Clmul => {
            let mut r = 0u64;
            for i in 0..64 {
                if b >> i & 1 != 0 {
                    r ^= a << i;
                }
            }
            r
        }
        AluOp::Clmulh => {
            let mut r = 0u64;
            for i in 1..64 {
                if b >> i & 1 != 0 {
                    r ^= a >> (64 - i);
                }
            }
            r
        }
        AluOp::Clmulr => {
            let mut r = 0u64;
            for i in 0..64 {
                if b >> i & 1 != 0 {
                    r ^= a >> (63 - i);
                }
            }
            r
        }
        // Zbkx: every nibble (byte) of `b` indexes a nibble (byte) of the table in `a`.
        // An index past the register selects zero (no wrap-around), matching spike.
        AluOp::Xperm4 => {
            let mut r = 0u64;
            for i in 0..16 {
                let pos = ((b >> (4 * i)) & 0xf) << 2;
                if pos < 64 {
                    r |= ((a >> pos) & 0xf) << (4 * i);
                }
            }
            r
        }
        AluOp::Xperm8 => {
            let mut r = 0u64;
            for i in 0..8 {
                let pos = ((b >> (8 * i)) & 0xff) << 3;
                if pos < 64 {
                    r |= ((a >> pos) & 0xff) << (8 * i);
                }
            }
            r
        }
        // Zbs.
        AluOp::Bclr => a & !(1u64 << sh),
        AluOp::Bext => (a >> sh) & 1,
        AluOp::Binv => a ^ (1u64 << sh),
        AluOp::Bset => a | (1u64 << sh),
        // Zicond.
        AluOp::CzeroEqz => {
            if b == 0 {
                0
            } else {
                a
            }
        }
        AluOp::CzeroNez => {
            if b != 0 {
                0
            } else {
                a
            }
        }
    }
}

/// 32-bit ALU for the `*w` instructions: uses the low 32 bits of the operands and the low
/// 5 bits of the shift amount, then sign-extends the 32-bit result.
#[inline]
fn alu32(op: AluOp32, a: u64, b: u64) -> u64 {
    let (a, b) = (a as u32, b as u32);
    let sh = b & 31;
    let r = match op {
        AluOp32::Add => a.wrapping_add(b),
        AluOp32::Sub => a.wrapping_sub(b),
        AluOp32::Sll => a << sh,
        AluOp32::Srl => a >> sh,
        AluOp32::Sra => ((a as i32) >> sh) as u32,
        AluOp32::Mul => a.wrapping_mul(b),
        AluOp32::Div if b == 0 => u32::MAX,
        AluOp32::Div => (a as i32).wrapping_div(b as i32) as u32,
        AluOp32::Divu => a.checked_div(b).unwrap_or(u32::MAX),
        AluOp32::Rem if b == 0 => a,
        AluOp32::Rem => (a as i32).wrapping_rem(b as i32) as u32,
        AluOp32::Remu => a.checked_rem(b).unwrap_or(a),
        // Zbb word rotations; the sign-extension below applies to them like the *w ops.
        AluOp32::Rol => a.rotate_left(sh),
        AluOp32::Ror => a.rotate_right(sh),
        // Zbb word counts: both use only the low 32 bits, and the results (0..=32) are
        // unchanged by the sign-extension below.
        AluOp32::Clz => a.leading_zeros(),
        AluOp32::Ctz => a.trailing_zeros(),
        AluOp32::Cpop => a.count_ones(),
    };
    r as i32 as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::enc::*;
    use crate::memory::FlatMemory;

    const BASE: u64 = 0x8000_0000;
    const DATA: u64 = BASE + 0x800;

    fn setup(prog: &[u32]) -> (CpuState, FlatMemory) {
        let mem = FlatMemory::new(BASE, 4096);
        mem.load_words(BASE, prog).unwrap();
        (CpuState::new(BASE), mem)
    }

    /// Run `prog` with the given initial registers until pc falls off its end.
    fn exec(prog: &[u32], regs: &[(u8, u64)]) -> (CpuState, FlatMemory) {
        let (mut cpu, mem) = setup(prog);
        for &(r, v) in regs {
            cpu.set_x(r, v);
        }
        let end = BASE + 4 * prog.len() as u64;
        while cpu.pc != end {
            step(&mut cpu, &mem).unwrap_or_else(|t| panic!("{t:?}\n{cpu:?}"));
        }
        (cpu, mem)
    }

    /// Execute one instruction with a1 = `a`, a2 = `b`, and return a0.
    fn one(word: u32, a: u64, b: u64) -> u64 {
        exec(&[word], &[(11, a), (12, b)]).0.x[10]
    }

    const NEG1: u64 = u64::MAX;
    const MIN: u64 = i64::MIN as u64;

    #[test]
    fn add_and_addi() {
        let (cpu, _) = exec(
            &[
                addi(10, 0, 40),  // a0 = 40
                addi(11, 0, 2),   // a1 = 2
                add(12, 10, 11),  // a2 = a0 + a1
                addi(13, 0, -1),  // a3 = -1
                add(14, 13, 13),  // a4 = -2 (wrapping)
                addi(0, 10, 123), // write to x0 is discarded
            ],
            &[],
        );
        assert_eq!(cpu.x[12], 42);
        assert_eq!(cpu.x[13], NEG1);
        assert_eq!(cpu.x[14], (-2i64) as u64);
        assert_eq!(cpu.x[0], 0);
    }

    #[test]
    fn lui_and_auipc() {
        let (cpu, _) = exec(
            &[
                lui(10, 0x12345),   // a0 = 0x12345000
                lui(11, 0x80000),   // a1 = 0xffffffff80000000 (sign-extended)
                auipc(12, 0),       // a2 = pc
                auipc(13, 0xfffff), // a3 = pc - 0x1000
                lui(0, 1),
            ],
            &[],
        );
        assert_eq!(cpu.x[10], 0x1234_5000);
        assert_eq!(cpu.x[11], 0xffff_ffff_8000_0000);
        assert_eq!(cpu.x[12], BASE + 8);
        assert_eq!(cpu.x[13], BASE + 12 - 0x1000);
        assert_eq!(cpu.x[0], 0);
    }

    #[test]
    fn alu_reg_imm() {
        // (encoding with rd=a0, rs1=a1; a1 value; expected a0)
        let cases = [
            (op_imm(0b000, 10, 11, -1), 0, NEG1),                   // addi
            (op_imm(0b010, 10, 11, 0), NEG1, 1),                    // slti: -1 < 0
            (op_imm(0b010, 10, 11, -1), 0, 0),                      // slti: 0 < -1 is false
            (op_imm(0b011, 10, 11, -1), 5, 1),                      // sltiu: 5 < u64::MAX
            (op_imm(0b011, 10, 11, 1), 0, 1),                       // sltiu rd, rs, 1 == seqz
            (op_imm(0b100, 10, 11, -1), 0xf0, !0xf0),               // xori -1 == not
            (op_imm(0b110, 10, 11, 0x7ff), 0x1000, 0x17ff),         // ori
            (op_imm(0b111, 10, 11, -16), 0x1234_5678, 0x1234_5670), // andi sign-extends imm
            (op_imm(0b001, 10, 11, 63), 1, MIN),                    // slli 63
            (op_imm(0b101, 10, 11, 63), MIN, 1),                    // srli 63
            (op_imm(0b101, 10, 11, 0x400 | 63), MIN, NEG1),         // srai 63
            (op_imm(0b101, 10, 11, 0x400 | 4), 0x80, 0x8),          // srai positive
        ];
        for (i, &(w, a, want)) in cases.iter().enumerate() {
            assert_eq!(one(w, a, 0), want, "case {i}: {:?}", decoder::decode(w));
        }
    }

    #[test]
    fn alu_reg_reg() {
        // (funct3, funct7, a1, a2, expected a0)
        let cases = [
            (0b000, 0x00, NEG1, 1, 0),             // add wraps
            (0b000, 0x20, 0, 1, NEG1),             // sub wraps
            (0b001, 0x00, 1, 64 + 3, 8),           // sll uses low 6 bits of rs2
            (0b010, 0x00, MIN, 0, 1),              // slt signed
            (0b011, 0x00, MIN, 0, 0),              // sltu unsigned
            (0b011, 0x00, 0, 7, 1),                // sltu rd, x0, rs == snez
            (0b100, 0x00, 0b1100, 0b1010, 0b0110), // xor
            (0b101, 0x00, MIN, 63, 1),             // srl logical
            (0b101, 0x20, MIN, 63, NEG1),          // sra arithmetic
            (0b101, 0x20, MIN, 64, MIN),           // sra by 64 == by 0
            (0b110, 0x00, 0b1100, 0b1010, 0b1110), // or
            (0b111, 0x00, 0b1100, 0b1010, 0b1000), // and
        ];
        for (i, &(f3, f7, a, b, want)) in cases.iter().enumerate() {
            assert_eq!(one(op(f3, f7, 10, 11, 12), a, b), want, "case {i}");
        }
    }

    #[test]
    fn word_ops_sign_extend() {
        // (encoding with rd=a0, rs1=a1, rs2=a2; a1; a2; expected a0)
        let cases = [
            (op_imm_32(0b000, 10, 11, 1), 0x7fff_ffff, 0, 0xffff_ffff_8000_0000), // addiw overflow
            (op_imm_32(0b000, 10, 11, 0), 0x1_0000_0001, 0, 1), // sext.w drops upper bits
            (op_imm_32(0b001, 10, 11, 31), 1, 0, 0xffff_ffff_8000_0000), // slliw
            (op_imm_32(0b101, 10, 11, 0), 0x8000_0000, 0, 0xffff_ffff_8000_0000), // srliw 0
            (op_imm_32(0b101, 10, 11, 1), 0xffff_ffff_8000_0000, 0, 0x4000_0000), // srliw ignores upper
            (op_imm_32(0b101, 10, 11, 0x400 | 31), 0x8000_0000, 0, NEG1),         // sraiw
            (op_32(0b000, 0x00, 10, 11, 12), 0xffff_ffff, 1, 0), // addw wraps at 32
            (op_32(0b000, 0x20, 10, 11, 12), 0, 1, NEG1),        // subw
            (op_32(0b001, 0x00, 10, 11, 12), 1, 32 + 31, 0xffff_ffff_8000_0000), // sllw: low 5 bits
            (op_32(0b101, 0x00, 10, 11, 12), NEG1, 4, 0x0fff_ffff), // srlw
            (op_32(0b101, 0x20, 10, 11, 12), 0x8000_0000, 4, 0xffff_ffff_f800_0000), // sraw
        ];
        for (i, &(w, a, b, want)) in cases.iter().enumerate() {
            assert_eq!(one(w, a, b), want, "case {i}: {:?}", decoder::decode(w));
        }
    }

    #[test]
    fn mul_div_edge_cases() {
        let m = |f3| op(f3, 0b000_0001, 10, 11, 12);
        let w = |f3| op_32(f3, 0b000_0001, 10, 11, 12);
        let (mul, mulh, mulhsu, mulhu) = (m(0b000), m(0b001), m(0b010), m(0b011));
        let (div, divu, rem, remu) = (m(0b100), m(0b101), m(0b110), m(0b111));
        let (mulw, divw, divuw, remw, remuw) = (w(0b000), w(0b100), w(0b101), w(0b110), w(0b111));
        // (encoding, a1, a2, expected a0)
        let cases = [
            (mul, NEG1, NEG1, 1),
            (mul, 1 << 32, 1 << 32, 0), // low 64 bits only
            (mulh, NEG1, NEG1, 0),      // (-1 * -1) >> 64
            (mulh, MIN, MIN, 1 << 62),
            (mulhu, NEG1, NEG1, NEG1 - 1),
            (mulhsu, NEG1, NEG1, NEG1), // -1 * (2^64 - 1) >> 64 = -1
            (mulhsu, 2, NEG1, 1),
            (div, 7, NEG1 - 1, NEG1 - 2),   // 7 / -2 = -3, truncating
            (rem, 7, NEG1 - 1, 1),          // remainder takes the dividend's sign
            (rem, (-7i64) as u64, 2, NEG1), // -7 % 2 = -1
            (div, 5, 0, NEG1),              // divide by zero
            (divu, 5, 0, NEG1),
            (rem, 5, 0, 5),
            (remu, 5, 0, 5),
            (div, MIN, NEG1, MIN), // signed overflow
            (rem, MIN, NEG1, 0),
            (divu, NEG1, 2, NEG1 >> 1),
            (mulw, 0x1_0000_0002, 0x4000_0000, 0xffff_ffff_8000_0000), // low 32 bits, sign-extended
            (divw, 0x1_0000_0007, 2, 3),                               // upper bits ignored
            (divw, 0x8000_0000, NEG1, 0xffff_ffff_8000_0000),          // 32-bit overflow
            (remw, 0x8000_0000, NEG1, 0),
            (divuw, 0xffff_fffe, 1, 0xffff_ffff_ffff_fffe), // result is sign-extended
            (divuw, 5, 0, NEG1),
            (remuw, 0x1_8000_0001, 0, 0xffff_ffff_8000_0001),
        ];
        for (i, &(w, a, b, want)) in cases.iter().enumerate() {
            assert_eq!(one(w, a, b), want, "case {i}: {:?}", decoder::decode(w));
        }
    }

    #[test]
    fn lr_sc_reservation() {
        let (lr_d, sc_d) = (amo(0b00010, 0b011, 10, 11, 0), amo(0b00011, 0b011, 13, 11, 12));
        let regs = [(11, DATA), (12, 0x55)];

        // lr; sc succeeds (rd = 0) and stores.
        let (cpu, mem) = exec(&[lr_d, sc_d], &regs);
        assert_eq!(cpu.x[13], 0);
        assert_eq!(mem.bytes()[0x800], 0x55);
        assert_eq!(cpu.reserved_addr, crate::cpu::NO_RESERVATION);

        // sc without lr fails (rd != 0) and does not store.
        let (cpu, mem) = exec(&[sc_d], &regs);
        assert_ne!(cpu.x[13], 0);
        assert_eq!(mem.bytes()[0x800], 0);

        // A second sc fails: the first one consumed the reservation.
        let (cpu, mem) = exec(&[lr_d, sc_d, addi(12, 0, 0x66), sc_d], &regs);
        assert_ne!(cpu.x[13], 0);
        assert_eq!(mem.bytes()[0x800], 0x55);

        // sc to a different address than the reservation fails.
        let (cpu, _) = exec(&[lr_d, addi(11, 11, 8), sc_d], &regs);
        assert_ne!(cpu.x[13], 0);

        // lr.w sign-extends.
        let (mut cpu, mem) = setup(&[amo(0b00010, 0b010, 10, 11, 0)]);
        mem.write(DATA, &0x8000_0000u32.to_le_bytes()).unwrap();
        cpu.set_x(11, DATA);
        step(&mut cpu, &mem).unwrap();
        assert_eq!(cpu.x[10], 0xffff_ffff_8000_0000);
        assert_eq!((cpu.reserved_addr, cpu.reserved_value), (DATA, 0xffff_ffff_8000_0000));
    }

    #[test]
    fn amo_ops() {
        // (funct5, funct3, memory before, rs2, memory after); rd always gets the old value.
        let w = 0b010;
        let d = 0b011;
        let cases = [
            (0b00001, d, 5, 9, 9),                                 // swap
            (0b00000, d, NEG1, 2, 1),                              // add wraps
            (0b00000, w, 0x7fff_ffff, 1, 0x8000_0000),             // add.w: only low 32 bits stored
            (0b00100, d, 0b1100, 0b1010, 0b0110),                  // xor
            (0b01100, d, 0b1100, 0b1010, 0b1000),                  // and
            (0b01000, d, 0b1100, 0b1010, 0b1110),                  // or
            (0b10000, d, NEG1, 1, NEG1),                           // min signed
            (0b10100, d, NEG1, 1, 1),                              // max signed
            (0b11000, d, NEG1, 1, 1),                              // minu
            (0b11100, d, NEG1, 1, NEG1),                           // maxu
            (0b10000, w, 0x8000_0000, 1, 0x8000_0000),             // min.w: 0x80000000 is negative
            (0b11100, w, 0x8000_0000, 0x1_0000_0001, 0x8000_0000), // maxu.w ignores upper rs2 bits
        ];
        for (i, &(f5, f3, before, src, after)) in cases.iter().enumerate() {
            let (mut cpu, mem) = setup(&[amo(f5, f3, 10, 11, 12)]);
            mem.write(DATA, &before.to_le_bytes()).unwrap();
            cpu.set_x(11, DATA);
            cpu.set_x(12, src);
            step(&mut cpu, &mem).unwrap();
            let mut got = [0; 8];
            mem.read(DATA, &mut got).unwrap();
            let mask = if f3 == w { 0xffff_ffff } else { NEG1 };
            let old = if f3 == w { before as u32 as i32 as u64 } else { before };
            assert_eq!(u64::from_le_bytes(got) & mask, after & mask, "case {i}: memory");
            assert_eq!(cpu.x[10], old, "case {i}: rd");
        }

        // rd == rs2: the operand is read before rd is overwritten.
        let (mut cpu, mem) = setup(&[amo(0b00000, d, 12, 11, 12)]);
        mem.write(DATA, &10u64.to_le_bytes()).unwrap();
        cpu.set_x(11, DATA);
        cpu.set_x(12, 5);
        step(&mut cpu, &mem).unwrap();
        assert_eq!(cpu.x[12], 10);
        assert_eq!(mem.bytes()[0x800], 15);
    }

    #[test]
    fn atomics_hold_across_host_threads() {
        use crate::memory::{GuestMemory, Prot};
        const CODE: u64 = 0x10000;
        const COUNTERS: u64 = 0x20000;
        const N: u64 = 20_000;
        let (amoadd_d, lr_d, sc_d) = (0b00000, 0b00010, 0b00011);
        // a0 = iterations, a1 = counter address, a2 = 1
        let amo_loop = [
            amo(amoadd_d, 0b011, 0, 11, 12), // 0x00: amoadd.d zero, a2, (a1)
            addi(10, 10, -1),                // 0x04: addi a0, a0, -1
            b(0b001, 10, 0, -8),             // 0x08: bnez a0, 0x00
        ];
        let lrsc_loop = [
            amo(lr_d, 0b011, 5, 11, 0), // 0x00: lr.d t0, (a1)
            addi(5, 5, 1),              // 0x04: addi t0, t0, 1
            amo(sc_d, 0b011, 6, 11, 5), // 0x08: sc.d t1, t0, (a1)
            b(0b001, 6, 0, -12),        // 0x0c: bnez t1, 0x00 (retry)
            addi(10, 10, -1),           // 0x10: addi a0, a0, -1
            b(0b001, 10, 0, -20),       // 0x14: bnez a0, 0x00
        ];
        let mem = GuestMemory::new().unwrap();
        mem.map(CODE, 0x1000, Prot::READ | Prot::EXEC).unwrap();
        mem.map(COUNTERS, 0x1000, Prot::RW).unwrap();
        let words: Vec<u8> =
            amo_loop.iter().chain(&lrsc_loop).flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(CODE, &words).unwrap();

        std::thread::scope(|s| {
            for i in 0..8 {
                let mem = &mem;
                s.spawn(move || {
                    let (entry, len, counter) = if i % 2 == 0 {
                        (CODE, amo_loop.len(), COUNTERS)
                    } else {
                        (CODE + 4 * amo_loop.len() as u64, lrsc_loop.len(), COUNTERS + 8)
                    };
                    let mut cpu = CpuState::new(entry);
                    cpu.set_x(10, N);
                    cpu.set_x(11, counter);
                    cpu.set_x(12, 1);
                    while cpu.pc != entry + 4 * len as u64 {
                        step(&mut cpu, mem).unwrap();
                    }
                });
            }
        });
        let mut b = [0u8; 16];
        mem.read(COUNTERS, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b[..8].try_into().unwrap()), 4 * N, "amoadd lost updates");
        assert_eq!(u64::from_le_bytes(b[8..].try_into().unwrap()), 4 * N, "lr/sc lost updates");
    }

    #[test]
    fn misaligned_atomics_trap() {
        // lr.w (rs2 must be 0), sc.w, amoadd.w
        for word in [
            amo(0b00010, 0b010, 10, 11, 0),
            amo(0b00011, 0b010, 10, 11, 12),
            amo(0, 0b010, 10, 11, 12),
        ] {
            let (mut cpu, mem) = setup(&[word]);
            cpu.set_x(11, DATA + 2);
            let before = cpu.clone();
            assert_eq!(
                step(&mut cpu, &mem),
                Err(Trap::MisalignedAtomic { pc: BASE, addr: DATA + 2 })
            );
            assert_eq!(cpu, before);
        }
    }

    #[test]
    fn compressed_instructions_advance_by_two() {
        // 0x00: c.li a0, 5        (2 bytes)
        // 0x02: c.jalr t0         (2 bytes) -> t0 = BASE + 0x0a, ra = BASE + 0x04
        // 0x04: addi a1, zero, 1  (4 bytes, at a 2-byte-aligned pc)
        // 0x08: c.j +4            -> 0x0c
        // 0x0a: c.addi a0, 1      (callee) ; c.jr ra
        let (mut cpu, mem) = setup(&[]);
        let halves: [u16; 2] = [0x4515, 0x9282]; // c.li a0, 5 ; c.jalr t0
        let mut bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        bytes.extend(addi(11, 0, 1).to_le_bytes());
        for h in [0xa011u16, 0x0505, 0x8082] {
            // c.j +4 ; c.addi a0, 1 ; c.jr ra
            bytes.extend(h.to_le_bytes());
        }
        mem.write(BASE, &bytes).unwrap();
        cpu.set_x(5, BASE + 0x0a);

        step(&mut cpu, &mem).unwrap(); // c.li
        assert_eq!((cpu.x[10], cpu.pc), (5, BASE + 2));
        step(&mut cpu, &mem).unwrap(); // c.jalr
        assert_eq!((cpu.x[1], cpu.pc), (BASE + 4, BASE + 0x0a));
        run(&mut cpu, &mem, 4); // c.addi, c.jr, addi, c.j
        assert_eq!((cpu.x[10], cpu.x[11]), (6, 1));
        assert_eq!(cpu.pc, BASE + 0x0c);
    }

    #[test]
    fn fp_loads_stores_and_moves() {
        let (mut cpu, mem) = setup(&[
            fload(0b011, 1, 11, 0),            // fld f1, 0(a1)
            fload(0b010, 2, 11, 8),            // flw f2, 8(a1): NaN-boxed
            fstore(0b011, 11, 1, 16),          // fsd f1, 16(a1)
            fstore(0b010, 11, 2, 24),          // fsw f2, 24(a1)
            op_fp(0b11100, 0b01, 0, 10, 1, 0), // fmv.x.d a0, f1
            op_fp(0b11100, 0b00, 0, 12, 2, 0), // fmv.x.w a2, f2: sign-extended
            op_fp(0b11110, 0b00, 0, 3, 13, 0), // fmv.w.x f3, a3: NaN-boxed
            op_fp(0b11110, 0b01, 0, 4, 13, 0), // fmv.d.x f4, a3
        ]);
        mem.write(DATA, &1.5f64.to_bits().to_le_bytes()).unwrap();
        mem.write(DATA + 8, &(-2.0f32).to_bits().to_le_bytes()).unwrap();
        cpu.set_x(11, DATA);
        cpu.set_x(13, 0x1234_5678_3f80_0000);
        run(&mut cpu, &mem, 8);

        assert_eq!(cpu.f[1], 1.5f64.to_bits());
        assert_eq!(cpu.f[2], 0xffff_ffff_c000_0000);
        let mut b = [0u8; 12];
        mem.read(DATA + 16, &mut b).unwrap();
        assert_eq!(b[..8], 1.5f64.to_bits().to_le_bytes());
        assert_eq!(b[8..], (-2.0f32).to_bits().to_le_bytes());
        assert_eq!(cpu.x[10], 1.5f64.to_bits());
        assert_eq!(cpu.x[12], 0xffff_ffff_c000_0000);
        assert_eq!(cpu.f[3], 0xffff_ffff_3f80_0000);
        assert_eq!(cpu.f[4], 0x1234_5678_3f80_0000);
    }

    #[test]
    fn fp_sign_injection() {
        let one = 1.0f64.to_bits();
        let neg = (-3.0f64).to_bits();
        let sgnj = |kind, fmt| op_fp(0b00100, fmt, kind, 3, 1, 2);
        // (encoding, f1, f2, expected f3)
        let cases = [
            (sgnj(0b000, 1), one, neg, (-1.0f64).to_bits()), // fsgnj.d
            (sgnj(0b001, 1), one, neg, one),                 // fsgnjn.d
            (sgnj(0b010, 1), neg, neg, 3.0f64.to_bits()),    // fsgnjx.d (fabs)
            (sgnj(0b001, 1), neg, neg, 3.0f64.to_bits()),    // fneg.d of -3
            (sgnj(0b000, 0), nan_box(0x3f80_0000), nan_box(0x8000_0000), nan_box(0xbf80_0000)),
            (sgnj(0b000, 0), 0x3f80_0000, nan_box(0), nan_box(0x7fc0_0000)), // unboxed -> NaN
        ];
        for (i, &(w, a, b, want)) in cases.iter().enumerate() {
            let (mut cpu, mem) = setup(&[w]);
            cpu.f[1] = a;
            cpu.f[2] = b;
            step(&mut cpu, &mem).unwrap();
            assert_eq!(cpu.f[3], want, "case {i}");
        }
    }

    #[test]
    fn csr_read_modify_write() {
        // csrrw/csrrs/csrrc and immediates on fflags/frm/fcsr, which share one register.
        let csr =
            |f3: u32, rd: u32, rs1: u32, n: u32| n << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x73;
        let (fflags, frm, fcsr) = (0x001, 0x002, 0x003);
        let (cpu, _) = exec(
            &[
                csr(0b001, 10, 11, fcsr),        // csrrw a0, fcsr, a1 (a1 = 0x1ff): a0 = 0
                csr(0b010, 12, 0, fflags),       // csrrs a2, fflags, zero: a2 = 0x1f, no write
                csr(0b010, 13, 0, frm),          // frrm a3: a3 = 0b111
                csr(0b111, 0, 0b00101, fflags),  // csrrci zero, fflags, 5
                csr(0b101, 14, 0b00001, frm),    // csrrwi a4, frm, 1: a4 = 7
                csr(0b110, 15, 0b00010, fflags), // csrrsi a5, fflags, 2: a5 = 0x1a
                csr(0b010, 16, 0, fcsr),         // frcsr a6
            ],
            &[(11, 0x1ff)],
        );
        assert_eq!(cpu.x[10], 0);
        assert_eq!(cpu.x[12], 0x1f, "fcsr keeps only 8 bits");
        assert_eq!(cpu.x[13], 0b111);
        assert_eq!(cpu.x[14], 0b111);
        assert_eq!(cpu.x[15], 0x1a);
        assert_eq!(cpu.x[16], 1 << 5 | 0x1a);
        assert_eq!(cpu.fcsr, 1 << 5 | 0x1a);
    }

    #[test]
    fn counters() {
        let csr = |rd: u32, n: u32| n << 20 | 0b010 << 12 | rd << 7 | 0x73;
        let (cpu, _) = exec(
            &[addi(0, 0, 0), csr(10, 0xc02), csr(11, 0xc00), csr(12, 0xc01), csr(13, 0xc01)],
            &[],
        );
        assert_eq!(cpu.x[10], 1, "instret counts instructions retired before it");
        assert_eq!(cpu.x[11], 2, "cycle reports instret");
        assert!(cpu.x[12] > 0 && cpu.x[13] >= cpu.x[12], "time is monotonic");
        assert_eq!(cpu.instret, 5);
    }

    #[test]
    fn branches_signed_vs_unsigned() {
        // a1 = -1, a2 = 1. Each branch skips the following `addi a0, a0, 1` when taken.
        // (funct3, taken)
        let cases = [
            (0b000, false), // beq
            (0b001, true),  // bne
            (0b100, true),  // blt: -1 < 1
            (0b101, false), // bge
            (0b110, false), // bltu: u64::MAX < 1 is false
            (0b111, true),  // bgeu
        ];
        for (f3, taken) in cases {
            let (cpu, _) = exec(&[b(f3, 11, 12, 8), addi(10, 10, 1)], &[(11, NEG1), (12, 1)]);
            assert_eq!(cpu.x[10], (!taken) as u64, "funct3 {f3:#05b}");
        }
    }

    #[test]
    fn beq_backward_loop() {
        // Count a0 up to 5 with a backward branch:
        //   0x00: addi a1, zero, 5
        //   0x04: addi a0, a0, 1
        //   0x08: beq  a0, a1, +8      -> exit
        //   0x0c: beq  zero, zero, -8  -> 0x04
        let (cpu, _) = exec(&[addi(11, 0, 5), addi(10, 10, 1), beq(10, 11, 8), beq(0, 0, -8)], &[]);
        assert_eq!(cpu.x[10], 5);
    }

    #[test]
    fn jal_links_and_jumps() {
        let (mut cpu, mem) = setup(&[
            jal(1, 12),     // 0x00: ra = 0x04, -> 0x0c
            addi(10, 0, 1), // 0x04: return lands here
            jal(0, 0),      // 0x08: j . (halt loop)
            addi(11, 0, 2), // 0x0c: "callee"
            jal(5, -12),    // 0x10: t0 = 0x14, -> 0x04
        ]);
        step(&mut cpu, &mem).unwrap();
        assert_eq!(cpu.x[1], BASE + 4);
        assert_eq!(cpu.pc, BASE + 0x0c);
        run(&mut cpu, &mem, 3);
        assert_eq!(cpu.x[10], 1);
        assert_eq!(cpu.x[11], 2);
        assert_eq!(cpu.x[5], BASE + 0x14);
        assert_eq!(cpu.pc, BASE + 0x08);
        // `j .` spins in place and never writes x0.
        run(&mut cpu, &mem, 10);
        assert_eq!(cpu.pc, BASE + 0x08);
        assert_eq!(cpu.x[0], 0);
    }

    #[test]
    fn jalr_call_and_return() {
        let (mut cpu, mem) = setup(&[
            auipc(5, 0),      // 0x00: t0 = BASE
            jalr(1, 5, 0x11), // 0x04: call BASE+0x10 (bit 0 cleared), ra = 0x08
            jal(0, 0),        // 0x08: j .
            0,                // 0x0c: padding
            addi(10, 0, 7),   // 0x10: callee
            jalr(0, 1, 0),    // 0x14: ret
        ]);
        run(&mut cpu, &mem, 5);
        assert_eq!(cpu.x[1], BASE + 8);
        assert_eq!(cpu.x[10], 7);
        assert_eq!(cpu.pc, BASE + 8);

        // rd == rs1: the target uses the old value.
        let (cpu, _) = exec(&[auipc(1, 0), jalr(1, 1, 12), 0], &[]);
        assert_eq!(cpu.x[1], BASE + 8);
    }

    #[test]
    fn loads_extend_correctly() {
        let (mut cpu, mem) = setup(&[]);
        mem.write(DATA, &0x8899_aabb_ccdd_eeffu64.to_le_bytes()).unwrap();
        // (funct3, expected)
        let cases = [
            (0b000, 0xffff_ffff_ffff_ffff), // lb  0xff
            (0b001, 0xffff_ffff_ffff_eeff), // lh  0xeeff
            (0b010, 0xffff_ffff_ccdd_eeff), // lw  0xccddeeff
            (0b011, 0x8899_aabb_ccdd_eeff), // ld
            (0b100, 0xff),                  // lbu
            (0b101, 0xeeff),                // lhu
            (0b110, 0xccdd_eeff),           // lwu
        ];
        for (f3, want) in cases {
            mem.load_words(BASE, &[load(f3, 10, 11, -8)]).unwrap();
            cpu.pc = BASE;
            cpu.set_x(11, DATA + 8);
            step(&mut cpu, &mem).unwrap();
            assert_eq!(cpu.x[10], want, "funct3 {f3:#05b}");
        }
    }

    #[test]
    fn stores_write_only_their_width() {
        for (f3, want) in [
            (0b000, 0xaaaa_aaaa_aaaa_aa88u64), // sb
            (0b001, 0xaaaa_aaaa_aaaa_7788),    // sh
            (0b010, 0xaaaa_aaaa_5566_7788),    // sw
            (0b011, 0x1122_3344_5566_7788),    // sd
        ] {
            let (mut cpu, mem) = setup(&[store(f3, 11, 12, 0)]);
            mem.write(DATA, &[0xaa; 8]).unwrap();
            cpu.set_x(11, DATA);
            cpu.set_x(12, 0x1122_3344_5566_7788);
            step(&mut cpu, &mem).unwrap();
            let mut got = [0; 8];
            mem.read(DATA, &mut got).unwrap();
            assert_eq!(u64::from_le_bytes(got), want, "funct3 {f3:#05b}");
        }
    }

    #[test]
    fn misaligned_access_is_allowed() {
        let (cpu, mem) = exec(
            &[store(3, 11, 12, 1), load(3, 10, 11, 1), load(1, 13, 11, 3)],
            &[(11, DATA), (12, 0x0102_0304_0506_0708)],
        );
        assert_eq!(cpu.x[10], 0x0102_0304_0506_0708);
        assert_eq!(cpu.x[13], 0x0506);
        assert_eq!(mem.bytes()[0x801..0x809], [8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn fence_is_a_nop() {
        let (cpu, _) = exec(&[FENCE, addi(10, 0, 1)], &[]);
        assert_eq!(cpu.x[10], 1);
    }

    #[test]
    fn traps_leave_state_untouched() {
        // Each trapping instruction sits after `addi a0, zero, 1` so we can check that pc
        // points at the trapping instruction and nothing else changed.
        let fault = MemFault { addr: 0x10 };
        let cases: [(u32, Trap); 5] = [
            (ECALL, Trap::Ecall { pc: BASE + 4 }),
            (EBREAK, Trap::Ebreak { pc: BASE + 4 }),
            (load(3, 12, 0, 0x10), Trap::LoadFault { pc: BASE + 4, fault }),
            (store(3, 0, 10, 0x10), Trap::StoreFault { pc: BASE + 4, fault }),
            (
                0xffff_ffff,
                Trap::IllegalInstruction { pc: BASE + 4, err: DecodeError::Illegal(0xffff_ffff) },
            ),
        ];
        for (w, want) in cases {
            let (mut cpu, mem) = setup(&[addi(10, 0, 1), w]);
            let before = {
                step(&mut cpu, &mem).unwrap();
                cpu.clone()
            };
            assert_eq!(step(&mut cpu, &mem), Err(want));
            assert_eq!(cpu, before);
        }

        let (_, mem) = setup(&[]);
        let mut cpu = CpuState::new(0x10);
        assert_eq!(step(&mut cpu, &mem), Err(Trap::FetchFault(MemFault { addr: 0x10 })));
    }

    #[test]
    fn zba_shifted_adds() {
        // shNadd: (rs1 << N) + rs2, wrapping.
        assert_eq!(one(op(0b010, 0b001_0000, 10, 11, 12), 5, 7), 17);
        assert_eq!(one(op(0b100, 0b001_0000, 10, 11, 12), 5, 7), 27);
        assert_eq!(one(op(0b110, 0b001_0000, 10, 11, 12), 5, 7), 47);
        assert_eq!(one(op(0b010, 0b001_0000, 10, 11, 12), 1 << 63, 1), 1);
        // The .uw forms use only the low word of rs1 but keep rs2 full width, and the
        // sum is 64-bit.
        assert_eq!(one(op_32(0b010, 0b001_0000, 10, 11, 12), 0xffff_ffff_0000_0003, 2), 8);
        assert_eq!(one(op_32(0b100, 0b001_0000, 10, 11, 12), 3, 0), 12);
        assert_eq!(one(op_32(0b110, 0b001_0000, 10, 11, 12), 3, 0), 24);
        assert_eq!(
            one(op_32(0b000, 0b000_0100, 10, 11, 12), 0xffff_ffff_ffff_ffff, 0x10),
            0x1_0000_000f
        );
        // slli.uw shifts the zero-extended low word by 0..63 (the sixth shamt bit is
        // part of funct7, so the immediate is funct6 000010 above the shamt).
        let slliuw = |shamt: i32| 0b000_010 << 6 | shamt;
        assert_eq!(one(op_imm_32(0b001, 10, 11, slliuw(0)), 0xffff_ffff_ffff_ffff, 0), 0xffff_ffff);
        assert_eq!(
            one(op_imm_32(0b001, 10, 11, slliuw(40)), 0xffff_ffff, 0),
            0xffff_ff00_0000_0000
        );
        assert_eq!(one(op_imm_32(0b001, 10, 11, slliuw(63)), 3, 0), 0x8000_0000_0000_0000);
    }

    #[test]
    fn zbb_registers() {
        assert_eq!(one(op(0b111, 0b010_0000, 10, 11, 12), 12, 10), 4);
        assert_eq!(one(op(0b110, 0b010_0000, 10, 11, 12), 12, 10), (!10) | 12);
        assert_eq!(one(op(0b100, 0b010_0000, 10, 11, 12), 12, 10), !6);
        // min/max are signed; minu/maxu are unsigned.
        assert_eq!(one(op(0b110, 0b000_0101, 10, 11, 12), NEG1, 1), 1);
        assert_eq!(one(op(0b111, 0b000_0101, 10, 11, 12), NEG1, 1), NEG1);
        assert_eq!(one(op(0b100, 0b000_0101, 10, 11, 12), NEG1, 1), NEG1);
        assert_eq!(one(op(0b101, 0b000_0101, 10, 11, 12), NEG1, 1), 1);
        // Rotates mask the amount to 6 bits.
        assert_eq!(one(op(0b001, 0b011_0000, 10, 11, 12), 0x8000_0000_0000_0001, 1), 3);
        assert_eq!(one(op(0b001, 0b011_0000, 10, 11, 12), 0x8000_0000_0000_0001, 65), 3);
        assert_eq!(one(op(0b101, 0b011_0000, 10, 11, 12), 1, 1), 0x8000_0000_0000_0000);
        // rori (funct6 011000 above the 6-bit shamt).
        assert_eq!(one(op_imm_selector(0b101, 0b011_000 << 6 | 60, 10, 11), 1, 0), 16);
    }

    #[test]
    fn zbb_unary() {
        // clz/ctz/cpop: funct6 011000 with rs2 0/1/2.
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6, 10, 11), 0, 0), 64);
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6, 10, 11), 1, 0), 63);
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6 | 1, 10, 11), 0, 0), 64);
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6 | 1, 10, 11), 1 << 63, 0), 63);
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6 | 2, 10, 11), 0, 0), 0);
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6 | 2, 10, 11), NEG1, 0), 64);
        // sext.b / sext.h (rs2 4/5).
        assert_eq!(one(op_imm_selector(0b001, 0b011_000 << 6 | 4, 10, 11), 0xff, 0), NEG1);
        assert_eq!(
            one(op_imm_selector(0b001, 0b011_000 << 6 | 4, 10, 11), 0x80, 0),
            -128i64 as u64
        );
        assert_eq!(
            one(op_imm_selector(0b001, 0b011_000 << 6 | 5, 10, 11), 0xff80, 0),
            -128i64 as u64
        );
        // zext.h zero-extends the low halfword (64-bit result).
        assert_eq!(one(op_32(0b100, 0b000_0100, 10, 11, 0), NEG1, 0), 0xffff);
        // rev8 byte-swaps (one byte swap on a 64-bit register).
        assert_eq!(
            one(op_imm_selector(0b101, 0b011_010 << 6 | 0b11000, 10, 11), 0x0102_0304_0506_0708, 0),
            0x0807_0605_0403_0201
        );
        // orc.b sets a byte to 0xff when it is nonzero (rs2 00111 in funct6 001010).
        assert_eq!(
            one(op_imm_selector(0b101, 0b001_010 << 6 | 0b00111, 10, 11), 0x0100_0002_0000_0000, 0),
            0xff00_00ff_0000_0000
        );
    }

    #[test]
    fn zbs_single_bits() {
        // bset/bclr/binv/bext; the bit index is rs2 & 63.
        assert_eq!(one(op(0b001, 0b001_0100, 10, 11, 12), 0, 63), 1 << 63);
        assert_eq!(one(op(0b001, 0b010_0100, 10, 11, 12), 1 << 63, 63), 0);
        assert_eq!(one(op(0b001, 0b011_0100, 10, 11, 12), 0, 63), 1 << 63);
        assert_eq!(one(op(0b001, 0b011_0100, 10, 11, 12), 1 << 63, 63), 0);
        assert_eq!(one(op(0b101, 0b010_0100, 10, 11, 12), 2, 65), 1);
        assert_eq!(one(op(0b101, 0b010_0100, 10, 11, 12), 2, 1), 1);
        // Immediate forms take a 6-bit bit index (funct6 above the shamt).
        assert_eq!(one(op_imm_selector(0b001, 0b001_010 << 6 | 63, 10, 11), 0, 0), 1 << 63);
        assert_eq!(one(op_imm_selector(0b001, 0b010_010 << 6 | 63, 10, 11), NEG1, 0), !(1 << 63));
        assert_eq!(one(op_imm_selector(0b101, 0b010_010 << 6 | 20, 10, 11), 1 << 20, 0), 1);
        assert_eq!(one(op_imm_selector(0b001, 0b011_010 << 6 | 33, 10, 11), 0, 0), 1 << 33);
        assert_eq!(one(op_imm_selector(0b001, 0b001_010 << 6 | 42, 10, 11), 0, 0), 1 << 42);
    }

    #[test]
    fn zicond_selects() {
        assert_eq!(one(op(0b101, 0b000_0111, 10, 11, 12), 5, 0), 0);
        assert_eq!(one(op(0b101, 0b000_0111, 10, 11, 12), 5, 9), 5);
        assert_eq!(one(op(0b111, 0b000_0111, 10, 11, 12), 5, 9), 0);
        assert_eq!(one(op(0b111, 0b000_0111, 10, 11, 12), 5, 0), 5);
    }

    #[test]
    fn zbkb_packs_and_brev8() {
        // brev8 reverses the bits within each byte.
        assert_eq!(
            one(op_imm_selector(0b101, 0b011_010 << 6 | 0b00111, 10, 11), 0x1234_5678_9abc_def0, 0),
            0x482c_6a1e_593d_7b0f
        );
        // pack puts the low word of each source together, packh the low bytes...
        assert_eq!(
            one(op(0b100, 0b000_0100, 10, 11, 12), 0xdead_beef_ffff_1234, 0xffff_0000_0000_5678),
            0x5678_ffff_1234
        );
        assert_eq!(one(op(0b111, 0b000_0100, 10, 11, 12), 0x00ab, 0x00cd), 0xcdab);
        // ...and packw sign-extends the word it packs (bit 31 comes from rs2's low 16).
        assert_eq!(
            one(op_32(0b100, 0b000_0100, 10, 11, 12), 0x1234, 0x8000),
            0xffff_ffff_8000_1234
        );
    }

    #[test]
    fn zbc_carry_less_multiplies() {
        // 0x3 × 0x5 without carries is 0xf, entirely in the low word.
        assert_eq!(one(op(0b001, 0b000_0101, 10, 11, 12), 0x3, 0x5), 0xf);
        assert_eq!(one(op(0b011, 0b000_0101, 10, 11, 12), 0x3, 0x5), 0);
        // 2^63 × 2^63 = 2^126: clmul sees nothing, clmulh bit 62, and clmulr bit 63.
        let top = 1u64 << 63;
        assert_eq!(one(op(0b001, 0b000_0101, 10, 11, 12), top, top), 0);
        assert_eq!(one(op(0b011, 0b000_0101, 10, 11, 12), top, top), 1 << 62);
        assert_eq!(one(op(0b010, 0b000_0101, 10, 11, 12), top, top), 1 << 63);
    }

    #[test]
    fn zbkx_permutes_elements() {
        // xperm8: byte i of rs2 indexes the bytes of the table in rs1.
        assert_eq!(
            one(op(0b100, 0b001_0100, 10, 11, 12), 0x8877_6655_4433_2211, 0x0706_0504_0302_0100),
            0x8877_6655_4433_2211
        );
        // An index past the last element selects zero rather than wrapping.
        assert_eq!(
            one(op(0b100, 0b001_0100, 10, 11, 12), 0x8877_6655_4433_2211, 0x08),
            0x1111_1111_1111_1100,
            "byte 0 indexes past the table (zero); the rest index table byte 0"
        );
        // xperm4: nibble i of rs2 indexes the table in rs1 (0x0123_4567_89ab_cdef has
        // nibble 0 = 0xf counting up).
        assert_eq!(
            one(op(0b010, 0b001_0100, 10, 11, 12), 0x0123_4567_89ab_cdef, 0x3210),
            0xffff_ffff_ffff_cdef
        );
    }

    #[test]
    fn zbb_word_rotates() {
        // rolw/rorw rotate the low word and sign-extend the result.
        assert_eq!(one(op_32(0b001, 0b011_0000, 10, 11, 12), 1, 31), 0xffff_ffff_8000_0000);
        assert_eq!(one(op_32(0b101, 0b011_0000, 10, 11, 12), 1, 1), 0xffff_ffff_8000_0000);
        assert_eq!(one(op_32(0b101, 0b011_0000, 10, 11, 12), 1, 33), 0xffff_ffff_8000_0000);
        assert_eq!(one(op_imm_32(0b101, 10, 11, 0b011_000 << 6 | 31), 1, 0), 2);
    }

    #[test]
    fn vector_merge_picks_the_masked_source() {
        // Hand-assembled helpers for the pieces this test needs.
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vlm = |vd: u32, rs1: u32| (1 << 25) | (0b01011 << 20) | (rs1 << 15) | (vd << 7) | 0x07;
        let vmerge = |vd: u32, vs2: u32, vs1: u32| {
            (0b010111 << 26) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        // vsetivli zero, 2, e64, m1, ta, ma
        let prog = [
            vsetivli(0, 2, 0x0d8),
            vle64(1, 11),    // v1 = src A, two 8-byte elements from (a1)
            vle64(2, 12),    // v2 = src B from (a2)
            vlm(0, 13),      // v0 = mask byte from (a3)
            vmerge(3, 2, 1), // vmerge.vvm v3, v2, v1, v0
            vse64(3, 14),    // store v3 to (a4)
        ];
        let mem = FlatMemory::new(BASE, 4096);
        mem.load_words(BASE, &prog).unwrap();
        mem.write(DATA, &0xaaaa_aaaa_aaaa_aaaau64.to_le_bytes()).unwrap();
        mem.write(DATA + 8, &0xbbbb_bbbb_bbbb_bbbbu64.to_le_bytes()).unwrap();
        mem.write(DATA + 16, &0x1111_1111_1111_1111u64.to_le_bytes()).unwrap();
        mem.write(DATA + 24, &0x2222_2222_2222_2222u64.to_le_bytes()).unwrap();
        mem.write(DATA + 32, &[0b01]).unwrap(); // mask: element 0 only
        let mut cpu = CpuState::new(BASE);
        for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, DATA + 48)] {
            cpu.set_x(r, v);
        }
        let end = BASE + 4 * prog.len() as u64;
        while cpu.pc != end {
            step(&mut cpu, &mem).unwrap_or_else(|t| panic!("{t:?}\n{cpu:?}"));
        }
        // The mask selects vs1 (v1 = src A) for element 0 and vs2 (v2) otherwise.
        let mut out = [0u8; 16];
        mem.read(DATA + 48, &mut out).unwrap();
        assert_eq!(u64::from_le_bytes(out[..8].try_into().unwrap()), 0xaaaa_aaaa_aaaa_aaaa);
        assert_eq!(u64::from_le_bytes(out[8..].try_into().unwrap()), 0x2222_2222_2222_2222);
    }

    #[test]
    fn zcb_aliases() {
        // Raw words from tests/fixtures/rv64zb.s: c.zext.w a4, c.not a5, c.zext.b a0,
        // c.sext.h a3, c.mul a0, a1. Each 2-byte word is paired with a c.nop so the
        // program fills its whole 4-byte word.
        let pair = |h: u32| h | 0x0001 << 16;
        assert_eq!(exec(&[pair(0x9f71)], &[(14, NEG1)]).0.x[14], 0xffff_ffff);
        assert_eq!(exec(&[pair(0x9ff5)], &[(15, 0)]).0.x[15], NEG1);
        assert_eq!(exec(&[pair(0x9d61)], &[(10, 0x1234)]).0.x[10], 0x34);
        assert_eq!(exec(&[pair(0x9eed)], &[(13, 0xff80)]).0.x[13], -128i64 as u64);
        assert_eq!(exec(&[pair(0x9d4d)], &[(10, 6), (11, 7)]).0.x[10], 42);
    }
}
