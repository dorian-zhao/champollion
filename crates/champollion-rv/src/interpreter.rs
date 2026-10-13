//! Reference interpreter. Slow on purpose: it is the oracle the JIT is diffed against.

use crate::cpu::CpuState;
use crate::decoder::{
    self, AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, DecodeError, FWidth, Inst, LoadWidth,
    SgnjKind, StoreWidth,
};
use crate::memory::{MemFault, Memory};

/// Why execution stopped. On every trap, `cpu.pc` still points at the trapping instruction
/// and no architectural state has been modified, like `sepc` on real hardware.
///
/// - `Ecall`: system call request. The handler reads `a7`/`a0..a5`, writes `a0`, and
///   resumes at `pc + 4` (`ecall` has no compressed form).
/// - `MisalignedAtomic`: `lr`/`sc`/AMO on an address not aligned to its width; Linux
///   delivers SIGBUS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    FetchFault(MemFault),
    LoadFault { pc: u64, fault: MemFault },
    StoreFault { pc: u64, fault: MemFault },
    IllegalInstruction { pc: u64, err: DecodeError },
    Ecall { pc: u64 },
    Ebreak { pc: u64 },
    MisalignedAtomic { pc: u64, addr: u64 },
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
        Inst::Ecall => return Err(Trap::Ecall { pc }),
        Inst::Ebreak => return Err(Trap::Ebreak { pc }),
        // Atomics map onto host atomics (sequentially consistent, which covers every aq/rl
        // combination). lr/sc use a value-based reservation and compare-and-swap.
        Inst::Lr { width, rd, rs1, .. } => {
            let addr = atomic_addr(cpu, rs1, width)?;
            let v = load_value(mem, addr, amo_load_width(width))
                .map_err(|fault| Trap::LoadFault { pc, fault })?;
            cpu.set_x(rd, v);
            cpu.reservation = Some((addr, v));
        }
        Inst::Sc { width, rd, rs1, rs2, .. } => {
            let addr = atomic_addr(cpu, rs1, width)?;
            let size = amo_bytes(width);
            let mask = if size == 4 { 0xffff_ffff } else { u64::MAX };
            let stored = match cpu.reservation {
                Some((a, v)) if a == addr => mem
                    .compare_exchange(addr, size, v & mask, cpu.x(rs2) & mask)
                    .map_err(|fault| Trap::StoreFault { pc, fault })?,
                _ => false,
            };
            cpu.set_x(rd, if stored { 0 } else { 1 });
            cpu.reservation = None;
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
    }
    cpu.pc = next;
    Ok(())
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
    }
}

/// 32-bit ALU for the `*w` instructions: uses the low 32 bits of the operands and the low
/// 5 bits of the shift amount, then sign-extends the 32-bit result.
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
        assert_eq!(cpu.reservation, None);

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
        assert_eq!(cpu.reservation, Some((DATA, 0xffff_ffff_8000_0000)));
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
}
