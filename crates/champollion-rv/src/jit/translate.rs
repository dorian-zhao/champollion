//! Translate a decoded basic block into arm64.
//!
//! Register convention inside generated code:
//!
//! | host | use |
//! | --- | --- |
//! | x0–x13, x19–x26 | 22 guest registers, pinned (see [`HOST_REG`]) |
//! | x14, x15 | scratch for unpinned sources |
//! | x16 | scratch for unpinned destinations; exit code on the way to the exit stub |
//! | x17 | scratch for addresses and jump targets |
//! | x18 | untouched (reserved by macOS) |
//! | x27 | host address of guest memory (`GuestMemory::host_base`) |
//! | x28 | `&mut CpuState` |
//! | x29, x30, sp | host frame, link register, and stack |
//!
//! The pinned set was chosen by measuring dynamic register use on a C benchmark, Lua, and
//! SQLite: a0–a7, sp, ra, and s0–s11 cover all three; t0–t6, gp, and tp were each under 1%.
//!
//! Block layout: an `instret` update, the translated body, then one out-of-line *tail* per
//! exit. Every exit leaves the body through a plain `b` to its tail, so a later chaining step
//! can repoint that `b` straight at the next block.

use std::mem::offset_of;

use super::arm64::{Asm, AtomicOp, Cond, FmaKind, Fp, FpOp, FpRound, Mem, Reg, XZR};
use crate::cpu::CpuState;
use crate::decoder::{
    AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, FArithOp, FCmpOp, FWidth, FmaOp, Inst, IntType,
    LoadWidth, RM_DYN, SgnjKind, StoreWidth,
};
use crate::memory::GUEST_SIZE;

pub const CTX: Reg = 28;
pub const MEMBASE: Reg = 27;
const SRC0: Reg = 14;
const SRC1: Reg = 15;
const DST: Reg = 16;
const ADDR: Reg = 17;
/// Register carrying the exit code into the exit stub.
pub const EXIT_CODE: Reg = 16;

/// Host register holding each guest register, or `None` if it lives in [`CpuState`].
pub const HOST_REG: [Option<Reg>; 32] = {
    let mut map = [None; 32];
    let mut i = 0;
    while i < 8 {
        map[10 + i] = Some(i as Reg); // a0–a7 -> x0–x7
        i += 1;
    }
    map[1] = Some(8); // ra
    map[2] = Some(9); // sp
    map[8] = Some(10); // s0
    map[9] = Some(11); // s1
    map[18] = Some(12); // s2
    map[19] = Some(13); // s3
    let mut j = 0;
    while j < 8 {
        map[20 + j] = Some(19 + j as Reg); // s4–s11 -> x19–x26
        j += 1;
    }
    map
};

/// Exit codes returned by generated code to the dispatcher. The guest pc to continue at
/// has been stored in `CpuState::pc`.
pub mod exit {
    /// Continue at `pc` (look up or translate the next block).
    pub const NEXT: u64 = 0;
    /// `ecall` at `pc`.
    pub const ECALL: u64 = 1;
    /// The instruction at `pc` is not translated; run it on the interpreter.
    pub const INTERPRET: u64 = 2;
    /// `fence.i` retired; drop all translations, then continue at `pc`.
    pub const FENCE_I: u64 = 3;
    /// A load or store at `pc` faulted (set by the fault handler, not by generated code).
    pub const FAULT_LOAD: u64 = 4;
    pub const FAULT_STORE: u64 = 5;
}

pub fn x_offset(r: u8) -> u32 {
    (offset_of!(CpuState, x) + 8 * r as usize) as u32
}
const PC: u32 = offset_of!(CpuState, pc) as u32;
const INSTRET: u32 = offset_of!(CpuState, instret) as u32;
const RESERVED_ADDR: u32 = offset_of!(CpuState, reserved_addr) as u32;
const FCSR: u32 = offset_of!(CpuState, fcsr) as u32;
/// Host v-register holding each guest f register, or `None` if it lives in `CpuState`.
/// f0–f15 -> v0–v15 and f16–f27 -> v20–v31; ft8–ft11 (f28–f31) stay in memory. v8–v15 are
/// callee-saved in the host ABI (their low halves), so the trampoline preserves them.
pub const FHOST_REG: [Option<Reg>; 32] = {
    let mut map = [None; 32];
    let mut i = 0;
    while i < 28 {
        map[i] = Some(if i < 16 { i as Reg } else { i as Reg + 4 });
        i += 1;
    }
    map
};

/// FP scratch registers (caller-saved, so the trampoline need not preserve them).
const FA: Reg = 16;
const FB: Reg = 17;
const FC: Reg = 18;
const FR: Reg = 19;

/// `FPCR` for running guest code with RISC-V rounding mode `frm` (0–3): default-NaN mode
/// plus the matching ARM RMode. `None` for RMM (4) and reserved values, which generated
/// code cannot honor; the dispatcher interprets instead.
pub fn guest_fpcr(frm: u32) -> Option<u64> {
    const DN: u64 = 1 << 25;
    let rmode = match frm {
        0 => 0b00, // RNE -> RN
        1 => 0b11, // RTZ -> RZ
        2 => 0b10, // RDN -> RM
        3 => 0b01, // RUP -> RP
        _ => return None,
    };
    Some(DN | rmode << 22)
}
const RESERVED_VALUE: u32 = offset_of!(CpuState, reserved_value) as u32;
/// Upper half of a NaN-boxed single-precision value.
const NAN_BOX: u64 = 0xffff_ffff_0000_0000;
const SIGN64: u64 = 1 << 63;

fn f_offset(r: u8) -> u32 {
    (offset_of!(CpuState, f) + 8 * r as usize) as u32
}
/// Guest addresses are masked to the guest address space, so generated code can never touch
/// host memory outside the reservation, whatever address the guest computes.
const ADDR_MASK: u64 = GUEST_SIZE - 1;

/// A way out of a translated block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitSite {
    /// Index (in instructions) of the `b` that leaves the body for this exit's tail.
    pub branch: usize,
    /// The guest pc this exit always continues at, if known statically (chainable).
    pub target: Option<u64>,
}

pub struct Translation {
    pub code: Vec<u32>,
    pub exits: Vec<ExitSite>,
    /// Loads and stores that may fault: instruction index and what to report.
    pub sites: Vec<(usize, super::fault::Site)>,
}

enum Tail {
    /// Continue at a static pc with an exit code.
    Static { pc: u64, code: u64 },
    /// Leave mid-block to interpret the instruction at `pc`, of which `retired_before`
    /// instructions of this block had retired (instret is corrected accordingly).
    Interpret { pc: u64, retired_before: u32 },
    /// Continue at the pc computed into `ADDR`: try the jump cache, else exit.
    Indirect,
}

/// Number of entries in the jump cache (a power of two): `(guest pc, host code address)`
/// pairs indexed by guest pc bits 1..=16. 4096 entries thrashed on Lua (600k misses for 3.4k
/// blocks, from conflicting targets); 65536 (1 MiB) leaves only cold misses.
pub const JUMP_CACHE_ENTRIES: usize = 65536;
const JUMP_CACHE_BITS: u32 = JUMP_CACHE_ENTRIES.trailing_zeros();

/// Jump-cache slot for a guest pc.
pub fn jump_cache_index(pc: u64) -> usize {
    ((pc >> 1) as usize) & (JUMP_CACHE_ENTRIES - 1)
}

struct Tx {
    a: Asm,
    /// (index of the body's `b`, tail) pairs, emitted after the body.
    tails: Vec<(usize, Tail)>,
    /// Indices of the tails' branches to the shared exit stub, fixed up by `translate`.
    stub_branches: Vec<usize>,
    /// Memory accesses: (instruction index, guest pc, instructions retired before, store).
    accesses: Vec<(usize, u64, u32, bool)>,
    /// Instructions of this block retired so far (while translating).
    retired: u32,
    /// Host address of the jump cache.
    jump_cache: u64,
}

fn fp(w: FWidth) -> Fp {
    if w == FWidth::S { Fp::S } else { Fp::D }
}

fn sext(imm: i32) -> u64 {
    imm as i64 as u64
}

impl Tx {
    /// Host register holding guest `r`'s value, loading it into `scratch` if unpinned.
    fn src(&mut self, r: u8, scratch: Reg) -> Reg {
        match (r, HOST_REG[r as usize]) {
            (0, _) => XZR,
            (_, Some(h)) => h,
            (_, None) => {
                self.a.ldr_off(scratch, CTX, x_offset(r));
                scratch
            }
        }
    }

    /// Host register to compute guest `r`'s new value into.
    fn dst(&self, r: u8) -> Reg {
        if r == 0 { DST } else { HOST_REG[r as usize].unwrap_or(DST) }
    }

    /// Store a result computed into `h` back to `CpuState` if `r` is unpinned.
    fn commit(&mut self, r: u8, h: Reg) {
        if r != 0 && HOST_REG[r as usize].is_none() {
            self.a.str_off(h, CTX, x_offset(r));
        }
    }

    /// `rd = (rs1 + imm)` into `out` (which may be any scratch); handles rs1 = x0.
    fn add_imm_to(&mut self, out: Reg, rs1: u8, imm: i32) {
        if rs1 == 0 {
            self.a.mov_imm(out, sext(imm));
            return;
        }
        let base = self.src(rs1, SRC0);
        if imm >= 0 {
            self.a.add_imm(out, base, imm as u32);
        } else {
            self.a.sub_imm(out, base, imm.unsigned_abs());
        }
    }

    /// Compute the host offset of guest address `rs1 + imm` into `ADDR`.
    fn address(&mut self, rs1: u8, imm: i32) {
        if rs1 == 0 {
            self.a.mov_imm(ADDR, sext(imm) & ADDR_MASK);
            return;
        }
        if imm == 0 {
            let base = self.src(rs1, SRC0);
            self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
        } else {
            self.add_imm_to(ADDR, rs1, imm);
            self.a.and_imm(ADDR, ADDR, ADDR_MASK).unwrap();
        }
    }

    /// Leave the body through a `b` to a tail.
    fn exit(&mut self, tail: Tail) {
        let at = self.a.pos();
        self.a.b(at); // patched once the tail exists
        self.tails.push((at, tail));
    }

    /// Translate one non-control-flow instruction. Returns false if unsupported.
    fn simple(&mut self, inst: Inst, pc: u64) -> bool {
        match inst {
            Inst::Lui { rd, imm } => self.constant(rd, sext(imm)),
            Inst::Auipc { rd, imm } => self.constant(rd, pc.wrapping_add(sext(imm))),
            Inst::OpImm { op, rd, rs1, imm } => self.op_imm(op, rd, rs1, imm),
            Inst::OpImm32 { op, rd, rs1, imm } => self.op_imm32(op, rd, rs1, imm),
            Inst::Op { op, rd, rs1, rs2 } => self.op(op, rd, rs1, rs2),
            Inst::Op32 { op, rd, rs1, rs2 } => self.op32(op, rd, rs1, rs2),
            Inst::Load { width, rd, rs1, offset } => {
                self.address(rs1, offset);
                let mem = match width {
                    LoadWidth::B => Mem::S8,
                    LoadWidth::H => Mem::S16,
                    LoadWidth::W => Mem::S32,
                    LoadWidth::D => Mem::U64,
                    LoadWidth::Bu => Mem::U8,
                    LoadWidth::Hu => Mem::U16,
                    LoadWidth::Wu => Mem::U32,
                };
                // A load to x0 still accesses memory (and may fault).
                let d = self.dst(rd);
                self.accesses.push((self.a.pos(), pc, self.retired, false));
                self.a.ldr_reg(mem, d, MEMBASE, ADDR);
                self.commit(rd, d);
            }
            Inst::Store { width, rs1, rs2, offset } => {
                let v = self.src(rs2, SRC1);
                self.address(rs1, offset);
                let mem = match width {
                    StoreWidth::B => Mem::U8,
                    StoreWidth::H => Mem::U16,
                    StoreWidth::W => Mem::U32,
                    StoreWidth::D => Mem::U64,
                };
                self.accesses.push((self.a.pos(), pc, self.retired, true));
                self.a.str_reg(mem, v, MEMBASE, ADDR);
            }
            // Conservatively a full barrier; mapping pred/succ to dmb variants comes later.
            Inst::Fence { .. } => self.a.dmb_ish(),
            // F/D data movement. Doubles are moved as raw bits; singles are NaN-boxed when
            // written to an f register.
            Inst::FLoad { width, rd, rs1, offset } => {
                self.address(rs1, offset);
                self.accesses.push((self.a.pos(), pc, self.retired, false));
                match (width, FHOST_REG[rd as usize]) {
                    (FWidth::D, Some(h)) => self.a.ldr_fp_reg(Fp::D, h, MEMBASE, ADDR),
                    (FWidth::D, None) => {
                        self.a.ldr_reg(Mem::U64, DST, MEMBASE, ADDR);
                        self.a.str_off(DST, CTX, f_offset(rd));
                    }
                    (FWidth::S, _) => {
                        self.a.ldr_reg(Mem::U32, DST, MEMBASE, ADDR);
                        self.a.orr_imm(DST, DST, NAN_BOX).unwrap();
                        self.fraw_store(rd, DST);
                    }
                }
            }
            Inst::FStore { width, rs1, rs2, offset } => {
                let size = fp(width);
                match FHOST_REG[rs2 as usize] {
                    Some(h) => {
                        self.address(rs1, offset);
                        self.accesses.push((self.a.pos(), pc, self.retired, true));
                        // A single store writes the low 32 bits, boxed or not.
                        self.a.str_fp_reg(size, h, MEMBASE, ADDR);
                    }
                    None => {
                        self.a.ldr_off(SRC1, CTX, f_offset(rs2));
                        self.address(rs1, offset);
                        self.accesses.push((self.a.pos(), pc, self.retired, true));
                        let mem = if width == FWidth::D { Mem::U64 } else { Mem::U32 };
                        self.a.str_reg(mem, SRC1, MEMBASE, ADDR);
                    }
                }
            }
            Inst::FMvFromInt { width, rd, rs1 } => {
                let v = self.src(rs1, SRC0);
                if width == FWidth::D {
                    self.fraw_store(rd, v);
                } else {
                    self.a.orr_imm(DST, v, NAN_BOX).unwrap();
                    self.fraw_store(rd, DST);
                }
            }
            Inst::FMvToInt { width, rd, rs1 } => {
                if rd != 0 {
                    let d = self.dst(rd);
                    match (width, FHOST_REG[rs1 as usize]) {
                        (FWidth::D, _) => self.fraw_load(rs1, d),
                        (FWidth::S, Some(h)) => self.a.fmov_to_gpr(Fp::S, d, h),
                        (FWidth::S, None) => self.a.ldr_off_w(d, CTX, f_offset(rs1)),
                    }
                    if width == FWidth::S {
                        self.a.sxtw(d, d);
                    }
                    self.commit(rd, d);
                }
            }
            // fmv.d / fneg.d / fabs.d (rs1 == rs2) are single sign-bit instructions, which,
            // like RISC-V, never touch NaN payloads or raise exceptions.
            Inst::FSgnj { kind, width: FWidth::D, rd, rs1, rs2 } if rs1 == rs2 => {
                let a = self.fsrc(FWidth::D, rs1, FA);
                let d = self.fdest(FWidth::D, rd);
                match kind {
                    SgnjKind::Sgnj => self.a.fmov_d(d, a),
                    SgnjKind::Sgnjn => self.a.fneg_d(d, a),
                    SgnjKind::Sgnjx => self.a.fabs_d(d, a),
                }
                self.fcommit(FWidth::D, rd, d);
            }
            Inst::FSgnj { kind, width: FWidth::D, rd, rs1, rs2 } => {
                // Single precision needs NaN-box checks; it stays on the interpreter for now.
                self.fraw_load(rs1, SRC0);
                self.fraw_load(rs2, SRC1);
                match kind {
                    SgnjKind::Sgnj => {}
                    SgnjKind::Sgnjn => self.a.eor_imm(SRC1, SRC1, SIGN64).unwrap(),
                    SgnjKind::Sgnjx => self.a.eor(SRC1, SRC0, SRC1),
                }
                self.a.and_imm(SRC1, SRC1, SIGN64).unwrap();
                self.a.and_imm(SRC0, SRC0, !SIGN64).unwrap();
                self.a.orr(DST, SRC0, SRC1);
                self.fraw_store(rd, DST);
            }
            Inst::Lr { width, rd, rs1, .. } => {
                let base = self.src(rs1, SRC0);
                self.require_aligned(base, width, pc);
                // Keep the address (rd may be rs1) and record the reservation only after the
                // load, so a faulting lr leaves the state untouched.
                self.a.mov(SRC1, base);
                self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
                let d = self.dst(rd);
                self.accesses.push((self.a.pos(), pc, self.retired, false));
                let mem = if width == AmoWidth::D { Mem::U64 } else { Mem::S32 };
                self.a.ldr_reg(mem, d, MEMBASE, ADDR);
                self.a.str_off(SRC1, CTX, RESERVED_ADDR);
                self.a.str_off(d, CTX, RESERVED_VALUE);
                self.commit(rd, d);
            }
            Inst::Sc { width, rd, rs1, rs2, .. } => self.sc(width, rd, rs1, rs2, pc),
            // FP arithmetic runs on the host FPU under the guest's FPCR, which the trampoline
            // set from frm, so only `dyn` rounding (what compilers emit) is translated;
            // exceptions accumulate in FPSR and fold into fflags at exit.
            Inst::FArith { op, width, rd, rs1, rs2, rm: RM_DYN } => {
                let a = self.fsrc(width, rs1, FA);
                let b = self.fsrc(width, rs2, FB);
                let op = match op {
                    FArithOp::Add => FpOp::Add,
                    FArithOp::Sub => FpOp::Sub,
                    FArithOp::Mul => FpOp::Mul,
                    FArithOp::Div => FpOp::Div,
                };
                let d = self.fdest(width, rd);
                self.a.fp_arith(op, fp(width), d, a, b);
                self.fcommit(width, rd, d);
            }
            Inst::FSqrt { width, rd, rs1, rm: RM_DYN } => {
                let a = self.fsrc(width, rs1, FA);
                let d = self.fdest(width, rd);
                self.a.fsqrt(fp(width), d, a);
                self.fcommit(width, rd, d);
            }
            Inst::FFma { op, width, rd, rs1, rs2, rs3, rm: RM_DYN } => {
                let a = self.fsrc(width, rs1, FA);
                let b = self.fsrc(width, rs2, FB);
                let c = self.fsrc(width, rs3, FC);
                // ARM names the negated forms the other way around (see fpu::fma).
                let kind = match op {
                    FmaOp::Madd => FmaKind::Fmadd,
                    FmaOp::Msub => FmaKind::Fnmsub,
                    FmaOp::Nmsub => FmaKind::Fmsub,
                    FmaOp::Nmadd => FmaKind::Fnmadd,
                };
                let d = self.fdest(width, rd);
                self.a.fma(kind, fp(width), d, a, b, c);
                self.fcommit(width, rd, d);
            }
            Inst::FCmp { op, width, rd, rs1, rs2 } => {
                let a = self.fsrc(width, rs1, FA);
                let b = self.fsrc(width, rs2, FB);
                // fcmp is quiet like feq; fcmpe signals on any NaN like flt/fle.
                self.a.fcmp(op != FCmpOp::Eq, fp(width), a, b);
                if rd != 0 {
                    let d = self.dst(rd);
                    let cond = match op {
                        FCmpOp::Eq => Cond::Eq,
                        FCmpOp::Lt => Cond::Mi,
                        FCmpOp::Le => Cond::Ls,
                    };
                    self.a.cset(d, cond);
                    self.commit(rd, d);
                }
            }
            // fcvt.d.s is exact, so its rounding mode never matters.
            Inst::FCvtFloat { to: FWidth::D, rd, rs1, .. }
            | Inst::FCvtFloat { to: FWidth::S, rd, rs1, rm: RM_DYN } => {
                let to = if matches!(inst, Inst::FCvtFloat { to: FWidth::D, .. }) {
                    FWidth::D
                } else {
                    FWidth::S
                };
                let from = if to == FWidth::D { FWidth::S } else { FWidth::D };
                let a = self.fsrc(from, rs1, FA);
                let d = self.fdest(to, rd);
                self.a.fcvt(fp(to), d, a);
                self.fcommit(to, rd, d);
            }
            // Float to integer: the rounding is part of the ARM opcode, so every static mode
            // (RMM included) is available; dyn would need frm at translation time.
            Inst::FCvtToInt { from, to, rd, rs1, rm } if rm != RM_DYN => {
                let round = match rm {
                    0 => FpRound::Nearest,
                    1 => FpRound::Zero,
                    2 => FpRound::Down,
                    3 => FpRound::Up,
                    _ => FpRound::Away,
                };
                let a = self.fsrc(from, rs1, FA);
                let signed = matches!(to, IntType::W | IntType::L);
                let wide = matches!(to, IntType::L | IntType::Lu);
                self.a.fcvt_to_int(round, signed, wide, fp(from), SRC0, a);
                // NaN converts to the largest integer on RISC-V, to 0 on ARM.
                let max = match to {
                    IntType::W => i32::MAX as u64,
                    IntType::Wu => u32::MAX as u64,
                    IntType::L => i64::MAX as u64,
                    IntType::Lu => u64::MAX,
                };
                self.a.fcmp(false, fp(from), a, a);
                self.a.mov_imm(SRC1, max);
                let d = self.dst(rd);
                self.a.csel(d, SRC1, SRC0, Cond::Vs);
                if !wide {
                    self.a.sxtw(d, d); // w and wu results are sign-extended
                }
                self.commit(rd, d);
            }
            // Integer to float rounds per FPCR (dyn), except int32 -> double, which is exact.
            Inst::FCvtFromInt { from, to, rd, rs1, rm }
                if rm == RM_DYN
                    || (to == FWidth::D && matches!(from, IntType::W | IntType::Wu)) =>
            {
                let v = self.src(rs1, SRC0);
                let signed = matches!(from, IntType::W | IntType::L);
                let wide = matches!(from, IntType::L | IntType::Lu);
                let d = self.fdest(to, rd);
                self.a.cvt_from_int(signed, wide, fp(to), d, v);
                self.fcommit(to, rd, d);
            }
            Inst::Amo { op, width, rd, rs1, rs2, .. } => {
                let base = self.src(rs1, SRC0);
                self.require_aligned(base, width, pc);
                self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
                self.a.add(ADDR, MEMBASE, ADDR);
                let mut v = self.src(rs2, SRC1);
                let lse = match op {
                    AmoOp::Swap => AtomicOp::Swp,
                    AmoOp::Add => AtomicOp::Add,
                    AmoOp::Xor => AtomicOp::Eor,
                    AmoOp::Or => AtomicOp::Set,
                    AmoOp::Min => AtomicOp::Smin,
                    AmoOp::Max => AtomicOp::Smax,
                    AmoOp::Minu => AtomicOp::Umin,
                    AmoOp::Maxu => AtomicOp::Umax,
                    AmoOp::And => {
                        // LSE has bit-clear, not and: [m] & v == [m] & !(!v).
                        self.a.mvn(SRC1, v);
                        v = SRC1;
                        AtomicOp::Clr
                    }
                };
                let d = self.dst(rd);
                let wide = width == AmoWidth::D;
                self.accesses.push((self.a.pos(), pc, self.retired, true));
                self.a.atomic(lse, wide, v, if rd == 0 { XZR } else { d }, ADDR);
                if rd != 0 {
                    if !wide {
                        self.a.sxtw(d, d); // word AMOs sign-extend the old value
                    }
                    self.commit(rd, d);
                }
            }
            _ => return false,
        }
        true
    }

    /// Host v-register holding `f[r]` as an operand of width `w`, loading it into `scratch`
    /// if needed. Singles must be properly NaN-boxed; anything else reads as the canonical
    /// NaN, as the spec requires, so singles always go through the check.
    fn fsrc(&mut self, w: FWidth, r: u8, scratch: Reg) -> Reg {
        match (w, FHOST_REG[r as usize]) {
            (FWidth::D, Some(h)) => h,
            (FWidth::D, None) => {
                self.a.ldr_fp(Fp::D, scratch, CTX, f_offset(r));
                scratch
            }
            (FWidth::S, _) => {
                self.fraw_load(r, SRC0);
                self.a.lsr_imm(SRC1, SRC0, 32);
                self.a.cmn_imm_w(SRC1, 1); // upper half == 0xffffffff?
                self.a.mov_imm(ADDR, 0x7fc0_0000);
                self.a.csel(SRC0, SRC0, ADDR, Cond::Eq);
                self.a.fmov_from_gpr(Fp::S, scratch, SRC0);
                scratch
            }
        }
    }

    /// Host v-register to compute `f[r]`'s new value of width `w` into.
    fn fdest(&self, w: FWidth, r: u8) -> Reg {
        match (w, FHOST_REG[r as usize]) {
            (FWidth::D, Some(h)) => h,
            _ => FR,
        }
    }

    /// Finish writing `f[r]` from `v` (as returned by `fdest`): store unpinned doubles, and
    /// NaN-box singles.
    fn fcommit(&mut self, w: FWidth, r: u8, v: Reg) {
        match (w, FHOST_REG[r as usize]) {
            (FWidth::D, Some(_)) => {}
            (FWidth::D, None) => self.a.str_fp(Fp::D, v, CTX, f_offset(r)),
            (FWidth::S, _) => {
                self.a.fmov_to_gpr(Fp::S, SRC0, v);
                self.a.orr_imm(SRC0, SRC0, NAN_BOX).unwrap();
                self.fraw_store(r, SRC0);
            }
        }
    }

    /// Copy `f[r]`'s raw 64 bits into general register `g`.
    fn fraw_load(&mut self, r: u8, g: Reg) {
        match FHOST_REG[r as usize] {
            Some(h) => self.a.fmov_to_gpr(Fp::D, g, h),
            None => self.a.ldr_off(g, CTX, f_offset(r)),
        }
    }

    /// Set `f[r]` to the raw 64 bits in general register `g` (which may be xzr).
    fn fraw_store(&mut self, r: u8, g: Reg) {
        match FHOST_REG[r as usize] {
            Some(h) => self.a.fmov_from_gpr(Fp::D, h, g),
            None => self.a.str_off(g, CTX, f_offset(r)),
        }
    }

    /// Leave for the interpreter (which raises the misaligned-atomic trap) unless `base` is
    /// naturally aligned for `width`.
    fn require_aligned(&mut self, base: Reg, width: AmoWidth, pc: u64) {
        let mask = if width == AmoWidth::D { 7 } else { 3 };
        self.a.tst_imm(base, mask).unwrap();
        let skip = self.a.pos();
        self.a.b_cond(Cond::Eq, skip + 2);
        self.exit(Tail::Interpret { pc, retired_before: self.retired });
    }

    /// `sc`: succeeds (rd = 0) only if the reservation is for this address and memory still
    /// holds the reserved value, checked and stored with one `casal`.
    fn sc(&mut self, width: AmoWidth, rd: u8, rs1: u8, rs2: u8, pc: u64) {
        let wide = width == AmoWidth::D;
        let base = self.src(rs1, SRC0);
        self.require_aligned(base, width, pc);
        let d = self.dst(rd);
        self.a.ldr_off(DST, CTX, RESERVED_ADDR);
        self.a.cmp(DST, base);
        let to_fail = self.a.pos();
        self.a.b_cond(Cond::Ne, to_fail); // patched below
        self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
        self.a.add(ADDR, MEMBASE, ADDR);
        let v = self.src(rs2, SRC1);
        self.a.ldr_off(DST, CTX, RESERVED_VALUE);
        self.a.mov(SRC0, DST);
        self.accesses.push((self.a.pos(), pc, self.retired, true));
        self.a.casal(wide, SRC0, v, ADDR);
        if wide {
            self.a.cmp(SRC0, DST);
        } else {
            self.a.cmp_w(SRC0, DST);
        }
        self.a.cset(d, Cond::Ne);
        let to_done = self.a.pos();
        self.a.b(to_done); // patched below
        let fail = self.a.pos();
        self.a.patch(to_fail, fail);
        self.a.movz(d, 1, 0);
        let done = self.a.pos();
        self.a.patch(to_done, done);
        self.commit(rd, d);
        self.a.movn(SRC0, 0, 0); // NO_RESERVATION
        self.a.str_off(SRC0, CTX, RESERVED_ADDR);
    }

    fn constant(&mut self, rd: u8, v: u64) {
        if rd != 0 {
            let d = self.dst(rd);
            self.a.mov_imm(d, v);
            self.commit(rd, d);
        }
    }

    fn op_imm(&mut self, op: AluOp, rd: u8, rs1: u8, imm: i32) {
        if rd == 0 {
            return; // no side effects
        }
        let v = sext(imm);
        if rs1 == 0 {
            // x0 is zero, so the whole result is a constant.
            let r = match op {
                AluOp::Add | AluOp::Xor | AluOp::Or => v,
                AluOp::Slt => (0 < imm as i64) as u64,
                AluOp::Sltu => (0 < v) as u64,
                _ => 0, // and, shifts of zero
            };
            return self.constant(rd, r);
        }
        let d = self.dst(rd);
        let a = self.src(rs1, SRC0);
        match op {
            AluOp::Add => {
                if imm >= 0 {
                    self.a.add_imm(d, a, imm as u32);
                } else {
                    self.a.sub_imm(d, a, imm.unsigned_abs());
                }
            }
            AluOp::Slt | AluOp::Sltu => {
                if imm >= 0 {
                    self.a.cmp_imm(a, imm as u32);
                } else {
                    self.a.mov_imm(SRC1, v);
                    self.a.cmp(a, SRC1);
                }
                self.a.cset(d, if op == AluOp::Slt { Cond::Lt } else { Cond::Lo });
            }
            AluOp::And | AluOp::Or | AluOp::Xor => {
                let ok = match op {
                    AluOp::And => self.a.and_imm(d, a, v),
                    AluOp::Or => self.a.orr_imm(d, a, v),
                    _ => self.a.eor_imm(d, a, v),
                };
                if ok.is_none() {
                    // Not a bitmask immediate: materialize it.
                    self.a.mov_imm(SRC1, v);
                    match op {
                        AluOp::And => self.a.and(d, a, SRC1),
                        AluOp::Or => self.a.orr(d, a, SRC1),
                        _ => self.a.eor(d, a, SRC1),
                    }
                }
            }
            AluOp::Sll => self.a.lsl_imm(d, a, imm as u32),
            AluOp::Srl => self.a.lsr_imm(d, a, imm as u32),
            AluOp::Sra => self.a.asr_imm(d, a, imm as u32),
            _ => unreachable!("no immediate form of {op:?}"),
        }
        self.commit(rd, d);
    }

    fn op_imm32(&mut self, op: AluOp32, rd: u8, rs1: u8, imm: i32) {
        if rd == 0 {
            return;
        }
        if rs1 == 0 {
            let r = if op == AluOp32::Add { sext(imm) } else { 0 };
            return self.constant(rd, r);
        }
        let d = self.dst(rd);
        let a = self.src(rs1, SRC0);
        match op {
            AluOp32::Add if imm >= 0 => self.a.add_imm_w(d, a, imm as u32),
            AluOp32::Add => self.a.sub_imm_w(d, a, imm.unsigned_abs()),
            AluOp32::Sll => self.a.lsl_imm_w(d, a, imm as u32),
            AluOp32::Srl => self.a.lsr_imm_w(d, a, imm as u32),
            AluOp32::Sra => self.a.asr_imm_w(d, a, imm as u32),
            _ => unreachable!("no immediate form of {op:?}"),
        }
        self.a.sxtw(d, d);
        self.commit(rd, d);
    }

    fn op(&mut self, op: AluOp, rd: u8, rs1: u8, rs2: u8) {
        if rd == 0 {
            return; // M-extension ops never trap, so nothing is observable
        }
        let d = self.dst(rd);
        let a = self.src(rs1, SRC0);
        let b = self.src(rs2, SRC1);
        match op {
            AluOp::Add => self.a.add(d, a, b),
            AluOp::Sub => self.a.sub(d, a, b),
            AluOp::Xor => self.a.eor(d, a, b),
            AluOp::Or => self.a.orr(d, a, b),
            AluOp::And => self.a.and(d, a, b),
            AluOp::Sll => self.a.lslv(d, a, b),
            AluOp::Srl => self.a.lsrv(d, a, b),
            AluOp::Sra => self.a.asrv(d, a, b),
            AluOp::Slt | AluOp::Sltu => {
                self.a.cmp(a, b);
                self.a.cset(d, if op == AluOp::Slt { Cond::Lt } else { Cond::Lo });
            }
            AluOp::Mul => self.a.mul(d, a, b),
            AluOp::Mulh => self.a.smulh(d, a, b),
            AluOp::Mulhu => self.a.umulh(d, a, b),
            AluOp::Mulhsu => {
                // signed a × unsigned b: umulh(a, b) - (a < 0 ? b : 0)
                self.a.umulh(ADDR, a, b);
                self.a.asr_imm(DST, a, 63);
                self.a.and(DST, DST, b);
                self.a.sub(d, ADDR, DST);
            }
            // RISC-V: x / 0 = all ones; ARM gives 0, so patch it with csinv. Signed overflow
            // (MIN / -1 = MIN) already agrees.
            AluOp::Div | AluOp::Divu if rs2 == 0 => self.a.mov_imm(d, u64::MAX),
            AluOp::Div | AluOp::Divu => {
                if op == AluOp::Div {
                    self.a.sdiv(ADDR, a, b);
                } else {
                    self.a.udiv(ADDR, a, b);
                }
                self.a.cmp_imm(b, 0);
                self.a.csinv(d, ADDR, XZR, Cond::Ne);
            }
            // a - (a / b) * b is exactly RISC-V's remainder, including x % 0 = x (ARM's
            // division by zero gives 0) and MIN % -1 = 0.
            AluOp::Rem | AluOp::Remu => {
                if op == AluOp::Rem {
                    self.a.sdiv(ADDR, a, b);
                } else {
                    self.a.udiv(ADDR, a, b);
                }
                self.a.msub(d, ADDR, b, a);
            }
        }
        self.commit(rd, d);
    }

    fn op32(&mut self, op: AluOp32, rd: u8, rs1: u8, rs2: u8) {
        if rd == 0 {
            return;
        }
        let d = self.dst(rd);
        let a = self.src(rs1, SRC0);
        let b = self.src(rs2, SRC1);
        match op {
            AluOp32::Add => self.a.add_w(d, a, b),
            AluOp32::Sub => self.a.sub_w(d, a, b),
            AluOp32::Sll => self.a.lslv_w(d, a, b),
            AluOp32::Srl => self.a.lsrv_w(d, a, b),
            AluOp32::Sra => self.a.asrv_w(d, a, b),
            AluOp32::Mul => self.a.mul_w(d, a, b),
            AluOp32::Div | AluOp32::Divu => {
                if op == AluOp32::Div {
                    self.a.sdiv_w(ADDR, a, b);
                } else {
                    self.a.udiv_w(ADDR, a, b);
                }
                // The divisor is the low 32 bits, so test those.
                self.a.cmp_imm_w(b, 0);
                self.a.csinv_w(d, ADDR, XZR, Cond::Ne);
            }
            AluOp32::Rem | AluOp32::Remu => {
                if op == AluOp32::Rem {
                    self.a.sdiv_w(ADDR, a, b);
                } else {
                    self.a.udiv_w(ADDR, a, b);
                }
                self.a.msub_w(d, ADDR, b, a);
            }
        }
        self.a.sxtw(d, d);
        self.commit(rd, d);
    }

    /// Write the link register of `jal`/`jalr`.
    fn link(&mut self, rd: u8, value: u64) {
        self.constant(rd, value);
    }

    fn emit_tails(&mut self) {
        let tails = std::mem::take(&mut self.tails);
        for (branch, tail) in tails {
            let here = self.a.pos();
            self.a.patch(branch, here);
            match tail {
                Tail::Static { pc, code } => {
                    self.a.mov_imm(ADDR, pc);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, code as u16, 0);
                }
                Tail::Interpret { pc, retired_before } => {
                    // The block added all its instructions to instret at entry.
                    self.a.ldr_off(DST, CTX, INSTRET);
                    self.a.sub_imm(DST, DST, self.retired - retired_before);
                    self.a.str_off(DST, CTX, INSTRET);
                    self.a.mov_imm(ADDR, pc);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, exit::INTERPRET as u16, 0);
                }
                Tail::Indirect => {
                    // Inline jump-cache probe: on a hit, jump straight to the target's
                    // translation; on a miss, let the dispatcher find it (and fill the slot).
                    self.a.ubfm(DST, ADDR, 1, JUMP_CACHE_BITS); // index = pc[16:1]
                    self.a.mov_imm(SRC0, self.jump_cache);
                    self.a.add_lsl(DST, SRC0, DST, 4);
                    self.a.ldr_off(SRC0, DST, 0);
                    self.a.ldr_off(SRC1, DST, 8);
                    self.a.cmp(SRC0, ADDR);
                    let miss = self.a.pos();
                    self.a.b_cond(Cond::Ne, miss + 2);
                    self.a.br(SRC1);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, exit::NEXT as u16, 0);
                }
            }
            // Branch to the shared exit stub; its offset is fixed up by `translate`.
            let at = self.a.pos();
            self.a.b(at);
            self.stub_branches.push(at);
        }
    }
}

/// Translate `block` (as decoded at `pc`) into code that will live at word index
/// `block_word` of the code buffer, whose exit stub is at word index `stub_word`. Returns
/// `None` if not even the first instruction can be translated. Indirect jumps probe the
/// jump cache at host address `jump_cache`.
pub fn translate(
    block: &[(Inst, u8)],
    pc: u64,
    block_word: i64,
    stub_word: i64,
    jump_cache: u64,
) -> Option<Translation> {
    let mut tx = Tx {
        a: Asm::new(),
        tails: Vec::new(),
        stub_branches: Vec::new(),
        accesses: Vec::new(),
        retired: 0,
        jump_cache,
    };
    // instret: add the number of instructions this block retires, up front. Every exit is
    // at the end of the block, so the count is static; this stays right under chaining.
    let instret_at = tx.a.pos();
    tx.a.ldr_off(DST, CTX, INSTRET);
    tx.a.add_imm(DST, DST, 0); // patched below
    tx.a.str_off(DST, CTX, INSTRET);

    let mut retired = 0u32;
    let mut at = pc;
    let mut exits = Vec::new();
    let mut ended = false;
    for &(inst, len) in block {
        let len = len as u64;
        let next = at + len;
        match inst {
            Inst::Jal { rd, offset } => {
                tx.link(rd, next);
                retired += 1;
                let target = at.wrapping_add(sext(offset));
                exits.push((tx.a.pos(), Some(target)));
                tx.exit(Tail::Static { pc: target, code: exit::NEXT });
                ended = true;
                break;
            }
            Inst::Jalr { rd, rs1, offset } => {
                // Target first: rd may be rs1.
                tx.add_imm_to(ADDR, rs1, offset);
                tx.a.and_imm(ADDR, ADDR, !1).unwrap();
                tx.link(rd, next);
                retired += 1;
                exits.push((tx.a.pos(), None));
                tx.exit(Tail::Indirect);
                ended = true;
                break;
            }
            Inst::Branch { cond, rs1, rs2, offset } => {
                let a = tx.src(rs1, SRC0);
                let b = tx.src(rs2, SRC1);
                tx.a.cmp(a, b);
                let c = match cond {
                    BranchCond::Eq => Cond::Eq,
                    BranchCond::Ne => Cond::Ne,
                    BranchCond::Lt => Cond::Lt,
                    BranchCond::Ge => Cond::Ge,
                    BranchCond::Ltu => Cond::Lo,
                    BranchCond::Geu => Cond::Hs,
                };
                retired += 1;
                // b.cond taken; b fallthrough-tail; taken: b taken-tail
                let bcond = tx.a.pos();
                tx.a.b_cond(c, bcond + 2);
                exits.push((tx.a.pos(), Some(next)));
                tx.exit(Tail::Static { pc: next, code: exit::NEXT });
                let target = at.wrapping_add(sext(offset));
                exits.push((tx.a.pos(), Some(target)));
                tx.exit(Tail::Static { pc: target, code: exit::NEXT });
                ended = true;
                break;
            }
            Inst::Ecall => {
                exits.push((tx.a.pos(), None));
                tx.exit(Tail::Static { pc: at, code: exit::ECALL });
                ended = true;
                break;
            }
            Inst::FenceI => {
                retired += 1;
                exits.push((tx.a.pos(), None));
                tx.exit(Tail::Static { pc: next, code: exit::FENCE_I });
                ended = true;
                break;
            }
            _ => {
                if !tx.simple(inst, at) {
                    if retired == 0 {
                        return None;
                    }
                    exits.push((tx.a.pos(), None));
                    tx.exit(Tail::Static { pc: at, code: exit::INTERPRET });
                    ended = true;
                    break;
                }
                retired += 1;
            }
        }
        tx.retired = retired;
        at = next;
    }
    if !ended {
        // The block stopped at a length or page limit: fall through to the next one.
        exits.push((tx.a.pos(), Some(at)));
        tx.exit(Tail::Static { pc: at, code: exit::NEXT });
    }
    tx.retired = retired;
    tx.a.code[instret_at + 1] = {
        let mut a = Asm::new();
        a.add_imm(DST, DST, retired);
        a.code[0]
    };
    tx.emit_tails();
    for &b in &tx.stub_branches {
        let delta = stub_word - (block_word + b as i64);
        tx.a.code[b] = super::arm64::encode_branch_to(tx.a.code[b], delta);
    }
    let exits = exits.into_iter().map(|(branch, target)| ExitSite { branch, target }).collect();
    let sites = tx
        .accesses
        .iter()
        .map(|&(index, guest_pc, before, store)| {
            (index, super::fault::Site { guest_pc, instret_undo: retired - before, store })
        })
        .collect();
    Some(Translation { code: tx.a.code, exits, sites })
}

/// Code that enters translated code: `enter(cpu, membase, code) -> exit code`, a C-ABI
/// function. It saves the host's callee-saved registers, loads the pinned guest registers,
/// and jumps to `code`. Returns the code and the word index of the exit stub within it.
pub fn trampoline() -> (Vec<u32>, usize) {
    let mut a = Asm::new();
    a.stp_pre(29, 30, 31, -16);
    for pair in [(27, 28), (25, 26), (23, 24), (21, 22), (19, 20)] {
        a.stp_pre(pair.0, pair.1, 31, -16);
    }
    // FP state: save the host's FPCR, install the guest's (x3: default NaN + the rounding
    // mode for `frm`), and clear FPSR so it accumulates exactly this run's exceptions.
    a.mrs_fpcr(ADDR);
    a.stp_pre(ADDR, XZR, 31, -16);
    a.msr_fpcr(3);
    a.msr_fpsr(XZR);
    for pair in [(8, 9), (10, 11), (12, 13), (14, 15)] {
        a.stp_pre_d(pair.0, pair.1, 31, -16);
    }
    a.add_imm(29, 31, 0); // frame pointer = sp
    a.mov(CTX, 0);
    a.mov(MEMBASE, 1);
    a.mov(ADDR, 2);
    for (guest, host) in HOST_REG.iter().enumerate() {
        if let Some(h) = host {
            a.ldr_off(*h, CTX, x_offset(guest as u8));
        }
    }
    for (guest, host) in FHOST_REG.iter().enumerate() {
        if let Some(h) = host {
            a.ldr_fp(Fp::D, *h, CTX, f_offset(guest as u8));
        }
    }
    a.br(ADDR);

    // Exit stub: generated code branches here with the exit code in EXIT_CODE.
    let stub = a.pos();
    for (guest, host) in HOST_REG.iter().enumerate() {
        if let Some(h) = host {
            a.str_off(*h, CTX, x_offset(guest as u8));
        }
    }
    for (guest, host) in FHOST_REG.iter().enumerate() {
        if let Some(h) = host {
            a.str_fp(Fp::D, *h, CTX, f_offset(guest as u8));
        }
    }
    // Fold the exceptions raised while running into fflags. FPSR's IOC..IXC (bits 0..4) are
    // RISC-V's NV..NX in reverse order, so reverse the five bits.
    a.mrs_fpsr(ADDR);
    a.and_imm_w(ADDR, ADDR, 0x1f).unwrap();
    a.rbit_w(ADDR, ADDR);
    a.lsr_imm_w(ADDR, ADDR, 27);
    a.ldr_off_w(SRC1, CTX, FCSR);
    a.orr(SRC1, SRC1, ADDR);
    a.str_off_w(SRC1, CTX, FCSR);
    a.mov(0, EXIT_CODE);
    for pair in [(14, 15), (12, 13), (10, 11), (8, 9)] {
        a.ldp_post_d(pair.0, pair.1, 31, 16);
    }
    a.ldp_post(ADDR, XZR, 31, 16);
    a.msr_fpcr(ADDR);
    for pair in [(19, 20), (21, 22), (23, 24), (25, 26), (27, 28)] {
        a.ldp_post(pair.0, pair.1, 31, 16);
    }
    a.ldp_post(29, 30, 31, 16);
    a.ret();
    (a.code, stub)
}
