//! Translate a decoded basic block into arm64.
//!
//! Register convention inside generated code:
//!
//! | host | use |
//! | --- | --- |
//! | x0–x13, x19–x26, x29 | 23 guest registers, pinned (see [`HOST_REG`]) |
//! | x14, x15 | scratch for unpinned sources |
//! | x16 | scratch for unpinned destinations; exit code on the way to the exit stub |
//! | x17 | scratch for addresses and jump targets; caches a base register's host address |
//! | x18 | untouched (reserved by macOS) |
//! | x27 | host address of guest memory (`GuestMemory::host_base`) |
//! | x28 | `&mut CpuState` |
//! | x30 | `CpuState::instret` |
//! | sp | host stack |
//!
//! Generated code never calls, so it needs neither a frame pointer nor a link register and
//! takes both x29 and x30. Keeping `instret` in a register matters: as a memory counter, its
//! load-add-store at every block entry forms a store-to-load forwarding chain through all
//! blocks, several cycles per block.
//!
//! The pinned set was chosen by measuring dynamic register use on a C benchmark, Lua, and
//! SQLite: a0–a7, sp, ra, and s0–s9 are common to all three, and t0–t2 carry SQLite's hot
//! string and memory loops. s10, s11, t3–t6, gp, and tp are each about 1% or less.
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
use crate::memory::{CONTROL_OFFSET, GUARD_SIZE, GUEST_SIZE};
const _: () = assert!(CONTROL_OFFSET.is_multiple_of(4096) && CONTROL_OFFSET >> 12 < 4096);

pub const CTX: Reg = 28;
/// Host address of guest address 0 (`GuestMemory::host_base`) in the offset layout. In the
/// identity layout, guest addresses are host addresses and this holds the control area
/// (`GuestMemory::control_area`) instead, which the interrupt check then reaches in one
/// load.
pub const MEMBASE: Reg = 27;
const SRC0: Reg = 14;
const SRC1: Reg = 15;
const DST: Reg = 16;
const ADDR: Reg = 17;
/// Register carrying the exit code into the exit stub.
pub const EXIT_CODE: Reg = 16;
/// Register holding `CpuState::instret` while generated code runs.
pub const INSTRET_REG: Reg = 30;

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
    while j < 6 {
        map[20 + j] = Some(19 + j as Reg); // s4–s9 -> x19–x24
        j += 1;
    }
    map[5] = Some(25); // t0
    map[6] = Some(26); // t1
    map[7] = Some(29); // t2
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
    /// The thread's interrupt flag was set (a signal to deliver); continue at `pc`.
    pub const INTERRUPT: u64 = 6;
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
///
/// Accesses through a base register mask the base, not the sum: `ADDR` gets the host
/// address `MEMBASE + (rs1 & ADDR_MASK)` (just the masked address in the identity layout),
/// and the access adds its offset as an immediate.
/// The offset (under 32 KiB, or down to -256) can then step just past either end of the
/// guest space, into the guard region that [`GUARD_SIZE`] reserves there, and fault. While
/// the base register is not written, later accesses through it reuse `ADDR`: a run of
/// stack accesses costs one instruction each, plus two for the first.
const ADDR_MASK: u64 = GUEST_SIZE - 1;
const _: () = assert!(GUARD_SIZE >= 4096 * 8 + 256);

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

#[derive(Clone, Copy)]
enum Tail {
    /// Continue at a static pc with an exit code.
    Static { pc: u64, code: u64 },
    /// Leave mid-block to interpret the instruction at `pc`, of which `retired_before`
    /// instructions of this block had retired (instret is corrected accordingly).
    Interpret { pc: u64, retired_before: u32 },
    /// A taken conditional branch in the middle of a block, of which `retired_before`
    /// instructions had retired (including the branch): correct instret, then a chainable
    /// jump to `pc`, after an interrupt check if it jumps `back`.
    Side { pc: u64, retired_before: u32, back: bool },
    /// Continue at the pc computed into `ADDR` with an exit code (a jump-cache miss, or an
    /// interrupt before an indirect jump).
    Dynamic { code: u64 },
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
    /// Byte offset of the jump cache from the start of the code buffer.
    jump_cache: i64,
    /// Word index of this block in the code buffer.
    block_word: i64,
    /// Chainable exits inside tails (side exits): instruction index and target pc.
    tail_exits: Vec<(usize, Option<u64>)>,
    /// Guest register whose host address (see [`ADDR_MASK`]) is in `ADDR`, if any.
    addr_base: Option<u8>,
    /// Whether the access being translated has its base register reused by a later one.
    base_reused: bool,
    /// The interrupt flag this thread's code checks (see [`Tx::interrupt_check`]).
    interrupt_slot: u32,
    /// Whether guest memory uses the identity layout (see [`MEMBASE`]).
    identity: bool,
}

fn fp(w: FWidth) -> Fp {
    if w == FWidth::S { Fp::S } else { Fp::D }
}

fn sext(imm: i32) -> u64 {
    imm as i64 as u64
}

impl Tx {
    /// Turn the masked guest address in `ADDR` into a host address.
    fn host_address(&mut self) {
        if !self.identity {
            self.a.add(ADDR, MEMBASE, ADDR);
        }
    }

    /// Base and index registers of a register-offset access to the masked guest address in
    /// `ADDR`: `[MEMBASE, ADDR]`, or `[ADDR, xzr]` in the identity layout.
    fn base_reg(&self) -> Reg {
        if self.identity { ADDR } else { MEMBASE }
    }

    fn index_reg(&self) -> Reg {
        if self.identity { XZR } else { ADDR }
    }

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

    /// Prepare a `size`-byte access to `rs1 + imm`. Returns the immediate offset to use from
    /// `ADDR` (which holds the base register's host address, see [`ADDR_MASK`]), or `None`
    /// to use a register-offset access to the masked guest address in `ADDR`.
    fn mem_operand(&mut self, rs1: u8, imm: i32, size: u32) -> Option<i32> {
        let cached = self.addr_base == Some(rs1);
        // With a zero offset and no reuse, the masked address alone is one instruction
        // shorter than setting up the base.
        if rs1 != 0 && Asm::imm_offset_fits(size, imm) && (cached || imm != 0 || self.base_reused) {
            if !cached {
                let base = self.src(rs1, SRC0);
                self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
                self.host_address();
                self.addr_base = Some(rs1);
            }
            return Some(imm);
        }
        self.address(rs1, imm);
        self.addr_base = None;
        None
    }

    /// Integer load into `d` through an operand from [`Tx::mem_operand`].
    fn load_int(&mut self, mem: Mem, d: Reg, operand: Option<i32>) {
        match operand {
            Some(off) => assert!(self.a.ldr_imm(mem, d, ADDR, off)),
            None => self.a.ldr_reg(mem, d, self.base_reg(), self.index_reg()),
        }
    }

    /// Leave the body through a `b` to a tail.
    fn exit(&mut self, tail: Tail) {
        let at = self.a.pos();
        self.a.b(at); // patched once the tail exists
        self.tails.push((at, tail));
    }

    /// Jump to the guest pc in `ADDR` through the jump cache: on a hit, straight to the
    /// cached host code; on a miss, exit to the dispatcher (which fills the slot).
    /// `check` adds an interrupt check (see [`Tx::interrupt_check`]).
    fn indirect(&mut self, check: bool) {
        if check {
            self.interrupt_check(None);
        }
        self.a.ubfm(DST, ADDR, 1, JUMP_CACHE_BITS); // index = pc[16:1]
        // The jump cache is page-aligned in the code buffer's data area.
        let here = (self.block_word + self.a.pos() as i64) * 4;
        self.a.adrp(SRC0, ((self.jump_cache >> 12) - (here >> 12)) as i32);
        self.a.add_lsl(DST, SRC0, DST, 4);
        self.a.ldr_off(SRC0, DST, 0);
        self.a.ldr_off(SRC1, DST, 8);
        self.a.cmp(SRC0, ADDR);
        let miss = self.a.pos();
        self.a.b_cond(Cond::Ne, miss); // patched once the tail exists
        self.tails.push((miss, Tail::Dynamic { code: exit::NEXT }));
        self.a.br(SRC1);
    }

    /// Leave with `exit::INTERRUPT` (continuing at `target`) if another thread asked this
    /// one to (a signal to deliver). Emitted before backward direct jumps and before
    /// indirect jumps other than returns: a cycle of forward jumps is impossible, and a
    /// return only closes a cycle together with a call, which is a backward jump (recursion)
    /// or an indirect one. So every loop passes a check, even a tight one, while forward
    /// edges and returns stay free.
    fn interrupt_check(&mut self, target: Option<u64>) {
        if self.identity {
            self.a.ldr_off_w(DST, MEMBASE, self.interrupt_slot * 4);
        } else {
            self.a.sub_imm_lsl12(DST, MEMBASE, (CONTROL_OFFSET >> 12) as u32);
            self.a.ldr_off_w(DST, DST, self.interrupt_slot * 4);
        }
        let check = self.a.pos();
        self.a.cbnz(DST, check); // patched once the tail exists
        let tail = match target {
            Some(pc) => Tail::Static { pc, code: exit::INTERRUPT },
            None => Tail::Dynamic { code: exit::INTERRUPT }, // target in ADDR
        };
        self.tails.push((check, tail));
    }

    /// Translate one non-control-flow instruction. Returns false if unsupported.
    fn simple(&mut self, inst: Inst, pc: u64) -> bool {
        match inst {
            Inst::Lui { rd, imm } => self.constant(rd, sext(imm)),
            Inst::Auipc { rd, imm } => self.constant(rd, pc.wrapping_add(sext(imm))),
            Inst::OpImm { op, rd, rs1, imm } => {
                if !self.op_imm(op, rd, rs1, imm) {
                    return false;
                }
            }
            Inst::OpImm32 { op, rd, rs1, imm } => {
                if !self.op_imm32(op, rd, rs1, imm) {
                    return false;
                }
            }
            Inst::Op { op, rd, rs1, rs2 } => {
                if !self.op(op, rd, rs1, rs2) {
                    return false;
                }
            }
            Inst::Op32 { op, rd, rs1, rs2 } => {
                if !self.op32(op, rd, rs1, rs2) {
                    return false;
                }
            }
            Inst::Load { width, rd, rs1, offset } => {
                let mem = match width {
                    LoadWidth::B => Mem::S8,
                    LoadWidth::H => Mem::S16,
                    LoadWidth::W => Mem::S32,
                    LoadWidth::D => Mem::U64,
                    LoadWidth::Bu => Mem::U8,
                    LoadWidth::Hu => Mem::U16,
                    LoadWidth::Wu => Mem::U32,
                };
                let operand = self.mem_operand(rs1, offset, mem.size());
                // A load to x0 still accesses memory (and may fault).
                let d = self.dst(rd);
                self.accesses.push((self.a.pos(), pc, self.retired, false));
                match operand {
                    Some(off) => assert!(self.a.ldr_imm(mem, d, ADDR, off)),
                    None => self.a.ldr_reg(mem, d, self.base_reg(), self.index_reg()),
                }
                self.commit(rd, d);
            }
            Inst::Store { width, rs1, rs2, offset } => {
                let v = self.src(rs2, SRC1);
                let mem = match width {
                    StoreWidth::B => Mem::U8,
                    StoreWidth::H => Mem::U16,
                    StoreWidth::W => Mem::U32,
                    StoreWidth::D => Mem::U64,
                };
                let operand = self.mem_operand(rs1, offset, mem.size());
                self.accesses.push((self.a.pos(), pc, self.retired, true));
                match operand {
                    Some(off) => assert!(self.a.str_imm(mem, v, ADDR, off)),
                    None => self.a.str_reg(mem, v, self.base_reg(), self.index_reg()),
                }
            }
            // Conservatively a full barrier; mapping pred/succ to dmb variants comes later.
            Inst::Fence { .. } => self.a.dmb_ish(),
            // F/D data movement. Doubles are moved as raw bits; singles are NaN-boxed when
            // written to an f register.
            Inst::FLoad { width, rd, rs1, offset } => {
                let size = if width == FWidth::D { 8 } else { 4 };
                let operand = self.mem_operand(rs1, offset, size);
                self.accesses.push((self.a.pos(), pc, self.retired, false));
                match (width, FHOST_REG[rd as usize]) {
                    (FWidth::D, Some(h)) => match operand {
                        Some(off) => assert!(self.a.ldr_fp_imm(Fp::D, h, ADDR, off)),
                        None => self.a.ldr_fp_reg(Fp::D, h, self.base_reg(), self.index_reg()),
                    },
                    (FWidth::D, None) => {
                        self.load_int(Mem::U64, DST, operand);
                        self.a.str_off(DST, CTX, f_offset(rd));
                    }
                    (FWidth::S, _) => {
                        self.load_int(Mem::U32, DST, operand);
                        self.a.orr_imm(DST, DST, NAN_BOX).unwrap();
                        self.fraw_store(rd, DST);
                    }
                }
            }
            Inst::FStore { width, rs1, rs2, offset } => {
                let size = fp(width);
                let bytes = if width == FWidth::D { 8 } else { 4 };
                match FHOST_REG[rs2 as usize] {
                    Some(h) => {
                        let operand = self.mem_operand(rs1, offset, bytes);
                        self.accesses.push((self.a.pos(), pc, self.retired, true));
                        // A single store writes the low 32 bits, boxed or not.
                        match operand {
                            Some(off) => assert!(self.a.str_fp_imm(size, h, ADDR, off)),
                            None => self.a.str_fp_reg(size, h, self.base_reg(), self.index_reg()),
                        }
                    }
                    None => {
                        self.a.ldr_off(SRC1, CTX, f_offset(rs2));
                        let operand = self.mem_operand(rs1, offset, bytes);
                        self.accesses.push((self.a.pos(), pc, self.retired, true));
                        let mem = if width == FWidth::D { Mem::U64 } else { Mem::U32 };
                        match operand {
                            Some(off) => assert!(self.a.str_imm(mem, SRC1, ADDR, off)),
                            None => self.a.str_reg(mem, SRC1, self.base_reg(), self.index_reg()),
                        }
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
                self.a.ldr_reg(mem, d, self.base_reg(), self.index_reg());
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
                self.host_address();
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
        self.host_address();
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

    fn op_imm(&mut self, op: AluOp, rd: u8, rs1: u8, imm: i32) -> bool {
        if rd == 0 {
            return true; // no side effects
        }
        let v = sext(imm);
        if rs1 == 0 {
            // x0 is zero, so the whole result is a constant.
            let r = match op {
                AluOp::Add | AluOp::Xor | AluOp::Or => v,
                AluOp::Slt => (0 < imm as i64) as u64,
                AluOp::Sltu => (0 < v) as u64,
                // Zba/Zbb/Zbs/Zicond immediate forms are interpreted only for now.
                AluOp::SllUw
                | AluOp::Clz
                | AluOp::Ctz
                | AluOp::Cpop
                | AluOp::SextB
                | AluOp::SextH
                | AluOp::ZextH
                | AluOp::Rev8
                | AluOp::OrcB
                | AluOp::Ror
                | AluOp::Bclr
                | AluOp::Bext
                | AluOp::Binv
                | AluOp::Bset => return false,
                _ => 0, // and, shifts of zero
            };
            self.constant(rd, r);
            return true;
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
            // Zba/Zbb/Zbs/Zicond immediate forms are interpreted only for now.
            AluOp::SllUw
            | AluOp::Clz
            | AluOp::Ctz
            | AluOp::Cpop
            | AluOp::SextB
            | AluOp::SextH
            | AluOp::ZextH
            | AluOp::Rev8
            | AluOp::OrcB
            | AluOp::Ror
            | AluOp::Bclr
            | AluOp::Bext
            | AluOp::Binv
            | AluOp::Bset => return false,
            _ => unreachable!("no immediate form of {op:?}"),
        }
        self.commit(rd, d);
        true
    }

    fn op_imm32(&mut self, op: AluOp32, rd: u8, rs1: u8, imm: i32) -> bool {
        if rd == 0 {
            return true;
        }
        if rs1 == 0 {
            if matches!(op, AluOp32::Clz | AluOp32::Ctz | AluOp32::Cpop) {
                return false; // clzw(0) = 32, not a foldable zero
            }
            let r = if op == AluOp32::Add { sext(imm) } else { 0 };
            self.constant(rd, r);
            return true;
        }
        let d = self.dst(rd);
        let a = self.src(rs1, SRC0);
        match op {
            AluOp32::Add if imm >= 0 => self.a.add_imm_w(d, a, imm as u32),
            AluOp32::Add => self.a.sub_imm_w(d, a, imm.unsigned_abs()),
            AluOp32::Sll => self.a.lsl_imm_w(d, a, imm as u32),
            AluOp32::Srl => self.a.lsr_imm_w(d, a, imm as u32),
            AluOp32::Sra => self.a.asr_imm_w(d, a, imm as u32),
            // Zbb word rotates and counts are interpreted only for now.
            AluOp32::Ror | AluOp32::Clz | AluOp32::Ctz | AluOp32::Cpop => return false,
            _ => unreachable!("no immediate form of {op:?}"),
        }
        self.a.sxtw(d, d);
        self.commit(rd, d);
        true
    }

    fn op(&mut self, op: AluOp, rd: u8, rs1: u8, rs2: u8) -> bool {
        if rd == 0 {
            return true; // M-extension ops never trap, so nothing is observable
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
            // Zba/Zbb/Zbs/Zicond (the immediate-only variants never reach here) are
            // interpreted only for now; the block falls back as a whole.
            AluOp::Sh1Add
            | AluOp::Sh2Add
            | AluOp::Sh3Add
            | AluOp::AddUw
            | AluOp::Sh1AddUw
            | AluOp::Sh2AddUw
            | AluOp::Sh3AddUw
            | AluOp::SllUw
            | AluOp::Andn
            | AluOp::Orn
            | AluOp::Xnor
            | AluOp::Max
            | AluOp::Maxu
            | AluOp::Min
            | AluOp::Minu
            | AluOp::Rol
            | AluOp::Ror
            | AluOp::Bclr
            | AluOp::Bext
            | AluOp::Binv
            | AluOp::Bset
            | AluOp::CzeroEqz
            | AluOp::CzeroNez
            | AluOp::Clz
            | AluOp::Ctz
            | AluOp::Cpop
            | AluOp::SextB
            | AluOp::SextH
            | AluOp::ZextH
            | AluOp::Rev8
            | AluOp::OrcB => return false,
        }
        self.commit(rd, d);
        true
    }

    fn op32(&mut self, op: AluOp32, rd: u8, rs1: u8, rs2: u8) -> bool {
        if rd == 0 {
            return true;
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
            // `cmp` below cannot test x0: register 31 in an immediate compare is wsp.
            AluOp32::Div | AluOp32::Divu if rs2 == 0 => self.a.mov_imm(d, u64::MAX),
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
            // Zbb word rotations and counts are interpreted only for now.
            AluOp32::Rol | AluOp32::Ror | AluOp32::Clz | AluOp32::Ctz | AluOp32::Cpop => {
                return false;
            }
        }
        self.a.sxtw(d, d);
        self.commit(rd, d);
        true
    }

    /// Write the link register of `jal`/`jalr`.
    fn link(&mut self, rd: u8, value: u64) {
        self.constant(rd, value);
    }

    fn emit_tails(&mut self) {
        // Tails may add tails (a side exit's interrupt check), so iterate by index.
        let mut i = 0;
        while i < self.tails.len() {
            let (branch, tail) = self.tails[i];
            i += 1;
            let here = self.a.pos();
            self.a.patch(branch, here);
            match tail {
                Tail::Static { pc, code } => {
                    self.a.mov_imm(ADDR, pc);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, code as u16, 0);
                }
                Tail::Side { pc, retired_before, back } => {
                    let undo = self.retired - retired_before;
                    if undo > 0 {
                        self.a.sub_imm(INSTRET_REG, INSTRET_REG, undo);
                    }
                    if back {
                        self.interrupt_check(Some(pc));
                    }
                    // Chained later; until then it falls through into the exit below.
                    let b = self.a.pos();
                    self.a.b(b + 1);
                    self.tail_exits.push((b, Some(pc)));
                    self.a.mov_imm(ADDR, pc);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, exit::NEXT as u16, 0);
                }
                Tail::Interpret { pc, retired_before } => {
                    // The block added all its instructions to instret at entry.
                    self.a.sub_imm(INSTRET_REG, INSTRET_REG, self.retired - retired_before);
                    self.a.mov_imm(ADDR, pc);
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, exit::INTERPRET as u16, 0);
                }
                Tail::Dynamic { code } => {
                    self.a.str_off(ADDR, CTX, PC);
                    self.a.movz(EXIT_CODE, code as u16, 0);
                }
            }
            // Branch to the shared exit stub; its offset is fixed up by `translate`.
            let at = self.a.pos();
            self.a.b(at);
            self.stub_branches.push(at);
        }
    }
}

/// Base register of a load or store (the accesses that go through [`Tx::mem_operand`]).
fn mem_base(inst: Inst) -> Option<u8> {
    match inst {
        Inst::Load { rs1, .. }
        | Inst::Store { rs1, .. }
        | Inst::FLoad { rs1, .. }
        | Inst::FStore { rs1, .. } => Some(rs1),
        _ => None,
    }
}

/// Whether translating `inst` leaves `ADDR` alone, apart from [`Tx::mem_operand`], so the
/// base-register cache survives it (unless it writes the base register, see [`int_dest`]).
fn keeps_addr(inst: Inst) -> bool {
    use AluOp::{Div, Divu, Mulhsu, Rem, Remu};
    match inst {
        Inst::Lui { .. }
        | Inst::Auipc { .. }
        | Inst::OpImm { .. }
        | Inst::OpImm32 { .. }
        | Inst::Branch { .. } => true,
        Inst::Op { op, .. } => !matches!(op, Mulhsu | Div | Divu | Rem | Remu),
        Inst::Op32 { op, .. } => {
            !matches!(op, AluOp32::Div | AluOp32::Divu | AluOp32::Rem | AluOp32::Remu)
        }
        _ => mem_base(inst).is_some(),
    }
}

/// The integer register written by an instruction that [`keeps_addr`].
fn int_dest(inst: Inst) -> Option<u8> {
    match inst {
        Inst::Lui { rd, .. }
        | Inst::Auipc { rd, .. }
        | Inst::OpImm { rd, .. }
        | Inst::OpImm32 { rd, .. }
        | Inst::Op { rd, .. }
        | Inst::Op32 { rd, .. }
        | Inst::Load { rd, .. } => Some(rd),
        _ => None,
    }
}

/// For each instruction of `block`: whether it is an access whose base register a later
/// access reuses, with nothing in between that drops the cached base.
fn base_reuse(block: &[(Inst, u8)]) -> Vec<bool> {
    let mut reused = vec![false; block.len()];
    for (i, &(inst, _)) in block.iter().enumerate() {
        let Some(base) = mem_base(inst) else { continue };
        if int_dest(inst) == Some(base) {
            continue;
        }
        for &(later, _) in &block[i + 1..] {
            if mem_base(later) == Some(base) {
                reused[i] = true;
                break;
            }
            if !keeps_addr(later) || int_dest(later) == Some(base) || mem_base(later).is_some() {
                break;
            }
        }
    }
    reused
}

/// Translate `block` (as decoded at `pc`) into code that will live at word index
/// `block_word` of the code buffer, whose exit stub is at word index `stub_word`. Returns
/// `None` if not even the first instruction can be translated. Indirect jumps probe the
/// jump cache at byte offset `jump_cache` in the code buffer. Backward and indirect jumps
/// check interrupt flag `interrupt_slot` (see [`GuestMemory::interrupt_flag`]).
///
/// [`GuestMemory::interrupt_flag`]: crate::memory::GuestMemory::interrupt_flag
pub fn translate(
    block: &[(Inst, u8)],
    pc: u64,
    block_word: i64,
    stub_word: i64,
    jump_cache: i64,
    interrupt_slot: u32,
    identity: bool,
) -> Option<Translation> {
    let mut tx = Tx {
        a: Asm::new(),
        tails: Vec::new(),
        stub_branches: Vec::new(),
        accesses: Vec::new(),
        retired: 0,
        jump_cache,
        block_word,
        tail_exits: Vec::new(),
        addr_base: None,
        base_reused: false,
        interrupt_slot,
        identity,
    };
    // instret: add the number of instructions this block retires, up front. Every exit is
    // at the end of the block, so the count is static; this stays right under chaining.
    let instret_at = tx.a.pos();
    tx.a.add_imm(INSTRET_REG, INSTRET_REG, 0); // patched below

    let mut retired = 0u32;
    let mut at = pc;
    let mut exits = Vec::new();
    let mut ended = false;
    // The register the previous instruction set to a constant (`lui`, `auipc`), and its value.
    let mut known: Option<(u8, u64)> = None;
    let reused = base_reuse(block);
    for (i, &(inst, len)) in block.iter().enumerate() {
        tx.base_reused = reused[i];
        let len = len as u64;
        let next = at + len;
        match inst {
            Inst::Jal { rd, offset } => {
                tx.link(rd, next);
                retired += 1;
                let target = at.wrapping_add(sext(offset));
                if target <= at {
                    tx.interrupt_check(Some(target));
                }
                exits.push((tx.a.pos(), Some(target)));
                tx.exit(Tail::Static { pc: target, code: exit::NEXT });
                ended = true;
                break;
            }
            Inst::Jalr { rd, rs1, offset } => {
                if let Some((_, base)) = known.filter(|&(r, _)| r == rs1) {
                    // `auipc`/`lui` + `jalr` (a far call or tail call): a direct jump that
                    // chains like `jal`.
                    let target = base.wrapping_add(sext(offset)) & !1;
                    tx.link(rd, next);
                    retired += 1;
                    if target <= at {
                        tx.interrupt_check(Some(target));
                    }
                    exits.push((tx.a.pos(), Some(target)));
                    tx.exit(Tail::Static { pc: target, code: exit::NEXT });
                    ended = true;
                    break;
                }
                // Target first: rd may be rs1.
                if offset == 0 && rs1 != 0 {
                    let base = tx.src(rs1, SRC0);
                    tx.a.and_imm(ADDR, base, !1).unwrap();
                } else {
                    tx.add_imm_to(ADDR, rs1, offset);
                    tx.a.and_imm(ADDR, ADDR, !1).unwrap();
                }
                tx.link(rd, next);
                retired += 1;
                tx.indirect(!(rd == 0 && rs1 == 1)); // no check on `ret`
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
                let target = at.wrapping_add(sext(offset));
                if i + 1 < block.len() {
                    // Not the end of the block: fall through into the rest of it, and
                    // leave through a side exit when taken.
                    let bcond = tx.a.pos();
                    tx.a.b_cond(c, bcond); // patched once the tail exists
                    let side =
                        Tail::Side { pc: target, retired_before: retired, back: target <= at };
                    tx.tails.push((bcond, side));
                    tx.retired = retired;
                    known = None;
                    at = next;
                    continue;
                }
                // b.cond taken; b fallthrough-tail; taken: b taken-tail
                let bcond = tx.a.pos();
                tx.a.b_cond(c, bcond + 2);
                exits.push((tx.a.pos(), Some(next)));
                tx.exit(Tail::Static { pc: next, code: exit::NEXT });
                if target <= at {
                    tx.interrupt_check(Some(target));
                }
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
        if !keeps_addr(inst) || int_dest(inst).is_some_and(|rd| Some(rd) == tx.addr_base) {
            tx.addr_base = None;
        }
        known = match inst {
            Inst::Lui { rd, imm } if rd != 0 => Some((rd, sext(imm))),
            Inst::Auipc { rd, imm } if rd != 0 => Some((rd, at.wrapping_add(sext(imm)))),
            _ => None,
        };
        at = next;
    }
    if !ended {
        // The block stopped at a length or page limit: fall through to the next one.
        exits.push((tx.a.pos(), Some(at)));
        tx.exit(Tail::Static { pc: at, code: exit::NEXT });
    }
    tx.retired = retired;
    tx.a.code[instret_at] = {
        let mut a = Asm::new();
        a.add_imm(INSTRET_REG, INSTRET_REG, retired);
        a.code[0]
    };
    tx.emit_tails();
    exits.append(&mut tx.tail_exits);
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
    a.ldr_off(INSTRET_REG, CTX, INSTRET);
    a.br(ADDR);

    // Exit stub: generated code branches here with the exit code in EXIT_CODE.
    let stub = a.pos();
    a.str_off(INSTRET_REG, CTX, INSTRET);
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
