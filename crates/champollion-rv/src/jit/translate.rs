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
use std::sync::atomic::{AtomicBool, Ordering};

use super::arm64::{Asm, AtomicOp, Cond, FmaKind, Fp, FpOp, FpRound, Mem, Reg, XZR};
use crate::cpu::CpuState;
use crate::decoder::{
    AluOp, AluOp32, AmoOp, AmoWidth, BranchCond, Csr, CsrOp, FArithOp, FCmpOp, FWidth, FmaOp, Inst,
    IntType, LoadWidth, RM_DYN, SgnjKind, StoreWidth, VArithOp, VCmpOp, VFArithOp, VFMaOp, VFRedOp,
    VMaskOp, VMemKind, VRedOp, VSel, VSetKind, VStoreKind, VUnaryOp,
};
use crate::memory::{
    CONTROL_OFFSET, FAST_RANGE_COUNT, FAST_RANGE_STRIDE, FAST_RANGES_OFFSET, GUARD_SIZE, GUEST_SIZE,
};

static STAT_PASSTHROUGH: AtomicBool = AtomicBool::new(false);

/// Whether a Linux identity guest can pass `newfstatat` straight to the host.
/// This is cleared when a sysroot changes guest paths.
pub fn set_stat_passthrough(enabled: bool) {
    STAT_PASSTHROUGH.store(enabled, Ordering::Relaxed);
}

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
const VL_OFF: u32 = offset_of!(CpuState, vl) as u32;
const VTYPE_OFF: u32 = offset_of!(CpuState, vtype) as u32;

fn v_offset(v: u8) -> u32 {
    (offset_of!(CpuState, v) + 16 * v as usize) as u32
}
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

/// A backward branch internalized into its own block: the taken path re-enters the loop
/// body in place (see [`Tx::emit_edge`]) instead of leaving through a side exit, which
/// would re-run the block's entry guards (boxed singles, the vector configuration) and
/// its instret charge on every iteration.
#[derive(Clone, Copy)]
struct Edge {
    /// The `b.cond` site, patched to the landing pad.
    bcond: usize,
    /// The target instruction's index in the block.
    t: u32,
    /// Instructions one pass retires (`branch index + 1 - t`), for the instret fix-ups.
    m: u32,
    /// The loop head's guest pc: where an interrupt resumes.
    pc: u64,
    /// Word index of the loop body's first instruction (the pad's jump target).
    lt: usize,
}

/// A loop body emitted twice (see `translate`): the first copy's back edge continues
/// into the second, and only the second's leaves to the landing pad, so the pad's
/// charge/check runs once per two iterations.
struct Unroll {
    /// Index of the back-edge branch ending one pass.
    b: usize,
    /// The loop body's first instruction.
    t: usize,
    /// Fall-through jumps out of each copy, patched to the code after the bodies.
    after0: Option<usize>,
    after1: Option<usize>,
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

/// What a translated memory access is, for the fault table and the `vle*ff.v` path.
#[derive(Clone, Copy)]
enum Access {
    Load,
    Store,
    /// `vle*ff.v`: a fault shortens `vl` instead of trapping. `base` is the address
    /// register (its live value is the access's base), `eew` the element size in bytes,
    /// and `vd` the destination vector register.
    FaultOnlyFirst {
        base: u8,
        eew: u8,
        vd: u8,
    },
}

struct Tx {
    a: Asm,
    /// (index of the body's `b`, tail) pairs, emitted after the body.
    tails: Vec<(usize, Tail)>,
    /// Indices of the tails' branches to the shared exit stub, fixed up by `translate`.
    stub_branches: Vec<usize>,
    /// Memory accesses: (instruction index, guest pc, instructions retired before, store).
    accesses: Vec<(usize, u64, u32, Access)>,
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
    /// Vector configuration this translation assumes, `(sew log2, element bytes)`, when
    /// it translates vector instructions: the block's entry guard requires the guest to
    /// still be in exactly this configuration (vl == vlmax, m1) or it interprets instead.
    vec: Option<(u32, u32)>,
    /// Registers per vector group in that configuration (1 for m1, 2 for m2 - the
    /// widest the translated subset expresses). Ops that cannot span a group must
    /// refuse when this is not 1.
    vec_regs: u32,
    /// Bit `r` set: f[r] is known to hold a properly NaN-boxed single at this point of
    /// the block. The block's straight-line shape makes this exact: an `S` result boxes
    /// and a `D` result clears it, so the canonical-NaN check can be skipped for singles
    /// the block itself produced.
    fboxed: u32,
    /// The block's internalized backward edge, if any: emitted at the very end, after
    /// the body it jumps back into (see [`Tx::emit_edge`]).
    edge: Option<Edge>,
    /// The guest vector register currently held in `FA` after a translated `VArith`.
    /// Consecutive operations on the same destination can skip its next `ldr q`.
    vec_last: Option<u8>,
    /// The guest m2 vector group currently held in `FA`/`FB`, when its two chunks can
    /// survive a vector load/store or the macc family.
    vec_m2: Option<u8>,
    /// A second m2 group in `FC`/`FR`. The nanogpt-shaped macc loop loads its source
    /// and accumulator back to back, so keeping both groups resident avoids reloading
    /// both sides from `CpuState`.
    vec_m2_b: Option<u8>,
    /// Set while translating a unit m2 load whose CpuState store is provably dead
    /// because the next vector instruction overwrites the same group.
    defer_m2_store: bool,
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
        self.a.ldp(SRC0, SRC1, DST); // (guest pc, code address) in one load
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
        if !(matches!(
            inst,
            Inst::VArith { mask: false, .. }
                | Inst::VFMa { mask: false, .. }
                | Inst::VLoad { mask: false, .. }
                | Inst::VStore { mask: false, .. }
        ) || is_integer_only(inst))
        {
            self.vec_last = None;
            self.vec_m2 = None;
            self.vec_m2_b = None;
        }
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
                self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
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
                self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                match operand {
                    Some(off) => assert!(self.a.str_imm(mem, v, ADDR, off)),
                    None => self.a.str_reg(mem, v, self.base_reg(), self.index_reg()),
                }
            }
            // Conservatively a full barrier; mapping pred/succ to dmb variants comes later.
            Inst::Fence { .. } => self.a.dmb_ish(),
            Inst::Mop { rd } => self.constant(rd, 0),
            // F/D data movement. Doubles are moved as raw bits; singles are NaN-boxed when
            // written to an f register.
            Inst::FLoad { width, rd, rs1, offset } => {
                let size = if width == FWidth::D { 8 } else { 4 };
                let operand = self.mem_operand(rs1, offset, size);
                self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
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
                self.set_fboxed(rd, width == FWidth::S);
            }
            Inst::FStore { width, rs1, rs2, offset } => {
                let size = fp(width);
                let bytes = if width == FWidth::D { 8 } else { 4 };
                match FHOST_REG[rs2 as usize] {
                    Some(h) => {
                        let operand = self.mem_operand(rs1, offset, bytes);
                        self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                        // A single store writes the low 32 bits, boxed or not.
                        match operand {
                            Some(off) => assert!(self.a.str_fp_imm(size, h, ADDR, off)),
                            None => self.a.str_fp_reg(size, h, self.base_reg(), self.index_reg()),
                        }
                    }
                    None => {
                        self.a.ldr_off(SRC1, CTX, f_offset(rs2));
                        let operand = self.mem_operand(rs1, offset, bytes);
                        self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                        let mem = if width == FWidth::D { Mem::U64 } else { Mem::U32 };
                        match operand {
                            Some(off) => assert!(self.a.str_imm(mem, SRC1, ADDR, off)),
                            None => self.a.str_reg(mem, SRC1, self.base_reg(), self.index_reg()),
                        }
                    }
                }
            }
            Inst::VLoad { .. } | Inst::VStore { .. } => {
                if !self.vector_mem(inst, pc) {
                    return false;
                }
            }
            Inst::VSetvl { rd, rs1, rs2, kind: VSetKind::ImmediateAvl } => {
                if !self.vsetivli(rd, u64::from(rs1), u64::from(rs2)) {
                    return false;
                }
            }
            Inst::VSetvl { rd, rs1, rs2, kind: VSetKind::RegisterAvl } => {
                self.vsetvli(rd, rs1, u64::from(rs2));
            }
            // Reading the vector CSRs (the scanners' `csrr t, vl` in every loop).
            Inst::Csr { op: CsrOp::Rs, csr, rd, src: 0, imm: false } => {
                let d = self.dst(rd);
                match csr {
                    Csr::Frm => {
                        self.a.ldr_off_w(d, CTX, FCSR);
                        self.a.lsr_imm_w(d, d, 5);
                        self.a.and_imm_w(d, d, 0b111).unwrap();
                    }
                    Csr::Time => self.time(d),
                    Csr::Vl => self.a.ldr_off(d, CTX, VL_OFF),
                    Csr::Vtype => self.a.ldr_off(d, CTX, VTYPE_OFF),
                    Csr::Vlenb => self.a.mov_imm(d, 16),
                    _ => return false,
                }
                self.commit(rd, d);
            }
            Inst::VArith { op, vd, vs2, src, mask } => {
                if !self.vector_arith(op, vd, vs2, src, mask) {
                    return false;
                }
            }
            Inst::VCmp { op, vd, vs2, src, mask } => {
                if !self.vector_cmp(op, vd, vs2, src, mask) {
                    return false;
                }
            }
            Inst::VUnary { op, mask, vd, vs2 } => {
                if !self.vector_unary(op, mask, vd, vs2) {
                    return false;
                }
            }
            Inst::VRed { op, vd, vs1, vs2 } => {
                if !self.vector_red(op, vd, vs1, vs2) {
                    return false;
                }
            }
            Inst::VMacc { add, vd, vs2, src } => {
                if !self.vector_macc(add, vd, vs2, src) {
                    return false;
                }
            }
            Inst::VExt { sext, narrow, vd, vs2 } => {
                if !self.vector_ext(sext, u32::from(narrow), vd, vs2) {
                    return false;
                }
            }
            Inst::VWShift { mask, vd, vs2, src } => {
                if !self.vector_wshift(mask, vd, vs2, src) {
                    return false;
                }
            }
            Inst::VNarrow { sra, mask, vd, vs2, src } => {
                if !self.vector_narrow(sra, mask, vd, vs2, src) {
                    return false;
                }
            }
            Inst::VMask { op, vd, vs1, vs2 } => {
                if !self.vector_mask(op, vd, vs1, vs2) {
                    return false;
                }
            }
            Inst::VFArith { op, vd, vs2, src, mask: false } => {
                if !self.vector_farith(op, vd, vs2, src, false) {
                    return false;
                }
            }
            Inst::VFMa { op, vd, vs2, src, mask: false } => {
                if !self.vector_fma(op, vd, vs2, src, false) {
                    return false;
                }
            }
            Inst::VFRed { op, vd, vs1, vs2 } => {
                if !self.vector_fred(op, vd, vs1, vs2) {
                    return false;
                }
            }
            Inst::VFHalfConvert { widening, vd, vs2, mask: false } => {
                if !self.vector_half_convert(widening, vd, vs2, false) {
                    return false;
                }
            }
            // vfslide1down.vf: like the integer slide, dropping the float scalar into
            // the freed top element. The up form stays on the interpreter until
            // something needs it.
            Inst::VFSlide1 { down: true, vd, vs2, rs1, mask: false } => {
                if !self.vector_fslide1(vd, vs2, rs1) {
                    return false;
                }
            }
            // The float-scalar moves; the general arms below only take integer sources.
            Inst::Vmv { vd, src: VSel::F(rs1) } => {
                if !self.vector_fmv_vf(vd, rs1) {
                    return false;
                }
            }
            Inst::VFmvFS { rd, vs2 } => {
                if !self.vector_fmv_fs(rd, vs2) {
                    return false;
                }
            }
            Inst::Vfirst { rd, vs2 } => {
                if !self.vector_first(rd, vs2) {
                    return false;
                }
            }
            Inst::Vmv { vd, src } => {
                if !self.vector_mv(vd, src) {
                    return false;
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
                self.set_fboxed(rd, width == FWidth::S);
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
                // Doubles are raw 64-bit patterns: combine the halves in GPRs.
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
                self.set_fboxed(rd, false);
            }
            // fsgnj.s reads both operands with the NaN-box rule (`fsrc` substitutes the
            // canonical NaN) and injects the sign with a masked `bit`, which is exact -
            // the host's scalar sign ops would flatten a NaN payload to the default NaN.
            Inst::FSgnj { kind, width: FWidth::S, rd, rs1, rs2 } => {
                let a = self.fsrc(FWidth::S, rs1, FA);
                let b = self.fsrc(FWidth::S, rs2, FB);
                self.a.vec_movi_sign(FC);
                let d = self.fdest(FWidth::S, rd);
                match kind {
                    SgnjKind::Sgnj => {
                        self.a.fmov_d(d, a);
                        self.a.vec_bit(d, b, FC);
                        self.fcommit(FWidth::S, rd, d);
                    }
                    SgnjKind::Sgnjn => {
                        self.a.vec_eor(FB, b, FC);
                        self.a.fmov_d(d, a);
                        self.a.vec_bit(d, FB, FC);
                        self.fcommit(FWidth::S, rd, d);
                    }
                    SgnjKind::Sgnjx => {
                        // magnitude(a) with sign(a) ^ sign(b) is `a ^ (b & m)`: only the
                        // sign bit differs from a, exactly where b's sign is set. `a` may
                        // live in a pinned register (or FA, when it needed the boxing
                        // check), so the xor goes into FA.
                        self.a.vec_and(d, b, FC);
                        self.a.vec_eor(FA, a, d);
                        self.fcommit(FWidth::S, rd, FA);
                    }
                }
            }
            Inst::Lr { width, rd, rs1, .. } => {
                let base = self.src(rs1, SRC0);
                self.require_aligned(base, width, pc);
                // Keep the address (rd may be rs1) and record the reservation only after the
                // load, so a faulting lr leaves the state untouched.
                self.a.mov(SRC1, base);
                self.a.and_imm(ADDR, base, ADDR_MASK).unwrap();
                let d = self.dst(rd);
                self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
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
            // fleq/fltq stay quiet for quiet NaNs but still signal for signaling ones;
            // that fix-up is not emitted here yet, so the engine handles them.
            Inst::FCmp { op: FCmpOp::Ltq | FCmpOp::Leq, .. } => return false,
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
                        FCmpOp::Ltq | FCmpOp::Leq => unreachable!("refused above"),
                    };
                    self.a.cset(d, cond);
                    self.commit(rd, d);
                }
            }
            Inst::FMinMax { max, canonical_nan, width, rd, rs1, rs2 } => {
                let a = self.fsrc(width, rs1, FA);
                let b = self.fsrc(width, rs2, FB);
                let d = self.fdest(width, rd);
                self.fminmax(max, canonical_nan, width, d, a, b);
                self.fcommit(width, rd, d);
            }
            Inst::FClass { width, rd, rs1 } => {
                if rd != 0 {
                    let a = self.fsrc(width, rs1, FA);
                    self.a.fmov_to_gpr(Fp::D, SRC0, a);
                    self.fclass(width, SRC0, rd);
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
            // A static rounding mode other than the ambient `frm`: swap FPCR around the
            // conversion, then restore the block's mode. RMM has no ARM equivalent and
            // stays on the interpreter.
            Inst::FCvtFromInt { from, to, rd, rs1, rm }
                if let Some(fpcr) = guest_fpcr(u32::from(rm)) =>
            {
                let v = self.src(rs1, SRC0);
                let signed = matches!(from, IntType::W | IntType::L);
                let wide = matches!(from, IntType::L | IntType::Lu);
                let d = self.fdest(to, rd);
                self.a.mrs_fpcr(SRC1);
                self.a.mov_imm(DST, fpcr);
                self.a.msr_fpcr(DST);
                self.a.cvt_from_int(signed, wide, fp(to), d, v);
                self.a.msr_fpcr(SRC1);
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
                self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
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
    /// NaN, as the spec requires, so singles go through the check unless this block is
    /// the one that boxed the value (see `Tx::fboxed`).
    fn fsrc(&mut self, w: FWidth, r: u8, scratch: Reg) -> Reg {
        if w == FWidth::S && self.fboxed(r) {
            return match FHOST_REG[r as usize] {
                Some(h) => h,
                None => {
                    self.a.ldr_fp(Fp::D, scratch, CTX, f_offset(r));
                    scratch
                }
            };
        }
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
                // SRC1, not ADDR: the checked path must not disturb the cached base in
                // ADDR (see `keeps_addr`), or an FP operand would drop an address a
                // later access could still reuse.
                self.a.mov_imm(SRC1, 0x7fc0_0000);
                self.a.csel(SRC0, SRC0, SRC1, Cond::Eq);
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
            // A register this block already boxed only needs lane 0 updated: a scalar
            // op would zero bits 63:32 (taking the box with it), an element insert
            // leaves them alone.
            (FWidth::S, Some(h)) if self.fboxed(r) => self.a.vec_ins_s0(h, v),
            (FWidth::S, _) => {
                self.a.fmov_to_gpr(Fp::S, SRC0, v);
                self.a.orr_imm(SRC0, SRC0, NAN_BOX).unwrap();
                self.fraw_store(r, SRC0);
            }
        }
        self.set_fboxed(r, w == FWidth::S);
    }

    /// Whether f[`r`] is known to hold a NaN-boxed single (see `Tx::fboxed`).
    fn fboxed(&self, r: u8) -> bool {
        self.fboxed >> r & 1 != 0
    }

    fn set_fboxed(&mut self, r: u8, boxed: bool) {
        if boxed {
            self.fboxed |= 1 << r;
        } else {
            self.fboxed &= !(1 << r);
        }
    }

    /// RISC-V `fclass`: a one-hot mask of the ten IEEE classes. Reads the raw bits after
    /// [`Tx::fsrc`] has applied the single-precision NaN-box rule.
    fn fclass(&mut self, w: FWidth, bits: Reg, rd: u8) {
        let d = self.dst(rd);
        let (exp_max, quiet, sign) = match w {
            FWidth::S => (0xff, 0x0040_0000, 1u64 << 31),
            FWidth::D => (0x7ff, 1u64 << 51, 1u64 << 63),
        };
        match w {
            FWidth::S => {
                self.a.ubfm_w(SRC1, bits, 23, 30); // exponent
                self.a.ubfm_w(ADDR, bits, 0, 22); // fraction
            }
            FWidth::D => {
                self.a.ubfm(SRC1, bits, 52, 62);
                self.a.ubfm(ADDR, bits, 0, 51);
            }
        }

        let mut done = Vec::new();
        match w {
            FWidth::S => self.a.cmp_imm_w(SRC1, exp_max),
            FWidth::D => self.a.cmp_imm(SRC1, exp_max),
        }
        let not_nan = self.a.pos();
        self.a.b_cond(Cond::Ne, not_nan); // patched after the NaN block
        let nan = self.a.pos();
        self.a.cbnz(ADDR, nan); // patched to the NaN block

        // exp == all ones, frac == 0: infinity.
        self.a.mov_imm(d, 7);
        self.a.tst_imm(bits, sign).unwrap();
        let b = self.a.pos();
        self.a.b_cond(Cond::Eq, b);
        done.push(b);
        self.a.mov_imm(d, 0);
        let b = self.a.pos();
        self.a.b(0);
        done.push(b);

        let nan_label = self.a.pos();
        self.a.patch(nan, nan_label);
        self.a.mov_imm(d, 8); // signaling NaN
        self.a.tst_imm(ADDR, quiet).unwrap();
        let b = self.a.pos();
        self.a.b_cond(Cond::Eq, b);
        done.push(b);
        self.a.mov_imm(d, 9); // quiet NaN
        let b = self.a.pos();
        self.a.b(0);
        done.push(b);

        let after_nan = self.a.pos();
        self.a.patch(not_nan, after_nan);
        match w {
            FWidth::S => self.a.cmp_imm_w(SRC1, 0),
            FWidth::D => self.a.cmp_imm(SRC1, 0),
        }
        let nonzero_exp = self.a.pos();
        self.a.b_cond(Cond::Ne, nonzero_exp); // patched to the normal case

        let subnormal = self.a.pos();
        self.a.cbnz(ADDR, subnormal);
        self.a.mov_imm(d, 4); // +0
        self.a.tst_imm(bits, sign).unwrap();
        let b = self.a.pos();
        self.a.b_cond(Cond::Eq, b);
        done.push(b);
        self.a.mov_imm(d, 3); // -0
        let b = self.a.pos();
        self.a.b(0);
        done.push(b);

        let subnormal_label = self.a.pos();
        self.a.patch(subnormal, subnormal_label);
        self.a.mov_imm(d, 5); // positive subnormal
        self.a.tst_imm(bits, sign).unwrap();
        let b = self.a.pos();
        self.a.b_cond(Cond::Eq, b);
        done.push(b);
        self.a.mov_imm(d, 2); // negative subnormal
        let b = self.a.pos();
        self.a.b(0);
        done.push(b);

        let normal_label = self.a.pos();
        self.a.patch(nonzero_exp, normal_label);
        self.a.mov_imm(d, 6); // positive normal
        self.a.tst_imm(bits, sign).unwrap();
        let b = self.a.pos();
        self.a.b_cond(Cond::Eq, b);
        done.push(b);
        self.a.mov_imm(d, 1); // negative normal

        let end = self.a.pos();
        for branch in done {
            self.a.patch(branch, end);
        }
        self.a.mov_imm(SRC0, 1);
        self.a.lslv(d, SRC0, d);
        self.commit(rd, d);
    }

    /// `fmin`/`fmax` and their Zfa `fminm`/`fmaxm` forms. ARM's minNum/maxNum cover
    /// the ordinary case; a signaling NaN needs RISC-V's "return the other operand" fix-up,
    /// and the Zfa forms replace every NaN result with the canonical quiet NaN.
    fn fminmax(&mut self, max: bool, canonical_nan: bool, w: FWidth, d: Reg, a: Reg, b: Reg) {
        // Keep the original operands alive when the destination aliases one of them:
        // the NaN fix-up runs after the base min/max has written `d`.
        let (mut a, mut b) = (a, b);
        if d == a {
            self.a.fmov_d(FC, a);
            a = FC;
            if d == b {
                b = FC;
            }
        } else if d == b {
            self.a.fmov_d(FC, b);
            b = FC;
        }
        self.a.fmov_to_gpr(Fp::D, SRC0, a);
        self.a.fmov_to_gpr(Fp::D, SRC1, b);
        // Bit 1: NaN; bit 0: signaling NaN.
        self.nan_flags(w, SRC0, SRC0);
        self.nan_flags(w, SRC1, SRC1);
        self.a.fp_minmax(max, !canonical_nan, fp(w), d, a, b);

        if canonical_nan {
            self.a.orr(ADDR, SRC0, SRC1);
            self.a.tst_imm(ADDR, 2).unwrap();
            let no_nan = self.a.pos();
            self.a.b_cond(Cond::Eq, no_nan);
            let quiet = match w {
                FWidth::S => 0x7fc0_0000,
                FWidth::D => 0x7ff8_0000_0000_0000,
            };
            self.a.mov_imm(SRC0, quiet);
            self.a.fmov_from_gpr(fp(w), d, SRC0);
            let end = self.a.pos();
            self.a.patch(no_nan, end);
            return;
        }

        // If a is a signaling NaN and b is a number, the result is b. The symmetric
        // case follows; both-NaN stays with the host's minNum/maxNum result.
        self.a.tst_imm(SRC0, 1).unwrap();
        let check_b = self.a.pos();
        self.a.b_cond(Cond::Eq, check_b);
        self.a.tst_imm(SRC1, 2).unwrap();
        self.a.b_cond(Cond::Ne, check_b);
        self.a.fmov_d(d, b);
        let done = self.a.pos();
        self.a.b(0);

        let check_b_label = self.a.pos();
        self.a.patch(check_b, check_b_label);
        self.a.tst_imm(SRC1, 1).unwrap();
        let skip_b = self.a.pos();
        self.a.b_cond(Cond::Eq, skip_b);
        self.a.tst_imm(SRC0, 2).unwrap();
        let skip_a = self.a.pos();
        self.a.b_cond(Cond::Ne, skip_a);
        self.a.fmov_d(d, a);
        let end = self.a.pos();
        self.a.patch(skip_b, end);
        self.a.patch(skip_a, end);
        self.a.patch(done, end);
    }

    /// Pack NaN information from raw bits: bit 1 means NaN, bit 0 means signaling NaN.
    fn nan_flags(&mut self, w: FWidth, bits: Reg, out: Reg) {
        let (exp_max, quiet) = match w {
            FWidth::S => (0xff, 0x0040_0000),
            FWidth::D => (0x7ff, 1u64 << 51),
        };
        match w {
            FWidth::S => {
                self.a.ubfm_w(ADDR, bits, 23, 30);
                self.a.ubfm_w(DST, bits, 0, 22);
            }
            FWidth::D => {
                self.a.ubfm(ADDR, bits, 52, 62);
                self.a.ubfm(DST, bits, 0, 51);
            }
        }
        match w {
            FWidth::S => self.a.cmp_imm_w(ADDR, exp_max),
            FWidth::D => self.a.cmp_imm(ADDR, exp_max),
        }
        self.a.cset(out, Cond::Eq);
        match w {
            FWidth::S => self.a.cmp_imm_w(DST, 0),
            FWidth::D => self.a.cmp_imm(DST, 0),
        }
        self.a.cset(ADDR, Cond::Ne);
        self.a.and(out, out, ADDR);
        self.a.tst_imm(DST, quiet).unwrap();
        self.a.cset(ADDR, Cond::Eq);
        self.a.and(ADDR, out, ADDR);
        self.a.lsl_imm(out, out, 1);
        self.a.orr(out, out, ADDR);
    }

    /// The guest's `time` CSR: CNTVCT scaled to the 10 MHz timebase the interpreter uses.
    fn time(&mut self, d: Reg) {
        self.a.mrs_cntvct(SRC0);
        self.a.mrs_cntfrq(SRC1);
        self.a.udiv(ADDR, SRC0, SRC1);
        self.a.msub(DST, ADDR, SRC1, SRC0);
        self.a.mov_imm(SRC0, 10_000_000);
        self.a.mul(ADDR, ADDR, SRC0);
        self.a.mul(DST, DST, SRC0);
        self.a.udiv(DST, DST, SRC1);
        self.a.add(d, ADDR, DST);
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
        self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
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
                // The Zb* ops on x0 (zero): the counts are 64, and the bit-writing
                // immediates plant their bit in the zero. The rest stay zero.
                AluOp::Clz | AluOp::Ctz => 64,
                AluOp::Bset | AluOp::Binv => 1u64 << (imm & 63),
                _ => 0, // and, shifts of zero, and the other Zb* ops
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
            // Zba slli.uw: the zero-extended low word, shifted (0..=63).
            AluOp::SllUw => {
                self.a.ubfm(d, a, 0, 31);
                if imm != 0 {
                    self.a.lsl_imm(d, d, imm as u32 & 63);
                }
            }
            AluOp::Ror => self.a.ror_imm(d, a, imm as u32 & 63),
            // Zbs: the 6-bit shamt is the bit index. `and_imm` refuses nothing here (a
            // run with one hole wraps, and a single bit is a run too), but falling back
            // before anything is emitted is free.
            AluOp::Bclr => {
                if self.a.and_imm(d, a, !(1u64 << (imm & 63))).is_none() {
                    return false;
                }
            }
            AluOp::Bext => self.a.ubfm(d, a, imm as u32 & 63, imm as u32 & 63),
            AluOp::Binv => {
                if self.a.eor_imm(d, a, 1u64 << (imm & 63)).is_none() {
                    return false;
                }
            }
            AluOp::Bset => {
                if self.a.orr_imm(d, a, 1u64 << (imm & 63)).is_none() {
                    return false;
                }
            }
            AluOp::Clz
            | AluOp::Ctz
            | AluOp::Cpop
            | AluOp::SextB
            | AluOp::SextH
            | AluOp::ZextH
            | AluOp::Rev8
            | AluOp::Brev8
            | AluOp::OrcB => self.bit_unary(op, d, a),
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
            let r = match op {
                AluOp32::Add => sext(imm),
                AluOp32::Clz | AluOp32::Ctz => 32, // clzw(0) = ctzw(0) = 32
                _ => 0,                            // a rotation of zero, and cpopw(0)
            };
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
            AluOp32::Ror => self.a.ror_imm_w(d, a, imm as u32 & 31),
            AluOp32::Clz => self.a.clz_w(d, a),
            AluOp32::Ctz => {
                self.a.rbit_w(d, a);
                self.a.clz_w(d, d);
            }
            AluOp32::Cpop => {
                self.a.ubfm(d, a, 0, 31); // the count covers the low word
                self.cpop(d, d);
            }
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
            // Zba: `b + (a << n)`, and the word-sourced forms, where the shift comes
            // from the zero-extending address mode.
            AluOp::Sh1Add => self.a.add_lsl(d, b, a, 1),
            AluOp::Sh2Add => self.a.add_lsl(d, b, a, 2),
            AluOp::Sh3Add => self.a.add_lsl(d, b, a, 3),
            AluOp::AddUw => self.add_uw(d, b, a, 0),
            AluOp::Sh1AddUw => self.add_uw(d, b, a, 1),
            AluOp::Sh2AddUw => self.add_uw(d, b, a, 2),
            AluOp::Sh3AddUw => self.add_uw(d, b, a, 3),
            AluOp::SllUw => return false, // immediate-only; the decoder never builds this
            AluOp::Andn => self.a.bic(d, a, b),
            AluOp::Orn => self.a.orn(d, a, b),
            AluOp::Xnor => self.a.eon(d, a, b),
            // RISC-V min/max are "select on the comparison"; the plain ARM min/max need
            // FEAT_CSSC, which base hardware does not have.
            AluOp::Max => {
                self.a.cmp(a, b);
                self.a.csel(d, a, b, Cond::Gt);
            }
            AluOp::Maxu => {
                self.a.cmp(a, b);
                self.a.csel(d, a, b, Cond::Hi);
            }
            AluOp::Min => {
                self.a.cmp(a, b);
                self.a.csel(d, a, b, Cond::Lt);
            }
            AluOp::Minu => {
                self.a.cmp(a, b);
                self.a.csel(d, a, b, Cond::Lo);
            }
            // ARM rotates only right; -shamt (mod 64) is the same rotation.
            AluOp::Rol => {
                self.a.sub(ADDR, XZR, b);
                self.a.rorv(d, a, ADDR);
            }
            AluOp::Ror => self.a.rorv(d, a, b),
            // Zbs: the register forms build `1 << (b & 63)`, then clear, flip, or set.
            AluOp::Bclr => {
                self.bit_shift(ADDR, b);
                self.a.bic(d, a, ADDR);
            }
            AluOp::Binv => {
                self.bit_shift(ADDR, b);
                self.a.eor(d, a, ADDR);
            }
            AluOp::Bset => {
                self.bit_shift(ADDR, b);
                self.a.orr(d, a, ADDR);
            }
            AluOp::Bext => {
                self.a.lsrv(d, a, b);
                assert!(self.a.and_imm(d, d, 1).is_some(), "1 is a bitmask immediate");
            }
            // Zicond: keep the value or zero, on whether the condition is zero. A zero
            // condition register fixes the result, and it must not go through the compare:
            // in the immediate form register 31 is SP, so `cmp xzr, #0` is not encodable.
            AluOp::CzeroEqz => {
                if b == XZR {
                    self.a.mov(d, XZR);
                } else {
                    self.a.cmp_imm(b, 0);
                    self.a.csel(d, a, XZR, Cond::Ne);
                }
            }
            AluOp::CzeroNez => {
                if b == XZR {
                    self.a.mov(d, a);
                } else {
                    self.a.cmp_imm(b, 0);
                    self.a.csel(d, a, XZR, Cond::Eq);
                }
            }
            AluOp::Clz
            | AluOp::Ctz
            | AluOp::Cpop
            | AluOp::SextB
            | AluOp::SextH
            | AluOp::ZextH
            | AluOp::Rev8
            | AluOp::Brev8
            | AluOp::OrcB => self.bit_unary(op, d, a),
            // Zbkb: the low half of each source, packed.
            AluOp::Pack => {
                assert!(self.a.and_imm(ADDR, b, 0xffff_ffff).is_some());
                self.a.lsl_imm(ADDR, ADDR, 32);
                assert!(self.a.and_imm(d, a, 0xffff_ffff).is_some());
                self.a.orr(d, d, ADDR);
            }
            AluOp::Packh => {
                assert!(self.a.and_imm(ADDR, b, 0xff).is_some());
                self.a.lsl_imm(ADDR, ADDR, 8);
                assert!(self.a.and_imm(d, a, 0xff).is_some());
                self.a.orr(d, d, ADDR);
            }
            AluOp::Packw => {
                assert!(self.a.and_imm(ADDR, b, 0xffff).is_some());
                self.a.lsl_imm_w(ADDR, ADDR, 16);
                assert!(self.a.and_imm(d, a, 0xffff).is_some());
                self.a.orr(d, d, ADDR);
                self.a.sxtw(d, d);
            }
            // Zbc: one NEON polynomial multiply gives all 128 product bits.
            AluOp::Clmul | AluOp::Clmulh | AluOp::Clmulr => {
                self.vec_last = None;
                self.vec_m2 = None;
                self.vec_m2_b = None;
                self.a.fmov_from_gpr(Fp::D, FA, a);
                self.a.fmov_from_gpr(Fp::D, FB, b);
                self.a.pmull_d(FC, FA, FB);
                match op {
                    AluOp::Clmul => self.a.fmov_to_gpr(Fp::D, d, FC),
                    AluOp::Clmulh => self.a.umov_d1(d, FC),
                    AluOp::Clmulr => {
                        self.a.fmov_to_gpr(Fp::D, ADDR, FC);
                        self.a.umov_d1(DST, FC);
                        // clmulr is product bits 126..63, i.e. the 128-bit product
                        // shifted right by 63. EXTR concatenates Rn (high) with Rm (low).
                        self.a.extr(d, DST, ADDR, 63);
                    }
                    _ => unreachable!(),
                }
            }
            // Zbkx: NEON TBL does the lookup; out-of-range indices naturally produce
            // zero because the table's unused upper half is zero.
            AluOp::Xperm4 | AluOp::Xperm8 => {
                self.vec_last = None;
                self.vec_m2 = None;
                self.vec_m2_b = None;
                self.a.fmov_from_gpr(Fp::D, FA, a);
                self.a.fmov_from_gpr(Fp::D, FB, b);
                if op == AluOp::Xperm8 {
                    self.a.vec_tbl(FC, FA, FB);
                    self.a.fmov_to_gpr(Fp::D, d, FC);
                } else {
                    // a's 16 nibbles, in order, become the table.
                    self.a.vec_shl_16b(FC, FA, 4);
                    self.a.vec_ushr_16b(FC, FC, 4);
                    self.a.vec_ushr_16b(FR, FA, 4);
                    self.a.vec_zip1_16b(FA, FC, FR);
                    // The low and high nibbles of b are the two index vectors.
                    self.a.vec_shl_16b(FC, FB, 4);
                    self.a.vec_ushr_16b(FC, FC, 4);
                    self.a.vec_ushr_16b(FR, FB, 4);
                    self.a.vec_tbl(FB, FA, FC);
                    self.a.vec_tbl(FC, FA, FR);
                    self.a.vec_shl_16b(FC, FC, 4);
                    self.a.vec_orr(FB, FB, FC);
                    self.a.fmov_to_gpr(Fp::D, d, FB);
                }
            }
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
            AluOp32::Rol => {
                self.a.sub(ADDR, XZR, b);
                self.a.rorv_w(d, a, ADDR);
            }
            AluOp32::Ror => self.a.rorv_w(d, a, b),
            AluOp32::Clz => self.a.clz_w(d, a),
            AluOp32::Ctz => {
                self.a.rbit_w(d, a);
                self.a.clz_w(d, d);
            }
            AluOp32::Cpop => {
                self.a.ubfm(d, a, 0, 31); // the count covers the low word
                self.cpop(d, d);
            }
        }
        self.a.sxtw(d, d);
        self.commit(rd, d);
        true
    }

    /// The unary Zbb/Zbs bit-manipulation ops, shared by their register and immediate
    /// forms. `cpop` and `orc.b` go through NEON with `FA` as scratch: the guest's f
    /// registers live in v0–v15 and v20–v31, so the FP scratch registers are free here.
    fn bit_unary(&mut self, op: AluOp, d: Reg, a: Reg) {
        match op {
            AluOp::Clz => self.a.clz(d, a),
            AluOp::Ctz => {
                self.a.rbit(d, a);
                self.a.clz(d, d);
            }
            AluOp::Cpop => self.cpop(d, a),
            AluOp::SextB => self.a.sbfm(d, a, 0, 7),
            AluOp::SextH => self.a.sbfm(d, a, 0, 15),
            AluOp::ZextH => self.a.ubfm(d, a, 0, 15),
            AluOp::Rev8 => self.a.rev(d, a),
            AluOp::Brev8 => {
                // Each byte reversed: NEON's per-byte bit reversal (rbit v.16b), with
                // the scalar register moved in and out.
                self.a.fmov_from_gpr(Fp::D, FA, a);
                self.a.rbit16b(FA, FA);
                self.a.fmov_to_gpr(Fp::D, d, FA);
            }
            AluOp::OrcB => {
                self.a.fmov_from_gpr(Fp::D, FA, a);
                self.a.cmeq_zero16b(FA, FA);
                self.a.mvn16b(FA, FA);
                self.a.fmov_to_gpr(Fp::D, d, FA);
            }
            _ => unreachable!("{op:?} is not unary"),
        }
    }

    /// `cpop`: NEON byte counts summed across the register (writing the scalar zeroes the
    /// upper half of the vector, so the eight zero bytes do not add anything).
    fn cpop(&mut self, d: Reg, a: Reg) {
        self.a.fmov_from_gpr(Fp::D, FA, a);
        self.a.cnt16b(FA, FA);
        self.a.addv16b(FA, FA);
        self.a.fmov_to_gpr(Fp::D, d, FA);
    }

    /// `tmp = 1 << (b & 63)`, the mask the register forms of the single-bit ops use.
    fn bit_shift(&mut self, tmp: Reg, b: Reg) {
        self.a.mov_imm(tmp, 1);
        self.a.lslv(tmp, tmp, b);
    }

    /// A unit-stride vector access at `vl == vlmax`: VLEN bits are one NEON register,
    /// whatever the element width, so the whole access is one 16-byte load or store
    /// (the entry guard pinned the configuration; unaligned addresses are fine).
    /// A `vle*ff.v` load translates the same way but records itself as fault-tolerant:
    /// its fault shortens `vl` instead of trapping (see `Jit::run_interruptible`).
    fn vector_mem(&mut self, inst: Inst, pc: u64) -> bool {
        let Some((_, bytes)) = self.vec else {
            return false;
        };
        if !matches!(
            inst,
            Inst::VLoad { kind: VMemKind::Unit, .. } | Inst::VStore { kind: VStoreKind::Unit, .. }
        ) {
            self.vec_m2_b = None;
        }
        match inst {
            Inst::VLoad { width, kind, vd, rs1, mask: false, nf: 0, .. }
                if matches!(kind, VMemKind::Unit | VMemKind::FaultOnlyFirst)
                    && width.bytes() == u64::from(bytes) =>
            {
                let regs = self.vec_regs;
                // The fault-only-first prefix shortening is modelled per element; a
                // group would need it on top of the multi-register move.
                if kind == VMemKind::FaultOnlyFirst && regs != 1 {
                    return false;
                }
                // With EEW == SEW the whole active set is vlmax * eew = VLEN bits, one
                // 16-byte move per register of the group (an EEW the configuration
                // does not cover interprets).
                let access = if kind == VMemKind::FaultOnlyFirst {
                    Access::FaultOnlyFirst { base: rs1, eew: bytes as u8, vd }
                } else {
                    Access::Load
                };
                let operand = self.mem_operand(rs1, 0, 16);
                if regs == 2 {
                    let (first, second) = self.m2_slot(vd);
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, access));
                            assert!(self.a.ldp_q_off(first, second, ADDR, 0));
                        }
                        None => {
                            self.host_address();
                            self.accesses.push((self.a.pos(), pc, self.retired, access));
                            self.a.ldp_q(first, second, ADDR);
                        }
                    }
                    if !(self.defer_m2_store && (bytes == 4 || bytes == 8)) {
                        self.v_str2(first, second, vd);
                    }
                    return true;
                }
                for r in 0..regs {
                    let dst = if regs == 2 && r == 1 { FB } else { FA };
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, access));
                            assert!(self.a.ldr_q_off(dst, ADDR, r * 16));
                        }
                        None => {
                            if r != 0 {
                                self.a.add_imm(ADDR, ADDR, 16);
                            }
                            self.accesses.push((self.a.pos(), pc, self.retired, access));
                            self.a.ldr_q_reg(dst, self.base_reg(), self.index_reg());
                        }
                    }
                    if regs != 2 {
                        self.v_str(dst, (vd + r as u8) & 31);
                    }
                }
                if regs == 2 {
                    self.v_str2(FA, FB, vd);
                }
                self.vec_last = if regs == 1 { Some(vd) } else { None };
                self.vec_m2 = if regs == 2 { Some(vd) } else { None };
                true
            }
            // Whole-register moves carry `nf` 16-byte registers whatever vl and vtype
            // say, so they translate with or without a pinned configuration - the
            // memcpy-shaped loops in vectorized code are built on them.
            Inst::VLoad { kind: VMemKind::WholeRegister, vd, rs1, mask: false, nf, .. } => {
                let n = u32::from(nf) + 1;
                let operand = self.mem_operand(rs1, 0, 16);
                if n == 2 {
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
                            assert!(self.a.ldp_q_off(FA, FB, ADDR, 0));
                        }
                        None => {
                            self.host_address();
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
                            self.a.ldp_q(FA, FB, ADDR);
                        }
                    }
                    self.v_str2(FA, FB, vd);
                    self.vec_m2 = Some(vd);
                    return true;
                }
                for k in 0..n {
                    let dst = if n == 2 && k == 1 { FB } else { FA };
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
                            assert!(self.a.ldr_q_off(dst, ADDR, k * 16));
                        }
                        None => {
                            if k != 0 {
                                self.a.add_imm(ADDR, ADDR, 16);
                            }
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Load));
                            self.a.ldr_q_reg(dst, self.base_reg(), self.index_reg());
                        }
                    }
                    if n > 2 {
                        self.v_str(dst, (vd + k as u8) & 31);
                    }
                }
                if n == 1 {
                    self.v_str(FA, vd);
                } else if n == 2 {
                    self.v_str2(FA, FB, vd);
                }
                self.vec_last = if n == 1 { Some(vd) } else { None };
                self.vec_m2 = if n == 2 { Some(vd) } else { None };
                true
            }
            Inst::VStore { kind: VStoreKind::WholeRegister, vs3, rs1, mask: false, nf, .. } => {
                let n = u32::from(nf) + 1;
                let operand = self.mem_operand(rs1, 0, 16);
                if n == 2 {
                    if self.vec_m2 != Some(vs3) {
                        self.v_ldr2(FA, FB, vs3);
                    }
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            assert!(self.a.stp_q_off(FA, FB, ADDR, 0));
                        }
                        None => {
                            self.host_address();
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            self.a.stp_q(FA, FB, ADDR);
                        }
                    }
                    self.vec_m2 = Some(vs3);
                    return true;
                }
                for k in 0..n {
                    let src = if n == 2 && k == 1 { FB } else { FA };
                    if !(n == 2 && self.vec_m2 == Some(vs3))
                        && !(n == 1 && self.vec_last == Some((vs3 + k as u8) & 31))
                    {
                        self.v_ldr(src, (vs3 + k as u8) & 31);
                    }
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            assert!(self.a.str_q_off(src, ADDR, k * 16));
                        }
                        None => {
                            if k != 0 {
                                self.a.add_imm(ADDR, ADDR, 16);
                            }
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            self.a.str_q_reg(src, self.base_reg(), self.index_reg());
                        }
                    }
                }
                self.vec_last = if n == 1 { Some(vs3) } else { None };
                self.vec_m2 = if n == 2 { Some(vs3) } else { None };
                true
            }
            Inst::VStore {
                width, kind: VStoreKind::Unit, vs3, rs1, mask: false, nf: 0, ..
            } if width.bytes() == u64::from(bytes) => {
                let regs = self.vec_regs;
                let operand = self.mem_operand(rs1, 0, 16);
                if regs == 2 {
                    let pair = match self.m2_cached(vs3) {
                        Some(pair) => pair,
                        None => {
                            let pair = self.m2_slot(vs3);
                            self.v_ldr2(pair.0, pair.1, vs3);
                            pair
                        }
                    };
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            assert!(self.a.stp_q_off(pair.0, pair.1, ADDR, 0));
                        }
                        None => {
                            self.host_address();
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            self.a.stp_q(pair.0, pair.1, ADDR);
                        }
                    }
                    return true;
                }
                for r in 0..regs {
                    let src = if regs == 2 && r == 1 { FB } else { FA };
                    if !(regs == 2 && self.vec_m2 == Some(vs3))
                        && (regs != 1 || self.vec_last != Some((vs3 + r as u8) & 31))
                    {
                        self.v_ldr(src, (vs3 + r as u8) & 31);
                    }
                    match operand {
                        Some(_) => {
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            assert!(self.a.str_q_off(src, ADDR, r * 16));
                        }
                        None => {
                            if r != 0 {
                                self.a.add_imm(ADDR, ADDR, 16);
                            }
                            self.accesses.push((self.a.pos(), pc, self.retired, Access::Store));
                            self.a.str_q_reg(src, self.base_reg(), self.index_reg());
                        }
                    }
                }
                self.vec_last = if regs == 1 { Some(vs3) } else { None };
                self.vec_m2 = if regs == 2 { Some(vs3) } else { None };
                true
            }
            _ => false,
        }
    }

    /// Zvbb's single-source bit operations at m1 with a full `vl`.
    fn vector_unary(&mut self, op: VUnaryOp, mask: bool, vd: u8, vs2: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1 {
            return false;
        }
        self.v_ldr(FA, vs2);
        match op {
            VUnaryOp::Rev8 => self.rev8_bytes(bytes, FA),
            VUnaryOp::Brev8 => self.brev8(FA),
            VUnaryOp::Brev => {
                self.brev8(FA);
                self.rev8_bytes(bytes, FA);
            }
            VUnaryOp::Clz => {
                if bytes == 8 {
                    return false; // NEON CLZ has no 2d form
                }
                self.a.vec_clz(bytes * 8, FA, FA);
            }
            VUnaryOp::Ctz => {
                if bytes == 8 {
                    return false;
                }
                self.brev8(FA);
                self.rev8_bytes(bytes, FA);
                self.a.vec_clz(bytes * 8, FA, FA);
            }
            VUnaryOp::Cpop => {
                self.a.vec_cnt(FA, FA);
                match bytes {
                    1 => {}
                    2 => self.a.vec_uaddlp(8, FA, FA),
                    4 => {
                        self.a.vec_uaddlp(8, FA, FA);
                        self.a.vec_uaddlp(16, FA, FA);
                    }
                    8 => {
                        self.a.vec_uaddlp(8, FA, FA);
                        self.a.vec_uaddlp(16, FA, FA);
                        self.a.vec_uaddlp(32, FA, FA);
                    }
                    _ => unreachable!(),
                }
            }
        }
        if mask {
            // Compute first, then merge: the bit-reversal helpers borrow every scratch
            // register, so the old destination can only be loaded afterwards.
            self.v_ldr(FB, vd);
            self.vector_mask_expand(bytes);
            self.a.vec_bit(FB, FA, FR);
            self.v_str(FB, vd);
        } else {
            self.v_str(FA, vd);
        }
        true
    }

    /// Reverse bytes within each element (the `rev8` part of a `vrev8.v`).
    fn rev8_bytes(&mut self, bytes: u32, r: Reg) {
        match bytes {
            1 => {}
            2 => self.a.vec_rev16(r, r),
            4 => self.a.vec_rev32(r, r),
            8 => self.a.vec_rev64(r, r),
            _ => unreachable!(),
        }
    }

    /// Reverse bits within every byte using a nibble table and NEON TBL.
    fn brev8(&mut self, r: Reg) {
        // Reverse-nibble table: [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15].
        self.a.mov_imm(ADDR, 0x0e06_0a02_0c04_0800);
        self.a.fmov_from_gpr(Fp::D, FB, ADDR);
        self.a.mov_imm(ADDR, 0x0f07_0b03_0d05_0901);
        self.a.fmov_from_gpr(Fp::D, FC, ADDR);
        self.a.vec_ins_last(FB, FC, 8);
        self.a.vec_shl_16b(FC, r, 4);
        self.a.vec_ushr_16b(FC, FC, 4);
        self.a.vec_ushr_16b(FR, r, 4);
        self.a.vec_tbl(r, FB, FC);
        self.a.vec_tbl(FC, FB, FR);
        self.a.vec_shl_16b(r, r, 4);
        self.a.vec_orr(r, r, FC);
    }

    /// The bitwise and sum reductions at m1 with a full `vl`.
    fn vector_red(&mut self, op: VRedOp, vd: u8, vs1: u8, vs2: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1 {
            return false;
        }
        self.v_ldr(FA, vs2);
        self.v_ldr(FB, vd);
        self.v_ldr(FR, vs1);
        match bytes {
            1 | 2 | 4 | 8 => {}
            _ => unreachable!(),
        }
        if op == VRedOp::Sum {
            match bytes {
                1 | 2 | 4 => {
                    self.a.vec_addv(bytes * 8, FA, FA);
                    self.a.fmov_to_gpr(Fp::S, SRC0, FA);
                }
                8 => {
                    self.a.vec_addp_d(FA, FA);
                    self.a.fmov_to_gpr(Fp::D, SRC0, FA);
                }
                _ => unreachable!(),
            }
        } else {
            let shifts: &[u32] = match bytes {
                1 => &[8, 4, 2, 1],
                2 => &[8, 4, 2],
                4 => &[8, 4],
                8 => &[8],
                _ => unreachable!(),
            };
            for &imm in shifts {
                self.a.vec_ext(FC, FA, FA, imm);
                match op {
                    VRedOp::And => self.a.vec_and(FA, FA, FC),
                    VRedOp::Or => self.a.vec_orr(FA, FA, FC),
                    VRedOp::Xor => self.a.vec_eor(FA, FA, FC),
                    VRedOp::Sum => unreachable!(),
                }
            }
            self.a.fmov_to_gpr(if bytes == 8 { Fp::D } else { Fp::S }, SRC0, FA);
        }
        self.a.fmov_to_gpr(if bytes == 8 { Fp::D } else { Fp::S }, SRC1, FR);
        match bytes {
            1 => {
                self.a.and_imm(SRC0, SRC0, 0xff).unwrap();
                self.a.and_imm(SRC1, SRC1, 0xff).unwrap();
                match op {
                    VRedOp::Sum => {
                        self.a.add(SRC0, SRC0, SRC1);
                        self.a.and_imm(SRC0, SRC0, 0xff).unwrap();
                    }
                    VRedOp::And => self.a.and(SRC0, SRC0, SRC1),
                    VRedOp::Or => self.a.orr(SRC0, SRC0, SRC1),
                    VRedOp::Xor => self.a.eor(SRC0, SRC0, SRC1),
                }
            }
            2 => {
                self.a.and_imm(SRC0, SRC0, 0xffff).unwrap();
                self.a.and_imm(SRC1, SRC1, 0xffff).unwrap();
                match op {
                    VRedOp::Sum => {
                        self.a.add(SRC0, SRC0, SRC1);
                        self.a.and_imm(SRC0, SRC0, 0xffff).unwrap();
                    }
                    VRedOp::And => self.a.and(SRC0, SRC0, SRC1),
                    VRedOp::Or => self.a.orr(SRC0, SRC0, SRC1),
                    VRedOp::Xor => self.a.eor(SRC0, SRC0, SRC1),
                }
            }
            4 => {
                self.a.and_imm(SRC0, SRC0, 0xffff_ffff).unwrap();
                self.a.and_imm(SRC1, SRC1, 0xffff_ffff).unwrap();
                match op {
                    VRedOp::Sum => self.a.add_w(SRC0, SRC0, SRC1),
                    VRedOp::And => self.a.and(SRC0, SRC0, SRC1),
                    VRedOp::Or => self.a.orr(SRC0, SRC0, SRC1),
                    VRedOp::Xor => self.a.eor(SRC0, SRC0, SRC1),
                }
            }
            8 => match op {
                VRedOp::Sum => self.a.add(SRC0, SRC0, SRC1),
                VRedOp::And => self.a.and(SRC0, SRC0, SRC1),
                VRedOp::Or => self.a.orr(SRC0, SRC0, SRC1),
                VRedOp::Xor => self.a.eor(SRC0, SRC0, SRC1),
            },
            _ => unreachable!(),
        }
        self.a.fmov_from_gpr(if bytes == 8 { Fp::D } else { Fp::S }, FA, SRC0);
        self.a.vec_ins_lane0(bytes, FB, FA);
        self.v_str(FB, vd);
        true
    }

    /// `vmacc` (`vd += vs2*src`) and `vmadd` (`vd = src*vd + vs2`) at m1.
    fn vector_macc(&mut self, add: bool, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1 || bytes == 8 {
            return false;
        }
        self.v_ldr(FA, vd);
        self.v_ldr(FB, vs2);
        if !self.vector_operand(src, FC, bytes) {
            return false;
        }
        if add {
            self.a.vec_mla(bytes * 8, FA, FB, FC);
            self.v_str(FA, vd);
        } else {
            self.a.vec_mla(bytes * 8, FB, FA, FC);
            self.v_str(FB, vd);
        }
        true
    }

    /// `vsext`/`vzext`: widen the low SEW/narrow bits of every element.
    fn vector_ext(&mut self, sext: bool, narrow: u32, vd: u8, vs2: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1 || !matches!(narrow, 2 | 4 | 8) {
            return false;
        }
        let mut bits = bytes * 8 / narrow;
        if !matches!(bits, 8 | 16 | 32) {
            return false;
        }
        self.v_ldr(FA, vs2);
        while bits < bytes * 8 {
            self.a.vec_shift_long(sext, bits, FA, FA);
            bits *= 2;
        }
        self.v_str(FA, vd);
        true
    }

    /// Zvbb `vwsll`: widen each SEW element to 2×SEW and shift it left by the masked
    /// amount. e8→e16 and e16→e32 fit two NEON lanes per destination register; e32→e64
    /// stays on the interpreter.
    fn vector_wshift(&mut self, mask: bool, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if mask || self.vec_regs != 1 || !matches!(bytes, 1 | 2) {
            return false;
        }
        self.v_ldr(FA, vs2);
        self.a.vec_ext(FB, FA, FA, 8);
        let amount = |tx: &mut Self| {
            if let VSel::I(imm) = src {
                tx.a.mov_imm(ADDR, u64::from(imm));
                tx.a.vec_dup(FC, ADDR, bytes);
                true
            } else {
                tx.vector_operand(src, FC, bytes)
            }
        };
        if !amount(self) {
            return false;
        }
        if bytes == 1 {
            // Mask each byte amount to log2(16) = 4 bits, then split into two halves.
            self.a.vec_shl_16b(FC, FC, 4);
            self.a.vec_ushr_16b(FC, FC, 4);
            self.a.vec_ext(FR, FC, FC, 8);
            self.a.vec_shl_16b(FR, FR, 4);
            self.a.vec_ushr_16b(FR, FR, 4);
            self.a.vec_shift_long(false, 8, FA, FA);
            self.a.vec_shift_long(false, 8, FB, FB);
            self.a.vec_shift_long(false, 8, FC, FC);
            self.a.vec_shift_long(false, 8, FR, FR);
            self.a.vec_ushl(16, FA, FA, FC);
            self.a.vec_ushl(16, FB, FB, FR);
        } else {
            // Mask 16-bit amounts to log2(32) = 5 bits, then widen both halves to 32-bit.
            self.a.vec_movi_8h(FR, 31);
            self.a.vec_and(FC, FC, FR);
            self.a.vec_ext(FR, FC, FC, 8);
            self.a.vec_shift_long(false, 16, FC, FC);
            self.a.vec_shift_long(false, 16, FR, FR);
            self.a.vec_shift_long(false, 16, FA, FA);
            self.a.vec_shift_long(false, 16, FB, FB);
            self.a.vec_ushl(32, FA, FA, FC);
            self.a.vec_ushl(32, FB, FB, FR);
        }
        self.v_str(FA, vd);
        self.v_str(FB, (vd + 1) & 31);
        true
    }

    /// `vnsrl.vi`/`vnsra.vi`: shift the m2 source and narrow it into one m1 register.
    fn vector_narrow(&mut self, sra: bool, mask: bool, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if mask || self.vec_regs != 1 || !matches!(bytes, 1 | 2 | 4) {
            return false;
        }
        let VSel::I(imm) = src else { return false };
        if !group_fits(vs2, 2) {
            return false;
        }
        let wide_bits = bytes * 16;
        let sh = u32::from(imm) & (wide_bits - 1);
        self.v_ldr2(FA, FB, vs2);
        if sh > 0 {
            if sra {
                self.a.vec_sshr_imm(wide_bits, FA, FA, sh);
                self.a.vec_sshr_imm(wide_bits, FB, FB, sh);
            } else {
                self.a.vec_ushr_imm(wide_bits, FA, FA, sh);
                self.a.vec_ushr_imm(wide_bits, FB, FB, sh);
            }
        }
        self.a.vec_xtn(wide_bits, FA, FA);
        self.a.vec_xtn(wide_bits, FB, FB);
        self.a.vec_ins_last(FA, FB, 8);
        self.v_str(FA, vd);
        true
    }

    /// A lane-wise unmasked arithmetic op, one NEON instruction per element width (the
    /// bitwise ones are element-size independent and use the byte forms).
    fn vector_arith(&mut self, op: VArithOp, vd: u8, vs2: u8, src: VSel, mask: bool) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        let bits = bytes * 8;
        self.vec_m2_b = None;
        if mask {
            if self.vec_regs != 1 {
                return false;
            }
            return self.vector_arith_masked(op, vd, vs2, src, bytes, bits);
        }
        if self.vec_regs > 1 {
            let regs = self.vec_regs;
            if !matches!(
                op,
                VArithOp::Add
                    | VArithOp::Rsub
                    | VArithOp::Sub
                    | VArithOp::And
                    | VArithOp::Andn
                    | VArithOp::Or
                    | VArithOp::Xor
                    | VArithOp::Mul
            ) || (op == VArithOp::Mul && bits == 64)
            {
                return false;
            }
            if !group_fits(vd, regs) || !group_fits(vs2, regs) || groups_overlap(vd, vs2, regs) {
                return false;
            }
            if let VSel::V(vs1) = src
                && (!group_fits(vs1, regs)
                    || groups_overlap(vd, vs1, regs)
                    || groups_overlap(vs2, vs1, regs))
            {
                return false;
            }
            self.vec_last = None;
            self.vec_m2 = None;
            for r in 0..regs {
                self.v_ldr(FA, vs2 + r as u8);
                let b = match src {
                    VSel::V(vs1) => {
                        self.v_ldr(FB, vs1 + r as u8);
                        FB
                    }
                    _ => {
                        if !self.vector_operand(src, FB, bytes) {
                            return false;
                        }
                        FB
                    }
                };
                match op {
                    VArithOp::Add => self.a.vec_add(bits, FA, FA, b),
                    VArithOp::Rsub => self.a.vec_sub(bits, FA, b, FA),
                    VArithOp::Sub => self.a.vec_sub(bits, FA, FA, b),
                    VArithOp::And => self.a.vec_and(FA, FA, b),
                    VArithOp::Andn => {
                        self.a.mvn16b(b, b);
                        self.a.vec_and(FA, FA, b);
                    }
                    VArithOp::Or => self.a.vec_orr(FA, FA, b),
                    VArithOp::Mul => self.a.vec_mul(bits, FA, FA, b),
                    VArithOp::Xor => self.a.vec_eor(FA, FA, b),
                    _ => unreachable!(),
                }
                self.v_str(FA, vd + r as u8);
            }
            return true;
        }
        if matches!(op, VArithOp::Rol | VArithOp::Ror) {
            return self.vector_rotate(vd, vs2, src, op == VArithOp::Rol);
        }
        if op == VArithOp::Sll {
            return self.vector_sll(vd, vs2, src);
        }
        if self.vec_regs != 1 {
            return false;
        }
        if op == VArithOp::SlideUp {
            return self.vector_slide_up(vd, vs2, src);
        }
        if op == VArithOp::Gather {
            return self.vector_gather(vd, vs2, src);
        }
        // vslidedown: elements move down by a constant, the top zero-filled. The guard
        // pins vl to VLMAX, where the spec's "source read bounded by VLMAX" is just the
        // register's own elements - one byte-extract against a zero register.
        if let (VArithOp::SlideDown, VSel::I(imm)) = (op, src) {
            let elems = 16 / bytes;
            if self.vec_last != Some(vs2) {
                self.v_ldr(FA, vs2);
            }
            if u32::from(imm) >= elems {
                self.a.vec_movi_zero(FA);
            } else if imm > 0 {
                self.a.vec_movi_zero(FB);
                self.a.vec_ext(FA, FA, FB, u32::from(imm) * bytes);
            }
            self.v_str(FA, vd);
            self.vec_last = Some(vd);
            return true;
        }
        // Decide before emitting anything that guest state depends on.
        let supported = matches!(
            op,
            VArithOp::Add
                | VArithOp::Rsub
                | VArithOp::Sub
                | VArithOp::And
                | VArithOp::Andn
                | VArithOp::Or
                | VArithOp::Xor
                | VArithOp::Mul
                | VArithOp::Srl
                | VArithOp::Sra
        );
        if !supported {
            return false;
        }
        if op == VArithOp::Mul && bits == 64 {
            return false; // NEON's MUL tops out at 4s
        }
        if self.vec_last != Some(vs2) {
            self.v_ldr(FA, vs2);
        }
        match (op, src) {
            // Shifts want the immediate zero-extended and the scalar masked to
            // log2(SEW) bits, neither of which `vector_operand` does.
            (VArithOp::Srl | VArithOp::Sra, VSel::I(imm)) => {
                let k = u32::from(imm) & (bits - 1);
                if k > 0 {
                    if op == VArithOp::Srl {
                        self.a.vec_ushr_imm(bits, FA, FA, k);
                    } else {
                        self.a.vec_sshr_imm(bits, FA, FA, k);
                    }
                }
            }
            // NEON's USHL/SSHL take a *signed* byte per lane: a negative amount
            // shifts right (SSHL sign-fills), so the masked amount is negated and
            // broadcast. RVV's `.vx` shift amount is the scalar modulo SEW.
            (VArithOp::Srl | VArithOp::Sra, VSel::X(r)) => {
                let x = self.src(r, SRC0);
                self.a.mov_imm(SRC1, u64::from(bits - 1));
                self.a.and(SRC1, x, SRC1);
                self.a.sub(SRC1, XZR, SRC1);
                self.a.vec_dup(FB, SRC1, bytes);
                self.vector_shift(op, bits);
            }
            (VArithOp::Srl | VArithOp::Sra, VSel::V(vs1)) => {
                self.v_ldr(FB, vs1);
                self.a.mov_imm(SRC0, u64::from(bits - 1));
                self.a.vec_dup(FC, SRC0, bytes);
                self.a.vec_and(FB, FB, FC);
                self.a.vec_dup(FC, XZR, bytes);
                self.a.vec_sub(bits, FB, FC, FB);
                self.vector_shift(op, bits);
            }
            _ => {
                if !self.vector_operand(src, FB, bytes) {
                    return false; // only scratch (FA) was touched
                }
                match op {
                    VArithOp::Add => self.a.vec_add(bits, FA, FA, FB),
                    VArithOp::Sub => self.a.vec_sub(bits, FA, FA, FB),
                    VArithOp::Rsub => self.a.vec_sub(bits, FA, FB, FA),
                    VArithOp::And => self.a.vec_and(FA, FA, FB),
                    VArithOp::Andn => {
                        self.a.mvn16b(FB, FB);
                        self.a.vec_and(FA, FA, FB);
                    }
                    VArithOp::Or => self.a.vec_orr(FA, FA, FB),
                    VArithOp::Mul => self.a.vec_mul(bits, FA, FA, FB),
                    _ => self.a.vec_eor(FA, FA, FB),
                }
            }
        }
        self.v_str(FA, vd);
        self.vec_last = Some(vd);
        true
    }

    /// A masked lane-wise integer op at m1: compute every lane, then blend the old
    /// destination back in where v0's bit is clear. Integer arithmetic has no flags, so
    /// evaluating the inactive lanes is unobservable.
    fn vector_arith_masked(
        &mut self,
        op: VArithOp,
        vd: u8,
        vs2: u8,
        src: VSel,
        bytes: u32,
        bits: u32,
    ) -> bool {
        let supported = matches!(
            op,
            VArithOp::Add
                | VArithOp::Rsub
                | VArithOp::Sub
                | VArithOp::And
                | VArithOp::Andn
                | VArithOp::Or
                | VArithOp::Xor
                | VArithOp::Mul
                | VArithOp::Srl
                | VArithOp::Sra
        );
        if !supported || (op == VArithOp::Mul && bits == 64) {
            return false;
        }
        // The spec reserves masked arithmetic whose destination overlaps the mask
        // register; leave those encodings to the interpreter rather than invent a
        // semantics for them.
        if vd == 0 {
            return false;
        }
        // The destination is a source here too. Load it before anything can overwrite it,
        // so the merge value wins over any source that aliases it.
        self.v_ldr(FB, vd);
        self.v_ldr(FA, vs2);
        match (op, src) {
            (VArithOp::Srl | VArithOp::Sra, VSel::I(imm)) => {
                let k = u32::from(imm) & (bits - 1);
                if k > 0 {
                    if op == VArithOp::Srl {
                        self.a.vec_ushr_imm(bits, FA, FA, k);
                    } else {
                        self.a.vec_sshr_imm(bits, FA, FA, k);
                    }
                }
            }
            (VArithOp::Srl | VArithOp::Sra, VSel::X(r)) => {
                let x = self.src(r, SRC0);
                self.a.mov_imm(SRC1, u64::from(bits - 1));
                self.a.and(SRC1, x, SRC1);
                self.a.sub(SRC1, XZR, SRC1);
                self.a.vec_dup(FC, SRC1, bytes);
                if op == VArithOp::Srl {
                    self.a.vec_ushl(bits, FA, FA, FC);
                } else {
                    self.a.vec_sshl(bits, FA, FA, FC);
                }
            }
            (VArithOp::Srl | VArithOp::Sra, VSel::V(vs1)) => {
                self.v_ldr(FC, vs1);
                self.a.mov_imm(SRC0, u64::from(bits - 1));
                self.a.vec_dup(FR, SRC0, bytes);
                self.a.vec_and(FC, FC, FR);
                self.a.vec_dup(FR, XZR, bytes);
                self.a.vec_sub(bits, FC, FR, FC);
                if op == VArithOp::Srl {
                    self.a.vec_ushl(bits, FA, FA, FC);
                } else {
                    self.a.vec_sshl(bits, FA, FA, FC);
                }
            }
            _ => {
                if !self.vector_operand(src, FC, bytes) {
                    return false;
                }
                match op {
                    VArithOp::Add => self.a.vec_add(bits, FA, FA, FC),
                    VArithOp::Sub => self.a.vec_sub(bits, FA, FA, FC),
                    VArithOp::Rsub => self.a.vec_sub(bits, FA, FC, FA),
                    VArithOp::And => self.a.vec_and(FA, FA, FC),
                    VArithOp::Andn => {
                        self.a.mvn16b(FC, FC);
                        self.a.vec_and(FA, FA, FC);
                    }
                    VArithOp::Or => self.a.vec_orr(FA, FA, FC),
                    VArithOp::Mul => self.a.vec_mul(bits, FA, FA, FC),
                    _ => self.a.vec_eor(FA, FA, FC),
                }
            }
        }
        self.vector_mask_expand(bytes);
        self.a.vec_bit(FB, FA, FR);
        self.v_str(FB, vd);
        self.vec_last = None;
        true
    }

    /// Expand the low `vlmax` bits of v0 into all-ones/all-zero elements of `FR`.
    ///
    /// Each mask byte becomes eight bytes holding its bits in the low bit of each byte;
    /// repeated `zip1.16b` then doubles those bytes until they fill the element width.
    fn vector_mask_expand(&mut self, bytes: u32) {
        let groups = (16 / bytes).div_ceil(8);
        self.a.ldr_off(SRC0, CTX, v_offset(0));
        // Broadcast each byte with a multiply, then keep bit k of byte k: exactly the
        // eight element selectors that byte holds.
        self.a.mov_imm(ADDR, 0x0101_0101_0101_0101);
        self.a.mov_imm(DST, 0x8040_2010_0804_0201);
        for k in 0..groups {
            if k == 0 {
                self.a.and_imm(SRC1, SRC0, 0xff).unwrap();
            } else {
                self.a.lsr_imm(SRC1, SRC0, 8 * k);
                self.a.and_imm(SRC1, SRC1, 0xff).unwrap();
            }
            self.a.mul(SRC1, SRC1, ADDR);
            self.a.and(SRC1, SRC1, DST);
            let dst = if k == 0 { FR } else { FC };
            self.a.fmov_from_gpr(Fp::D, dst, SRC1);
            if k > 0 {
                self.a.vec_ins_last(FR, dst, 8);
            }
        }
        // Each representative byte still holds its bit's value (1, 2, 4, ...); clamp it
        // to one before duplicating it across the element.
        self.a.movi16b_one(FC);
        self.a.vec_umin(8, FR, FR, FC);
        for _ in 0..bytes.trailing_zeros() {
            self.a.vec_zip1_16b(FR, FR, FR);
        }
        // Element lanes are 1 where selected (0x0101.. after zipping); compare against
        // zero and invert for the all-ones mask.
        self.a.vec_cmeq_zero(bytes * 8, FR, FR);
        self.a.mvn16b(FR, FR);
    }

    /// Zvbb rotates: a pair of signed shifts (NEON's `ushl` shifts left for positive
    /// amounts and right for negative amounts) plus OR.
    fn vector_rotate(&mut self, vd: u8, vs2: u8, src: VSel, left: bool) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        let bits = bytes * 8;
        let regs = self.vec_regs;
        if !matches!(src, VSel::I(_) | VSel::X(_) | VSel::V(_)) {
            return false;
        }
        // NEON's `ushl` reads a signed byte per lane: a positive amount shifts left and
        // a negative one shifts right. The amount is reduced modulo the element width.
        // Rotating left by `k` is `(x << k) | (x >> (bits - k))`; rotating right swaps
        // the two halves. When `k` is zero, the out-of-range half shifts to zero.
        self.a.mov_imm(SRC0, u64::from(bits));
        self.a.vec_dup(FR, SRC0, bytes);
        for r in 0..regs {
            let data = (vs2 + r as u8) & 31;
            if self.vec_last != Some(data) {
                self.v_ldr(FA, data);
            }
            self.vector_amount(src, r as u8, bits, bytes, FB);
            self.a.vec_orr(FC, FA, FA);
            if left {
                self.a.vec_ushl(bits, FA, FA, FB);
                self.a.vec_sub(bits, FB, FB, FR);
                self.a.vec_ushl(bits, FC, FC, FB);
            } else {
                self.a.vec_sub(bits, FB, FR, FB);
                self.a.vec_ushl(bits, FC, FC, FB);
                self.a.vec_sub(bits, FB, FB, FR);
                self.a.vec_ushl(bits, FA, FA, FB);
            }
            self.a.vec_orr(FA, FA, FC);
            self.v_str(FA, (vd + r as u8) & 31);
        }
        self.vec_last = if regs == 1 { Some(vd) } else { None };
        self.vec_m2 = None;
        true
    }

    /// `vsll.vi` / `vsll.vx` / `vsll.vv`: the amount is reduced modulo the element width
    /// before NEON's signed per-lane shift sees it.
    fn vector_sll(&mut self, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        let bits = bytes * 8;
        let regs = self.vec_regs;
        if !matches!(src, VSel::I(_) | VSel::X(_) | VSel::V(_)) {
            return false;
        }
        for r in 0..regs {
            let data = (vs2 + r as u8) & 31;
            if self.vec_last != Some(data) {
                self.v_ldr(FA, data);
            }
            self.vector_amount(src, r as u8, bits, bytes, FB);
            self.a.vec_ushl(bits, FA, FA, FB);
            self.v_str(FA, (vd + r as u8) & 31);
        }
        self.vec_last = if regs == 1 { Some(vd) } else { None };
        self.vec_m2 = None;
        true
    }

    /// `vslideup.vi`: shift the source toward higher indices and keep the low
    /// `offset * SEW` bytes of the old destination.
    fn vector_slide_up(&mut self, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        let VSel::I(imm) = src else { return false };
        let elems = 16 / bytes;
        let offset = u32::from(imm);
        if offset >= elems {
            return true;
        }
        let shift = offset * bytes;
        self.v_ldr(FA, vd);
        self.v_ldr(FB, vs2);
        if shift == 0 {
            self.a.vec_orr(FA, FB, FB);
        } else {
            let low = if shift >= 8 { u64::MAX } else { (1u64 << (8 * shift)) - 1 };
            let high = if shift <= 8 { 0 } else { (1u64 << (8 * (shift - 8))) - 1 };
            self.a.mov_imm(SRC0, low);
            self.a.vec_dup(FC, SRC0, 8);
            self.a.mov_imm(SRC1, high);
            self.a.vec_dup(FR, SRC1, 8);
            self.a.vec_ext(FC, FC, FR, 8); // low half of FC, high half of FR
            self.a.vec_movi_zero(FR);
            self.a.vec_ext(FB, FR, FB, 16 - shift);
            self.a.vec_bit(FB, FA, FC);
            self.a.vec_orr(FA, FB, FB);
        }
        self.v_str(FA, vd);
        self.vec_last = Some(vd);
        self.vec_m2 = None;
        self.vec_m2_b = None;
        true
    }

    /// `vrgather.vi` / `vrgather.vv` at m1: NEON `tbl` gathers bytes from the
    /// 16-byte table. Wider elements expand each element index into its byte
    /// indices; out-of-range indices become byte indices outside the table, so
    /// `tbl` produces the architectural zero.
    fn vector_gather(&mut self, vd: u8, vs2: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if !matches!(bytes, 1 | 2 | 4 | 8) {
            return false;
        }
        if !matches!(src, VSel::I(_) | VSel::V(_)) {
            return false;
        }
        self.v_ldr(FA, vs2);
        match src {
            VSel::I(imm) => {
                self.a.mov_imm(SRC0, u64::from(imm));
                self.a.vec_dup(FB, SRC0, bytes);
            }
            VSel::V(vs1) => self.v_ldr(FB, vs1),
            _ => unreachable!("checked above"),
        }
        match bytes {
            1 => self.a.vec_tbl(FC, FA, FB),
            2 => {
                self.a.mov_imm(SRC0, 8);
                self.a.vec_dup(FC, SRC0, 2);
                self.a.vec_umin(16, FB, FB, FC);
                self.a.vec_add(16, FB, FB, FB);
                self.a.mov_imm(SRC0, 8);
                self.a.vec_dup(FC, SRC0, 2);
                self.a.vec_ushl(16, FC, FB, FC);
                self.a.vec_orr(FB, FB, FC);
                self.a.mov_imm(SRC0, 0x0100);
                self.a.vec_dup(FC, SRC0, 2);
                self.a.vec_add(16, FB, FB, FC);
                self.a.vec_tbl(FC, FA, FB);
            }
            4 => {
                self.a.mov_imm(SRC0, 4);
                self.a.vec_dup(FC, SRC0, 4);
                self.a.vec_umin(32, FB, FB, FC);
                self.a.vec_add(32, FB, FB, FB);
                self.a.vec_add(32, FB, FB, FB);
                self.a.mov_imm(SRC0, 8);
                self.a.vec_dup(FC, SRC0, 4);
                self.a.vec_ushl(32, FC, FB, FC);
                self.a.vec_orr(FB, FB, FC);
                self.a.mov_imm(SRC0, 16);
                self.a.vec_dup(FC, SRC0, 4);
                self.a.vec_ushl(32, FC, FB, FC);
                self.a.vec_orr(FB, FB, FC);
                self.a.mov_imm(SRC0, 0x0302_0100_0302_0100);
                self.a.vec_dup(FC, SRC0, 8);
                self.a.vec_add(8, FB, FB, FC);
                self.a.vec_tbl(FC, FA, FB);
            }
            _ => {
                self.a.mov_imm(SRC0, 2);
                self.a.vec_dup(FC, SRC0, 8);
                self.a.vec_cmhs_2d(FR, FB, FC);
                self.a.vec_bit(FB, FC, FR);
                self.a.vec_add(64, FB, FB, FB);
                self.a.vec_add(64, FB, FB, FB);
                self.a.vec_add(64, FB, FB, FB);
                for shift in [8, 16, 32] {
                    self.a.mov_imm(SRC0, shift);
                    self.a.vec_dup(FC, SRC0, 8);
                    self.a.vec_ushl(64, FC, FB, FC);
                    self.a.vec_orr(FB, FB, FC);
                }
                self.a.mov_imm(SRC0, 0x0706_0504_0302_0100);
                self.a.vec_dup(FC, SRC0, 8);
                self.a.vec_add(8, FB, FB, FC);
                self.a.vec_tbl(FC, FA, FB);
            }
        }
        self.v_str(FC, vd);
        self.vec_last = None;
        self.vec_m2 = None;
        self.vec_m2_b = None;
        true
    }

    /// Put a shift amount masked to `bits - 1` into `dst` for vector group `r`.
    fn vector_amount(&mut self, src: VSel, r: u8, bits: u32, bytes: u32, dst: Reg) {
        match src {
            VSel::I(imm) => {
                let k = (imm as u32) & (bits - 1);
                self.a.mov_imm(SRC0, u64::from(k));
                self.a.vec_dup(dst, SRC0, bytes);
            }
            VSel::X(rs1) => {
                let x = self.src(rs1, SRC0);
                self.a.mov_imm(SRC1, u64::from(bits - 1));
                self.a.and(SRC1, x, SRC1);
                self.a.vec_dup(dst, SRC1, bytes);
            }
            VSel::V(vs1) => {
                self.v_ldr(dst, (vs1 + r) & 31);
                self.a.mov_imm(SRC0, u64::from(bits - 1));
                self.a.vec_dup(FC, SRC0, bytes);
                self.a.vec_and(dst, dst, FC);
            }
            _ => unreachable!("only vector shift sources are passed here"),
        }
    }

    /// The right shift for a prepared per-lane amount in `FB`.
    fn vector_shift(&mut self, op: VArithOp, bits: u32) {
        if op == VArithOp::Srl {
            self.a.vec_ushl(bits, FA, FA, FB);
        } else {
            self.a.vec_sshl(bits, FA, FA, FB);
        }
    }

    /// The `.vf` scalar as a per-lane broadcast in `dst`, with the F extension's
    /// boxing rule: a scalar this block boxed is one lane-0 dup (the pinned `f`
    /// registers are host vector registers), anything else needs the check.
    fn vector_fscalar(&mut self, rs1: u8, dst: Reg, bytes: u32) {
        let bits = bytes * 8;
        if bytes == 8 || self.fboxed(rs1) {
            match FHOST_REG[rs1 as usize] {
                Some(h) => self.a.vec_dup_lane0(bits, dst, h),
                None => {
                    self.a.ldr_fp(Fp::D, dst, CTX, f_offset(rs1));
                    self.a.vec_dup_lane0(bits, dst, dst);
                }
            }
            return;
        }
        self.fraw_load(rs1, SRC0);
        self.a.lsr_imm(SRC1, SRC0, 32);
        self.a.cmn_imm_w(SRC1, 1);
        self.a.mov_imm(SRC1, 0x7fc0_0000);
        self.a.csel(SRC0, SRC0, SRC1, Cond::Eq);
        self.a.vec_dup(dst, SRC0, 4);
    }

    /// Float vector arithmetic (`vfadd` ... `vfdiv`, `.vv` or `.vf`) at the pinned
    /// configuration. The block-level FPCR (guest rounding, default NaN) is what the
    /// NEON ops run under, exactly like the translated scalar FP.
    fn vector_farith(&mut self, op: VFArithOp, vd: u8, vs2: u8, src: VSel, mask: bool) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if mask || bytes < 4 {
            return false;
        }
        let bits = bytes * 8;
        let regs = self.vec_regs;
        // The `.vf` scalar broadcasts once; register operands move per group register.
        let vs1 = match src {
            VSel::F(rs1) => {
                self.vector_fscalar(rs1, FB, bytes);
                None
            }
            VSel::V(vs1) => Some(vs1),
            _ => return false,
        };
        for r in 0..regs {
            self.v_ldr(FA, (vs2 + r as u8) & 31);
            if let Some(vs1) = vs1 {
                self.v_ldr(FB, (vs1 + r as u8) & 31);
            }
            match op {
                VFArithOp::Add => self.a.vec_fadd(bits, FA, FA, FB),
                VFArithOp::Sub => self.a.vec_fsub(bits, FA, FA, FB),
                VFArithOp::Mul => self.a.vec_fmul(bits, FA, FA, FB),
                _ => self.a.vec_fdiv(bits, FA, FA, FB),
            }
            self.v_str(FA, (vd + r as u8) & 31);
        }
        true
    }

    /// Zvfhmin's widening/narrowing f16↔f32 conversions at the pinned e16/m1 state.
    /// The widening form writes a two-register destination group; the narrowing form
    /// reads a two-register source group. Masked forms stay on the interpreter.
    fn vector_half_convert(&mut self, widening: bool, vd: u8, vs2: u8, mask: bool) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if mask || bytes != 2 || self.vec_regs != 1 {
            return false;
        }
        if widening {
            if vd > 30 {
                return false;
            }
            self.v_ldr(FA, vs2);
            self.a.vec_fcvtl(FB, FA);
            self.a.vec_fcvtl2(FC, FA);
            self.v_str(FB, vd);
            self.v_str(FC, vd + 1);
        } else {
            if vs2 > 30 || vd > 29 {
                return false;
            }
            self.v_ldr(FA, vs2);
            self.v_ldr(FB, vs2 + 1);
            self.a.vec_fcvtn(FC, FA);
            self.a.vec_fcvtn2(FC, FB);
            self.v_str(FC, vd);
        }
        self.vec_last = None;
        self.vec_m2 = None;
        true
    }

    /// The eight fused multiply-adds, each rebuilt from FMLA/FMLS (both fused, matching
    /// the spec's single rounding) plus exact sign flips. With `a` = vs2, `b` = the
    /// scalar or vs1, `d` = the old vd: the macc family computes `±a×b ± d` into d's
    /// register, the madd family `±d×b ± a` into vs2's.
    fn vector_fma(&mut self, op: VFMaOp, vd: u8, vs2: u8, src: VSel, mask: bool) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if mask || bytes < 4 {
            return false;
        }
        let bits = bytes * 8;
        let regs = self.vec_regs;
        if regs == 2
            && op == VFMaOp::Macc
            && let VSel::F(rs1) = src
        {
            return self.vector_fma_m2_macc(vd, vs2, rs1, bytes);
        }
        self.vec_m2 = None;
        self.vec_m2_b = None;
        let vs1 = match src {
            VSel::F(rs1) => {
                self.vector_fscalar(rs1, FB, bytes);
                None
            }
            VSel::V(vs1) => Some(vs1),
            _ => return false,
        };
        for r in 0..regs {
            self.v_ldr(FC, (vs2 + r as u8) & 31); // a
            self.v_ldr(FA, (vd + r as u8) & 31); // d
            if let Some(vs1) = vs1 {
                self.v_ldr(FB, (vs1 + r as u8) & 31);
            }
            let out = match op {
                VFMaOp::Macc => {
                    self.a.vec_fmla(bits, FA, FC, FB); // d + a×b
                    FA
                }
                VFMaOp::Nmacc => {
                    self.a.vec_fneg(bits, FA, FA);
                    self.a.vec_fneg(bits, FC, FC);
                    self.a.vec_fmla(bits, FA, FC, FB); // -d + -a×b
                    FA
                }
                VFMaOp::Msac => {
                    self.a.vec_fmls(bits, FA, FC, FB); // d - a×b
                    self.a.vec_fneg(bits, FA, FA); // -(d - a×b)
                    FA
                }
                VFMaOp::Nmsac => {
                    self.a.vec_fmls(bits, FA, FC, FB); // d - a×b
                    FA
                }
                VFMaOp::Madd => {
                    self.a.vec_fmla(bits, FC, FA, FB); // a + d×b
                    FC
                }
                VFMaOp::Nmadd => {
                    self.a.vec_fneg(bits, FC, FC);
                    self.a.vec_fneg(bits, FA, FA);
                    self.a.vec_fmla(bits, FC, FA, FB); // -a + -d×b
                    FC
                }
                VFMaOp::Msub => {
                    self.a.vec_fneg(bits, FC, FC);
                    self.a.vec_fmla(bits, FC, FA, FB); // -a + d×b
                    FC
                }
                VFMaOp::Nmsub => {
                    self.a.vec_fneg(bits, FA, FA);
                    self.a.vec_fmla(bits, FC, FA, FB); // a + -d×b
                    FC
                }
            };
            self.v_str(out, (vd + r as u8) & 31);
        }
        true
    }

    /// `vfredusum`/`vfredosum`: seed element 0 plus the source lanes, folded left to
    /// right. At m1 f32 has four lanes and f64 two, so scalar FADD is both simple and
    /// matches the ordered semantics.
    fn vector_fred(&mut self, op: VFRedOp, vd: u8, vs1: u8, vs2: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1
            || !matches!(bytes, 4 | 8)
            || !matches!(op, VFRedOp::Usum | VFRedOp::Osum)
        {
            return false;
        }
        let (fp, lanes) = if bytes == 4 { (Fp::S, 4) } else { (Fp::D, 2) };
        self.v_ldr(FA, vs1);
        self.v_ldr(FB, vs2);
        self.a.fmov_to_gpr(fp, SRC0, FA);
        self.a.fmov_from_gpr(fp, FA, SRC0);
        for i in 0..lanes {
            if bytes == 4 {
                self.a.vec_umov_s(SRC0, FB, i);
                self.a.fmov_from_gpr(Fp::S, FC, SRC0);
            } else {
                self.a.vec_umov_d(SRC0, FB, i);
                self.a.fmov_from_gpr(Fp::D, FC, SRC0);
            }
            self.a.fp_arith(FpOp::Add, fp, FA, FA, FC);
        }
        self.a.fmov_to_gpr(fp, SRC0, FA);
        self.v_ldr(FB, vd);
        self.a.fmov_from_gpr(fp, FA, SRC0);
        self.a.vec_ins_lane0(bytes, FB, FA);
        self.v_str(FB, vd);
        true
    }

    /// The e32/m2 `vfmacc.vf` shape: keep the two-q accumulator in FA/FB and broadcast
    /// the scalar with the by-element form of FMLA. Two groups can stay resident (one
    /// in each pair of scratch registers), which is exactly the load/load/macc/store
    /// loop the GPT benchmark emits.
    fn vector_fma_m2_macc(&mut self, vd: u8, vs2: u8, rs1: u8, bytes: u32) -> bool {
        let bits = bytes * 8;
        let Some(f) = FHOST_REG[rs1 as usize] else {
            return false;
        };
        let dest = match self.m2_cached(vd) {
            Some(pair) => pair,
            None => {
                let pair = self.m2_slot(vd);
                self.v_ldr2(pair.0, pair.1, vd);
                pair
            }
        };
        if vs2 == vd {
            self.a.vec_fmla_lane0(bits, dest.0, dest.0, f);
            self.a.vec_fmla_lane0(bits, dest.1, dest.1, f);
        } else if let Some(src) = self.m2_cached(vs2) {
            self.a.vec_fmla_lane0(bits, dest.0, src.0, f);
            self.a.vec_fmla_lane0(bits, dest.1, src.1, f);
        } else {
            // No free pair: borrow the pair the destination does not occupy, one q at
            // a time. That pair no longer holds a valid cached group.
            let tmp = if dest.0 == FA { FC } else { FA };
            if tmp == FC {
                self.vec_m2_b = None;
            } else {
                self.vec_m2 = None;
            }
            self.v_ldr(tmp, vs2);
            self.a.vec_fmla_lane0(bits, dest.0, tmp, f);
            self.v_ldr(tmp, vs2 + 1);
            self.a.vec_fmla_lane0(bits, dest.1, tmp, f);
        }
        self.v_str2(dest.0, dest.1, vd);
        true
    }

    /// `vfmv.v.f`: broadcast the scalar to every lane.
    fn vector_fmv_vf(&mut self, vd: u8, rs1: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if bytes < 4 {
            return false;
        }
        let regs = self.vec_regs;
        self.vector_fscalar(rs1, FA, bytes);
        for r in 0..regs {
            self.v_str(FA, (vd + r as u8) & 31);
        }
        true
    }

    /// `vfmv.f.s`: lane 0 into an `f` register, boxed for singles.
    fn vector_fmv_fs(&mut self, rd: u8, vs2: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if bytes < 4 {
            return false;
        }
        self.v_ldr(FA, vs2);
        if bytes == 4 {
            self.a.fmov_to_gpr(Fp::S, SRC0, FA);
            self.a.orr_imm(SRC0, SRC0, NAN_BOX).unwrap();
        } else {
            self.a.fmov_to_gpr(Fp::D, SRC0, FA);
        }
        self.fraw_store(rd, SRC0);
        self.set_fboxed(rd, bytes == 4);
        true
    }

    /// `vfslide1down.vf` under the pinned configuration: the elements move down one and
    /// the float scalar (read with the NaN-box rule) drops into the freed top element.
    fn vector_fslide1(&mut self, vd: u8, vs2: u8, rs1: u8) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        if self.vec_regs != 1 || bytes < 4 {
            return false;
        }
        self.v_ldr(FA, vs2);
        self.a.vec_movi_zero(FB);
        self.a.vec_ext(FA, FA, FB, bytes);
        let s =
            if bytes == 4 { self.fsrc(FWidth::S, rs1, FB) } else { self.fsrc(FWidth::D, rs1, FB) };
        self.a.vec_ins_last(FA, s, bytes);
        self.v_str(FA, vd);
        true
    }

    /// `vmv.v.*`: the whole register becomes the broadcast of the source.
    fn vector_mv(&mut self, vd: u8, src: VSel) -> bool {
        let Some((_, bytes)) = self.vec else { return false };
        let regs = self.vec_regs;
        match src {
            // A register source moves each group register too.
            VSel::V(vs1) => {
                for r in 0..regs {
                    self.v_ldr(FA, (vs1 + r as u8) & 31);
                    self.v_str(FA, (vd + r as u8) & 31);
                }
            }
            _ => {
                if !self.vector_operand(src, FA, bytes) {
                    return false;
                }
                for r in 0..regs {
                    self.v_str(FA, (vd + r as u8) & 31);
                }
            }
        }
        true
    }

    /// `vmseq`/`vmsne` at e8/m1 unmasked: one NEON compare, then the per-byte result is
    /// packed into a 16-bit mask (each half through the classic byte-to-bits multiply)
    /// and written over the low two bytes of `vd`. Bits at or above `vl` are left as
    /// they were, exactly as the interpreter leaves them.
    fn vector_cmp(&mut self, op: VCmpOp, vd: u8, vs2: u8, src: VSel, mask: bool) -> bool {
        let Some((sew, bytes)) = self.vec else { return false };
        if mask || sew != 0 || bytes != 1 || self.vec_regs != 1 {
            return false;
        }
        if !matches!(op, VCmpOp::Eq | VCmpOp::Ne) {
            return false;
        }
        self.v_ldr(FA, vs2);
        if !self.vector_operand(src, FB, 1) {
            return false;
        }
        self.a.cmeq16b(FA, FA, FB);
        if op == VCmpOp::Ne {
            self.a.mvn16b(FA, FA);
        }
        // Bytes are 0xFF/0x00; keep the low bit, then gather each half's bits into the
        // low byte of a 64-bit product (y = (y * 0x0102040810204080) >> 56).
        self.a.movi16b_one(FB);
        self.a.vec_and(FA, FA, FB);
        self.a.fmov_to_gpr(Fp::D, DST, FA);
        self.a.ext16b(FC, FA, FA, 8);
        self.a.fmov_to_gpr(Fp::D, ADDR, FC);
        self.a.mov_imm(SRC1, 0x0102_0408_1020_4080);
        self.a.mul(DST, DST, SRC1);
        self.a.mul(ADDR, ADDR, SRC1);
        self.a.lsr_imm(DST, DST, 56);
        self.a.lsr_imm(ADDR, ADDR, 56);
        self.a.lsl_imm(ADDR, ADDR, 8);
        self.a.orr(DST, DST, ADDR);
        assert!(self.a.str_imm(Mem::U16, DST, CTX, v_offset(vd) as i32));
        true
    }

    /// The `vm*.mm` mask-register logical ops at the supported widths: the bitmask
    /// lives in the low bytes, so this is a read-modify-write of exactly the bits
    /// below `vl`, matching the interpreter's per-bit update.
    fn vector_mask(&mut self, op: VMaskOp, vd: u8, vs1: u8, vs2: u8) -> bool {
        let Some((_, _)) = self.vec else { return false };
        if self.vec_regs != 1 {
            return false;
        }
        let mask = (1u64 << self.vec_vl()) - 1;
        self.a.ldr_off(DST, CTX, v_offset(vd));
        self.a.ldr_off(SRC0, CTX, v_offset(vs2));
        self.a.ldr_off(SRC1, CTX, v_offset(vs1));
        match op {
            VMaskOp::And => self.a.and(SRC0, SRC0, SRC1),
            VMaskOp::Andn => self.a.bic(SRC0, SRC0, SRC1),
            VMaskOp::Nand => {
                self.a.and(SRC0, SRC0, SRC1);
                self.a.mvn(SRC0, SRC0);
            }
            VMaskOp::Or => self.a.orr(SRC0, SRC0, SRC1),
            VMaskOp::Orn => self.a.orn(SRC0, SRC0, SRC1),
            VMaskOp::Nor => {
                self.a.orr(SRC0, SRC0, SRC1);
                self.a.mvn(SRC0, SRC0);
            }
            VMaskOp::Xor => self.a.eor(SRC0, SRC0, SRC1),
            VMaskOp::Xnor => {
                self.a.eor(SRC0, SRC0, SRC1);
                self.a.mvn(SRC0, SRC0);
            }
        }
        self.a.mov_imm(SRC1, mask);
        self.a.and(SRC0, SRC0, SRC1);
        self.a.bic(DST, DST, SRC1);
        self.a.orr(DST, DST, SRC0);
        self.a.str_off(DST, CTX, v_offset(vd));
        true
    }

    /// `vfirst.m`: the lowest set mask bit below `vl`, or -1. Two instructions' worth of
    /// work once the mask word is in a GPR: `clz(rbit(m))`.
    fn vector_first(&mut self, rd: u8, vs2: u8) -> bool {
        let Some((_, _)) = self.vec else { return false };
        if self.vec_regs != 1 {
            return false;
        }
        let mask = (1u64 << self.vec_vl()) - 1;
        self.a.ldr_off(DST, CTX, v_offset(vs2));
        self.a.mov_imm(SRC1, mask);
        self.a.and(DST, DST, SRC1);
        self.a.rbit(DST, DST);
        self.a.clz(DST, DST);
        self.a.cmp_imm(DST, 64);
        let d = self.dst(rd);
        self.a.mov_imm(SRC0, u64::MAX);
        self.a.csel(d, SRC0, DST, Cond::Eq);
        self.commit(rd, d);
        true
    }

    /// `vl` this translation assumes (its `vlmax`): only called with `vec` set.
    fn vec_vl(&self) -> u32 {
        let (_, bytes) = self.vec.expect("a vector translation always pins its config");
        16 / bytes
    }

    /// `vsetivli`: its AVL and vtype are constants, so the resulting state is known at
    /// translation time. Write it like the interpreter would and continue under it,
    /// which lets a loop that opens with a `vsetivli` translate as one block.
    fn vsetivli(&mut self, rd: u8, avl: u64, bits: u64) -> bool {
        let (vtype, vl) = crate::vector::vset_compute(avl, bits);
        self.a.mov_imm(SRC0, vtype);
        self.a.str_off(SRC0, CTX, VTYPE_OFF);
        self.a.mov_imm(SRC0, vl);
        self.a.str_off(SRC0, CTX, VL_OFF);
        self.constant(rd, vl);
        // Later vector ops in this block run under the new configuration; if it is not
        // one this subset can express, the first of them ends the block for the
        // interpreter (the state write above stays, as it must).
        let (vec, regs) = vector_config((vtype, vl)).map_or((None, 1), |(v, r)| (Some(v), r));
        self.vec = vec;
        self.vec_regs = regs;
        true
    }

    /// `vsetvli` with a register AVL (the shape every glibc loop head uses): the state
    /// write is a saturating minimum against vlmax. The resulting configuration is
    /// dynamic, so no vector op after it in this block may translate (they end the
    /// block, and the interpreter runs them under the freshly written state).
    fn vsetvli(&mut self, rd: u8, rs1: u8, bits: u64) {
        let (vtype, vlmax) = crate::vector::vset_compute(u64::MAX, bits);
        self.a.mov_imm(SRC0, vlmax);
        if rs1 != 0 {
            let avl = self.src(rs1, SRC1);
            self.a.cmp(avl, SRC0);
            self.a.csel(SRC0, avl, SRC0, Cond::Lo);
        } else if rd == 0 {
            // The AVL table's keep form: (rd, rs1) = (x0, x0) leaves vl as it is,
            // clipped by the new vlmax. rs1 = x0 with a real rd asks for VLMAX, which
            // SRC0 already holds.
            self.a.ldr_off(SRC1, CTX, VL_OFF);
            self.a.cmp(SRC1, SRC0);
            self.a.csel(SRC0, SRC1, SRC0, Cond::Lo);
        }
        let d = self.dst(rd);
        if d != SRC0 {
            self.a.mov(d, SRC0);
        }
        self.a.mov_imm(SRC1, vtype);
        self.a.str_off(SRC1, CTX, VTYPE_OFF);
        self.a.str_off(SRC0, CTX, VL_OFF);
        self.commit(rd, d);
        self.vec = None;
    }

    /// The vector instruction's source: another register, a broadcast scalar from an x
    /// register, or the sign-extended 5-bit immediate (arithmetic forms).
    fn vector_operand(&mut self, src: VSel, dst: Reg, bytes: u32) -> bool {
        match src {
            VSel::V(vs1) => {
                self.v_ldr(dst, vs1);
            }
            VSel::X(rs1) => {
                let x = self.src(rs1, SRC0);
                self.a.vec_dup(dst, x, bytes);
            }
            // Float scalars stay on the interpreter for now.
            VSel::F(_) => return false,
            VSel::I(imm) => {
                self.a.mov_imm(ADDR, crate::vector::sext5(imm));
                self.a.vec_dup(dst, ADDR, bytes);
            }
        }
        true
    }

    /// Load one 16-byte chunk of guest vector register `v` into `dst`. The register file
    /// is 16-byte aligned (see `CpuState`), so a single immediate-offset move reaches it.
    fn v_ldr(&mut self, dst: Reg, v: u8) {
        assert!(self.a.ldr_q_off(dst, CTX, v_offset(v)));
    }

    /// Store `src` into guest vector register `v` (see [`Tx::v_ldr`]).
    fn v_str(&mut self, src: Reg, v: u8) {
        assert!(self.a.str_q_off(src, CTX, v_offset(v)));
    }

    /// Load an m2 group's two chunks with one `ldp q`.
    fn v_ldr2(&mut self, first: Reg, second: Reg, v: u8) {
        assert!(v_offset(v) + 16 <= 1008);
        assert!(self.a.ldp_q_off(first, second, CTX, v_offset(v)));
    }

    /// Store an m2 group's two chunks with one `stp q`.
    fn v_str2(&mut self, first: Reg, second: Reg, v: u8) {
        assert!(v_offset(v) + 16 <= 1008);
        assert!(self.a.stp_q_off(first, second, CTX, v_offset(v)));
    }

    /// The host registers holding a cached m2 group, if any.
    fn m2_cached(&self, group: u8) -> Option<(Reg, Reg)> {
        if self.vec_m2 == Some(group) {
            Some((FA, FB))
        } else if self.vec_m2_b == Some(group) {
            Some((FC, FR))
        } else {
            None
        }
    }

    /// Pick the slot for a new m2 group, preferring an empty slot and evicting the
    /// second slot when both are occupied. The caller still has to load the group.
    fn m2_slot(&mut self, group: u8) -> (Reg, Reg) {
        if self.vec_m2 == Some(group) {
            (FA, FB)
        } else if self.vec_m2_b == Some(group) {
            (FC, FR)
        } else if self.vec_m2.is_none() {
            self.vec_m2 = Some(group);
            (FA, FB)
        } else {
            self.vec_m2_b = Some(group);
            (FC, FR)
        }
    }

    /// Zba word adds: `b + (zero-extended a << shift)`. In the ARM extended-register
    /// form register 31 reads as SP, not XZR, so a zero `b` cannot take the one-instruction
    /// form — the zero-extended source is the whole result then.
    fn add_uw(&mut self, d: Reg, b: Reg, a: Reg, shift: u32) {
        if b == XZR {
            self.a.ubfm(d, a, 0, 31);
            if shift != 0 {
                self.a.lsl_imm(d, d, shift);
            }
        } else if shift == 0 {
            self.a.add_uxtw(d, b, a);
        } else {
            self.a.add_uxtw_lsl(d, b, a, shift);
        }
    }

    /// Write the link register of `jal`/`jalr`.
    fn link(&mut self, rd: u8, value: u64) {
        self.constant(rd, value);
    }

    /// Emit the landing pad of an internalized backward edge (see [`Edge`]): scale the
    /// optimistic instret charge down to what this pass retired, check for interrupts
    /// with the count exact, then re-charge the block like a fresh entry and jump back
    /// to the loop body's first instruction.
    ///
    /// One pass retires `m` instructions, but the entry charged the whole block: each
    /// edge nets `+m`, the same as the dispatcher re-entry used to (undo down to the
    /// branch, full charge back), so every exit's static `retired - retired_before`
    /// correction stays valid. The entry guards need not run again: what they establish
    /// is either re-materialized by the block's own writes every pass (boxed singles) or
    /// a value this re-entry never touches (`vsetvl` blocks never internalize).
    fn emit_edge(&mut self, edge: Edge) {
        let here = self.a.pos();
        self.a.patch(edge.bcond, here);
        let skipped = self.retired - (edge.t + edge.m);
        if skipped > 0 {
            self.a.sub_imm(INSTRET_REG, INSTRET_REG, skipped);
        }
        self.interrupt_check(Some(edge.pc));
        self.a.add_imm(INSTRET_REG, INSTRET_REG, self.retired - edge.t);
        self.a.b(edge.lt);
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
        | Inst::FStore { rs1, .. }
        | Inst::VLoad { rs1, .. }
        | Inst::VStore { rs1, .. } => Some(rs1),
        _ => None,
    }
}

/// The `f` registers an instruction reads as *singles*, where the boxing rule matters
/// (raw readers like `fmv.x.s` and single stores never need it). Empty if the JIT does
/// not translate the instruction.
fn f_single_reads(inst: Inst) -> [Option<u8>; 3] {
    match inst {
        Inst::FArith { width: FWidth::S, rs1, rs2, rm: RM_DYN, .. } => [Some(rs1), Some(rs2), None],
        Inst::FSqrt { width: FWidth::S, rs1, rm: RM_DYN, .. } => [Some(rs1), None, None],
        Inst::FFma { width: FWidth::S, rs1, rs2, rs3, rm: RM_DYN, .. } => {
            [Some(rs1), Some(rs2), Some(rs3)]
        }
        Inst::FCmp { width: FWidth::S, rs1, rs2, .. } => [Some(rs1), Some(rs2), None],
        Inst::FMinMax { width: FWidth::S, rs1, rs2, .. } => [Some(rs1), Some(rs2), None],
        Inst::FClass { width: FWidth::S, rs1, .. } => [Some(rs1), None, None],
        // fcvt.d.s reads its single source with the boxing rule; both int conversions too.
        Inst::FCvtFloat { to: FWidth::D, rs1, .. } => [Some(rs1), None, None],
        Inst::FCvtToInt { from: FWidth::S, rs1, rm, .. } if rm != RM_DYN => [Some(rs1), None, None],
        Inst::VFArith { src: VSel::F(rs1), .. } => [Some(rs1), None, None],
        Inst::VFMa { src: VSel::F(rs1), .. } => [Some(rs1), None, None],
        Inst::VFSlide1 { rs1, .. } => [Some(rs1), None, None],
        Inst::Vmv { src: VSel::F(rs1), .. } => [Some(rs1), None, None],
        _ => [None; 3],
    }
}

/// The `f` register an instruction writes, if any (any width: a write ends the entry
/// assumption for later reads).
fn f_write(inst: Inst) -> Option<u8> {
    match inst {
        Inst::FLoad { rd, .. }
        | Inst::FMvFromInt { rd, .. }
        | Inst::FArith { rd, .. }
        | Inst::FSqrt { rd, .. }
        | Inst::FFma { rd, .. }
        | Inst::FMinMax { rd, .. }
        | Inst::FSgnj { rd, .. }
        | Inst::FCvtFloat { rd, .. }
        | Inst::FCvtFromInt { rd, .. } => Some(rd),
        _ => None,
    }
}

/// The f registers `block` reads as singles before writing them, up to two (the guard
/// costs a few instructions each). Their boxing is assumed at entry and checked by the
/// translation's head guard.
fn scan_fboxed_reads(block: &[(Inst, u8)]) -> u32 {
    let mut assumed = 0u32;
    let mut written = 0u32;
    for &(inst, _) in block {
        for r in f_single_reads(inst).into_iter().flatten() {
            if written >> r & 1 == 0 && assumed.count_ones() < 2 {
                assumed |= 1 << r;
            }
        }
        if let Some(w) = f_write(inst) {
            written |= 1 << w;
        }
    }
    assumed
}

/// The vector configuration `(sew log2, element bytes)` a translation may assume for
/// `(vtype, vl)`: m1 with no `vill`, a supported element width, and `vl` at its maximum
/// (VLEN = 128 bits, one NEON register). Anything else must interpret.
fn vector_config((vtype, vl): (u64, u64)) -> Option<((u32, u32), u32)> {
    let sew = (vtype >> 3) & 7;
    if vtype >> 63 != 0 || sew > 3 {
        return None;
    }
    // m1 through m8 are the LMULs the basic translated subset handles; operations
    // without a group-aware path still fall back inside the block.
    let lmul = vtype & 7;
    if lmul > 3 {
        return None;
    }
    let regs = 1u32 << lmul;
    let bytes = 1u32 << sew;
    if vl == u64::from(16 / bytes * regs) { Some(((sew as u32, bytes), regs)) } else { None }
}

/// A whole-register group that stays inside v0..v31. Groups that wrap are left to the
/// interpreter rather than teaching every arithmetic path about modular register numbers.
fn group_fits(v: u8, regs: u32) -> bool {
    u32::from(v) + regs <= 32
}

fn groups_overlap(a: u8, b: u8, regs: u32) -> bool {
    u32::from(a) < u32::from(b) + regs && u32::from(b) < u32::from(a) + regs
}

/// Whether the instruction reads or writes vector state, so its translation has to pin
/// the vector configuration. `vset*` is not included: it ends the block instead.
fn is_vector(inst: Inst) -> bool {
    matches!(
        inst,
        Inst::VLoad { .. }
            | Inst::VStore { .. }
            | Inst::VArith { .. }
            | Inst::VFRed { .. }
            | Inst::VFSlide1 { .. }
            | Inst::VFHalfConvert { .. }
            | Inst::VCmp { .. }
            | Inst::VMask { .. }
            | Inst::Vmv { .. }
            | Inst::VmvNr { .. }
            | Inst::Vmerge { .. }
            | Inst::VRed { .. }
            | Inst::VExt { .. }
            | Inst::VMacc { .. }
            | Inst::Vcpop { .. }
            | Inst::Vfirst { .. }
            | Inst::Vid { .. }
            | Inst::VmvSx { .. }
            | Inst::VmvXs { .. }
            | Inst::VNarrow { .. }
            | Inst::VUnary { .. }
            | Inst::VCompress { .. }
            | Inst::VIndexLoad { .. }
            | Inst::VSegLoad { .. }
            | Inst::VSegStore { .. }
            | Inst::VIndexStore { .. }
            | Inst::VWiden { .. }
            | Inst::VWShift { .. }
    )
}

/// Whether an instruction only touches the integer register file, so the vector
/// scratch-register caches stay valid across it.
fn is_integer_only(inst: Inst) -> bool {
    match inst {
        Inst::Op { op, .. } | Inst::OpImm { op, .. } => !matches!(
            op,
            AluOp::Cpop
                | AluOp::Brev8
                | AluOp::OrcB
                | AluOp::Clmul
                | AluOp::Clmulh
                | AluOp::Clmulr
                | AluOp::Xperm4
                | AluOp::Xperm8
        ),
        Inst::Op32 { op, .. } => op != AluOp32::Cpop,
        Inst::OpImm32 { .. }
        | Inst::Load { .. }
        | Inst::Store { .. }
        | Inst::Branch { .. }
        | Inst::Jal { .. }
        | Inst::Jalr { .. }
        | Inst::Lui { .. }
        | Inst::Auipc { .. } => true,
        _ => false,
    }
}

/// Whether an instruction is a pure integer ALU operation that cannot fault or use the
/// vector scratch registers.
fn is_pure_integer_alu(inst: Inst) -> bool {
    match inst {
        Inst::Op { op, .. } | Inst::OpImm { op, .. } => !matches!(
            op,
            AluOp::Cpop
                | AluOp::Brev8
                | AluOp::OrcB
                | AluOp::Clmul
                | AluOp::Clmulh
                | AluOp::Clmulr
                | AluOp::Xperm4
                | AluOp::Xperm8
        ),
        Inst::Op32 { op, .. } => op != AluOp32::Cpop,
        Inst::OpImm32 { .. } | Inst::Lui { .. } | Inst::Auipc { .. } => true,
        _ => false,
    }
}

/// Find unit m2 loads whose architectural store is dead because the next vector
/// instruction is a macc that overwrites the same destination group. Only pure integer
/// ALU instructions and constant formation may sit between them, so no fault or exit
/// can observe the skipped store.
fn plan_deferred_m2_stores(block: &[(Inst, u8)], pcs: &[u64], branch_targets: &[u64]) -> Vec<bool> {
    let mut defer = vec![false; block.len()];
    for i in 0..block.len() {
        let Inst::VLoad { kind: VMemKind::Unit, vd, mask: false, nf: 0, .. } = block[i].0 else {
            continue;
        };
        let mut j = i + 1;
        while j < block.len() {
            if branch_targets.contains(&pcs[j]) {
                break;
            }
            if let Inst::VFMa { op: VFMaOp::Macc, vd: d, src: VSel::F(rs1), mask: false, .. } =
                block[j].0
            {
                if d == vd && FHOST_REG[rs1 as usize].is_some() {
                    defer[i] = true;
                }
                break;
            }
            if is_pure_integer_alu(block[j].0) {
                j += 1;
                continue;
            }
            break;
        }
    }
    defer
}

/// Whether translating `inst` leaves `ADDR` alone, apart from [`Tx::mem_operand`], so the
/// base-register cache survives it (unless it writes the base register, see [`int_dest`]).
fn keeps_addr(inst: Inst) -> bool {
    // `ADDR` is scratch: the instructions whose translation writes it must say so, or a
    // later access would reuse a stale cached base.
    use AluOp::{Bclr, Binv, Bset, Clmulr, Div, Divu, Mulhsu, Pack, Packh, Packw, Rem, Remu, Rol};
    match inst {
        Inst::Lui { .. }
        | Inst::Auipc { .. }
        | Inst::OpImm { .. }
        | Inst::OpImm32 { .. }
        | Inst::Branch { .. } => true,
        Inst::Op { op, .. } => !matches!(
            op,
            Mulhsu
                | Div
                | Divu
                | Rem
                | Remu
                | Rol
                | Bclr
                | Binv
                | Bset
                | Pack
                | Packh
                | Packw
                | Clmulr
        ),
        Inst::Op32 { op, .. } => !matches!(
            op,
            AluOp32::Div | AluOp32::Divu | AluOp32::Rem | AluOp32::Remu | AluOp32::Rol
        ),
        // The FP arms never emit through ADDR: operands load into SRC0/SRC1 (the
        // singles' canonical-NaN fixup included) and results commit through CTX.
        Inst::FArith { .. }
        | Inst::FSqrt { .. }
        | Inst::FFma { .. }
        | Inst::FCmp { .. }
        | Inst::FCvtFloat { .. }
        | Inst::FCvtToInt { .. }
        | Inst::FCvtFromInt { .. }
        | Inst::FMvFromInt { .. }
        | Inst::FMvToInt { .. }
        | Inst::FSgnj { .. }
        | Inst::VFArith { .. }
        | Inst::VFMa { .. } => true,
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
        | Inst::Load { rd, .. }
        | Inst::Mop { rd } => Some(rd),
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

/// The block index a backward branch's target names, if the branch can re-enter the
/// body in place (see [`Tx::emit_edge`]): the target is an instruction of this very
/// block, no `vsetvl` intervenes (it would change the vector state across a pass), and
/// the translation carried no register fact (`known`, `ADDR`) into the target - a fresh
/// entry and a re-entry after a pass then start from the same translation state.
fn internal_edge(
    pcs: &[u64],
    carry_clean: &[bool],
    target: u64,
    upto: usize,
    has_vset: bool,
    already: Option<Edge>,
) -> Option<usize> {
    if has_vset || already.is_some() {
        return None;
    }
    let t = pcs[..upto].iter().position(|&p| p == target)?;
    carry_clean[t].then_some(t)
}

/// A fused rotate: the source, destination, right-rotate amount, and the shift whose
/// result the OR did not replace.
struct Rotate {
    rs: u8,
    rd: u8,
    amount: u32,
    keep_op: AluOp,
    keep_rd: u8,
    keep_imm: u32,
}

/// Match a rotate written as two shifts and an OR at `i`, which is what RV64GC compilers
/// emit because the base ISA has no rotate instruction: `slli t, a, m; srli a, a, 64-m;
/// or a, t, a` (or the same with the shifts in the other order).
///
/// The first shift must not write the shared source: otherwise the second shift reads
/// the already-shifted value, and the three instructions are not a rotate. Writing the
/// source in the second shift is fine (and common), because both shifts have read it by
/// then.
///
/// The shift the OR did not replace still has to run: its destination may stay live
/// after the OR, so dropping it would leave the guest register file wrong.
fn rotate_pattern(block: &[(Inst, u8)], i: usize) -> Option<Rotate> {
    let Inst::OpImm { op: op0, rd: d0, rs1: s0, imm: k0 } = block.get(i)?.0 else {
        return None;
    };
    let Inst::OpImm { op: op1, rd: d1, rs1: s1, imm: k1 } = block.get(i + 1)?.0 else {
        return None;
    };
    let Inst::Op { op: AluOp::Or, rd, rs1: o0, rs2: o1 } = block.get(i + 2)?.0 else {
        return None;
    };
    if s0 != s1 || d0 == 0 || d1 == 0 || rd == 0 || d0 == d1 || d0 == s0 {
        return None;
    }
    let (right, left) = match (op0, op1) {
        (AluOp::Srl, AluOp::Sll) => (k0, k1),
        (AluOp::Sll, AluOp::Srl) => (k1, k0),
        _ => return None,
    };
    let (right, left) = (right as u32, left as u32);
    if right == 0 || left == 0 || right + left != 64 {
        return None;
    }
    if !((o0 == d0 && o1 == d1) || (o0 == d1 && o1 == d0)) {
        return None;
    }
    // With a third destination both shift results are intermediates, so fusion would
    // emit as many instructions as the original three and buy nothing.
    let rotate = if rd == d0 {
        Rotate { rs: s0, rd, amount: right, keep_op: op1, keep_rd: d1, keep_imm: k1 as u32 }
    } else if rd == d1 {
        Rotate { rs: s0, rd, amount: right, keep_op: op0, keep_rd: d0, keep_imm: k0 as u32 }
    } else {
        return None;
    };
    Some(rotate)
}

/// How far ahead of an OR a 32-bit shift pair may be scheduled. Compilers interleave
/// independent rotates to hide latency, so the OR often follows the next rotate's first
/// shift.
const ROTATE32_WINDOW: usize = 8;

#[derive(Clone, Copy)]
enum Rotate32Shift {
    Left(u32),
    Right(u32),
}

/// A 32-bit rotate found in a window ending at an OR. `first` and `second` are the two
/// shifts, in program order; both are skipped and recomputed at the OR.
#[derive(Clone, Copy)]
struct Rotate32 {
    rs: u8,
    rd: u8,
    amount: u32,
    keep: (Rotate32Shift, u8),
    d0: u8,
    d1: u8,
    first: usize,
    second: usize,
    at: usize,
}

/// The integer registers an ALU instruction reads, and the one it writes. `None` for
/// anything that is not a pure integer ALU operation, which keeps the window analysis
/// free of faults and control flow.
fn alu_effect(inst: Inst) -> Option<(u32, u8)> {
    match inst {
        Inst::Lui { rd, .. } | Inst::Auipc { rd, .. } => Some((0, rd)),
        Inst::OpImm { rd, rs1, .. } | Inst::OpImm32 { rd, rs1, .. } => Some((1 << rs1, rd)),
        Inst::Op { rd, rs1, rs2, .. } | Inst::Op32 { rd, rs1, rs2, .. } => {
            Some((1 << rs1 | 1 << rs2, rd))
        }
        _ => None,
    }
}

fn shift32(op: AluOp32, imm: u32) -> Option<Rotate32Shift> {
    match op {
        AluOp32::Sll => Some(Rotate32Shift::Left(imm)),
        AluOp32::Srl => Some(Rotate32Shift::Right(imm)),
        _ => None,
    }
}

/// Whether the instructions between a 32-bit rotate's two shifts and its OR still leave
/// the shift results intact. Instructions that are themselves part of another fused
/// rotate are judged by their fused effect: the rotate reads its source and writes its
/// destination and the kept shift's destination. A skipped shift has no effect at its
/// old position; its result is materialized at its own OR.
fn rotate32_window_ok(
    block: &[(Inst, u8)],
    plans: &[Option<Rotate32>],
    skip: &[bool],
    r: &Rotate32,
) -> bool {
    for p in r.first + 1..r.at {
        if p == r.second || skip[p] {
            continue;
        }
        let (reads, writes) = match plans[p] {
            Some(q) => (1u32 << q.rs, [q.rd, q.keep.1]),
            None => match alu_effect(block[p].0) {
                Some((reads, w)) => (reads, [w, w]),
                None => return false,
            },
        };
        if p < r.second {
            // The second shift overwrites d1, so any earlier write to it is dead.
            for w in writes {
                if w == r.rs || w == r.d0 {
                    return false;
                }
            }
            if reads & (1 << r.d0) != 0 {
                return false;
            }
        } else {
            for w in writes {
                if w == r.rs || w == r.d0 || w == r.d1 {
                    return false;
                }
            }
            if reads & (1 << r.d0 | 1 << r.d1) != 0 || (r.d1 == r.rs && reads & (1 << r.rs) != 0) {
                return false;
            }
        }
    }
    true
}

/// Find windowed 32-bit rotates. Returns which instructions to skip and the plan for each
/// OR that replaces them. Windows may overlap in the scheduling pattern compilers emit:
/// the next rotate's first shift sits between this rotate's second shift and its OR.
fn plan_rotates32(
    block: &[(Inst, u8)],
    pcs: &[u64],
    branch_targets: &[u64],
) -> (Vec<bool>, Vec<Option<Rotate32>>) {
    let mut skip = vec![false; block.len()];
    let mut plans = vec![None; block.len()];
    for i in 0..block.len() {
        let Inst::Op { op: AluOp::Or, rd, rs1: o0, rs2: o1 } = block[i].0 else {
            continue;
        };
        let start = i.saturating_sub(ROTATE32_WINDOW);
        let mut shifts = Vec::new();
        for (p, &(inst, _)) in block.iter().enumerate().take(i).skip(start) {
            if let Inst::OpImm32 { op, rd, rs1, imm } = inst
                && let Some(shift) = shift32(op, imm as u32)
            {
                shifts.push((p, shift, rd, rs1));
            }
        }
        let mut found = None;
        'pairs: for a in 0..shifts.len() {
            for b in a + 1..shifts.len() {
                let (j, s0, d0, src0) = shifts[a];
                let (k, s1, d1, src1) = shifts[b];
                let (right, left) = match (s0, s1) {
                    (Rotate32Shift::Left(m), Rotate32Shift::Right(k)) => (k, m),
                    (Rotate32Shift::Right(k), Rotate32Shift::Left(m)) => (k, m),
                    _ => continue,
                };
                if src0 != src1
                    || d0 == 0
                    || d1 == 0
                    || d0 == d1
                    || d0 == src0
                    || right == 0
                    || left == 0
                    || right + left != 32
                {
                    continue;
                }
                if !((o0 == d0 && o1 == d1) || (o0 == d1 && o1 == d0)) {
                    continue;
                }
                let keep = if rd == d0 {
                    (s1, d1)
                } else if rd == d1 {
                    (s0, d0)
                } else {
                    continue;
                };
                // A branch entering the window would skip the shifts but still reach the
                // fused OR, which would then compute a rotate from the wrong state.
                if (j..=i).any(|p| branch_targets.contains(&pcs[p])) {
                    continue;
                }
                let r = Rotate32 {
                    rs: src0,
                    rd,
                    amount: right,
                    keep,
                    d0,
                    d1,
                    first: j,
                    second: k,
                    at: i,
                };
                // Two plans cannot share a shift: the second would need the first's
                // materialized value at a different point.
                if skip[j] || skip[k] || !rotate32_window_ok(block, &plans, &skip, &r) {
                    continue;
                }
                found = Some(r);
                break 'pairs;
            }
        }
        if let Some(r) = found {
            plans[i] = Some(r);
            skip[r.first] = true;
            skip[r.second] = true;
        }
    }
    // A later plan can change the effect of an earlier plan's window. Recheck with the
    // final set and drop anything that no longer holds.
    loop {
        let mut changed = false;
        for i in 0..plans.len() {
            if let Some(r) = plans[i]
                && !rotate32_window_ok(block, &plans, &skip, &r)
            {
                plans[i] = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        skip.fill(false);
        for r in plans.iter().flatten() {
            skip[r.first] = true;
            skip[r.second] = true;
        }
    }
    (skip, plans)
}

/// Translate `block` (as decoded at `pc`) into code that will live at word index
/// `block_word` of the code buffer, whose exit stub is at word index `stub_word`. Returns
/// `None` if not even the first instruction can be translated. Indirect jumps probe the
/// jump cache at byte offset `jump_cache` in the code buffer. Backward and indirect jumps
/// check interrupt flag `interrupt_slot` (see [`GuestMemory::interrupt_flag`]).
///
/// [`GuestMemory::interrupt_flag`]: crate::memory::GuestMemory::interrupt_flag
#[allow(clippy::too_many_arguments)]
pub fn translate(
    block: &[(Inst, u8)],
    pc: u64,
    block_word: i64,
    stub_word: i64,
    jump_cache: i64,
    interrupt_slot: u32,
    identity: bool,
    vstate: (u64, u64),
) -> Option<Translation> {
    // A block with vector instructions is translated for one exact vector configuration
    // (the one it is being translated in), and its entry guard re-checks that at run
    // time. A configuration the subset cannot express leaves `vec` empty: the scalar
    // prefix still translates (and is cached), and the first vector instruction ends the
    // block, exactly as untranslated instructions always have.
    let (vec, vec_regs) = if block.iter().any(|&(inst, _)| is_vector(inst)) {
        vector_config(vstate).map_or((None, 1), |(v, r)| (Some(v), r))
    } else {
        (None, 1)
    };
    // The entry guard is only needed while a translated vector op can run under the
    // entry's vector state: a `vsetivli` in the block re-establishes it from constants.
    let guard = vec.is_some()
        && block
            .iter()
            .take_while(|&&(inst, _)| {
                !matches!(
                    inst,
                    Inst::VSetvl { kind: VSetKind::ImmediateAvl | VSetKind::RegisterAvl, .. }
                )
            })
            .any(|&(inst, _)| is_vector(inst));
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
        vec,
        vec_regs,
        fboxed: 0,
        edge: None,
        vec_last: None,
        vec_m2: None,
        vec_m2_b: None,
        defer_m2_store: false,
    };
    // instret: add the number of instructions this block retires, up front. Every exit is
    // at the end of the block, so the count is static; this stays right under chaining.
    let instret_at = tx.a.pos();
    tx.a.add_imm(INSTRET_REG, INSTRET_REG, 0); // patched below

    // The vector entry guard (after the instret update, which its tail corrects): if
    // vtype or vl differ from what this translation assumed (a different loop, a later
    // vset), leave to the interpreter at this pc.
    if guard {
        let (vtype, vl) = vstate;
        tx.a.ldr_off(SRC0, CTX, VTYPE_OFF);
        tx.a.mov_imm(SRC1, vtype);
        tx.a.eor(SRC0, SRC0, SRC1);
        tx.a.ldr_off(SRC1, CTX, VL_OFF);
        tx.a.mov_imm(DST, vl);
        tx.a.eor(SRC1, SRC1, DST);
        tx.a.orr(SRC0, SRC0, SRC1);
        let to_tail = tx.a.pos();
        tx.a.cbnz(SRC0, to_tail); // patched to the interpret tail by emit_tails
        tx.tails.push((to_tail, Tail::Interpret { pc, retired_before: 0 }));
    }

    // Boxing entry guards: a register the block reads as a single before writing it is
    // assumed boxed - the loop-invariant float a compiler keeps in a register - and
    // checked once here instead of canonical-NaN-fixed at every use. A register that
    // is not boxed falls back to the interpreter for this pc, which is always correct.
    let assumed_fboxed = scan_fboxed_reads(block);
    tx.fboxed = assumed_fboxed;
    for r in 0..32u8 {
        if assumed_fboxed >> r & 1 == 0 {
            continue;
        }
        tx.fraw_load(r, SRC0);
        tx.a.lsr_imm(SRC1, SRC0, 32);
        tx.a.cmn_imm_w(SRC1, 1); // Z set when the upper half is all ones
        let to_tail = tx.a.pos();
        tx.a.b_cond(Cond::Ne, to_tail); // patched by emit_tails
        tx.tails.push((to_tail, Tail::Interpret { pc, retired_before: 0 }));
    }

    let mut retired = 0u32;
    let mut exits = Vec::new();
    let mut ended = false;
    // The register the previous instruction set to a constant (`lui`, `auipc`), and its value.
    let mut known: Option<(u8, u64)> = None;
    let reused = base_reuse(block);
    // Per-instruction translation state, for internalizing a backward branch (see
    // [`Edge`]) and for unrolling its body ([`Unroll`]): each instruction's guest pc
    // and code position, whether the translation carried no register fact (`ADDR` or
    // `known`) into it, and which `f` registers were known boxed there.
    let mut pcs: Vec<u64> = Vec::with_capacity(block.len());
    {
        let mut a = pc;
        for &(_, len) in block {
            pcs.push(a);
            a += u64::from(len);
        }
    }
    // Instructions a branch or jump inside this block can enter. A fused rotate spans
    // three instructions, so a branch must not be able to land in its middle.
    let branch_targets: Vec<u64> = block
        .iter()
        .enumerate()
        .filter_map(|(i, &(inst, _))| match inst {
            Inst::Branch { offset, .. } | Inst::Jal { offset, .. } => {
                Some(pcs[i].wrapping_add(sext(offset)))
            }
            _ => None,
        })
        .collect();
    let (skip32, plans32) = plan_rotates32(block, &pcs, &branch_targets);
    let defer_m2 = plan_deferred_m2_stores(block, &pcs, &branch_targets);
    let mut code_pos = vec![0usize; block.len()];
    let mut carry_clean = vec![false; block.len()];
    let mut fboxed_at = vec![0u32; block.len()];
    // The emission runs the block's instructions in order, except that an unrolled back
    // edge sends it back over the loop body for a second copy (`pass` 1); the tracker
    // resets to the loop head's entry state so the copy translates exactly like the
    // first, and the counters line back up when the main pass resumes after the body.
    let mut i = 0usize;
    let mut pass = 0u8;
    let mut unroll: Option<Unroll> = None;
    while i < block.len() {
        if pass == 1 && unroll.as_ref().is_some_and(|u| i == u.b + 1) {
            // The second copy is done: the main pass resumes, and both copies' exit
            // jumps land here.
            pass = 0;
            if let Some(u) = &unroll {
                let after = tx.a.pos();
                for site in [u.after0, u.after1].into_iter().flatten() {
                    tx.a.patch(site, after);
                }
            }
        }
        let (inst, len) = block[i];
        let len = u64::from(len);
        let at = pcs[i];
        let next = at + len;
        if pass == 0 {
            code_pos[i] = tx.a.pos();
            carry_clean[i] = tx.addr_base.is_none() && known.is_none();
            fboxed_at[i] = tx.fboxed;
        }
        tx.base_reused = reused[i];
        tx.defer_m2_store = defer_m2[i];
        if skip32[i] {
            // The shift runs as part of the rotate at the waiting OR. Its result is
            // materialized there, so only bookkeeping happens here.
            retired += 1;
            tx.retired = retired;
            if let Some(rd) = int_dest(inst) {
                if tx.addr_base == Some(rd) {
                    tx.addr_base = None;
                }
                if known.is_some_and(|(r, _)| r == rd) {
                    known = None;
                }
            }
            i += 1;
            continue;
        }
        if let Some(r) = plans32[i] {
            let a = tx.src(r.rs, SRC0);
            let d = tx.dst(r.rd);
            let keep_after = r.keep.1 == r.rs;
            let emit_shift = |tx: &mut Tx, h: Reg| {
                match r.keep.0 {
                    Rotate32Shift::Left(imm) => tx.a.lsl_imm_w(h, a, imm),
                    Rotate32Shift::Right(imm) => tx.a.lsr_imm_w(h, a, imm),
                }
                tx.a.sxtw(h, h);
            };
            if !keep_after {
                let h = tx.dst(r.keep.1);
                emit_shift(&mut tx, h);
                tx.commit(r.keep.1, h);
            }
            tx.a.ror_imm_w(d, a, r.amount);
            tx.a.sxtw(d, d);
            tx.commit(r.rd, d);
            if keep_after {
                let h = tx.dst(r.keep.1);
                emit_shift(&mut tx, h);
                tx.commit(r.keep.1, h);
            }
            retired += 1;
            tx.retired = retired;
            if tx.addr_base == Some(r.rd) || tx.addr_base == Some(r.keep.1) {
                tx.addr_base = None;
            }
            known = None;
            i += 1;
            continue;
        }
        if i + 2 < block.len()
            && !branch_targets.contains(&pcs[i + 1])
            && !branch_targets.contains(&pcs[i + 2])
            && let Some(r) = rotate_pattern(block, i)
        {
            let a = tx.src(r.rs, SRC0);
            let d = tx.dst(r.rd);
            // The kept shift still reads the original source. If its destination is
            // that source, it has to run after the rotate, which does not write rs
            // (rd and the kept destination differ).
            let keep_after = r.keep_rd == r.rs;
            if !keep_after {
                let h = tx.dst(r.keep_rd);
                match r.keep_op {
                    AluOp::Sll => tx.a.lsl_imm(h, a, r.keep_imm),
                    AluOp::Srl => tx.a.lsr_imm(h, a, r.keep_imm),
                    _ => unreachable!(),
                }
                tx.commit(r.keep_rd, h);
            }
            tx.a.ror_imm(d, a, r.amount);
            tx.commit(r.rd, d);
            if keep_after {
                let h = tx.dst(r.keep_rd);
                match r.keep_op {
                    AluOp::Sll => tx.a.lsl_imm(h, a, r.keep_imm),
                    AluOp::Srl => tx.a.lsr_imm(h, a, r.keep_imm),
                    _ => unreachable!(),
                }
                tx.commit(r.keep_rd, h);
            }
            retired += 3;
            tx.retired = retired;
            if tx.addr_base == Some(r.rd) || tx.addr_base == Some(r.keep_rd) {
                tx.addr_base = None;
            }
            known = None;
            i += 3;
            continue;
        }
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
                    // leave through a side exit when taken. A target inside the block
                    // is the loop's back edge: emit the body a second time and re-enter
                    // in place instead (see [`Unroll`] and [`Tx::emit_edge`]).
                    let bcond = tx.a.pos();
                    if pass == 1 && unroll.as_ref().is_some_and(|u| i == u.b) {
                        // The second copy's back edge: taken -> the landing pad,
                        // fall-through -> the code after the bodies.
                        tx.a.b_cond(c, bcond); // patched by `emit_edge`
                        let after = tx.a.pos();
                        tx.a.b(after); // patched to the code after the bodies
                        unroll.as_mut().unwrap().after1 = Some(after);
                        let u = unroll.as_ref().unwrap();
                        tx.edge = Some(Edge {
                            bcond,
                            t: u.t as u32,
                            m: retired - u.t as u32,
                            pc: target,
                            lt: code_pos[u.t],
                        });
                    } else if let Some(t) = if unroll.is_none() {
                        // A vset after this branch cannot affect the loop body.
                        let has_vset = block[..=i]
                            .iter()
                            .any(|&(inst, _)| matches!(inst, Inst::VSetvl { .. }));
                        internal_edge(&pcs, &carry_clean, target, i, has_vset, tx.edge)
                    } else {
                        None
                    } {
                        // The loop's back edge, discovered here: the first copy's taken
                        // path continues into the second copy right below it.
                        tx.a.b_cond(c, bcond + 2); // over the fall-through jump
                        let after = tx.a.pos();
                        tx.a.b(after); // patched to the code after the bodies
                        tx.a.add_imm(INSTRET_REG, INSTRET_REG, retired - t as u32);
                        unroll = Some(Unroll { b: i, t, after0: Some(after), after1: None });
                        tx.fboxed = fboxed_at[t];
                        tx.addr_base = None;
                        known = None;
                        retired = t as u32;
                        pass = 1;
                        i = t;
                        tx.retired = retired;
                        continue;
                    } else {
                        tx.a.b_cond(c, bcond); // patched once the tail exists
                        let side =
                            Tail::Side { pc: target, retired_before: retired, back: target <= at };
                        tx.tails.push((bcond, side));
                    }
                    tx.retired = retired;
                    known = None;
                    i += 1;
                    continue;
                }
                // b.cond taken; b fallthrough-tail; taken: the second copy (the first
                // copy's back edge), the landing pad (the second copy's), or a tail.
                let bcond = tx.a.pos();
                if pass == 1 && unroll.as_ref().is_some_and(|u| i == u.b) {
                    tx.a.b_cond(c, bcond); // patched by `emit_edge`
                    let u = unroll.as_ref().unwrap();
                    tx.edge = Some(Edge {
                        bcond,
                        t: u.t as u32,
                        m: retired - u.t as u32,
                        pc: target,
                        lt: code_pos[u.t],
                    });
                    exits.push((tx.a.pos(), Some(next)));
                    tx.exit(Tail::Static { pc: next, code: exit::NEXT });
                    ended = true;
                    break;
                }
                let t = if unroll.is_none() {
                    let has_vset =
                        block[..=i].iter().any(|&(inst, _)| matches!(inst, Inst::VSetvl { .. }));
                    internal_edge(&pcs, &carry_clean, target, i, has_vset, tx.edge)
                } else {
                    None
                };
                match t {
                    Some(t) => {
                        // The loop's back edge, discovered here: taken -> the second
                        // copy below the fall-through exit.
                        tx.a.b_cond(c, bcond); // patched to the second copy
                        exits.push((tx.a.pos(), Some(next)));
                        tx.exit(Tail::Static { pc: next, code: exit::NEXT });
                        let copy2 = tx.a.pos();
                        tx.a.patch(bcond, copy2);
                        tx.a.add_imm(INSTRET_REG, INSTRET_REG, retired - t as u32);
                        unroll = Some(Unroll { b: i, t, after0: None, after1: None });
                        tx.fboxed = fboxed_at[t];
                        tx.addr_base = None;
                        known = None;
                        retired = t as u32;
                        pass = 1;
                        i = t;
                        tx.retired = retired;
                        continue;
                    }
                    None => tx.a.b_cond(c, bcond + 2),
                }
                exits.push((tx.a.pos(), Some(next)));
                tx.exit(Tail::Static { pc: next, code: exit::NEXT });
                if t.is_none() {
                    if target <= at {
                        tx.interrupt_check(Some(target));
                    }
                    exits.push((tx.a.pos(), Some(target)));
                    tx.exit(Tail::Static { pc: target, code: exit::NEXT });
                }
                ended = true;
                break;
            }
            Inst::Ecall if tx.identity => {
                // Linux's asm-generic numbers, arguments, and return register line up
                // for these calls. close and getpid do not touch guest memory; read and
                // write can run inline when their buffer falls in one of the recently
                // mapped ranges recorded in the control area.
                let stat_passthrough = STAT_PASSTHROUGH.load(Ordering::Relaxed);
                let mut close_branches = Vec::new();
                for nr in [57u32, 172] {
                    tx.a.cmp_imm(7, nr);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Eq, b);
                    close_branches.push(b);
                }
                let mut rw_branches = Vec::new();
                for nr in [63u32, 64] {
                    tx.a.cmp_imm(7, nr);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Eq, b);
                    rw_branches.push(b);
                }
                let mut stat_branches = Vec::new();
                if stat_passthrough {
                    tx.a.cmp_imm(7, 79);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Eq, b);
                    stat_branches.push(b);
                }
                let slow = tx.a.pos();
                exits.push((slow, None));
                tx.exit(Tail::Static { pc: at, code: exit::ECALL });
                let rw = tx.a.pos();
                for b in rw_branches {
                    tx.a.patch(b, rw);
                }
                // read needs a writable buffer; write needs a readable one.
                tx.a.and_imm_w(15, 7, 1).unwrap();
                tx.a.add_imm_w(15, 15, 1);
                let mut hit_branches = Vec::new();
                for slot in 0..FAST_RANGE_COUNT {
                    let base = FAST_RANGES_OFFSET + slot * FAST_RANGE_STRIDE;
                    let mut failures = Vec::new();
                    // The version is odd while a mapping update is in progress.
                    tx.a.ldr_off_w(16, MEMBASE, base as u32);
                    tx.a.and_imm_w(17, 16, 1).unwrap();
                    let b = tx.a.pos();
                    tx.a.cbnz(17, b);
                    failures.push(b);
                    // start <= buffer
                    tx.a.ldr_off(14, MEMBASE, (base + 8) as u32);
                    tx.a.cmp(1, 14);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Lo, b);
                    failures.push(b);
                    // end = buffer + len, without wrapping
                    tx.a.add(14, 1, 2);
                    tx.a.cmp(14, 1);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Lo, b);
                    failures.push(b);
                    // end <= range end
                    tx.a.ldr_off(17, MEMBASE, (base + 16) as u32);
                    tx.a.cmp(14, 17);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Hi, b);
                    failures.push(b);
                    // the mapping has the permission this syscall needs
                    tx.a.ldr_off_w(14, MEMBASE, (base + 4) as u32);
                    tx.a.and(14, 14, 15);
                    let b = tx.a.pos();
                    tx.a.cbz(14, b);
                    failures.push(b);
                    // the seqlock did not move while we read the fields
                    tx.a.dmb_ish();
                    tx.a.ldr_off_w(14, MEMBASE, base as u32);
                    tx.a.cmp(16, 14);
                    let b = tx.a.pos();
                    tx.a.b_cond(Cond::Ne, b);
                    failures.push(b);
                    let hit = tx.a.pos();
                    tx.a.b(hit); // patched to the shared fast body
                    hit_branches.push(hit);
                    let next_slot = tx.a.pos();
                    for b in failures {
                        tx.a.patch(b, next_slot);
                    }
                }
                tx.a.b(slow);
                let fast = tx.a.pos();
                for b in close_branches {
                    tx.a.patch(b, fast);
                }
                for b in stat_branches {
                    tx.a.patch(b, fast);
                }
                for b in hit_branches {
                    tx.a.patch(b, fast);
                }
                // The ecall retires once; the block's entry counted everything before it.
                tx.a.add_imm(INSTRET_REG, INSTRET_REG, 1);
                // Host x8 is guest ra; the host ABI wants the syscall number there.
                tx.a.mov(ADDR, 8);
                tx.a.mov(8, 7); // host syscall number is guest a7
                tx.a.svc();
                tx.a.mov(8, ADDR);
                tx.interrupt_check(Some(next));
                tx.a.mov_imm(ADDR, next);
                tx.indirect(false);
                ended = true;
                break;
            }
            Inst::Ecall => {
                exits.push((tx.a.pos(), None));
                tx.exit(Tail::Static { pc: at, code: exit::ECALL });
                ended = true;
                break;
            }
            // A register-AVL `vsetvli` leaves a runtime configuration (compilers
            // strip-mine with it). End the block right here: the dispatcher looks the
            // body's first pc up under the just-written vtype/vl and translates it with
            // that configuration, so the whole loop stays in generated code instead of
            // being interpreted up to its first vector instruction.
            Inst::VSetvl { rd, rs1, rs2, kind: VSetKind::RegisterAvl } => {
                tx.vsetvli(rd, rs1, u64::from(rs2));
                retired += 1;
                exits.push((tx.a.pos(), Some(next)));
                tx.exit(Tail::Static { pc: next, code: exit::NEXT });
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
        i += 1;
    }
    if !ended {
        // The block stopped at a length or page limit: fall through to the next one.
        let (_, last_len) = block[block.len() - 1];
        let after = pcs[block.len() - 1] + u64::from(last_len);
        exits.push((tx.a.pos(), Some(after)));
        tx.exit(Tail::Static { pc: after, code: exit::NEXT });
    }
    tx.retired = retired;
    tx.a.code[instret_at] = {
        let mut a = Asm::new();
        a.add_imm(INSTRET_REG, INSTRET_REG, retired);
        a.code[0]
    };
    // The landing pad of an internalized backward edge sits behind the body (the tails
    // follow it): a taken back edge lands here, corrects instret, checks interrupts,
    // re-charges the block, and jumps back into the loop body.
    if let Some(edge) = tx.edge {
        tx.emit_edge(edge);
    }
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
        .map(|&(index, guest_pc, before, access)| {
            let (store, ff) = match access {
                Access::Load => (false, None),
                Access::Store => (true, None),
                Access::FaultOnlyFirst { base, eew, vd } => (false, Some((base, eew, vd))),
            };
            (index, super::fault::Site { guest_pc, instret_undo: retired - before, store, ff })
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
