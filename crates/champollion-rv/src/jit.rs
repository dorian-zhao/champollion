//! Basic-block JIT to arm64 (M4).
//!
//! - [`arm64`]: instruction encoder, checked against an assembler.
//! - [`code`]: executable memory (MAP_JIT and W^X on macOS).
//! - [`translate`]: guest block -> arm64, plus the enter/exit trampoline.
//! - [`fault`]: host faults in generated loads/stores -> precise guest faults.
//!
//! [`Jit::run`] is the dispatcher: it looks up (or translates) the block at the guest pc,
//! enters it, and acts on the exit code. Instructions the translator does not handle yet
//! run on the reference interpreter, one at a time, so coverage can grow incrementally
//! without ever being wrong.

pub mod arm64;
pub mod code;
pub mod fault;
pub mod profile;
pub mod translate;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::cpu::CpuState;
use crate::engine::decode_block;
use crate::interpreter::{self, Trap};
use crate::memory::{GuestMemory, MemFault};
use code::CodeBuffer;
use translate::exit;

/// Size of the code buffer; when full, all translations are dropped and rebuilt.
const CODE_SIZE: usize = 64 << 20;

/// Hasher for guest pcs (see the engine's): one multiply.
#[derive(Default)]
struct PcHasher(u64);

impl Hasher for PcHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed")
    }
    fn write_u64(&mut self, pc: u64) {
        self.0 = (pc >> 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type EnterFn = unsafe extern "C" fn(*mut CpuState, *mut u8, *const u8, u64) -> u64;

pub struct Jit {
    code: CodeBuffer,
    enter: EnterFn,
    /// Word index of the exit stub in the code buffer.
    stub_word: usize,
    /// Guest pc -> byte offset of its translation.
    blocks: HashMap<u64, usize, BuildHasherDefault<PcHasher>>,
    /// `GuestMemory::code_generation` the translations are valid for.
    generation: u64,
    faults: fault::Table,
    /// Exits waiting for their target to be translated: target pc -> byte offsets of the
    /// exits' `b` instructions, patched to jump straight to the target when it appears.
    unchained: HashMap<u64, Vec<usize>, BuildHasherDefault<PcHasher>>,
    /// Set to disable chaining (every block returns to the dispatcher), for comparison.
    pub chaining: bool,
    /// Indirect-jump cache probed inline by generated code; `(guest pc, host address)`
    /// per slot, with guest pc 1 (never a valid pc) marking an empty slot.
    jump_cache: Box<[[u64; 2]; translate::JUMP_CACHE_ENTRIES]>,
    pub stats: Stats,
    /// Sampling profiler, if enabled with [`Jit::start_profiling`].
    pub profiler: Option<profile::Profiler>,
}

/// Counters for `RVRUN_STATS`, to see where time goes outside generated code. The cheap
/// counters are always kept; the per-kind histogram costs a decode per interpreted
/// instruction, so it is collected only when `detailed` is set.
#[derive(Debug, Default)]
pub struct Stats {
    pub detailed: bool,
    /// Calls into generated code (each one a full save/restore round trip).
    pub entries: u64,
    pub translations: u64,
    pub resets: u64,
    /// Exits by code (see `translate::exit`), and NEXT exits whose target was already
    /// translated (a missed chain or jump-cache slot rather than new code).
    pub exits: [u64; 6],
    pub next_known: u64,
    /// Instructions run on the interpreter, by kind (the `Inst` variant name).
    pub interpreted: HashMap<&'static str, u64>,
}

impl Stats {
    pub fn report(&self) -> String {
        let total: u64 = self.interpreted.values().sum();
        let mut top: Vec<_> = self.interpreted.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> = top.iter().take(6).map(|(k, v)| format!("{k} {v}")).collect();
        format!(
            "jit: {} entries, {} translations, {} resets, exits next/ecall/interp/fence.i/fault \
             {:?} ({} next to known blocks), {total} interpreted [{}]",
            self.entries,
            self.translations,
            self.resets,
            self.exits,
            self.next_known,
            top.join(", ")
        )
    }
}

/// The variant name of an instruction, for statistics.
fn kind(inst: &crate::decoder::Inst) -> &'static str {
    let name = format!("{inst:?}");
    let end = name.find([' ', '{']).unwrap_or(name.len());
    // Leak once per distinct name; there are only a few dozen.
    thread_local! {
        static NAMES: std::cell::RefCell<HashMap<String, &'static str>> = Default::default();
    }
    NAMES.with(|n| {
        *n.borrow_mut().entry(name[..end].to_string()).or_insert_with_key(|k| k.clone().leak())
    })
}

impl Jit {
    pub fn new() -> std::io::Result<Self> {
        let code = CodeBuffer::new(CODE_SIZE)?;
        // SAFETY: placeholder, replaced by `reset` before any use.
        let enter: EnterFn = unsafe { std::mem::transmute::<*const u8, EnterFn>(code.base()) };
        fault::install();
        let faults = fault::Table {
            code_base: code.base() as usize,
            code_len: CODE_SIZE,
            stub: 0,
            sites: HashMap::new(),
            fault_addr: std::cell::Cell::new(0),
        };
        let mut jit = Jit {
            code,
            enter,
            stub_word: 0,
            blocks: HashMap::default(),
            generation: 0,
            faults,
            unchained: HashMap::default(),
            chaining: true,
            jump_cache: Box::new([[1, 0]; translate::JUMP_CACHE_ENTRIES]),
            stats: Stats::default(),
            profiler: None,
        };
        jit.reset();
        Ok(jit)
    }

    /// Drop every translation and regenerate the trampoline at the start of the buffer.
    fn reset(&mut self) {
        self.code.reset();
        self.stats.resets += 1;
        if let Some(p) = &mut self.profiler {
            p.reset();
        }
        self.blocks.clear();
        self.unchained.clear();
        self.jump_cache.fill([1, 0]);
        let (words, stub) = translate::trampoline();
        let at = self.code.append(&words).expect("trampoline fits");
        // SAFETY: the trampoline is a C-ABI function with this signature.
        self.enter = unsafe { std::mem::transmute::<*const u8, EnterFn>(self.code.ptr(at)) };
        self.stub_word = at / 4 + stub;
        self.faults.sites.clear();
        self.faults.stub = self.code.ptr(self.stub_word * 4) as usize;
    }

    /// Byte offset of the translation for `pc`, translating it on first use. `Ok(None)`
    /// means its first instruction must be interpreted.
    fn lookup(&mut self, pc: u64, mem: &GuestMemory) -> Result<Option<usize>, Trap> {
        if let Some(&at) = self.blocks.get(&pc) {
            return Ok(Some(at));
        }
        let block = decode_block(mem, pc)?;
        for attempt in 0..2 {
            let word = (self.code.used() / 4) as i64;
            let cache = self.jump_cache.as_ptr() as u64;
            let Some(t) = translate::translate(&block, pc, word, self.stub_word as i64, cache)
            else {
                return Ok(None);
            };
            match self.code.append(&t.code) {
                Some(at) => {
                    self.stats.translations += 1;
                    self.blocks.insert(pc, at);
                    if let Some(p) = &mut self.profiler {
                        p.blocks.push(profile::BlockInfo {
                            guest_pc: pc,
                            guest: block.iter().map(|&(inst, _)| inst).collect(),
                            host_offset: at,
                            host_words: t.code.clone(),
                        });
                    }
                    for (index, site) in t.sites {
                        self.faults.sites.insert(at + 4 * index, site);
                    }
                    if self.chaining {
                        self.chain(pc, at, &t.exits);
                    }
                    return Ok(Some(at));
                }
                None if attempt == 0 => self.reset(), // buffer full: start over
                None => panic!("a single block does not fit in the code buffer"),
            }
        }
        unreachable!()
    }

    /// Start sampling generated code (see [`profile`]); report with `profiler.report()`.
    pub fn start_profiling(&mut self) {
        self.profiler = Some(profile::Profiler::start(self.code.base() as usize, CODE_SIZE));
    }

    /// Link the new block at `at` (guest `pc`) into the graph: exits already waiting for
    /// `pc` now jump straight to it, and its own static exits jump straight to their
    /// targets if those are translated (or wait for them).
    fn chain(&mut self, pc: u64, at: usize, exits: &[translate::ExitSite]) {
        for branch in self.unchained.remove(&pc).unwrap_or_default() {
            self.patch_branch(branch, at);
        }
        for exit in exits {
            let Some(target) = exit.target else { continue };
            let branch = at + 4 * exit.branch;
            match self.blocks.get(&target) {
                Some(&dest) => self.patch_branch(branch, dest),
                None => self.unchained.entry(target).or_default().push(branch),
            }
        }
    }

    /// Repoint the `b` at byte offset `branch` to byte offset `dest`.
    fn patch_branch(&mut self, branch: usize, dest: usize) {
        let delta = (dest as i64 - branch as i64) / 4;
        self.code.write(branch, &[arm64::encode_branch_to(0x1400_0000, delta)]);
    }

    /// Run until a trap (including `ecall`), like [`interpreter::run`] with no step limit.
    pub fn run(&mut self, cpu: &mut CpuState, mem: &GuestMemory) -> Trap {
        loop {
            let generation = mem.code_generation();
            if generation != self.generation {
                self.reset();
                self.generation = generation;
            }
            let at = match self.lookup(cpu.pc, mem) {
                Ok(Some(at)) => {
                    // Let generated indirect jumps find this block without us next time.
                    self.jump_cache[translate::jump_cache_index(cpu.pc)] =
                        [cpu.pc, self.code.ptr(at) as u64];
                    at
                }
                Ok(None) => {
                    if let Err(trap) = interpreter::step(cpu, mem) {
                        return trap;
                    }
                    continue;
                }
                Err(trap) => return trap,
            };
            // SAFETY: `at` holds a translation produced for this guest memory layout; it
            // only touches `cpu`, guest memory below `host_base() + GUEST_SIZE` (addresses
            // are masked), and the host stack, and returns through the exit stub.
            // Generated FP code runs under the guest's rounding mode; RMM (and reserved frm
            // values, which make dyn instructions illegal) can only be honored by the
            // interpreter.
            let Some(fpcr) = translate::guest_fpcr(cpu.fcsr >> 5 & 0b111) else {
                if let Err(trap) = interpreter::step(cpu, mem) {
                    return trap;
                }
                continue;
            };
            self.stats.entries += 1;
            let code = {
                let _active = fault::Active::enter(&self.faults);
                unsafe { (self.enter)(cpu, mem.host_base(), self.code.ptr(at), fpcr) }
            };
            self.stats.exits[code.min(5) as usize] += 1;
            if code == exit::NEXT && self.blocks.contains_key(&cpu.pc) {
                self.stats.next_known += 1;
            }
            match code {
                exit::NEXT => {}
                exit::ECALL => return Trap::Ecall { pc: cpu.pc },
                exit::INTERPRET => {
                    if self.stats.detailed
                        && let Ok(block) = decode_block(mem, cpu.pc)
                    {
                        *self.stats.interpreted.entry(kind(&block[0].0)).or_default() += 1;
                    }
                    if let Err(trap) = interpreter::step(cpu, mem) {
                        return trap;
                    }
                }
                exit::FENCE_I => self.reset(),
                exit::FAULT_LOAD | exit::FAULT_STORE => {
                    let fault = MemFault { addr: self.faults.fault_addr.get() };
                    let pc = cpu.pc;
                    return if code == exit::FAULT_LOAD {
                        Trap::LoadFault { pc, fault }
                    } else {
                        Trap::StoreFault { pc, fault }
                    };
                }
                other => panic!("bad exit code {other}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::enc::*;
    use crate::memory::Prot;

    const CODE: u64 = 0x10000;
    const DATA: u64 = 0x20000;

    fn setup(prog: &[u32]) -> GuestMemory {
        let mem = GuestMemory::new().unwrap();
        mem.map(CODE, 0x1000, Prot::READ | Prot::EXEC).unwrap();
        mem.map(DATA, 0x1000, Prot::RW).unwrap();
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(CODE, &bytes).unwrap();
        mem
    }

    /// Run `prog` on the interpreter and on the JIT until a trap; both must agree exactly.
    fn same_as_interpreter(prog: &[u32]) -> (Trap, CpuState) {
        let mem = setup(prog);
        let mut oracle = CpuState::new(CODE);
        oracle.x[11] = DATA;
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem) {
                break t;
            }
        };
        let mem = setup(prog);
        let mut cpu = CpuState::new(CODE);
        cpu.x[11] = DATA;
        let got = Jit::new().unwrap().run(&mut cpu, &mem);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        (got, cpu)
    }

    #[test]
    fn faults_are_precise() {
        // Pinned (a0, s0) and unpinned (t0) registers written before the fault must be
        // visible; the instruction after the fault must not run; instret counts 3.
        let (trap, cpu) = same_as_interpreter(&[
            addi(10, 0, 5),       // a0 = 5
            addi(5, 0, 7),        // t0 = 7 (unpinned)
            addi(8, 0, 9),        // s0 = 9
            load(3, 12, 0, 0x40), // ld a2, 0x40(zero): unmapped page 0
            addi(10, 0, 99),      // must not run
            ECALL,
        ]);
        assert!(matches!(trap, Trap::LoadFault { pc, .. } if pc == CODE + 12));
        assert_eq!((cpu.x[10], cpu.x[5], cpu.x[8], cpu.instret), (5, 7, 9, 3));

        // A store fault, in a later block reached through a jump.
        same_as_interpreter(&[
            store(3, 11, 10, 0), // sd a0, 0(a1): fine
            jal(0, 8),           // -> +8
            0,                   // (skipped)
            addi(5, 0, 1),
            store(2, 0, 5, 0x123), // sw t0, 0x123(zero): faults
            ECALL,
        ]);
    }

    #[test]
    fn auipc_and_lui_jumps_become_direct() {
        let (_, cpu) = same_as_interpreter(&[
            auipc(1, 0),       // 0x00: ra = CODE
            jalr(1, 1, 17),    // 0x04: -> 0x10 (bit 0 cleared), ra = 0x08; rd == rs1
            addi(10, 10, 100), // 0x08
            ECALL,             // 0x0c
            addi(10, 0, 1),    // 0x10
            lui(7, 0x10),      // 0x14: t2 = CODE
            jalr(0, 7, 0x24),  // 0x18: -> 0x24
            0,                 // 0x1c (skipped)
            0,                 // 0x20 (skipped)
            jalr(0, 1, 0),     // 0x24: indirect, back to 0x08
        ]);
        assert_eq!((cpu.x[10], cpu.x[1], cpu.x[7]), (101, CODE + 8, CODE));
    }

    #[test]
    fn atomics_match_the_interpreter() {
        let (lr_d, sc_d, lr_w, sc_w) = (0b00010, 0b00011, 0b00010, 0b00011);
        // a1 = DATA. lr/sc success, sc without a reservation, sc to another address,
        // word AMOs (sign extension), amoand (bit-clear with inverted operand), min/max.
        let (_, cpu) = same_as_interpreter(&[
            addi(12, 0, -5),                 // a2 = -5
            store(3, 11, 12, 0),             // [DATA] = -5
            amo(lr_d, 0b011, 13, 11, 0),     // lr.d a3, (a1)
            amo(sc_d, 0b011, 14, 11, 12),    // sc.d a4, a2, (a1): success -> 0
            amo(sc_d, 0b011, 15, 11, 12),    // sc.d again: no reservation -> 1
            amo(lr_w, 0b010, 16, 11, 0),     // lr.w a6, (a1): sign-extended
            addi(5, 11, 8),                  // t0 = DATA + 8
            amo(sc_w, 0b010, 17, 5, 12),     // sc.w a7, a2, (t0): wrong address -> 1
            addi(28, 0, 7),                  // t3 = 7 (unpinned)
            amo(0b10100, 0b010, 29, 11, 28), // amomax.w t4, t3, (a1): old = -5
            amo(0b01100, 0b011, 30, 11, 28), // amoand.d t5, t3, (a1)
            amo(0b11100, 0b011, 31, 11, 12), // amomaxu.d t6, a2, (a1)
            amo(0b00001, 0b011, 0, 11, 28),  // amoswap.d zero, t3, (a1)
            load(3, 18, 11, 0),              // ld s2, 0(a1)
            ECALL,
        ]);
        assert_eq!((cpu.x[14], cpu.x[15], cpu.x[17]), (0, 1, 1));
        assert_eq!(cpu.x[16] as i64, -5);
        assert_eq!(cpu.x[18], 7);

        // A misaligned atomic leaves through the interpreter, which raises the trap.
        let (trap, _) = same_as_interpreter(&[addi(11, 11, 4), amo(0, 0b011, 13, 11, 12), ECALL]);
        assert!(matches!(trap, Trap::MisalignedAtomic { .. }));

        // A faulting lr leaves the reservation (and rd) untouched.
        let (trap, cpu) = same_as_interpreter(&[
            amo(lr_d, 0b011, 13, 11, 0), // reserve DATA
            addi(5, 0, 64),
            amo(lr_d, 0b011, 13, 5, 0), // lr.d from page 0: faults
            ECALL,
        ]);
        assert!(matches!(trap, Trap::LoadFault { .. }));
        assert_eq!(cpu.reserved_addr, DATA);
    }

    #[test]
    fn fp_moves_match_the_interpreter() {
        let (fld, fsd, flw, fsw) = (0b011, 0b011, 0b010, 0b010);
        let (_, cpu) = same_as_interpreter(&[
            addi(12, 0, -1),
            op_fp(0b11110, 0b01, 0, 1, 12, 0),    // fmv.d.x f1, a2
            op_fp(0b11110, 0b00, 0, 2, 12, 0),    // fmv.w.x f2, a2: NaN-boxed
            fstore(fsd, 11, 1, 0),                // fsd f1, 0(a1)
            fload(flw, 3, 11, 0),                 // flw f3, 0(a1): boxed
            fstore(fsw, 11, 2, 8),                // fsw f2, 8(a1)
            fload(fld, 4, 11, 8),                 // fld f4, 8(a1)
            op_fp(0b11100, 0b00, 0, 13, 3, 0),    // fmv.x.w a3, f3: sign-extended
            op_fp(0b11100, 0b01, 0, 14, 4, 0),    // fmv.x.d a4, f4
            op_fp(0b00100, 0b01, 0b001, 5, 1, 1), // fsgnjn.d f5, f1, f1 (fneg)
            op_fp(0b00100, 0b01, 0b010, 6, 5, 1), // fsgnjx.d f6, f5, f1
            ECALL,
        ]);
        assert_eq!(cpu.f[2], 0xffff_ffff_ffff_ffff);
        assert_eq!(cpu.x[13], u64::MAX);
        assert_eq!(cpu.f[5], u64::MAX >> 1);
    }

    #[test]
    fn fp_arithmetic_matches_the_interpreter() {
        let dyn_ = 0b111;
        let fp = |f5, fmt, rm, rd, rs1, rs2| op_fp(f5, fmt, rm, rd, rs1, rs2);
        let (_, cpu) = same_as_interpreter(&[
            addi(12, 0, 3),
            fp(0b11010, 0b01, dyn_, 1, 12, 0), // fcvt.d.w f1, a2 = 3.0
            addi(12, 0, 7),
            fp(0b11010, 0b01, dyn_, 2, 12, 0),  // f2 = 7.0
            fp(0b00011, 0b01, dyn_, 3, 2, 1),   // fdiv.d f3 = 7/3: inexact
            fp(0b00011, 0b01, dyn_, 4, 2, 0),   // fdiv.d f4 = 7/0: divide by zero
            fp(0b00001, 0b01, dyn_, 5, 4, 4),   // fsub.d f5 = inf - inf: invalid, NaN
            fp(0b11000, 0b01, 0b001, 13, 5, 0), // fcvt.w.d a3, f5 (rtz): NaN -> INT_MAX
            fp(0b11000, 0b01, 0b001, 14, 3, 1), // fcvt.wu.d a4, f3: 2
            fp(0b01000, 0b00, dyn_, 6, 3, 1),   // fcvt.s.d f6, f3: boxed single
            fp(0b00010, 0b00, dyn_, 7, 6, 6),   // fmul.s f7, f6, f6
            fp(0b00000, 0b00, dyn_, 8, 3, 6),   // fadd.s f8, f3 (not boxed!) + f6
            fp(0b10100, 0b01, 0b001, 15, 3, 2), // flt.d a5, f3, f2: 1
            fp(0b10100, 0b01, 0b010, 16, 5, 5), // feq.d a6, NaN, NaN: 0
            fp(0b01000, 0b01, dyn_, 9, 7, 0),   // fcvt.d.s f9, f7
            fp(0b01011, 0b01, dyn_, 10, 2, 0),  // fsqrt.d f10, f2
            ECALL,
        ]);
        assert_eq!(cpu.x[13], i32::MAX as u64);
        assert_eq!((cpu.x[14], cpu.x[15], cpu.x[16]), (2, 1, 0));
        assert_eq!(cpu.f[8], 0xffff_ffff_7fc0_0000, "unboxed operand reads as canonical NaN");
        assert_eq!(cpu.fcsr & 0x1f, 0b11001, "NV, DZ, and NX accrued");
    }

    #[test]
    fn pinned_and_unpinned_fp_registers_agree() {
        // f28-f31 live in memory, the rest in host registers; mix them in every role, and
        // use the fmv/fneg/fabs fast paths (fsgnj* with rs1 == rs2).
        let dyn_ = 0b111;
        let fp = |f5, fmt, rm, rd, rs1, rs2| op_fp(f5, fmt, rm, rd, rs1, rs2);
        let (_, cpu) = same_as_interpreter(&[
            addi(12, 0, -9),
            fp(0b11010, 0b01, dyn_, 28, 12, 2), // fcvt.d.l f28, a2 = -9 (unpinned)
            fp(0b00100, 0b01, 0b010, 1, 28, 28), // fabs.d f1, f28 = 9
            fp(0b00100, 0b01, 0b001, 29, 1, 1), // fneg.d f29, f1 = -9 (unpinned)
            fp(0b00100, 0b01, 0b000, 30, 29, 29), // fmv.d f30, f29
            fp(0b00010, 0b01, dyn_, 31, 30, 1), // fmul.d f31, f30, f1 = -81
            fp(0b01011, 0b01, dyn_, 20, 1, 0),  // fsqrt.d f20, f1 = 3 (pinned high)
            fstore(0b011, 11, 31, 0),           // fsd f31, 0(a1)
            fload(0b011, 9, 11, 0),             // fld f9 (callee-saved host v9)
            fp(0b00000, 0b01, dyn_, 27, 9, 20), // fadd.d f27 = -78
            fp(0b11000, 0b01, 0b001, 13, 27, 2), // fcvt.l.d a3 = -78
            fp(0b00100, 0b01, 0b001, 28, 27, 30), // fsgnjn.d f28, f27, f30 (general)
            fp(0b11100, 0b01, 0, 14, 28, 0),    // fmv.x.d a4, f28
            ECALL,
        ]);
        assert_eq!(cpu.x[13] as i64, -78);
        assert_eq!(f64::from_bits(cpu.x[14]), 78.0);
    }

    #[test]
    fn untranslated_instructions_fall_back_to_the_interpreter() {
        // CSR accesses are not translated: they run on the interpreter mid-stream, and
        // instret stays exact around them.
        let csr = |rd: u32, n: u32| n << 20 | 0b010 << 12 | rd << 7 | 0x73; // csrrs rd, n, zero
        let (trap, cpu) = same_as_interpreter(&[
            addi(12, 0, 3),
            csr(13, 0xc02), // rdinstret a3 = 1
            addi(12, 12, 1),
            csr(14, 0xc02), // rdinstret a4 = 3
            ECALL,
        ]);
        assert!(matches!(trap, Trap::Ecall { .. }));
        assert_eq!((cpu.x[12], cpu.x[13], cpu.x[14]), (4, 1, 3));
    }
}
