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
mod persist;
pub mod profile;
pub mod translate;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::cpu::CpuState;
use crate::engine::decode_block;
use crate::interpreter::{self, Trap};
use crate::memory::{GuestMemory, MemFault, Tlb, TlbMemory, TlbReader};
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
    /// Generated code, then (in the data area) the indirect-jump cache probed inline by
    /// generated code: `(guest pc, host address)` per slot, with guest pc 1 (never a valid
    /// pc) marking an empty slot.
    code: CodeBuffer,
    enter: EnterFn,
    /// Word index of the exit stub in the code buffer.
    stub_word: usize,
    /// Guest pc -> byte offset of its translation.
    blocks: HashMap<u64, usize, BuildHasherDefault<PcHasher>>,
    /// `GuestMemory::code_generation` the translations are valid for.
    generation: u64,
    /// Instruction fetch for translation, valid for `GuestMemory::generation` `tlb_generation`.
    tlb: Tlb,
    tlb_generation: u64,
    /// Tells unmapping which TLB generation the fetch may still use (created on first use).
    tlb_reader: Option<TlbReader>,
    faults: fault::Table,
    /// Exits waiting for their target to be translated: target pc -> byte offsets of the
    /// exits' `b` instructions, patched to jump straight to the target when it appears.
    unchained: HashMap<u64, Vec<usize>, BuildHasherDefault<PcHasher>>,
    /// Slot of the interrupt flag generated code checks.
    interrupt_slot: u32,
    /// Whether the translations are for guest memory in the identity layout.
    identity: bool,
    /// Whether translations were never dropped since the JIT started (or restored), so all of
    /// them come from the startup image and may be saved (see [`persist`]).
    pristine: bool,
    /// Set to disable chaining (every block returns to the dispatcher), for comparison.
    pub chaining: bool,
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
    /// Blocks loaded from a persistent code cache.
    pub restored: u64,
    pub resets: u64,
    /// Exits by code (see `translate::exit`), and NEXT exits whose target was already
    /// translated (a missed chain or jump-cache slot rather than new code).
    pub exits: [u64; 7],
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
            "jit: {} entries, {} translations, {} restored, {} resets, exits \
             next/ecall/interp/fence.i/fault/interrupt {:?} ({} next to known blocks), {total} \
             interpreted [{}]",
            self.entries,
            self.translations,
            self.restored,
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
        let code = CodeBuffer::new(CODE_SIZE, translate::JUMP_CACHE_ENTRIES * 16)?;
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
            tlb: Tlb::default(),
            tlb_generation: u64::MAX,
            tlb_reader: None,
            faults,
            unchained: HashMap::default(),
            chaining: true,
            pristine: true,
            interrupt_slot: 0,
            identity: false,
            stats: Stats::default(),
            profiler: None,
        };
        jit.reset();
        Ok(jit)
    }

    /// Drop every translation and regenerate the trampoline at the start of the buffer.
    fn reset(&mut self) {
        if !self.blocks.is_empty() {
            self.pristine = false;
        }
        self.code.reset();
        self.stats.resets += 1;
        if let Some(p) = &mut self.profiler {
            p.reset();
        }
        self.blocks.clear();
        self.unchained.clear();
        self.jump_cache().fill([1, 0]);
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
        // Fetch through a TLB: a lookup per instruction in the mapping table showed up as
        // 11% of a short program's run time.
        let reader = self.tlb_reader.get_or_insert_with(|| mem.tlb_reader());
        let generation = reader.enter(mem);
        if generation != self.tlb_generation {
            self.tlb.flush();
            self.tlb_generation = generation;
        }
        let block = decode_block(&TlbMemory { mem, tlb: &self.tlb }, pc, false);
        reader.leave();
        let block = block?;
        for attempt in 0..2 {
            let word = (self.code.used() / 4) as i64;
            let cache = self.code.data_offset() as i64;
            let stub = self.stub_word as i64;
            let slot = self.interrupt_slot;
            let identity = self.identity;
            let Some(t) = translate::translate(&block, pc, word, stub, cache, slot, identity)
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
                            host_len: t.code.len(),
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

    /// The jump cache, at the start of the code buffer's data area (see the field docs).
    fn jump_cache(&mut self) -> &mut [[u64; 2]] {
        let data = self.code.data_mut();
        assert!(data.len() >= translate::JUMP_CACHE_ENTRIES * 16);
        // SAFETY: the data area is page-aligned and large enough; any bits are valid u64s.
        unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), translate::JUMP_CACHE_ENTRIES)
        }
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
            if let Some(trap) = self.run_interruptible(cpu, mem) {
                return trap;
            }
        }
    }

    /// Generated code checks this thread's interrupt flag (slot `slot` of
    /// [`GuestMemory::interrupt_flag`]); slot 0 by default. Set it before running anything.
    pub fn set_interrupt_slot(&mut self, slot: usize) {
        assert!(self.blocks.is_empty(), "translations check the old slot");
        self.interrupt_slot = slot as u32;
    }

    /// Like [`Jit::run`], but return `None` when generated code sees the interrupt flag set
    /// (the caller delivers signals and clears it), with `cpu` at the next instruction.
    pub fn run_interruptible(&mut self, cpu: &mut CpuState, mem: &GuestMemory) -> Option<Trap> {
        loop {
            // Loops through the interpreter (untranslated instructions) pass here.
            if mem
                .interrupt_flag(self.interrupt_slot as usize)
                .load(std::sync::atomic::Ordering::Relaxed)
                != 0
            {
                return None;
            }
            let generation = mem.code_generation();
            if generation != self.generation || mem.is_identity() != self.identity {
                self.reset();
                self.generation = generation;
                self.identity = mem.is_identity();
            }
            let at = match self.lookup(cpu.pc, mem) {
                Ok(Some(at)) => {
                    // Let generated indirect jumps find this block without us next time.
                    self.jump_cache()[translate::jump_cache_index(cpu.pc)] =
                        [cpu.pc, self.code.ptr(at) as u64];
                    at
                }
                Ok(None) => {
                    if let Err(trap) = interpreter::step(cpu, mem) {
                        return Some(trap);
                    }
                    continue;
                }
                Err(trap) => return Some(trap),
            };
            // SAFETY: `at` holds a translation produced for this guest memory layout; it
            // only touches `cpu`, guest memory below `host_base() + GUEST_SIZE` (addresses
            // are masked), and the host stack, and returns through the exit stub.
            // Generated FP code runs under the guest's rounding mode; RMM (and reserved frm
            // values, which make dyn instructions illegal) can only be honored by the
            // interpreter.
            let Some(fpcr) = translate::guest_fpcr(cpu.fcsr >> 5 & 0b111) else {
                if let Err(trap) = interpreter::step(cpu, mem) {
                    return Some(trap);
                }
                continue;
            };
            self.stats.entries += 1;
            self.code.commit();
            let code = {
                let _active = fault::Active::enter(&self.faults);
                let membase =
                    if self.identity { mem.control_area().cast() } else { mem.host_base() };
                unsafe { (self.enter)(cpu, membase, self.code.ptr(at), fpcr) }
            };
            self.stats.exits[code.min(6) as usize] += 1;
            if code == exit::NEXT && self.blocks.contains_key(&cpu.pc) {
                self.stats.next_known += 1;
            }
            match code {
                exit::NEXT => {}
                exit::ECALL => return Some(Trap::Ecall { pc: cpu.pc }),
                exit::INTERPRET => {
                    if self.stats.detailed
                        && let Ok(block) = decode_block(mem, cpu.pc, false)
                    {
                        *self.stats.interpreted.entry(kind(&block[0].0)).or_default() += 1;
                    }
                    if let Err(trap) = interpreter::step(cpu, mem) {
                        return Some(trap);
                    }
                }
                exit::FENCE_I => self.reset(),
                exit::INTERRUPT => return None,
                exit::FAULT_LOAD | exit::FAULT_STORE => {
                    let host = self.faults.fault_addr.get();
                    let fault = MemFault { addr: host.wrapping_sub(mem.host_base() as u64) };
                    let pc = cpu.pc;
                    return Some(if code == exit::FAULT_LOAD {
                        Trap::LoadFault { pc, fault }
                    } else {
                        Trap::StoreFault { pc, fault }
                    });
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
    use crate::memory::{GUEST_SIZE, Prot};

    const CODE: u64 = 0x10000;
    const DATA: u64 = 0x20000;

    fn setup(prog: &[u32]) -> GuestMemory {
        let mem = GuestMemory::new().unwrap();
        load_program(&mem, prog);
        mem
    }

    /// Map code (holding `prog`) and data pages.
    fn load_program(mem: &GuestMemory, prog: &[u32]) {
        mem.map(CODE, 0x1000, Prot::READ | Prot::EXEC).unwrap();
        mem.map(DATA, 0x1000, Prot::RW).unwrap();
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(CODE, &bytes).unwrap();
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
    fn division_by_x0_matches_the_interpreter() {
        // Found by the fuzzer: `divuw rd, rs, zero` returned 0 instead of all ones.
        let (_, cpu) = same_as_interpreter(&[
            addi(10, 0, 7),
            op_32(4, 1, 12, 10, 0), // divw a2, a0, zero
            op_32(5, 1, 13, 10, 0), // divuw a3, a0, zero
            op_32(6, 1, 14, 10, 0), // remw a4, a0, zero
            op(4, 1, 15, 10, 0),    // div a5, a0, zero
            ECALL,
        ]);
        assert_eq!((cpu.x[12], cpu.x[13], cpu.x[14], cpu.x[15]), (u64::MAX, u64::MAX, 7, u64::MAX));
    }

    #[test]
    fn restored_translations_run_like_fresh_ones() {
        let prog = [
            addi(10, 10, 1),      // 0x00: a0 += 1
            op_imm(7, 12, 10, 1), // 0x04: andi a2, a0, 1
            beq(12, 0, 8),        // 0x08: even -> 0x10
            store(3, 11, 10, 0),  // 0x0c: sd a0, 0(a1)
            addi(14, 0, 5),       // 0x10: a4 = 5
            b(4, 10, 14, -0x14),  // 0x14: blt a0, a4, 0x00
            load(3, 12, 0, 0x40), // 0x18: faults
        ];
        let run = |jit: &mut Jit, mem: &GuestMemory| {
            let mut cpu = CpuState::new(CODE);
            cpu.x[11] = DATA;
            (jit.run(&mut cpu, mem), cpu)
        };
        let mem = setup(&prog);
        let mut first = Jit::new().unwrap();
        let want = run(&mut first, &mem);
        assert!(first.cacheable());
        let saved = first.save(42).unwrap();

        let mem = setup(&prog);
        let mut second = Jit::new().unwrap();
        assert_eq!(second.restore(&saved, 42, &mem), Ok(first.blocks.len()));
        assert_eq!(run(&mut second, &mem), want, "restored code, chains, and fault sites work");
        assert_eq!(second.stats.translations, 0);

        // Snapshots for another key, truncated, or padded are refused, leaving the JIT empty
        // but working.
        let mut third = Jit::new().unwrap();
        assert!(third.restore(&saved, 7, &mem).is_err());
        assert!(third.restore(&saved[..saved.len() - 1], 42, &mem).is_err());
        assert!(third.restore(&[&saved[..], &[0]].concat(), 42, &mem).is_err());
        assert!(third.blocks.is_empty());
        assert_eq!(run(&mut third, &setup(&prog)), want);
    }

    #[test]
    fn side_exits_match_the_interpreter() {
        // One block with two mid-block branches; the loop takes each both ways, so side
        // exits (with their instret correction) and fall-through paths both run.
        let (_, cpu) = same_as_interpreter(&[
            addi(10, 10, 1),      // 0x00: a0 += 1
            op_imm(7, 12, 10, 1), // 0x04: andi a2, a0, 1
            beq(12, 0, 8),        // 0x08: even -> 0x10 (side exit)
            addi(13, 13, 1),      // 0x0c: a3 += 1 (odd)
            addi(14, 0, 5),       // 0x10: a4 = 5
            b(4, 10, 14, -0x14),  // 0x14: blt a0, a4, 0x00 (side exit, backward)
            store(3, 11, 10, 0),  // 0x18: sd a0, 0(a1)
            ECALL,                // 0x1c
        ]);
        assert_eq!((cpu.x[10], cpu.x[13]), (5, 3));
        // A fault after a mid-block branch fell through.
        let (trap, cpu) = same_as_interpreter(&[
            addi(10, 0, 1),       // a0 = 1
            beq(10, 0, 8),        // not taken
            load(3, 12, 0, 0x40), // ld a2, 0x40(zero): faults
            ECALL,
        ]);
        assert!(matches!(trap, Trap::LoadFault { .. }));
        assert_eq!(cpu.instret, 2);
    }

    #[test]
    fn base_register_cache_matches_the_interpreter() {
        let (_, cpu) = same_as_interpreter(&[
            addi(10, 0, 0x55),     // a0 = 0x55
            store(3, 11, 10, 0),   // sd a0, 0(a1): sets up a1's host address
            store(3, 11, 10, 8),   // sd a0, 8(a1): reuses it
            load(3, 12, 11, 8),    // ld a2, 8(a1)
            addi(11, 11, 16),      // a1 += 16: the cached address is stale now
            store(2, 11, 12, -4),  // sw a2, -4(a1): negative offset (ldur)
            load(2, 13, 11, -4),   // lw a3, -4(a1)
            op(4, 1, 14, 13, 10),  // div a4, a3, a0: borrows the address register
            load(0, 14, 11, -16),  // lb a4, -16(a1)
            addi(3, 11, 0),        // gp = a1: an unpinned base
            store(3, 3, 10, 24),   // sd a0, 24(gp)
            load(3, 15, 3, 24),    // ld a5, 24(gp)
            load(3, 3, 3, 24),     // ld gp, 24(gp): overwrites its own base
            load(2, 16, 11, 1022), // lw a6, 1022(a1): misaligned, no immediate form
            load(3, 17, 11, 2000), // ld a7, 2000(a1): large scaled offset
            ECALL,
        ]);
        assert_eq!((cpu.x[12], cpu.x[13], cpu.x[15], cpu.x[3]), (0x55, 0x55, 0x55, 0x55));
    }

    #[test]
    fn offsets_past_the_guest_space_fault_precisely() {
        // Base just below the top of the guest space, offset past it: the access lands in
        // the guard region above, and the fault reports the unmasked address.
        let (trap, _) = same_as_interpreter(&[
            addi(5, 0, 1),
            op_imm(1, 5, 5, 38), // t0 = GUEST_SIZE
            addi(5, 5, -8),      // t0 = GUEST_SIZE - 8
            load(3, 10, 5, 16),  // ld a0, 16(t0)
            ECALL,
        ]);
        assert!(matches!(trap, Trap::LoadFault { fault, .. } if fault.addr == GUEST_SIZE + 8));
        // And below the bottom.
        let (trap, _) = same_as_interpreter(&[
            addi(5, 0, 8),        // t0 = 8
            store(3, 5, 10, -16), // sd a0, -16(t0)
            ECALL,
        ]);
        assert!(
            matches!(trap, Trap::StoreFault { fault, .. } if fault.addr == 8u64.wrapping_sub(16))
        );
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

    /// Differential fuzzing: random programs must leave the JIT (fresh, and restored from a
    /// saved snapshot) and the interpreter in exactly the same state (registers, fcsr,
    /// reservation, instret, trap, and memory).
    mod fuzz {
        use super::*;

        struct Rng(u64);

        impl Rng {
            fn next(&mut self) -> u64 {
                // xorshift64*
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
            fn pick<T: Copy>(&mut self, items: &[T]) -> T {
                items[self.below(items.len() as u64) as usize]
            }
            fn chance(&mut self, one_in: u64) -> bool {
                self.below(one_in) == 0
            }
        }

        /// Base registers, pointing into the data page (only moved by a few bytes, never
        /// overwritten): a1 (pinned) -> DATA, s0 (pinned) -> DATA + 0x800, t3 (unpinned)
        /// -> DATA + 0x400. s1 counts loop iterations.
        const BASES: [(u32, u64); 3] = [(11, DATA), (8, DATA + 0x800), (28, DATA + 0x400)];
        const COUNTER: u32 = 9;

        fn int_dest(rng: &mut Rng) -> u32 {
            loop {
                let r = rng.below(32) as u32;
                if !matches!(r, 8 | 9 | 11 | 28) {
                    return r;
                }
            }
        }

        fn reg(rng: &mut Rng) -> u32 {
            rng.below(32) as u32
        }

        /// A load or store address: a base register and an offset that stays inside the
        /// first data page, or rarely any offset (faulting below the data). Random registers
        /// are no bases: addresses at or above 2^38 alias in generated code (a known
        /// deviation).
        fn mem_operand(rng: &mut Rng, size: u64) -> (u32, i32) {
            let (base, at) = rng.pick(&BASES);
            if rng.chance(64) {
                return (base, rng.below(4096) as i32 - 2048);
            }
            let addr = DATA + rng.below(0x1000 - size + 1);
            let addr = if rng.chance(4) { addr } else { addr & !(size - 1) };
            (base, (addr as i64 - at as i64) as i32)
        }

        /// A body item: a plain instruction, or a forward jump whose target is chosen once
        /// the body is laid out (so jumps only ever land on the first word of an item).
        enum Item {
            Plain(u32),
            /// Conditional branch with funct3 and sources.
            Branch(u32, u32, u32),
            /// `jal rd`.
            Jal(u32),
            /// `auipc t, 0; jalr rd, t, off`: a direct far jump.
            Direct(u32, u32),
            /// `auipc t, 0; addi t, t, off; jalr rd, t, 0`: an indirect jump, to an odd
            /// address (bit 0 is cleared by `jalr`) if the flag is set.
            Indirect(u32, u32, bool),
        }

        impl Item {
            fn len(&self) -> usize {
                match self {
                    Item::Direct(..) => 2,
                    Item::Indirect(..) => 3,
                    _ => 1,
                }
            }
        }

        fn instruction(rng: &mut Rng) -> Item {
            use Item::Plain;
            match rng.below(20) {
                0..=4 => {
                    let (f3, f7) = rng.pick(&[
                        (0, 0),
                        (0, 0x20),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (5, 0x20),
                        (6, 0),
                        (7, 0),
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                    ]);
                    Plain(op(f3, f7, int_dest(rng), reg(rng), reg(rng)))
                }
                5..=7 => {
                    let f3 = rng.below(8) as u32;
                    let imm = match f3 {
                        1 => rng.below(64) as i32,
                        5 => rng.below(64) as i32 | if rng.chance(2) { 0x400 } else { 0 },
                        _ => rng.below(4096) as i32 - 2048,
                    };
                    Plain(op_imm(f3, int_dest(rng), reg(rng), imm))
                }
                8 => {
                    let (f3, f7) = rng.pick(&[
                        (0, 0),
                        (0, 0x20),
                        (1, 0),
                        (5, 0),
                        (5, 0x20),
                        (0, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                    ]);
                    if rng.chance(2) {
                        Plain(op_32(f3, f7, int_dest(rng), reg(rng), reg(rng)))
                    } else {
                        let (f3, imm) = match f3 {
                            1 => (1, rng.below(32) as i32),
                            5 => (5, rng.below(32) as i32 | if f7 != 0 { 0x400 } else { 0 }),
                            _ => (0, rng.below(4096) as i32 - 2048),
                        };
                        Plain(op_imm_32(f3, int_dest(rng), reg(rng), imm))
                    }
                }
                9 => Plain(if rng.chance(2) {
                    lui(int_dest(rng), rng.next() as u32)
                } else {
                    auipc(int_dest(rng), rng.next() as u32)
                }),
                10..=11 => {
                    let f3 = rng.below(7) as u32;
                    let (base, off) = mem_operand(rng, 1 << (f3 & 3));
                    Plain(load(f3, int_dest(rng), base, off))
                }
                12..=13 => {
                    let f3 = rng.below(4) as u32;
                    let (base, off) = mem_operand(rng, 1 << f3);
                    Plain(store(f3, base, reg(rng), off))
                }
                14 => Item::Branch(rng.pick(&[0, 1, 4, 5, 6, 7]), reg(rng), reg(rng)),
                15 => Item::Jal(int_dest(rng)),
                16 if rng.chance(2) => Item::Direct(int_dest(rng), int_dest(rng)),
                16 => Item::Indirect(int_dest(rng), int_dest(rng), rng.chance(2)),
                17 => {
                    // AMOs and lr/sc on aligned data through a base register.
                    let (base, _) = rng.pick(&BASES);
                    let f5 = rng.pick(&[0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28]);
                    let rs2 = if f5 == 2 { 0 } else { reg(rng) };
                    Plain(amo(f5, rng.pick(&[2, 3]), int_dest(rng), base, rs2))
                }
                18 if rng.chance(2) => {
                    // Move a base register a little (accesses through it then drift by as
                    // much, at worst off the page), so cached base addresses go stale.
                    let (base, _) = rng.pick(&BASES);
                    Plain(addi(base, base, (rng.below(9) as i32 - 4) * 8))
                }
                _ => Plain(fp(rng)),
            }
        }

        /// A random F/D instruction, including ones the JIT leaves to the interpreter.
        fn fp(rng: &mut Rng) -> u32 {
            let f = |rng: &mut Rng| rng.below(32) as u32;
            let fmt = rng.below(2) as u32;
            let rm = rng.pick(&[0, 1, 2, 3, 4, 7, 7, 7]);
            match rng.below(9) {
                0 => {
                    let (base, off) = mem_operand(rng, 4 << fmt);
                    fload(2 + fmt, f(rng), base, off)
                }
                1 => {
                    let (base, off) = mem_operand(rng, 4 << fmt);
                    fstore(2 + fmt, base, f(rng), off)
                }
                2 => op_fp(rng.below(4) as u32, fmt, rm, f(rng), f(rng), f(rng)),
                3 => op_fp(0b01011, fmt, rm, f(rng), f(rng), 0),
                // fsgnj/fsgnjn/fsgnjx, fmin/fmax
                4 => op_fp(rng.pick(&[4, 4, 5]), fmt, rng.below(3) as u32, f(rng), f(rng), f(rng)),
                5 => op_fp(0b10100, fmt, rng.below(3) as u32, int_dest(rng), f(rng), f(rng)),
                6 => op_fp(0b11000, fmt, rm, int_dest(rng), f(rng), rng.below(4) as u32),
                7 => op_fp(0b11010, fmt, rm, f(rng), reg(rng), rng.below(4) as u32),
                _ => match rng.below(3) {
                    0 => op_fp(0b11100, fmt, 0, int_dest(rng), f(rng), 0), // fmv.x.*
                    1 => op_fp(0b11110, fmt, 0, f(rng), reg(rng), 0),      // fmv.*.x
                    _ => op_fp(0b01000, fmt, rm, f(rng), f(rng), 1 - fmt), // fcvt.s.d / fcvt.d.s
                },
            }
        }

        /// A loop over a body of about `n` random words, run `COUNTER` times, then `ecall`.
        /// Jumps only go forward, so every program terminates.
        fn program(rng: &mut Rng, n: usize) -> Vec<u32> {
            let mut items = Vec::new();
            let mut len = 0;
            while len < n {
                let item = instruction(rng);
                len += item.len();
                items.push(item);
            }
            // Word index of each item; the loop tail (at `len`) is a target too.
            let starts: Vec<usize> = items
                .iter()
                .scan(0, |at, item| {
                    let start = *at;
                    *at += item.len();
                    Some(start)
                })
                .chain([len])
                .collect();
            let mut words = Vec::new();
            for (k, item) in items.iter().enumerate() {
                // A byte offset from word `from` to a random item after this one.
                let mut target = |from: usize| {
                    let to = starts[k + 1 + rng.below((items.len() - k) as u64) as usize];
                    ((to - from) * 4) as i32
                };
                let at = starts[k];
                match *item {
                    Item::Plain(w) => words.push(w),
                    Item::Branch(f3, rs1, rs2) => words.push(b(f3, rs1, rs2, target(at))),
                    Item::Jal(rd) => words.push(jal(rd, target(at))),
                    Item::Direct(t, rd) => {
                        let off = target(at);
                        words.extend([auipc(t, 0), jalr(rd, t, off)]);
                    }
                    Item::Indirect(t, rd, odd) => {
                        let off = target(at) + odd as i32;
                        words.extend([auipc(t, 0), addi(t, t, off), jalr(rd, t, 0)]);
                    }
                }
            }
            let back = -(words.len() as i32 + 1) * 4;
            words.push(addi(COUNTER, COUNTER, -1));
            words.push(b(1, COUNTER, 0, back)); // bne s1, zero, body
            words.push(ECALL);
            words
        }

        /// Bytes of data: a whole host page, so that accesses leaving it fault in generated
        /// code too (a guest page sharing a host page with a mapped one does not fault).
        const DATA_SIZE: u64 = 0x4000;

        /// How [`run`] executes a program.
        #[derive(Clone, Copy, PartialEq)]
        enum Mode {
            Interpreter,
            Jit,
            /// On translations saved by an earlier JIT run and restored into a fresh JIT.
            Restored,
        }

        /// Runs `prog` with guest memory in the identity layout if `identity`, or returns
        /// `None` if that layout is not available (macOS, or another test has it).
        fn run(
            prog: &[u32],
            init: &CpuState,
            data: &[u8],
            mode: Mode,
            identity: bool,
        ) -> Option<(Trap, CpuState, Vec<u8>)> {
            let memory = || {
                let mem = if identity {
                    GuestMemory::new_identity()?
                } else {
                    GuestMemory::new().ok()?
                };
                load_program(&mem, prog);
                mem.map(DATA + 0x1000, DATA_SIZE - 0x1000, Prot::RW).unwrap();
                mem.write_privileged(DATA, data).unwrap();
                Some(mem)
            };
            let mut mem = memory()?;
            let mut cpu = init.clone();
            let trap = if mode != Mode::Interpreter {
                let mut jit = Jit::new().unwrap();
                let trap = jit.run(&mut cpu, &mem);
                // (Nothing to save if the program ended before a block was translated.)
                if let (Mode::Restored, Some(saved)) = (mode, jit.save(1)) {
                    let mut jit = Jit::new().unwrap();
                    drop(mem); // (it holds the identity layout's range)
                    mem = memory()?;
                    cpu = init.clone();
                    jit.restore(&saved, 1, &mem).unwrap();
                    jit.run(&mut cpu, &mem)
                } else {
                    trap
                }
            } else {
                let mut steps = 0;
                loop {
                    if let Err(t) = interpreter::step(&mut cpu, &mem) {
                        break t;
                    }
                    steps += 1;
                    assert!(steps < 1 << 20, "generated program does not terminate");
                }
            };
            let mut out = vec![0; data.len()];
            mem.read_privileged(DATA, &mut out).unwrap();
            Some((trap, cpu, out))
        }

        #[test]
        fn random_programs_match_the_interpreter() {
            // FUZZ_CASES=n runs more (or fewer) cases, FUZZ_SEED=x a different sequence.
            let env = |name, default| {
                std::env::var(name)
                    .ok()
                    .and_then(|v: String| {
                        let v = v.trim_start_matches("0x");
                        u64::from_str_radix(v, if v.len() > 10 { 16 } else { 10 }).ok()
                    })
                    .unwrap_or(default)
            };
            let mut rng = Rng(env("FUZZ_SEED", 0x9e37_79b9_7f4a_7c15));
            let mut identity_runs = 0;
            for case in 0..env("FUZZ_CASES", 1000) {
                // Each case: a seed that reproduces it alone (set `rng.0` to it).
                let seed = rng.0;
                let n = 16 + rng.below(160) as usize;
                let prog = program(&mut rng, n);
                let mut init = CpuState::new(CODE);
                for r in 1..32 {
                    init.x[r] = match rng.below(4) {
                        0 => rng.below(16),
                        1 => rng.pick(&[
                            0,
                            1,
                            u64::MAX,
                            i64::MIN as u64,
                            i64::MAX as u64,
                            0xffff_ffff,
                            0x8000_0000,
                        ]),
                        _ => rng.next(),
                    };
                    init.f[r] = if rng.chance(2) {
                        rng.next()
                    } else {
                        (rng.next() as f64 % 1e6).to_bits()
                    };
                }
                for (base, at) in BASES {
                    init.x[base as usize] = at;
                }
                init.x[COUNTER as usize] = 1 + rng.below(3);
                init.fcsr = (rng.below(5) as u32) << 5;
                let data: Vec<u8> = (0..DATA_SIZE).map(|_| rng.next() as u8).collect();
                let want = run(&prog, &init, &data, Mode::Interpreter, false).unwrap();
                for (mode, identity) in [
                    (Mode::Jit, false),
                    (Mode::Restored, false),
                    (Mode::Jit, true),
                    (Mode::Restored, true),
                ] {
                    let Some(got) = run(&prog, &init, &data, mode, identity) else { continue };
                    identity_runs += u32::from(identity);
                    let which = match (mode, identity) {
                        (Mode::Jit, false) => "jit",
                        (Mode::Restored, false) => "restored",
                        (Mode::Jit, true) => "jit (identity layout)",
                        _ => "restored (identity layout)",
                    };
                    assert_eq!(got.0, want.0, "case {case} (seed {seed:#x}): {which} trap");
                    assert_eq!(got.1, want.1, "case {case} (seed {seed:#x}): {which} state");
                    assert!(got.2 == want.2, "case {case} (seed {seed:#x}): {which} memory");
                }
            }
            // Other tests may hold the identity layout's range for a while, but not always.
            if cfg!(target_os = "linux") {
                assert!(identity_runs > 0, "the identity layout was never available");
            }
        }
    }
}
