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
use crate::memory::{GuestMemory, MemFault, Memory, Tlb, TlbMemory, TlbReader};
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
    /// Guest pc -> the vector state its translation was refused under: RVV-heavy code
    /// re-enters such pcs every iteration (a block whose first instruction never
    /// translates), and re-decoding them each time dominated the dispatcher. Cleared
    /// with the translations.
    refused: HashMap<u64, (u64, u64), BuildHasherDefault<PcHasher>>,
    /// Instructions the interpreter fallback runs per batch: starts small so refusals
    /// in hot code hand back to the JIT quickly, grows while the dispatcher keeps
    /// landing inside untranslatable stretches, and resets on a translated hit.
    batch: u64,
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
    /// Lookups that missed the cache and had to decode, and blocks those decodes then
    /// refused to translate (their first instruction ends on the interpreter): how much
    /// time goes to re-deciding a pc that never becomes a block of its own.
    pub misses: u64,
    pub refused: u64,
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
            "jit: {} entries, {} translations, {} restored, {} resets, misses {}/{}, exits \
             next/ecall/interp/fence.i/fault/interrupt {:?} ({} next to known blocks), {total} \
             interpreted [{}]",
            self.entries,
            self.translations,
            self.restored,
            self.resets,
            self.misses,
            self.refused,
            self.exits,
            self.next_known,
            top.join(", ")
        )
    }
}

/// The variant name of an instruction, for statistics.
fn kind(inst: &crate::decoder::Inst) -> &'static str {
    use crate::decoder::Inst;
    // Vector accesses carry their kind (unit-stride, fault-only-first, ...): which
    // shapes fall back is exactly what decides what to translate next.
    let name = match inst {
        Inst::VLoad { kind, mask, .. } => {
            format!("VLoad({kind:?}{})", if *mask { ",masked" } else { "" })
        }
        Inst::VStore { kind, mask, .. } => {
            format!("VStore({kind:?}{})", if *mask { ",masked" } else { "" })
        }
        _ => format!("{inst:?}"),
    };
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
            fault_site: std::cell::Cell::new(None),
        };
        let mut jit = Jit {
            code,
            enter,
            stub_word: 0,
            blocks: HashMap::default(),
            refused: HashMap::default(),
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
            batch: 32,
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
        self.refused.clear();
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
    fn lookup(
        &mut self,
        pc: u64,
        mem: &GuestMemory,
        vstate: (u64, u64),
    ) -> Result<Option<usize>, Trap> {
        if let Some(&at) = self.blocks.get(&pc) {
            return Ok(Some(at));
        }
        if self.refused.get(&pc) == Some(&vstate) {
            return Ok(None); // decoded and refused already, for this very state
        }
        self.stats.misses += 1;
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
            let Some(t) =
                translate::translate(&block, pc, word, stub, cache, slot, identity, vstate)
            else {
                self.stats.refused += 1;
                self.refused.insert(pc, vstate);
                if std::env::var_os("RVRUN_DIAG").is_some() && self.stats.refused < 40 {
                    eprintln!(
                        "[diag] refused pc={pc:#x} vtype={:#x} vl={} first={:?}",
                        vstate.0,
                        vstate.1,
                        block.first().map(|(i, _)| i)
                    );
                }
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

    /// Skip unmapping the code buffer at drop; only for a process that is about to exit,
    /// where the kernel reclaims the address space.
    pub fn leak_at_exit(&mut self) {
        self.code.leak_at_exit();
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
        let mut engine = crate::engine::Engine::new();
        loop {
            if let Some(trap) = self.run_interruptible(cpu, &mut engine, mem) {
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

    /// Run a short stretch on the fast interpreter. Untranslatable code comes in runs
    /// (the tails of vectorized loops, masked shapes): interpreting one instruction per
    /// dispatcher round trip was the cost, so batch a few. The stretch is small enough
    /// that the interrupt flag is checked promptly between batches.
    fn fallback(
        &self,
        cpu: &mut CpuState,
        engine: &mut crate::engine::Engine,
        mem: &GuestMemory,
        batch: u64,
    ) -> Option<Trap> {
        engine.run(cpu, mem, batch).0
    }

    /// Like [`Jit::run`], but return `None` when generated code sees the interrupt flag set
    /// (the caller delivers signals and clears it), with `cpu` at the next instruction.
    pub fn run_interruptible(
        &mut self,
        cpu: &mut CpuState,
        engine: &mut crate::engine::Engine,
        mem: &GuestMemory,
    ) -> Option<Trap> {
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
            let at = match self.lookup(cpu.pc, mem, (cpu.vtype, cpu.vl)) {
                Ok(Some(at)) => {
                    // Generated code again: drop back to short interpreter stretches,
                    // so a refusal inside hot code hands back to the JIT quickly.
                    self.batch = 32;
                    // Let generated indirect jumps find this block without us next time.
                    self.jump_cache()[translate::jump_cache_index(cpu.pc)] =
                        [cpu.pc, self.code.ptr(at) as u64];
                    at
                }
                Ok(None) => {
                    // A stretch of untranslatable code: stay in the interpreter longer,
                    // growing while the dispatcher keeps landing back inside it (the
                    // memcpy loops of vectorized code); any translated hit resets.
                    let batch = self.batch;
                    self.batch = (batch * 4).min(4096);
                    if let Some(trap) = self.fallback(cpu, engine, mem, batch) {
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
                        let k = kind(&block[0].0);
                        *self.stats.interpreted.entry(k).or_default() += 1;
                        // One line per fallback: which instruction keeps leaving generated code.
                        if std::env::var_os("RVRUN_DIAG_PC").is_some() {
                            eprintln!("[interp] pc={:#x} kind={k}", cpu.pc);
                        }
                    }
                    let batch = self.batch;
                    self.batch = (batch * 4).min(4096);
                    if let Some(trap) = self.fallback(cpu, engine, mem, batch) {
                        return Some(trap);
                    }
                }
                exit::FENCE_I => self.reset(),
                exit::INTERRUPT => return None,
                exit::FAULT_LOAD | exit::FAULT_STORE => {
                    let host = self.faults.fault_addr.get();
                    // A `vle*ff.v` load does not trap: re-run it element by element and
                    // keep the prefix that worked, shortening `vl` to it (the faulting
                    // element and everything past it are left alone).
                    if code == exit::FAULT_LOAD
                        && let Some(site) = self.faults.fault_site.get()
                        && let Some((base, eew, vd)) = site.ff
                    {
                        let eew = usize::from(eew);
                        let addr = cpu.x(base);
                        let vl = cpu.vl;
                        let mut loaded = 0;
                        while loaded < vl {
                            let mut b = [0u8; 8];
                            let at = addr.wrapping_add(loaded.wrapping_mul(eew as u64));
                            if mem.read(at, &mut b[..eew]).is_err() {
                                break;
                            }
                            crate::vector::set_elem(
                                cpu,
                                vd,
                                loaded,
                                eew as u64,
                                u64::from_le_bytes(b),
                            );
                            loaded += 1;
                        }
                        cpu.vl = loaded;
                        cpu.instret += 1; // the fault-tolerant load retires
                        cpu.pc = site.guest_pc + 4;
                        continue;
                    }
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
    fn vector_float_ops_match_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle32 = |vd: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse32 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vv = |f6: u32, vd: u32, vs2: u32, vs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (0b001 << 12) | (vd << 7) | 0x57
        };
        let vf = |f6: u32, vd: u32, vs2: u32, rs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (rs1 << 15) | (0b101 << 12) | (vd << 7) | 0x57
        };
        const VT: u32 = 0xd0; // e32, m1, ta, ma
        let prog = [
            vsetivli(0, 4, VT),
            vle32(1, 11),                      // v1 = a
            vle32(2, 12),                      // v2 = b
            op_fp(0b11110, 0b00, 0, 0, 10, 0), // fmv.w.x f0, a0 (2.0): boxed in this block
            vv(0b000000, 3, 1, 2),             // vfadd.vv v3, v1, v2
            vse32(3, 13),
            vv(0b000010, 4, 1, 2), // vfsub.vv v4, v1, v2
            vse32(4, 14),
            vv(0b100000, 5, 1, 2), // vfdiv.vv v5, v1, v2
            vse32(5, 15),
            vf(0b100100, 6, 1, 0), // vfmul.vf v6, v1, f0
            vse32(6, 16),
            vv(0b101100, 7, 1, 2), // vfmacc.vv v7, v1, v2 (v7 starts 0)
            vse32(7, 17),
            vf(0b101100, 8, 1, 0), // vfmacc.vf v8, f0, v1
            vse32(8, 18),
            vv(0b101010, 9, 2, 1), // vfmsub.vv v9, v1, v2
            vse32(9, 19),
            // vfmv.v.f v10, f0 (vs2 = 0 selects the broadcast).
            (0x17 << 26) | (1 << 25) | (0b101 << 12) | (10 << 7) | 0x57,
            vse32(10, 20),
            vv(0b101101, 11, 2, 1), // vfnmacc.vv v11, v1, v2
            vse32(11, 21),
            (0x10 << 26) | (1 << 25) | (1 << 20) | (0b001 << 12) | (5 << 7) | 0x57, // vfmv.f.s f5, v1
            fstore(0b011, 22, 5, 0),                                                // fsd f5, 0(a6)
        ];
        let setup_data = |mem: &GuestMemory| {
            let a: [f32; 4] = [1.5, -2.5, 0.5, 8.0];
            let b: [f32; 4] = [4.0, 2.0, -0.5, 2.0];
            for (i, v) in a.iter().chain(b.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &v.to_bits().to_le_bytes()).unwrap();
            }
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [
                (11u8, DATA),
                (12, DATA + 16),
                (13, DATA + 64),
                (14, DATA + 80),
                (15, DATA + 96),
                (16, DATA + 112),
                (17, DATA + 128),
                (18, DATA + 144),
                (19, DATA + 160),
                (20, DATA + 176),
                (21, DATA + 192),
                (22, DATA + 208),
                (10, 2.0f32.to_bits() as u64),
            ] {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        setup_data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        setup_data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "state");
        for name in ["VFArith", "VFMa", "VFmvFS"] {
            assert_eq!(jit.stats.interpreted.get(name), None, "{name} stayed in the JIT");
        }
        let get = |off: u64| {
            let mut b = [0u8; 4];
            mem_j.read_privileged(DATA + off, &mut b).unwrap();
            f32::from_bits(u32::from_le_bytes(b))
        };
        assert_eq!(get(64), 5.5, "1.5 + 4.0");
        assert_eq!(get(80), -2.5, "1.5 - 4.0");
        assert_eq!(get(96), 0.375, "1.5 / 4.0");
        assert_eq!(get(112), 3.0, "1.5 × 2.0");
        assert_eq!(get(128), 6.0, "1.5 × 4.0");
        assert_eq!(get(144), 3.0, "2.0 × 1.5");
        assert_eq!(get(160), -4.0, "vfmsub: -vs2");
        assert_eq!(get(176), 2.0, "vfmv.v.f");
        assert_eq!(get(192), -6.0, "vfnmacc: -(vs2 × vs1)");
        let mut b = [0u8; 8];
        mem_j.read_privileged(DATA + 208, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b), 0xffff_ffff_3fc0_0000, "vfmv.f.s boxes");
    }

    #[test]
    fn vector_half_conversions_match_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let convert = |selector: u32, vd: u32, vs2: u32| {
            (0b010010 << 26) | (1 << 25) | (vs2 << 20) | (selector << 15) | (vd << 7) | 0x57
        };
        let prog = [
            vsetivli(0, 8, 0xc8), // e16, m1, ta, ma
            convert(0x0c, 4, 2),  // vfwcvt.f.f.v v4, v2
            convert(0x14, 6, 4),  // vfncvt.f.f.w v6, v4
            ECALL,
        ];
        let init = |cpu: &mut CpuState| {
            for (i, bits) in [0x3c00u16, 0xc000, 0x7c00, 0x7e00].into_iter().enumerate() {
                cpu.v[2][2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
            }
        };
        let mem_i = setup(&prog);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(jit.stats.interpreted.get("VFHalfConvert"), None);
    }

    #[test]
    fn vector_m2_groups_match_the_interpreter() {
        // The axpy shape clang emits for `a += s * b`: e32/m2 with whole-register
        // moves and a vfmacc.vf over the two-register group.
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vl2re32 = |vd: u32, rs1: u32| {
            (1 << 29) | (1 << 25) | (0b01000 << 20) | (rs1 << 15) | (vd << 7) | 0x07
        };
        let vs2r32 = |vs3: u32, rs1: u32| {
            (1 << 29) | (1 << 25) | (0b01000 << 20) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vfmacc_vf = |vd: u32, vs2: u32, f: u32| {
            (0x2c << 26) | (1 << 25) | (vs2 << 20) | (f << 15) | (0b101 << 12) | (vd << 7) | 0x57
        };
        let vle32 = |vd: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse32 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let prog = [
            vsetivli(0, 8, 0xd1),              // e32, m2, ta, ma (VLMAX = 8)
            op_fp(0b11110, 0b00, 0, 0, 10, 0), // fmv.w.x f0, a0 (2.0)
            vl2re32(8, 11),                    // v8-v9 = b
            vl2re32(10, 12),                   // v10-v11 = a
            vfmacc_vf(10, 8, 0),               // v10-v11 += f0 × v8-v9
            vs2r32(10, 12),                    // store back over a
            vle32(2, 11),                      // both unit-stride forms too
            vse32(2, 13),
        ];
        let setup_data = |mem: &GuestMemory| {
            let b: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let a: [f32; 8] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
            for (i, v) in b.iter().chain(a.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &v.to_bits().to_le_bytes()).unwrap();
            }
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in
                [(11u8, DATA), (12, DATA + 32), (13, DATA + 128), (10, 2.0f32.to_bits() as u64)]
            {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        setup_data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        setup_data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(jit.stats.interpreted.get("VFMa"), None, "m2 fma stayed in the JIT");
        let mut out = [0u8; 16];
        mem_j.read_privileged(DATA + 32, &mut out).unwrap();
        assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 2.5);
        assert_eq!(f32::from_le_bytes(out[12..16].try_into().unwrap()), 10.0);
        let mut unit = [0u8; 4];
        mem_j.read_privileged(DATA + 128, &mut unit).unwrap();
        assert_eq!(f32::from_le_bytes(unit), 1.0, "unit-stride group load");
    }

    #[test]
    fn vector_m2_unit_groups_stay_resident() {
        // The GPT inner loop loads the source and accumulator with unit-stride m2
        // loads, then runs vfmacc.vf before storing the accumulator. Both groups should
        // stay in scratch registers through the macc.
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle32 = |vd: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse32 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vfmacc_vf = |vd: u32, vs2: u32, f: u32| {
            (0x2c << 26) | (1 << 25) | (vs2 << 20) | (f << 15) | (0b101 << 12) | (vd << 7) | 0x57
        };
        let prog = [
            vsetivli(0, 8, 0xd1),              // e32, m2
            op_fp(0b11110, 0b00, 0, 0, 10, 0), // fmv.w.x f0, a0 (2.0)
            vle32(8, 11),                      // v8-v9 = b
            vle32(10, 12),                     // v10-v11 = a
            vfmacc_vf(10, 8, 0),               // a += 2 × b
            vse32(10, 13),
            ECALL,
        ];
        let data = |mem: &GuestMemory| {
            let b: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let a: [f32; 8] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
            for (i, v) in b.iter().chain(a.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &v.to_bits().to_le_bytes()).unwrap();
            }
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in
                [(11u8, DATA), (12, DATA + 32), (13, DATA + 64), (10, 2.0f32.to_bits() as u64)]
            {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(jit.stats.interpreted.get("VFMa"), None);
        assert_eq!(jit.stats.interpreted.get("VLoad"), None);
        assert_eq!(jit.stats.interpreted.get("VStore"), None);
        let mut out = [0u8; 32];
        mem_j.read_privileged(DATA + 64, &mut out).unwrap();
        for i in 0..8 {
            let want = 0.5 + 0.5 * i as f32 + 2.0 * (i as f32 + 1.0);
            assert_eq!(
                f32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap()),
                want,
                "lane {i}"
            );
        }
    }

    #[test]
    fn vector_m2_f64_macc_stays_resident() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vfmacc_vf = |vd: u32, vs2: u32, f: u32| {
            (0x2c << 26) | (1 << 25) | (vs2 << 20) | (f << 15) | (0b101 << 12) | (vd << 7) | 0x57
        };
        let prog = [
            vsetivli(0, 4, 0xd9),              // e64, m2
            op_fp(0b11110, 0b01, 0, 0, 10, 0), // fmv.d.x f0, a0 (2.0)
            vle64(8, 11),
            vle64(10, 12),
            vfmacc_vf(10, 8, 0),
            vse64(10, 13),
            ECALL,
        ];
        let data = |mem: &GuestMemory| {
            let b: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
            let a: [f64; 4] = [0.5, 1.5, 2.5, 3.5];
            for (i, v) in b.iter().chain(a.iter()).enumerate() {
                mem.write_privileged(DATA + 8 * i as u64, &v.to_bits().to_le_bytes()).unwrap();
            }
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 32), (13, DATA + 64), (10, 2.0f64.to_bits())] {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(jit.stats.interpreted.get("VFMa"), None);
        let mut out = [0u8; 32];
        mem_j.read_privileged(DATA + 64, &mut out).unwrap();
        for i in 0..4 {
            let want = 0.5 + i as f64 + 2.0 * (i as f64 + 1.0);
            assert_eq!(
                f64::from_le_bytes(out[8 * i..8 * i + 8].try_into().unwrap()),
                want,
                "lane {i}"
            );
        }
    }

    #[test]
    fn fboxed_entry_guard_matches_the_interpreter() {
        // A loop that reads a single a *previous* block loaded: the translation assumes
        // it boxed, checks that once at the loop head, and skips the per-use canonical-
        // NaN fixup. With a plain double in the register the guard must fall back to
        // the interpreter and still agree exactly.
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        for (load_f3, data) in [(0b010u32, 0xffff_ffff_3fc0_0000u64), (0b011, 1.5f64.to_bits())] {
            let prog = [
                fload(load_f3, 1, 11, 0),         // flw/fld ft1, 0(a1)
                vsetivli(0, 4, 0xd0),             // e32, m1
                addi(5, 0, 3),                    // t0 = 3
                op_fp(0b00000, 0b00, 7, 0, 1, 0), // fadd.s ft0, ft1, ft0
                addi(5, 5, -1),
                beq(5, 0, -8),           // bnez t0 -> the fadd
                fstore(0b010, 12, 0, 0), // fsw ft0, 0(a2)
            ];
            let fill = |mem: &GuestMemory| mem.write_privileged(DATA, &data.to_le_bytes()).unwrap();
            let init = |cpu: &mut CpuState| {
                for (r, v) in [(11u8, DATA), (12, DATA + 16)] {
                    cpu.set_x(r, v);
                }
            };
            let mem_i = setup(&prog);
            fill(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(&prog);
            fill(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want, "load f3={load_f3:#05b}");
            assert_eq!(cpu, oracle, "load f3={load_f3:#05b}");
            if load_f3 == 0b010 {
                assert_eq!(
                    jit.stats.interpreted.get("FArith"),
                    None,
                    "the boxed loop should stay in the JIT"
                );
            }
            let mut out = [0u8; 4];
            mem_j.read_privileged(DATA + 16, &mut out).unwrap();
            let mut want_out = [0u8; 4];
            mem_i.read_privileged(DATA + 16, &mut want_out).unwrap();
            assert_eq!(out, want_out, "stored value");
        }
    }

    #[test]
    fn f32_boxing_elision_matches_the_interpreter() {
        // The known-boxed fast path must agree with the interpreter's per-operand
        // canonical-NaN rule in both directions: a single this block boxed itself
        // skips the check, a double read as a single and a raw half still take it.
        let prog = [
            fload(0b010, 0, 11, 0),           // flw ft0, 0(a1): -Inf
            fload(0b010, 1, 11, 4),           // flw ft1, 4(a1): 3.0
            op_fp(0b00010, 0b00, 7, 1, 1, 0), // fmul.s ft1, ft1, ft0: a boxed chain
            fload(0b011, 2, 11, 8),           // fld ft2, 8(a1): a double, not a boxed single
            op_fp(0b00000, 0b00, 7, 3, 2, 1), // fadd.s ft3, ft2, ft1: reads canonical NaN
            op_fp(0b00000, 0b01, 7, 4, 2, 3), // fadd.d ft4, ft2, ft3: raw doubles
            op_fp(0b11010, 0b00, 7, 5, 0, 0), // fcvt.s.w ft5, zero: +0.0, boxed
            op_fp(0b00010, 0b00, 7, 6, 5, 1), // fmul.s ft6, ft5, ft1: 0 * -Inf raises NV
            ECALL,
        ];
        let setup_data = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &(-f32::INFINITY).to_bits().to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 4, &3.0f32.to_bits().to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &1.0f64.to_bits().to_le_bytes()).unwrap();
        };
        let mem_i = setup(&prog);
        setup_data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        setup_data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(
            cpu.f[0..7],
            [
                0xffff_ffff_ff80_0000, // -Inf, boxed
                0xffff_ffff_ff80_0000, // 3 * -Inf
                1.0f64.to_bits(),      // the double as stored
                0xffff_ffff_7fc0_0000, // canonical NaN: the double is not a boxed single
                f64::NAN.to_bits(),    // and as a double operand it stays canonical
                0xffff_ffff_0000_0000, // +0.0, boxed
                0xffff_ffff_7fc0_0000, // 0 * -Inf, canonical NaN, NV
            ]
        );
        assert_eq!(cpu, oracle, "whole state");
    }

    #[test]
    fn fclass_matches_the_interpreter() {
        fn fclass(fmt: u32, rd: u32, rs1: u32) -> u32 {
            op_fp(0b11100, fmt, 0b001, rd, rs1, 0)
        }
        fn run(prog: &[u32], fill: &dyn Fn(&GuestMemory)) -> (CpuState, GuestMemory) {
            let mem_i = setup(prog);
            fill(&mem_i);
            let mut oracle = CpuState::new(CODE);
            oracle.set_x(11, DATA);
            oracle.set_x(12, DATA + 0x200);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            fill(&mem_j);
            let mut cpu = CpuState::new(CODE);
            cpu.set_x(11, DATA);
            cpu.set_x(12, DATA + 0x200);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu, oracle, "fclass state");
            assert_eq!(jit.stats.interpreted.get("FClass"), None, "fclass stays in the JIT");
            (cpu, mem_j)
        }

        let s = [
            0xff80_0000u32, // -inf
            0xbfc0_0000,    // -normal
            0x8000_0001,    // -subnormal
            0x8000_0000,    // -0
            0x0000_0000,    // +0
            0x0000_0001,    // +subnormal
            0x3fc0_0000,    // +normal
            0x7f80_0000,    // +inf
            0x7f80_0001,    // sNaN
            0x7fc0_0000,    // qNaN
        ];
        let mut prog = Vec::new();
        for (i, _) in s.iter().enumerate() {
            prog.push(fload(0b010, 0, 11, 4 * i as i32));
            prog.push(fclass(0, 10, 0));
            prog.push(store(0b011, 12, 10, 8 * i as i32));
        }
        prog.push(ECALL);
        let (_, mem) = run(&prog, &|mem| {
            for (i, v) in s.iter().enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &v.to_le_bytes()).unwrap();
            }
        });
        let mut got = [0u8; 80];
        mem.read_privileged(DATA + 0x200, &mut got).unwrap();
        let want: Vec<u8> = (0..10).flat_map(|i| (1u64 << i).to_le_bytes()).collect();
        assert_eq!(got.as_slice(), want.as_slice(), "single classes");

        let d = [
            0xfff0_0000_0000_0000u64, // -inf
            0xbff8_0000_0000_0000,    // -1.5
            0x8000_0000_0000_0001,    // -subnormal
            0x8000_0000_0000_0000,    // -0
            0x0000_0000_0000_0000,    // +0
            0x0000_0000_0000_0001,    // +subnormal
            0x3ff8_0000_0000_0000,    // +1.5
            0x7ff0_0000_0000_0000,    // +inf
            0x7ff0_0000_0000_0001,    // sNaN
            0x7ff8_0000_0000_0000,    // qNaN
        ];
        let mut prog = Vec::new();
        for (i, _) in d.iter().enumerate() {
            prog.push(fload(0b011, 0, 11, 8 * i as i32));
            prog.push(fclass(1, 10, 0));
            prog.push(store(0b011, 12, 10, 8 * i as i32));
        }
        prog.push(ECALL);
        let (_, mem) = run(&prog, &|mem| {
            for (i, v) in d.iter().enumerate() {
                mem.write_privileged(DATA + 8 * i as u64, &v.to_le_bytes()).unwrap();
            }
        });
        mem.read_privileged(DATA + 0x200, &mut got).unwrap();
        assert_eq!(got.as_slice(), want.as_slice(), "double classes");

        // A raw, unboxed single reads as the canonical qNaN.
        let prog = [fclass(0, 10, 0), store(0b011, 12, 10, 0), ECALL];
        let mem_i = setup(&prog);
        let mut oracle = CpuState::new(CODE);
        oracle.f[0] = 0x1234_5678;
        oracle.set_x(12, DATA + 0x200);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        let mut cpu = CpuState::new(CODE);
        cpu.f[0] = 0x1234_5678;
        cpu.set_x(12, DATA + 0x200);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(cpu.x[10], 1 << 9, "unboxed single is a quiet NaN");
    }

    #[test]
    fn fminmax_matches_the_interpreter() {
        fn check(single: bool, a: u64, b: u64, max: bool, canonical_nan: bool, rd: u32) {
            let (fmt, f3_load, step) = if single { (0, 0b010, 4) } else { (1, 0b011, 8) };
            let f3 = match (max, canonical_nan) {
                (false, false) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (true, true) => 3,
            };
            let prog = [
                fload(f3_load, 0, 11, 0),
                fload(f3_load, 1, 11, step),
                op_fp(0b00101, fmt, f3, rd, 0, 1),
                fstore(f3_load, 12, rd, 0),
                ECALL,
            ];
            let fill = |mem: &GuestMemory| {
                if single {
                    mem.write_privileged(DATA, &(a as u32).to_le_bytes()).unwrap();
                    mem.write_privileged(DATA + 4, &(b as u32).to_le_bytes()).unwrap();
                } else {
                    mem.write_privileged(DATA, &a.to_le_bytes()).unwrap();
                    mem.write_privileged(DATA + 8, &b.to_le_bytes()).unwrap();
                }
            };
            let mem_i = setup(&prog);
            fill(&mem_i);
            let mut oracle = CpuState::new(CODE);
            oracle.set_x(11, DATA);
            oracle.set_x(12, DATA + 0x100);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(&prog);
            fill(&mem_j);
            let mut cpu = CpuState::new(CODE);
            cpu.set_x(11, DATA);
            cpu.set_x(12, DATA + 0x100);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu, oracle, "fminmax state");
            assert_eq!(jit.stats.interpreted.get("FMinMax"), None, "fminmax stays in the JIT");
            let step = step as usize;
            let mut got_bytes = [0u8; 8];
            let mut want_bytes = [0u8; 8];
            mem_j.read_privileged(DATA + 0x100, &mut got_bytes[..step]).unwrap();
            mem_i.read_privileged(DATA + 0x100, &mut want_bytes[..step]).unwrap();
            assert_eq!(&got_bytes[..step], &want_bytes[..step], "fminmax result");
        }

        let qnan = 0x7fc0_0000u64;
        let snan = 0x7f80_0001u64;
        for single in [true, false] {
            let (one, two, zero, negative_zero, qnan, snan) = if single {
                (1.5f32.to_bits() as u64, 2.5f32.to_bits() as u64, 0, 0x8000_0000, qnan, snan)
            } else {
                (
                    1.5f64.to_bits(),
                    2.5f64.to_bits(),
                    0,
                    0x8000_0000_0000_0000,
                    0x7ff8_0000_0000_0000,
                    0x7ff0_0000_0000_0001,
                )
            };
            for max in [false, true] {
                for canonical_nan in [false, true] {
                    check(single, one, two, max, canonical_nan, 2);
                    check(single, negative_zero, zero, max, canonical_nan, 2);
                    check(single, qnan, one, max, canonical_nan, 2);
                    check(single, snan, one, max, canonical_nan, 2);
                    // The destination aliases the signaling-NaN source.
                    check(single, snan, one, max, canonical_nan, 0);
                }
            }
        }
    }

    #[test]
    fn csr_frm_and_time_stay_in_the_jit() {
        let csr = |rd: u32, n: u32| n << 20 | 0b010 << 12 | rd << 7 | 0x73;
        let prog = [
            csr(10, 0x002),           // a0 = frm
            csr(11, 0xc01),           // a1 = time
            csr(12, 0xc01),           // a2 = time again
            store(0b011, 13, 10, 0),  // sd a0, 0(a3)
            store(0b011, 13, 11, 8),  // sd a1, 8(a3)
            store(0b011, 13, 12, 16), // sd a2, 16(a3)
            ECALL,
        ];
        let init = |cpu: &mut CpuState| {
            cpu.fcsr = 3 << 5; // frm = RZ, so the JIT can enter with a valid FPCR
            cpu.set_x(13, DATA + 0x200);
        };
        let mem_i = setup(&prog);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        loop {
            if interpreter::step(&mut oracle, &mem_i).is_err() {
                break;
            }
        }
        let mem_j = setup(&prog);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let _ = jit.run(&mut cpu, &mem_j);
        assert_eq!(cpu.x[10], 3, "frm");
        assert!(cpu.x[11] > 0, "time is a positive 10 MHz tick count");
        assert!(cpu.x[12] >= cpu.x[11], "time is monotonic");
        assert_eq!(jit.stats.interpreted.get("Csr"), None, "CSR reads stay in the JIT");
    }

    #[test]
    fn static_fcvt_from_int_rounding_stays_in_the_jit() {
        // fcvt.d.l with RTZ and fcvt.s.l with RDN: the ambient frm is RNE, so these
        // only work if the translator swaps FPCR around the conversion.
        let prog = [
            load(0b011, 11, 11, 0),         // ld a1, 0(a1)
            op_fp(0b11010, 1, 1, 0, 11, 2), // fcvt.d.l ft0, a1, rtz
            fstore(0b011, 12, 0, 0),        // fsd ft0, 0(a2)
            op_fp(0b11010, 0, 2, 1, 11, 2), // fcvt.s.l ft1, a1, rdn
            fstore(0b010, 12, 1, 8),        // fsw ft1, 8(a2)
            ECALL,
        ];
        let value = i64::MAX;
        let fill = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &value.to_le_bytes()).unwrap();
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        oracle.set_x(12, DATA + 0x100);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        cpu.set_x(12, DATA + 0x100);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "static conversion rounding");
        assert_eq!(jit.stats.interpreted.get("FCvtFromInt"), None, "conversion stays in the JIT");
    }

    #[test]
    fn zbc_and_zbkb_ops_stay_in_the_jit() {
        let prog = [
            load(0b011, 13, 11, 0),   // a3 = a
            load(0b011, 14, 11, 8),   // a4 = b
            op(0b100, 4, 15, 13, 14), // pack a5, a3, a4
            store(0b011, 12, 15, 0),
            op(0b111, 4, 15, 13, 14), // packh
            store(0b011, 12, 15, 8),
            op_32(0b100, 4, 15, 13, 14), // packw
            store(0b011, 12, 15, 16),
            op(0b001, 5, 15, 13, 14), // clmul
            store(0b011, 12, 15, 24),
            op(0b011, 5, 15, 13, 14), // clmulh
            store(0b011, 12, 15, 32),
            op(0b010, 5, 15, 13, 14), // clmulr
            store(0b011, 12, 15, 40),
            ECALL,
        ];
        let a = 0x1122_3344_5566_7788u64;
        let b = 0x99aa_bbcc_ddee_ff00u64;
        assert!(matches!(
            crate::decoder::decode(prog[12]),
            Ok(crate::decoder::Inst::Op { op: crate::decoder::AluOp::Clmulr, .. })
        ));
        let fill = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &a.to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &b.to_le_bytes()).unwrap();
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        oracle.set_x(12, DATA + 0x100);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        cpu.set_x(12, DATA + 0x100);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "Zb* state");
        assert_eq!(jit.stats.interpreted.get("Op"), None, "Zb* ops stay in the JIT");
        let mut got_bytes = [0u8; 48];
        let mut want_bytes = [0u8; 48];
        mem_j.read_privileged(DATA + 0x100, &mut got_bytes).unwrap();
        mem_i.read_privileged(DATA + 0x100, &mut want_bytes).unwrap();
        assert_eq!(got_bytes, want_bytes, "Zb* results");
    }

    #[test]
    fn zbkx_permutes_stay_in_the_jit() {
        let prog = [
            load(0b011, 13, 11, 0),
            load(0b011, 14, 11, 8),
            op(0b010, 20, 15, 13, 14), // xperm4
            store(0b011, 12, 15, 0),
            op(0b100, 20, 15, 13, 14), // xperm8
            store(0b011, 12, 15, 8),
            ECALL,
        ];
        let a = 0x0123_4567_89ab_cdefu64;
        let b = 0xfedc_ba98_7654_3210u64;
        assert!(matches!(
            crate::decoder::decode(prog[2]),
            Ok(crate::decoder::Inst::Op { op: crate::decoder::AluOp::Xperm4, .. })
        ));
        assert!(matches!(
            crate::decoder::decode(prog[4]),
            Ok(crate::decoder::Inst::Op { op: crate::decoder::AluOp::Xperm8, .. })
        ));
        let fill = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &a.to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &b.to_le_bytes()).unwrap();
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        oracle.set_x(12, DATA + 0x100);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        cpu.set_x(12, DATA + 0x100);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "Zbkx state");
        assert_eq!(jit.stats.interpreted.get("Op"), None, "Zbkx ops stay in the JIT");
        let mut got_bytes = [0u8; 16];
        let mut want_bytes = [0u8; 16];
        mem_j.read_privileged(DATA + 0x100, &mut got_bytes).unwrap();
        mem_i.read_privileged(DATA + 0x100, &mut want_bytes).unwrap();
        assert_eq!(got_bytes, want_bytes, "Zbkx results");
    }

    #[test]
    fn fsgnj_single_matches_the_interpreter() {
        // Sign injection reads both operands with the NaN-box rule, and a double read
        // as a single produces the canonical NaN whose sign is then injected; the
        // payload of a boxed NaN must survive untouched.
        let prog = [
            fload(0b010, 0, 11, 0),               // flw ft0, 0(a1): 1.5
            fload(0b010, 1, 11, 4),               // flw ft1, 4(a1): -2.25
            op_fp(0b00100, 0b00, 0b000, 2, 0, 1), // fsgnj.s ft2, ft0, ft1: -1.5
            op_fp(0b00100, 0b00, 0b001, 3, 0, 1), // fsgnjn.s ft3: 1.5
            op_fp(0b00100, 0b00, 0b010, 4, 0, 1), // fsgnjx.s ft4: -1.5
            fload(0b011, 5, 11, 8), // fld ft5, 8(a1): a raw double, not a boxed single
            op_fp(0b00100, 0b00, 0b000, 6, 5, 1), // fsgnj.s ft6, ft5, ft1: canonical NaN in
            op_fp(0b00100, 0b00, 0b010, 7, 6, 1), // fsgnjx.s ft7, ft6, ft1: signs xor to +
            ECALL,
        ];
        let setup_data = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &1.5f32.to_bits().to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 4, &(-2.25f32).to_bits().to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &1.0f64.to_bits().to_le_bytes()).unwrap();
        };
        let mem_i = setup(&prog);
        setup_data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        setup_data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "whole state");
        assert_eq!(cpu.f[2], 0xffff_ffff_bfc0_0000); // -1.5, boxed
        assert_eq!(cpu.f[3], 0xffff_ffff_3fc0_0000); // +1.5, boxed
        assert_eq!(cpu.f[4], 0xffff_ffff_bfc0_0000); // -1.5, boxed
        assert_eq!(cpu.f[6], 0xffff_ffff_ffc0_0000); // canonical NaN, sign from ft1
        assert_eq!(cpu.f[7], 0xffff_ffff_7fc0_0000); // and exclusive-or takes it back to +
        assert_eq!(cpu, oracle, "whole state");
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
    fn internal_back_edges_match_the_interpreter() {
        // A backward branch to an instruction of the same block re-enters the loop body
        // in place instead of a side exit through the block head. instret must stay
        // exact across passes (the loop leaves through its fall-through exit), the
        // prefix before the loop head runs once, and the entry guard's facts must
        // survive without re-running it.
        let (_, cpu) = same_as_interpreter(&[
            addi(9, 0, 7),   // 0x00: a1 = 7 (before the loop head: runs once)
            addi(10, 10, 1), // 0x04: a0 += 1 <- loop head
            addi(12, 12, 2), // 0x08: a2 += 2
            b(1, 10, 9, -8), // 0x0c: bne a0, a1 -> 0x04 (the internalized back edge)
            ECALL,           // 0x10: leaves through the fall-through exit
        ]);
        assert_eq!((cpu.x[10], cpu.x[12]), (7, 14));

        // The same shape with singles the block boxed itself: ft5 and ft6 are read in
        // every pass, but only written once before the loop head, so the carried boxed
        // facts must hold again on each re-entry.
        let (_, cpu) = same_as_interpreter(&[
            op_fp(0b11010, 0b00, 7, 5, 0, 0), // 0x00: fcvt.s.w ft5, zero: +0.0, boxed
            op_fp(0b11010, 0b00, 7, 6, 0, 0), // 0x04: fcvt.s.w ft6, zero: +0.0, boxed
            addi(9, 0, 4),                    // 0x08: a1 = 4
            op_fp(0b00000, 0b00, 7, 6, 6, 5), // 0x0c: fadd.s ft6, ft6, ft5 <- loop head
            addi(10, 10, 1),                  // 0x10: a0 += 1
            b(1, 10, 9, -8),                  // 0x14: bne a0, a1 -> 0x0c
            ECALL,                            // 0x18
        ]);
        let zero = 0xffff_ffff_0000_0000; // +0.0, NaN-boxed
        assert_eq!((cpu.x[10], cpu.f[5], cpu.f[6]), (4, zero, zero));

        // And with the branch as the block's last instruction (decode stops at the
        // 64-instruction limit): the edge replaces the taken exit entirely.
        let mut prog = vec![addi(9, 0, 4)]; // a1 = 4
        prog.extend(vec![addi(0, 0, 0); 61]); // pad the block up to the limit
        prog.push(addi(10, 10, 1)); // 0x100f8: a0 += 1 <- loop head
        prog.push(b(1, 10, 9, -4)); // 0x100fc: bne -> 0x100f8 (64th instruction)
        prog.push(ECALL); // 0x10100: a block of its own
        let (_, cpu) = same_as_interpreter(&prog);
        assert_eq!(cpu.x[10], 4);
    }

    #[test]
    fn unrolled_loops_match_the_interpreter() {
        // A mid-body side exit inside the unrolled loop: firing from the first and from
        // the second copy, with the instret corrections both copies' exits rely on.
        let prog = |target: i32| {
            vec![
                addi(9, 0, 10),      // 0x00: a1 = 10, counting passes
                addi(12, 0, target), // 0x04: a2 = the a0 the loop exits on
                addi(10, 10, 1),     // 0x08: a0 += 1 <- loop head
                beq(10, 12, 12),     // 0x0c: a0 == a2 -> 0x18 (the side exit)
                addi(9, 9, -1),      // 0x10: a1 -= 1
                b(1, 9, 0, -12),     // 0x14: bne a1, zero -> 0x08 (the back edge)
                ECALL,               // 0x18
            ]
        };
        let (_, cpu) = same_as_interpreter(&prog(3)); // exits from the first copy
        assert_eq!((cpu.x[10], cpu.x[9]), (3, 8));
        let (_, cpu) = same_as_interpreter(&prog(2)); // exits from the second copy
        assert_eq!((cpu.x[10], cpu.x[9]), (2, 9));
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

    /// A vector load, lane-wise add, and store at `vl == vlmax`; the JIT translates the
    /// unit-stride accesses as one 16-byte move each and the add per element width, and
    /// the block's entry guard keeps other vector configurations on the interpreter.
    #[test]
    fn vector_lane_ops_match_the_interpreter() {
        // Hand-assembled helpers for the pieces this test needs (as in the interpreter's
        // own vector test).
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vadd_vv =
            |vd: u32, vs2: u32, vs1: u32| (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57;
        let vand_vv = |vd: u32, vs2: u32, vs1: u32| {
            (1 << 25) | (0b001001 << 26) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };

        // The second case is a configuration the translation refuses (vl < vlmax), so the
        // whole block goes through the interpreter; both engines must still agree.
        let cases = [("e64 m1", 0x0d8, 2, true), ("e64 m1, vl = 1 falls back", 0x0d8, 1, false)];
        for (name, vtype, avl, check_values) in cases {
            let prog = [
                vsetivli(0, avl, vtype),
                vle64(1, 11),     // v1 = [a1]
                vle64(2, 12),     // v2 = [a2]
                vadd_vv(3, 1, 2), // v3 = v1 + v2
                vand_vv(4, 3, 2), // v4 = v3 & v2 (the byte-wise form)
                vse64(3, 13),     // [a3] = v3
                vse64(4, 14),     // [a4] = v4
            ];
            let setup_data = |mem: &GuestMemory| {
                mem.write_privileged(DATA, &0x0000_0000_0000_0001u64.to_le_bytes()).unwrap();
                mem.write_privileged(DATA + 8, &0x0000_0000_0000_0002u64.to_le_bytes()).unwrap();
                mem.write_privileged(DATA + 16, &0x0000_0000_0000_0010u64.to_le_bytes()).unwrap();
                mem.write_privileged(DATA + 24, &0x0000_0000_0000_0020u64.to_le_bytes()).unwrap();
            };
            let init = |cpu: &mut CpuState| {
                for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, DATA + 48)] {
                    cpu.set_x(r, v);
                }
            };
            let mem_i = setup(&prog);
            setup_data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(&prog);
            setup_data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
            assert_eq!(got, want, "{name}: trap");
            assert_eq!(cpu.x[13..15], oracle.x[13..15], "{name}: pointers");
            assert_eq!(cpu.v, oracle.v, "{name}: vector register file");
            assert_eq!(cpu, oracle, "{name}: state");
            for off in [32u64, 40, 48, 56] {
                let mut a = [0u8; 8];
                let mut b = [0u8; 8];
                mem_i.read_privileged(DATA + off, &mut a).unwrap();
                mem_j.read_privileged(DATA + off, &mut b).unwrap();
                assert_eq!(a, b, "{name}: stores differ at +{off}");
                if check_values {
                    // v3 = v1 + v2 = [0x11, 0x22] at a3, v4 = v3 & v2 = [0x10, 0x20] at a4.
                    let want = match off {
                        32 => 0x11,
                        40 => 0x22,
                        48 => 0x10,
                        _ => 0x20,
                    };
                    assert_eq!(u64::from_le_bytes(a), want, "{name}: value at +{off}");
                }
            }
        }
    }

    #[test]
    fn vector_shift_sub_mul_match_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |f6: u32, vd: u32, vs2: u32, vs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vx = |f6: u32, vd: u32, vs2: u32, rs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (rs1 << 15) | (0b100 << 12) | (vd << 7) | 0x57
        };
        let vi = |f6: u32, vd: u32, vs2: u32, imm: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (imm << 15) | (0b011 << 12) | (vd << 7) | 0x57
        };
        let mvv = |vd: u32, vs2: u32, vs1: u32| {
            (0b100101 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (vs1 << 15)
                | (0b010 << 12)
                | (vd << 7)
                | 0x57
        };
        let mvx = |vd: u32, vs2: u32, rs1: u32| {
            (0b100101 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b110 << 12)
                | (vd << 7)
                | 0x57
        };

        // e8: every lane active; the amounts cover the log2(SEW) masking (8..15 wrap
        // to 0..7) and the scalar 9 wraps to 1. v1 and v2 are the two inputs, the
        // seven results go to a3 upward in 16-byte slats.
        let data = |mem: &GuestMemory| {
            mem.write_privileged(
                DATA,
                &[
                    0xff, 0x80, 0x7f, 0x01, 0xfe, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
                    0x90, 0xa0, 0xb0,
                ],
            )
            .unwrap();
            mem.write_privileged(
                DATA + 16,
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            )
            .unwrap();
        };
        let prog = [
            vsetivli(0, 16, 0xc0), // e8, m1, ta, ma
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vi(0b101000, 3, 1, 3),  // vsrl.vi v3, v1, 3
            vx(0b101001, 4, 1, 14), // vsra.vx v4, v1, a4 (9 wraps to 1)
            vv(0b101000, 5, 1, 2),  // vsrl.vv v5, v1, v2
            vv(0b101001, 6, 1, 2),  // vsra.vv v6, v1, v2
            vv(0b000010, 7, 1, 2),  // vsub.vv v7, v1, v2
            mvv(8, 1, 2),           // vmul.vv v8, v1, v2
            mvx(9, 1, 15),          // vmul.vx v9, v1, a5 (= 3)
            vse(0b000, 3, 13),
            addi(13, 13, 16),
            vse(0b000, 4, 13),
            addi(13, 13, 16),
            vse(0b000, 5, 13),
            addi(13, 13, 16),
            vse(0b000, 6, 13),
            addi(13, 13, 16),
            vse(0b000, 7, 13),
            addi(13, 13, 16),
            vse(0b000, 8, 13),
            addi(13, 13, 16),
            vse(0b000, 9, 13),
            ECALL,
        ];
        let want_rows: [[u8; 16]; 7] = [
            [
                0x1f, 0x10, 0x0f, 0x00, 0x1f, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12,
                0x14, 0x16,
            ],
            [
                0xff, 0xc0, 0x3f, 0x00, 0xff, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0xc0, 0xc8,
                0xd0, 0xd8,
            ],
            [
                0xff, 0x40, 0x1f, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x40, 0x28, 0x18, 0x0e, 0x08, 0x04,
                0x02, 0x01,
            ],
            [
                0xff, 0xc0, 0x1f, 0x00, 0xff, 0x00, 0x00, 0x00, 0x40, 0x28, 0x18, 0x0e, 0xf8, 0xfc,
                0xfe, 0xff,
            ],
            [
                0xff, 0x7f, 0x7d, 0xfe, 0xfa, 0x0b, 0x1a, 0x29, 0x38, 0x47, 0x56, 0x65, 0x74, 0x83,
                0x92, 0xa1,
            ],
            [
                0x00, 0x80, 0xfe, 0x03, 0xf8, 0x50, 0xc0, 0x50, 0x00, 0xd0, 0xc0, 0xd0, 0x00, 0x50,
                0xc0, 0x50,
            ],
            [
                0xfd, 0x80, 0x7d, 0x03, 0xfa, 0x30, 0x60, 0x90, 0xc0, 0xf0, 0x20, 0x50, 0x80, 0xb0,
                0xe0, 0x10,
            ],
        ];
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 48), (14, 9), (15, 3)] {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want, "e8: trap");
        assert_eq!(cpu, oracle, "e8: state");
        assert_eq!(
            jit.stats.interpreted.get("VArith"),
            None,
            "e8: every arithmetic op should stay in the JIT"
        );
        for (row, want) in want_rows.iter().enumerate() {
            let mut b = [0u8; 16];
            mem_j.read_privileged(DATA + 48 + 16 * row as u64, &mut b).unwrap();
            assert_eq!(&b, want, "e8: row {row}");
        }

        // e64 exercises the 2d forms: one logical and one arithmetic shift.
        let prog64 = [
            vsetivli(0, 2, 0xd8), // e64, m1, ta, ma
            vle(0b111, 1, 11),
            vx(0b101000, 2, 1, 14), // vsrl.vx v2, v1, a4 (65 wraps to 1)
            vi(0b101001, 3, 1, 5),  // vsra.vi v3, v1, 5
            vse(0b111, 2, 13),
            addi(13, 13, 16),
            vse(0b111, 3, 13),
            ECALL,
        ];
        let data64 = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &u64::MAX.to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &(1u64 << 63).to_le_bytes()).unwrap();
        };
        let init64 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (13, DATA + 48), (14, 65)] {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog64);
        data64(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init64(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog64);
        data64(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init64(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want, "e64: trap");
        assert_eq!(cpu, oracle, "e64: state");
        assert_eq!(jit.stats.interpreted.get("VArith"), None, "e64: in the JIT");
        let mut b = [0u8; 16];
        mem_j.read_privileged(DATA + 48, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b[..8].try_into().unwrap()), 0x7fff_ffff_ffff_ffff);
        assert_eq!(u64::from_le_bytes(b[8..].try_into().unwrap()), 1u64 << 62);
        mem_j.read_privileged(DATA + 64, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b[..8].try_into().unwrap()), u64::MAX);
        assert_eq!(u64::from_le_bytes(b[8..].try_into().unwrap()), 0xfc00_0000_0000_0000);
    }

    #[test]
    fn vector_rol_matches_the_interpreter() {
        fn rol(x: u64, bits: u32, sh: u64) -> u64 {
            let sh = (sh % u64::from(bits)) as u32;
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            ((x << sh) | (x >> ((bits - sh) % bits))) & mask
        }
        fn check(
            prog: &[u32],
            data: &dyn Fn(&GuestMemory),
            init: &dyn Fn(&mut CpuState),
        ) -> GuestMemory {
            let mem_i = setup(prog);
            data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu.v, oracle.v, "vector register file");
            assert_eq!(cpu, oracle);
            assert_eq!(jit.stats.interpreted.get("VArith"), None, "vrol should stay in the JIT");
            mem_j
        }
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |vd: u32, vs2: u32, vs1: u32| {
            (0b010101 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vx = |vd: u32, vs2: u32, rs1: u32| {
            (0b010101 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b100 << 12)
                | (vd << 7)
                | 0x57
        };

        // e8/m1: rotate in place through `.vv`, then a separate `.vx` and another
        // in-place `.vx`. The byte amounts include 8..15, which must wrap modulo 8.
        let bytes: [u8; 16] = [
            0x81, 0x12, 0x40, 0x01, 0xff, 0x80, 0x7f, 0xaa, 0x55, 0x0f, 0xf0, 0x13, 0x37, 0x99,
            0xcc, 0x66,
        ];
        let byte_amounts: [u8; 16] = [1, 3, 7, 9, 15, 0, 2, 4, 6, 8, 10, 12, 14, 5, 11, 13];
        let prog8 = [
            vsetivli(0, 16, 0xc0),
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vv(1, 1, 2),
            vx(3, 1, 14),
            vx(1, 1, 15),
            vse(0b000, 3, 13),
            addi(13, 13, 16),
            vse(0b000, 1, 13),
            ECALL,
        ];
        let data8 = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &bytes).unwrap();
            mem.write_privileged(DATA + 16, &byte_amounts).unwrap();
        };
        let init8 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, 9), (15, 3)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&prog8, &data8, &init8);
        let rotated: Vec<u8> = bytes
            .iter()
            .zip(byte_amounts)
            .map(|(&x, amount)| rol(u64::from(x), 8, u64::from(amount)) as u8)
            .collect();
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        for (i, &x) in rotated.iter().enumerate() {
            assert_eq!(row[i], rol(u64::from(x), 8, 9) as u8, "e8 vx lane {i}");
        }
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        for (i, &x) in rotated.iter().enumerate() {
            assert_eq!(row[i], rol(u64::from(x), 8, 3) as u8, "e8 in-place lane {i}");
        }

        // e32/m1: the per-lane amount 36 wraps to 4, and the scalar 36 does too.
        let words: [u32; 4] = [0x8000_0001, 0x1234_5678, 0xdead_beef, 0xffff_ffff];
        let word_amounts: [u32; 4] = [1, 8, 31, 36];
        let prog32 = [
            vsetivli(0, 4, 0xd0),
            vle(0b110, 1, 11),
            vle(0b110, 2, 12),
            vv(1, 1, 2),
            vx(3, 1, 14),
            vse(0b110, 3, 13),
            addi(13, 13, 16),
            vse(0b110, 1, 13),
            ECALL,
        ];
        let data32 = |mem: &GuestMemory| {
            for (i, &x) in words.iter().chain(word_amounts.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let init32 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, 36)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&prog32, &data32, &init32);
        let rotated: Vec<u32> = words
            .iter()
            .zip(word_amounts)
            .map(|(&x, amount)| rol(u64::from(x), 32, u64::from(amount)) as u32)
            .collect();
        let mut row = [0u8; 16];
        for (offset, want) in [(32u64, 4u64), (48, 0)] {
            mem.read_privileged(DATA + offset, &mut row).unwrap();
            for (i, &x) in rotated.iter().enumerate() {
                let got = u32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
                assert_eq!(got, rol(u64::from(x), 32, want) as u32, "e32 lane {i}");
            }
        }

        // e32/m2: one in-place `.vv` over v10-v11 and one `.vx` over v14-v15.
        let words8: [u32; 8] = [
            0x8000_0001,
            0x1234_5678,
            0xdead_beef,
            0xffff_ffff,
            0x0102_0304,
            0x7fff_ffff,
            0x0000_0001,
            0x89ab_cdef,
        ];
        let amounts8: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let progm2 = [
            vsetivli(0, 8, 0xd1),
            vle(0b110, 10, 11),
            vle(0b110, 12, 12),
            vv(10, 10, 12),
            vx(14, 10, 14),
            vse(0b110, 10, 13),
            addi(13, 13, 32),
            vse(0b110, 14, 13),
            ECALL,
        ];
        let datam2 = |mem: &GuestMemory| {
            for (i, &x) in words8.iter().chain(amounts8.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let initm2 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 32), (13, DATA + 64), (14, 33)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&progm2, &datam2, &initm2);
        let rotated: Vec<u32> = words8
            .iter()
            .zip(amounts8)
            .map(|(&x, amount)| rol(u64::from(x), 32, u64::from(amount)) as u32)
            .collect();
        for (offset, want) in [(64u64, 0u64), (96, 1)] {
            let mut row = [0u8; 32];
            mem.read_privileged(DATA + offset, &mut row).unwrap();
            for (i, &x) in rotated.iter().enumerate() {
                let got = u32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
                assert_eq!(got, rol(u64::from(x), 32, want) as u32, "m2 lane {i}");
            }
        }
    }

    #[test]
    fn vector_sll_and_ror_match_the_interpreter() {
        fn ror(x: u64, bits: u32, sh: u64) -> u64 {
            let sh = (sh % u64::from(bits)) as u32;
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            ((x >> sh) | (x << ((bits - sh) % bits))) & mask
        }
        fn check(
            prog: &[u32],
            data: &dyn Fn(&GuestMemory),
            init: &dyn Fn(&mut CpuState),
        ) -> GuestMemory {
            let mem_i = setup(prog);
            data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu.v, oracle.v, "vector register file");
            assert_eq!(cpu, oracle);
            assert_eq!(
                jit.stats.interpreted.get("VArith"),
                None,
                "vsll/vror should stay in the JIT"
            );
            mem_j
        }
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |f6: u32, vd: u32, vs2: u32, vs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vx = |f6: u32, vd: u32, vs2: u32, rs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (rs1 << 15) | (0b100 << 12) | (vd << 7) | 0x57
        };
        let vi = |f6: u32, vd: u32, vs2: u32, imm: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (imm << 15) | (0b011 << 12) | (vd << 7) | 0x57
        };

        let bytes: [u8; 16] = [
            0x81, 0x12, 0x40, 0x01, 0xff, 0x80, 0x7f, 0xaa, 0x55, 0x0f, 0xf0, 0x13, 0x37, 0x99,
            0xcc, 0x66,
        ];
        let byte_amounts: [u8; 16] = [1, 3, 7, 9, 15, 0, 2, 4, 6, 8, 10, 12, 14, 5, 11, 13];
        let prog8 = [
            vsetivli(0, 16, 0xc0),
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vv(0b100101, 1, 1, 2),  // vsll.vv v1, v1, v2
            vx(0b010100, 3, 1, 14), // vror.vx v3, v1, a4
            vx(0b100101, 1, 1, 15), // vsll.vx v1, v1, a5
            vi(0b010100, 4, 1, 13), // vror.vi v4, v1, 13
            vi(0b100101, 5, 1, 2),  // vsll.vi v5, v1, 2
            vse(0b000, 3, 13),
            addi(13, 13, 16),
            vse(0b000, 1, 13),
            addi(13, 13, 16),
            vse(0b000, 4, 13),
            addi(13, 13, 16),
            vse(0b000, 5, 13),
            ECALL,
        ];
        let data8 = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &bytes).unwrap();
            mem.write_privileged(DATA + 16, &byte_amounts).unwrap();
        };
        let init8 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, 9), (15, 3)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&prog8, &data8, &init8);
        let shifted: Vec<u8> = bytes
            .iter()
            .zip(byte_amounts)
            .map(|(&x, amount)| x.wrapping_shl(u32::from(amount & 7)))
            .collect();
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        for (i, &x) in shifted.iter().enumerate() {
            assert_eq!(row[i], ror(u64::from(x), 8, 9) as u8, "e8 vror lane {i}");
        }
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        for (i, &x) in shifted.iter().enumerate() {
            assert_eq!(row[i], x.wrapping_shl(3), "e8 vsll lane {i}");
        }
        mem.read_privileged(DATA + 64, &mut row).unwrap();
        for (i, &x) in shifted.iter().enumerate() {
            let x = x.wrapping_shl(3);
            assert_eq!(row[i], ror(u64::from(x), 8, 13) as u8, "e8 vror.vi lane {i}");
        }
        mem.read_privileged(DATA + 80, &mut row).unwrap();
        for (i, &x) in shifted.iter().enumerate() {
            assert_eq!(row[i], x.wrapping_shl(3).wrapping_shl(2), "e8 vsll.vi lane {i}");
        }

        let words: [u32; 8] = [
            0x8000_0001,
            0x1234_5678,
            0xdead_beef,
            0xffff_ffff,
            0x0102_0304,
            0x7fff_ffff,
            0x0000_0001,
            0x89ab_cdef,
        ];
        let amounts: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let progm2 = [
            vsetivli(0, 8, 0xd1),
            vle(0b110, 10, 11),
            vle(0b110, 12, 12),
            vv(0b100101, 10, 10, 12), // vsll.vv m2
            vx(0b010100, 14, 10, 14), // vror.vx m2
            vse(0b110, 10, 13),
            addi(13, 13, 32),
            vse(0b110, 14, 13),
            ECALL,
        ];
        let datam2 = |mem: &GuestMemory| {
            for (i, &x) in words.iter().chain(amounts.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let initm2 = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 32), (13, DATA + 64), (14, 33)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&progm2, &datam2, &initm2);
        let shifted: Vec<u32> =
            words.iter().zip(amounts).map(|(&x, amount)| x.wrapping_shl(amount & 31)).collect();
        for (offset, want_ror) in [(64u64, false), (96, true)] {
            let mut row = [0u8; 32];
            mem.read_privileged(DATA + offset, &mut row).unwrap();
            for (i, &x) in shifted.iter().enumerate() {
                let got = u32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
                let want = if want_ror { ror(u64::from(x), 32, 1) as u32 } else { x };
                assert_eq!(got, want, "m2 lane {i}");
            }
        }
    }

    #[test]
    fn vector_andn_matches_the_interpreter() {
        fn check(
            prog: &[u32],
            data: &dyn Fn(&GuestMemory),
            init: &dyn Fn(&mut CpuState),
        ) -> GuestMemory {
            let mem_i = setup(prog);
            data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu, oracle);
            assert_eq!(jit.stats.interpreted.get("VArith"), None);
            mem_j
        }
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |vd: u32, vs2: u32, vs1: u32| {
            (0b000001 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vx = |vd: u32, vs2: u32, rs1: u32| {
            (0b000001 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b100 << 12)
                | (vd << 7)
                | 0x57
        };
        let bytes: [u8; 16] = [
            0x81, 0x12, 0x40, 0x01, 0xff, 0x80, 0x7f, 0xaa, 0x55, 0x0f, 0xf0, 0x13, 0x37, 0x99,
            0xcc, 0x66,
        ];
        let mask: [u8; 16] = [
            0x0f, 0xf0, 0xaa, 0x55, 0x00, 0xff, 0x80, 0x7f, 0x33, 0xcc, 0x0f, 0xf0, 0x5a, 0xa5,
            0x01, 0xfe,
        ];
        let prog = [
            vsetivli(0, 16, 0xc0),
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vv(3, 1, 2),
            vx(4, 1, 14),
            vse(0b000, 3, 13),
            addi(13, 13, 16),
            vse(0b000, 4, 13),
            ECALL,
        ];
        let data = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &bytes).unwrap();
            mem.write_privileged(DATA + 16, &mask).unwrap();
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, 0x0f)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&prog, &data, &init);
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        for i in 0..16 {
            assert_eq!(row[i], bytes[i] & !mask[i], "vandn.vv lane {i}");
        }
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        for i in 0..16 {
            assert_eq!(row[i], bytes[i] & !0x0f, "vandn.vx lane {i}");
        }
    }

    #[test]
    fn vector_gather_e8_matches_the_interpreter() {
        fn check(
            prog: &[u32],
            data: &dyn Fn(&GuestMemory),
            init: &dyn Fn(&mut CpuState),
        ) -> GuestMemory {
            let mem_i = setup(prog);
            data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu, oracle);
            assert_eq!(jit.stats.interpreted.get("VArith"), None);
            mem_j
        }
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |vd: u32, vs2: u32, vs1: u32| {
            (0b001100 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vi = |vd: u32, vs2: u32, imm: u32| {
            (0b001100 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (imm << 15)
                | (0b011 << 12)
                | (vd << 7)
                | 0x57
        };
        let table: [u8; 16] = [
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ];
        let indices: [u8; 16] = [15, 0, 16, 3, 3, 0, 255, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let prog = [
            vsetivli(0, 16, 0xc0),
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vv(3, 1, 2),
            vi(4, 1, 3),
            vse(0b000, 3, 13),
            addi(13, 13, 16),
            vse(0b000, 4, 13),
            ECALL,
        ];
        let data = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &table).unwrap();
            mem.write_privileged(DATA + 16, &indices).unwrap();
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32)] {
                cpu.set_x(r, v);
            }
        };
        let mem = check(&prog, &data, &init);
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        for i in 0..16 {
            let want =
                if usize::from(indices[i]) < 16 { table[usize::from(indices[i])] } else { 0 };
            assert_eq!(row[i], want, "vrgather.vv lane {i}");
        }
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        assert_eq!(row, [table[3]; 16], "vrgather.vi");
    }

    #[test]
    fn vector_gather_wide_matches_the_interpreter() {
        fn check(
            prog: &[u32],
            data: &dyn Fn(&GuestMemory),
            init: &dyn Fn(&mut CpuState),
        ) -> GuestMemory {
            let mem_i = setup(prog);
            data(&mem_i);
            let mut oracle = CpuState::new(CODE);
            init(&mut oracle);
            let want = loop {
                if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                    break t;
                }
            };
            let mem_j = setup(prog);
            data(&mem_j);
            let mut cpu = CpuState::new(CODE);
            init(&mut cpu);
            let mut jit = Jit::new().unwrap();
            let got = jit.run(&mut cpu, &mem_j);
            assert_eq!(got, want);
            assert_eq!(cpu, oracle);
            assert_eq!(jit.stats.interpreted.get("VArith"), None);
            mem_j
        }
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |vd: u32, vs2: u32, vs1: u32| {
            (0b001100 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vi = |vd: u32, vs2: u32, imm: u32| {
            (0b001100 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (imm << 15)
                | (0b011 << 12)
                | (vd << 7)
                | 0x57
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32)] {
                cpu.set_x(r, v);
            }
        };

        let table16: [u16; 8] = [0x1000, 0x1001, 0x1002, 0x1003, 0x1004, 0x1005, 0x1006, 0x1007];
        let index16: [u16; 8] = [7, 0, 8, 3, 3, 0, 99, 1];
        let data16 = |mem: &GuestMemory| {
            for (i, &x) in table16.iter().chain(index16.iter()).enumerate() {
                mem.write_privileged(DATA + 2 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let prog16 = [
            vsetivli(0, 8, 0xc8),
            vle(0b101, 1, 11),
            vle(0b101, 2, 12),
            vv(3, 1, 2),
            vi(4, 1, 3),
            vse(0b101, 3, 13),
            addi(13, 13, 16),
            vse(0b101, 4, 13),
            ECALL,
        ];
        let mem = check(&prog16, &data16, &init);
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        let got: Vec<u16> = row.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        assert_eq!(got[..4], [table16[7], table16[0], 0, table16[3]]);
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        assert!(row.as_chunks::<2>().0.iter().all(|b| u16::from_le_bytes(*b) == table16[3]));

        let table32: [u32; 4] = [0x1000_0001, 0x2000_0002, 0x3000_0003, 0x4000_0004];
        let index32: [u32; 4] = [3, 0, 4, 1];
        let data32 = |mem: &GuestMemory| {
            for (i, &x) in table32.iter().chain(index32.iter()).enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let prog32 = [
            vsetivli(0, 4, 0xd0),
            vle(0b110, 1, 11),
            vle(0b110, 2, 12),
            vv(3, 1, 2),
            vi(4, 1, 2),
            vse(0b110, 3, 13),
            addi(13, 13, 16),
            vse(0b110, 4, 13),
            ECALL,
        ];
        let mem = check(&prog32, &data32, &init);
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        let got: Vec<u32> = row.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        assert_eq!(got, [table32[3], table32[0], 0, table32[1]]);
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        assert!(row.as_chunks::<4>().0.iter().all(|b| u32::from_le_bytes(*b) == table32[2]));

        let table64: [u64; 2] = [0x1000_0000_0000_0001, 0x2000_0000_0000_0002];
        let index64: [u64; 2] = [1, 2];
        let data64 = |mem: &GuestMemory| {
            for (i, &x) in table64.iter().chain(index64.iter()).enumerate() {
                mem.write_privileged(DATA + 8 * i as u64, &x.to_le_bytes()).unwrap();
            }
        };
        let prog64 = [
            vsetivli(0, 2, 0xd8),
            vle(0b111, 1, 11),
            vle(0b111, 2, 12),
            vv(3, 1, 2),
            vi(4, 1, 1),
            vse(0b111, 3, 13),
            addi(13, 13, 16),
            vse(0b111, 4, 13),
            ECALL,
        ];
        let mem = check(&prog64, &data64, &init);
        let mut row = [0u8; 16];
        mem.read_privileged(DATA + 32, &mut row).unwrap();
        let got: Vec<u64> = row.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
        assert_eq!(got, [table64[1], 0]);
        mem.read_privileged(DATA + 48, &mut row).unwrap();
        assert!(row.as_chunks::<8>().0.iter().all(|b| u64::from_le_bytes(*b) == table64[1]));
    }

    #[test]
    fn vector_arith_chain_reuses_the_previous_destination() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle =
            |wf3: u32, vd: u32, rs1: u32| (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse = |wf3: u32, vs3: u32, rs1: u32| {
            (1 << 25) | (wf3 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27
        };
        let vv = |f6: u32, vd: u32, vs2: u32, vs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57
        };
        let vx = |f6: u32, vd: u32, vs2: u32, rs1: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (rs1 << 15) | (0b100 << 12) | (vd << 7) | 0x57
        };
        let vi = |f6: u32, vd: u32, vs2: u32, imm: u32| {
            (f6 << 26) | (1 << 25) | (vs2 << 20) | (imm << 15) | (0b011 << 12) | (vd << 7) | 0x57
        };
        let mvx = |vd: u32, vs2: u32, rs1: u32| {
            (0b100101 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b110 << 12)
                | (vd << 7)
                | 0x57
        };
        let prog = [
            vsetivli(0, 16, 0xc0), // e8, m1, ta, ma
            vle(0b000, 1, 11),
            vle(0b000, 2, 12),
            vi(0b101000, 1, 1, 3),  // vsrl.vi v1, v1, 3
            vx(0b101001, 1, 1, 14), // vsra.vx v1, v1, a4
            vv(0b000010, 1, 1, 2),  // vsub.vv v1, v1, v2
            mvx(1, 1, 15),          // vmul.vx v1, v1, a5
            vse(0b000, 1, 13),
            ECALL,
        ];
        let data = |mem: &GuestMemory| {
            mem.write_privileged(
                DATA,
                &[
                    0xff, 0x80, 0x7f, 0x01, 0xfe, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
                    0x90, 0xa0, 0xb0,
                ],
            )
            .unwrap();
            mem.write_privileged(
                DATA + 16,
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            )
            .unwrap();
        };
        let init = |cpu: &mut CpuState| {
            for (r, v) in [(11u8, DATA), (12, DATA + 16), (13, DATA + 32), (14, 3), (15, 3)] {
                cpu.set_x(r, v);
            }
        };
        let mem_i = setup(&prog);
        data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want, "trap");
        assert_eq!(cpu, oracle, "state");
        assert_eq!(jit.stats.interpreted.get("VArith"), None);
    }

    /// A loop whose head is a `vsetivli`: the whole body translates as one block (the
    /// constant vset is emitted inline), and the engines must agree after every trip.
    #[test]
    fn vector_loop_with_vsetivli_head_matches_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vadd_vv =
            |vd: u32, vs2: u32, vs1: u32| (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57;
        // vsetivli zero, 2, e64, m1; v2 = v1 + v1; [a2] = v2; a3 -= 1; bne a3, x0, -20
        let prog = [
            vsetivli(0, 2, 0x0d8),
            vle64(1, 11),
            vadd_vv(2, 1, 1),
            vse64(2, 12),
            addi(13, 13, -1),
            b(0b001, 13, 0, -20),
        ];
        let init = |cpu: &mut CpuState| {
            cpu.set_x(11, DATA);
            cpu.set_x(12, DATA + 16);
            cpu.set_x(13, 3);
        };
        let mem_i = setup(&prog);
        mem_i.write_privileged(DATA, &0x0000_0000_0000_0001u64.to_le_bytes()).unwrap();
        mem_i.write_privileged(DATA + 8, &0x0000_0000_0000_0002u64.to_le_bytes()).unwrap();
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        mem_j.write_privileged(DATA, &0x0000_0000_0000_0001u64.to_le_bytes()).unwrap();
        mem_j.write_privileged(DATA + 8, &0x0000_0000_0000_0002u64.to_le_bytes()).unwrap();
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        mem_i.read_privileged(DATA + 16, &mut a).unwrap();
        mem_j.read_privileged(DATA + 16, &mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(u64::from_le_bytes(a), 2, "v2 = v1 + v1");
        assert_eq!(cpu.x[13], 0, "three trips");
    }

    /// The glibc loop-head shape: `vsetvli` with a register AVL. The vset translates
    /// (a saturating min against vlmax) and the body translates under whatever state it
    /// wrote, so after the first round the whole loop runs natively.
    #[test]
    fn vector_loop_with_vsetvli_head_matches_the_interpreter() {
        let vsetvli = |rd: u32, rs1: u32, vtype: u32| {
            (vtype << 20) | (rs1 << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let vadd_vv =
            |vd: u32, vs2: u32, vs1: u32| (1 << 25) | (vs2 << 20) | (vs1 << 15) | (vd << 7) | 0x57;
        // vsetvli t0, a3, e64, m1; v2 = v1 + v1; [a2] = v2; a4 -= 1; bne a4, x0, -20
        let prog = [
            vsetvli(5, 13, 0x0d8),
            vle64(1, 11),
            vadd_vv(2, 1, 1),
            vse64(2, 12),
            addi(14, 14, -1),
            b(0b001, 14, 0, -20),
        ];
        let init = |cpu: &mut CpuState| {
            cpu.set_x(11, DATA);
            cpu.set_x(12, DATA + 16);
            cpu.set_x(13, 9); // AVL 9 saturates to vlmax = 2
            cpu.set_x(14, 3);
        };
        let mem_i = setup(&prog);
        mem_i.write_privileged(DATA, &0x0000_0000_0000_0001u64.to_le_bytes()).unwrap();
        mem_i.write_privileged(DATA + 8, &0x0000_0000_0000_0002u64.to_le_bytes()).unwrap();
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        mem_j.write_privileged(DATA, &0x0000_0000_0000_0001u64.to_le_bytes()).unwrap();
        mem_j.write_privileged(DATA + 8, &0x0000_0000_0000_0002u64.to_le_bytes()).unwrap();
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle);
        assert_eq!(cpu.x[5], 2, "t0 = vl = min(9, vlmax)");
        assert_eq!(cpu.x[14], 0, "three trips");
    }

    /// `vslidedown.vi` and `vfslide1down.vf` (the shapes the gelu's scalar expf calls
    /// live around), against the interpreter: the aliased and zero-filled slides, and a
    /// boxed and an unboxed scalar for the float form.
    #[test]
    fn vector_slides_match_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle32 = |vd: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse32 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b110 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        // Encodings from the assembler: the down-slides have funct6 = 001111
        // (vslidedown.vi funct3 = 011, vfslide1down.vf funct3 = 101); the up forms
        // use 001110.
        let vslidedown_vi = |vd: u32, vs2: u32, imm: u32| {
            (0b001111 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (imm << 15)
                | (0b011 << 12)
                | (vd << 7)
                | 0x57
        };
        let vslideup_vi = |vd: u32, vs2: u32, imm: u32| {
            (0b001110 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (imm << 15)
                | (0b011 << 12)
                | (vd << 7)
                | 0x57
        };
        let vfslide1down_vf = |vd: u32, vs2: u32, rs1: u32| {
            (0b001111 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (rs1 << 15)
                | (0b101 << 12)
                | (vd << 7)
                | 0x57
        };
        let prog = [
            vsetivli(0, 4, 0xd0),    // e32, m1, ta, ma: vl = 4
            vle32(8, 11),            // v8 = {1.0, 2.0, 3.0, 4.0}
            vslidedown_vi(8, 8, 1),  // v8 = {2, 3, 4, 0}, aliased vd == vs2
            vslidedown_vi(9, 8, 2),  // v9 = {4, 0, 0, 0}
            vslidedown_vi(10, 8, 5), // v10 = {0, 0, 0, 0}: offset past VLMAX
            addi(12, 11, 32),
            vse32(8, 12),
            addi(12, 11, 48),
            vse32(9, 12),
            addi(12, 11, 64),
            vse32(10, 12),
            fload(0b010, 0, 11, 16),   // flw ft0, 16(a1): 2.5, a boxed single
            fload(0b011, 1, 11, 24),   // fld ft1, 24(a1): a raw double, not boxed
            vfslide1down_vf(16, 9, 0), // v16 = {v9[1..4], 2.5} = {0, 0, 0, 2.5}
            vfslide1down_vf(17, 9, 1), // v17 = {0, 0, 0, canonical NaN}
            addi(12, 11, 80),
            vse32(16, 12),
            addi(12, 11, 96),
            vse32(17, 12),
            addi(12, 11, 160),
            vle32(19, 12),
            vslideup_vi(18, 8, 2),
            vslideup_vi(19, 19, 1),
            vslideup_vi(20, 8, 3),
            addi(12, 11, 112),
            vse32(18, 12),
            addi(12, 11, 128),
            vse32(19, 12),
            addi(12, 11, 144),
            vse32(20, 12),
            ECALL,
        ];
        let setup_data = |mem: &GuestMemory| {
            for (i, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
                mem.write_privileged(DATA + 4 * i as u64, &v.to_bits().to_le_bytes()).unwrap();
            }
            mem.write_privileged(DATA + 16, &2.5f32.to_bits().to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 24, &1.0f64.to_bits().to_le_bytes()).unwrap();
            for (i, v) in [10.0f32, 20.0, 30.0, 40.0].iter().enumerate() {
                mem.write_privileged(DATA + 160 + 4 * i as u64, &v.to_bits().to_le_bytes())
                    .unwrap();
            }
        };
        let mem_i = setup(&prog);
        setup_data(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        setup_data(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        let mut jit = Jit::new().unwrap();
        let got = jit.run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu.v, oracle.v, "vector register file");
        assert_eq!(cpu, oracle, "whole state");
        assert_eq!(jit.stats.interpreted.get("VArith"), None);
        let elem = |reg: usize, lane: usize| {
            u32::from_le_bytes(cpu.v[reg][4 * lane..4 * lane + 4].try_into().unwrap())
        };
        assert_eq!(elem(8, 0), 2.0f32.to_bits()); // the aliased slide's low element
        assert_eq!(elem(16, 3), 2.5f32.to_bits()); // the inserted scalar
        assert_eq!(elem(17, 3), 0x7fc0_0000); // canonical NaN: the double is not boxed
        assert_eq!(
            (elem(18, 0), elem(18, 1), elem(18, 2), elem(18, 3)),
            (0, 0, 2.0f32.to_bits(), 3.0f32.to_bits())
        );
        assert_eq!(
            (elem(19, 0), elem(19, 1), elem(19, 2), elem(19, 3)),
            (10.0f32.to_bits(), 10.0f32.to_bits(), 20.0f32.to_bits(), 30.0f32.to_bits())
        );
        assert_eq!(
            (elem(20, 0), elem(20, 1), elem(20, 2), elem(20, 3)),
            (0, 0, 0, 2.0f32.to_bits())
        );
    }

    /// The glibc scanner shape: `vle8ff.v`, `vmsne.vi`, `vmor.mm`, `vfirst.m`, and
    /// reading `vl`, all against the interpreter.
    #[test]
    fn vector_scanner_ops_match_the_interpreter() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle8ff =
            |vd: u32, rs1: u32| (0b10000 << 20) | (1 << 25) | (rs1 << 15) | (vd << 7) | 0x07;
        let vmsne_vi = |vd: u32, vs2: u32, imm: u32| {
            (0b011001 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (imm << 15)
                | (0b011 << 12)
                | (vd << 7)
                | 0x57
        };
        let vmor_mm = |vd: u32, vs1: u32, vs2: u32| {
            (0b011010 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (vs1 << 15)
                | (0b010 << 12)
                | (vd << 7)
                | 0x57
        };
        let vfirst_m = |rd: u32, vs2: u32| {
            (0b010000 << 26)
                | (1 << 25)
                | (vs2 << 20)
                | (0b10001 << 15)
                | (0b010 << 12)
                | (rd << 7)
                | 0x57
        };
        let csrr = |rd: u32, csr: u32| (csr << 20) | (0b010 << 12) | (rd << 7) | 0x73;
        let prog = [
            vsetivli(0, 16, 0xc0), // e8, m1, ta, ma
            vle8ff(1, 11),         // v1 = [a1] (fault-tolerant)
            vmsne_vi(2, 1, 0),     // v2 = v1 != 0
            vmor_mm(3, 2, 2),      // v3 = v2 | v2
            csrr(6, 0xc20),        // t1 = vl
            vfirst_m(5, 3),        // t0 = index of the first set mask bit
        ];
        // Hand-encodings must decode to the instructions this test means to exercise.
        for (i, want) in [(1, "VLoad"), (2, "VCmp"), (3, "VMask"), (5, "Vfirst")] {
            let name = format!("{:?}", crate::decode(prog[i]).unwrap());
            assert!(name.starts_with(want), "{i}: {name}");
        }
        let fill = |mem: &GuestMemory| {
            let mut bytes = [0u8; 16];
            bytes[1] = 7;
            bytes[3] = 0x80;
            mem.write_privileged(DATA, &bytes).unwrap();
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        oracle.set_x(11, DATA);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        cpu.set_x(11, DATA);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "scanner sequence");
        assert_eq!(cpu.x[6], 16, "t1 = vl");
        assert_eq!(cpu.x[5], 1, "the first set element is index 1");
        assert_eq!(cpu.v[2][0], 0b0000_1010, "vmsne packed bits");
    }

    /// A `vle8ff.v` that runs into the end of the mapping: `vl` shortens to the elements
    /// that loaded, no trap, and both engines land in the same state.
    #[test]
    fn vector_ff_load_shortens_at_a_page_boundary() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle8ff =
            |vd: u32, rs1: u32| (0b10000 << 20) | (1 << 25) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse8 = |vs3: u32, rs1: u32| (1 << 25) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let prog = [
            vsetivli(0, 16, 0xc0),
            vle8ff(1, 11), // a1 = the last 8 bytes of the DATA page: element 8 faults
            vse8(1, 12),   // dump v1 (only the loaded elements matter) to a2
        ];
        let fill = |mem: &GuestMemory| {
            // The probe crosses a host-page boundary (16 KiB on Apple silicon), the only
            // boundary both engines see alike: guest memory's host pages cover four
            // guest pages, and a sub-page gap inside a mapped host page is invisible to
            // generated code but not to the interpreter.
            mem.map(DATA + 0x1000, 0x3000, Prot::RW).unwrap();
            mem.write_privileged(DATA + 0x3ff8, &[1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        };
        let init = |cpu: &mut CpuState| {
            cpu.set_x(11, DATA + 0x3ff8);
            cpu.set_x(12, DATA + 16);
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu.vl, oracle.vl, "vl");
        assert_eq!(cpu.vtype, oracle.vtype, "vtype");
        assert_eq!(cpu.instret, oracle.instret, "instret");
        assert_eq!(cpu.v, oracle.v, "v registers");
        assert_eq!(cpu, oracle, "ff shortening");
        assert_eq!(cpu.vl, 8, "eight elements loaded before the fault");
        assert_eq!(cpu.v[1][..8], [1, 2, 3, 4, 5, 6, 7, 8]);
        let mut stored = [0u8; 8];
        mem_j.read_privileged(DATA + 16, &mut stored).unwrap();
        assert_eq!(stored, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// Two vector accesses sharing a base register in one block: the second used to
    /// trust the base-register cache, but the first left `ADDR` holding the address of a
    /// guest vector register instead of the base. (Coreutils caught this one: `uname`
    /// crashed with a load fault at address 8 before the fix.)
    #[test]
    fn vector_accesses_do_not_reuse_a_stale_cached_base() {
        let vsetivli = |rd: u32, avl: u32, vtype: u32| {
            (0b11 << 30) | (vtype << 20) | (avl << 15) | (0b111 << 12) | (rd << 7) | 0x57
        };
        let vle64 = |vd: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vd << 7) | 0x07;
        let vse64 =
            |vs3: u32, rs1: u32| (1 << 25) | (0b111 << 12) | (rs1 << 15) | (vs3 << 7) | 0x27;
        let prog = [
            vsetivli(0, 2, 0x0d8),
            vle64(1, 11), // v1 = [a1]
            vle64(2, 11), // v2 = [a1]: the second access must recompute the address
            vse64(2, 12), // [a2] = v2
        ];
        let fill = |mem: &GuestMemory| {
            mem.write_privileged(DATA, &0x1111_2222_3333_4444u64.to_le_bytes()).unwrap();
            mem.write_privileged(DATA + 8, &0x5555_6666_7777_8888u64.to_le_bytes()).unwrap();
        };
        let init = |cpu: &mut CpuState| {
            cpu.set_x(11, DATA);
            cpu.set_x(12, DATA + 16);
        };
        let mem_i = setup(&prog);
        fill(&mem_i);
        let mut oracle = CpuState::new(CODE);
        init(&mut oracle);
        let want = loop {
            if let Err(t) = interpreter::step(&mut oracle, &mem_i) {
                break t;
            }
        };
        let mem_j = setup(&prog);
        fill(&mem_j);
        let mut cpu = CpuState::new(CODE);
        init(&mut cpu);
        let got = Jit::new().unwrap().run(&mut cpu, &mem_j);
        assert_eq!(got, want);
        assert_eq!(cpu, oracle, "stale-base reuse");
        assert_eq!(cpu.v[1], cpu.v[2], "both loads read the same guest bytes");
        let mut stored = [0u8; 16];
        mem_j.read_privileged(DATA + 16, &mut stored).unwrap();
        assert_eq!(u64::from_le_bytes(stored[..8].try_into().unwrap()), 0x1111_2222_3333_4444);
        assert_eq!(u64::from_le_bytes(stored[8..].try_into().unwrap()), 0x5555_6666_7777_8888);
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
                        // Zba, Zbb, Zbs, and Zicond register forms.
                        (2, 0x10),
                        (4, 0x10),
                        (6, 0x10),
                        (7, 0x10),
                        (6, 0x20),
                        (4, 0x20),
                        (4, 0x5),
                        (5, 0x5),
                        (6, 0x5),
                        (7, 0x5),
                        (1, 0x30),
                        (5, 0x30),
                        (1, 0x14),
                        (1, 0x24),
                        (5, 0x24),
                        (1, 0x34),
                        (5, 0x7),
                        (7, 0x7),
                    ]);
                    Plain(op(f3, f7, int_dest(rng), reg(rng), reg(rng)))
                }
                5..=7 => {
                    let f3 = rng.below(8) as u32;
                    let imm = match f3 {
                        // The Zbb/Zbs unary and immediate forms pack a funct6 above a
                        // 5-bit rs2/shift field; slli.uw is funct6 000100 over a 6-bit
                        // shift.
                        1 if rng.chance(2) => match rng.below(5) {
                            0 => (0b011_000 << 6) | rng.pick(&[0, 1, 2, 4, 5]),
                            1 => (0b010_010 << 6) | rng.below(64) as i32,
                            2 => (0b011_010 << 6) | rng.below(64) as i32,
                            3 => (0b001_010 << 6) | rng.below(64) as i32,
                            _ => (0b000_100 << 6) | rng.below(64) as i32,
                        },
                        5 if rng.chance(2) => match rng.below(4) {
                            0 => (0b011_000 << 6) | rng.below(64) as i32, // rori
                            1 => (0b001_010 << 6) | 0b00111,              // orc.b
                            2 => (0b011_010 << 6) | 0b11000,              // rev8
                            _ => (0b010_010 << 6) | rng.below(64) as i32, // bexti
                        },
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
                        // Zba word adds, Zbb zext.h, and the Zbb word rotates.
                        (2, 0x10),
                        (4, 0x10),
                        (6, 0x10),
                        (0, 0x4),
                        (4, 0x4),
                        (1, 0x30),
                        (5, 0x30),
                    ]);
                    if rng.chance(2) {
                        // zext.h keeps its second source at zero.
                        let rs2 = if (f3, f7) == (4, 0x4) { 0 } else { reg(rng) };
                        Plain(op_32(f3, f7, int_dest(rng), reg(rng), rs2))
                    } else {
                        let (f3, imm) = match f3 {
                            // clzw/ctzw/cpopw and roriw share a funct6 011000 selector.
                            1 if rng.chance(2) => (1, 0x600 | rng.below(3) as i32),
                            5 if rng.chance(2) => (5, 0x600 | rng.below(32) as i32),
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
