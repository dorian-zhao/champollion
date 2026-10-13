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

type EnterFn = unsafe extern "C" fn(*mut CpuState, *mut u8, *const u8) -> u64;

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
            "jit: {} entries, {} translations, {} resets, {total} interpreted [{}]",
            self.entries,
            self.translations,
            self.resets,
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
        };
        jit.reset();
        Ok(jit)
    }

    /// Drop every translation and regenerate the trampoline at the start of the buffer.
    fn reset(&mut self) {
        self.code.reset();
        self.stats.resets += 1;
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
            self.stats.entries += 1;
            let code = {
                let _active = fault::Active::enter(&self.faults);
                unsafe { (self.enter)(cpu, mem.host_base(), self.code.ptr(at)) }
            };
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
    fn untranslated_instructions_fall_back_to_the_interpreter() {
        // amoadd (not translated yet) between translated instructions.
        let (trap, cpu) = same_as_interpreter(&[
            addi(12, 0, 3),
            amo(0, 0b011, 13, 11, 12), // amoadd.d a3, a2, (a1)
            amo(0, 0b011, 13, 11, 12),
            load(3, 14, 11, 0), // ld a4, 0(a1)
            ECALL,
        ]);
        assert!(matches!(trap, Trap::Ecall { .. }));
        assert_eq!((cpu.x[13], cpu.x[14]), (3, 6));
    }
}
