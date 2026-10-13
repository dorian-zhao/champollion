//! Basic-block JIT to arm64 (M4).
//!
//! - [`arm64`]: instruction encoder, checked against an assembler.
//! - [`code`]: executable memory (MAP_JIT and W^X on macOS).
//! - [`translate`]: guest block -> arm64, plus the enter/exit trampoline.
//!
//! [`Jit::run`] is the dispatcher: it looks up (or translates) the block at the guest pc,
//! enters it, and acts on the exit code. Instructions the translator does not handle yet
//! run on the reference interpreter, one at a time, so coverage can grow incrementally
//! without ever being wrong.

pub mod arm64;
pub mod code;
pub mod translate;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::cpu::CpuState;
use crate::engine::decode_block;
use crate::interpreter::{self, Trap};
use crate::memory::GuestMemory;
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
    generation: u64,
}

impl Jit {
    pub fn new() -> std::io::Result<Self> {
        let code = CodeBuffer::new(CODE_SIZE)?;
        // SAFETY: placeholder, replaced by `reset` before any use.
        let enter: EnterFn = unsafe { std::mem::transmute::<*const u8, EnterFn>(code.base()) };
        let mut jit = Jit { code, enter, stub_word: 0, blocks: HashMap::default(), generation: 0 };
        jit.reset();
        Ok(jit)
    }

    /// Drop every translation and regenerate the trampoline at the start of the buffer.
    fn reset(&mut self) {
        self.code.reset();
        self.blocks.clear();
        let (words, stub) = translate::trampoline();
        let at = self.code.append(&words).expect("trampoline fits");
        // SAFETY: the trampoline is a C-ABI function with this signature.
        self.enter = unsafe { std::mem::transmute::<*const u8, EnterFn>(self.code.ptr(at)) };
        self.stub_word = at / 4 + stub;
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
            let Some(t) = translate::translate(&block, pc, word, self.stub_word as i64) else {
                return Ok(None);
            };
            match self.code.append(&t.code) {
                Some(at) => {
                    self.blocks.insert(pc, at);
                    return Ok(Some(at));
                }
                None if attempt == 0 => self.reset(), // buffer full: start over
                None => panic!("a single block does not fit in the code buffer"),
            }
        }
        unreachable!()
    }

    /// Run until a trap (including `ecall`), like [`interpreter::run`] with no step limit.
    pub fn run(&mut self, cpu: &mut CpuState, mem: &GuestMemory) -> Trap {
        loop {
            let generation = mem.generation();
            if generation != self.generation {
                self.reset();
                self.generation = generation;
            }
            let at = match self.lookup(cpu.pc, mem) {
                Ok(Some(at)) => at,
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
            let code = unsafe { (self.enter)(cpu, mem.host_base(), self.code.ptr(at)) };
            match code {
                exit::NEXT => {}
                exit::ECALL => return Trap::Ecall { pc: cpu.pc },
                exit::INTERPRET => {
                    if let Err(trap) = interpreter::step(cpu, mem) {
                        return trap;
                    }
                }
                exit::FENCE_I => self.reset(),
                other => panic!("bad exit code {other}"),
            }
        }
    }
}
