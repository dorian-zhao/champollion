//! The fast interpreter: decoded-block cache plus software TLB.
//!
//! Borrowed from how Rosetta 2 and QEMU structure translation: work is done per basic
//! block, once, and cached by guest pc. Here the "translation" is just the decoded
//! [`Inst`]s of the block; in M4 the same cache, keyed and invalidated the same way, will
//! hold arm64 code instead. Executing a cached block skips fetch and decode entirely, and
//! data accesses go through the [`Tlb`] instead of the locked mapping table.
//!
//! Behavior is identical to [`interpreter::step`] (the oracle), which the riscv-tests
//! suites check register-for-register. Cached blocks are dropped when:
//! - [`GuestMemory::generation`] moves (any mapping change, `riscv_flush_icache`),
//!   checked before every block;
//! - this thread executes `fence.i`, which is exactly when RISC-V requires a hart to
//!   observe its own code modifications. Code modified without `fence.i` may keep running
//!   stale instructions, as it may on real hardware.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::cpu::CpuState;
use crate::decoder::{self, Inst};
use crate::interpreter::{self, Trap};
use crate::memory::{GUEST_PAGE_SIZE, GuestMemory, Memory, Tlb, TlbMemory};

/// Longest block we decode ahead, in instructions.
const MAX_BLOCK: usize = 64;

/// A decoded basic block: instructions with their lengths in bytes (2 or 4).
type Block = Box<[(Inst, u8)]>;

/// Hasher for guest pcs: they are already well distributed in their low bits (after
/// dropping bit 0), so one multiply is plenty.
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

/// A cached block plus the blocks most recently seen to follow it. Following a successor
/// link skips the hash lookup; it is the interpreter's version of block chaining.
struct CachedBlock {
    insts: Block,
    /// `(pc, index into Engine::blocks)` of the last two distinct successors, newest
    /// first; `u32::MAX` marks an empty slot. Two cover both sides of a branch.
    successors: [(u64, u32); 2],
}

impl CachedBlock {
    #[inline]
    fn successor(&self, pc: u64) -> Option<u32> {
        self.successors.iter().find(|&&(p, i)| p == pc && i != u32::MAX).map(|&(_, i)| i)
    }

    fn remember(&mut self, pc: u64, index: u32) {
        self.successors = [(pc, index), self.successors[0]];
    }
}

/// Decoded blocks; indices stay valid until the next flush.
#[derive(Default)]
struct BlockCache {
    blocks: Vec<CachedBlock>,
    by_pc: HashMap<u64, u32, BuildHasherDefault<PcHasher>>,
}

impl BlockCache {
    fn clear(&mut self) {
        self.blocks.clear();
        self.by_pc.clear();
    }

    /// Index of the block at `pc`, decoding it on first use.
    fn lookup(&mut self, tm: &TlbMemory, pc: u64) -> Result<u32, Trap> {
        if let Some(&i) = self.by_pc.get(&pc) {
            return Ok(i);
        }
        let insts = decode_block(tm, pc)?;
        let i = self.blocks.len() as u32;
        self.blocks.push(CachedBlock { insts, successors: [(0, u32::MAX); 2] });
        self.by_pc.insert(pc, i);
        Ok(i)
    }
}

/// Per-thread execution engine.
#[derive(Default)]
pub struct Engine {
    tlb: Tlb,
    generation: u64,
    cache: BlockCache,
}

/// Whether `inst` may change the pc non-sequentially (or must end a block for another
/// reason), so nothing after it belongs to the same block.
fn ends_block(inst: Inst) -> bool {
    matches!(
        inst,
        Inst::Jal { .. }
            | Inst::Jalr { .. }
            | Inst::Branch { .. }
            | Inst::Ecall
            | Inst::Ebreak
            | Inst::FenceI
    )
}

/// Decode the block starting at `pc`. Fails only if its very first instruction cannot be
/// fetched or decoded, with the same trap `interpreter::step` would raise; later failures
/// just end the block early, so the trap is raised when execution actually gets there.
fn decode_block(mem: &impl Memory, pc: u64) -> Result<Block, Trap> {
    let mut insts = Vec::new();
    let mut at = pc;
    loop {
        let fetched = mem.fetch_u16(at).and_then(|lo| {
            let len = decoder::inst_len(lo);
            let word = if len == 4 { mem.fetch_u32(at)? } else { lo as u32 };
            Ok((word, len))
        });
        let (word, len) = match fetched {
            Ok(v) => v,
            Err(fault) if insts.is_empty() => return Err(Trap::FetchFault(fault)),
            Err(_) => break,
        };
        let inst = match decoder::decode(word) {
            Ok(i) => i,
            Err(err) if insts.is_empty() => return Err(Trap::IllegalInstruction { pc: at, err }),
            Err(_) => break,
        };
        insts.push((inst, len as u8));
        at += len;
        // Stop at page boundaries so a block never depends on two pages' permissions.
        if ends_block(inst) || insts.len() == MAX_BLOCK || at.is_multiple_of(GUEST_PAGE_SIZE) {
            break;
        }
    }
    Ok(insts.into_boxed_slice())
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    fn flush(&mut self) {
        self.tlb.flush();
        self.cache.clear();
    }

    /// Run until a trap or `max_steps` retired instructions; same contract as
    /// [`interpreter::run`].
    pub fn run(
        &mut self,
        cpu: &mut CpuState,
        mem: &GuestMemory,
        max_steps: u64,
    ) -> (Option<Trap>, u64) {
        let mut retired = 0;
        let mut prev: Option<u32> = None;
        while retired < max_steps {
            let generation = mem.generation();
            if generation != self.generation {
                self.flush();
                self.generation = generation;
                prev = None;
            }
            let tm = TlbMemory { mem, tlb: &self.tlb };
            let cache = &mut self.cache;
            let pc = cpu.pc;
            let index = match prev.and_then(|p| cache.blocks[p as usize].successor(pc)) {
                Some(i) => i,
                None => match cache.lookup(&tm, pc) {
                    Ok(i) => {
                        if let Some(p) = prev {
                            cache.blocks[p as usize].remember(pc, i);
                        }
                        i
                    }
                    Err(trap) => return (Some(trap), retired),
                },
            };

            let mut fence_i = false;
            for &(inst, len) in cache.blocks[index as usize].insts.iter() {
                if retired == max_steps {
                    return (None, retired);
                }
                if let Err(trap) = interpreter::execute(cpu, &tm, inst, len as u64) {
                    return (Some(trap), retired);
                }
                retired += 1;
                fence_i = inst == Inst::FenceI;
            }
            prev = Some(index);
            if fence_i {
                cache.clear();
                prev = None;
            }
        }
        (None, retired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::enc::*;
    use crate::memory::Prot;

    const CODE: u64 = 0x10000;
    const DATA: u64 = 0x20000;

    fn setup(prog: &[u32]) -> (CpuState, GuestMemory) {
        let mem = GuestMemory::new().unwrap();
        mem.map(CODE, 0x1000, Prot::READ | Prot::EXEC).unwrap();
        mem.map(DATA, 0x1000, Prot::RW).unwrap();
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(CODE, &bytes).unwrap();
        (CpuState::new(CODE), mem)
    }

    #[test]
    fn runs_loops_from_the_cache() {
        // a0 = 1000 iterations of a1 += 3
        let (mut cpu, mem) = setup(&[
            addi(10, 0, 1000),   // 0x00
            addi(11, 11, 3),     // 0x04
            addi(10, 10, -1),    // 0x08
            b(0b001, 10, 0, -8), // 0x0c: bnez a0, 0x04
            jal(0, 0),           // 0x10: j .
        ]);
        let mut e = Engine::new();
        let (trap, n) = e.run(&mut cpu, &mem, 1 + 3 * 1000);
        assert_eq!((trap, n), (None, 3001));
        assert_eq!(cpu.x[11], 3000);
        assert_eq!(cpu.pc, CODE + 0x10);
        assert!(e.cache.blocks.len() <= 3, "the loop body is decoded once");
        assert!(
            e.cache.blocks.iter().any(|b| b.successor(CODE + 4).is_some()),
            "the loop chains to itself"
        );
    }

    #[test]
    fn stops_exactly_at_max_steps_and_traps_like_the_oracle() {
        let prog = [addi(10, 0, 1), addi(10, 10, 1), addi(10, 10, 1), 0xffff_ffff];
        let (mut cpu, mem) = setup(&prog);
        let mut e = Engine::new();
        assert_eq!(e.run(&mut cpu, &mem, 2), (None, 2));
        assert_eq!((cpu.x[10], cpu.pc), (2, CODE + 8));

        let (trap, n) = e.run(&mut cpu, &mem, 100);
        let (mut cpu2, mem2) = setup(&prog);
        cpu2.pc = CODE + 12;
        cpu2.x[10] = 3;
        assert_eq!(trap, interpreter::step(&mut cpu2, &mem2).err());
        assert_eq!(n, 1);
        assert_eq!(cpu.pc, CODE + 12, "pc stays on the trapping instruction");

        // Fetching from a non-executable page faults.
        cpu.pc = DATA;
        assert_eq!(
            e.run(&mut cpu, &mem, 1),
            (Some(Trap::FetchFault(crate::memory::MemFault { addr: DATA })), 0)
        );
    }

    #[test]
    fn mapping_changes_and_fence_i_invalidate_blocks() {
        // Code lives in a writable page so the guest can patch it.
        let mem = GuestMemory::new().unwrap();
        mem.map(CODE, 0x1000, Prot::READ | Prot::WRITE | Prot::EXEC).unwrap();
        let patch = addi(10, 0, 2); // replaces `addi a0, zero, 1` at 0x00
        let prog = [
            addi(10, 0, 1), // 0x00
            jal(0, 0),      // 0x04: j .
        ];
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(CODE, &bytes).unwrap();
        let mut e = Engine::new();
        let mut cpu = CpuState::new(CODE);
        e.run(&mut cpu, &mem, 1);
        assert_eq!(cpu.x[10], 1);

        // A store without fence.i may keep running the cached (stale) block...
        mem.write(CODE, &patch.to_le_bytes()).unwrap();
        cpu.pc = CODE;
        e.run(&mut cpu, &mem, 1);
        assert_eq!(cpu.x[10], 1, "stale until fence.i, as RISC-V allows");

        // ...but fence.i makes the new code visible.
        let fence_i_at = CODE + 0x100;
        mem.write_privileged(fence_i_at, &0x0000_100fu32.to_le_bytes()).unwrap();
        mem.write_privileged(fence_i_at + 4, &jal(0, 0).to_le_bytes()).unwrap();
        cpu.pc = fence_i_at;
        e.run(&mut cpu, &mem, 1);
        cpu.pc = CODE;
        e.run(&mut cpu, &mem, 1);
        assert_eq!(cpu.x[10], 2);

        // So does any mapping change, via the generation counter.
        mem.write(CODE, &addi(10, 0, 3).to_le_bytes()).unwrap();
        mem.map(0x30000, 0x1000, Prot::RW).unwrap();
        cpu.pc = CODE;
        e.run(&mut cpu, &mem, 1);
        assert_eq!(cpu.x[10], 3);
    }
}
