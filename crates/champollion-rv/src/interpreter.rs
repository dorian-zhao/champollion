//! Reference interpreter. Slow on purpose: it is the oracle the JIT is diffed against.

use crate::cpu::CpuState;
use crate::decoder::{self, DecodeError, Inst};
use crate::memory::{MemFault, Memory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    FetchFault(MemFault),
    IllegalInstruction {
        pc: u64,
        err: DecodeError,
    },
    /// Control transfer to a pc that is not 2-byte aligned (4 without the C extension,
    /// but we plan for C from the start).
    MisalignedFetch {
        pc: u64,
    },
}

/// Fetch, decode and execute one instruction.
pub fn step(cpu: &mut CpuState, mem: &mut impl Memory) -> Result<(), Trap> {
    let pc = cpu.pc;
    let lo = mem.read_u16(pc).map_err(Trap::FetchFault)?;
    let len = decoder::inst_len(lo);
    let word = if len == 4 {
        mem.read_u32(pc).map_err(Trap::FetchFault)?
    } else {
        lo as u32
    };
    let inst = decoder::decode(word).map_err(|err| Trap::IllegalInstruction { pc, err })?;
    execute(cpu, inst, len)
}

/// Run until a trap. Returns the trap and the number of instructions retired before it.
pub fn run(cpu: &mut CpuState, mem: &mut impl Memory, max_steps: u64) -> (Option<Trap>, u64) {
    for n in 0..max_steps {
        if let Err(t) = step(cpu, mem) {
            return (Some(t), n);
        }
    }
    (None, max_steps)
}

fn execute(cpu: &mut CpuState, inst: Inst, len: u64) -> Result<(), Trap> {
    let pc = cpu.pc;
    let mut next = pc.wrapping_add(len);
    match inst {
        Inst::Add { rd, rs1, rs2 } => cpu.set_x(rd, cpu.x(rs1).wrapping_add(cpu.x(rs2))),
        Inst::Addi { rd, rs1, imm } => cpu.set_x(rd, cpu.x(rs1).wrapping_add(imm as u64)),
        Inst::Beq { rs1, rs2, offset } => {
            if cpu.x(rs1) == cpu.x(rs2) {
                next = pc.wrapping_add(offset as u64);
            }
        }
        Inst::Jal { rd, offset } => {
            next = pc.wrapping_add(offset as u64);
            // Check before writing rd so a trapping jal leaves state untouched.
            if next & 1 != 0 {
                return Err(Trap::MisalignedFetch { pc: next });
            }
            cpu.set_x(rd, pc.wrapping_add(len));
        }
    }
    cpu.pc = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::tests::{add, addi, beq, jal};
    use crate::memory::FlatMemory;

    const BASE: u64 = 0x8000_0000;

    fn setup(prog: &[u32]) -> (CpuState, FlatMemory) {
        let mut mem = FlatMemory::new(BASE, 4096);
        mem.load_words(BASE, prog).unwrap();
        (CpuState::new(BASE), mem)
    }

    #[test]
    fn add_and_addi() {
        let (mut cpu, mut mem) = setup(&[
            addi(10, 0, 40),  // a0 = 40
            addi(11, 0, 2),   // a1 = 2
            add(12, 10, 11),  // a2 = a0 + a1
            addi(13, 0, -1),  // a3 = -1
            add(14, 13, 13),  // a4 = -2 (wrapping)
            addi(0, 10, 123), // write to x0 is discarded
        ]);
        let (trap, n) = run(&mut cpu, &mut mem, 6);
        assert_eq!((trap, n), (None, 6));
        assert_eq!(cpu.x[12], 42);
        assert_eq!(cpu.x[13], u64::MAX);
        assert_eq!(cpu.x[14], (-2i64) as u64);
        assert_eq!(cpu.x[0], 0);
        assert_eq!(cpu.pc, BASE + 24);
    }

    #[test]
    fn beq_taken_and_not_taken() {
        let (mut cpu, mut mem) = setup(&[
            addi(10, 0, 1),  // 0x00
            beq(10, 0, 8),   // 0x04: not taken
            beq(0, 0, 8),    // 0x08: taken -> 0x10
            addi(11, 0, 99), // 0x0c: skipped
            addi(12, 0, 7),  // 0x10
        ]);
        run(&mut cpu, &mut mem, 4);
        assert_eq!(cpu.x[11], 0);
        assert_eq!(cpu.x[12], 7);
        assert_eq!(cpu.pc, BASE + 0x14);
    }

    #[test]
    fn beq_backward_loop() {
        // Count a0 up to 5 with a backward branch:
        //   0x00: addi a1, zero, 5
        //   0x04: addi a0, a0, 1
        //   0x08: beq  a0, a1, +8   -> exit
        //   0x0c: beq  zero, zero, -8 -> 0x04
        //   0x10: (exit)
        let (mut cpu, mut mem) = setup(&[
            addi(11, 0, 5),
            addi(10, 10, 1),
            beq(10, 11, 8),
            beq(0, 0, -8),
        ]);
        while cpu.pc != BASE + 0x10 {
            step(&mut cpu, &mut mem).unwrap();
        }
        assert_eq!(cpu.x[10], 5);
    }

    #[test]
    fn jal_links_and_jumps() {
        let (mut cpu, mut mem) = setup(&[
            jal(1, 12),     // 0x00: ra = 0x04, -> 0x0c
            addi(10, 0, 1), // 0x04: return lands here
            jal(0, 0),      // 0x08: j . (halt loop)
            addi(11, 0, 2), // 0x0c: "callee"
            jal(5, -12),    // 0x10: t0 = 0x14, -> 0x04
        ]);
        step(&mut cpu, &mut mem).unwrap();
        assert_eq!(cpu.x[1], BASE + 4);
        assert_eq!(cpu.pc, BASE + 0x0c);
        run(&mut cpu, &mut mem, 3);
        assert_eq!(cpu.x[10], 1);
        assert_eq!(cpu.x[11], 2);
        assert_eq!(cpu.x[5], BASE + 0x14);
        assert_eq!(cpu.pc, BASE + 0x08);
        // `j .` spins in place and never writes x0.
        run(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.pc, BASE + 0x08);
        assert_eq!(cpu.x[0], 0);
    }

    #[test]
    fn traps_are_reported() {
        let (mut cpu, mut mem) = setup(&[0xffff_ffff]);
        assert!(matches!(
            step(&mut cpu, &mut mem),
            Err(Trap::IllegalInstruction { pc: BASE, .. })
        ));

        let mut cpu = CpuState::new(0x10);
        assert_eq!(
            step(&mut cpu, &mut mem),
            Err(Trap::FetchFault(MemFault { addr: 0x10 }))
        );
    }
}
