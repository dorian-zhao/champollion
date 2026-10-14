//! Linux signals: dispositions, masks, pending sets, and the riscv64 signal frame.
//!
//! Dispositions (`rt_sigaction`) belong to the process; masks, pending thread-directed
//! signals, and the alternate stack to each thread. Signals are delivered by the thread's
//! run loop at safe points: after a syscall, and when it traps (a fault becomes SIGSEGV
//! and so on, if the guest handles it). Delivering pushes the frame Linux builds on riscv64
//! (`siginfo`, then `ucontext` with the registers) and enters the handler with `ra` at a
//! trampoline that calls `rt_sigreturn`, which restores the frame.
//!
//! A sender also sets the recipient's interrupt flag (in guest memory's control area),
//! which translated code checks on backward and indirect jumps and the engine between
//! slices of instructions, so even a thread spinning in a loop takes the signal promptly.
//!
//! Limitations: blocked syscalls are not interrupted; realtime signals do not queue (one
//! pending instance per signal, like standard signals).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cpu::CpuState;
use crate::memory::{MemFault, Memory};

pub const SIGHUP: i32 = 1;
pub const SIGINT: i32 = 2;
pub const SIGQUIT: i32 = 3;
pub const SIGILL: i32 = 4;
pub const SIGTRAP: i32 = 5;
pub const SIGABRT: i32 = 6;
pub const SIGBUS: i32 = 7;
pub const SIGKILL: i32 = 9;
pub const SIGSEGV: i32 = 11;
pub const SIGPIPE: i32 = 13;
pub const SIGTERM: i32 = 15;
pub const SIGCHLD: i32 = 17;
pub const SIGCONT: i32 = 18;
pub const SIGSTOP: i32 = 19;
pub const SIGTSTP: i32 = 20;
pub const SIGTTIN: i32 = 21;
pub const SIGTTOU: i32 = 22;
pub const SIGURG: i32 = 23;
pub const SIGWINCH: i32 = 28;
pub const NSIG: i32 = 64;

pub const SIG_DFL: u64 = 0;
pub const SIG_IGN: u64 = 1;
pub const SA_SIGINFO: u64 = 4;
pub const SA_ONSTACK: u64 = 0x0800_0000;
pub const SA_RESTART: u64 = 0x1000_0000;
pub const SA_NODEFER: u64 = 0x4000_0000;
pub const SA_RESETHAND: u64 = 0x8000_0000;

pub const SS_ONSTACK: u32 = 1;
pub const SS_DISABLE: u32 = 2;

/// `si_code` values.
pub const SI_USER: i32 = 0;
pub const SI_TKILL: i32 = -6;
pub const SEGV_MAPERR: i32 = 1;
pub const ILL_ILLOPC: i32 = 1;
pub const BUS_ADRALN: i32 = 1;
pub const TRAP_BRKPT: i32 = 1;

/// The bit for signal `sig` (1-based) in a signal set.
pub fn bit(sig: i32) -> u64 {
    1 << (sig - 1)
}

/// Signals that can be neither caught, blocked, nor ignored.
pub const UNBLOCKABLE: u64 = 1 << (SIGKILL - 1) | 1 << (SIGSTOP - 1);

/// A signal's disposition, as in the riscv64 kernel `struct sigaction` (no restorer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Action {
    pub handler: u64,
    pub flags: u64,
    pub mask: u64,
}

impl Action {
    pub fn from_bytes(b: &[u8; 24]) -> Action {
        let word = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        Action { handler: word(0), flags: word(1), mask: word(2) }
    }

    pub fn to_bytes(self) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[..8].copy_from_slice(&self.handler.to_le_bytes());
        b[8..16].copy_from_slice(&self.flags.to_le_bytes());
        b[16..].copy_from_slice(&self.mask.to_le_bytes());
        b
    }
}

/// What a signal does when its handler is `SIG_DFL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultAction {
    Terminate,
    Ignore,
    /// Job control (stop/continue). There is no job control here, so these are ignored.
    Stop,
}

pub fn default_action(sig: i32) -> DefaultAction {
    match sig {
        SIGCHLD | SIGURG | SIGWINCH | SIGCONT => DefaultAction::Ignore,
        SIGSTOP | SIGTSTP | SIGTTIN | SIGTTOU => DefaultAction::Stop,
        _ => DefaultAction::Terminate,
    }
}

/// The `siginfo_t` fields we fill in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Info {
    /// Sent with kill/tgkill: `si_code`, sender pid and uid.
    Sent { code: i32, pid: u32, uid: u32 },
    /// Raised by a fault: `si_code` and `si_addr`.
    Fault { code: i32, addr: u64 },
}

impl Info {
    /// The 128-byte `siginfo_t` for signal `sig`.
    pub fn to_bytes(self, sig: i32) -> [u8; 128] {
        let mut b = [0u8; 128];
        b[0..4].copy_from_slice(&sig.to_le_bytes());
        match self {
            Info::Sent { code, pid, uid } => {
                b[8..12].copy_from_slice(&code.to_le_bytes());
                b[16..20].copy_from_slice(&pid.to_le_bytes());
                b[20..24].copy_from_slice(&uid.to_le_bytes());
            }
            Info::Fault { code, addr } => {
                b[8..12].copy_from_slice(&code.to_le_bytes());
                b[16..24].copy_from_slice(&addr.to_le_bytes());
            }
        }
        b
    }
}

/// Pending signals and their info: one instance per signal.
#[derive(Debug, Default)]
pub struct Pending {
    pub set: u64,
    info: Vec<(i32, Info)>,
}

impl Pending {
    pub fn add(&mut self, sig: i32, info: Info) {
        if self.set & bit(sig) == 0 {
            self.set |= bit(sig);
            self.info.push((sig, info));
        }
    }

    /// Remove and return the lowest-numbered pending signal not in `blocked`.
    pub fn take(&mut self, blocked: u64) -> Option<(i32, Info)> {
        let ready = self.set & !blocked;
        if ready == 0 {
            return None;
        }
        let sig = ready.trailing_zeros() as i32 + 1;
        self.set &= !bit(sig);
        let i = self.info.iter().position(|&(s, _)| s == sig)?;
        Some(self.info.swap_remove(i))
    }

    /// Drop pending instances of the signals in `set` (they became ignored).
    pub fn discard(&mut self, set: u64) {
        self.set &= !set;
        self.info.retain(|&(s, _)| set & bit(s) == 0);
    }
}

/// A thread's signal state that other threads touch: its mask (to pick a recipient for
/// process-directed signals) and its pending thread-directed signals.
#[derive(Debug, Default)]
pub struct ThreadSignals {
    pub mask: AtomicU64,
    pub pending: Mutex<Pending>,
    /// The thread's interrupt flag slot in guest memory's control area: senders set the
    /// flag to make the thread leave generated code and take the signal.
    pub slot: usize,
}

impl ThreadSignals {
    pub fn mask(&self) -> u64 {
        self.mask.load(Ordering::Relaxed)
    }

    pub fn set_mask(&self, mask: u64) {
        self.mask.store(mask & !UNBLOCKABLE, Ordering::Relaxed);
    }
}

/// `sigaltstack` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AltStack {
    pub sp: u64,
    pub flags: u32,
    pub size: u64,
}

impl AltStack {
    pub fn enabled(&self) -> bool {
        self.flags & SS_DISABLE == 0 && self.size != 0
    }

    pub fn contains(&self, sp: u64) -> bool {
        self.enabled() && sp > self.sp && sp - self.sp <= self.size
    }

    /// `stack_t` as `sigaltstack` reports it for a thread whose stack pointer is `sp`.
    pub fn to_bytes(self, sp: u64) -> [u8; 24] {
        let flags = if !self.enabled() {
            SS_DISABLE
        } else if self.contains(sp) {
            SS_ONSTACK
        } else {
            0
        };
        let mut b = [0u8; 24];
        b[..8].copy_from_slice(&self.sp.to_le_bytes());
        b[8..12].copy_from_slice(&flags.to_le_bytes());
        b[16..].copy_from_slice(&self.size.to_le_bytes());
        b
    }
}

// rt_sigframe layout (see the module docs): siginfo, then ucontext at 128.
const UC: usize = 128;
const UC_STACK: usize = UC + 16;
const UC_SIGMASK: usize = UC + 40;
/// `uc_mcontext`, 16-byte aligned: 32 GPRs (pc first), then the FP state.
const MCONTEXT: usize = UC + 176;
const FPREGS: usize = MCONTEXT + 256;
const FCSR: usize = FPREGS + 256;
/// Size of `rt_sigframe`: `__riscv_q_ext_state` reserves 528 bytes of FP state.
pub const FRAME_SIZE: u64 = (FPREGS + 528) as u64;

/// Push a signal frame for `sig` onto the guest stack and point `cpu` at `handler`, with
/// `ra` at `trampoline`. `mask` is the mask to restore on return, `alt` the thread's
/// alternate stack (used if the handler asks for it). On failure (an unwritable stack)
/// `cpu` is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn push_frame(
    mem: &impl Memory,
    cpu: &mut CpuState,
    sig: i32,
    info: Info,
    action: &Action,
    mask: u64,
    alt: AltStack,
    trampoline: u64,
) -> Result<(), MemFault> {
    let on_alt = action.flags & SA_ONSTACK != 0 && alt.enabled() && !alt.contains(cpu.x[2]);
    let top = if on_alt { alt.sp + alt.size } else { cpu.x[2] };
    let frame = top.wrapping_sub(FRAME_SIZE) & !15;
    let mut b = vec![0u8; FRAME_SIZE as usize];
    b[..128].copy_from_slice(&info.to_bytes(sig));
    b[UC_STACK..UC_STACK + 24].copy_from_slice(&alt.to_bytes(cpu.x[2]));
    b[UC_SIGMASK..UC_SIGMASK + 8].copy_from_slice(&mask.to_le_bytes());
    for r in 0..32 {
        let v = if r == 0 { cpu.pc } else { cpu.x[r] };
        b[MCONTEXT + 8 * r..MCONTEXT + 8 * r + 8].copy_from_slice(&v.to_le_bytes());
        b[FPREGS + 8 * r..FPREGS + 8 * r + 8].copy_from_slice(&cpu.f[r].to_le_bytes());
    }
    b[FCSR..FCSR + 4].copy_from_slice(&cpu.fcsr.to_le_bytes());
    mem.write(frame, &b)?;
    cpu.x[10] = sig as u64;
    cpu.x[11] = frame;
    cpu.x[12] = frame + UC as u64;
    cpu.x[2] = frame;
    cpu.x[1] = trampoline;
    cpu.pc = action.handler;
    Ok(())
}

/// `rt_sigreturn`: restore the registers saved by [`push_frame`] at the stack pointer, and
/// return the signal mask saved there.
pub fn pop_frame(mem: &impl Memory, cpu: &mut CpuState) -> Result<u64, MemFault> {
    let mut b = vec![0u8; FRAME_SIZE as usize];
    mem.read(cpu.x[2], &mut b)?;
    let word = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    cpu.pc = word(MCONTEXT);
    for r in 1..32 {
        cpu.x[r] = word(MCONTEXT + 8 * r);
    }
    for r in 0..32 {
        cpu.f[r] = word(FPREGS + 8 * r);
    }
    cpu.fcsr = u32::from_le_bytes(b[FCSR..FCSR + 4].try_into().unwrap()) & 0xff;
    cpu.reserved_addr = crate::cpu::NO_RESERVATION;
    Ok(word(UC_SIGMASK))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::FlatMemory;

    #[test]
    fn frames_round_trip() {
        let mem = FlatMemory::new(0x1000, 0x4000);
        let mut cpu = CpuState::new(0x2468);
        for r in 1..32 {
            cpu.x[r] = 0x100 * r as u64;
        }
        cpu.x[2] = 0x5000 - 8; // unaligned: the frame is aligned down
        cpu.f[7] = 0x4000_0000_0000_0000;
        cpu.fcsr = 0x61;
        let saved = cpu.clone();
        let action = Action { handler: 0x8000, flags: SA_SIGINFO, mask: 0 };
        let info = Info::Fault { code: SEGV_MAPERR, addr: 0xdead };
        push_frame(&mem, &mut cpu, SIGSEGV, info, &action, 0b1010, AltStack::default(), 0x9000)
            .unwrap();
        assert_eq!(cpu.pc, 0x8000);
        assert_eq!((cpu.x[10], cpu.x[1]), (SIGSEGV as u64, 0x9000));
        assert_eq!(cpu.x[2] % 16, 0);
        assert!(cpu.x[2] + FRAME_SIZE <= saved.x[2]);
        assert_eq!((cpu.x[11], cpu.x[12]), (cpu.x[2], cpu.x[2] + 128));
        let mut si = [0u8; 24];
        mem.read(cpu.x[11], &mut si).unwrap();
        assert_eq!(i32::from_le_bytes(si[..4].try_into().unwrap()), SIGSEGV);
        assert_eq!(u64::from_le_bytes(si[16..24].try_into().unwrap()), 0xdead);
        // The handler may clobber anything; sigreturn restores it all.
        let frame_sp = cpu.x[2];
        cpu.x = [0x77; 32];
        cpu.x[0] = 0;
        cpu.x[2] = frame_sp;
        assert_eq!(pop_frame(&mem, &mut cpu).unwrap(), 0b1010);
        assert_eq!(cpu.x, saved.x);
        assert_eq!((cpu.pc, cpu.f, cpu.fcsr), (saved.pc, saved.f, saved.fcsr));
    }

    #[test]
    fn alternate_stack_is_used_once() {
        let mem = FlatMemory::new(0x1000, 0x8000);
        let mut cpu = CpuState::new(0);
        cpu.x[2] = 0x8000;
        let alt = AltStack { sp: 0x2000, flags: 0, size: 0x2000 };
        let action = Action { handler: 0x10, flags: SA_ONSTACK, mask: 0 };
        let info = Info::Sent { code: SI_USER, pid: 1, uid: 0 };
        push_frame(&mem, &mut cpu, SIGINT, info, &action, 0, alt, 0).unwrap();
        assert!(alt.contains(cpu.x[2]), "switched to the alternate stack");
        let nested = cpu.x[2];
        push_frame(&mem, &mut cpu, SIGINT, info, &action, 0, alt, 0).unwrap();
        assert!(cpu.x[2] < nested, "a nested signal stays on it, below the first frame");
    }

    #[test]
    fn pending_signals_come_out_lowest_first_unless_blocked() {
        let mut p = Pending::default();
        let info = Info::Sent { code: SI_USER, pid: 1, uid: 0 };
        p.add(SIGTERM, info);
        p.add(SIGINT, info);
        p.add(SIGINT, info); // standard signals do not queue
        assert_eq!(p.take(bit(SIGINT)), Some((SIGTERM, info)));
        assert_eq!(p.take(bit(SIGINT)), None);
        assert_eq!(p.take(0), Some((SIGINT, info)));
        assert_eq!(p.set, 0);
    }
}
