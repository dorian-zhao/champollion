//! Turning host faults in generated code into precise guest faults.
//!
//! Generated loads and stores access guest memory directly, so a guest access to an
//! unmapped page faults on the host (`SIGSEGV`/`SIGBUS`). The handler installed here checks
//! that the faulting host pc is a load or store the translator recorded, then rewrites the
//! interrupted context: the guest pc and `instret` (in its host register) are set to the
//! faulting instruction's, and the host pc is sent to the shared exit stub with a fault
//! exit code. On return from the signal, the stub saves the pinned guest registers as
//! usual, so the guest state is exactly as the interpreter would leave it at the fault.
//!
//! Faults anywhere else (or in code that is not a recorded access) are passed to the
//! handler that was installed before ours, e.g. Rust's stack-overflow reporter.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;

use super::translate::{CTX, EXIT_CODE, INSTRET_REG, MEMBASE, exit};
use crate::cpu::CpuState;

/// What generated code was doing at a recorded host instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    pub guest_pc: u64,
    /// Instructions the block counted in `instret` at entry but did not retire.
    pub instret_undo: u32,
    pub store: bool,
}

/// Recorded accesses of one JIT, by byte offset in its code buffer, plus where to resume.
pub struct Table {
    pub code_base: usize,
    pub code_len: usize,
    pub stub: usize,
    pub sites: HashMap<usize, Site>,
    /// Guest address of the last fault, set by the handler.
    pub fault_addr: Cell<u64>,
}

thread_local! {
    /// The table of the JIT running on this thread, while it runs generated code.
    static ACTIVE: Cell<*const Table> = const { Cell::new(std::ptr::null()) };
}

/// Marks `table` as active on this thread for the duration of a call into generated code.
pub struct Active;

impl Active {
    pub fn enter(table: &Table) -> Active {
        ACTIVE.with(|a| a.set(table));
        Active
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(std::ptr::null()));
    }
}

static PREVIOUS: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

/// Install the handler for SIGSEGV and SIGBUS (once per process).
pub fn install() {
    PREVIOUS.get_or_init(|| {
        let mut previous: [libc::sigaction; 2] = unsafe { std::mem::zeroed() };
        for (i, sig) in [libc::SIGSEGV, libc::SIGBUS].into_iter().enumerate() {
            // SAFETY: installing a SA_SIGINFO handler with a valid function pointer.
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = handler as *const () as usize;
                sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(sig, &sa, &mut previous[i]);
            }
        }
        previous
    });
}

/// Hand a fault we do not own back to whoever handled it before us.
///
/// # Safety
/// Called from the signal handler with its arguments.
unsafe fn chain(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let index = usize::from(sig == libc::SIGBUS);
    let prev = PREVIOUS.get().map(|p| p[index]);
    match prev {
        Some(p) if p.sa_flags & libc::SA_SIGINFO != 0 && p.sa_sigaction > 1 => {
            let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                // SAFETY: the previous handler was installed with SA_SIGINFO.
                unsafe { std::mem::transmute(p.sa_sigaction) };
            f(sig, info, ctx);
        }
        _ => {
            // Restore the default action; returning re-executes the faulting instruction,
            // which then terminates the process as usual.
            // SAFETY: plain sigaction call.
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = libc::SIG_DFL;
                libc::sigaction(sig, &sa, std::ptr::null_mut());
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let table = ACTIVE.with(|a| a.get());
    // SAFETY: the kernel passes a valid ucontext; `table` is live while ACTIVE is set.
    unsafe {
        let uc = ctx as *mut libc::ucontext_t;
        let mc = (*uc).uc_mcontext;
        let pc = (*mc).__ss.__pc as usize;
        let site = table.as_ref().and_then(|t| {
            let off = pc.checked_sub(t.code_base).filter(|&o| o < t.code_len)?;
            t.sites.get(&off).map(|s| (t, *s))
        });
        let Some((t, site)) = site else {
            return chain(sig, info, ctx);
        };
        let regs = &mut (*mc).__ss.__x;
        let cpu = regs[CTX as usize] as *mut CpuState;
        (*cpu).pc = site.guest_pc;
        // x30 is `lr` in the thread state.
        const _: () = assert!(INSTRET_REG == 30);
        (*mc).__ss.__lr -= site.instret_undo as u64;
        let far = (*mc).__es.__far;
        t.fault_addr.set(far.wrapping_sub(regs[MEMBASE as usize]));
        regs[EXIT_CODE as usize] = if site.store { exit::FAULT_STORE } else { exit::FAULT_LOAD };
        (*mc).__ss.__pc = t.stub as u64;
    }
}

/// The same as the macOS handler, for the Linux layout of the interrupted context.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let table = ACTIVE.with(|a| a.get());
    // SAFETY: the kernel passes a valid ucontext; `table` is live while ACTIVE is set.
    unsafe {
        let mc = &mut (*(ctx as *mut libc::ucontext_t)).uc_mcontext;
        let pc = mc.pc as usize;
        let site = table.as_ref().and_then(|t| {
            let off = pc.checked_sub(t.code_base).filter(|&o| o < t.code_len)?;
            t.sites.get(&off).map(|s| (t, *s))
        });
        let Some((t, site)) = site else {
            return chain(sig, info, ctx);
        };
        let cpu = mc.regs[CTX as usize] as *mut CpuState;
        (*cpu).pc = site.guest_pc;
        mc.regs[INSTRET_REG as usize] -= site.instret_undo as u64;
        t.fault_addr.set(mc.fault_address.wrapping_sub(mc.regs[MEMBASE as usize]));
        mc.regs[EXIT_CODE as usize] = if site.store { exit::FAULT_STORE } else { exit::FAULT_LOAD };
        mc.pc = t.stub as u64;
    }
}

#[cfg(not(all(any(target_os = "macos", target_os = "linux"), target_arch = "aarch64")))]
extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    // SAFETY: forwarding the handler's own arguments.
    unsafe { chain(sig, info, ctx) }
}
