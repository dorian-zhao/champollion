//! Sampling profiler for generated code (`RVRUN_PROF`).
//!
//! Host profilers cannot attribute samples inside JIT code, so we sample ourselves: an
//! `ITIMER_PROF` timer delivers SIGPROF about every millisecond of CPU time, and the handler
//! counts the interrupted pc, per host instruction when it lies in the code buffer. The
//! dispatcher keeps each block's guest instructions so the report can show, for the hottest
//! blocks, the guest code next to the generated code with per-instruction sample counts
//! (`tools/jitprof.py` disassembles the latter).

use std::fmt::Write as _;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64};

use crate::decoder::Inst;

/// Sample counts, shared with the signal handler.
pub struct Samples {
    code_base: usize,
    /// One counter per instruction slot of the code buffer.
    per_insn: Box<[AtomicU32]>,
    /// Samples outside generated code (dispatcher, interpreter, syscalls, Rust).
    outside: AtomicU64,
}

static ACTIVE: AtomicPtr<Samples> = AtomicPtr::new(std::ptr::null_mut());

/// A translated block, as recorded for the report.
pub struct BlockInfo {
    pub guest_pc: u64,
    pub guest: Vec<Inst>,
    pub host_offset: usize,
    pub host_words: Vec<u32>,
}

pub struct Profiler {
    samples: &'static Samples,
    pub blocks: Vec<BlockInfo>,
}

impl Profiler {
    /// Start sampling. Only one profiler may run per process.
    pub fn start(code_base: usize, code_size: usize) -> Profiler {
        let samples: &'static Samples = Box::leak(Box::new(Samples {
            code_base,
            per_insn: (0..code_size / 4).map(|_| AtomicU32::new(0)).collect(),
            outside: AtomicU64::new(0),
        }));
        ACTIVE.store(samples as *const Samples as *mut Samples, Relaxed);
        // SAFETY: installing a SA_SIGINFO handler and arming the profiling timer.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_sample as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGPROF, &sa, std::ptr::null_mut());
            let tick = libc::timeval { tv_sec: 0, tv_usec: 1000 };
            let timer = libc::itimerval { it_interval: tick, it_value: tick };
            libc::setitimer(libc::ITIMER_PROF, &timer, std::ptr::null_mut());
        }
        Profiler { samples, blocks: Vec::new() }
    }

    /// Forget recorded blocks and their samples when the JIT resets (their code is about to
    /// be overwritten). Resets are rare, so the report covers the run after the last one.
    pub fn reset(&mut self) {
        self.blocks.clear();
        for c in self.samples.per_insn.iter() {
            c.store(0, Relaxed);
        }
    }

    /// Stop sampling and render the report: totals, then the `top` hottest blocks.
    pub fn report(&self, top: usize) -> String {
        // SAFETY: disarming the timer.
        unsafe {
            let zero = libc::timeval { tv_sec: 0, tv_usec: 0 };
            let off = libc::itimerval { it_interval: zero, it_value: zero };
            libc::setitimer(libc::ITIMER_PROF, &off, std::ptr::null_mut());
        }
        let counts = |b: &BlockInfo| -> Vec<u32> {
            (0..b.host_words.len())
                .map(|i| self.samples.per_insn[b.host_offset / 4 + i].load(Relaxed))
                .collect()
        };
        let mut hot: Vec<(u64, &BlockInfo)> =
            self.blocks.iter().map(|b| (counts(b).iter().map(|&c| c as u64).sum(), b)).collect();
        let in_jit: u64 = hot.iter().map(|h| h.0).sum();
        let outside = self.samples.outside.load(Relaxed);
        let total = (in_jit + outside).max(1);
        hot.sort_by_key(|h| std::cmp::Reverse(h.0));

        let mut out = String::new();
        let pct = |n: u64| 100.0 * n as f64 / total as f64;
        let _ = writeln!(
            out,
            "samples: {total} ({:.1}% in generated code, {:.1}% outside)",
            pct(in_jit),
            pct(outside)
        );
        for (n, b) in hot.iter().take(top).filter(|h| h.0 > 0) {
            let _ = writeln!(out, "\n== block {:#x}: {n} samples ({:.1}%)", b.guest_pc, pct(*n));
            for inst in &b.guest {
                let _ = writeln!(out, "  guest {inst:?}");
            }
            for (w, c) in b.host_words.iter().zip(counts(b)) {
                let _ = writeln!(out, "  host {w:08x} {c}");
            }
        }
        out
    }
}

extern "C" fn on_sample(_sig: libc::c_int, _info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let samples = ACTIVE.load(Relaxed);
    if samples.is_null() {
        return;
    }
    // SAFETY: `samples` is leaked and never freed; the kernel passes a valid ucontext.
    unsafe {
        let s = &*samples;
        let pc = host_pc(ctx);
        let slot = pc.wrapping_sub(s.code_base) / 4;
        match s.per_insn.get(slot) {
            Some(c) => {
                c.fetch_add(1, Relaxed);
            }
            None => {
                s.outside.fetch_add(1, Relaxed);
            }
        }
    }
}

/// The interrupted pc from a signal context.
///
/// # Safety
/// `ctx` must be the ucontext passed to a SA_SIGINFO handler.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn host_pc(ctx: *mut libc::c_void) -> usize {
    // SAFETY: per the contract.
    unsafe { (*(*(ctx as *mut libc::ucontext_t)).uc_mcontext).__ss.__pc as usize }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
unsafe fn host_pc(_ctx: *mut libc::c_void) -> usize {
    0 // TODO(M6): Linux ucontext layout.
}
