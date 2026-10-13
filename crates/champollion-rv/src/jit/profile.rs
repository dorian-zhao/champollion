//! Sampling profiler for generated code (`RVRUN_PROF`).
//!
//! Host profilers cannot attribute samples inside JIT code, so we sample ourselves. A
//! sampler thread wakes about every half millisecond, suspends the guest thread, reads its
//! pc with `thread_get_state`, and resumes it, the way `sample(1)` does. (`ITIMER_PROF` is no
//! good here: macOS delivers SIGPROF only when the thread next enters the kernel, so samples
//! pile up on page-faulting stores and syscall returns.) The pc is counted per host
//! instruction when it lies in the code buffer and kept for symbolization otherwise. The
//! dispatcher keeps each block's guest instructions so the report can show, for the hottest
//! blocks, the guest code next to the generated code with per-instruction sample counts
//! (`tools/jitprof.py` disassembles the latter).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::decoder::Inst;

/// Sample counts, shared with the sampler thread.
struct Samples {
    code_base: usize,
    /// One counter per instruction slot of the code buffer.
    per_insn: Box<[AtomicU32]>,
    /// Samples outside generated code (dispatcher, interpreter, syscalls, Rust).
    outside: AtomicU64,
    /// The host pcs of the first [`OUTSIDE_PCS`] of those, symbolized in the report.
    outside_pcs: Box<[AtomicUsize]>,
    stop: AtomicBool,
}

const OUTSIDE_PCS: usize = 1 << 16;
const INTERVAL: Duration = Duration::from_micros(500);

impl Samples {
    fn record(&self, pc: usize) {
        match self.per_insn.get(pc.wrapping_sub(self.code_base) / 4) {
            Some(c) => {
                c.fetch_add(1, Relaxed);
            }
            None => {
                let n = self.outside.fetch_add(1, Relaxed) as usize;
                if let Some(slot) = self.outside_pcs.get(n) {
                    slot.store(pc, Relaxed);
                }
            }
        }
    }
}

/// A translated block, as recorded for the report.
pub struct BlockInfo {
    pub guest_pc: u64,
    pub guest: Vec<Inst>,
    pub host_offset: usize,
    pub host_words: Vec<u32>,
}

pub struct Profiler {
    samples: Arc<Samples>,
    sampler: Option<JoinHandle<()>>,
    pub blocks: Vec<BlockInfo>,
}

impl Profiler {
    /// Start sampling the calling thread.
    pub fn start(code_base: usize, code_size: usize) -> Profiler {
        let samples = Arc::new(Samples {
            code_base,
            per_insn: (0..code_size / 4).map(|_| AtomicU32::new(0)).collect(),
            outside: AtomicU64::new(0),
            outside_pcs: (0..OUTSIDE_PCS).map(|_| AtomicUsize::new(0)).collect(),
            stop: AtomicBool::new(false),
        });
        let target = mach::current_thread();
        let shared = samples.clone();
        let sampler = std::thread::Builder::new()
            .name("rvrun-profiler".into())
            .spawn(move || {
                while !shared.stop.load(Relaxed) {
                    std::thread::sleep(INTERVAL);
                    if let Some(pc) = mach::sample_pc(target) {
                        shared.record(pc);
                    }
                }
            })
            .ok();
        Profiler { samples, sampler, blocks: Vec::new() }
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
    pub fn report(&mut self, top: usize) -> String {
        self.samples.stop.store(true, Relaxed);
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.join();
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
        let recorded = (outside as usize).min(OUTSIDE_PCS);
        let mut symbols: HashMap<String, u64> = HashMap::new();
        for slot in &self.samples.outside_pcs[..recorded] {
            *symbols.entry(symbol(slot.load(Relaxed))).or_default() += 1;
        }
        let mut symbols: Vec<_> = symbols.into_iter().collect();
        symbols.sort_by_key(|s| std::cmp::Reverse(s.1));
        let _ = writeln!(out, "\n== outside: {outside} samples ({:.1}%)", pct(outside));
        for (name, n) in symbols.iter().take(top) {
            let _ = writeln!(out, "  symbol {n} {name}");
        }
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

/// The name of the function containing host `pc`, or its image and address.
fn symbol(pc: usize) -> String {
    // SAFETY: dladdr only reads the loaded images' symbol tables.
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(pc as *const libc::c_void, &mut info) == 0 {
            return format!("{pc:#x}");
        }
        let name = |p: *const libc::c_char| {
            (!p.is_null()).then(|| std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned())
        };
        match (name(info.dli_sname), name(info.dli_fname)) {
            (Some(s), _) => s,
            (None, Some(f)) => format!("{f}+{:#x}", pc - info.dli_fbase as usize),
            (None, None) => format!("{pc:#x}"),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod mach {
    unsafe extern "C" {
        fn pthread_mach_thread_np(thread: libc::pthread_t) -> u32;
        fn thread_suspend(thread: u32) -> i32;
        fn thread_resume(thread: u32) -> i32;
        fn thread_get_state(thread: u32, flavor: i32, state: *mut u32, count: *mut u32) -> i32;
    }

    /// `ARM_THREAD_STATE64`: x0-x28, fp, lr, sp, pc, cpsr, pad.
    const ARM_THREAD_STATE64: i32 = 6;
    const STATE_WORDS: usize = 68;
    const PC_INDEX: usize = 32;

    pub fn current_thread() -> u32 {
        // SAFETY: plain query of the calling thread's Mach port.
        unsafe { pthread_mach_thread_np(libc::pthread_self()) }
    }

    /// Stop `thread`, read its pc, and let it go again.
    pub fn sample_pc(thread: u32) -> Option<usize> {
        let mut state = [0u64; STATE_WORDS / 2];
        let mut count = STATE_WORDS as u32;
        // SAFETY: `state` holds `count` 32-bit words; the thread is resumed on every path
        // after a successful suspend, and we touch no locks while it is stopped.
        unsafe {
            if thread_suspend(thread) != 0 {
                return None;
            }
            let kr =
                thread_get_state(thread, ARM_THREAD_STATE64, state.as_mut_ptr().cast(), &mut count);
            thread_resume(thread);
            (kr == 0).then_some(state[PC_INDEX] as usize)
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod mach {
    pub fn current_thread() -> u32 {
        0
    }

    pub fn sample_pc(_thread: u32) -> Option<usize> {
        None // TODO(M6): sample on Linux.
    }
}
