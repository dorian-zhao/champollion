//! Executable memory for generated code.
//!
//! One fixed region per JIT, filled bump-style and reset wholesale on flush. Writes collect
//! until [`CodeBuffer::commit`], which must run before generated code does: switching the
//! region between writable and executable and invalidating the instruction cache cost
//! barriers, so they happen once per batch rather than once per write (a block and the
//! branches chained to it). The platform details sit behind `write` and `commit`:
//!
//! - macOS: the region is mapped `MAP_JIT`, and each thread toggles it between writable and
//!   executable with `pthread_jit_write_protect_np` (W^X per thread); `sys_icache_invalidate`
//!   makes new code visible to instruction fetch.
//! - Linux (route A, M6): a plain RWX mapping plus explicit cache maintenance.

use std::io;

/// Byte offset of generated code within the buffer.
pub type CodeOffset = usize;

pub struct CodeBuffer {
    base: *mut u8,
    size: usize,
    used: usize,
    /// Byte ranges written since the last commit.
    dirty: Vec<(usize, usize)>,
    /// Whether this thread has the region writable (macOS).
    writable: bool,
}

/// Instruction cache line size, for merging nearby dirty ranges.
const LINE: usize = 64;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
}

impl CodeBuffer {
    pub fn new(size: usize) -> io::Result<Self> {
        #[cfg(target_os = "macos")]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
        #[cfg(not(target_os = "macos"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        // SAFETY: a fresh anonymous mapping.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                flags,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { base: base.cast(), size, used: 0, dirty: Vec::new(), writable: false })
    }

    pub fn base(&self) -> *const u8 {
        self.base
    }

    pub fn used(&self) -> usize {
        self.used
    }

    /// Whether `addr` points into generated code.
    pub fn contains(&self, addr: usize) -> bool {
        (self.base as usize..self.base as usize + self.used).contains(&addr)
    }

    /// Discard all generated code.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Append `words` and return their offset, or `None` if the buffer is full.
    pub fn append(&mut self, words: &[u32]) -> Option<CodeOffset> {
        let len = words.len() * 4;
        if self.used + len > self.size {
            return None;
        }
        let at = self.used;
        self.write(at, words);
        self.used += len;
        Some(at)
    }

    /// Overwrite already generated code at `at` (e.g. to chain a branch). Takes effect for
    /// execution at the next [`CodeBuffer::commit`].
    pub fn write(&mut self, at: CodeOffset, words: &[u32]) {
        let len = words.len() * 4;
        assert!(at.is_multiple_of(4) && at + len <= self.size);
        // SAFETY: the range is inside our mapping. On macOS, making it writable affects only
        // this thread, which runs no generated code until `commit`.
        unsafe {
            #[cfg(target_os = "macos")]
            if !self.writable {
                libc::pthread_jit_write_protect_np(0);
                self.writable = true;
            }
            std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), self.base.add(at), len);
        }
        self.dirty.push((at, at + len));
    }

    /// Make everything written since the last commit executable and visible to instruction
    /// fetch. Call before running generated code.
    pub fn commit(&mut self) {
        if self.dirty.is_empty() {
            return;
        }
        #[cfg(target_os = "macos")]
        if self.writable {
            // SAFETY: switching this thread's view of the MAP_JIT region back to executable.
            unsafe { libc::pthread_jit_write_protect_np(1) };
            self.writable = false;
        }
        self.dirty.sort_unstable();
        let mut ranges = self.dirty.drain(..);
        let mut current = ranges.next();
        for (start, end) in ranges.chain([(usize::MAX, usize::MAX)]) {
            let Some((lo, hi)) = current else { break };
            if start <= hi + LINE {
                current = Some((lo, hi.max(end)));
                continue;
            }
            // SAFETY: the range lies inside our mapping.
            unsafe {
                let dst = self.base.add(lo);
                #[cfg(target_os = "macos")]
                sys_icache_invalidate(dst.cast(), hi - lo);
                #[cfg(not(target_os = "macos"))]
                flush_icache(dst, hi - lo);
            }
            current = Some((start, end));
        }
    }

    /// Address of code at `at`.
    pub fn ptr(&self, at: CodeOffset) -> *const u8 {
        self.base.wrapping_add(at)
    }
}

/// Make freshly written code visible to instruction fetch (AArch64 cache maintenance).
///
/// # Safety
/// `start..start + len` must be mapped.
#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
unsafe fn flush_icache(start: *mut u8, len: usize) {
    use std::arch::asm;
    let begin = start as usize & !(LINE - 1);
    let end = start as usize + len;
    // SAFETY: cache maintenance by virtual address on mapped memory.
    unsafe {
        for a in (begin..end).step_by(LINE) {
            asm!("dc cvau, {}", in(reg) a);
        }
        asm!("dsb ish");
        for a in (begin..end).step_by(LINE) {
            asm!("ic ivau, {}", in(reg) a);
        }
        asm!("dsb ish", "isb");
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what `new` mapped.
        unsafe { libc::munmap(self.base.cast(), self.size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::arm64::Asm;

    #[test]
    fn generated_code_runs_and_can_be_patched() {
        let mut buf = CodeBuffer::new(1 << 16).unwrap();
        let mut a = Asm::new();
        a.mov_imm(0, 42);
        a.ret();
        let at = buf.append(&a.code).unwrap();
        buf.commit();
        // SAFETY: the buffer holds a valid function with the C ABI: `mov x0, #42; ret`.
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(buf.ptr(at)) };
        assert_eq!(f(), 42);

        let mut a = Asm::new();
        a.mov_imm(0, 7);
        buf.write(at, &a.code);
        buf.commit();
        assert_eq!(f(), 7, "patched code is visible to instruction fetch");
        assert!(buf.contains(buf.ptr(at) as usize));
    }
}
