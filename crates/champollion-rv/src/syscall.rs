//! Linux syscall layer.
//!
//! The guest issues `ecall` with the syscall number in `a7` and arguments in `a0..a5`
//! (Linux RISC-V ABI); the result goes back in `a0`, negative errno on failure. This module
//! owns that ABI and the Linux-specific struct layouts. Everything host-facing goes through
//! [`HostOs`], so the backend can be native Darwin (route B-lite) or, as an escape hatch,
//! passthrough to a Linux VM (route A).

/// Linux errno values (asm-generic), which differ from Darwin's for anything past ~34.
pub mod errno {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const EBADF: i32 = 9;
    pub const ENOMEM: i32 = 12;
    pub const EFAULT: i32 = 14;
    pub const EINVAL: i32 = 22;
    pub const ENOSYS: i32 = 38;
}

/// Linux RISC-V syscall numbers (asm-generic table).
pub mod nr {
    pub const IOCTL: u64 = 29;
    pub const OPENAT: u64 = 56;
    pub const CLOSE: u64 = 57;
    pub const LSEEK: u64 = 62;
    pub const READ: u64 = 63;
    pub const WRITE: u64 = 64;
    pub const FSTAT: u64 = 80;
    pub const EXIT: u64 = 93;
    pub const EXIT_GROUP: u64 = 94;
    pub const CLOCK_GETTIME: u64 = 113;
    pub const BRK: u64 = 214;
    pub const MUNMAP: u64 = 215;
    pub const MMAP: u64 = 222;
    pub const MPROTECT: u64 = 226;
    pub const GETRANDOM: u64 = 278;
}

/// Result of a host call: `Ok(value)` or `Err(linux_errno)` (positive).
/// Implementations are responsible for translating host errno to Linux errno.
pub type SysResult = Result<u64, i32>;

/// Host operating-system services the syscall layer needs.
///
/// Arguments are already translated out of guest memory into host terms (byte slices,
/// host paths, Linux flag values still un-translated where noted). Methods grow as M2/M3
/// add syscalls; default impls return `ENOSYS` so a backend can be filled in incrementally.
pub trait HostOs {
    fn read(&mut self, fd: i32, buf: &mut [u8]) -> SysResult {
        let _ = (fd, buf);
        Err(errno::ENOSYS)
    }

    fn write(&mut self, fd: i32, buf: &[u8]) -> SysResult {
        let _ = (fd, buf);
        Err(errno::ENOSYS)
    }

    /// `flags` are Linux `O_*` values; the backend maps them to its own.
    fn openat(&mut self, dirfd: i32, path: &std::ffi::CStr, flags: i32, mode: u32) -> SysResult {
        let _ = (dirfd, path, flags, mode);
        Err(errno::ENOSYS)
    }

    fn close(&mut self, fd: i32) -> SysResult {
        let _ = fd;
        Err(errno::ENOSYS)
    }

    fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> SysResult {
        let _ = (fd, offset, whence);
        Err(errno::ENOSYS)
    }

    /// Returns `(seconds, nanoseconds)` for a Linux clock id.
    fn clock_gettime(&mut self, clock_id: i32) -> Result<(i64, i64), i32> {
        let _ = clock_id;
        Err(errno::ENOSYS)
    }

    fn getrandom(&mut self, buf: &mut [u8], flags: u32) -> SysResult {
        let _ = (buf, flags);
        Err(errno::ENOSYS)
    }

    /// Terminates the whole guest process. `exit` and `exit_group` are the same thing
    /// while we are single-threaded.
    fn exit_group(&mut self, code: i32) -> !;
}
