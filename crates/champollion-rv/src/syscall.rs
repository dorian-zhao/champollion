//! Linux syscall layer.
//!
//! The guest issues `ecall` with the syscall number in `a7` and arguments in `a0..a5`
//! (Linux RISC-V ABI); the result goes back in `a0`, negative errno on failure. This module
//! owns that ABI: numbers, Linux struct layouts, and copying arguments in and out of guest
//! memory. Memory-management calls (`brk`, `mmap`, ...) are served by [`Process`] itself,
//! since the guest address space is ours. Everything else goes through [`HostOs`], so the
//! backend can be native Darwin (route B) or passthrough to Linux (route A).

use std::collections::HashSet;
use std::ffi::CStr;
use std::io::IoSlice;

use crate::memory::{MemFault, Memory, Prot};
use crate::process::Process;

/// Linux errno values (asm-generic), which differ from Darwin's past 34 (and at 11).
pub mod errno {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const EIO: i32 = 5;
    pub const EBADF: i32 = 9;
    pub const ENOMEM: i32 = 12;
    pub const EFAULT: i32 = 14;
    pub const EEXIST: i32 = 17;
    pub const ENODEV: i32 = 19;
    pub const EINVAL: i32 = 22;
    pub const ENOTTY: i32 = 25;
    pub const ENOSYS: i32 = 38;
}

/// Linux RISC-V syscall numbers (asm-generic table).
pub mod nr {
    pub const FCNTL: u64 = 25;
    pub const IOCTL: u64 = 29;
    pub const OPENAT: u64 = 56;
    pub const CLOSE: u64 = 57;
    pub const LSEEK: u64 = 62;
    pub const READ: u64 = 63;
    pub const WRITE: u64 = 64;
    pub const READV: u64 = 65;
    pub const WRITEV: u64 = 66;
    pub const PREAD64: u64 = 67;
    pub const NEWFSTATAT: u64 = 79;
    pub const FSTAT: u64 = 80;
    pub const EXIT: u64 = 93;
    pub const EXIT_GROUP: u64 = 94;
    pub const SET_TID_ADDRESS: u64 = 96;
    pub const CLOCK_GETTIME: u64 = 113;
    pub const RT_SIGACTION: u64 = 134;
    pub const RT_SIGPROCMASK: u64 = 135;
    pub const SCHED_GETAFFINITY: u64 = 123;
    pub const UNAME: u64 = 160;
    pub const GETPID: u64 = 172;
    pub const GETPPID: u64 = 173;
    pub const GETUID: u64 = 174;
    pub const GETEUID: u64 = 175;
    pub const GETGID: u64 = 176;
    pub const GETEGID: u64 = 177;
    pub const GETTID: u64 = 178;
    pub const BRK: u64 = 214;
    pub const MUNMAP: u64 = 215;
    pub const MMAP: u64 = 222;
    pub const MPROTECT: u64 = 226;
    pub const GETRANDOM: u64 = 278;
}

/// Result of a syscall: `Ok(value)` or `Err(linux_errno)` (positive).
/// Backends are responsible for translating host errno to Linux errno.
pub type SysResult = Result<u64, i32>;

/// File status, in host-neutral form; written to the guest as Linux `struct stat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    /// File type and permission bits; `S_IF*` values are the same on Linux and Darwin.
    pub mode: u32,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u64,
    pub size: i64,
    pub blksize: i32,
    pub blocks: i64,
    /// `(seconds, nanoseconds)`.
    pub atime: (i64, i64),
    pub mtime: (i64, i64),
    pub ctime: (i64, i64),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity {
    pub pid: u32,
    pub ppid: u32,
    pub uid: u32,
    pub euid: u32,
    pub gid: u32,
    pub egid: u32,
}

/// Host operating-system services the syscall layer needs.
///
/// Arguments carry **Linux** values (flags, `whence`, clock ids, `AT_FDCWD`, ...); each
/// backend translates them, and translates host errno back to Linux errno. Default
/// implementations return `ENOSYS`, so a backend (or a test fake) can be filled in
/// incrementally.
#[allow(unused_variables)]
pub trait HostOs {
    fn read(&mut self, fd: i32, buf: &mut [u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn write(&mut self, fd: i32, buf: &[u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn writev(&mut self, fd: i32, bufs: &[IoSlice<'_>]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn pread(&mut self, fd: i32, buf: &mut [u8], offset: i64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn openat(&mut self, dirfd: i32, path: &CStr, flags: i32, mode: u32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn close(&mut self, fd: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `fcntl` with a Linux command and, for `F_SETFL`, Linux flag values; `F_GETFL`
    /// returns Linux flag values.
    fn fcntl(&mut self, fd: i32, cmd: i32, arg: u64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fstat(&mut self, fd: i32) -> Result<Stat, i32> {
        Err(errno::ENOSYS)
    }
    fn fstatat(&mut self, dirfd: i32, path: &CStr, flags: i32) -> Result<Stat, i32> {
        Err(errno::ENOSYS)
    }
    /// Returns `(seconds, nanoseconds)` for a Linux clock id.
    fn clock_gettime(&mut self, clock_id: i32) -> Result<(i64, i64), i32> {
        Err(errno::ENOSYS)
    }
    fn getrandom(&mut self, buf: &mut [u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Terminal size as `struct winsize` (rows, cols, xpixel, ypixel); `ENOTTY` if `fd`
    /// is not a terminal.
    fn winsize(&mut self, fd: i32) -> Result<[u16; 4], i32> {
        Err(errno::ENOTTY)
    }
    fn identity(&self) -> Identity {
        Identity::default()
    }
    fn hostname(&self) -> Vec<u8> {
        b"localhost".to_vec()
    }
}

/// What the run loop should do after a syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// The guest exited with this status (already masked to 8 bits).
    Exit(i32),
}

/// Syscall tracing and one-time warnings for unsupported calls.
#[derive(Default)]
pub struct SyscallLog {
    pub strace: bool,
    warned: HashSet<u64>,
}

const IOV_MAX: u64 = 1024;
const PATH_MAX: usize = 4096;
const TIOCGWINSZ: u64 = 0x5413;
/// `dirfd` value meaning "relative to the current directory" in Linux syscalls.
pub const AT_FDCWD: i32 = -100;

fn efault(_: MemFault) -> i32 {
    errno::EFAULT
}

/// Copy a syscall result out to guest memory, which must be writable by the guest.
fn put(p: &mut Process, addr: u64, bytes: &[u8]) -> Result<(), i32> {
    p.mem.write(addr, bytes).map_err(efault)
}

/// Handle the `ecall` at `p.cpu.pc`: read the request from the registers, perform it, write
/// the result to `a0`, and advance past the `ecall`.
pub fn dispatch(p: &mut Process, host: &mut dyn HostOs) -> Flow {
    let nr = p.cpu.x[17];
    let a: [u64; 6] = std::array::from_fn(|i| p.cpu.x[10 + i]);

    let result = match nr {
        nr::EXIT | nr::EXIT_GROUP => {
            if p.log.strace {
                eprintln!("[rvrun] exit_group({}) = ?", a[0] as i32);
            }
            return Flow::Exit(a[0] as i32 & 0xff);
        }
        _ => call(p, host, nr, a),
    };

    if p.log.strace {
        let ret = match result {
            Ok(v) => format!("{v:#x}"),
            Err(e) => format!("-{e} ({})", errno_name(e)),
        };
        eprintln!(
            "[rvrun] {}({:#x}, {:#x}, {:#x}, {:#x}) = {ret}",
            name(nr).unwrap_or("?"),
            a[0],
            a[1],
            a[2],
            a[3]
        );
    }
    p.cpu.x[10] = match result {
        Ok(v) => v,
        Err(e) => (-(e as i64)) as u64,
    };
    p.cpu.pc += 4;
    Flow::Continue
}

fn call(p: &mut Process, host: &mut dyn HostOs, nr: u64, a: [u64; 6]) -> SysResult {
    let fd = a[0] as i32;
    match nr {
        nr::READ => {
            let buf = p.mem.slice_mut(a[1], a[2], Prot::WRITE).map_err(efault)?;
            host.read(fd, buf)
        }
        nr::WRITE => {
            let buf = p.mem.slice(a[1], a[2], Prot::READ).map_err(efault)?;
            host.write(fd, buf)
        }
        nr::READV => {
            // Reads into each buffer in turn. Unlike a real readv this is not atomic with
            // respect to other readers, which only matters once there are threads.
            let iov = read_iovec(p, a[1], a[2])?;
            let mut total = 0;
            for (base, len) in iov {
                let buf = p.mem.slice_mut(base, len, Prot::WRITE).map_err(efault)?;
                let n = host.read(fd, buf)?;
                total += n;
                if n < len {
                    break;
                }
            }
            Ok(total)
        }
        nr::WRITEV => {
            let iov = read_iovec(p, a[1], a[2])?;
            let bufs: Vec<IoSlice> = iov
                .iter()
                .map(|&(base, len)| p.mem.slice(base, len, Prot::READ).map(IoSlice::new))
                .collect::<Result<_, _>>()
                .map_err(efault)?;
            host.writev(fd, &bufs)
        }
        nr::PREAD64 => {
            let buf = p.mem.slice_mut(a[1], a[2], Prot::WRITE).map_err(efault)?;
            host.pread(fd, buf, a[3] as i64)
        }
        nr::OPENAT => {
            let path = p.mem.read_cstr(a[1], PATH_MAX).map_err(efault)?;
            host.openat(fd, &path, a[2] as i32, a[3] as u32)
        }
        nr::CLOSE => host.close(fd),
        nr::FCNTL => host.fcntl(fd, a[1] as i32, a[2]),
        nr::LSEEK => host.lseek(fd, a[1] as i64, a[2] as i32),
        nr::FSTAT => {
            let st = host.fstat(fd)?;
            write_stat(p, a[1], &st)
        }
        nr::NEWFSTATAT => {
            let path = p.mem.read_cstr(a[1], PATH_MAX).map_err(efault)?;
            let st = host.fstatat(fd, &path, a[3] as i32)?;
            write_stat(p, a[2], &st)
        }
        nr::IOCTL => match a[1] {
            TIOCGWINSZ => {
                let ws = host.winsize(fd)?;
                let bytes: Vec<u8> = ws.iter().flat_map(|v| v.to_le_bytes()).collect();
                put(p, a[2], &bytes)?;
                Ok(0)
            }
            _ => Err(errno::ENOTTY),
        },
        nr::CLOCK_GETTIME => {
            let (sec, nsec) = host.clock_gettime(a[0] as i32)?;
            let mut ts = sec.to_le_bytes().to_vec();
            ts.extend(nsec.to_le_bytes());
            put(p, a[1], &ts)?;
            Ok(0)
        }
        nr::GETRANDOM => {
            let buf = p.mem.slice_mut(a[0], a[1], Prot::WRITE).map_err(efault)?;
            host.getrandom(buf)
        }
        nr::SCHED_GETAFFINITY => {
            // Until threads arrive in M5, report a single CPU, so programs that size
            // thread pools by CPU count start one worker. Linux returns the number of
            // bytes it wrote: one `unsigned long` of mask.
            if a[1] < 8 || !a[1].is_multiple_of(8) {
                return Err(errno::EINVAL);
            }
            put(p, a[2], &1u64.to_le_bytes())?;
            Ok(8)
        }
        nr::UNAME => {
            // struct utsname: six 65-byte NUL-padded fields.
            let fields: [&[u8]; 6] = [
                b"Linux",
                &host.hostname(),
                b"6.12.0-champollion",
                b"#1 SMP champollion",
                b"riscv64",
                b"(none)",
            ];
            let mut buf = vec![0u8; 6 * 65];
            for (i, f) in fields.iter().enumerate() {
                let n = f.len().min(64);
                buf[i * 65..i * 65 + n].copy_from_slice(&f[..n]);
            }
            put(p, a[0], &buf)?;
            Ok(0)
        }
        nr::SET_TID_ADDRESS | nr::GETTID | nr::GETPID => Ok(host.identity().pid as u64),
        nr::GETPPID => Ok(host.identity().ppid as u64),
        nr::GETUID => Ok(host.identity().uid as u64),
        nr::GETEUID => Ok(host.identity().euid as u64),
        nr::GETGID => Ok(host.identity().gid as u64),
        nr::GETEGID => Ok(host.identity().egid as u64),
        nr::RT_SIGACTION => {
            // Signals arrive in M5. Until then, accept handlers and report SIG_DFL as the
            // old one (riscv64 kernel struct sigaction: handler, flags, mask = 24 bytes).
            if a[2] != 0 {
                put(p, a[2], &[0; 24])?;
            }
            Ok(0)
        }
        nr::RT_SIGPROCMASK => {
            if a[2] != 0 {
                put(p, a[2], &[0; 8])?;
            }
            Ok(0)
        }
        nr::BRK => Ok(p.brk(a[0])),
        nr::MMAP => p.mmap(host, a[0], a[1], a[2] as u32, a[3] as u32, a[4] as i32, a[5]),
        nr::MUNMAP => p.munmap(a[0], a[1]),
        nr::MPROTECT => p.mprotect(a[0], a[1], a[2] as u32),
        _ => {
            if p.log.warned.insert(nr) {
                eprintln!("rvrun: unsupported syscall {nr} ({})", name(nr).unwrap_or("unknown"));
            }
            Err(errno::ENOSYS)
        }
    }
}

/// Read a guest `struct iovec` array as `(base, len)` pairs.
fn read_iovec(p: &Process, addr: u64, count: u64) -> Result<Vec<(u64, u64)>, i32> {
    if count > IOV_MAX {
        return Err(errno::EINVAL);
    }
    let raw = p.mem.slice(addr, count * 16, Prot::READ).map_err(efault)?;
    Ok(raw
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            let (base, len) = c.split_at(8);
            (
                u64::from_le_bytes(base.try_into().unwrap()),
                u64::from_le_bytes(len.try_into().unwrap()),
            )
        })
        .collect())
}

/// Write `st` to the guest as the asm-generic `struct stat` used by riscv64 (128 bytes).
fn write_stat(p: &mut Process, addr: u64, st: &Stat) -> SysResult {
    let mut b = Vec::with_capacity(128);
    b.extend(st.dev.to_le_bytes());
    b.extend(st.ino.to_le_bytes());
    b.extend(st.mode.to_le_bytes());
    b.extend(st.nlink.to_le_bytes());
    b.extend(st.uid.to_le_bytes());
    b.extend(st.gid.to_le_bytes());
    b.extend(st.rdev.to_le_bytes());
    b.extend(0u64.to_le_bytes()); // __pad1
    b.extend(st.size.to_le_bytes());
    b.extend(st.blksize.to_le_bytes());
    b.extend(0i32.to_le_bytes()); // __pad2
    b.extend(st.blocks.to_le_bytes());
    for (sec, nsec) in [st.atime, st.mtime, st.ctime] {
        b.extend(sec.to_le_bytes());
        b.extend(nsec.to_le_bytes());
    }
    b.extend([0u8; 8]); // __unused4, __unused5
    debug_assert_eq!(b.len(), 128);
    put(p, addr, &b)?;
    Ok(0)
}

pub fn name(nr: u64) -> Option<&'static str> {
    Some(match nr {
        nr::FCNTL => "fcntl",
        nr::IOCTL => "ioctl",
        nr::OPENAT => "openat",
        nr::CLOSE => "close",
        nr::LSEEK => "lseek",
        nr::READ => "read",
        nr::WRITE => "write",
        nr::READV => "readv",
        nr::WRITEV => "writev",
        nr::PREAD64 => "pread64",
        nr::NEWFSTATAT => "newfstatat",
        nr::FSTAT => "fstat",
        nr::EXIT => "exit",
        nr::EXIT_GROUP => "exit_group",
        nr::SET_TID_ADDRESS => "set_tid_address",
        nr::CLOCK_GETTIME => "clock_gettime",
        nr::RT_SIGACTION => "rt_sigaction",
        nr::RT_SIGPROCMASK => "rt_sigprocmask",
        nr::SCHED_GETAFFINITY => "sched_getaffinity",
        nr::UNAME => "uname",
        nr::GETPID => "getpid",
        nr::GETPPID => "getppid",
        nr::GETUID => "getuid",
        nr::GETEUID => "geteuid",
        nr::GETGID => "getgid",
        nr::GETEGID => "getegid",
        nr::GETTID => "gettid",
        nr::BRK => "brk",
        nr::MUNMAP => "munmap",
        nr::MMAP => "mmap",
        nr::MPROTECT => "mprotect",
        nr::GETRANDOM => "getrandom",
        // Not implemented yet, but common enough to name in warnings.
        17 => "getcwd",
        23 => "dup",
        24 => "dup3",
        35 => "unlinkat",
        48 => "faccessat",
        49 => "chdir",
        59 => "pipe2",
        61 => "getdents64",
        73 => "ppoll",
        78 => "readlinkat",
        98 => "futex",
        99 => "set_robust_list",
        101 => "nanosleep",
        124 => "sched_yield",
        129 => "kill",
        131 => "tgkill",
        220 => "clone",
        221 => "execve",
        233 => "madvise",
        260 => "wait4",
        261 => "prlimit64",
        293 => "rseq",
        _ => return None,
    })
}

fn errno_name(e: i32) -> &'static str {
    match e {
        errno::EPERM => "EPERM",
        errno::ENOENT => "ENOENT",
        errno::EIO => "EIO",
        errno::EBADF => "EBADF",
        errno::ENOMEM => "ENOMEM",
        errno::EFAULT => "EFAULT",
        errno::EEXIST => "EEXIST",
        errno::ENODEV => "ENODEV",
        errno::EINVAL => "EINVAL",
        errno::ENOTTY => "ENOTTY",
        errno::ENOSYS => "ENOSYS",
        _ => "errno",
    }
}
