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
use std::os::unix::ffi::OsStrExt;
use std::sync::Mutex;

use crate::futex;
use crate::memory::{GuestBuf, MemFault, Memory, Prot};
use crate::process::{Process, Thread};
use crate::signal;

/// Linux errno values (asm-generic), which differ from Darwin's past 34 (and at 11).
pub mod errno {
    pub const EPERM: i32 = 1;
    pub const E2BIG: i32 = 7;
    pub const ENOEXEC: i32 = 8;
    pub const EACCES: i32 = 13;
    pub const ENOENT: i32 = 2;
    pub const EIO: i32 = 5;
    pub const EBADF: i32 = 9;
    pub const EAGAIN: i32 = 11;
    pub const ENOMEM: i32 = 12;
    pub const EFAULT: i32 = 14;
    pub const EEXIST: i32 = 17;
    pub const ENODEV: i32 = 19;
    pub const EINVAL: i32 = 22;
    pub const ENOTTY: i32 = 25;
    pub const ESRCH: i32 = 3;
    pub const ERANGE: i32 = 34;
    pub const ENAMETOOLONG: i32 = 36;
    pub const ENOSYS: i32 = 38;
    pub const ENOPROTOOPT: i32 = 92;
    pub const EAFNOSUPPORT: i32 = 97;
    pub const ETIMEDOUT: i32 = 110;
    pub const EINTR: i32 = 4;
}

/// Declares the syscall table once: `nr::NAME` constants plus [`name`] for traces and
/// warnings. Entries without an implementation are listed too, so warnings can name them.
macro_rules! syscalls {
    ($($id:ident = $num:literal $name:literal,)*) => {
        /// Linux RISC-V syscall numbers (asm-generic table).
        pub mod nr {
            $(pub const $id: u64 = $num;)*
        }

        pub fn name(n: u64) -> Option<&'static str> {
            match n {
                $($num => Some($name),)*
                _ => None,
            }
        }
    };
}

syscalls! {
    GETCWD = 17 "getcwd",
    EVENTFD2 = 19 "eventfd2",
    EPOLL_CREATE1 = 20 "epoll_create1",
    EPOLL_CTL = 21 "epoll_ctl",
    EPOLL_PWAIT = 22 "epoll_pwait",
    DUP = 23 "dup",
    DUP3 = 24 "dup3",
    FCNTL = 25 "fcntl",
    IOCTL = 29 "ioctl",
    MKDIRAT = 34 "mkdirat",
    UNLINKAT = 35 "unlinkat",
    RENAMEAT = 38 "renameat",
    FTRUNCATE = 46 "ftruncate",
    FACCESSAT = 48 "faccessat",
    CHDIR = 49 "chdir",
    FCHDIR = 50 "fchdir",
    FCHMOD = 52 "fchmod",
    FCHMODAT = 53 "fchmodat",
    OPENAT = 56 "openat",
    CLOSE = 57 "close",
    PIPE2 = 59 "pipe2",
    GETDENTS64 = 61 "getdents64",
    LSEEK = 62 "lseek",
    READ = 63 "read",
    WRITE = 64 "write",
    READV = 65 "readv",
    WRITEV = 66 "writev",
    PREAD64 = 67 "pread64",
    PWRITE64 = 68 "pwrite64",
    PPOLL = 73 "ppoll",
    READLINKAT = 78 "readlinkat",
    NEWFSTATAT = 79 "newfstatat",
    FSTAT = 80 "fstat",
    FSYNC = 82 "fsync",
    FDATASYNC = 83 "fdatasync",
    UTIMENSAT = 88 "utimensat",
    EXIT = 93 "exit",
    EXIT_GROUP = 94 "exit_group",
    SET_TID_ADDRESS = 96 "set_tid_address",
    FUTEX = 98 "futex",
    SET_ROBUST_LIST = 99 "set_robust_list",
    NANOSLEEP = 101 "nanosleep",
    CLOCK_GETTIME = 113 "clock_gettime",
    CLOCK_NANOSLEEP = 115 "clock_nanosleep",
    SCHED_GETAFFINITY = 123 "sched_getaffinity",
    SCHED_YIELD = 124 "sched_yield",
    KILL = 129 "kill",
    TKILL = 130 "tkill",
    TGKILL = 131 "tgkill",
    SIGALTSTACK = 132 "sigaltstack",
    RT_SIGSUSPEND = 133 "rt_sigsuspend",
    RT_SIGACTION = 134 "rt_sigaction",
    RT_SIGPROCMASK = 135 "rt_sigprocmask",
    RT_SIGPENDING = 136 "rt_sigpending",
    RT_SIGRETURN = 139 "rt_sigreturn",
    SETREGID = 143 "setregid",
    SETGID = 144 "setgid",
    SETREUID = 145 "setreuid",
    SETUID = 146 "setuid",
    SETRESUID = 147 "setresuid",
    GETRESUID = 148 "getresuid",
    SETRESGID = 149 "setresgid",
    GETRESGID = 150 "getresgid",
    GETGROUPS = 158 "getgroups",
    SETGROUPS = 159 "setgroups",
    UNAME = 160 "uname",
    GETRUSAGE = 165 "getrusage",
    UMASK = 166 "umask",
    PRCTL = 167 "prctl",
    GETPID = 172 "getpid",
    GETPPID = 173 "getppid",
    GETUID = 174 "getuid",
    GETEUID = 175 "geteuid",
    GETGID = 176 "getgid",
    GETEGID = 177 "getegid",
    GETTID = 178 "gettid",
    SOCKET = 198 "socket",
    SOCKETPAIR = 199 "socketpair",
    BIND = 200 "bind",
    LISTEN = 201 "listen",
    ACCEPT = 202 "accept",
    CONNECT = 203 "connect",
    GETSOCKNAME = 204 "getsockname",
    GETPEERNAME = 205 "getpeername",
    SENDTO = 206 "sendto",
    RECVFROM = 207 "recvfrom",
    SETSOCKOPT = 208 "setsockopt",
    GETSOCKOPT = 209 "getsockopt",
    SHUTDOWN = 210 "shutdown",
    SENDMSG = 211 "sendmsg",
    RECVMSG = 212 "recvmsg",
    BRK = 214 "brk",
    MUNMAP = 215 "munmap",
    MREMAP = 216 "mremap",
    CLONE = 220 "clone",
    EXECVE = 221 "execve",
    MMAP = 222 "mmap",
    MPROTECT = 226 "mprotect",
    MADVISE = 233 "madvise",
    ACCEPT4 = 242 "accept4",
    WAIT4 = 260 "wait4",
    RISCV_HWPROBE = 258 "riscv_hwprobe",
    RISCV_FLUSH_ICACHE = 259 "riscv_flush_icache",
    PRLIMIT64 = 261 "prlimit64",
    RENAMEAT2 = 276 "renameat2",
    GETRANDOM = 278 "getrandom",
    RSEQ = 293 "rseq",
    FACCESSAT2 = 439 "faccessat2",
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

/// `struct pollfd`, identical on Linux and Darwin; event bits are Linux values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PollFd {
    pub fd: i32,
    pub events: i16,
    pub revents: i16,
}

/// A POSIX record lock with Linux `l_type` values (`F_RDLCK` 0, `F_WRLCK` 1, `F_UNLCK` 2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flock {
    pub typ: i16,
    pub whence: i16,
    pub start: i64,
    pub len: i64,
    pub pid: i32,
}

/// The parts of `struct rusage` we report; times are `(seconds, microseconds)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rusage {
    pub utime: (i64, i64),
    pub stime: (i64, i64),
    pub maxrss_kib: i64,
    pub minflt: i64,
    pub majflt: i64,
    pub nvcsw: i64,
    pub nivcsw: i64,
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
///
/// Methods take `&self` and the trait requires `Sync`: guest threads call into one shared
/// backend concurrently, and a call that blocks (a `read` on a pipe) must only block its
/// own thread. Buffers are [`GuestBuf`]s, validated but not borrowed, so no lock is held
/// while the host blocks.
#[allow(unused_variables)]
pub trait HostOs: Send + Sync {
    fn read(&self, fd: i32, buf: GuestBuf) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn epoll_create1(&self, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `event` is `(events, data)`; ignored for `EPOLL_CTL_DEL`.
    fn epoll_ctl(&self, epfd: i32, op: i32, fd: i32, event: (u32, u64)) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Up to `max` ready `(events, data)` pairs, waiting up to `timeout_ms` (negative:
    /// forever).
    fn epoll_wait(&self, epfd: i32, max: usize, timeout_ms: i32) -> Result<Vec<(u32, u64)>, i32> {
        Err(errno::ENOSYS)
    }
    fn eventfd(&self, initval: u32, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `linux_dirent64` records for the next entries of directory `fd`, at most `max` bytes.
    fn getdents64(&self, fd: i32, max: usize) -> Result<Vec<u8>, i32> {
        Err(errno::ENOSYS)
    }
    /// `wait4`: the child's pid and its wait status, Linux-encoded.
    fn wait4(&self, pid: i32, options: i32) -> Result<(u64, i32), i32> {
        Err(errno::ENOSYS)
    }

    // Sockets. Addresses are Linux `sockaddr` bytes, flags Linux values.
    fn socket(&self, domain: i32, ty: i32, protocol: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn socketpair(&self, domain: i32, ty: i32, protocol: i32) -> Result<[i32; 2], i32> {
        Err(errno::ENOSYS)
    }
    fn bind(&self, fd: i32, addr: &[u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn connect(&self, fd: i32, addr: &[u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn listen(&self, fd: i32, backlog: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// The new socket and its peer's address.
    fn accept4(&self, fd: i32, flags: i32) -> Result<(u64, Vec<u8>), i32> {
        Err(errno::ENOSYS)
    }
    /// `getsockname`, or `getpeername` if `peer`.
    fn sockname(&self, fd: i32, peer: bool) -> Result<Vec<u8>, i32> {
        Err(errno::ENOSYS)
    }
    fn setsockopt(&self, fd: i32, level: i32, name: i32, value: &[u8]) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// The option's value, at most `len` bytes.
    fn getsockopt(&self, fd: i32, level: i32, name: i32, len: usize) -> Result<Vec<u8>, i32> {
        Err(errno::ENOSYS)
    }
    fn sendmsg(&self, fd: i32, to: Option<&[u8]>, bufs: &[GuestBuf], flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Bytes received, the sender's address (if any), and the message flags.
    fn recvmsg(&self, fd: i32, bufs: &[GuestBuf], flags: i32) -> Result<(u64, Vec<u8>, i32), i32> {
        Err(errno::ENOSYS)
    }
    fn shutdown(&self, fd: i32, how: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn write(&self, fd: i32, buf: GuestBuf) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn readv(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn writev(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn pread(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn openat(&self, dirfd: i32, path: &CStr, flags: i32, mode: u32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn close(&self, fd: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `fcntl` with a Linux command and, for `F_SETFL`, Linux flag values; `F_GETFL`
    /// returns Linux flag values.
    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn lseek(&self, fd: i32, offset: i64, whence: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fstat(&self, fd: i32) -> Result<Stat, i32> {
        Err(errno::ENOSYS)
    }
    fn fstatat(&self, dirfd: i32, path: &CStr, flags: i32) -> Result<Stat, i32> {
        Err(errno::ENOSYS)
    }
    /// Returns `(seconds, nanoseconds)` for a Linux clock id.
    fn clock_gettime(&self, clock_id: i32) -> Result<(i64, i64), i32> {
        Err(errno::ENOSYS)
    }
    fn getrandom(&self, buf: GuestBuf) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Terminal size as `struct winsize` (rows, cols, xpixel, ypixel); `ENOTTY` if `fd`
    /// is not a terminal.
    fn winsize(&self, fd: i32) -> Result<[u16; 4], i32> {
        Err(errno::ENOTTY)
    }
    fn pwrite(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn unlinkat(&self, dirfd: i32, path: &CStr, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn renameat2(
        &self,
        olddirfd: i32,
        old: &CStr,
        newdirfd: i32,
        new: &CStr,
        flags: u32,
    ) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn mkdirat(&self, dirfd: i32, path: &CStr, mode: u32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn faccessat(&self, dirfd: i32, path: &CStr, mode: i32, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn readlinkat(&self, dirfd: i32, path: &CStr, buf: GuestBuf) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn getcwd(&self) -> Result<Vec<u8>, i32> {
        Err(errno::ENOSYS)
    }
    fn chdir(&self, path: &CStr) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fchdir(&self, fd: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn dup(&self, fd: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `dup3`; `flags` may contain Linux `O_CLOEXEC`.
    fn dup3(&self, old: i32, new: i32, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// `pipe2`; `flags` are Linux `O_CLOEXEC`/`O_NONBLOCK`.
    fn pipe2(&self, flags: i32) -> Result<[i32; 2], i32> {
        Err(errno::ENOSYS)
    }
    fn ftruncate(&self, fd: i32, len: i64) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fsync(&self, fd: i32, data_only: bool) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fchmod(&self, fd: i32, mode: u32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn fchmodat(&self, dirfd: i32, path: &CStr, mode: u32, flags: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn umask(&self, mask: u32) -> u32 {
        0o022
    }
    /// Record locks: `F_GETLK`/`F_SETLK`/`F_SETLKW` (Linux command numbers 5/6/7) with the
    /// lock in host-neutral form. `F_GETLK` updates `lock` in place.
    fn lock(&self, fd: i32, cmd: i32, lock: &mut Flock) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Sleep for `(seconds, nanoseconds)` on the host.
    fn nanosleep(&self, duration: (i64, i64)) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Resource usage of the process (`who` = 0) or its children (-1).
    fn getrusage(&self, who: i32) -> Result<Rusage, i32> {
        Err(errno::ENOSYS)
    }
    /// `(soft, hard)` limit for a Linux `RLIMIT_*` resource; `u64::MAX` is unlimited.
    fn getrlimit(&self, resource: u32) -> Result<(u64, u64), i32> {
        Err(errno::ENOSYS)
    }
    /// Wait for events on `fds` (`(fd, events, revents)` with Linux `POLL*` bits) for at
    /// most `timeout` (`None` = forever). Returns the number of ready descriptors.
    fn poll(&self, fds: &mut [PollFd], timeout: Option<(i64, i64)>) -> SysResult {
        Err(errno::ENOSYS)
    }
    /// Send a Linux signal number to another host process.
    fn kill(&self, pid: i32, sig: i32) -> SysResult {
        Err(errno::ENOSYS)
    }
    fn identity(&self) -> Identity {
        Identity::default()
    }
    fn hostname(&self) -> Vec<u8> {
        b"localhost".to_vec()
    }
}

/// What the run loop should do after a syscall. Exit statuses are already masked to 8 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// `exit`: end the calling thread only.
    ExitThread(i32),
    /// `exit_group`: end the whole process.
    ExitGroup(i32),
    /// The thread must die by this signal now (a corrupt signal frame on `rt_sigreturn`).
    Killed(i32),
}

/// Syscall tracing and one-time warnings for unsupported calls. Shared by all threads.
#[derive(Default)]
pub struct SyscallLog {
    /// Trace every syscall to stderr (set before the guest starts).
    pub strace: std::sync::atomic::AtomicBool,
    warned: Mutex<HashSet<u64>>,
}

const IOV_MAX: u64 = 1024;
const PATH_MAX: usize = 4096;
const TIOCGWINSZ: u64 = 0x5413;
/// `dirfd` value meaning "relative to the current directory" in Linux syscalls.
pub const AT_FDCWD: i32 = -100;

fn efault(_: MemFault) -> i32 {
    errno::EFAULT
}

const F_GETLK: i32 = 5;
const F_SETLK: i32 = 6;
const F_SETLKW: i32 = 7;

/// Whether sending `sig` to ourselves must end the process now: its default action
/// terminates, and handlers are not supported yet. Returns the signal if so.
fn signal_number(sig: u64) -> Result<i32, i32> {
    let sig = sig as i32;
    if (0..=signal::NSIG).contains(&sig) { Ok(sig) } else { Err(errno::EINVAL) }
}

/// `riscv_hwprobe(pairs, count, cpusetsize, cpus, flags)`: describe the hart we emulate.
/// Each pair is `(key: i64, value: u64)`; unknown keys get key -1.
fn hwprobe(p: &Process, a: [u64; 6]) -> SysResult {
    const BASE_BEHAVIOR_IMA: u64 = 1;
    const IMA_FD: u64 = 1 << 0;
    const IMA_C: u64 = 1 << 1;
    if a[4] != 0 {
        return Err(errno::EINVAL);
    }
    for i in 0..a[1] {
        let at = a[0] + 16 * i;
        let mut b = [0u8; 8];
        p.mem.read(at, &mut b).map_err(efault)?;
        let value = match i64::from_le_bytes(b) {
            0..=2 => Some(0), // mvendorid, marchid, mimpid
            3 => Some(BASE_BEHAVIOR_IMA),
            4 => Some(IMA_FD | IMA_C),
            _ => None,
        };
        match value {
            Some(v) => put(p, at + 8, &v.to_le_bytes())?,
            None => put(p, at, &(-1i64).to_le_bytes())?,
        }
    }
    Ok(0)
}

/// `kill`, `tkill`, `tgkill`. Signals to this process or its threads are queued and
/// delivered by the recipient's run loop (right after this syscall, for the caller).
fn kill(p: &Process, nr: u64, a: [u64; 6]) -> SysResult {
    let me = p.host.identity();
    let pid = me.pid as i32;
    let sent = |code| signal::Info::Sent { code, pid: me.pid, uid: me.uid };
    match nr {
        nr::KILL => {
            let (target, sig) = (a[0] as i32, signal_number(a[1])?);
            if target != pid && target != 0 && target != -pid {
                return p.host.kill(target, sig);
            }
            if sig != 0 {
                p.send_to_process(sig, sent(signal::SI_USER));
            }
            Ok(0)
        }
        _ => {
            let (tgid, tid, sig) = match nr {
                nr::TKILL => (pid, a[0] as i32, signal_number(a[1])?),
                _ => (a[0] as i32, a[1] as i32, signal_number(a[2])?),
            };
            if tgid != pid || tid <= 0 {
                return Err(errno::ESRCH);
            }
            let exists = if sig == 0 {
                p.thread_exists(tid as u32) // signal 0 only checks the target
            } else {
                p.send_to_thread(tid as u32, sig, sent(signal::SI_TKILL))
            };
            if exists { Ok(0) } else { Err(errno::ESRCH) }
        }
    }
}

fn read_timespec(p: &Process, addr: u64) -> Result<(i64, i64), i32> {
    let mut b = [0u8; 16];
    p.mem.read(addr, &mut b).map_err(efault)?;
    let sec = i64::from_le_bytes(b[..8].try_into().unwrap());
    let nsec = i64::from_le_bytes(b[8..].try_into().unwrap());
    if sec < 0 || !(0..1_000_000_000).contains(&nsec) {
        return Err(errno::EINVAL);
    }
    Ok((sec, nsec))
}

/// asm-generic `struct flock`: l_type, l_whence (i16), l_start, l_len (i64), l_pid (i32).
fn read_flock(p: &Process, addr: u64) -> Result<Flock, i32> {
    let mut b = [0u8; 32];
    p.mem.read(addr, &mut b).map_err(efault)?;
    Ok(Flock {
        typ: i16::from_le_bytes([b[0], b[1]]),
        whence: i16::from_le_bytes([b[2], b[3]]),
        start: i64::from_le_bytes(b[8..16].try_into().unwrap()),
        len: i64::from_le_bytes(b[16..24].try_into().unwrap()),
        pid: i32::from_le_bytes(b[24..28].try_into().unwrap()),
    })
}

fn write_flock(p: &Process, addr: u64, l: &Flock) -> Result<(), i32> {
    let mut b = [0u8; 32];
    b[0..2].copy_from_slice(&l.typ.to_le_bytes());
    b[2..4].copy_from_slice(&l.whence.to_le_bytes());
    b[8..16].copy_from_slice(&l.start.to_le_bytes());
    b[16..24].copy_from_slice(&l.len.to_le_bytes());
    b[24..28].copy_from_slice(&l.pid.to_le_bytes());
    put(p, addr, &b)
}

/// Linux `struct rusage` (144 bytes): two timevals, then 14 longs starting with ru_maxrss
/// (in KiB).
fn linux_rusage(ru: &Rusage) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    for (sec, usec) in [ru.utime, ru.stime] {
        b.extend(sec.to_le_bytes());
        b.extend(usec.to_le_bytes());
    }
    let longs =
        [ru.maxrss_kib, 0, 0, 0, ru.minflt, ru.majflt, 0, 0, 0, 0, 0, 0, ru.nvcsw, ru.nivcsw];
    for v in longs {
        b.extend(v.to_le_bytes());
    }
    b
}

/// `clone(flags, newsp, parent_tid, tls, child_tid)` (the riscv64 argument order). Only
/// threads are supported: a new thread in this process, not a new process.
fn clone(p: &Process, t: &mut Thread, a: [u64; 6]) -> SysResult {
    const CLONE_VM: u64 = 0x100;
    const CLONE_SIGHAND: u64 = 0x800;
    const CLONE_THREAD: u64 = 0x1_0000;
    const CLONE_SETTLS: u64 = 0x8_0000;
    const CLONE_PARENT_SETTID: u64 = 0x10_0000;
    const CLONE_CHILD_CLEARTID: u64 = 0x20_0000;
    const CLONE_CHILD_SETTID: u64 = 0x100_0000;
    let [flags, newsp, parent_tid, tls, child_tid, _] = a;
    let thread = CLONE_VM | CLONE_SIGHAND | CLONE_THREAD;
    if flags & CLONE_THREAD == 0 {
        // A new process: fork, or vfork (CLONE_VM | CLONE_VFORK), which may copy memory
        // instead of sharing it, since its child only runs until execve or _exit.
        let pid = p.fork(t)?;
        if pid == 0 {
            if flags & CLONE_VM != 0 && newsp != 0 {
                t.cpu.x[2] = newsp;
            }
            if flags & CLONE_SETTLS != 0 {
                t.cpu.x[4] = tls;
            }
            if flags & CLONE_CHILD_SETTID != 0 {
                put(p, child_tid, &t.tid.to_le_bytes())?;
            }
            if flags & CLONE_CHILD_CLEARTID != 0 {
                t.clear_child_tid = child_tid;
            }
        } else if flags & CLONE_PARENT_SETTID != 0 {
            put(p, parent_tid, &pid.to_le_bytes())?;
        }
        return Ok(pid as u64);
    }
    if flags & thread != thread {
        return Err(errno::EINVAL);
    }
    let tid = p.new_tid();
    let mut cpu = t.cpu.clone();
    // The child returns 0 from this ecall, on its own stack and TLS if given.
    cpu.pc += 4;
    cpu.instret += 1;
    cpu.x[10] = 0;
    cpu.reserved_addr = crate::cpu::NO_RESERVATION;
    if newsp != 0 {
        cpu.x[2] = newsp;
    }
    if flags & CLONE_SETTLS != 0 {
        cpu.x[4] = tls;
    }
    let set = |flag, addr| -> Result<(), i32> {
        if flags & flag != 0 { put(p, addr, &tid.to_le_bytes()) } else { Ok(()) }
    };
    let clear = if flags & CLONE_CHILD_CLEARTID != 0 { child_tid } else { 0 };
    let started = set(CLONE_PARENT_SETTID, parent_tid)
        .and_then(|()| set(CLONE_CHILD_SETTID, child_tid))
        .and_then(|()| {
            p.start_thread(cpu, tid, t.name, t.signals.mask(), clear).map_err(|_| errno::EAGAIN)
        });
    if let Err(e) = started {
        p.forget_tid(tid);
        return Err(e);
    }
    Ok(tid as u64)
}

/// A NULL-terminated array of C strings in guest memory (`argv`, `envp`).
fn read_cstr_array(p: &Process, mut addr: u64) -> Result<Vec<Vec<u8>>, i32> {
    const MAX: usize = 1 << 16;
    let mut out = Vec::new();
    if addr == 0 {
        return Ok(out);
    }
    loop {
        let mut b = [0u8; 8];
        p.mem.read(addr, &mut b).map_err(efault)?;
        let s = u64::from_le_bytes(b);
        if s == 0 {
            return Ok(out);
        }
        if out.len() == MAX {
            return Err(errno::E2BIG);
        }
        out.push(p.mem.read_cstr(s, 1 << 17).map_err(efault)?.into_bytes());
        addr += 8;
    }
}

/// `execve(path, argv, envp)`: the embedder starts the new program (see
/// [`Process::exec`]); this only returns on failure.
fn execve(p: &Process, t: &Thread, a: [u64; 6]) -> SysResult {
    let path = p.mem.read_cstr(a[0], PATH_MAX).map_err(efault)?;
    let argv = read_cstr_array(p, a[1])?;
    let envp = read_cstr_array(p, a[2])?;
    let host =
        std::path::PathBuf::from(std::ffi::OsStr::from_bytes(p.host_path(&path, true).to_bytes()));
    // Refuse what cannot run here before giving up this process: only riscv64 ELF files
    // and scripts (the embedder handles `#!`).
    let mut head = [0u8; 20];
    let n = std::fs::File::open(&host)
        .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => errno::ENOENT,
            std::io::ErrorKind::PermissionDenied => errno::EACCES,
            _ => errno::EIO,
        })?;
    const EM_RISCV: u16 = 243;
    let elf =
        n >= 20 && head[..4] == *b"\x7fELF" && u16::from_le_bytes([head[18], head[19]]) == EM_RISCV;
    if !elf && !head.starts_with(b"#!") {
        return Err(errno::ENOEXEC);
    }
    let ignored = p.ignored_signals();
    Err(p.exec(&crate::process::Exec { path: host, argv, envp, ignored, mask: t.signals.mask() }))
}

/// `futex(uaddr, op, val, timeout or val2, uaddr2, val3)`: the wait, wake, and requeue
/// operations, private or not (guest processes never share memory).
fn futex(p: &Process, t: &Thread, a: [u64; 6]) -> SysResult {
    const PRIVATE: u64 = 128;
    const CLOCK_REALTIME: u64 = 256;
    let [addr, op, val, timeout, addr2, val3] = a;
    let count = (val as i32).max(0) as u32;
    // A timespec at `timeout`: relative for WAIT, absolute for WAIT_BITSET.
    let deadline = |absolute: bool| -> Result<Option<std::time::Instant>, i32> {
        if timeout == 0 {
            return Ok(None);
        }
        let (sec, nsec) = read_timespec(p, timeout)?;
        let mut d = std::time::Duration::new(sec as u64, nsec as u32);
        let now = std::time::Instant::now();
        if absolute {
            // The guest's clocks, as clock_gettime reports them.
            let clock = if op & CLOCK_REALTIME != 0 { 0 } else { 1 };
            let (s, ns) = p.host.clock_gettime(clock)?;
            let clock_now = std::time::Duration::new(s as u64, ns as u32);
            d = d.saturating_sub(clock_now);
        }
        Ok(Some(now + d))
    };
    match op & !(PRIVATE | CLOCK_REALTIME) {
        0 => {
            let who = p.blocked(t);
            p.futexes.wait(&p.mem, addr, val as u32, deadline(false)?, futex::MATCH_ANY, Some(&who))
        }
        1 => p.futexes.wake(addr, count, futex::MATCH_ANY),
        3 => p.futexes.requeue(&p.mem, addr, count, addr2, timeout as u32, None),
        4 => p.futexes.requeue(&p.mem, addr, count, addr2, timeout as u32, Some(val3 as u32)),
        9 => {
            let who = p.blocked(t);
            p.futexes.wait(&p.mem, addr, val as u32, deadline(true)?, val3 as u32, Some(&who))
        }
        10 => p.futexes.wake(addr, count, val3 as u32),
        _ => Err(errno::ENOSYS),
    }
}

/// Copy a syscall result out to guest memory, which must be writable by the guest.
fn put(p: &Process, addr: u64, bytes: &[u8]) -> Result<(), i32> {
    p.mem.write(addr, bytes).map_err(efault)
}

/// Handle the `ecall` at `t.cpu.pc`: read the request from the thread's registers, perform
/// it, write the result to `a0`, and advance past the `ecall`.
pub fn dispatch(p: &Process, t: &mut Thread) -> Flow {
    let nr = t.cpu.x[17];
    let a: [u64; 6] = std::array::from_fn(|i| t.cpu.x[10 + i]);

    let result = match nr {
        nr::EXIT | nr::EXIT_GROUP => {
            if p.log.strace.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[rvrun {}] {}({}) = ?", t.tid, name(nr).unwrap(), a[0] as i32);
            }
            let status = a[0] as i32 & 0xff;
            return if nr == nr::EXIT { Flow::ExitThread(status) } else { Flow::ExitGroup(status) };
        }
        nr::RT_SIGRETURN => {
            let Ok(mask) = signal::pop_frame(&p.mem, &mut t.cpu) else {
                return Flow::Killed(signal::SIGSEGV); // a corrupt frame, as on Linux
            };
            t.signals.set_mask(mask);
            if p.log.strace.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[rvrun {}] rt_sigreturn() = {:#x}", t.tid, t.cpu.x[10]);
            }
            // The restored registers are the result: no a0, no pc advance.
            return Flow::Continue;
        }
        _ => call(p, t, nr, a),
    };
    // A blocking call interrupted by a signal: restart it after the handler if every
    // handler waiting asked for SA_RESTART (and the call is one Linux restarts), or if no
    // signal is waiting after all; otherwise return EINTR. Restarting leaves pc at the
    // ecall and the arguments in place, so delivery runs the handler first.
    if result == Err(errno::EINTR) && nr != nr::RT_SIGSUSPEND {
        let restartable = !matches!(
            nr,
            nr::EPOLL_PWAIT | nr::PPOLL | nr::NANOSLEEP | nr::CLOCK_NANOSLEEP | nr::FUTEX
        );
        if p.signal_waiting(t).is_none_or(|all_restart| all_restart && restartable) {
            return Flow::Continue;
        }
    }
    // Writing to a pipe or socket with no reader raises SIGPIPE (whose default action
    // kills), unless a send asked for MSG_NOSIGNAL. The host only returns EPIPE.
    const EPIPE: i32 = 32;
    const MSG_NOSIGNAL: u64 = 0x4000;
    if result == Err(EPIPE) {
        let quiet = match nr {
            nr::SENDTO => a[3] & MSG_NOSIGNAL != 0,
            nr::SENDMSG => a[2] & MSG_NOSIGNAL != 0,
            _ => false,
        };
        if !quiet {
            let id = p.host.identity();
            let info = signal::Info::Sent { code: signal::SI_USER, pid: id.pid, uid: id.uid };
            p.send_to_thread(t.tid, signal::SIGPIPE, info);
        }
    }

    if p.log.strace.load(std::sync::atomic::Ordering::Relaxed) {
        let ret = match result {
            Ok(v) => format!("{v:#x}"),
            Err(e) => format!("-{e} ({})", errno_name(e)),
        };
        eprintln!(
            "[rvrun {}] {}({:#x}, {:#x}, {:#x}, {:#x}) = {ret}",
            t.tid,
            name(nr).unwrap_or("?"),
            a[0],
            a[1],
            a[2],
            a[3]
        );
    }
    t.cpu.x[10] = match result {
        Ok(v) => v,
        Err(e) => (-(e as i64)) as u64,
    };
    t.cpu.pc += 4;
    t.cpu.instret += 1;
    Flow::Continue
}

fn call(p: &Process, t: &mut Thread, nr: u64, a: [u64; 6]) -> SysResult {
    let host = &*p.host;
    let fd = a[0] as i32;
    let buf = |addr, len, need| p.mem.host_buf(addr, len, need).map_err(efault);
    let cstr = |addr| p.mem.read_cstr(addr, PATH_MAX).map_err(efault);
    match nr {
        nr::READ => host.read(fd, buf(a[1], a[2], Prot::WRITE)?),
        nr::WRITE => host.write(fd, buf(a[1], a[2], Prot::READ)?),
        nr::READV | nr::WRITEV => {
            let need = if nr == nr::READV { Prot::WRITE } else { Prot::READ };
            let bufs = read_iovec(p, a[1], a[2])?
                .into_iter()
                .map(|(base, len)| buf(base, len, need))
                .collect::<Result<Vec<_>, _>>()?;
            if nr == nr::READV { host.readv(fd, &bufs) } else { host.writev(fd, &bufs) }
        }
        nr::PREAD64 => host.pread(fd, buf(a[1], a[2], Prot::WRITE)?, a[3] as i64),
        nr::OPENAT => {
            let path = p.mem.read_cstr(a[1], PATH_MAX).map_err(efault)?;
            host.openat(fd, &p.host_path(&path, true), a[2] as i32, a[3] as u32)
        }
        nr::CLOSE => host.close(fd),
        nr::FCNTL => match a[1] as i32 {
            cmd @ (F_GETLK | F_SETLK | F_SETLKW) => {
                let mut lock = read_flock(p, a[2])?;
                let r = host.lock(fd, cmd, &mut lock)?;
                if cmd == F_GETLK {
                    write_flock(p, a[2], &lock)?;
                }
                Ok(r)
            }
            cmd => host.fcntl(fd, cmd, a[2]),
        },
        nr::LSEEK => host.lseek(fd, a[1] as i64, a[2] as i32),
        nr::FSTAT => {
            let st = host.fstat(fd)?;
            write_stat(p, a[1], &st)
        }
        nr::NEWFSTATAT => {
            let path = p.mem.read_cstr(a[1], PATH_MAX).map_err(efault)?;
            const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
            let follow = a[3] & AT_SYMLINK_NOFOLLOW == 0;
            let st = host.fstatat(fd, &p.host_path(&path, follow), a[3] as i32)?;
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
        nr::GETRANDOM => host.getrandom(buf(a[0], a[1], Prot::WRITE)?),
        nr::SCHED_GETAFFINITY => {
            // Every host CPU, up to 64. Linux returns the number of bytes it wrote: one
            // `unsigned long` of mask.
            if a[1] < 8 || !a[1].is_multiple_of(8) {
                return Err(errno::EINVAL);
            }
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get()).min(64);
            let mask = if cpus == 64 { u64::MAX } else { (1u64 << cpus) - 1 };
            put(p, a[2], &mask.to_le_bytes())?;
            Ok(8)
        }
        nr::CLONE => clone(p, t, a),
        nr::EXECVE => execve(p, t, a),
        nr::WAIT4 => {
            let (pid, status) = host.wait4(a[0] as i32, a[2] as i32)?;
            if a[1] != 0 {
                put(p, a[1], &status.to_le_bytes())?;
            }
            if a[3] != 0 {
                put(p, a[3], &[0; 144])?; // struct rusage: not tracked per child
            }
            Ok(pid)
        }
        nr::FUTEX => futex(p, t, a),
        // Robust futex lists only matter when a thread dies holding a lock, which ends the
        // whole process here anyway.
        nr::SET_ROBUST_LIST => Ok(0),
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
        nr::SET_TID_ADDRESS => {
            t.clear_child_tid = a[0];
            Ok(t.tid as u64)
        }
        nr::GETTID => Ok(t.tid as u64),
        nr::GETPID => Ok(host.identity().pid as u64),
        nr::GETPPID => Ok(host.identity().ppid as u64),
        nr::GETUID => Ok(host.identity().uid as u64),
        nr::GETEUID => Ok(host.identity().euid as u64),
        nr::GETGID => Ok(host.identity().gid as u64),
        nr::GETEGID => Ok(host.identity().egid as u64),
        // The guest cannot change its credentials: setting them to what they are succeeds
        // (programs do that to drop privileges they never had), anything else is EPERM.
        nr::SETUID | nr::SETGID | nr::SETREUID | nr::SETREGID | nr::SETRESUID | nr::SETRESGID => {
            let id = host.identity();
            let (real, effective) = match nr {
                nr::SETUID | nr::SETREUID | nr::SETRESUID => (id.uid, id.euid),
                _ => (id.gid, id.egid),
            };
            let n = match nr {
                nr::SETUID | nr::SETGID => 1,
                nr::SETREUID | nr::SETREGID => 2,
                _ => 3,
            };
            let ok = a[..n]
                .iter()
                .all(|&v| v as u32 == u32::MAX || v as u32 == real || v as u32 == effective);
            if ok { Ok(0) } else { Err(errno::EPERM) }
        }
        nr::GETRESUID | nr::GETRESGID => {
            let id = host.identity();
            let (real, effective) =
                if nr == nr::GETRESUID { (id.uid, id.euid) } else { (id.gid, id.egid) };
            for (addr, v) in [(a[0], real), (a[1], effective), (a[2], effective)] {
                put(p, addr, &v.to_le_bytes())?;
            }
            Ok(0)
        }
        // Supplementary groups: just the primary one.
        nr::GETGROUPS => {
            if a[0] > 0 {
                put(p, a[1], &host.identity().gid.to_le_bytes())?;
            }
            Ok(1)
        }
        nr::SETGROUPS => Err(errno::EPERM),
        nr::KILL | nr::TKILL | nr::TGKILL => kill(p, nr, a),
        nr::RT_SIGSUSPEND => {
            if a[1] != 8 {
                return Err(errno::EINVAL);
            }
            let mut b = [0u8; 8];
            p.mem.read(a[0], &mut b).map_err(efault)?;
            p.sigsuspend(t, u64::from_le_bytes(b))
        }
        nr::EVENTFD2 => host.eventfd(a[0] as u32, a[1] as i32),
        nr::EPOLL_CREATE1 => host.epoll_create1(a[0] as i32),
        nr::EPOLL_CTL => {
            const EPOLL_CTL_DEL: i32 = 2;
            // riscv64 `struct epoll_event` is not packed: events, padding, data.
            let event = if a[1] as i32 == EPOLL_CTL_DEL || a[3] == 0 {
                (0, 0)
            } else {
                let mut b = [0u8; 16];
                p.mem.read(a[3], &mut b).map_err(efault)?;
                let events = u32::from_le_bytes(b[..4].try_into().unwrap());
                (events, u64::from_le_bytes(b[8..].try_into().unwrap()))
            };
            host.epoll_ctl(fd, a[1] as i32, a[2] as i32, event)
        }
        nr::EPOLL_PWAIT => {
            // The signal mask argument is not applied: blocked waits are not interrupted
            // by signals yet anyway.
            let ready = host.epoll_wait(fd, a[2] as i32 as usize, a[3] as i32)?;
            let mut b = Vec::with_capacity(16 * ready.len());
            for (events, data) in &ready {
                b.extend(events.to_le_bytes());
                b.extend([0; 4]);
                b.extend(data.to_le_bytes());
            }
            put(p, a[1], &b)?;
            Ok(ready.len() as u64)
        }
        nr::RISCV_HWPROBE => hwprobe(p, a),
        nr::SOCKET => host.socket(a[0] as i32, a[1] as i32, a[2] as i32),
        nr::SOCKETPAIR => {
            let [s0, s1] = host.socketpair(a[0] as i32, a[1] as i32, a[2] as i32)?;
            let mut b = s0.to_le_bytes().to_vec();
            b.extend(s1.to_le_bytes());
            if let Err(e) = put(p, a[3], &b) {
                let _ = (host.close(s0), host.close(s1));
                return Err(e);
            }
            Ok(0)
        }
        nr::BIND | nr::CONNECT => {
            let addr = read_sockaddr(p, a[1], a[2])?;
            if nr == nr::BIND { host.bind(fd, &addr) } else { host.connect(fd, &addr) }
        }
        nr::LISTEN => host.listen(fd, a[1] as i32),
        nr::ACCEPT | nr::ACCEPT4 => {
            let flags = if nr == nr::ACCEPT4 { a[3] as i32 } else { 0 };
            let (new, addr) = host.accept4(fd, flags)?;
            if let Err(e) = put_sockaddr(p, a[1], a[2], &addr) {
                let _ = host.close(new as i32);
                return Err(e);
            }
            Ok(new)
        }
        nr::GETSOCKNAME | nr::GETPEERNAME => {
            let addr = host.sockname(fd, nr == nr::GETPEERNAME)?;
            put_sockaddr(p, a[1], a[2], &addr)?;
            Ok(0)
        }
        nr::SETSOCKOPT => {
            let mut value = vec![0u8; (a[4] as usize).min(256)];
            p.mem.read(a[3], &mut value).map_err(efault)?;
            host.setsockopt(fd, a[1] as i32, a[2] as i32, &value)
        }
        nr::GETSOCKOPT => {
            let mut len = [0u8; 4];
            p.mem.read(a[4], &mut len).map_err(efault)?;
            let len = u32::from_le_bytes(len) as usize;
            let value = host.getsockopt(fd, a[1] as i32, a[2] as i32, len)?;
            let n = value.len().min(len);
            put(p, a[3], &value[..n])?;
            put(p, a[4], &(n as u32).to_le_bytes())?;
            Ok(0)
        }
        nr::SENDTO => {
            let to = if a[4] != 0 { Some(read_sockaddr(p, a[4], a[5])?) } else { None };
            host.sendmsg(fd, to.as_deref(), &[buf(a[1], a[2], Prot::READ)?], a[3] as i32)
        }
        nr::RECVFROM => {
            let (n, from, _) = host.recvmsg(fd, &[buf(a[1], a[2], Prot::WRITE)?], a[3] as i32)?;
            put_sockaddr(p, a[4], a[5], &from)?;
            Ok(n)
        }
        nr::SENDMSG | nr::RECVMSG => {
            // struct msghdr: name, namelen (u32, padded), iov, iovlen, control,
            // controllen, flags. Ancillary data (e.g. SCM_RIGHTS) is not supported.
            let mut m = [0u8; 56];
            p.mem.read(a[1], &mut m).map_err(efault)?;
            let word = |i: usize| u64::from_le_bytes(m[i..i + 8].try_into().unwrap());
            let (name, namelen) = (word(0), u32::from_le_bytes(m[8..12].try_into().unwrap()));
            let need = if nr == nr::SENDMSG { Prot::READ } else { Prot::WRITE };
            let bufs = read_iovec(p, word(16), word(24))?
                .into_iter()
                .map(|(base, len)| buf(base, len, need))
                .collect::<Result<Vec<_>, _>>()?;
            if nr == nr::SENDMSG {
                let to =
                    if name != 0 { Some(read_sockaddr(p, name, namelen as u64)?) } else { None };
                host.sendmsg(fd, to.as_deref(), &bufs, a[2] as i32)
            } else {
                let (n, from, flags) = host.recvmsg(fd, &bufs, a[2] as i32)?;
                if name != 0 {
                    let len = from.len().min(namelen as usize);
                    put(p, name, &from[..len])?;
                    put(p, a[1] + 8, &(from.len() as u32).to_le_bytes())?;
                }
                put(p, a[1] + 40, &0u64.to_le_bytes())?; // msg_controllen
                put(p, a[1] + 48, &flags.to_le_bytes())?;
                Ok(n)
            }
        }
        nr::SHUTDOWN => host.shutdown(fd, a[1] as i32),
        nr::GETDENTS64 => {
            p.mem.host_buf(a[1], a[2], Prot::WRITE).map_err(efault)?;
            let records = host.getdents64(fd, a[2] as usize)?;
            put(p, a[1], &records)?;
            Ok(records.len() as u64)
        }
        nr::RT_SIGACTION => {
            let sig = signal_number(a[0])?;
            if sig == 0 || a[3] != 8 {
                return Err(errno::EINVAL);
            }
            let new = if a[1] != 0 {
                if signal::UNBLOCKABLE & signal::bit(sig) != 0 {
                    return Err(errno::EINVAL);
                }
                let mut b = [0u8; 24];
                p.mem.read(a[1], &mut b).map_err(efault)?;
                Some(signal::Action::from_bytes(&b))
            } else {
                None
            };
            let old = match new {
                Some(action) => p.set_action(sig, action),
                None => p.action(sig),
            };
            if a[2] != 0 {
                put(p, a[2], &old.to_bytes())?;
            }
            Ok(0)
        }
        nr::SIGALTSTACK => {
            let sp = t.cpu.x[2];
            let old = t.altstack.to_bytes(sp);
            if a[0] != 0 {
                let mut b = [0u8; 24];
                p.mem.read(a[0], &mut b).map_err(efault)?;
                if t.altstack.contains(sp) {
                    return Err(errno::EPERM); // cannot change the stack we are running on
                }
                let flags = u32::from_le_bytes(b[8..12].try_into().unwrap());
                let size = u64::from_le_bytes(b[16..24].try_into().unwrap());
                const MINSIGSTKSZ: u64 = 2048;
                t.altstack = if flags & signal::SS_DISABLE != 0 {
                    signal::AltStack { sp: 0, flags: signal::SS_DISABLE, size: 0 }
                } else if flags & !signal::SS_ONSTACK != 0 {
                    return Err(errno::EINVAL);
                } else if size < MINSIGSTKSZ {
                    return Err(errno::ENOMEM);
                } else {
                    let base = u64::from_le_bytes(b[..8].try_into().unwrap());
                    signal::AltStack { sp: base, flags: 0, size }
                };
            }
            if a[1] != 0 {
                put(p, a[1], &old)?;
            }
            Ok(0)
        }
        nr::RT_SIGPROCMASK => {
            const SIG_BLOCK: u64 = 0;
            const SIG_UNBLOCK: u64 = 1;
            const SIG_SETMASK: u64 = 2;
            if a[3] != 8 {
                return Err(errno::EINVAL);
            }
            let old = t.signals.mask();
            if a[1] != 0 {
                let mut b = [0u8; 8];
                p.mem.read(a[1], &mut b).map_err(efault)?;
                let set = u64::from_le_bytes(b);
                t.signals.set_mask(match a[0] {
                    SIG_BLOCK => old | set,
                    SIG_UNBLOCK => old & !set,
                    SIG_SETMASK => set,
                    _ => return Err(errno::EINVAL),
                });
            }
            if a[2] != 0 {
                put(p, a[2], &old.to_le_bytes())?;
            }
            Ok(0)
        }
        nr::RT_SIGPENDING => {
            if a[1] != 8 {
                return Err(errno::EINVAL);
            }
            put(p, a[0], &p.pending_for(t).to_le_bytes())?;
            Ok(0)
        }
        nr::RISCV_FLUSH_ICACHE => {
            // Make code written by any thread visible to all threads' instruction fetch:
            // bump the generation so every engine drops its decoded blocks. The range and
            // flags are only hints.
            p.mem.invalidate_caches();
            Ok(0)
        }
        nr::PWRITE64 => host.pwrite(fd, buf(a[1], a[2], Prot::READ)?, a[3] as i64),
        nr::UNLINKAT => host.unlinkat(fd, &cstr(a[1])?, a[2] as i32),
        nr::RENAMEAT => host.renameat2(fd, &cstr(a[1])?, a[2] as i32, &cstr(a[3])?, 0),
        nr::RENAMEAT2 => host.renameat2(fd, &cstr(a[1])?, a[2] as i32, &cstr(a[3])?, a[4] as u32),
        nr::MKDIRAT => host.mkdirat(fd, &cstr(a[1])?, a[2] as u32),
        nr::FACCESSAT => host.faccessat(fd, &p.host_path(&cstr(a[1])?, true), a[2] as i32, 0),
        nr::FACCESSAT2 => {
            host.faccessat(fd, &p.host_path(&cstr(a[1])?, true), a[2] as i32, a[3] as i32)
        }
        nr::READLINKAT => {
            let path = cstr(a[1])?;
            // There is no /proc on macOS; answer the one entry nearly every program asks.
            if path.as_bytes() == b"/proc/self/exe" {
                let n = p.exe.len().min(a[3] as usize);
                put(p, a[2], &p.exe[..n])?;
                return Ok(n as u64);
            }
            host.readlinkat(fd, &p.host_path(&path, false), buf(a[2], a[3], Prot::WRITE)?)
        }
        nr::GETCWD => {
            // Linux returns the length including the terminating NUL.
            let mut cwd = host.getcwd()?;
            cwd.push(0);
            if cwd.len() as u64 > a[1] {
                return Err(errno::ERANGE);
            }
            put(p, a[0], &cwd)?;
            Ok(cwd.len() as u64)
        }
        nr::CHDIR => host.chdir(&cstr(a[0])?),
        nr::FCHDIR => host.fchdir(fd),
        nr::DUP => host.dup(fd),
        nr::DUP3 => host.dup3(fd, a[1] as i32, a[2] as i32),
        nr::PIPE2 => {
            let fds = host.pipe2(a[1] as i32)?;
            let bytes: Vec<u8> = fds.iter().flat_map(|f| f.to_le_bytes()).collect();
            put(p, a[0], &bytes)?;
            Ok(0)
        }
        nr::FTRUNCATE => host.ftruncate(fd, a[1] as i64),
        nr::FSYNC => host.fsync(fd, false),
        nr::FDATASYNC => host.fsync(fd, true),
        nr::FCHMOD => host.fchmod(fd, a[1] as u32),
        nr::FCHMODAT => host.fchmodat(fd, &cstr(a[1])?, a[2] as u32, 0),
        nr::UMASK => Ok(host.umask(a[0] as u32) as u64),
        nr::NANOSLEEP => {
            let (sec, nsec) = read_timespec(p, a[0])?;
            let start = std::time::Instant::now();
            match host.nanosleep((sec, nsec)) {
                Err(errno::EINTR) if p.signal_waiting(t).is_some() => {
                    // Interrupted by a signal: report the time left.
                    if a[1] != 0 {
                        let total = std::time::Duration::new(sec as u64, nsec as u32);
                        let left = total.saturating_sub(start.elapsed());
                        let mut b = (left.as_secs() as i64).to_le_bytes().to_vec();
                        b.extend((left.subsec_nanos() as i64).to_le_bytes());
                        put(p, a[1], &b)?;
                    }
                    Err(errno::EINTR)
                }
                Err(errno::EINTR) => Err(errno::EINTR), // spurious: restarted by dispatch
                other => other,
            }
        }
        nr::CLOCK_NANOSLEEP => {
            const TIMER_ABSTIME: u64 = 1;
            let (sec, nsec) = read_timespec(p, a[2])?;
            let duration = if a[1] & TIMER_ABSTIME != 0 {
                let (now_s, now_ns) = host.clock_gettime(a[0] as i32)?;
                let total = (sec - now_s) * 1_000_000_000 + (nsec - now_ns);
                if total <= 0 {
                    return Ok(0);
                }
                (total / 1_000_000_000, total % 1_000_000_000)
            } else {
                host.clock_gettime(a[0] as i32)?; // validate the clock id
                (sec, nsec)
            };
            host.nanosleep(duration)?;
            Ok(0)
        }
        nr::PPOLL => {
            // The signal mask argument is ignored until signals arrive (M5).
            const MAX_FDS: u64 = 1 << 16;
            if a[1] > MAX_FDS {
                return Err(errno::EINVAL);
            }
            let mut raw = vec![0u8; a[1] as usize * 8];
            p.mem.read(a[0], &mut raw).map_err(efault)?;
            let mut fds: Vec<PollFd> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| PollFd {
                    fd: i32::from_le_bytes(c[..4].try_into().unwrap()),
                    events: i16::from_le_bytes([c[4], c[5]]),
                    revents: 0,
                })
                .collect();
            let timeout = if a[2] == 0 { None } else { Some(read_timespec(p, a[2])?) };
            let n = host.poll(&mut fds, timeout)?;
            for (i, f) in fds.iter().enumerate() {
                put(p, a[0] + i as u64 * 8 + 6, &f.revents.to_le_bytes())?;
            }
            Ok(n)
        }
        nr::PRCTL => {
            // Thread names (libtest and std name their threads); the rest are unsupported.
            const PR_SET_NAME: u64 = 15;
            const PR_GET_NAME: u64 = 16;
            match a[0] {
                PR_SET_NAME => {
                    let mut name = [0u8; 16];
                    let s = p.mem.read_cstr(a[1], 4096).map_err(efault)?;
                    let s = s.as_bytes();
                    let n = s.len().min(15);
                    name[..n].copy_from_slice(&s[..n]);
                    t.name = name;
                    Ok(0)
                }
                PR_GET_NAME => put(p, a[1], &t.name).map(|()| 0),
                _ => Err(errno::EINVAL),
            }
        }
        nr::SCHED_YIELD => {
            std::thread::yield_now();
            Ok(0)
        }
        nr::GETRUSAGE => {
            const RUSAGE_THREAD: i32 = 1;
            let who = a[0] as i32;
            let ru = host.getrusage(if who == RUSAGE_THREAD { 0 } else { who })?;
            put(p, a[1], &linux_rusage(&ru))?;
            Ok(0)
        }
        nr::PRLIMIT64 => {
            // Only the calling process (pid 0 or our own pid) is supported. New limits are
            // accepted and ignored: we do not enforce them.
            if a[0] != 0 && a[0] != host.identity().pid as u64 {
                return Err(errno::ESRCH);
            }
            if a[3] != 0 {
                const RLIMIT_STACK: u32 = 3;
                let (soft, hard) = if a[1] as u32 == RLIMIT_STACK {
                    (crate::process::STACK_SIZE, u64::MAX)
                } else {
                    host.getrlimit(a[1] as u32)?
                };
                let mut bytes = soft.to_le_bytes().to_vec();
                bytes.extend(hard.to_le_bytes());
                put(p, a[3], &bytes)?;
            }
            Ok(0)
        }
        nr::BRK => Ok(p.brk(a[0])),
        nr::MREMAP => p.mremap(a[0], a[1], a[2], a[3] as u32, a[4]),
        nr::MADVISE => p.madvise(a[0], a[1], a[2] as u32),
        nr::MMAP => p.mmap(host, a[0], a[1], a[2] as u32, a[3] as u32, a[4] as i32, a[5]),
        nr::MUNMAP => p.munmap(a[0], a[1]),
        nr::MPROTECT => p.mprotect(a[0], a[1], a[2] as u32),
        _ => {
            if p.log.warned.lock().unwrap_or_else(|e| e.into_inner()).insert(nr) {
                eprintln!("rvrun: unsupported syscall {nr} ({})", name(nr).unwrap_or("unknown"));
            }
            Err(errno::ENOSYS)
        }
    }
}

/// Read a guest `struct iovec` array as `(base, len)` pairs.
/// A `sockaddr` of `len` bytes from the guest.
fn read_sockaddr(p: &Process, addr: u64, len: u64) -> Result<Vec<u8>, i32> {
    if len > 128 {
        return Err(errno::EINVAL);
    }
    let mut b = vec![0u8; len as usize];
    p.mem.read(addr, &mut b).map_err(efault)?;
    Ok(b)
}

/// Return a `sockaddr` the way the kernel does: into the buffer at `addr` whose size is
/// the u32 at `len_ptr` (truncated if short), then the full size into `*len_ptr`. Nothing
/// if `addr` is null.
fn put_sockaddr(p: &Process, addr: u64, len_ptr: u64, sockaddr: &[u8]) -> Result<(), i32> {
    if addr == 0 {
        return Ok(());
    }
    let mut len = [0u8; 4];
    p.mem.read(len_ptr, &mut len).map_err(efault)?;
    let room = u32::from_le_bytes(len) as usize;
    put(p, addr, &sockaddr[..sockaddr.len().min(room)])?;
    put(p, len_ptr, &(sockaddr.len() as u32).to_le_bytes())
}

fn read_iovec(p: &Process, addr: u64, count: u64) -> Result<Vec<(u64, u64)>, i32> {
    if count > IOV_MAX {
        return Err(errno::EINVAL);
    }
    let mut raw = vec![0u8; count as usize * 16];
    p.mem.read(addr, &mut raw).map_err(efault)?;
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
fn write_stat(p: &Process, addr: u64, st: &Stat) -> SysResult {
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
