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
use std::sync::Mutex;

use crate::memory::{GuestBuf, MemFault, Memory, Prot};
use crate::process::{Process, Thread};

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
    pub const ESRCH: i32 = 3;
    pub const ERANGE: i32 = 34;
    pub const ENOSYS: i32 = 38;
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
    RT_SIGACTION = 134 "rt_sigaction",
    RT_SIGPROCMASK = 135 "rt_sigprocmask",
    UNAME = 160 "uname",
    GETRUSAGE = 165 "getrusage",
    UMASK = 166 "umask",
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
    WAIT4 = 260 "wait4",
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
    /// The process sent itself a signal whose default action terminates it (`abort()`
    /// does this with SIGABRT). Handlers arrive with signals in M5.
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
fn signal_to_self(to_self: bool, sig: i32) -> Option<i32> {
    // Default action ignore: SIGCHLD 17, SIGCONT 18, SIGURG 23, SIGWINCH 28. Signal 0 only
    // checks that the target exists.
    const IGNORED: [i32; 5] = [0, 17, 18, 23, 28];
    (to_self && !IGNORED.contains(&sig) && (1..=64).contains(&sig)).then_some(sig)
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

/// Copy a syscall result out to guest memory, which must be writable by the guest.
fn put(p: &Process, addr: u64, bytes: &[u8]) -> Result<(), i32> {
    p.mem.write(addr, bytes).map_err(efault)
}

/// Handle the `ecall` at `t.cpu.pc`: read the request from the thread's registers, perform
/// it, write the result to `a0`, and advance past the `ecall`.
pub fn dispatch(p: &Process, t: &mut Thread) -> Flow {
    let host = &*p.host;
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
        nr::KILL | nr::TKILL | nr::TGKILL => {
            let pid = host.identity().pid as i64;
            let (target, sig) = match nr {
                nr::KILL => (a[0] as i32 as i64, a[1] as i32),
                nr::TKILL => (a[0] as i32 as i64, a[1] as i32),
                _ => (a[1] as i32 as i64, a[2] as i32),
            };
            let to_self = target == pid || (nr == nr::KILL && target == 0);
            match signal_to_self(to_self, sig) {
                Some(sig) => {
                    if p.log.strace.load(std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[rvrun {}] {}(..., {sig}) = ? (fatal)",
                            t.tid,
                            name(nr).unwrap()
                        );
                    }
                    return Flow::Killed(sig);
                }
                None if to_self => Ok(0),
                None => host.kill(target as i32, sig),
            }
        }
        _ => call(p, t, nr, a),
    };

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
            host.openat(fd, &path, a[2] as i32, a[3] as u32)
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
        nr::GETRANDOM => host.getrandom(buf(a[0], a[1], Prot::WRITE)?),
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
        nr::RT_SIGACTION => {
            // Signals arrive in M5. Until then, accept handlers and report SIG_DFL as the
            // old one (riscv64 kernel struct sigaction: handler, flags, mask = 24 bytes).
            if a[2] != 0 {
                put(p, a[2], &[0; 24])?;
            }
            Ok(0)
        }
        nr::SIGALTSTACK => {
            // No signal is ever delivered before M5, so accept the new stack and report the
            // old one as disabled (stack_t: ss_sp, ss_flags = SS_DISABLE, ss_size).
            if a[1] != 0 {
                const SS_DISABLE: u64 = 2;
                let mut old = 0u64.to_le_bytes().to_vec();
                old.extend(SS_DISABLE.to_le_bytes());
                old.extend(0u64.to_le_bytes());
                put(p, a[1], &old)?;
            }
            Ok(0)
        }
        nr::RT_SIGPROCMASK => {
            if a[2] != 0 {
                put(p, a[2], &[0; 8])?;
            }
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
        nr::FACCESSAT => host.faccessat(fd, &cstr(a[1])?, a[2] as i32, 0),
        nr::FACCESSAT2 => host.faccessat(fd, &cstr(a[1])?, a[2] as i32, a[3] as i32),
        nr::READLINKAT => {
            let path = cstr(a[1])?;
            // There is no /proc on macOS; answer the one entry nearly every program asks.
            if path.as_bytes() == b"/proc/self/exe" {
                let n = p.exe.len().min(a[3] as usize);
                put(p, a[2], &p.exe[..n])?;
                return Ok(n as u64);
            }
            host.readlinkat(fd, &path, buf(a[2], a[3], Prot::WRITE)?)
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
            host.nanosleep(read_timespec(p, a[0])?)?;
            if a[1] != 0 {
                put(p, a[1], &[0; 16])?; // no signals yet, so never interrupted
            }
            Ok(0)
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
