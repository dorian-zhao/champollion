//! Route A backend: serve the guest's Linux syscalls with the host Linux kernel's.
//!
//! riscv64 and arm64 share the asm-generic syscall ABI, so almost every value passes
//! through unchanged: errno, signal numbers, `AT_*`, `MSG_*`, socket addresses and options,
//! clock ids, and the layouts of `struct stat`, `statfs`, and `linux_dirent64`. The
//! exception is four `open` flags, whose bits arm64 defines differently.

use std::ffi::CStr;

use crate::memory::GuestBuf;
use crate::syscall::{Flock, HostOs, Identity, PollFd, Rusage, Stat, SysResult, errno};

/// The Linux backend. Guest fds are host fds, and it keeps no state.
#[derive(Default)]
pub struct LinuxHost;

/// asm-generic (riscv64) `open` flags whose arm64 values differ.
mod generic_o {
    pub const DIRECT: i32 = 0o40000;
    pub const LARGEFILE: i32 = 0o100000;
    pub const DIRECTORY: i32 = 0o200000;
    pub const NOFOLLOW: i32 = 0o400000;
}

/// The flags that differ, as `(asm-generic, host)` pairs.
const OPEN_FLAG_PAIRS: [(i32, i32); 4] = [
    (generic_o::DIRECT, libc::O_DIRECT),
    (generic_o::LARGEFILE, libc::O_LARGEFILE),
    (generic_o::DIRECTORY, libc::O_DIRECTORY),
    (generic_o::NOFOLLOW, libc::O_NOFOLLOW),
];

/// Guest (asm-generic) `open`/`F_SETFL` flags -> host.
fn host_open_flags(guest: i32) -> i32 {
    let mask = OPEN_FLAG_PAIRS.iter().fold(0, |m, &(g, h)| m | g | h);
    OPEN_FLAG_PAIRS.iter().fold(guest & !mask, |f, &(g, h)| if guest & g != 0 { f | h } else { f })
}

/// Host file status flags (`F_GETFL`) -> guest.
fn guest_open_flags(host: i32) -> i32 {
    let mask = OPEN_FLAG_PAIRS.iter().fold(0, |m, &(g, h)| m | g | h);
    OPEN_FLAG_PAIRS.iter().fold(host & !mask, |f, &(g, h)| if host & h != 0 { f | g } else { f })
}

fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(errno::EIO)
}

/// Convert a `-1`-on-error return into a [`SysResult`].
fn check(ret: isize) -> SysResult {
    if ret < 0 { Err(last_errno()) } else { Ok(ret as u64) }
}

/// A raw system call with up to six arguments.
macro_rules! sys {
    ($nr:ident $(, $arg:expr)* $(,)?) => {
        // SAFETY: each call site passes arguments valid for that system call; guest buffers
        // go straight to the kernel, which fails with EFAULT instead of crashing if another
        // guest thread unmaps them mid-call.
        check(unsafe { libc::syscall(libc::$nr $(, $arg)*) } as isize)
    };
}

fn to_stat(st: &libc::stat) -> Stat {
    Stat {
        dev: st.st_dev,
        ino: st.st_ino,
        mode: st.st_mode,
        nlink: st.st_nlink,
        uid: st.st_uid,
        gid: st.st_gid,
        rdev: st.st_rdev,
        size: st.st_size,
        blksize: st.st_blksize,
        blocks: st.st_blocks,
        atime: (st.st_atime, st.st_atime_nsec),
        mtime: (st.st_mtime, st.st_mtime_nsec),
        ctime: (st.st_ctime, st.st_ctime_nsec),
    }
}

fn timeval(tv: libc::timeval) -> (i64, i64) {
    (tv.tv_sec, tv.tv_usec)
}

/// `struct statfs` is the same 120 bytes on both architectures.
fn statfs_bytes(raw: [u64; 15]) -> Vec<u8> {
    raw.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `fcntl` commands whose argument is a plain integer, passed through as they are.
const PLAIN_FCNTL: [i32; 12] = [
    0,    // F_DUPFD
    1,    // F_GETFD
    2,    // F_SETFD
    8,    // F_SETOWN
    9,    // F_GETOWN
    10,   // F_SETSIG
    11,   // F_GETSIG
    1024, // F_SETLEASE
    1025, // F_GETLEASE
    1030, // F_DUPFD_CLOEXEC
    1031, // F_SETPIPE_SZ
    1032, // F_GETPIPE_SZ
];

impl HostOs for LinuxHost {
    fn epoll_create1(&self, flags: i32) -> SysResult {
        sys!(SYS_epoll_create1, flags)
    }

    fn epoll_ctl(&self, epfd: i32, op: i32, fd: i32, (events, data): (u32, u64)) -> SysResult {
        let mut ev = libc::epoll_event { events, u64: data };
        sys!(SYS_epoll_ctl, epfd, op, fd, &raw mut ev)
    }

    fn epoll_wait(&self, epfd: i32, max: usize, timeout_ms: i32) -> Result<Vec<(u32, u64)>, i32> {
        let mut evs = vec![libc::epoll_event { events: 0, u64: 0 }; max];
        let n =
            sys!(SYS_epoll_pwait, epfd, evs.as_mut_ptr(), max as i32, timeout_ms, 0usize, 8usize)?;
        Ok(evs[..n as usize].iter().map(|e| (e.events, e.u64)).collect())
    }

    fn pidfd_open(&self, pid: i32, flags: u32) -> SysResult {
        sys!(SYS_pidfd_open, pid, flags)
    }

    fn eventfd(&self, initval: u32, flags: i32) -> SysResult {
        sys!(SYS_eventfd2, initval, flags)
    }

    fn getdents64(&self, fd: i32, max: usize) -> Result<Vec<u8>, i32> {
        let mut buf = vec![0u8; max];
        let n = sys!(SYS_getdents64, fd, buf.as_mut_ptr(), max)?;
        buf.truncate(n as usize);
        Ok(buf)
    }

    fn wait4(&self, pid: i32, options: i32) -> Result<(u64, i32), i32> {
        let mut status = 0i32;
        let got = sys!(SYS_wait4, pid, &raw mut status, options, 0usize)?;
        Ok((got, status))
    }

    fn socket(&self, domain: i32, ty: i32, protocol: i32) -> SysResult {
        sys!(SYS_socket, domain, ty, protocol)
    }

    fn socketpair(&self, domain: i32, ty: i32, protocol: i32) -> Result<[i32; 2], i32> {
        let mut fds = [0i32; 2];
        sys!(SYS_socketpair, domain, ty, protocol, fds.as_mut_ptr())?;
        Ok(fds)
    }

    fn bind(&self, fd: i32, addr: &[u8]) -> SysResult {
        sys!(SYS_bind, fd, addr.as_ptr(), addr.len())
    }

    fn connect(&self, fd: i32, addr: &[u8]) -> SysResult {
        sys!(SYS_connect, fd, addr.as_ptr(), addr.len())
    }

    fn listen(&self, fd: i32, backlog: i32) -> SysResult {
        sys!(SYS_listen, fd, backlog)
    }

    fn accept4(&self, fd: i32, flags: i32) -> Result<(u64, Vec<u8>), i32> {
        let mut addr = [0u8; 128];
        let mut len = addr.len() as libc::socklen_t;
        let new = sys!(SYS_accept4, fd, addr.as_mut_ptr(), &raw mut len, flags)?;
        Ok((new, addr[..(len as usize).min(addr.len())].to_vec()))
    }

    fn sockname(&self, fd: i32, peer: bool) -> Result<Vec<u8>, i32> {
        let mut addr = [0u8; 128];
        let mut len = addr.len() as libc::socklen_t;
        if peer {
            sys!(SYS_getpeername, fd, addr.as_mut_ptr(), &raw mut len)?;
        } else {
            sys!(SYS_getsockname, fd, addr.as_mut_ptr(), &raw mut len)?;
        }
        Ok(addr[..(len as usize).min(addr.len())].to_vec())
    }

    fn setsockopt(&self, fd: i32, level: i32, name: i32, value: &[u8]) -> SysResult {
        sys!(SYS_setsockopt, fd, level, name, value.as_ptr(), value.len())
    }

    fn getsockopt(&self, fd: i32, level: i32, name: i32, len: usize) -> Result<Vec<u8>, i32> {
        let mut value = vec![0u8; len];
        let mut n = len as libc::socklen_t;
        sys!(SYS_getsockopt, fd, level, name, value.as_mut_ptr(), &raw mut n)?;
        value.truncate(n as usize);
        Ok(value)
    }

    fn sendmsg(&self, fd: i32, to: Option<&[u8]>, bufs: &[GuestBuf], flags: i32) -> SysResult {
        // SAFETY: an all-zero msghdr is valid.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        if let Some(to) = to {
            msg.msg_name = to.as_ptr().cast_mut().cast();
            msg.msg_namelen = to.len() as libc::socklen_t;
        }
        // GuestBuf is layout-compatible with struct iovec.
        msg.msg_iov = bufs.as_ptr().cast_mut().cast();
        msg.msg_iovlen = bufs.len() as _;
        sys!(SYS_sendmsg, fd, &raw const msg, flags)
    }

    fn recvmsg(&self, fd: i32, bufs: &[GuestBuf], flags: i32) -> Result<(u64, Vec<u8>, i32), i32> {
        let mut addr = [0u8; 128];
        // SAFETY: an all-zero msghdr is valid.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_name = addr.as_mut_ptr().cast();
        msg.msg_namelen = addr.len() as libc::socklen_t;
        msg.msg_iov = bufs.as_ptr().cast_mut().cast();
        msg.msg_iovlen = bufs.len() as _;
        let n = sys!(SYS_recvmsg, fd, &raw mut msg, flags)?;
        let from = addr[..(msg.msg_namelen as usize).min(addr.len())].to_vec();
        Ok((n, from, msg.msg_flags))
    }

    fn shutdown(&self, fd: i32, how: i32) -> SysResult {
        sys!(SYS_shutdown, fd, how)
    }

    fn read(&self, fd: i32, buf: GuestBuf) -> SysResult {
        sys!(SYS_read, fd, buf.ptr, buf.len)
    }

    fn write(&self, fd: i32, buf: GuestBuf) -> SysResult {
        sys!(SYS_write, fd, buf.ptr, buf.len)
    }

    fn readv(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        sys!(SYS_readv, fd, bufs.as_ptr(), bufs.len())
    }

    fn writev(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        sys!(SYS_writev, fd, bufs.as_ptr(), bufs.len())
    }

    fn pread(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        sys!(SYS_pread64, fd, buf.ptr, buf.len, offset)
    }

    fn pwrite(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        sys!(SYS_pwrite64, fd, buf.ptr, buf.len, offset)
    }

    fn openat(&self, dirfd: i32, path: &CStr, flags: i32, mode: u32) -> SysResult {
        sys!(SYS_openat, dirfd, path.as_ptr(), host_open_flags(flags), mode)
    }

    fn close(&self, fd: i32) -> SysResult {
        sys!(SYS_close, fd)
    }

    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> SysResult {
        const F_GETFL: i32 = 3;
        const F_SETFL: i32 = 4;
        match cmd {
            F_GETFL => sys!(SYS_fcntl, fd, cmd).map(|f| guest_open_flags(f as i32) as u64),
            F_SETFL => sys!(SYS_fcntl, fd, cmd, host_open_flags(arg as i32)),
            _ if PLAIN_FCNTL.contains(&cmd) => sys!(SYS_fcntl, fd, cmd, arg),
            _ => Err(errno::EINVAL),
        }
    }

    fn lseek(&self, fd: i32, offset: i64, whence: i32) -> SysResult {
        sys!(SYS_lseek, fd, offset, whence)
    }

    fn futex_wait_shared(
        &self,
        word: GuestBuf,
        expected: u32,
        timeout: Option<std::time::Duration>,
    ) -> SysResult {
        const FUTEX_WAIT: i32 = 0; // shared: not FUTEX_PRIVATE_FLAG
        let ts = timeout.map(|t| libc::timespec {
            tv_sec: t.as_secs() as i64,
            tv_nsec: t.subsec_nanos() as i64,
        });
        let ts_ptr = ts.as_ref().map_or(std::ptr::null(), |t| t as *const libc::timespec);
        // A guest signal interrupts the wait with EINTR (see Process::interrupt).
        sys!(SYS_futex, word.ptr, FUTEX_WAIT, expected, ts_ptr).map(|_| 0)
    }

    fn futex_wake_shared(&self, word: GuestBuf, count: u32) -> SysResult {
        const FUTEX_WAKE: i32 = 1;
        sys!(SYS_futex, word.ptr, FUTEX_WAKE, count.min(i32::MAX as u32))
    }

    fn linkat(
        &self,
        olddirfd: i32,
        old: &CStr,
        newdirfd: i32,
        new: &CStr,
        flags: i32,
    ) -> SysResult {
        sys!(SYS_linkat, olddirfd, old.as_ptr(), newdirfd, new.as_ptr(), flags)
    }

    fn symlinkat(&self, target: &CStr, newdirfd: i32, new: &CStr) -> SysResult {
        sys!(SYS_symlinkat, target.as_ptr(), newdirfd, new.as_ptr())
    }

    fn bytes_readable(&self, fd: i32) -> SysResult {
        let mut n: libc::c_int = 0;
        sys!(SYS_ioctl, fd, libc::FIONREAD, &raw mut n)?;
        Ok(n.max(0) as u64)
    }

    fn statfs(&self, path: &CStr) -> Result<Vec<u8>, i32> {
        let mut raw = [0u64; 15];
        sys!(SYS_statfs, path.as_ptr(), raw.as_mut_ptr())?;
        Ok(statfs_bytes(raw))
    }

    fn fstatfs(&self, fd: i32) -> Result<Vec<u8>, i32> {
        let mut raw = [0u64; 15];
        sys!(SYS_fstatfs, fd, raw.as_mut_ptr())?;
        Ok(statfs_bytes(raw))
    }

    fn fstat(&self, fd: i32) -> Result<Stat, i32> {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        sys!(SYS_fstat, fd, st.as_mut_ptr())?;
        // SAFETY: initialized by the successful call.
        Ok(to_stat(&unsafe { st.assume_init() }))
    }

    fn fstatat(&self, dirfd: i32, path: &CStr, flags: i32) -> Result<Stat, i32> {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        sys!(SYS_newfstatat, dirfd, path.as_ptr(), st.as_mut_ptr(), flags)?;
        // SAFETY: initialized by the successful call.
        Ok(to_stat(&unsafe { st.assume_init() }))
    }

    fn clock_gettime(&self, clock_id: i32) -> Result<(i64, i64), i32> {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: a valid out-pointer; libc goes through the vDSO.
        check(unsafe { libc::clock_gettime(clock_id, &mut ts) } as isize)?;
        Ok((ts.tv_sec, ts.tv_nsec))
    }

    fn getrandom(&self, buf: GuestBuf) -> SysResult {
        sys!(SYS_getrandom, buf.ptr, buf.len, 0)
    }

    fn winsize(&self, fd: i32) -> Result<[u16; 4], i32> {
        let mut ws = libc::winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
        sys!(SYS_ioctl, fd, libc::TIOCGWINSZ, &raw mut ws)?;
        Ok([ws.ws_row, ws.ws_col, ws.ws_xpixel, ws.ws_ypixel])
    }

    fn unlinkat(&self, dirfd: i32, path: &CStr, flags: i32) -> SysResult {
        sys!(SYS_unlinkat, dirfd, path.as_ptr(), flags)
    }

    fn renameat2(&self, ofd: i32, old: &CStr, nfd: i32, new: &CStr, flags: u32) -> SysResult {
        sys!(SYS_renameat2, ofd, old.as_ptr(), nfd, new.as_ptr(), flags)
    }

    fn mkdirat(&self, dirfd: i32, path: &CStr, mode: u32) -> SysResult {
        sys!(SYS_mkdirat, dirfd, path.as_ptr(), mode)
    }

    fn faccessat(&self, dirfd: i32, path: &CStr, mode: i32, flags: i32) -> SysResult {
        if flags == 0 {
            return sys!(SYS_faccessat, dirfd, path.as_ptr(), mode);
        }
        sys!(SYS_faccessat2, dirfd, path.as_ptr(), mode, flags)
    }

    fn readlinkat(&self, dirfd: i32, path: &CStr, buf: GuestBuf) -> SysResult {
        sys!(SYS_readlinkat, dirfd, path.as_ptr(), buf.ptr, buf.len)
    }

    fn getcwd(&self) -> Result<Vec<u8>, i32> {
        use std::os::unix::ffi::OsStringExt;
        std::env::current_dir()
            .map(|p| p.into_os_string().into_vec())
            .map_err(|e| e.raw_os_error().unwrap_or(errno::EIO))
    }

    fn chdir(&self, path: &CStr) -> SysResult {
        sys!(SYS_chdir, path.as_ptr())
    }

    fn fchdir(&self, fd: i32) -> SysResult {
        sys!(SYS_fchdir, fd)
    }

    fn dup(&self, fd: i32) -> SysResult {
        sys!(SYS_dup, fd)
    }

    fn dup3(&self, old: i32, new: i32, flags: i32) -> SysResult {
        sys!(SYS_dup3, old, new, flags)
    }

    fn pipe2(&self, flags: i32) -> Result<[i32; 2], i32> {
        let mut fds = [0i32; 2];
        // O_DIRECT (packet mode) is one of the flags whose bit differs.
        sys!(SYS_pipe2, fds.as_mut_ptr(), host_open_flags(flags))?;
        Ok(fds)
    }

    fn ftruncate(&self, fd: i32, len: i64) -> SysResult {
        sys!(SYS_ftruncate, fd, len)
    }

    fn fsync(&self, fd: i32, data_only: bool) -> SysResult {
        if data_only { sys!(SYS_fdatasync, fd) } else { sys!(SYS_fsync, fd) }
    }

    fn fchmod(&self, fd: i32, mode: u32) -> SysResult {
        sys!(SYS_fchmod, fd, mode)
    }

    fn fchmodat(&self, dirfd: i32, path: &CStr, mode: u32, flags: i32) -> SysResult {
        // libc implements AT_SYMLINK_NOFOLLOW, which the system call lacks.
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::fchmodat(dirfd, path.as_ptr(), mode, flags) } as isize)
    }

    fn umask(&self, mask: u32) -> u32 {
        // SAFETY: no pointers.
        unsafe { libc::umask(mask) }
    }

    fn lock(&self, fd: i32, cmd: i32, lock: &mut Flock) -> SysResult {
        let mut fl = libc::flock {
            l_type: lock.typ,
            l_whence: lock.whence,
            l_start: lock.start,
            l_len: lock.len,
            l_pid: lock.pid,
        };
        let r = sys!(SYS_fcntl, fd, cmd, &raw mut fl)?;
        (lock.typ, lock.whence, lock.start, lock.len, lock.pid) =
            (fl.l_type, fl.l_whence, fl.l_start, fl.l_len, fl.l_pid);
        Ok(r)
    }

    fn nanosleep(&self, (sec, nsec): (i64, i64)) -> SysResult {
        let req = libc::timespec { tv_sec: sec, tv_nsec: nsec };
        // A guest signal interrupts the sleep with EINTR (see Process::interrupt).
        sys!(SYS_nanosleep, &raw const req, 0usize)
    }

    fn getrusage(&self, who: i32) -> Result<Rusage, i32> {
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
        sys!(SYS_getrusage, who, ru.as_mut_ptr())?;
        // SAFETY: initialized by the successful call.
        let ru = unsafe { ru.assume_init() };
        Ok(Rusage {
            utime: timeval(ru.ru_utime),
            stime: timeval(ru.ru_stime),
            maxrss_kib: ru.ru_maxrss,
            minflt: ru.ru_minflt,
            majflt: ru.ru_majflt,
            nvcsw: ru.ru_nvcsw,
            nivcsw: ru.ru_nivcsw,
        })
    }

    fn getrlimit(&self, resource: u32) -> Result<(u64, u64), i32> {
        let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        sys!(SYS_prlimit64, 0, resource, 0usize, &raw mut rl)?;
        Ok((rl.rlim_cur, rl.rlim_max))
    }

    fn poll(&self, fds: &mut [PollFd], timeout: Option<(i64, i64)>) -> SysResult {
        let ts = timeout.map(|(s, ns)| libc::timespec { tv_sec: s, tv_nsec: ns });
        let ts_ptr = ts.as_ref().map_or(std::ptr::null(), |t| t as *const libc::timespec);
        // PollFd is layout-compatible with struct pollfd.
        sys!(SYS_ppoll, fds.as_mut_ptr(), fds.len(), ts_ptr, 0usize, 8usize)
    }

    fn kill(&self, pid: i32, sig: i32) -> SysResult {
        sys!(SYS_kill, pid, sig)
    }

    fn fadvise(&self, fd: i32, offset: i64, len: i64, advice: i32) -> SysResult {
        sys!(SYS_fadvise64, fd, offset, len, advice)
    }

    fn sysinfo(&self) -> Result<Vec<u8>, i32> {
        let mut raw = [0u64; 14];
        sys!(SYS_sysinfo, raw.as_mut_ptr())?;
        Ok(raw.iter().flat_map(|w| w.to_le_bytes()).collect())
    }

    fn raw(&self, nr: u64, a: [u64; 6]) -> Option<SysResult> {
        // SAFETY: the caller only passes calls whose pointer arguments are host pointers
        // (guest memory in the identity layout); the kernel validates them.
        Some(check(
            unsafe { libc::syscall(nr as libc::c_long, a[0], a[1], a[2], a[3], a[4], a[5]) }
                as isize,
        ))
    }

    fn setpgid(&self, pid: i32, pgid: i32) -> SysResult {
        sys!(SYS_setpgid, pid, pgid)
    }

    fn getpgid(&self, pid: i32) -> SysResult {
        sys!(SYS_getpgid, pid)
    }

    fn getsid(&self, pid: i32) -> SysResult {
        sys!(SYS_getsid, pid)
    }

    fn setsid(&self) -> SysResult {
        sys!(SYS_setsid)
    }

    fn identity(&self) -> Identity {
        // SAFETY: these calls have no preconditions and cannot fail.
        unsafe {
            Identity {
                pid: libc::getpid() as u32,
                ppid: libc::getppid() as u32,
                uid: libc::getuid(),
                euid: libc::geteuid(),
                gid: libc::getgid(),
                egid: libc::getegid(),
            }
        }
    }

    fn hostname(&self) -> Vec<u8> {
        let mut buf = [0u8; 256];
        // SAFETY: buf is valid for buf.len() bytes.
        if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
            return b"localhost".to_vec();
        }
        buf.iter().take_while(|&&b| b != 0).copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syscall::AT_FDCWD;

    #[test]
    fn open_flags_translate_both_ways() {
        let g = 0o2 | 0o100 | generic_o::DIRECTORY | generic_o::NOFOLLOW | generic_o::LARGEFILE;
        let h = host_open_flags(g);
        assert_eq!(
            h,
            0o2 | libc::O_CREAT | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_LARGEFILE
        );
        assert_eq!(guest_open_flags(h), g);
        assert_eq!(host_open_flags(0o2000000), libc::O_CLOEXEC);
    }

    #[test]
    fn opens_reads_and_stats_files() {
        let dir = std::env::temp_dir().join(format!("champollion-host-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, b"hello").unwrap();
        let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let h = LinuxHost;
        let fd = h.openat(AT_FDCWD, &cpath, 0, 0).unwrap() as i32;
        let st = h.fstat(fd).unwrap();
        assert_eq!(st.size, 5);
        assert_eq!(st.mode & 0o170000, 0o100000); // S_IFREG
        let mut buf = [0u8; 8];
        assert_eq!(h.read(fd, GuestBuf { ptr: buf.as_mut_ptr(), len: buf.len() }), Ok(5));
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(h.fcntl(fd, 3, 0), Ok(generic_o::LARGEFILE as u64)); // F_GETFL
        h.close(fd).unwrap();
        assert_eq!(h.close(fd), Err(errno::EBADF));
        let dirpath = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
        let d = h.openat(AT_FDCWD, &dirpath, generic_o::DIRECTORY, 0).unwrap() as i32;
        let ents = h.getdents64(d, 4096).unwrap();
        assert!(ents.windows(5).any(|w| w == b"f.txt"));
        h.close(d).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
