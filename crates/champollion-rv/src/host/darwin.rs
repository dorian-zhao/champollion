//! Route B backend: serve Linux syscalls with Darwin (macOS) calls.
//!
//! Guest file descriptors are host file descriptors. Every Linux-specific value coming in
//! (open flags, `AT_*` constants, `whence`, clock ids) is translated to its Darwin
//! counterpart, and every Darwin errno going out is translated to Linux.

use std::ffi::CStr;

use crate::memory::GuestBuf;
use crate::syscall::{AT_FDCWD, Flock, HostOs, Identity, PollFd, Rusage, Stat, SysResult, errno};

/// The Darwin backend. Guest fds are host fds; the only state kept here is for the Linux
/// fd types Darwin lacks (eventfd; epoll instances are plain kqueues).
#[derive(Default)]
pub struct DarwinHost {
    eventfds: super::kqueue::EventFds,
}

/// Linux (asm-generic) `open` flags.
mod linux_o {
    pub const ACCMODE: i32 = 0o3;
    pub const CREAT: i32 = 0o100;
    pub const EXCL: i32 = 0o200;
    pub const NOCTTY: i32 = 0o400;
    pub const TRUNC: i32 = 0o1000;
    pub const APPEND: i32 = 0o2000;
    pub const NONBLOCK: i32 = 0o4000;
    pub const DSYNC: i32 = 0o10000;
    pub const DIRECT: i32 = 0o40000;
    pub const LARGEFILE: i32 = 0o100000;
    pub const DIRECTORY: i32 = 0o200000;
    pub const NOFOLLOW: i32 = 0o400000;
    pub const NOATIME: i32 = 0o1000000;
    pub const CLOEXEC: i32 = 0o2000000;
    /// `O_SYNC` is `__O_SYNC | O_DSYNC`.
    pub const SYNC: i32 = 0o4010000;
    pub const PATH: i32 = 0o10000000;
    pub const TMPFILE: i32 = 0o20000000;
}

const LINUX_AT_SYMLINK_NOFOLLOW: i32 = 0x100;
const LINUX_AT_REMOVEDIR: i32 = 0x200;
const LINUX_AT_EACCESS: i32 = 0x200;
const LINUX_AT_EMPTY_PATH: i32 = 0x1000;

/// Linux signal number -> Darwin. 1–6, 8–9, 11, 13–15, and 21–28 agree.
fn darwin_signal(sig: i32) -> Option<i32> {
    Some(match sig {
        0..=6 | 8 | 9 | 11 | 13..=15 | 21..=28 => sig,
        7 => libc::SIGBUS,
        10 => libc::SIGUSR1,
        12 => libc::SIGUSR2,
        17 => libc::SIGCHLD,
        18 => libc::SIGCONT,
        19 => libc::SIGSTOP,
        20 => libc::SIGTSTP,
        29 => libc::SIGIO,
        31 => libc::SIGSYS,
        _ => return None, // SIGSTKFLT, SIGPWR, real-time signals
    })
}

/// Linux `RLIMIT_*` -> Darwin, or `None` for limits Darwin does not have.
fn darwin_rlimit(resource: u32) -> Option<libc::c_int> {
    Some(match resource {
        0 => libc::RLIMIT_CPU,
        1 => libc::RLIMIT_FSIZE,
        2 => libc::RLIMIT_DATA,
        3 => libc::RLIMIT_STACK,
        4 => libc::RLIMIT_CORE,
        5 => libc::RLIMIT_RSS,
        6 => libc::RLIMIT_NPROC,
        7 => libc::RLIMIT_NOFILE,
        8 => libc::RLIMIT_MEMLOCK,
        9 => libc::RLIMIT_AS,
        _ => return None,
    })
}

/// Set FD_CLOEXEC / O_NONBLOCK on `fd` per Linux `O_CLOEXEC` / `O_NONBLOCK` bits.
fn apply_fd_flags(fd: i32, linux_flags: i32) -> SysResult {
    // SAFETY: fcntl on a descriptor we just created.
    unsafe {
        if linux_flags & linux_o::CLOEXEC != 0 {
            check(libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) as isize)?;
        }
        if linux_flags & linux_o::NONBLOCK != 0 {
            let fl = check(libc::fcntl(fd, libc::F_GETFL) as isize)? as i32;
            check(libc::fcntl(fd, libc::F_SETFL, fl | libc::O_NONBLOCK) as isize)?;
        }
    }
    Ok(0)
}

/// Linux `POLL*` bits -> Darwin and back. Bits 0x1–0x80 agree; Linux POLLWRNORM (0x100)
/// and POLLWRBAND (0x200) are Darwin 0x4 and 0x100.
fn darwin_poll_events(linux: i16) -> i16 {
    let mut d = linux & 0xff;
    if linux & 0x100 != 0 {
        d |= libc::POLLWRNORM;
    }
    if linux & 0x200 != 0 {
        d |= libc::POLLWRBAND;
    }
    d
}

fn linux_poll_events(darwin: i16, requested: i16) -> i16 {
    let mut l = darwin & 0xff & !libc::POLLWRBAND;
    // Darwin's POLLWRNORM is POLLOUT; report it only if the guest asked for it.
    if darwin & libc::POLLOUT != 0 && requested & 0x100 != 0 {
        l |= 0x100;
    }
    if darwin & libc::POLLWRBAND != 0 {
        l |= 0x200;
    }
    l
}

fn timeval(tv: libc::timeval) -> (i64, i64) {
    (tv.tv_sec, tv.tv_usec as i64)
}

fn open_flags(linux: i32) -> Result<i32, i32> {
    use linux_o::*;
    if linux & (PATH | TMPFILE) != 0 {
        return Err(errno::EINVAL);
    }
    // Same access-mode encoding on both systems.
    let mut f = linux & ACCMODE;
    let table = [
        (CREAT, libc::O_CREAT),
        (EXCL, libc::O_EXCL),
        (NOCTTY, libc::O_NOCTTY),
        (TRUNC, libc::O_TRUNC),
        (APPEND, libc::O_APPEND),
        (NONBLOCK, libc::O_NONBLOCK),
        (DIRECTORY, libc::O_DIRECTORY),
        (NOFOLLOW, libc::O_NOFOLLOW),
        (CLOEXEC, libc::O_CLOEXEC),
    ];
    for (l, d) in table {
        if linux & l != 0 {
            f |= d;
        }
    }
    if linux & SYNC == SYNC {
        f |= libc::O_SYNC;
    } else if linux & DSYNC != 0 {
        f |= libc::O_DSYNC;
    }
    // DIRECT, LARGEFILE, NOATIME: hints with no Darwin equivalent, safe to drop.
    let _ = (DIRECT, LARGEFILE, NOATIME);
    Ok(f)
}

/// Darwin file status flags -> Linux, for `F_GETFL`.
fn linux_status_flags(darwin: i32) -> i32 {
    let mut f = darwin & libc::O_ACCMODE;
    for (d, l) in [
        (libc::O_APPEND, linux_o::APPEND),
        (libc::O_NONBLOCK, linux_o::NONBLOCK),
        (libc::O_DSYNC, linux_o::DSYNC),
    ] {
        if darwin & d != 0 {
            f |= l;
        }
    }
    if darwin & libc::O_SYNC != 0 {
        f |= linux_o::SYNC;
    }
    f
}

fn dirfd(linux: i32) -> i32 {
    if linux == AT_FDCWD { libc::AT_FDCWD } else { linux }
}

/// Darwin errno -> Linux errno. Values 1–34 agree except 11 (Darwin `EDEADLK`, Linux
/// `EAGAIN`).
pub(super) fn linux_errno(e: i32) -> i32 {
    match e {
        libc::EDEADLK => 35,
        libc::EAGAIN => 11,
        1..=34 => e,
        libc::EINPROGRESS => 115,
        libc::EALREADY => 114,
        libc::ENOTSOCK => 88,
        libc::EDESTADDRREQ => 89,
        libc::EMSGSIZE => 90,
        libc::EPROTOTYPE => 91,
        libc::ENOPROTOOPT => 92,
        libc::EPROTONOSUPPORT => 93,
        libc::ENOTSUP => 95,
        libc::EAFNOSUPPORT => 97,
        libc::EADDRINUSE => 98,
        libc::EADDRNOTAVAIL => 99,
        libc::ENETDOWN => 100,
        libc::ENETUNREACH => 101,
        libc::ECONNABORTED => 103,
        libc::ECONNRESET => 104,
        libc::ENOBUFS => 105,
        libc::EISCONN => 106,
        libc::ENOTCONN => 107,
        libc::ETIMEDOUT => 110,
        libc::ECONNREFUSED => 111,
        libc::EHOSTUNREACH => 113,
        libc::ELOOP => 40,
        libc::ENAMETOOLONG => 36,
        libc::ENOTEMPTY => 39,
        libc::EDQUOT => 122,
        libc::ENOLCK => 37,
        libc::ENOSYS => 38,
        libc::EOVERFLOW => 75,
        libc::ECANCELED => 125,
        libc::EILSEQ => 84,
        _ => errno::EIO,
    }
}

fn last_errno() -> i32 {
    linux_errno(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
}

/// Convert a `-1`-on-error return into a [`SysResult`].
pub(super) fn check(ret: isize) -> SysResult {
    if ret < 0 { Err(last_errno()) } else { Ok(ret as u64) }
}

fn to_stat(st: &libc::stat) -> Stat {
    Stat {
        dev: st.st_dev as u64,
        ino: st.st_ino,
        mode: st.st_mode as u32,
        nlink: st.st_nlink as u32,
        uid: st.st_uid,
        gid: st.st_gid,
        rdev: st.st_rdev as u64,
        size: st.st_size,
        blksize: st.st_blksize,
        blocks: st.st_blocks,
        atime: (st.st_atime, st.st_atime_nsec),
        mtime: (st.st_mtime, st.st_mtime_nsec),
        ctime: (st.st_ctime, st.st_ctime_nsec),
    }
}

impl HostOs for DarwinHost {
    // Guest buffers go straight to the kernel, which fails with EFAULT instead of crashing
    // if another guest thread unmaps them mid-call.

    fn epoll_create1(&self, flags: i32) -> SysResult {
        super::kqueue::epoll_create1(flags)
    }

    fn epoll_ctl(&self, epfd: i32, op: i32, fd: i32, event: (u32, u64)) -> SysResult {
        super::kqueue::epoll_ctl(epfd, op, fd, event)
    }

    fn epoll_wait(&self, epfd: i32, max: usize, timeout_ms: i32) -> Result<Vec<(u32, u64)>, i32> {
        super::kqueue::epoll_wait(epfd, max, timeout_ms)
    }

    fn eventfd(&self, initval: u32, flags: i32) -> SysResult {
        self.eventfds.create(initval, flags)
    }

    fn read(&self, fd: i32, buf: GuestBuf) -> SysResult {
        if let Some(r) = self.eventfds.read(fd, buf) {
            return r;
        }
        // SAFETY: the kernel validates the destination.
        check(unsafe { libc::read(fd, buf.ptr.cast(), buf.len) })
    }

    fn write(&self, fd: i32, buf: GuestBuf) -> SysResult {
        if let Some(r) = self.eventfds.write(fd, buf) {
            return r;
        }
        // SAFETY: the kernel validates the source.
        check(unsafe { libc::write(fd, buf.ptr.cast(), buf.len) })
    }

    fn readv(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        // SAFETY: GuestBuf is layout-compatible with struct iovec; the kernel validates.
        check(unsafe { libc::readv(fd, bufs.as_ptr().cast(), bufs.len() as i32) })
    }

    fn writev(&self, fd: i32, bufs: &[GuestBuf]) -> SysResult {
        // SAFETY: as in readv.
        check(unsafe { libc::writev(fd, bufs.as_ptr().cast(), bufs.len() as i32) })
    }

    fn pread(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        // SAFETY: the kernel validates the destination.
        check(unsafe { libc::pread(fd, buf.ptr.cast(), buf.len, offset) })
    }

    fn openat(&self, fd: i32, path: &CStr, flags: i32, mode: u32) -> SysResult {
        let flags = open_flags(flags)?;
        // SAFETY: path is NUL-terminated.
        check(
            unsafe { libc::openat(dirfd(fd), path.as_ptr(), flags, mode as libc::c_uint) } as isize
        )
    }

    fn close(&self, fd: i32) -> SysResult {
        self.eventfds.close(fd);
        // SAFETY: closing an arbitrary fd is memory-safe.
        check(unsafe { libc::close(fd) } as isize)
    }

    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> SysResult {
        const F_DUPFD: i32 = 0;
        const F_GETFD: i32 = 1;
        const F_SETFD: i32 = 2;
        const F_GETFL: i32 = 3;
        const F_SETFL: i32 = 4;
        const F_DUPFD_CLOEXEC: i32 = 1030;
        // SAFETY: none of these commands take a pointer argument.
        let call = |dcmd: i32, darg: i32| check(unsafe { libc::fcntl(fd, dcmd, darg) } as isize);
        match cmd {
            F_DUPFD => call(libc::F_DUPFD, arg as i32),
            F_DUPFD_CLOEXEC => call(libc::F_DUPFD_CLOEXEC, arg as i32),
            // FD_CLOEXEC is 1 on both systems.
            F_GETFD => call(libc::F_GETFD, 0),
            F_SETFD => call(libc::F_SETFD, arg as i32),
            F_GETFL => call(libc::F_GETFL, 0).map(|f| linux_status_flags(f as i32) as u64),
            // An eventfd's host fd stays nonblocking; only the guest's view changes.
            F_SETFL if self.eventfds.set_nonblock(fd, arg as i32 & linux_o::NONBLOCK != 0) => Ok(0),
            F_SETFL => call(libc::F_SETFL, open_flags(arg as i32)?),
            _ => Err(errno::EINVAL),
        }
    }

    fn lseek(&self, fd: i32, offset: i64, whence: i32) -> SysResult {
        let whence = match whence {
            0..=2 => whence, // SEEK_SET/CUR/END agree
            3 => libc::SEEK_DATA,
            4 => libc::SEEK_HOLE,
            _ => return Err(errno::EINVAL),
        };
        // SAFETY: no pointers involved.
        check(unsafe { libc::lseek(fd, offset, whence) } as isize)
    }

    fn fstat(&self, fd: i32) -> Result<Stat, i32> {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: st is a valid out-pointer.
        check(unsafe { libc::fstat(fd, st.as_mut_ptr()) } as isize)?;
        // SAFETY: fstat succeeded, so st is initialized.
        Ok(to_stat(&unsafe { st.assume_init() }))
    }

    fn fstatat(&self, fd: i32, path: &CStr, flags: i32) -> Result<Stat, i32> {
        if flags & LINUX_AT_EMPTY_PATH != 0 && path.is_empty() {
            return self.fstat(fd);
        }
        let mut dflags = 0;
        if flags & LINUX_AT_SYMLINK_NOFOLLOW != 0 {
            dflags |= libc::AT_SYMLINK_NOFOLLOW;
        }
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: path is NUL-terminated and st is a valid out-pointer.
        check(unsafe { libc::fstatat(dirfd(fd), path.as_ptr(), st.as_mut_ptr(), dflags) } as isize)?;
        // SAFETY: fstatat succeeded, so st is initialized.
        Ok(to_stat(&unsafe { st.assume_init() }))
    }

    fn clock_gettime(&self, clock_id: i32) -> Result<(i64, i64), i32> {
        let clock = match clock_id {
            0 | 5 => libc::CLOCK_REALTIME,       // REALTIME, REALTIME_COARSE
            1 | 6 => libc::CLOCK_MONOTONIC,      // MONOTONIC, MONOTONIC_COARSE
            4 => libc::CLOCK_MONOTONIC_RAW,      // MONOTONIC_RAW
            7 => libc::CLOCK_MONOTONIC,          // BOOTTIME (Darwin's MONOTONIC counts sleep)
            2 => libc::CLOCK_PROCESS_CPUTIME_ID, // PROCESS_CPUTIME_ID
            3 => libc::CLOCK_THREAD_CPUTIME_ID,  // THREAD_CPUTIME_ID
            _ => return Err(errno::EINVAL),
        };
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: ts is a valid out-pointer.
        check(unsafe { libc::clock_gettime(clock, &mut ts) } as isize)?;
        Ok((ts.tv_sec, ts.tv_nsec))
    }

    fn getrandom(&self, buf: GuestBuf) -> SysResult {
        // getentropy is a system call (so a bad buffer is EFAULT, not a crash) limited to
        // 256 bytes per call.
        let mut done = 0;
        while done < buf.len {
            let n = (buf.len - done).min(256);
            // SAFETY: the kernel validates the destination.
            check(unsafe { libc::getentropy(buf.ptr.add(done).cast(), n) } as isize)?;
            done += n;
        }
        Ok(buf.len as u64)
    }

    fn winsize(&self, fd: i32) -> Result<[u16; 4], i32> {
        let mut ws = libc::winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
        // SAFETY: TIOCGWINSZ writes a struct winsize through the pointer.
        check(unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) } as isize)?;
        Ok([ws.ws_row, ws.ws_col, ws.ws_xpixel, ws.ws_ypixel])
    }

    fn pwrite(&self, fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
        // SAFETY: the kernel validates the source.
        check(unsafe { libc::pwrite(fd, buf.ptr.cast(), buf.len, offset) })
    }

    fn unlinkat(&self, fd: i32, path: &CStr, flags: i32) -> SysResult {
        if flags & !LINUX_AT_REMOVEDIR != 0 {
            return Err(errno::EINVAL);
        }
        let dflags = if flags & LINUX_AT_REMOVEDIR != 0 { libc::AT_REMOVEDIR } else { 0 };
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::unlinkat(dirfd(fd), path.as_ptr(), dflags) } as isize)
    }

    fn renameat2(&self, ofd: i32, old: &CStr, nfd: i32, new: &CStr, flags: u32) -> SysResult {
        const NOREPLACE: u32 = 1;
        const EXCHANGE: u32 = 2;
        let dflags = match flags {
            0 => 0,
            NOREPLACE => libc::RENAME_EXCL,
            EXCHANGE => libc::RENAME_SWAP,
            _ => return Err(errno::EINVAL),
        };
        // SAFETY: both paths are NUL-terminated.
        check(unsafe {
            libc::renameatx_np(dirfd(ofd), old.as_ptr(), dirfd(nfd), new.as_ptr(), dflags)
        } as isize)
    }

    fn mkdirat(&self, fd: i32, path: &CStr, mode: u32) -> SysResult {
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::mkdirat(dirfd(fd), path.as_ptr(), mode as libc::mode_t) } as isize)
    }

    fn faccessat(&self, fd: i32, path: &CStr, mode: i32, flags: i32) -> SysResult {
        if flags & LINUX_AT_EMPTY_PATH != 0 && path.is_empty() {
            return self.fstat(fd).map(|_| 0); // existence check of the fd itself
        }
        let mut dflags = 0;
        if flags & LINUX_AT_EACCESS != 0 {
            dflags |= libc::AT_EACCESS;
        }
        if flags & LINUX_AT_SYMLINK_NOFOLLOW != 0 {
            dflags |= libc::AT_SYMLINK_NOFOLLOW;
        }
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::faccessat(dirfd(fd), path.as_ptr(), mode, dflags) } as isize)
    }

    fn readlinkat(&self, fd: i32, path: &CStr, buf: GuestBuf) -> SysResult {
        // SAFETY: path is NUL-terminated; the kernel validates the destination.
        check(unsafe { libc::readlinkat(dirfd(fd), path.as_ptr(), buf.ptr.cast(), buf.len) })
    }

    fn getcwd(&self) -> Result<Vec<u8>, i32> {
        use std::os::unix::ffi::OsStringExt;
        std::env::current_dir()
            .map(|p| p.into_os_string().into_vec())
            .map_err(|e| linux_errno(e.raw_os_error().unwrap_or(libc::EIO)))
    }

    fn chdir(&self, path: &CStr) -> SysResult {
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::chdir(path.as_ptr()) } as isize)
    }

    fn fchdir(&self, fd: i32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::fchdir(fd) } as isize)
    }

    fn dup(&self, fd: i32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::dup(fd) } as isize)
    }

    fn dup3(&self, old: i32, new: i32, flags: i32) -> SysResult {
        if old == new || flags & !linux_o::CLOEXEC != 0 {
            return Err(errno::EINVAL);
        }
        // SAFETY: no pointers.
        let fd = check(unsafe { libc::dup2(old, new) } as isize)? as i32;
        apply_fd_flags(fd, flags)?;
        Ok(fd as u64)
    }

    fn pipe2(&self, flags: i32) -> Result<[i32; 2], i32> {
        if flags & !(linux_o::CLOEXEC | linux_o::NONBLOCK) != 0 {
            return Err(errno::EINVAL);
        }
        let mut fds = [0i32; 2];
        // SAFETY: fds has room for two descriptors.
        check(unsafe { libc::pipe(fds.as_mut_ptr()) } as isize)?;
        for fd in fds {
            apply_fd_flags(fd, flags)?;
        }
        Ok(fds)
    }

    fn ftruncate(&self, fd: i32, len: i64) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::ftruncate(fd, len) } as isize)
    }

    fn fsync(&self, fd: i32, _data_only: bool) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::fsync(fd) } as isize)
    }

    fn fchmod(&self, fd: i32, mode: u32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::fchmod(fd, mode as libc::mode_t) } as isize)
    }

    fn fchmodat(&self, fd: i32, path: &CStr, mode: u32, flags: i32) -> SysResult {
        let dflags =
            if flags & LINUX_AT_SYMLINK_NOFOLLOW != 0 { libc::AT_SYMLINK_NOFOLLOW } else { 0 };
        // SAFETY: path is NUL-terminated.
        check(unsafe { libc::fchmodat(dirfd(fd), path.as_ptr(), mode as libc::mode_t, dflags) }
            as isize)
    }

    fn umask(&self, mask: u32) -> u32 {
        // SAFETY: no pointers.
        unsafe { libc::umask(mask as libc::mode_t) as u32 }
    }

    fn lock(&self, fd: i32, cmd: i32, lock: &mut Flock) -> SysResult {
        let dcmd = match cmd {
            5 => libc::F_GETLK,
            6 => libc::F_SETLK,
            7 => libc::F_SETLKW,
            _ => return Err(errno::EINVAL),
        };
        let dtype = match lock.typ {
            0 => libc::F_RDLCK,
            1 => libc::F_WRLCK,
            2 => libc::F_UNLCK,
            _ => return Err(errno::EINVAL),
        };
        let mut fl = libc::flock {
            l_start: lock.start,
            l_len: lock.len,
            l_pid: lock.pid,
            l_type: dtype,
            l_whence: lock.whence,
        };
        // SAFETY: fl is a valid struct flock for the duration of the call.
        let r = check(unsafe { libc::fcntl(fd, dcmd, &mut fl) } as isize)?;
        if cmd == 5 {
            lock.typ = match fl.l_type {
                libc::F_RDLCK => 0,
                libc::F_WRLCK => 1,
                _ => 2,
            };
            (lock.whence, lock.start, lock.len, lock.pid) =
                (fl.l_whence, fl.l_start, fl.l_len, fl.l_pid);
        }
        Ok(r)
    }

    fn nanosleep(&self, (sec, nsec): (i64, i64)) -> SysResult {
        let mut req = libc::timespec { tv_sec: sec, tv_nsec: nsec };
        let mut rem = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // Host signals (none reach the guest yet) just restart the remaining sleep.
        // SAFETY: valid timespec pointers.
        while unsafe { libc::nanosleep(&req, &mut rem) } != 0 {
            if last_errno() != 4 {
                return Err(last_errno());
            }
            req = rem;
        }
        Ok(0)
    }

    fn getrusage(&self, who: i32) -> Result<Rusage, i32> {
        let dwho = match who {
            0 => libc::RUSAGE_SELF,
            -1 => libc::RUSAGE_CHILDREN,
            _ => return Err(errno::EINVAL),
        };
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: valid out-pointer.
        check(unsafe { libc::getrusage(dwho, ru.as_mut_ptr()) } as isize)?;
        // SAFETY: getrusage succeeded.
        let ru = unsafe { ru.assume_init() };
        Ok(Rusage {
            utime: timeval(ru.ru_utime),
            stime: timeval(ru.ru_stime),
            maxrss_kib: ru.ru_maxrss / 1024, // Darwin reports bytes, Linux KiB
            minflt: ru.ru_minflt,
            majflt: ru.ru_majflt,
            nvcsw: ru.ru_nvcsw,
            nivcsw: ru.ru_nivcsw,
        })
    }

    fn getrlimit(&self, resource: u32) -> Result<(u64, u64), i32> {
        if resource > 15 {
            return Err(errno::EINVAL);
        }
        let Some(r) = darwin_rlimit(resource) else { return Ok((u64::MAX, u64::MAX)) };
        let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        // SAFETY: valid out-pointer.
        check(unsafe { libc::getrlimit(r, &mut rl) } as isize)?;
        let lim = |v: libc::rlim_t| if v == libc::RLIM_INFINITY { u64::MAX } else { v };
        Ok((lim(rl.rlim_cur), lim(rl.rlim_max)))
    }

    fn poll(&self, fds: &mut [PollFd], timeout: Option<(i64, i64)>) -> SysResult {
        let mut host: Vec<libc::pollfd> = fds
            .iter()
            .map(|f| libc::pollfd { fd: f.fd, events: darwin_poll_events(f.events), revents: 0 })
            .collect();
        // Round up to whole milliseconds so short timeouts do not become busy polls.
        let ms = timeout.map_or(-1, |(s, ns)| {
            (s.saturating_mul(1000).saturating_add((ns + 999_999) / 1_000_000)).min(i32::MAX as i64)
                as i32
        });
        // SAFETY: `host` is a valid array of `host.len()` pollfds.
        let n = check(
            unsafe { libc::poll(host.as_mut_ptr(), host.len() as libc::nfds_t, ms) } as isize
        )?;
        for (f, h) in fds.iter_mut().zip(&host) {
            f.revents = linux_poll_events(h.revents, f.events);
        }
        Ok(n)
    }

    fn kill(&self, pid: i32, sig: i32) -> SysResult {
        let sig = darwin_signal(sig).ok_or(errno::EINVAL)?;
        // SAFETY: no pointers.
        check(unsafe { libc::kill(pid, sig) } as isize)
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

    #[test]
    fn open_flags_translate() {
        assert_eq!(open_flags(0), Ok(libc::O_RDONLY));
        assert_eq!(
            open_flags(1 | linux_o::CREAT | linux_o::TRUNC | linux_o::CLOEXEC),
            Ok(libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_CLOEXEC)
        );
        assert_eq!(open_flags(linux_o::DIRECTORY | linux_o::LARGEFILE), Ok(libc::O_DIRECTORY));
        assert_eq!(open_flags(linux_o::SYNC), Ok(libc::O_SYNC));
        assert_eq!(open_flags(linux_o::PATH), Err(errno::EINVAL));
    }

    #[test]
    fn errno_translates() {
        assert_eq!(linux_errno(libc::ENOENT), 2);
        assert_eq!(linux_errno(libc::EAGAIN), 11);
        assert_eq!(linux_errno(libc::EDEADLK), 35);
        assert_eq!(linux_errno(libc::ENOSYS), 38);
        assert_eq!(linux_errno(libc::ENOTEMPTY), 39);
    }

    #[test]
    fn opens_reads_and_stats_files() {
        let dir = std::env::temp_dir().join(format!("champollion-host-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, b"hello").unwrap();
        let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let h = DarwinHost::default();
        let fd = h.openat(AT_FDCWD, &cpath, 0, 0).unwrap() as i32;
        let st = h.fstat(fd).unwrap();
        assert_eq!(st.size, 5);
        assert_eq!(st.mode & 0o170000, 0o100000); // S_IFREG
        let mut buf = [0u8; 8];
        assert_eq!(h.read(fd, GuestBuf { ptr: buf.as_mut_ptr(), len: buf.len() }), Ok(5));
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(h.lseek(fd, 1, 0), Ok(1));
        assert_eq!(h.fcntl(fd, 2, 1), Ok(0)); // F_SETFD FD_CLOEXEC
        assert_eq!(h.fcntl(fd, 1, 0), Ok(1)); // F_GETFD
        assert_eq!(h.fcntl(fd, 3, 0), Ok(0)); // F_GETFL: O_RDONLY
        let dup = h.fcntl(fd, 1030, 10).unwrap() as i32; // F_DUPFD_CLOEXEC
        assert!(dup >= 10);
        h.close(dup).unwrap();
        h.close(fd).unwrap();
        assert_eq!(h.close(fd), Err(errno::EBADF));

        let missing = std::ffi::CString::new(dir.join("nope").to_str().unwrap()).unwrap();
        assert_eq!(h.openat(AT_FDCWD, &missing, 0, 0), Err(errno::ENOENT));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
