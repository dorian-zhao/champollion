//! Direct mode's backend: serve Linux syscalls with Darwin (macOS) calls.
//!
//! Guest file descriptors are host file descriptors. Every Linux-specific value coming in
//! (open flags, `AT_*` constants, `whence`, clock ids) is translated to its Darwin
//! counterpart, and every Darwin errno going out is translated to Linux.

use std::ffi::CStr;

use crate::memory::GuestBuf;
use crate::syscall::{
    AT_FDCWD, Flock, HostOs, Identity, PollFd, Rusage, Stat, SysResult, TimeSpec, errno,
};

/// The Darwin backend. Guest fds are host fds; the only state kept here is for what Darwin
/// lacks: eventfds (epoll instances are plain kqueues) and directory streams for
/// `getdents64`.
#[derive(Default)]
pub struct DarwinHost {
    eventfds: super::kqueue::EventFds,
    dirs: super::dirs::Dirs,
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

/// Darwin signal number -> Linux.
pub(super) fn linux_signal(sig: i32) -> i32 {
    (1..=31).find(|&l| darwin_signal(l) == Some(sig)).unwrap_or(sig)
}

/// A Darwin `statfs` as Linux's riscv64 `struct statfs`: type, bsize, blocks, bfree, bavail,
/// files, ffree (i64 each), fsid (2 x i32), namelen, frsize, flags, spare[4].
fn linux_statfs(st: &libc::statfs) -> Vec<u8> {
    // Linux magic numbers for the file system types a Mac has.
    let fs_type: i64 =
        match unsafe { std::ffi::CStr::from_ptr(st.f_fstypename.as_ptr()) }.to_bytes() {
            b"apfs" | b"hfs" => 0x1badface, // reported as "other"; there is no Linux magic for APFS
            b"msdos" => 0x4d44,
            b"nfs" => 0x6969,
            b"smbfs" => 0xff534d42u32 as i64,
            b"devfs" => 0x1373,
            _ => 0x1badface,
        };
    let mut b = Vec::with_capacity(120);
    for v in [
        fs_type,
        st.f_bsize as i64,
        st.f_blocks as i64,
        st.f_bfree as i64,
        st.f_bavail as i64,
        st.f_files as i64,
        st.f_ffree as i64,
    ] {
        b.extend(v.to_le_bytes());
    }
    // SAFETY: fsid_t is two i32s (its field is private in the libc crate).
    let fsid: [i32; 2] = unsafe { std::mem::transmute(st.f_fsid) };
    b.extend(fsid[0].to_le_bytes());
    b.extend(fsid[1].to_le_bytes());
    for v in [255i64, st.f_bsize as i64, 0, 0, 0, 0, 0] {
        b.extend(v.to_le_bytes());
    }
    b
}

// Public since macOS 14.4 (<os/os_sync_wait_on_address.h>): futexes that work across
// processes on shared memory.
unsafe extern "C" {
    fn os_sync_wait_on_address(
        addr: *mut libc::c_void,
        value: u64,
        size: usize,
        flags: u32,
    ) -> libc::c_int;
    fn os_sync_wait_on_address_with_timeout(
        addr: *mut libc::c_void,
        value: u64,
        size: usize,
        flags: u32,
        clockid: u32,
        timeout_ns: u64,
    ) -> libc::c_int;
    fn os_sync_wake_by_address_any(addr: *mut libc::c_void, size: usize, flags: u32)
    -> libc::c_int;
    fn os_sync_wake_by_address_all(addr: *mut libc::c_void, size: usize, flags: u32)
    -> libc::c_int;
}

/// A fixed-size `sysctl` value by name.
fn sysctl<T: Copy>(name: &str) -> Option<T> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let mut len = std::mem::size_of::<T>();
    // SAFETY: the kernel writes at most `len` bytes into `value`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    // SAFETY: fully written when the call succeeds with the size we asked for.
    (rc == 0 && len == std::mem::size_of::<T>()).then(|| unsafe { value.assume_init() })
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

    fn pidfd_open(&self, pid: i32, flags: u32) -> SysResult {
        super::kqueue::pidfd_open(pid, flags)
    }

    fn eventfd(&self, initval: u32, flags: i32) -> SysResult {
        self.eventfds.create(initval, flags)
    }

    fn getdents64(&self, fd: i32, max: usize) -> Result<Vec<u8>, i32> {
        self.dirs.getdents64(fd, max)
    }

    fn wait4(&self, pid: i32, options: i32) -> Result<(u64, i32), i32> {
        const WNOHANG: i32 = 1;
        const WUNTRACED: i32 = 2;
        const WCONTINUED: i32 = 8;
        let mut flags = 0;
        for (linux, darwin) in
            [(WNOHANG, libc::WNOHANG), (WUNTRACED, libc::WUNTRACED), (WCONTINUED, libc::WCONTINUED)]
        {
            if options & linux != 0 {
                flags |= darwin;
            }
        }
        let mut status = 0;
        // SAFETY: wait4 into a local status; no rusage.
        let got =
            check(unsafe { libc::wait4(pid, &mut status, flags, std::ptr::null_mut()) } as isize)?;
        // The encoding agrees (exit status << 8, or the signal in the low 7 bits, 0x7f for a
        // stop); only signal numbers differ.
        let sig = status & 0x7f;
        let status = if libc::WIFSIGNALED(status) {
            (status & !0x7f) | linux_signal(sig)
        } else if libc::WIFSTOPPED(status) {
            (linux_signal((status >> 8) & 0xff) << 8) | 0x7f
        } else {
            status
        };
        Ok((got, status))
    }

    fn socket(&self, domain: i32, ty: i32, protocol: i32) -> SysResult {
        super::socket::socket(domain, ty, protocol)
    }

    fn socketpair(&self, domain: i32, ty: i32, protocol: i32) -> Result<[i32; 2], i32> {
        super::socket::socketpair(domain, ty, protocol)
    }

    fn bind(&self, fd: i32, addr: &[u8]) -> SysResult {
        super::socket::bind(fd, addr)
    }

    fn connect(&self, fd: i32, addr: &[u8]) -> SysResult {
        super::socket::connect(fd, addr)
    }

    fn listen(&self, fd: i32, backlog: i32) -> SysResult {
        // SAFETY: plain syscall.
        check(unsafe { libc::listen(fd, backlog) } as isize)
    }

    fn accept4(&self, fd: i32, flags: i32) -> Result<(u64, Vec<u8>), i32> {
        super::socket::accept4(fd, flags)
    }

    fn sockname(&self, fd: i32, peer: bool) -> Result<Vec<u8>, i32> {
        super::socket::name(fd, peer)
    }

    fn setsockopt(&self, fd: i32, level: i32, name: i32, value: &[u8]) -> SysResult {
        super::socket::setsockopt(fd, level, name, value)
    }

    fn getsockopt(&self, fd: i32, level: i32, name: i32, len: usize) -> Result<Vec<u8>, i32> {
        super::socket::getsockopt(fd, level, name, len)
    }

    fn sendmsg(&self, fd: i32, to: Option<&[u8]>, bufs: &[GuestBuf], flags: i32) -> SysResult {
        let addr = to.map(super::socket::to_darwin).transpose()?;
        // SAFETY: GuestBuf is layout-compatible with iovec; the kernel validates buffers.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        if let Some((ss, len)) = &addr {
            msg.msg_name = (ss as *const libc::sockaddr_storage).cast_mut().cast();
            msg.msg_namelen = *len;
        }
        msg.msg_iov = bufs.as_ptr().cast_mut().cast();
        msg.msg_iovlen = bufs.len() as i32;
        check(unsafe { libc::sendmsg(fd, &msg, super::socket::msg_flags_in(flags)) })
    }

    fn recvmsg(&self, fd: i32, bufs: &[GuestBuf], flags: i32) -> Result<(u64, Vec<u8>, i32), i32> {
        // SAFETY: as in `sendmsg`, receiving the address into a local storage.
        let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_name = (&raw mut ss).cast();
        msg.msg_namelen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        msg.msg_iov = bufs.as_ptr().cast_mut().cast();
        msg.msg_iovlen = bufs.len() as i32;
        let n = check(unsafe { libc::recvmsg(fd, &mut msg, super::socket::msg_flags_in(flags)) })?;
        let from = if msg.msg_namelen > 0 {
            super::socket::to_linux(&ss, msg.msg_namelen)
        } else {
            Vec::new()
        };
        Ok((n, from, super::socket::msg_flags_out(msg.msg_flags)))
    }

    fn shutdown(&self, fd: i32, how: i32) -> SysResult {
        // SAFETY: plain syscall (SHUT_RD/WR/RDWR agree).
        check(unsafe { libc::shutdown(fd, how) } as isize)
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

    fn sendfile(&self, out_fd: i32, in_fd: i32, offset: Option<u64>, count: usize) -> SysResult {
        if count == 0 {
            return Ok(0);
        }
        let start = match offset {
            Some(start) => start,
            None => {
                // SAFETY: no pointers.
                let pos = unsafe { libc::lseek(in_fd, 0, libc::SEEK_CUR) };
                if pos < 0 {
                    return Err(last_errno());
                }
                pos as u64
            }
        };
        let mut buf = vec![0u8; count.clamp(1, 64 * 1024)];
        let mut done = 0u64;
        let mut error = None;
        while (done as usize) < count {
            let Some(at) = start.checked_add(done).filter(|&v| v <= i64::MAX as u64) else {
                if done == 0 {
                    return Err(errno::EINVAL);
                }
                break;
            };
            let want = (count - done as usize).min(buf.len());
            // SAFETY: buf is valid for `want` bytes; the host validates the file offsets.
            let n = unsafe { libc::pread(in_fd, buf.as_mut_ptr().cast(), want, at as libc::off_t) };
            if n < 0 {
                let e = last_errno();
                if e == errno::EINTR {
                    continue;
                }
                if done == 0 {
                    return Err(e);
                }
                break;
            }
            if n == 0 {
                break;
            }
            let mut written = 0usize;
            while written < n as usize {
                // SAFETY: the source range is inside `buf`; the kernel validates out_fd.
                let w = unsafe {
                    libc::write(out_fd, buf[written..].as_ptr().cast(), n as usize - written)
                };
                if w < 0 {
                    let e = last_errno();
                    if e == errno::EINTR {
                        continue;
                    }
                    error = Some(e);
                    break;
                }
                if w == 0 {
                    break;
                }
                written += w as usize;
            }
            done += written as u64;
            if written < n as usize || error.is_some() {
                break;
            }
        }
        if offset.is_none() && done > 0 {
            // SAFETY: no pointers; the input fd is seekable when sendfile is meaningful.
            let ret = unsafe { libc::lseek(in_fd, done as libc::off_t, libc::SEEK_CUR) };
            if ret < 0 {
                return Err(last_errno());
            }
        }
        if done == 0
            && let Some(e) = error
        {
            return Err(e);
        }
        Ok(done)
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

    fn openat2(&self, fd: i32, path: &CStr, flags: i32, mode: u64, resolve: u64) -> SysResult {
        // Darwin has no `RESOLVE_*` resolution. When the guest does not ask for it (tar's
        // probe does not), the call is a plain `openat`; otherwise `ENOSYS` lets the
        // caller fall back.
        if resolve == 0 && mode <= u32::MAX as u64 {
            self.openat(fd, path, flags, mode as u32)
        } else {
            Err(errno::ENOSYS)
        }
    }

    fn close(&self, fd: i32) -> SysResult {
        self.eventfds.close(fd);
        self.dirs.forget(fd);
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
        self.dirs.forget(fd); // a directory stream restarts at the new offset
        let whence = match whence {
            0..=2 => whence, // SEEK_SET/CUR/END agree
            3 => libc::SEEK_DATA,
            4 => libc::SEEK_HOLE,
            _ => return Err(errno::EINVAL),
        };
        // SAFETY: no pointers involved.
        check(unsafe { libc::lseek(fd, offset, whence) } as isize)
    }

    fn futex_wait_shared(
        &self,
        word: GuestBuf,
        expected: u32,
        timeout: Option<std::time::Duration>,
    ) -> SysResult {
        const SHARED: u32 = 1; // OS_SYNC_WAIT_ON_ADDRESS_SHARED
        const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;
        let addr = word.ptr.cast::<u32>();
        // SAFETY: a validated, aligned guest word; read atomically.
        let current = unsafe { std::sync::atomic::AtomicU32::from_ptr(addr) }
            .load(std::sync::atomic::Ordering::SeqCst);
        if current != expected {
            return Err(errno::EAGAIN);
        }
        // SAFETY: the kernel compares and sleeps on the word; a guest signal interrupts it
        // with EINTR (see Process::interrupt).
        let rc = unsafe {
            match timeout {
                None => os_sync_wait_on_address(addr.cast(), expected as u64, 4, SHARED),
                Some(t) => os_sync_wait_on_address_with_timeout(
                    addr.cast(),
                    expected as u64,
                    4,
                    SHARED,
                    OS_CLOCK_MACH_ABSOLUTE_TIME,
                    t.as_nanos().clamp(1, u64::MAX as u128) as u64,
                ),
            }
        };
        if rc < 0 { Err(last_errno()) } else { Ok(0) }
    }

    fn futex_wake_shared(&self, word: GuestBuf, count: u32) -> SysResult {
        const SHARED: u32 = 1; // OS_SYNC_WAKE_BY_ADDRESS_SHARED
        let addr = word.ptr.cast::<libc::c_void>();
        if count >= i32::MAX as u32 {
            // SAFETY: waking by address has no memory effects.
            unsafe { os_sync_wake_by_address_all(addr, 4, SHARED) };
            return Ok(0); // the count of woken waiters is not reported for "all"
        }
        let mut woken = 0;
        // SAFETY: as above; ENOENT ends the loop when nobody is left waiting.
        while woken < count && unsafe { os_sync_wake_by_address_any(addr, 4, SHARED) } == 0 {
            woken += 1;
        }
        Ok(woken as u64)
    }

    fn bytes_readable(&self, fd: i32) -> SysResult {
        let mut n: libc::c_int = 0;
        // SAFETY: FIONREAD writes one int.
        check(unsafe { libc::ioctl(fd, libc::FIONREAD, &mut n) } as isize)?;
        Ok(n.max(0) as u64)
    }

    fn statfs(&self, path: &CStr) -> Result<Vec<u8>, i32> {
        let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: a valid path and out-pointer.
        check(unsafe { libc::statfs(path.as_ptr(), st.as_mut_ptr()) } as isize)?;
        // SAFETY: initialized by the successful call.
        let st = unsafe { st.assume_init() };
        Ok(linux_statfs(&st))
    }

    fn fstatfs(&self, fd: i32) -> Result<Vec<u8>, i32> {
        let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: a valid out-pointer.
        check(unsafe { libc::fstatfs(fd, st.as_mut_ptr()) } as isize)?;
        // SAFETY: initialized by the successful call.
        let st = unsafe { st.assume_init() };
        Ok(linux_statfs(&st))
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

    fn linkat(
        &self,
        olddirfd: i32,
        old: &CStr,
        newdirfd: i32,
        new: &CStr,
        flags: i32,
    ) -> SysResult {
        const LINUX_AT_SYMLINK_FOLLOW: i32 = 0x400;
        if flags & !LINUX_AT_SYMLINK_FOLLOW != 0 {
            return Err(errno::EINVAL);
        }
        let dflags = if flags & LINUX_AT_SYMLINK_FOLLOW != 0 { libc::AT_SYMLINK_FOLLOW } else { 0 };
        // SAFETY: NUL-terminated paths.
        check(unsafe {
            libc::linkat(dirfd(olddirfd), old.as_ptr(), dirfd(newdirfd), new.as_ptr(), dflags)
        } as isize)
    }

    fn symlinkat(&self, target: &CStr, newdirfd: i32, new: &CStr) -> SysResult {
        // SAFETY: NUL-terminated paths.
        check(unsafe { libc::symlinkat(target.as_ptr(), dirfd(newdirfd), new.as_ptr()) } as isize)
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

    fn fallocate(&self, fd: i32, mode: i32, offset: i64, len: i64) -> SysResult {
        if mode != 0 {
            return Err(errno::EOPNOTSUPP);
        }
        if offset < 0 || len < 0 {
            return Err(errno::EINVAL);
        }
        let end = offset.checked_add(len).ok_or(errno::EINVAL)?;
        let mut store = libc::fstore_t {
            fst_flags: libc::F_ALLOCATEALL,
            fst_posmode: libc::F_PEOFPOSMODE,
            fst_offset: 0,
            fst_length: end,
            fst_bytesalloc: 0,
        };
        // SAFETY: the fstore_t pointer is valid for the call.
        let r = unsafe { libc::fcntl(fd, libc::F_PREALLOCATE, &mut store) };
        if r < 0 {
            let e = last_errno();
            if e != errno::EOPNOTSUPP {
                return Err(e);
            }
        }
        // SAFETY: ftruncate has no pointer arguments.
        check(unsafe { libc::ftruncate(fd, end) } as isize).map(|_| 0)
    }

    fn copy_file_range(
        &self,
        fd_in: i32,
        off_in: Option<u64>,
        fd_out: i32,
        off_out: Option<u64>,
        len: usize,
    ) -> SysResult {
        if len == 0 {
            return Ok(0);
        }
        let start_in = match off_in {
            Some(v) => v,
            None => {
                // SAFETY: no pointers.
                let p = unsafe { libc::lseek(fd_in, 0, libc::SEEK_CUR) };
                if p < 0 {
                    return Err(last_errno());
                }
                p as u64
            }
        };
        let start_out = match off_out {
            Some(v) => v,
            None => {
                // SAFETY: no pointers.
                let p = unsafe { libc::lseek(fd_out, 0, libc::SEEK_CUR) };
                if p < 0 {
                    return Err(last_errno());
                }
                p as u64
            }
        };
        let mut buf = vec![0u8; len.clamp(1, 64 * 1024)];
        let mut done = 0u64;
        let mut error = None;
        while (done as usize) < len {
            let Some(at) = start_in.checked_add(done).filter(|&v| v <= i64::MAX as u64) else {
                if done == 0 {
                    return Err(errno::EINVAL);
                }
                break;
            };
            let want = (len - done as usize).min(buf.len());
            // SAFETY: buf is valid for want bytes.
            let n = unsafe { libc::pread(fd_in, buf.as_mut_ptr().cast(), want, at as libc::off_t) };
            if n < 0 {
                let e = last_errno();
                if e == errno::EINTR {
                    continue;
                }
                if done == 0 {
                    return Err(e);
                }
                break;
            }
            if n == 0 {
                break;
            }
            let mut written = 0usize;
            while written < n as usize {
                let Some(out_at) =
                    start_out.checked_add(done + written as u64).filter(|&v| v <= i64::MAX as u64)
                else {
                    error = Some(errno::EINVAL);
                    break;
                };
                // SAFETY: source range is inside buf.
                let w = unsafe {
                    libc::pwrite(
                        fd_out,
                        buf[written..].as_ptr().cast(),
                        n as usize - written,
                        out_at as libc::off_t,
                    )
                };
                if w < 0 {
                    let e = last_errno();
                    if e == errno::EINTR {
                        continue;
                    }
                    error = Some(e);
                    break;
                }
                if w == 0 {
                    break;
                }
                written += w as usize;
            }
            done += written as u64;
            if written < n as usize || error.is_some() {
                break;
            }
        }
        if off_in.is_none() && done > 0 {
            // SAFETY: no pointers.
            if unsafe { libc::lseek(fd_in, done as libc::off_t, libc::SEEK_CUR) } < 0 {
                return Err(last_errno());
            }
        }
        if off_out.is_none() && done > 0 {
            // SAFETY: no pointers.
            if unsafe { libc::lseek(fd_out, done as libc::off_t, libc::SEEK_CUR) } < 0 {
                return Err(last_errno());
            }
        }
        if done == 0
            && let Some(e) = error
        {
            return Err(e);
        }
        Ok(done)
    }

    fn close_range(&self, first: u32, last: u32, cloexec: bool) -> SysResult {
        if first > last {
            return Err(errno::EINVAL);
        }
        // SAFETY: getdtablesize has no arguments and no side effects.
        let limit = unsafe { libc::getdtablesize() }.max(0) as u32;
        if first >= limit {
            return Ok(0);
        }
        for fd in first..=last.min(limit - 1) {
            let fd = fd as i32;
            if cloexec {
                // SAFETY: fcntl reads a descriptor's flags.
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                if flags < 0 {
                    if last_errno() == errno::EBADF {
                        continue;
                    }
                    return Err(last_errno());
                }
                // SAFETY: no pointers.
                if unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
                    return Err(last_errno());
                }
            } else {
                // SAFETY: closing an arbitrary fd is safe; EBADF means it was already closed.
                unsafe { libc::close(fd) };
            }
        }
        Ok(0)
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

    fn utimensat(
        &self,
        fd: i32,
        path: Option<&CStr>,
        times: Option<[TimeSpec; 2]>,
        flags: i32,
    ) -> SysResult {
        let times = times.map(|times| {
            times.map(|time| match time {
                TimeSpec::At(sec, nsec) => {
                    libc::timespec { tv_sec: sec as libc::time_t, tv_nsec: nsec as libc::c_long }
                }
                TimeSpec::Now => libc::timespec { tv_sec: 0, tv_nsec: libc::UTIME_NOW },
                TimeSpec::Omit => libc::timespec { tv_sec: 0, tv_nsec: libc::UTIME_OMIT },
            })
        });
        let ptr = times.as_ref().map_or(std::ptr::null(), |times| times.as_ptr());
        match path {
            Some(path) => {
                let dflags = if flags & LINUX_AT_SYMLINK_NOFOLLOW != 0 {
                    libc::AT_SYMLINK_NOFOLLOW
                } else {
                    0
                };
                // SAFETY: path is NUL-terminated and times, when present, has two entries.
                check(unsafe { libc::utimensat(dirfd(fd), path.as_ptr(), ptr, dflags) } as isize)
            }
            // SAFETY: no path; futimens validates the descriptor and the optional array.
            None => check(unsafe { libc::futimens(fd, ptr) } as isize),
        }
    }

    fn umask(&self, mask: u32) -> u32 {
        // SAFETY: no pointers.
        unsafe { libc::umask(mask as libc::mode_t) as u32 }
    }

    fn flock(&self, fd: i32, op: i32) -> SysResult {
        // `LOCK_SH`/`LOCK_EX`/`LOCK_NB`/`LOCK_UN` agree between Linux and Darwin.
        // SAFETY: no pointers.
        check(unsafe { libc::flock(fd, op) } as isize)
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
        let req = libc::timespec { tv_sec: sec, tv_nsec: nsec };
        // A guest signal interrupts the sleep with EINTR (see Process::interrupt).
        // SAFETY: a valid timespec pointer.
        check(unsafe { libc::nanosleep(&req, std::ptr::null_mut()) } as isize)
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

    fn fadvise(&self, fd: i32, _offset: i64, _len: i64, advice: i32) -> SysResult {
        // Only a hint, with nothing like it on Darwin; check what Linux would.
        if !(0..=5).contains(&advice) {
            return Err(errno::EINVAL);
        }
        // SAFETY: no pointers.
        check(unsafe { libc::fcntl(fd, libc::F_GETFD) } as isize).map(|_| 0)
    }

    fn sysinfo(&self) -> Result<Vec<u8>, i32> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64);
        let boot: libc::timeval =
            sysctl("kern.boottime").unwrap_or(libc::timeval { tv_sec: now, tv_usec: 0 });
        let mut loads = [0f64; 3];
        // SAFETY: room for three samples.
        unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
        let page = sysctl::<u64>("hw.pagesize").unwrap_or(16384);
        let free = sysctl::<u32>("vm.page_free_count").unwrap_or(0) as u64 * page;
        let swap: libc::xsw_usage = sysctl("vm.swapusage").unwrap_or(libc::xsw_usage {
            xsu_total: 0,
            xsu_avail: 0,
            xsu_used: 0,
            xsu_pagesize: 0,
            xsu_encrypted: 0,
        });
        let mut b = Vec::with_capacity(112);
        b.extend((now - boot.tv_sec).to_le_bytes()); // uptime
        for l in loads {
            b.extend(((l * 65536.0) as u64).to_le_bytes()); // SI_LOAD_SHIFT 16
        }
        let total = sysctl::<u64>("hw.memsize").unwrap_or(0);
        for v in [total, free, 0, 0, swap.xsu_total, swap.xsu_avail] {
            b.extend(v.to_le_bytes()); // ram, free, shared, buffers, swap, free swap
        }
        b.extend([0u8; 8]); // procs, padding
        b.extend([0u8; 16]); // totalhigh, freehigh
        b.extend(1u32.to_le_bytes()); // mem_unit
        b.resize(112, 0);
        Ok(b)
    }

    fn setpgid(&self, pid: i32, pgid: i32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::setpgid(pid, pgid) } as isize)
    }

    fn getpgid(&self, pid: i32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::getpgid(pid) } as isize)
    }

    fn getsid(&self, pid: i32) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::getsid(pid) } as isize)
    }

    fn setsid(&self) -> SysResult {
        // SAFETY: no pointers.
        check(unsafe { libc::setsid() } as isize)
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
    fn flock_conflicts_across_open_descriptions() {
        use std::os::fd::AsRawFd;
        const LOCK_EX: i32 = 2;
        const LOCK_NB: i32 = 4;
        const LOCK_UN: i32 = 8;
        let dir = std::env::temp_dir().join(format!("champollion-flock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lock");
        let f1 = std::fs::File::create(&path).unwrap();
        let f2 = std::fs::File::open(&path).unwrap();
        let h = DarwinHost::default();
        assert_eq!(h.flock(f1.as_raw_fd(), LOCK_EX | LOCK_NB), Ok(0));
        assert_eq!(h.flock(f2.as_raw_fd(), LOCK_EX | LOCK_NB), Err(errno::EAGAIN));
        assert_eq!(h.flock(f1.as_raw_fd(), LOCK_UN), Ok(0));
        assert_eq!(h.flock(f2.as_raw_fd(), LOCK_EX | LOCK_NB), Ok(0));
        drop((f1, f2));
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn openat2_falls_back_to_openat_without_resolve() {
        let dir = std::env::temp_dir().join(format!("champollion-openat2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, b"hi").unwrap();
        let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let h = DarwinHost::default();
        let fd = h.openat2(AT_FDCWD, &cpath, 0, 0, 0).unwrap() as i32;
        h.close(fd).unwrap();
        // RESOLVE_NO_MAGICLINKS: not representable on Darwin, and callers retry with
        // `openat` when they see `ENOSYS`.
        assert_eq!(h.openat2(AT_FDCWD, &cpath, 0, 0, 0x02), Err(errno::ENOSYS));
        std::fs::remove_dir_all(&dir).unwrap();
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

    #[test]
    fn sendfile_copies_from_an_offset_and_tracks_the_file_position() {
        use std::os::fd::AsRawFd;
        let dir = std::env::temp_dir().join(format!("champollion-sendfile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in");
        std::fs::write(&input, b"hello world").unwrap();
        let infile = std::fs::File::open(&input).unwrap();
        let out1 = std::fs::File::create(dir.join("out1")).unwrap();
        let out2 = std::fs::File::create(dir.join("out2")).unwrap();
        let h = DarwinHost::default();

        assert_eq!(h.sendfile(out1.as_raw_fd(), infile.as_raw_fd(), Some(6), 5), Ok(5));
        assert_eq!(std::fs::read(dir.join("out1")).unwrap(), b"world");
        assert_eq!(h.lseek(infile.as_raw_fd(), 0, libc::SEEK_CUR), Ok(0));

        assert_eq!(h.sendfile(out2.as_raw_fd(), infile.as_raw_fd(), None, 5), Ok(5));
        assert_eq!(std::fs::read(dir.join("out2")).unwrap(), b"hello");
        assert_eq!(h.lseek(infile.as_raw_fd(), 0, libc::SEEK_CUR), Ok(5));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fallocate_copy_range_and_close_range() {
        use std::os::fd::AsRawFd;
        let dir = std::env::temp_dir().join(format!("champollion-fileops-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in");
        std::fs::write(&input, b"0123456789").unwrap();
        let infile = std::fs::File::open(&input).unwrap();
        let outfile = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.join("out"))
            .unwrap();
        let h = DarwinHost::default();
        let size = h.fallocate(outfile.as_raw_fd(), 0, 0, 4096).unwrap();
        assert_eq!(size, 0);
        assert_eq!(h.fstat(outfile.as_raw_fd()).unwrap().size, 4096);

        let mut in_off = 2u64;
        let mut out_off = 8u64;
        assert_eq!(
            h.copy_file_range(
                infile.as_raw_fd(),
                Some(in_off),
                outfile.as_raw_fd(),
                Some(out_off),
                4,
            ),
            Ok(4)
        );
        in_off += 4;
        out_off += 4;
        assert_eq!((in_off, out_off), (6, 12));
        let mut b = [0u8; 4];
        assert_eq!(unsafe { libc::pread(outfile.as_raw_fd(), b.as_mut_ptr().cast(), 4, 8) }, 4);
        assert_eq!(&b, b"2345");

        let closee = std::fs::File::open(&input).unwrap();
        assert_eq!(
            h.close_range(closee.as_raw_fd() as u32, closee.as_raw_fd() as u32, false),
            Ok(0)
        );
        assert_eq!(h.fstat(closee.as_raw_fd()), Err(errno::EBADF));
        std::mem::forget(closee); // the fd is already closed; do not close a reused one
        let cloexec = std::fs::File::open(&input).unwrap();
        assert_eq!(
            h.close_range(cloexec.as_raw_fd() as u32, cloexec.as_raw_fd() as u32, true),
            Ok(0)
        );
        assert_ne!(h.fcntl(cloexec.as_raw_fd(), 1, 0).unwrap() & 1, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn utimensat_sets_times_and_translates_now_and_omit() {
        use std::os::fd::AsRawFd;
        let dir = std::env::temp_dir().join(format!("champollion-utimens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        std::fs::write(&path, b"x").unwrap();
        let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let h = DarwinHost::default();
        let at = 1_600_000_000;

        assert_eq!(
            h.utimensat(
                AT_FDCWD,
                Some(&cpath),
                Some([TimeSpec::At(at, 0), TimeSpec::At(at + 1, 0)]),
                0,
            ),
            Ok(0)
        );
        let st = h.fstatat(AT_FDCWD, &cpath, 0).unwrap();
        assert_eq!((st.atime.0, st.mtime.0), (at, at + 1));

        assert_eq!(
            h.utimensat(AT_FDCWD, Some(&cpath), Some([TimeSpec::Now, TimeSpec::Omit]), 0),
            Ok(0)
        );
        let st = h.fstatat(AT_FDCWD, &cpath, 0).unwrap();
        assert!(st.atime.0 > at);
        assert_eq!(st.mtime.0, at + 1);

        assert_eq!(
            h.utimensat(file.as_raw_fd(), None, Some([TimeSpec::At(at + 2, 0); 2]), 0),
            Ok(0)
        );
        let st = h.fstat(file.as_raw_fd()).unwrap();
        assert_eq!((st.atime.0, st.mtime.0), (at + 2, at + 2));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
