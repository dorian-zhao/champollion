//! Route B backend: serve Linux syscalls with Darwin (macOS) calls.
//!
//! Guest file descriptors are host file descriptors. Every Linux-specific value coming in
//! (open flags, `AT_*` constants, `whence`, clock ids) is translated to its Darwin
//! counterpart, and every Darwin errno going out is translated to Linux.

use std::ffi::CStr;

use crate::memory::GuestBuf;
use crate::syscall::{AT_FDCWD, HostOs, Identity, Stat, SysResult, errno};

pub struct DarwinHost;

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
const LINUX_AT_EMPTY_PATH: i32 = 0x1000;

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
fn linux_errno(e: i32) -> i32 {
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
fn check(ret: isize) -> SysResult {
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

    fn read(&self, fd: i32, buf: GuestBuf) -> SysResult {
        // SAFETY: the kernel validates the destination.
        check(unsafe { libc::read(fd, buf.ptr.cast(), buf.len) })
    }

    fn write(&self, fd: i32, buf: GuestBuf) -> SysResult {
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

        let h = DarwinHost;
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
