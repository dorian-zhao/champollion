//! Linux sockets on Darwin: the BSD API is the same, the encodings are not.
//!
//! - `sockaddr`: Linux starts with a 16-bit family; Darwin with a length byte and an 8-bit
//!   family, and numbers `AF_INET6` 30 instead of 10. The rest of `sockaddr_in`,
//!   `sockaddr_in6`, and `sockaddr_un` agree (Darwin's `sun_path` is 104 bytes, not 108).
//! - `socket` types carry `SOCK_NONBLOCK`/`SOCK_CLOEXEC`, which Darwin sets with `fcntl`.
//! - `SOL_SOCKET` options, a few TCP and IPv6 options, and `MSG_*` flags are numbered
//!   differently.

use super::darwin::{check, linux_errno};
use crate::syscall::{SysResult, errno};

const LINUX_AF_UNIX: u16 = 1;
const LINUX_AF_INET: u16 = 2;
const LINUX_AF_INET6: u16 = 10;
const LINUX_SOCK_NONBLOCK: i32 = 0o4000;
const LINUX_SOCK_CLOEXEC: i32 = 0o2000000;

fn last_errno() -> i32 {
    linux_errno(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

pub fn domain(linux: i32) -> Result<i32, i32> {
    match linux as u16 {
        LINUX_AF_UNIX => Ok(libc::AF_UNIX),
        LINUX_AF_INET => Ok(libc::AF_INET),
        LINUX_AF_INET6 => Ok(libc::AF_INET6),
        _ => Err(errno::EAFNOSUPPORT),
    }
}

/// Apply `SOCK_NONBLOCK`/`SOCK_CLOEXEC` (or the same `accept4` flags) to `fd`.
fn apply_flags(fd: i32, flags: i32) {
    // SAFETY: plain fcntl calls on a descriptor we just made.
    unsafe {
        if flags & LINUX_SOCK_CLOEXEC != 0 {
            libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
        }
        if flags & LINUX_SOCK_NONBLOCK != 0 {
            let fl = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, fl | libc::O_NONBLOCK);
        }
        // Writes to a closed peer return EPIPE instead of raising the host SIGPIPE.
        let one: libc::c_int = 1;
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_NOSIGPIPE,
            (&raw const one).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

pub fn socket(domain_: i32, ty: i32, protocol: i32) -> SysResult {
    let family = domain(domain_)?;
    let flags = ty & (LINUX_SOCK_NONBLOCK | LINUX_SOCK_CLOEXEC);
    // SAFETY: plain syscall.
    let fd = unsafe { libc::socket(family, ty & 0xf, protocol) };
    if fd < 0 {
        return Err(last_errno());
    }
    apply_flags(fd, flags);
    Ok(fd as u64)
}

pub fn socketpair(domain_: i32, ty: i32, protocol: i32) -> Result<[i32; 2], i32> {
    let family = domain(domain_)?;
    let mut fds = [0; 2];
    // SAFETY: socketpair into a local array.
    if unsafe { libc::socketpair(family, ty & 0xf, protocol, fds.as_mut_ptr()) } < 0 {
        return Err(last_errno());
    }
    let flags = ty & (LINUX_SOCK_NONBLOCK | LINUX_SOCK_CLOEXEC);
    fds.iter().for_each(|&fd| apply_flags(fd, flags));
    Ok(fds)
}

/// A Linux `sockaddr` as a Darwin one.
pub fn to_darwin(linux: &[u8]) -> Result<(libc::sockaddr_storage, libc::socklen_t), i32> {
    if linux.len() < 2 {
        return Err(errno::EINVAL);
    }
    let family = u16::from_le_bytes([linux[0], linux[1]]);
    // SAFETY: an all-zero sockaddr_storage is valid.
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let bytes = (&raw mut ss).cast::<u8>();
    let (darwin_family, len) = match family {
        0 => (libc::AF_UNSPEC, 16), // connect(AF_UNSPEC) dissolves a UDP association
        LINUX_AF_INET if linux.len() >= 16 => (libc::AF_INET, 16),
        LINUX_AF_INET6 if linux.len() >= 28 => (libc::AF_INET6, 28),
        LINUX_AF_UNIX => {
            let path = &linux[2..];
            let path = &path[..path.iter().position(|&b| b == 0).unwrap_or(path.len())];
            if linux.len() > 2 && linux[2] == 0 {
                return Err(errno::EINVAL); // the abstract namespace is Linux-only
            }
            if path.len() >= 104 {
                return Err(errno::ENAMETOOLONG);
            }
            (libc::AF_UNIX, 2 + path.len() + 1)
        }
        LINUX_AF_INET | LINUX_AF_INET6 => return Err(errno::EINVAL),
        _ => return Err(errno::EAFNOSUPPORT),
    };
    // SAFETY: `len` <= 2 + 104 fits sockaddr_storage; `linux` holds at least `len` - 1
    // bytes past the family (the NUL of a path is added by zeroing).
    unsafe {
        *bytes = len as u8;
        *bytes.add(1) = darwin_family as u8;
        let body = (len - 2).min(linux.len() - 2);
        std::ptr::copy_nonoverlapping(linux[2..].as_ptr(), bytes.add(2), body);
    }
    Ok((ss, len as libc::socklen_t))
}

/// A Darwin `sockaddr` (of `len` bytes) as a Linux one.
pub fn to_linux(ss: &libc::sockaddr_storage, len: libc::socklen_t) -> Vec<u8> {
    // SAFETY: reading within the storage.
    let bytes = unsafe {
        std::slice::from_raw_parts((ss as *const libc::sockaddr_storage).cast::<u8>(), len as usize)
    };
    if bytes.len() < 2 {
        return vec![0; 2];
    }
    let family = match bytes[1] as i32 {
        libc::AF_INET => LINUX_AF_INET,
        libc::AF_INET6 => LINUX_AF_INET6,
        libc::AF_UNIX => LINUX_AF_UNIX,
        f => f as u16,
    };
    let mut out = family.to_le_bytes().to_vec();
    out.extend_from_slice(&bytes[2..]);
    out
}

pub fn bind(fd: i32, addr: &[u8]) -> SysResult {
    let (ss, len) = to_darwin(addr)?;
    // SAFETY: a valid address of `len` bytes.
    check(unsafe { libc::bind(fd, (&raw const ss).cast(), len) } as isize)
}

pub fn connect(fd: i32, addr: &[u8]) -> SysResult {
    let (ss, len) = to_darwin(addr)?;
    // SAFETY: as in `bind`.
    check(unsafe { libc::connect(fd, (&raw const ss).cast(), len) } as isize)
}

pub fn accept4(fd: i32, flags: i32) -> Result<(u64, Vec<u8>), i32> {
    if flags & !(LINUX_SOCK_NONBLOCK | LINUX_SOCK_CLOEXEC) != 0 {
        return Err(errno::EINVAL);
    }
    // SAFETY: accept into a local storage.
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let new = unsafe { libc::accept(fd, (&raw mut ss).cast(), &mut len) };
    if new < 0 {
        return Err(last_errno());
    }
    // Linux sockets from accept do not inherit O_NONBLOCK; Darwin's do.
    // SAFETY: plain fcntl on the new descriptor.
    unsafe {
        let fl = libc::fcntl(new, libc::F_GETFL);
        libc::fcntl(new, libc::F_SETFL, fl & !libc::O_NONBLOCK);
    }
    apply_flags(new, flags);
    Ok((new as u64, to_linux(&ss, len)))
}

/// `getsockname` (`peer` false) or `getpeername`.
pub fn name(fd: i32, peer: bool) -> Result<Vec<u8>, i32> {
    // SAFETY: into a local storage.
    let mut ss: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let r = unsafe {
        if peer {
            libc::getpeername(fd, (&raw mut ss).cast(), &mut len)
        } else {
            libc::getsockname(fd, (&raw mut ss).cast(), &mut len)
        }
    };
    if r < 0 {
        return Err(last_errno());
    }
    Ok(to_linux(&ss, len))
}

/// A Linux socket option `(level, name)` as Darwin's.
fn option(level: i32, name: i32) -> Result<(i32, i32), i32> {
    const SOL_SOCKET: i32 = 1;
    const IPPROTO_IPV6: i32 = 41;
    Ok(match (level, name) {
        (SOL_SOCKET, _) => (
            libc::SOL_SOCKET,
            match name {
                2 => libc::SO_REUSEADDR,
                3 => libc::SO_TYPE,
                4 => libc::SO_ERROR,
                6 => libc::SO_BROADCAST,
                7 => libc::SO_SNDBUF,
                8 => libc::SO_RCVBUF,
                9 => libc::SO_KEEPALIVE,
                10 => libc::SO_OOBINLINE,
                13 => libc::SO_LINGER,
                15 => libc::SO_REUSEPORT,
                18 => libc::SO_RCVLOWAT,
                19 => libc::SO_SNDLOWAT,
                20 => libc::SO_RCVTIMEO,
                21 => libc::SO_SNDTIMEO,
                _ => return Err(errno::ENOPROTOOPT),
            },
        ),
        (libc::IPPROTO_TCP, _) => (
            libc::IPPROTO_TCP,
            match name {
                1 => libc::TCP_NODELAY,
                4 => libc::TCP_KEEPALIVE, // TCP_KEEPIDLE
                5 => libc::TCP_KEEPINTVL,
                6 => libc::TCP_KEEPCNT,
                _ => return Err(errno::ENOPROTOOPT),
            },
        ),
        (IPPROTO_IPV6, 26) => (libc::IPPROTO_IPV6, libc::IPV6_V6ONLY),
        (libc::IPPROTO_IP, 1) => (libc::IPPROTO_IP, libc::IP_TOS),
        (libc::IPPROTO_IP, 2) => (libc::IPPROTO_IP, libc::IP_TTL),
        _ => return Err(errno::ENOPROTOOPT),
    })
}

pub fn setsockopt(fd: i32, level: i32, name: i32, value: &[u8]) -> SysResult {
    let (level, name) = option(level, name)?;
    // SAFETY: `value` is a local copy of the guest's option.
    check(unsafe {
        libc::setsockopt(fd, level, name, value.as_ptr().cast(), value.len() as libc::socklen_t)
    } as isize)
}

pub fn getsockopt(fd: i32, level: i32, name: i32, len: usize) -> Result<Vec<u8>, i32> {
    let (dlevel, dname) = option(level, name)?;
    let mut v = vec![0u8; len.clamp(4, 256)];
    let mut vlen = v.len() as libc::socklen_t;
    // SAFETY: into a local buffer of `vlen` bytes.
    if unsafe { libc::getsockopt(fd, dlevel, dname, v.as_mut_ptr().cast(), &mut vlen) } < 0 {
        return Err(last_errno());
    }
    v.truncate(vlen as usize);
    if dlevel == libc::SOL_SOCKET && dname == libc::SO_ERROR && v.len() >= 4 {
        // A pending error is an errno: translate it.
        let e = i32::from_le_bytes(v[..4].try_into().unwrap());
        let e = if e == 0 { 0 } else { linux_errno(e) };
        v[..4].copy_from_slice(&e.to_le_bytes());
    }
    Ok(v)
}

/// Linux `MSG_*` flags as Darwin's.
pub fn msg_flags_in(linux: i32) -> i32 {
    let mut f = linux & (libc::MSG_OOB | libc::MSG_PEEK);
    for (l, d) in [
        (0x08, libc::MSG_CTRUNC),
        (0x20, libc::MSG_TRUNC),
        (0x40, libc::MSG_DONTWAIT),
        (0x100, libc::MSG_WAITALL),
        (0x80, libc::MSG_EOR),
    ] {
        if linux & l != 0 {
            f |= d;
        }
    }
    f
}

/// Darwin `MSG_*` flags (as `recvmsg` reports them) as Linux's.
pub fn msg_flags_out(darwin: i32) -> i32 {
    let mut f = darwin & (libc::MSG_OOB | libc::MSG_PEEK);
    for (l, d) in [(0x08, libc::MSG_CTRUNC), (0x20, libc::MSG_TRUNC), (0x80, libc::MSG_EOR)] {
        if darwin & d != 0 {
            f |= l;
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_round_trip() {
        // 127.0.0.1:8080, IPv6 [::1]:443, and a Unix path.
        let inet = [2, 0, 0x1f, 0x90, 127, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let (ss, len) = to_darwin(&inet).unwrap();
        assert_eq!((len, ss.ss_len, ss.ss_family), (16, 16, libc::AF_INET as u8));
        assert_eq!(to_linux(&ss, len), inet);
        let mut inet6 = vec![10, 0, 0x01, 0xbb, 0, 0, 0, 0];
        inet6.extend([0; 15]);
        inet6.extend([1, 0, 0, 0, 0]);
        let (ss, len) = to_darwin(&inet6).unwrap();
        assert_eq!((len, ss.ss_family), (28, libc::AF_INET6 as u8));
        assert_eq!(to_linux(&ss, len), inet6);
        let mut unix = vec![1, 0];
        unix.extend(b"/tmp/sock\0");
        let (ss, len) = to_darwin(&unix).unwrap();
        assert_eq!(to_linux(&ss, len), unix);
        assert_eq!(to_darwin(&[1, 0, 0, b'a']).map(|_| ()), Err(errno::EINVAL), "abstract");
        assert_eq!(to_darwin(&[2, 0, 1]).map(|_| ()), Err(errno::EINVAL), "short");
    }
}
