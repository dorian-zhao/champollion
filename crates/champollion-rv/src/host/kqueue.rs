//! Linux `epoll` and `eventfd` on top of Darwin's kqueue.
//!
//! An epoll instance is a kqueue. Registering an fd adds a read and/or write filter whose
//! `udata` carries the guest's `epoll_data`, so waiting needs no lookup table:
//! `EPOLLIN` is `EVFILT_READ`, `EPOLLOUT` is `EVFILT_WRITE`, `EPOLLET` is `EV_CLEAR`, and
//! `EPOLLONESHOT` is `EV_DISPATCH`. Regular files and directories are refused with `EPERM`,
//! as Linux does (Go relies on that to fall back to blocking I/O).
//!
//! An eventfd is a counter kept here, plus one end of a socketpair that holds a byte while
//! the counter is nonzero, so the fd polls readable exactly when a read would succeed.
//! Reads and writes of the counter go through a pipe, so the kernel, not Rust, touches the
//! guest buffer (see [`GuestBuf`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::darwin::{check, linux_errno as darwin_to_linux_errno};
use crate::memory::GuestBuf;
use crate::syscall::{SysResult, errno};

pub mod linux {
    pub const EPOLLIN: u32 = 0x001;
    pub const EPOLLPRI: u32 = 0x002;
    pub const EPOLLOUT: u32 = 0x004;
    pub const EPOLLERR: u32 = 0x008;
    pub const EPOLLHUP: u32 = 0x010;
    pub const EPOLLRDHUP: u32 = 0x2000;
    pub const EPOLLONESHOT: u32 = 1 << 30;
    pub const EPOLLET: u32 = 1 << 31;
    pub const EPOLL_CTL_ADD: i32 = 1;
    pub const EPOLL_CTL_DEL: i32 = 2;
    pub const EPOLL_CTL_MOD: i32 = 3;
    /// `O_CLOEXEC`, which `EPOLL_CLOEXEC` and `EFD_CLOEXEC` equal.
    pub const CLOEXEC: i32 = 0o2000000;
    /// `O_NONBLOCK` (`EFD_NONBLOCK`).
    pub const NONBLOCK: i32 = 0o4000;
    pub const EFD_SEMAPHORE: i32 = 1;
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn last_errno() -> i32 {
    darwin_to_linux_errno(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

fn set_cloexec(fd: i32) {
    // SAFETY: plain fcntl on our own fd.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
}

fn set_nonblock(fd: i32) {
    // SAFETY: plain fcntl on our own fd.
    unsafe {
        let fl = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, fl | libc::O_NONBLOCK);
    }
}

/// `epoll_create1`: a kqueue.
pub fn epoll_create1(flags: i32) -> SysResult {
    if flags & !linux::CLOEXEC != 0 {
        return Err(errno::EINVAL);
    }
    // SAFETY: plain syscall.
    let kq = unsafe { libc::kqueue() };
    if kq < 0 {
        return Err(last_errno());
    }
    if flags & linux::CLOEXEC != 0 {
        set_cloexec(kq);
    }
    Ok(kq as u64)
}

/// `epoll_ctl`. `event` is `(events, data)`, unused for `EPOLL_CTL_DEL`.
pub fn epoll_ctl(epfd: i32, op: i32, fd: i32, event: (u32, u64)) -> SysResult {
    use linux::*;
    if !matches!(op, EPOLL_CTL_ADD | EPOLL_CTL_DEL | EPOLL_CTL_MOD) {
        return Err(errno::EINVAL);
    }
    if op != EPOLL_CTL_DEL {
        // SAFETY: fstat into a local.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } < 0 {
            return Err(last_errno());
        }
        let kind = st.st_mode & libc::S_IFMT;
        if kind == libc::S_IFREG || kind == libc::S_IFDIR {
            return Err(errno::EPERM);
        }
    }
    let (events, data) = event;
    let mut flags = libc::EV_ADD | libc::EV_ENABLE | libc::EV_RECEIPT;
    if events & EPOLLET != 0 {
        flags |= libc::EV_CLEAR;
    }
    if events & EPOLLONESHOT != 0 {
        flags |= libc::EV_DISPATCH;
    }
    let change = |filter, wanted: bool| libc::kevent {
        ident: fd as usize,
        filter,
        flags: if op != EPOLL_CTL_DEL && wanted {
            flags
        } else {
            libc::EV_DELETE | libc::EV_RECEIPT
        },
        fflags: 0,
        data: 0,
        udata: data as *mut libc::c_void,
    };
    let changes = [
        change(libc::EVFILT_READ, events & (EPOLLIN | EPOLLPRI | EPOLLRDHUP) != 0),
        change(libc::EVFILT_WRITE, events & EPOLLOUT != 0),
    ];
    // SAFETY: valid change and receipt arrays; zero timeout.
    let mut receipts: [libc::kevent; 2] = unsafe { std::mem::zeroed() };
    let zero = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    let n = unsafe { libc::kevent(epfd, changes.as_ptr(), 2, receipts.as_mut_ptr(), 2, &zero) };
    if n < 0 {
        return Err(last_errno());
    }
    // With EV_RECEIPT every change reports back, with EV_ERROR and its errno in `data`.
    let mut deleted_missing = 0;
    for r in &receipts[..n as usize] {
        let err = r.data as i32;
        if r.flags & libc::EV_ERROR == 0 || err == 0 {
            continue;
        }
        if r.flags & libc::EV_DELETE != 0 && err == libc::ENOENT {
            deleted_missing += 1; // removing a filter that was never added is fine...
            continue;
        }
        return Err(darwin_to_linux_errno(err));
    }
    if op == EPOLL_CTL_DEL && deleted_missing == 2 {
        return Err(errno::ENOENT); // ...unless the fd was not registered at all
    }
    Ok(0)
}

/// `epoll_pwait`: up to `max` ready `(events, data)` pairs, waiting at most `timeout_ms`
/// (forever if negative). Read and write readiness of one fd are merged into one event.
pub fn epoll_wait(epfd: i32, max: usize, timeout_ms: i32) -> Result<Vec<(u32, u64)>, i32> {
    use linux::*;
    if max == 0 {
        return Err(errno::EINVAL);
    }
    let max = max.min(1024);
    // SAFETY: kevent fills at most `max` entries of the buffer.
    let mut out: Vec<libc::kevent> = vec![unsafe { std::mem::zeroed() }; max];
    let ts = (timeout_ms >= 0).then(|| libc::timespec {
        tv_sec: (timeout_ms / 1000) as libc::time_t,
        tv_nsec: (timeout_ms % 1000) as libc::c_long * 1_000_000,
    });
    let tp = ts.as_ref().map_or(std::ptr::null(), |t| t as *const libc::timespec);
    // SAFETY: as above.
    let n = unsafe { libc::kevent(epfd, std::ptr::null(), 0, out.as_mut_ptr(), max as i32, tp) };
    if n < 0 {
        return Err(last_errno());
    }
    let mut events: Vec<(usize, u32, u64)> = Vec::with_capacity(n as usize);
    for k in &out[..n as usize] {
        let eof = k.flags & libc::EV_EOF != 0;
        let mut ev = if k.flags & libc::EV_ERROR != 0 { EPOLLERR } else { 0 };
        if k.filter == libc::EVFILT_READ {
            ev |= EPOLLIN | if eof { EPOLLRDHUP | EPOLLHUP } else { 0 };
        } else if k.filter == libc::EVFILT_WRITE {
            ev |= EPOLLOUT | if eof { EPOLLERR } else { 0 };
        }
        match events.iter_mut().find(|e| e.0 == k.ident) {
            Some(e) => e.1 |= ev,
            None => events.push((k.ident, ev, k.udata as u64)),
        }
    }
    Ok(events.into_iter().map(|(_, ev, data)| (ev, data)).collect())
}

/// Copies between Rust and guest buffers through a pipe, so the kernel does the guest side.
struct Bounce {
    pipe: Mutex<Option<[i32; 2]>>,
}

impl Bounce {
    fn with<T>(&self, f: impl FnOnce(i32, i32) -> Result<T, i32>) -> Result<T, i32> {
        let mut pipe = lock(&self.pipe);
        if pipe.is_none() {
            let mut fds = [0; 2];
            // SAFETY: pipe into a local array.
            if unsafe { libc::pipe(fds.as_mut_ptr()) } < 0 {
                return Err(last_errno());
            }
            fds.iter().for_each(|&fd| set_cloexec(fd));
            *pipe = Some(fds);
        }
        let [r, w] = pipe.unwrap();
        f(r, w)
    }

    /// The first 8 bytes of a guest buffer.
    fn read_u64(&self, buf: GuestBuf) -> Result<u64, i32> {
        self.with(|r, w| {
            // SAFETY: the kernel reads the guest buffer; we read into a local.
            check(unsafe { libc::write(w, buf.ptr.cast(), 8) })?;
            let mut v = [0u8; 8];
            check(unsafe { libc::read(r, v.as_mut_ptr().cast(), 8) })?;
            Ok(u64::from_le_bytes(v))
        })
    }

    /// Store 8 bytes into a guest buffer.
    fn write_u64(&self, buf: GuestBuf, v: u64) -> Result<(), i32> {
        self.with(|r, w| {
            // SAFETY: we write from a local; the kernel writes the guest buffer.
            check(unsafe { libc::write(w, v.to_le_bytes().as_ptr().cast(), 8) })?;
            match check(unsafe { libc::read(r, buf.ptr.cast(), 8) }) {
                Ok(_) => Ok(()),
                Err(e) => {
                    // Drain the pipe so the next copy starts clean.
                    let mut junk = [0u8; 8];
                    unsafe { libc::read(r, junk.as_mut_ptr().cast(), 8) };
                    Err(e)
                }
            }
        })
    }
}

struct EventFd {
    /// Socketpair end that the guest never sees: writing a byte here makes the guest's
    /// end readable.
    peer: i32,
    semaphore: bool,
    nonblock: AtomicBool,
    count: Mutex<u64>,
}

/// The eventfds of a process, by guest fd.
pub struct EventFds {
    fds: Mutex<HashMap<i32, Arc<EventFd>>>,
    /// Fast path: whether any eventfd exists (reads and writes check every fd otherwise).
    any: AtomicBool,
    bounce: Bounce,
}

impl Default for EventFds {
    fn default() -> Self {
        Self {
            fds: Mutex::default(),
            any: AtomicBool::new(false),
            bounce: Bounce { pipe: Mutex::new(None) },
        }
    }
}

impl EventFds {
    pub fn create(&self, initval: u32, flags: i32) -> SysResult {
        use linux::*;
        if flags & !(CLOEXEC | NONBLOCK | EFD_SEMAPHORE) != 0 {
            return Err(errno::EINVAL);
        }
        let mut fds = [0; 2];
        // SAFETY: socketpair into a local array.
        if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) } < 0 {
            return Err(last_errno());
        }
        let [guest, peer] = fds;
        set_cloexec(peer);
        set_nonblock(guest);
        set_nonblock(peer);
        if flags & CLOEXEC != 0 {
            set_cloexec(guest);
        }
        let e = Arc::new(EventFd {
            peer,
            semaphore: flags & EFD_SEMAPHORE != 0,
            nonblock: AtomicBool::new(flags & NONBLOCK != 0),
            count: Mutex::new(0),
        });
        if initval > 0 {
            e.add(initval as u64);
        }
        lock(&self.fds).insert(guest, e);
        self.any.store(true, Ordering::Relaxed);
        Ok(guest as u64)
    }

    fn get(&self, fd: i32) -> Option<Arc<EventFd>> {
        if !self.any.load(Ordering::Relaxed) {
            return None;
        }
        lock(&self.fds).get(&fd).cloned()
    }

    /// `read` if `fd` is an eventfd.
    pub fn read(&self, fd: i32, buf: GuestBuf) -> Option<SysResult> {
        let e = self.get(fd)?;
        if buf.len < 8 {
            return Some(Err(errno::EINVAL));
        }
        Some(loop {
            let mut count = lock(&e.count);
            if *count == 0 {
                drop(count);
                if e.nonblock.load(Ordering::Relaxed) {
                    break Err(errno::EAGAIN);
                }
                let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
                // SAFETY: poll on one local pollfd.
                unsafe { libc::poll(&mut p, 1, -1) };
                continue;
            }
            let value = if e.semaphore { 1 } else { *count };
            if let Err(err) = self.bounce.write_u64(buf, value) {
                break Err(err);
            }
            *count -= value;
            if *count == 0 {
                // Not readable any more: take back the byte.
                let mut b = 0u8;
                // SAFETY: read into a local.
                unsafe { libc::read(fd, (&raw mut b).cast(), 1) };
            }
            break Ok(8);
        })
    }

    /// `write` if `fd` is an eventfd.
    pub fn write(&self, fd: i32, buf: GuestBuf) -> Option<SysResult> {
        let e = self.get(fd)?;
        if buf.len < 8 {
            return Some(Err(errno::EINVAL));
        }
        Some(self.bounce.read_u64(buf).and_then(|v| {
            if v == u64::MAX {
                return Err(errno::EINVAL);
            }
            // Writes that would overflow block on Linux; that never happens in practice.
            e.add(v);
            Ok(8)
        }))
    }

    /// `close` if `fd` is an eventfd: also close its peer.
    pub fn close(&self, fd: i32) {
        if let Some(e) = self.get(fd) {
            lock(&self.fds).remove(&fd);
            // SAFETY: closing our own fd.
            unsafe { libc::close(e.peer) };
        }
    }

    /// `fcntl(F_SETFL)` on an eventfd: track `O_NONBLOCK` (the guest fd stays nonblocking).
    pub fn set_nonblock(&self, fd: i32, nonblock: bool) -> bool {
        match self.get(fd) {
            Some(e) => {
                e.nonblock.store(nonblock, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }
}

impl EventFd {
    fn add(&self, v: u64) {
        if v == 0 {
            return;
        }
        let mut count = lock(&self.count);
        if *count == 0 {
            let b = 1u8;
            // SAFETY: write from a local.
            unsafe { libc::write(self.peer, (&raw const b).cast(), 1) };
        }
        *count = count.saturating_add(v).min(u64::MAX - 1);
    }
}
