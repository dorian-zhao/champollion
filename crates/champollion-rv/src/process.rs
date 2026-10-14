//! A guest process and its threads.
//!
//! [`Process`] is everything guest threads share: the address space, the program break,
//! syscall logging, futexes, and the thread count. It is `Sync`, shared through an `Arc`,
//! and each guest thread runs on its own host thread against it. [`Thread`] is per-thread:
//! registers (including the `lr` reservation), the tid bookkeeping of `clone`, and the
//! thread's own engine and JIT.
//!
//! As on Linux, `exit` ends one thread and the process ends with its last thread, while
//! `exit_group` (or a fatal fault) in any thread ends the whole process. The main thread's
//! [`Process::run`] returns when the process ends; when another thread ends it, that
//! thread exits the host process directly, since the main thread may be blocked.
//!
//! Guest address-space layout (2^38 bytes, like Sv39 user space):
//!
//! ```text
//! GUEST_SIZE   ┬ STACK_TOP
//!              │ stack (STACK_SIZE, mapped up front)
//!              ├ STACK_TOP - STACK_SIZE
//!              │ gap for stack growth
//! MMAP_TOP     ┼ mmap allocations, top-down
//!              ⋮
//!              │ heap (brk), growing up
//! image end    ┼ brk start
//!              │ ELF image: fixed-address images where the linker put them
//!              │ (typically ~0x10000); static-PIE images at PIE_BASE (2/3 up)
//! 0            ┴
//! ```

use crate::cpu::CpuState;
use crate::engine::Engine;
use crate::interpreter::Trap;
use crate::jit::Jit;
use crate::loader::{self, LoadError, StackInit, auxv};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use crate::futex::{self, Futexes};

use crate::memory::{GUEST_PAGE_SIZE, GUEST_SIZE, GuestBuf, GuestMemory, MapError, Memory, Prot};
use crate::syscall::{self, Flow, HostOs, SysResult, SyscallLog, errno};

pub const STACK_TOP: u64 = GUEST_SIZE;
pub const STACK_SIZE: u64 = 8 << 20;
pub const MMAP_TOP: u64 = STACK_TOP - (1 << 30);
/// Load base for static-PIE executables: two thirds up the address space, as Linux does
/// (`ELF_ET_DYN_BASE`), leaving room below for fixed-address images and above for brk.
pub const PIE_BASE: u64 = (GUEST_SIZE / 3 * 2) & !(GUEST_PAGE_SIZE - 1);

/// `AT_HWCAP`: one bit per single-letter extension we implement (bit n = letter 'a' + n).
const HWCAP: u64 = hwcap(b"imafdc");

const fn hwcap(letters: &[u8]) -> u64 {
    let mut bits = 0;
    let mut i = 0;
    while i < letters.len() {
        bits |= 1 << (letters[i] - b'a');
        i += 1;
    }
    bits
}

#[derive(Debug)]
pub enum SpawnError {
    Load(LoadError),
    Map(MapError),
    Reserve(std::io::Error),
    /// The image overlaps the stack or the mmap region.
    BadLayout,
}

/// Why a guest stopped running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// `exit`/`exit_group` with this status.
    Status(i32),
    /// Killed by a fault, as Linux would with the signal [`Exit::signal`] returns.
    Trap(Trap),
    /// Killed by a signal the guest sent itself (e.g. SIGABRT from `abort()`).
    Signal(i32),
}

impl Exit {
    /// Tell the user how the guest died (nothing for a normal exit), and return the exit
    /// code a shell would see: the status, or 128 + the signal that killed it.
    pub fn report(self, cpu: &CpuState) -> u8 {
        match self {
            Exit::Status(status) => status as u8,
            Exit::Signal(sig) => {
                eprintln!("rvrun: guest killed by signal {sig}");
                128 + sig as u8
            }
            Exit::Trap(trap) => {
                let sig = self.signal().unwrap();
                eprintln!("rvrun: guest killed by signal {sig}: {trap:?}\n{cpu:?}");
                128 + sig as u8
            }
        }
    }

    /// The Linux signal number that would have killed the process, if any.
    pub fn signal(self) -> Option<i32> {
        const SIGILL: i32 = 4;
        const SIGTRAP: i32 = 5;
        const SIGBUS: i32 = 7;
        const SIGSEGV: i32 = 11;
        match self {
            Exit::Status(_) => None,
            Exit::Signal(sig) => Some(sig),
            Exit::Trap(t) => Some(match t {
                Trap::IllegalInstruction { .. } | Trap::ReservedRoundingMode { .. } => SIGILL,
                Trap::Ebreak { .. } => SIGTRAP,
                Trap::MisalignedAtomic { .. } => SIGBUS,
                Trap::FetchFault(_) | Trap::LoadFault { .. } | Trap::StoreFault { .. } => SIGSEGV,
                Trap::Ecall { .. } => unreachable!("ecall is handled by the run loop"),
            }),
        }
    }
}

pub struct Process {
    pub mem: GuestMemory,
    /// The host the guest's syscalls go to.
    pub host: Arc<dyn HostOs>,
    /// This process's own `Arc`, for the host threads of new guest threads.
    this: Weak<Process>,
    pub futexes: Futexes,
    threads: Mutex<Threads>,
    /// Signaled when the last thread exits.
    all_exited: Condvar,
    /// Whether new threads get a JIT (set by the embedder, like the main thread's).
    pub jit: AtomicBool,
    brk: Mutex<Brk>,
    pub log: SyscallLog,
    /// Absolute path of the executable, for `/proc/self/exe`.
    pub exe: Vec<u8>,
}

struct Threads {
    /// Guest threads that have not exited.
    live: u32,
    next_tid: u32,
    /// Status of the most recent `exit`, which becomes the process's when it is the last.
    last_status: i32,
}

/// How a thread's run loop ended.
enum End {
    /// `exit`: this thread only.
    Thread(i32),
    /// The whole process ends.
    Process(Exit),
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct Brk {
    start: u64,
    cur: u64,
}

/// One guest thread.
pub struct Thread {
    pub cpu: CpuState,
    pub tid: u32,
    /// `prctl(PR_SET_NAME)`: up to 15 bytes, NUL-padded.
    pub name: [u8; 16],
    /// Set by `set_tid_address`; on thread exit the kernel zeroes it and wakes futex
    /// waiters (M5, for `pthread_join`).
    pub clear_child_tid: u64,
    /// This thread's decoded-block cache and TLB.
    pub engine: Engine,
    /// This thread's JIT, if enabled (the default); otherwise `engine` runs everything.
    pub jit: Option<Jit>,
}

fn page_up(a: u64) -> Option<u64> {
    a.checked_add(GUEST_PAGE_SIZE - 1).map(|v| v & !(GUEST_PAGE_SIZE - 1))
}

impl Process {
    /// Load a static executable and set up its stack. Returns the process and its main
    /// thread, ready to run from the entry point. The main thread's tid is the pid.
    pub fn spawn(
        file: &[u8],
        argv: &[Vec<u8>],
        envp: &[Vec<u8>],
        execfn: &[u8],
        host: Arc<dyn HostOs>,
    ) -> Result<(Arc<Self>, Thread), SpawnError> {
        let mut elf = loader::parse(file).map_err(SpawnError::Load)?;
        if elf.pie {
            elf.relocate(PIE_BASE);
        }
        if elf.span().is_none_or(|(_, hi)| hi > MMAP_TOP) {
            return Err(SpawnError::BadLayout);
        }
        let mem = GuestMemory::new().map_err(SpawnError::Reserve)?;
        let brk = loader::map_elf(&mem, &elf).map_err(SpawnError::Map)?;
        mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::RW).map_err(SpawnError::Map)?;

        let id = host.identity();
        let mut random = [0u8; 16];
        let rbuf = GuestBuf { ptr: random.as_mut_ptr(), len: random.len() };
        let _ = host.getrandom(rbuf);
        let auxv = [
            (auxv::AT_PHDR, elf.phdr.unwrap_or(0)),
            (auxv::AT_PHENT, 56),
            (auxv::AT_PHNUM, elf.phnum as u64),
            (auxv::AT_PAGESZ, GUEST_PAGE_SIZE),
            (auxv::AT_BASE, 0),
            (auxv::AT_FLAGS, 0),
            (auxv::AT_ENTRY, elf.entry),
            (auxv::AT_UID, id.uid as u64),
            (auxv::AT_EUID, id.euid as u64),
            (auxv::AT_GID, id.gid as u64),
            (auxv::AT_EGID, id.egid as u64),
            (auxv::AT_HWCAP, HWCAP),
            (auxv::AT_CLKTCK, 100),
            (auxv::AT_SECURE, 0),
        ];
        let init = StackInit { argv, envp, execfn, random, auxv: &auxv };
        let sp = loader::build_stack(&mem, STACK_TOP, &init)
            .map_err(|_| SpawnError::Map(MapError::NoSpace))?;

        let mut cpu = CpuState::new(elf.entry);
        cpu.x[2] = sp;
        let exe = {
            use std::os::unix::ffi::{OsStrExt, OsStringExt};
            let path = std::ffi::OsStr::from_bytes(execfn);
            std::fs::canonicalize(path)
                .map_or_else(|_| execfn.to_vec(), |p| p.into_os_string().into_vec())
        };
        let brk = Mutex::new(Brk { start: brk, cur: brk });
        let threads = Threads { live: 1, next_tid: id.pid + 1, last_status: 0 };
        let process = Arc::new_cyclic(|this| Self {
            mem,
            host,
            this: this.clone(),
            futexes: Futexes::default(),
            threads: Mutex::new(threads),
            all_exited: Condvar::new(),
            jit: AtomicBool::new(false),
            brk,
            log: SyscallLog::default(),
            exe,
        });
        Ok((
            process,
            Thread {
                cpu,
                tid: id.pid,
                name: *b"rvrun\0\0\0\0\0\0\0\0\0\0\0",
                clear_child_tid: 0,
                engine: Engine::new(),
                jit: None,
            },
        ))
    }

    /// Run the main thread until the process ends (see the module docs).
    pub fn run(&self, thread: &mut Thread) -> Exit {
        match self.run_thread(thread) {
            End::Process(exit) => exit,
            End::Thread(status) => {
                self.thread_exited(thread, status);
                let mut threads = lock(&self.threads);
                while threads.live > 0 {
                    threads = self.all_exited.wait(threads).unwrap_or_else(|e| e.into_inner());
                }
                Exit::Status(threads.last_status)
            }
        }
    }

    /// Start a new guest thread (from `clone`) on its own host thread, with tid `tid` from
    /// [`Process::new_tid`]. Its engine and JIT are made there: they are per-thread.
    pub fn start_thread(
        &self,
        cpu: CpuState,
        tid: u32,
        name: [u8; 16],
        clear_child_tid: u64,
    ) -> std::io::Result<()> {
        let this = self.this.upgrade().expect("a running process is alive");
        std::thread::Builder::new().name(format!("guest thread {tid}")).stack_size(4 << 20).spawn(
            move || {
                let jit = if this.jit.load(Ordering::Relaxed) { Jit::new().ok() } else { None };
                let mut thread =
                    Thread { cpu, tid, name, clear_child_tid, engine: Engine::new(), jit };
                match this.run_thread(&mut thread) {
                    End::Thread(status) => this.thread_exited(&thread, status),
                    End::Process(exit) => std::process::exit(exit.report(&thread.cpu).into()),
                }
            },
        )?;
        Ok(())
    }

    /// Reserve a tid for a thread about to start, counting it as live.
    pub fn new_tid(&self) -> u32 {
        let mut threads = lock(&self.threads);
        threads.live += 1;
        threads.next_tid += 1;
        threads.next_tid - 1
    }

    /// Undo [`Process::new_tid`] for a thread that could not be started.
    pub fn forget_tid(&self) {
        lock(&self.threads).live -= 1;
    }

    /// Bookkeeping for a thread's `exit`: clear and wake its `clear_child_tid` word (how
    /// `pthread_join` learns of the exit), and end the process if it was the last thread.
    fn thread_exited(&self, thread: &Thread, status: i32) {
        if thread.clear_child_tid != 0 {
            let addr = thread.clear_child_tid;
            if self.mem.write(addr, &0u32.to_le_bytes()).is_ok() {
                let _ = self.futexes.wake(addr, 1, futex::MATCH_ANY);
            }
        }
        let mut threads = lock(&self.threads);
        threads.live -= 1;
        threads.last_status = status;
        if threads.live == 0 {
            self.all_exited.notify_all();
        }
    }

    /// Run `thread` until it exits or the process ends.
    fn run_thread(&self, thread: &mut Thread) -> End {
        loop {
            let trap = match &mut thread.jit {
                Some(jit) => jit.run(&mut thread.cpu, &self.mem),
                None => match thread.engine.run(&mut thread.cpu, &self.mem, u64::MAX) {
                    (Some(trap), _) => trap,
                    (None, _) => unreachable!("run() with u64::MAX steps returned"),
                },
            };
            match trap {
                Trap::Ecall { .. } => match syscall::dispatch(self, thread) {
                    Flow::Continue => {}
                    Flow::ExitThread(status) => return End::Thread(status),
                    Flow::ExitGroup(status) => return End::Process(Exit::Status(status)),
                    Flow::Killed(sig) => return End::Process(Exit::Signal(sig)),
                },
                trap => return End::Process(Exit::Trap(trap)),
            }
        }
    }

    /// `brk`: move the program break. Returns the new break, or the old one if the
    /// request is out of range or would collide with another mapping (Linux semantics:
    /// failure is signaled by not moving).
    pub fn brk(&self, addr: u64) -> u64 {
        let mut brk = self.brk.lock().unwrap_or_else(|e| e.into_inner());
        if addr < brk.start || addr > MMAP_TOP {
            return brk.cur;
        }
        let (old_end, new_end) = (page_up(brk.cur).unwrap(), page_up(addr).unwrap());
        if new_end > old_end {
            if self.mem.map_if_free(old_end, new_end - old_end, Prot::RW) != Ok(true) {
                return brk.cur;
            }
        } else if new_end < old_end {
            let _ = self.mem.unmap(new_end, old_end - new_end);
        }
        brk.cur = addr;
        addr
    }

    /// `mmap`. Anonymous and private file mappings are supported; shared file mappings
    /// that are writable are not, since writes could never reach the file.
    #[allow(clippy::too_many_arguments)]
    pub fn mmap(
        &self,
        host: &dyn HostOs,
        addr: u64,
        len: u64,
        prot: u32,
        flags: u32,
        fd: i32,
        offset: u64,
    ) -> SysResult {
        const MAP_SHARED: u32 = 0x01;
        const MAP_PRIVATE: u32 = 0x02;
        const MAP_TYPE: u32 = 0x0f;
        const MAP_FIXED: u32 = 0x10;
        const MAP_ANONYMOUS: u32 = 0x20;
        const MAP_FIXED_NOREPLACE: u32 = 0x10_0000;
        const PROT_WRITE: u32 = 2;

        let len = page_up(len).filter(|&l| l != 0).ok_or(errno::EINVAL)?;
        if prot & !7 != 0 || !offset.is_multiple_of(GUEST_PAGE_SIZE) {
            return Err(errno::EINVAL);
        }
        let shared = match flags & MAP_TYPE {
            MAP_PRIVATE => false,
            MAP_SHARED | 0x03 => true, // 0x03 = MAP_SHARED_VALIDATE
            _ => return Err(errno::EINVAL),
        };
        let anonymous = flags & MAP_ANONYMOUS != 0;
        if shared && !anonymous && prot & PROT_WRITE != 0 {
            return Err(errno::ENODEV);
        }

        let in_range = |a: u64| {
            a.is_multiple_of(GUEST_PAGE_SIZE) && a.checked_add(len).is_some_and(|e| e <= GUEST_SIZE)
        };
        // Map read/write first so file contents can be copied in, then apply `prot`.
        // Choosing an address and mapping it is one atomic step, so concurrent mmaps
        // from other threads cannot pick the same range.
        let enomem = |_| errno::ENOMEM;
        let start = if flags & MAP_FIXED_NOREPLACE != 0 {
            if !in_range(addr) {
                return Err(errno::EINVAL);
            }
            if !self.mem.map_if_free(addr, len, Prot::RW).map_err(enomem)? {
                return Err(errno::EEXIST);
            }
            addr
        } else if flags & MAP_FIXED != 0 {
            if !in_range(addr) {
                return Err(errno::EINVAL);
            }
            self.mem.map(addr, len, Prot::RW).map_err(enomem)?;
            addr
        } else if addr != 0
            && in_range(addr)
            && self.mem.map_if_free(addr, len, Prot::RW) == Ok(true)
        {
            addr // a usable hint
        } else {
            self.mem.map_anywhere(len, MMAP_TOP, Prot::RW).map_err(enomem)?
        };

        if !anonymous {
            // Copy the file contents in; bytes past EOF stay zero, as with a real mapping.
            let buf = self.mem.host_buf(start, len, Prot::NONE).unwrap();
            let mut done = 0;
            while done < buf.len {
                let rest = GuestBuf { ptr: buf.ptr.wrapping_add(done), len: buf.len - done };
                match host.pread(fd, rest, (offset + done as u64) as i64) {
                    Ok(0) => break,
                    Ok(n) => done += n as usize,
                    Err(e) => {
                        let _ = self.mem.unmap(start, len);
                        return Err(e);
                    }
                }
            }
        }
        self.mem.protect(start, len, Prot(prot)).unwrap();
        Ok(start)
    }

    /// `mremap`: resize a mapping in place if the pages after it are free, otherwise move
    /// it (with `MREMAP_MAYMOVE`), or put it at `new_addr` (with `MREMAP_FIXED`).
    pub fn mremap(
        &self,
        old: u64,
        old_len: u64,
        new_len: u64,
        flags: u32,
        new_addr: u64,
    ) -> SysResult {
        const MAYMOVE: u32 = 1;
        const FIXED: u32 = 2;
        let old_len = page_up(old_len).ok_or(errno::EINVAL)?;
        let new_len = page_up(new_len).filter(|&l| l != 0).ok_or(errno::EINVAL)?;
        if !old.is_multiple_of(GUEST_PAGE_SIZE)
            || flags & !(MAYMOVE | FIXED) != 0
            || (flags & FIXED != 0 && flags & MAYMOVE == 0)
            || old_len == 0
        {
            return Err(errno::EINVAL);
        }
        // The whole old range must be one protection, which the new pages inherit.
        let prot = self.mem.page_prot(old).ok_or(errno::EFAULT)?;
        let fixed = flags & FIXED != 0;
        if !fixed {
            if new_len <= old_len {
                if new_len < old_len {
                    self.mem.unmap(old + new_len, old_len - new_len).map_err(|_| errno::EINVAL)?;
                }
                return Ok(old);
            }
            let tail = old + old_len;
            if self.mem.map_if_free(tail, new_len - old_len, prot) == Ok(true) {
                return Ok(old);
            }
            if flags & MAYMOVE == 0 {
                return Err(errno::ENOMEM);
            }
        }
        let dst = if fixed {
            if !new_addr.is_multiple_of(GUEST_PAGE_SIZE) {
                return Err(errno::EINVAL);
            }
            self.mem.map(new_addr, new_len, prot).map_err(|_| errno::EINVAL)?;
            new_addr
        } else {
            self.mem.map_anywhere(new_len, MMAP_TOP, prot).map_err(|_| errno::ENOMEM)?
        };
        self.mem.copy_privileged(old, dst, old_len.min(new_len)).map_err(|_| errno::EFAULT)?;
        self.mem.unmap(old, old_len).map_err(|_| errno::EINVAL)?;
        Ok(dst)
    }

    /// `madvise`: only `MADV_DONTNEED` changes semantics (private pages read as zero
    /// afterwards); every other advice is a hint we may ignore.
    pub fn madvise(&self, addr: u64, len: u64, advice: u32) -> SysResult {
        const MADV_DONTNEED: u32 = 4;
        if !addr.is_multiple_of(GUEST_PAGE_SIZE) {
            return Err(errno::EINVAL);
        }
        let len = page_up(len).ok_or(errno::EINVAL)?;
        if advice == MADV_DONTNEED && len > 0 {
            self.mem.zero_privileged(addr, len).map_err(|_| errno::ENOMEM)?;
        }
        Ok(0)
    }

    pub fn munmap(&self, addr: u64, len: u64) -> SysResult {
        let len = page_up(len).filter(|&l| l != 0).ok_or(errno::EINVAL)?;
        self.mem.unmap(addr, len).map_err(|_| errno::EINVAL)?;
        Ok(0)
    }

    pub fn mprotect(&self, addr: u64, len: u64, prot: u32) -> SysResult {
        if prot & !7 != 0 {
            return Err(errno::EINVAL);
        }
        let len = page_up(len).ok_or(errno::EINVAL)?;
        if len == 0 {
            return if addr.is_multiple_of(GUEST_PAGE_SIZE) { Ok(0) } else { Err(errno::EINVAL) };
        }
        self.mem.protect(addr, len, Prot(prot)).map_err(|e| match e {
            MapError::NotMapped => errno::ENOMEM,
            _ => errno::EINVAL,
        })?;
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;
    use crate::syscall::{dispatch, nr};

    const HELLO: &[u8] = include_bytes!("../../../tests/fixtures/hello");
    const P: u64 = GUEST_PAGE_SIZE;
    const RW: u32 = 3;
    const PRIVATE_ANON: u32 = 0x02 | 0x20;

    /// Captures writes; `pread` serves `file`. Single-threaded tests only, so touching
    /// guest buffers from Rust is fine here.
    #[derive(Default)]
    struct FakeHost {
        out: Mutex<Vec<u8>>,
        file: Vec<u8>,
    }

    impl HostOs for FakeHost {
        fn write(&self, _fd: i32, buf: GuestBuf) -> SysResult {
            // SAFETY: validated guest memory, and no other guest thread exists.
            let data = unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) };
            self.out.lock().unwrap().extend_from_slice(data);
            Ok(buf.len as u64)
        }
        fn pread(&self, _fd: i32, buf: GuestBuf, offset: i64) -> SysResult {
            let rest = self.file.get(offset as usize..).unwrap_or(&[]);
            let n = rest.len().min(buf.len);
            // SAFETY: as in `write`.
            unsafe { std::ptr::copy_nonoverlapping(rest.as_ptr(), buf.ptr, n) };
            Ok(n as u64)
        }
    }

    fn spawn() -> (Arc<Process>, Thread, Arc<FakeHost>) {
        let host = Arc::new(FakeHost::default());
        let (p, t) =
            Process::spawn(HELLO, &[b"hello".to_vec()], &[], b"hello", host.clone()).unwrap();
        (p, t, host)
    }

    #[test]
    fn spawn_sets_up_registers_and_stack() {
        let (p, t, _) = spawn();
        let elf = loader::parse(HELLO).unwrap();
        assert_eq!(t.cpu.pc, elf.entry);
        assert_eq!(t.tid, 0, "FakeHost reports pid 0");
        let sp = t.cpu.x[2];
        assert!(sp % 16 == 0 && sp < STACK_TOP && sp > STACK_TOP - STACK_SIZE);
        let mut argc = [0u8; 8];
        p.mem.read(sp, &mut argc).unwrap();
        assert_eq!(u64::from_le_bytes(argc), 1);
    }

    #[test]
    fn brk_grows_and_shrinks() {
        let (p, _, _) = spawn();
        let start = p.brk(0);
        assert_eq!(start % P, 0);
        assert_eq!(p.brk(start - 1), start, "below the start: no change");

        let top = start + 3 * P + 100;
        assert_eq!(p.brk(top), top);
        p.mem.write(top - 1, &[1]).unwrap();
        assert!(p.mem.write(top + P, &[1]).is_err());

        assert_eq!(p.brk(start + P), start + P);
        assert!(p.mem.write(start + P, &[1]).is_err(), "shrunk pages are unmapped");
        assert_eq!(p.brk(start + 2 * P), start + 2 * P);
        let mut b = [9u8];
        p.mem.read(start + P, &mut b).unwrap();
        assert_eq!(b, [0], "regrown pages are zero");

        // Growing into an existing mapping fails by not moving.
        p.mmap(&FakeHost::default(), start + 4 * P, P, RW, PRIVATE_ANON | 0x10, -1, 0).unwrap();
        assert_eq!(p.brk(start + 5 * P), start + 2 * P);
    }

    #[test]
    fn mmap_anonymous() {
        let (p, _, h) = spawn();
        let a = p.mmap(&*h, 0, 100, RW, PRIVATE_ANON, -1, 0).unwrap();
        assert!(a % P == 0 && a + P <= MMAP_TOP);
        let b = p.mmap(&*h, 0, 2 * P, 1, PRIVATE_ANON, -1, 0).unwrap();
        assert!(b + 2 * P <= a, "allocations go top-down without overlap");
        p.mem.write(a + 99, &[1]).unwrap();
        assert!(p.mem.write(b, &[1]).is_err(), "PROT_READ is enforced");

        // A free hint is honored; MAP_FIXED replaces; MAP_FIXED_NOREPLACE refuses.
        let hint = 0x2000_0000;
        assert_eq!(p.mmap(&*h, hint, P, RW, PRIVATE_ANON, -1, 0), Ok(hint));
        p.mem.write(hint, &[7]).unwrap();
        assert_eq!(p.mmap(&*h, hint, P, RW, PRIVATE_ANON | 0x10, -1, 0), Ok(hint));
        let mut v = [9u8];
        p.mem.read(hint, &mut v).unwrap();
        assert_eq!(v, [0]);
        assert_eq!(p.mmap(&*h, hint, P, RW, PRIVATE_ANON | 0x10_0000, -1, 0), Err(errno::EEXIST));

        assert_eq!(p.mmap(&*h, 0, 0, RW, PRIVATE_ANON, -1, 0), Err(errno::EINVAL));
        assert_eq!(p.mmap(&*h, 0, P, RW, 0x20, -1, 0), Err(errno::EINVAL), "no SHARED/PRIVATE");
        assert_eq!(p.mmap(&*h, 0, P, RW, PRIVATE_ANON, -1, 1), Err(errno::EINVAL));
        assert_eq!(p.mmap(&*h, 0x1001, P, RW, PRIVATE_ANON | 0x10, -1, 0), Err(errno::EINVAL));
    }

    #[test]
    fn mmap_file_copies_contents() {
        let (p, _, _) = spawn();
        let h = FakeHost { file: b"0123456789".to_vec(), ..Default::default() };
        let a = p.mmap(&h, 0, P, 1, 0x02, 3, 0).unwrap();
        let mut b = [0u8; 12];
        p.mem.read(a, &mut b).unwrap();
        assert_eq!(&b, b"0123456789\0\0", "bytes past EOF are zero");
        // Writable shared file mappings are refused.
        assert_eq!(p.mmap(&h, 0, P, RW, 0x01, 3, 0), Err(errno::ENODEV));
    }

    #[test]
    fn munmap_and_mprotect() {
        let (p, _, h) = spawn();
        let a = p.mmap(&*h, 0, 3 * P, RW, PRIVATE_ANON, -1, 0).unwrap();
        assert_eq!(p.munmap(a + P, P), Ok(0));
        assert!(p.mem.write(a + P, &[1]).is_err());
        p.mem.write(a + 2 * P, &[1]).unwrap();
        assert_eq!(p.mprotect(a, 3 * P, 1), Err(errno::ENOMEM), "range has a hole");
        assert_eq!(p.mprotect(a, P, 1), Ok(0));
        assert!(p.mem.write(a, &[1]).is_err());
        assert_eq!(p.munmap(a + 1, P), Err(errno::EINVAL));
    }

    #[test]
    fn dispatch_reads_registers_and_writes_result() {
        let (p, mut t, h) = spawn();
        let buf = p.mmap(&*h, 0, P, RW, PRIVATE_ANON, -1, 0).unwrap();
        p.mem.write(buf, b"hi").unwrap();
        let pc = t.cpu.pc;

        t.cpu.x[17] = nr::WRITE;
        t.cpu.x[10..13].copy_from_slice(&[1, buf, 2]);
        assert_eq!(dispatch(&p, &mut t), Flow::Continue);
        assert_eq!(*h.out.lock().unwrap(), b"hi");
        assert_eq!(t.cpu.x[10], 2);
        assert_eq!(t.cpu.pc, pc + 4);

        // Bad pointer -> -EFAULT; unknown syscall -> -ENOSYS.
        t.cpu.x[17] = nr::WRITE;
        t.cpu.x[10..13].copy_from_slice(&[1, 0x10, 2]);
        dispatch(&p, &mut t);
        assert_eq!(t.cpu.x[10] as i64, -(errno::EFAULT as i64));
        t.cpu.x[17] = 4000;
        dispatch(&p, &mut t);
        assert_eq!(t.cpu.x[10] as i64, -(errno::ENOSYS as i64));

        // Exit statuses are masked to 8 bits; exit ends a thread, exit_group the process.
        t.cpu.x[17] = nr::EXIT_GROUP;
        t.cpu.x[10] = 0x1_01;
        assert_eq!(dispatch(&p, &mut t), Flow::ExitGroup(1));
        t.cpu.x[17] = nr::EXIT;
        assert_eq!(dispatch(&p, &mut t), Flow::ExitThread(1));
    }
}
