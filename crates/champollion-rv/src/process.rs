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
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use crate::futex::{self, Futexes};
use crate::signal::{self, Action, AltStack, Info, Pending, ThreadSignals};

use crate::memory::{
    GUEST_PAGE_SIZE, GUEST_SIZE, GuestBuf, GuestMemory, INTERRUPT_SLOTS, MapError, Memory, Prot,
};
use crate::syscall::{self, Flow, HostOs, SysResult, SyscallLog, errno};

pub const STACK_TOP: u64 = GUEST_SIZE;
pub const STACK_SIZE: u64 = 8 << 20;
pub const MMAP_TOP: u64 = STACK_TOP - (1 << 30);
/// Load base for static-PIE executables: two thirds up the address space, as Linux does
/// (`ELF_ET_DYN_BASE`), leaving room below for fixed-address images and above for brk.
pub const PIE_BASE: u64 = (GUEST_SIZE / 3 * 2) & !(GUEST_PAGE_SIZE - 1);
/// A page with `li a7, 139; ecall` (`rt_sigreturn`), where signal handlers return to: the
/// vDSO's job on Linux. It sits just above the mmap region.
pub const SIGRETURN_TRAMPOLINE: u64 = MMAP_TOP;
/// Load base of a dynamically linked program's interpreter (the dynamic loader), at the
/// top of the mmap region as on Linux; the libraries it maps go below.
pub const INTERP_BASE: u64 = MMAP_TOP - (64 << 20);

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
    /// The program interpreter (dynamic loader) at this host path could not be read.
    Interp(std::path::PathBuf, std::io::Error),
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
    /// End the host process the way the guest ended. A guest killed by a signal that only
    /// terminates (and has the same number on the host) kills rvrun with that signal, so
    /// the parent sees exactly what Linux would show it; for the others (faults, signals
    /// that dump core, which would make macOS file a crash report for rvrun) this reports
    /// what happened and exits with 128 + the signal, as [`Exit::report`] does.
    pub fn terminate(self, cpu: &CpuState) -> ! {
        if let Exit::Signal(sig @ (1 | 2 | 9 | 13 | 14 | 15)) = self {
            // SAFETY: restoring the default action and raising it ends the process.
            unsafe {
                libc::signal(sig, libc::SIG_DFL);
                let mut set: libc::sigset_t = std::mem::zeroed();
                libc::sigemptyset(&mut set);
                libc::sigaddset(&mut set, sig);
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
                // To the process, not this thread: a thread-directed signal would be lost if
                // another thread execs first, while a process-directed one survives exec.
                libc::kill(libc::getpid(), sig);
                // Let it arrive here before exiting another way.
                libc::sigsuspend(&{
                    let mut none: libc::sigset_t = std::mem::zeroed();
                    libc::sigemptyset(&mut none);
                    none
                });
            }
        }
        std::process::exit(self.report(cpu).into())
    }

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
    /// Signal dispositions, by signal number - 1.
    actions: Mutex<[Action; signal::NSIG as usize]>,
    /// Process-directed pending signals (`kill`), taken by whichever thread can first.
    pending: Mutex<Pending>,
    /// Every live thread's shared signal state, by tid (for `tgkill`).
    thread_signals: Mutex<std::collections::HashMap<u32, Arc<ThreadSignals>>>,
    /// Interrupt flag slots in use (see [`GuestMemory::interrupt_flag`]). The main thread
    /// has slot 0; the last slot is never set, for threads beyond the supply.
    slots: Mutex<Vec<bool>>,
    brk: Mutex<Brk>,
    pub log: SyscallLog,
    /// Absolute path of the executable, for `/proc/self/exe`.
    pub exe: Vec<u8>,
    /// Where the guest's absolute paths are looked up first (see [`Process::host_path`]).
    pub sysroot: Option<std::path::PathBuf>,
    /// tid of the thread that ends the process by returning from [`Process::run`].
    main_tid: std::sync::atomic::AtomicU32,
    /// Set in a child forked from another thread: there is no main thread to wait for the
    /// others, so the last thread to exit ends the process itself.
    orphaned: AtomicBool,
    /// How to start a new program for `execve` (the embedder knows how to launch itself).
    exec: std::sync::OnceLock<ExecFn>,
    /// `ITIMER_REAL`, served by a thread started on first use.
    itimer: Mutex<ITimer>,
    itimer_changed: Condvar,
    /// Run in a forked child, which keeps only the forking thread: restarts whatever host
    /// threads the process relies on (the host signal pump).
    on_fork: std::sync::OnceLock<Box<dyn Fn(Arc<Process>) + Send + Sync>>,
}

/// What `execve` asks the embedder to run: a guest executable (a host path) with its
/// arguments (including `argv[0]`) and environment. Returns only on failure, with a Linux
/// errno.
pub type ExecFn = Box<dyn Fn(&Exec) -> i32 + Send + Sync>;

pub struct Exec {
    pub path: std::path::PathBuf,
    pub argv: Vec<Vec<u8>>,
    pub envp: Vec<Vec<u8>>,
    /// Signals the new program starts with ignored (ignored dispositions survive `execve`;
    /// handlers do not), and its signal mask.
    pub ignored: u64,
    pub mask: u64,
}

#[derive(Default)]
struct ITimer {
    /// When SIGALRM is due, if armed.
    deadline: Option<std::time::Instant>,
    /// The period to rearm with after it fires (zero: once).
    interval: std::time::Duration,
    /// Whether the timer thread runs.
    started: bool,
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
    /// Signal mask and pending thread-directed signals, shared with senders.
    pub signals: Arc<ThreadSignals>,
    pub altstack: AltStack,
    /// The mask to restore when the next handler returns, if not the current one
    /// (`rt_sigsuspend` runs the handler under its temporary mask).
    pub saved_mask: Option<u64>,
    /// Set by `set_tid_address`; on thread exit the kernel zeroes it and wakes futex
    /// waiters (M5, for `pthread_join`).
    pub clear_child_tid: u64,
    /// This thread's decoded-block cache and TLB.
    pub engine: Engine,
    /// This thread's JIT, if enabled (the default); otherwise `engine` runs everything.
    pub jit: Option<Jit>,
}

/// The host signal that interrupts a guest thread's blocking host syscall (EINTR) when a
/// guest signal arrives for it. Its handler does nothing; guest code never sees it.
const WAKE_SIGNAL: i32 = libc::SIGURG;

extern "C" fn wake_handler(_: libc::c_int) {}

fn install_wake_handler() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: installing a handler without SA_RESTART, so blocking calls return EINTR.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = wake_handler as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(WAKE_SIGNAL, &sa, std::ptr::null_mut());
        }
    });
}

/// A thread blocked in a wait of ours; see [`Process::blocked`].
pub struct Blocked<'a> {
    process: &'a Process,
    thread: &'a Thread,
}

impl futex::Interruptible for Blocked<'_> {
    fn arm(&self, waker: Option<signal::Waker>) {
        *lock(&self.thread.signals.waker) = waker;
    }

    fn interrupted(&self) -> bool {
        self.process.signal_waiting(self.thread).is_some()
    }
}

/// `path` under `sysroot` if it is absolute and exists there, else `path` itself. Paths
/// are resolved as in a chroot: symlinks (with absolute targets inside the sysroot, as
/// distributions make them) and `..` never leave it. The final component is followed if
/// it is a symlink only with `follow`.
pub fn resolve(
    sysroot: Option<&std::path::Path>,
    path: &std::path::Path,
    follow: bool,
) -> std::path::PathBuf {
    match sysroot {
        Some(root) if path.is_absolute() => {
            resolve_in(root, path, follow).unwrap_or_else(|| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

fn resolve_in(
    root: &std::path::Path,
    path: &std::path::Path,
    follow: bool,
) -> Option<std::path::PathBuf> {
    use std::path::Component;
    // Components still to walk, last first; `cur` is the host path reached so far.
    let mut todo: Vec<std::ffi::OsString> = Vec::new();
    let push_path = |todo: &mut Vec<std::ffi::OsString>, p: &std::path::Path| {
        for c in p.components().rev() {
            match c {
                Component::Normal(n) => todo.push(n.to_os_string()),
                Component::ParentDir => todo.push("..".into()),
                _ => {}
            }
        }
    };
    push_path(&mut todo, path);
    let mut cur = root.to_path_buf();
    let mut depth = 0; // within the sysroot
    let mut links = 0;
    while let Some(name) = todo.pop() {
        if name == ".." {
            if depth > 0 {
                cur.pop();
                depth -= 1;
            }
            continue;
        }
        let next = cur.join(&name);
        let meta = std::fs::symlink_metadata(&next).ok()?;
        let last = todo.is_empty();
        if meta.file_type().is_symlink() && (follow || !last) {
            links += 1;
            if links > 40 {
                return None; // ELOOP: let the host report it
            }
            let target = std::fs::read_link(&next).ok()?;
            if target.is_absolute() {
                cur = root.to_path_buf();
                depth = 0;
            }
            push_path(&mut todo, &target);
            continue;
        }
        cur = next;
        depth += 1;
    }
    Some(cur)
}

fn page_up(a: u64) -> Option<u64> {
    a.checked_add(GUEST_PAGE_SIZE - 1).map(|v| v & !(GUEST_PAGE_SIZE - 1))
}

impl Process {
    /// Load an executable and set up its stack. Returns the process and its main thread,
    /// ready to run from the entry point (the interpreter's, for a dynamically linked
    /// program, found under `sysroot`). The main thread's tid is the pid.
    pub fn spawn(
        file: &[u8],
        argv: &[Vec<u8>],
        envp: &[Vec<u8>],
        execfn: &[u8],
        host: Arc<dyn HostOs>,
        sysroot: Option<std::path::PathBuf>,
    ) -> Result<(Arc<Self>, Thread), SpawnError> {
        let mut elf = loader::parse(file).map_err(SpawnError::Load)?;
        if elf.pie {
            elf.relocate(PIE_BASE);
        }
        if elf.span().is_none_or(|(_, hi)| hi > MMAP_TOP) {
            return Err(SpawnError::BadLayout);
        }
        // The identity layout lets system calls pass guest pointers through.
        let mem = match GuestMemory::new_identity() {
            Some(mem) => mem,
            None => GuestMemory::new().map_err(SpawnError::Reserve)?,
        };
        let brk = loader::map_elf(&mem, &elf).map_err(SpawnError::Map)?;
        mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::RW).map_err(SpawnError::Map)?;
        mem.map(SIGRETURN_TRAMPOLINE, GUEST_PAGE_SIZE, Prot::READ | Prot::EXEC)
            .map_err(SpawnError::Map)?;
        const LI_A7_139: u32 = 0x08b0_0893;
        const ECALL: u32 = 0x0000_0073;
        let code: Vec<u8> = [LI_A7_139, ECALL].iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write_privileged(SIGRETURN_TRAMPOLINE, &code).map_err(|_| SpawnError::BadLayout)?;

        // A dynamically linked program starts in its interpreter, which finds the program
        // through the auxiliary vector (AT_PHDR, AT_ENTRY) and itself through AT_BASE.
        let (start, interp_base) = match elf.interp {
            None => (elf.entry, 0),
            Some(path) => {
                let path = std::ffi::OsStr::from_bytes(path);
                let host_path = resolve(sysroot.as_deref(), std::path::Path::new(path), true);
                let bytes = std::fs::read(&host_path)
                    .map_err(|e| SpawnError::Interp(host_path.clone(), e))?;
                let mut interp = loader::parse(&bytes).map_err(SpawnError::Load)?;
                if !interp.pie || interp.interp.is_some() {
                    return Err(SpawnError::BadLayout);
                }
                interp.relocate(INTERP_BASE);
                if interp.span().is_none_or(|(_, hi)| hi > MMAP_TOP) {
                    return Err(SpawnError::BadLayout);
                }
                loader::map_elf(&mem, &interp).map_err(SpawnError::Map)?;
                (interp.entry, INTERP_BASE)
            }
        };

        let id = host.identity();
        let mut random = [0u8; 16];
        let rbuf = GuestBuf { ptr: random.as_mut_ptr(), len: random.len() };
        let _ = host.getrandom(rbuf);
        let auxv = [
            (auxv::AT_PHDR, elf.phdr.unwrap_or(0)),
            (auxv::AT_PHENT, 56),
            (auxv::AT_PHNUM, elf.phnum as u64),
            (auxv::AT_PAGESZ, GUEST_PAGE_SIZE),
            (auxv::AT_BASE, interp_base),
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

        let mut cpu = CpuState::new(start);
        cpu.x[2] = sp;
        let exe = {
            use std::os::unix::ffi::{OsStrExt, OsStringExt};
            let path = std::ffi::OsStr::from_bytes(execfn);
            std::fs::canonicalize(path)
                .map_or_else(|_| execfn.to_vec(), |p| p.into_os_string().into_vec())
        };
        let brk = Mutex::new(Brk { start: brk, cur: brk });
        let threads = Threads { live: 1, next_tid: id.pid + 1, last_status: 0 };
        let main_signals = Arc::new(ThreadSignals::default());
        let process = Arc::new_cyclic(|this| Self {
            mem,
            host,
            this: this.clone(),
            futexes: Futexes::default(),
            threads: Mutex::new(threads),
            all_exited: Condvar::new(),
            jit: AtomicBool::new(false),
            actions: Mutex::new([Action::default(); signal::NSIG as usize]),
            pending: Mutex::default(),
            thread_signals: Mutex::new(std::collections::HashMap::from([(
                id.pid,
                main_signals.clone(),
            )])),
            slots: Mutex::new({
                let mut slots = vec![false; INTERRUPT_SLOTS];
                slots[0] = true;
                slots
            }),
            brk,
            log: SyscallLog::default(),
            exe,
            sysroot,
            main_tid: std::sync::atomic::AtomicU32::new(id.pid),
            orphaned: AtomicBool::new(false),
            exec: std::sync::OnceLock::new(),
            on_fork: std::sync::OnceLock::new(),
            itimer: Mutex::default(),
            itimer_changed: Condvar::new(),
        });
        Ok((
            process,
            Thread {
                cpu,
                tid: id.pid,
                name: *b"rvrun\0\0\0\0\0\0\0\0\0\0\0",
                signals: main_signals,
                altstack: AltStack::default(),
                saved_mask: None,
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
        mask: u64,
        clear_child_tid: u64,
    ) -> std::io::Result<()> {
        let this = self.this.upgrade().expect("a running process is alive");
        // A new thread inherits the signal mask, but not the alternate stack.
        let slot = {
            let mut slots = lock(&self.slots);
            let free = slots[..INTERRUPT_SLOTS - 1].iter().position(|used| !used);
            let slot = free.unwrap_or(INTERRUPT_SLOTS - 1);
            slots[slot] = true;
            slot
        };
        let signals = Arc::new(ThreadSignals { slot, ..ThreadSignals::default() });
        signals.set_mask(mask);
        lock(&self.thread_signals).insert(tid, signals.clone());
        std::thread::Builder::new().name(format!("guest thread {tid}")).stack_size(4 << 20).spawn(
            move || {
                let mut jit = if this.jit.load(Ordering::Relaxed) { Jit::new().ok() } else { None };
                if let Some(jit) = &mut jit {
                    jit.set_interrupt_slot(slot);
                }
                let mut thread = Thread {
                    cpu,
                    tid,
                    name,
                    signals,
                    altstack: AltStack::default(),
                    saved_mask: None,
                    clear_child_tid,
                    engine: Engine::new(),
                    jit,
                };
                match this.run_thread(&mut thread) {
                    End::Thread(status) => this.thread_exited(&thread, status),
                    End::Process(exit) => exit.terminate(&thread.cpu),
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

    fn free_slot(&self, slot: usize) {
        if slot != INTERRUPT_SLOTS - 1 {
            lock(&self.slots)[slot] = false;
        }
    }

    /// Undo [`Process::new_tid`] for a thread that could not be started.
    pub fn forget_tid(&self, tid: u32) {
        if let Some(t) = lock(&self.thread_signals).remove(&tid) {
            self.free_slot(t.slot);
        }
        lock(&self.threads).live -= 1;
    }

    /// Bookkeeping for a thread's `exit`: clear and wake its `clear_child_tid` word (how
    /// `pthread_join` learns of the exit), and end the process if it was the last thread.
    fn thread_exited(&self, thread: &Thread, status: i32) {
        lock(&self.thread_signals).remove(&thread.tid);
        thread.signals.host_thread.store(0, Ordering::SeqCst);
        self.free_slot(thread.signals.slot);
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
            if self.orphaned.load(Ordering::Relaxed) {
                std::process::exit(status);
            }
            self.all_exited.notify_all();
        }
    }

    /// Run `thread` until it exits or the process ends.
    fn run_thread(&self, thread: &mut Thread) -> End {
        // SAFETY: pthread_self has no preconditions.
        thread
            .signals
            .host_thread
            .store(unsafe { libc::pthread_self() } as usize, Ordering::SeqCst);
        install_wake_handler();
        loop {
            if let Some(end) = self.deliver_signals(thread) {
                return end;
            }
            // Both come back now and then without a trap, to take signals: the JIT when its
            // interrupt flag is set, the engine after a slice of instructions.
            let trap = match &mut thread.jit {
                Some(jit) => jit.run_interruptible(&mut thread.cpu, &self.mem),
                None => thread.engine.run(&mut thread.cpu, &self.mem, 1 << 20).0,
            };
            let Some(trap) = trap else { continue };
            match trap {
                Trap::Ecall { .. } => match syscall::dispatch(self, thread) {
                    Flow::Continue => {}
                    Flow::ExitThread(status) => return End::Thread(status),
                    Flow::ExitGroup(status) => return End::Process(Exit::Status(status)),
                    Flow::Killed(sig) => return End::Process(Exit::Signal(sig)),
                },
                trap => {
                    if let Some(end) = self.trap_signal(thread, trap) {
                        return end;
                    }
                }
            }
        }
    }

    /// The host path for guest path `path`: on macOS, `/dev/shm` (POSIX shared memory and
    /// named semaphores) is a per-user directory under the host's temporary directory; others
    /// are under the sysroot if they are absolute and exist there (as QEMU's `-L` does),
    /// else unchanged. See [`resolve`] for `follow`.
    pub fn host_path<'a>(
        &self,
        path: &'a std::ffi::CStr,
        follow: bool,
    ) -> std::borrow::Cow<'a, std::ffi::CStr> {
        use std::os::unix::ffi::OsStrExt;
        #[cfg(target_os = "macos")]
        if let Some(rest) = path.to_bytes().strip_prefix(b"/dev/shm")
            && (rest.is_empty() || rest[0] == b'/')
        {
            let id = self.host.identity();
            let dir = std::env::temp_dir().join(format!("rvrun-shm-{}", id.uid));
            let _ = std::fs::create_dir_all(&dir);
            let mut host = dir.into_os_string().into_vec();
            host.extend_from_slice(rest);
            return std::ffi::CString::new(host).map_or(path.into(), Into::into);
        }
        let Some(root) = &self.sysroot else { return path.into() };
        let guest = std::path::Path::new(std::ffi::OsStr::from_bytes(path.to_bytes()));
        let host = resolve(Some(root), guest, follow);
        if host.as_path() == guest {
            return path.into();
        }
        std::ffi::CString::new(host.into_os_string().into_vec()).map_or(path.into(), Into::into)
    }

    /// Set how `execve` starts a new program (once).
    pub fn set_exec(&self, exec: ExecFn) {
        let _ = self.exec.set(exec);
    }

    /// Set what a forked child must restart (once); see the field.
    pub fn set_on_fork(&self, f: Box<dyn Fn(Arc<Process>) + Send + Sync>) {
        let _ = self.on_fork.set(f);
    }

    /// `execve`: replace this program. Returns only on failure, with a Linux errno.
    pub fn exec(&self, request: &Exec) -> i32 {
        match self.exec.get() {
            Some(exec) => exec(request),
            None => crate::syscall::errno::ENOSYS,
        }
    }

    /// A guest `fork` (or `vfork`, which may copy): fork the host process. Returns the
    /// child's pid in the parent and 0 in the child, where `thread` is the only thread left,
    /// as on Linux. The locks other threads could hold are held across the fork, so the
    /// child never inherits them locked by a thread it does not have.
    pub fn fork(&self, thread: &mut Thread) -> Result<u32, i32> {
        let brk = lock(&self.brk);
        let mut threads = lock(&self.threads);
        let mut registry = lock(&self.thread_signals);
        let mut slots = lock(&self.slots);
        let mut pending = lock(&self.pending);
        let mut own_pending = lock(&thread.signals.pending);
        // The signal pump can hold these while it forwards a signal (it reads `actions`
        // for every one, and the forking thread's `waker` when interrupting a wait): a
        // child that inherited them locked would deadlock on its first syscall.
        let own_waker = lock(&thread.signals.waker);
        let actions = lock(&self.actions);
        let itimer = lock(&self.itimer);
        let mut maps = self.mem.lock_for_fork();
        // SAFETY: plain fork; the child continues on this thread only, holding the locks.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(crate::syscall::errno::EAGAIN);
        }
        if pid > 0 {
            return Ok(pid as u32);
        }
        let new = self.host.identity().pid;
        if thread.tid != self.main_tid.load(Ordering::Relaxed) {
            self.orphaned.store(true, Ordering::Relaxed);
        }
        self.main_tid.store(new, Ordering::Relaxed);
        threads.live = 1;
        let mine = registry.remove(&thread.tid).expect("a running thread is registered");
        registry.clear();
        slots.fill(false);
        slots[mine.slot] = true;
        registry.insert(new, mine);
        // A child starts with no pending signals.
        *pending = Pending::default();
        *own_pending = Pending::default();
        thread.tid = new;
        // The forking thread is in a syscall, so its own TLB readers are idle too.
        maps.forget_other_threads();
        drop((
            brk,
            threads,
            registry,
            slots,
            pending,
            own_pending,
            own_waker,
            actions,
            itimer,
            maps,
        ));
        if let (Some(f), Some(this)) = (self.on_fork.get(), self.this.upgrade()) {
            f(this);
        }
        Ok(0)
    }

    /// `setitimer(ITIMER_REAL)` with `(value, interval)` (a zero value disarms), or just
    /// `getitimer` with `None`. Returns the old `(time left, interval)`.
    pub fn itimer(
        &self,
        new: Option<(std::time::Duration, std::time::Duration)>,
    ) -> (std::time::Duration, std::time::Duration) {
        let now = std::time::Instant::now();
        let mut t = lock(&self.itimer);
        let old = (
            t.deadline.map_or_else(Default::default, |d| d.saturating_duration_since(now)),
            t.interval,
        );
        if let Some((value, interval)) = new {
            t.deadline = (!value.is_zero()).then(|| now + value);
            t.interval = interval;
            if t.deadline.is_some()
                && !t.started
                && let Some(this) = self.this.upgrade()
            {
                t.started = std::thread::Builder::new()
                    .name("rvrun itimer".into())
                    .spawn(move || this.run_itimer())
                    .is_ok();
            }
            self.itimer_changed.notify_all();
        }
        old
    }

    /// The `ITIMER_REAL` thread: sleep until the deadline, send SIGALRM, rearm.
    fn run_itimer(&self) {
        let mut t = lock(&self.itimer);
        loop {
            let now = std::time::Instant::now();
            t = match t.deadline {
                None => self.itimer_changed.wait(t).unwrap_or_else(|e| e.into_inner()),
                Some(due) if now < due => {
                    self.itimer_changed
                        .wait_timeout(t, due - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                }
                Some(due) => {
                    t.deadline = (!t.interval.is_zero()).then(|| due + t.interval);
                    drop(t);
                    let info = Info::Sent { code: 0x80, pid: 0, uid: 0 }; // SI_KERNEL
                    self.send_to_process(signal::SIGALRM, info);
                    lock(&self.itimer)
                }
            };
        }
    }

    /// Signals whose disposition is `SIG_IGN`.
    pub fn ignored_signals(&self) -> u64 {
        let actions = lock(&self.actions);
        (1..=signal::NSIG)
            .filter(|&sig| actions[sig as usize - 1].handler == signal::SIG_IGN)
            .fold(0, |set, sig| set | signal::bit(sig))
    }

    /// The disposition of `sig` (1..=64).
    pub fn action(&self, sig: i32) -> Action {
        lock(&self.actions)[sig as usize - 1]
    }

    /// `rt_sigaction`: set the disposition of `sig`, returning the old one. Signals that
    /// become ignored are dropped from every pending set.
    pub fn set_action(&self, sig: i32, action: Action) -> Action {
        let old = std::mem::replace(&mut lock(&self.actions)[sig as usize - 1], action);
        let ignored = action.handler == signal::SIG_IGN
            || (action.handler == signal::SIG_DFL
                && signal::default_action(sig) != signal::DefaultAction::Terminate);
        if ignored {
            lock(&self.pending).discard(signal::bit(sig));
            for t in lock(&self.thread_signals).values() {
                lock(&t.pending).discard(signal::bit(sig));
            }
        }
        old
    }

    /// Send `sig` to the process (`kill`): any thread that does not block it takes it.
    pub fn send_to_process(&self, sig: i32, info: Info) {
        lock(&self.pending).add(sig, info);
        for t in lock(&self.thread_signals).values() {
            self.interrupt(t, sig);
        }
    }

    /// Send `sig` to thread `tid` (`tgkill`). False if there is no such thread.
    pub fn send_to_thread(&self, tid: u32, sig: i32, info: Info) -> bool {
        let Some(t) = lock(&self.thread_signals).get(&tid).cloned() else { return false };
        lock(&t.pending).add(sig, info);
        self.interrupt(&t, sig);
        true
    }

    /// Make thread `t` take `sig` soon if it can: leave generated code, end a wait of ours,
    /// or return from a blocking host syscall with EINTR.
    fn interrupt(&self, t: &ThreadSignals, sig: i32) {
        if t.mask() & signal::bit(sig) != 0 {
            return;
        }
        self.mem.interrupt_flag(t.slot).store(1, Ordering::SeqCst);
        if let Some(wake) = lock(&t.waker).clone() {
            wake();
        }
        let host = t.host_thread.load(Ordering::SeqCst);
        // SAFETY: pthread_self has no preconditions.
        if host != 0 && host != unsafe { libc::pthread_self() } as usize {
            // SAFETY: the host thread is alive while it is registered (it unregisters
            // before it ends); the signal's handler does nothing (see `wake_handler`).
            unsafe { libc::pthread_kill(host as libc::pthread_t, WAKE_SIGNAL) };
        }
    }

    /// Whether `thread` has a signal it can take now, and if so whether every such signal's
    /// handler restarts interrupted syscalls (`SA_RESTART`).
    pub fn signal_waiting(&self, thread: &Thread) -> Option<bool> {
        let ready = self.pending_for(thread) & !thread.signals.mask();
        if ready == 0 {
            return None;
        }
        let actions = lock(&self.actions);
        let restart = (1..=signal::NSIG)
            .filter(|&sig| ready & signal::bit(sig) != 0)
            .all(|sig| actions[sig as usize - 1].flags & signal::SA_RESTART != 0);
        Some(restart)
    }

    /// A blocking wait of ours for `thread` that signals end (see [`Process::interrupt`]).
    pub fn blocked<'a>(&'a self, thread: &'a Thread) -> Blocked<'a> {
        Blocked { process: self, thread }
    }

    /// `rt_sigsuspend`: wait under `mask` until a signal arrives that it lets through;
    /// its handler runs under `mask` too, and the old mask comes back when it returns.
    pub fn sigsuspend(&self, thread: &mut Thread, mask: u64) -> Result<u64, i32> {
        let old = thread.signals.mask();
        thread.signals.set_mask(mask);
        thread.saved_mask = Some(old);
        let wakeup = Arc::new((Mutex::new(false), Condvar::new()));
        let w = wakeup.clone();
        *lock(&thread.signals.waker) = Some(Arc::new(move || {
            *lock(&w.0) = true;
            w.1.notify_one();
        }));
        let mut woken = lock(&wakeup.0);
        while self.signal_waiting(thread).is_none() {
            woken = wakeup.1.wait(woken).unwrap_or_else(|e| e.into_inner());
            *woken = false;
        }
        drop(woken);
        *lock(&thread.signals.waker) = None;
        Err(crate::syscall::errno::EINTR)
    }

    pub fn thread_exists(&self, tid: u32) -> bool {
        lock(&self.thread_signals).contains_key(&tid)
    }

    /// Signals pending for `thread`, directed at it or at the process.
    pub fn pending_for(&self, thread: &Thread) -> u64 {
        lock(&thread.signals.pending).set | lock(&self.pending).set
    }

    /// Deliver the signals `thread` can take now. Returns how the process ends if one of
    /// them terminates it.
    fn deliver_signals(&self, thread: &mut Thread) -> Option<End> {
        // Clear the flag first: a signal sent after this sets it again.
        self.mem.interrupt_flag(thread.signals.slot).store(0, Ordering::SeqCst);
        loop {
            let blocked = thread.signals.mask();
            let next = lock(&thread.signals.pending).take(blocked);
            let (sig, info) = next.or_else(|| lock(&self.pending).take(blocked))?;
            if let Some(end) = self.deliver(thread, sig, info) {
                return Some(end);
            }
        }
    }

    /// Act on `sig` for `thread`: ignore it, end the process, or run its handler.
    fn deliver(&self, thread: &mut Thread, sig: i32, info: Info) -> Option<End> {
        let action = {
            let mut actions = lock(&self.actions);
            let action = actions[sig as usize - 1];
            if action.flags & signal::SA_RESETHAND != 0 && action.handler > signal::SIG_IGN {
                actions[sig as usize - 1] = Action::default();
            }
            action
        };
        match action.handler {
            signal::SIG_IGN => None,
            signal::SIG_DFL => match signal::default_action(sig) {
                signal::DefaultAction::Terminate => Some(End::Process(Exit::Signal(sig))),
                _ => None,
            },
            _ => {
                let mask = thread.signals.mask();
                let pushed = signal::push_frame(
                    &self.mem,
                    &mut thread.cpu,
                    sig,
                    info,
                    &action,
                    thread.saved_mask.take().unwrap_or(mask),
                    thread.altstack,
                    SIGRETURN_TRAMPOLINE,
                );
                if pushed.is_err() {
                    // No room for the frame: Linux kills with SIGSEGV.
                    return Some(End::Process(Exit::Signal(signal::SIGSEGV)));
                }
                let defer =
                    if action.flags & signal::SA_NODEFER == 0 { signal::bit(sig) } else { 0 };
                thread.signals.set_mask(mask | action.mask | defer);
                None
            }
        }
    }

    /// A trap in guest code: run the guest's handler for the matching signal, or end the
    /// process if there is none (or the signal is blocked, as Linux does for faults).
    fn trap_signal(&self, thread: &mut Thread, trap: Trap) -> Option<End> {
        let (sig, info) = match trap {
            Trap::FetchFault(f) => {
                (signal::SIGSEGV, Info::Fault { code: signal::SEGV_MAPERR, addr: f.addr })
            }
            Trap::LoadFault { fault, .. } | Trap::StoreFault { fault, .. } => {
                (signal::SIGSEGV, Info::Fault { code: signal::SEGV_MAPERR, addr: fault.addr })
            }
            Trap::IllegalInstruction { pc, .. } | Trap::ReservedRoundingMode { pc } => {
                (signal::SIGILL, Info::Fault { code: signal::ILL_ILLOPC, addr: pc })
            }
            Trap::Ebreak { pc } => {
                (signal::SIGTRAP, Info::Fault { code: signal::TRAP_BRKPT, addr: pc })
            }
            Trap::MisalignedAtomic { addr, .. } => {
                (signal::SIGBUS, Info::Fault { code: signal::BUS_ADRALN, addr })
            }
            Trap::Ecall { .. } => unreachable!("ecall is a syscall"),
        };
        let handled = self.action(sig).handler > signal::SIG_IGN;
        if !handled || thread.signals.mask() & signal::bit(sig) != 0 {
            return Some(End::Process(Exit::Trap(trap)));
        }
        self.deliver(thread, sig, info)
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
        let _ = PROT_WRITE;
        if shared {
            // Shared memory is real host shared memory: writes reach the file, and a fork's
            // child shares it. It needs host-page alignment (guest fds are host fds here).
            let at = (flags & (MAP_FIXED | MAP_FIXED_NOREPLACE) != 0).then_some(addr);
            let file = (!anonymous).then_some((fd, offset));
            return self.mem.map_shared(at, len, MMAP_TOP, Prot(prot), file).map_err(|e| match e {
                MapError::Host(e) => e,
                MapError::Invalid => errno::EINVAL,
                _ => errno::ENOMEM,
            });
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
            MapError::Host(e) => e,
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
            Process::spawn(HELLO, &[b"hello".to_vec()], &[], b"hello", host.clone(), None).unwrap();
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
    }

    #[test]
    fn shared_mappings_reach_the_file() {
        use std::os::fd::AsRawFd;
        let (p, _, h) = spawn();
        let path = std::env::temp_dir().join(format!("rvrun-shared-{}", std::process::id()));
        std::fs::write(&path, b"hello").unwrap();
        let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let a = p.mmap(&*h, 0, P, RW, 0x01, file.as_raw_fd(), 0).unwrap();
        assert!(p.mem.is_shared(a));
        let mut b = [0u8; 5];
        p.mem.read(a, &mut b).unwrap();
        assert_eq!(&b, b"hello");
        p.mem.write(a, b"J").unwrap();
        assert_eq!(p.munmap(a, P), Ok(0));
        assert_eq!(std::fs::read(&path).unwrap(), b"Jello", "the write reached the file");
        std::fs::remove_file(&path).unwrap();
        // Host-page alignment is required for a fixed address (16 KiB pages on macOS).
        // SAFETY: sysconf has no preconditions.
        if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64 > P {
            assert_eq!(p.mmap(&*h, 0x1000_1000, P, RW, 0x21 | 0x10, -1, 0), Err(errno::EINVAL));
        }
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
