//! GDB remote debugging (`rvrun -g PORT`), on top of the `gdbstub` crate.
//!
//! While a debugger is attached, the guest runs on the reference interpreter one
//! instruction at a time, so breakpoints, single-stepping, and interrupts are exact. Syscalls
//! are served as usual, so the program's I/O works while being debugged. Breakpoints are
//! kept on our side (GDB's `Z0` packets); guest code is never patched.
//!
//! A fault (e.g. SIGSEGV) stops the guest with that signal so it can be inspected;
//! continuing with the signal delivers it, which (until signal handlers arrive in M5)
//! terminates the guest.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use gdbstub::common::Signal;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::{SingleThreadStopReason, run_blocking};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::base::singlethread::{
    SingleThreadBase, SingleThreadResume, SingleThreadResumeOps, SingleThreadSingleStep,
    SingleThreadSingleStepOps,
};
use gdbstub::target::ext::breakpoints::{
    Breakpoints, BreakpointsOps, SwBreakpoint, SwBreakpointOps,
};
use gdbstub::target::{Target, TargetResult};
use gdbstub_arch::riscv::Riscv64;
use gdbstub_arch::riscv::reg::RiscvCoreRegs;

use crate::interpreter::{self, Trap};
use crate::memory::GUEST_PAGE_SIZE;
use crate::process::{Exit, Process, Thread};
use crate::syscall::{self, Flow, HostOs};

/// Instructions to run between checks for an interrupt (Ctrl-C) from the debugger.
const POLL_INTERVAL: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Continue,
    Step,
}

/// What ended a run of the guest.
enum Event {
    DoneStep,
    Breakpoint,
    /// A fault: stop with this signal, guest still inspectable.
    Fault(Trap),
    Exited(Exit),
    IncomingData,
}

/// The debugged guest: its process, main thread, and host, plus debugger state.
pub struct Debuggee<'a> {
    process: &'a Process,
    thread: &'a mut Thread,
    host: &'a dyn HostOs,
    breakpoints: BTreeSet<u64>,
    mode: Mode,
    /// Signal GDB asked to deliver on resume; any signal is fatal until M5.
    pending_signal: Option<i32>,
    /// Set once the guest has exited or been killed.
    pub exit: Option<Exit>,
}

impl<'a> Debuggee<'a> {
    pub fn new(process: &'a Process, thread: &'a mut Thread, host: &'a dyn HostOs) -> Self {
        Debuggee {
            process,
            thread,
            host,
            breakpoints: BTreeSet::new(),
            mode: Mode::Continue,
            pending_signal: None,
            exit: None,
        }
    }

    fn run(&mut self, mut poll_incoming: impl FnMut() -> bool) -> Event {
        if let Some(sig) = self.pending_signal.take() {
            return Event::Exited(Exit::Signal(sig));
        }
        let mut until_poll = POLL_INTERVAL;
        loop {
            match interpreter::step(&mut self.thread.cpu, &self.process.mem) {
                Ok(()) => {}
                Err(Trap::Ecall { .. }) => {
                    match syscall::dispatch(self.process, self.thread, self.host) {
                        Flow::Continue => {}
                        Flow::ExitThread(status) | Flow::ExitGroup(status) => {
                            return Event::Exited(Exit::Status(status));
                        }
                        Flow::Killed(sig) => return Event::Exited(Exit::Signal(sig)),
                    }
                }
                Err(trap) => return Event::Fault(trap),
            }
            if self.mode == Mode::Step {
                return Event::DoneStep;
            }
            if self.breakpoints.contains(&self.thread.cpu.pc) {
                return Event::Breakpoint;
            }
            until_poll -= 1;
            if until_poll == 0 {
                until_poll = POLL_INTERVAL;
                if poll_incoming() {
                    return Event::IncomingData;
                }
            }
        }
    }
}

/// GDB signal numbers are Linux-like for the ones a fault can produce.
fn gdb_signal(sig: i32) -> Signal {
    match sig {
        4 => Signal::SIGILL,
        5 => Signal::SIGTRAP,
        6 => Signal::SIGABRT,
        7 => Signal::SIGBUS,
        8 => Signal::SIGFPE,
        9 => Signal::SIGKILL,
        11 => Signal::SIGSEGV,
        15 => Signal::SIGTERM,
        n => Signal(n as u8),
    }
}

impl Target for Debuggee<'_> {
    type Arch = Riscv64;
    type Error = &'static str;

    fn base_ops(&mut self) -> BaseOps<'_, Self::Arch, Self::Error> {
        BaseOps::SingleThread(self)
    }

    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadBase for Debuggee<'_> {
    fn read_registers(&mut self, regs: &mut RiscvCoreRegs<u64>) -> TargetResult<(), Self> {
        regs.x = self.thread.cpu.x;
        regs.pc = self.thread.cpu.pc;
        Ok(())
    }

    fn write_registers(&mut self, regs: &RiscvCoreRegs<u64>) -> TargetResult<(), Self> {
        self.thread.cpu.x = regs.x;
        self.thread.cpu.x[0] = 0;
        self.thread.cpu.pc = regs.pc;
        Ok(())
    }

    fn read_addrs(&mut self, start: u64, data: &mut [u8]) -> TargetResult<usize, Self> {
        // Read page by page; stop at the first unmapped page, as GDB expects.
        let mut done = 0;
        while done < data.len() {
            let addr = start + done as u64;
            let chunk =
                ((GUEST_PAGE_SIZE - addr % GUEST_PAGE_SIZE) as usize).min(data.len() - done);
            if self.process.mem.read_privileged(addr, &mut data[done..done + chunk]).is_err() {
                break;
            }
            done += chunk;
        }
        Ok(done)
    }

    fn write_addrs(&mut self, start: u64, data: &[u8]) -> TargetResult<(), Self> {
        // The debugger may write anywhere mapped, including read-only code. The reference
        // interpreter refetches every instruction, so no cache needs flushing here; the
        // engine's caches see the generation bump below once debugging ends.
        self.process
            .mem
            .write_privileged(start, data)
            .map_err(|_| gdbstub::target::TargetError::NonFatal)?;
        self.process.mem.invalidate_caches();
        Ok(())
    }

    fn support_resume(&mut self) -> Option<SingleThreadResumeOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadResume for Debuggee<'_> {
    fn resume(&mut self, signal: Option<Signal>) -> Result<(), Self::Error> {
        self.mode = Mode::Continue;
        self.pending_signal = signal.map(|s| s.0 as i32);
        Ok(())
    }

    fn support_single_step(&mut self) -> Option<SingleThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl SingleThreadSingleStep for Debuggee<'_> {
    fn step(&mut self, signal: Option<Signal>) -> Result<(), Self::Error> {
        self.mode = Mode::Step;
        self.pending_signal = signal.map(|s| s.0 as i32);
        Ok(())
    }
}

impl Breakpoints for Debuggee<'_> {
    fn support_sw_breakpoint(&mut self) -> Option<SwBreakpointOps<'_, Self>> {
        Some(self)
    }
}

impl SwBreakpoint for Debuggee<'_> {
    fn add_sw_breakpoint(&mut self, addr: u64, _kind: usize) -> TargetResult<bool, Self> {
        self.breakpoints.insert(addr);
        Ok(true)
    }

    fn remove_sw_breakpoint(&mut self, addr: u64, _kind: usize) -> TargetResult<bool, Self> {
        Ok(self.breakpoints.remove(&addr))
    }
}

/// The `gdbstub` event loop: run the guest until it stops or the debugger speaks.
pub struct EventLoop<'a>(PhantomData<&'a ()>);

impl<'a> run_blocking::BlockingEventLoop for EventLoop<'a> {
    type Target = Debuggee<'a>;
    type Connection = Box<dyn ConnectionExt<Error = std::io::Error>>;
    type StopReason = SingleThreadStopReason<u64>;

    #[allow(clippy::type_complexity)]
    fn wait_for_stop_reason(
        target: &mut Debuggee<'a>,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<Self::StopReason>,
        run_blocking::WaitForStopReasonError<&'static str, std::io::Error>,
    > {
        let event = target.run(|| conn.peek().map(|b| b.is_some()).unwrap_or(true));
        let reason = match event {
            Event::IncomingData => {
                let byte = conn.read().map_err(run_blocking::WaitForStopReasonError::Connection)?;
                return Ok(run_blocking::Event::IncomingData(byte));
            }
            Event::DoneStep => SingleThreadStopReason::DoneStep,
            Event::Breakpoint => SingleThreadStopReason::SwBreak(()),
            Event::Fault(trap) => {
                let sig = Exit::Trap(trap).signal().unwrap_or(5);
                SingleThreadStopReason::Signal(gdb_signal(sig))
            }
            Event::Exited(exit) => {
                target.exit = Some(exit);
                match exit {
                    Exit::Status(code) => SingleThreadStopReason::Exited(code as u8),
                    other => {
                        SingleThreadStopReason::Terminated(gdb_signal(other.signal().unwrap_or(9)))
                    }
                }
            }
        };
        Ok(run_blocking::Event::TargetStopped(reason))
    }

    fn on_interrupt(_target: &mut Debuggee<'a>) -> Result<Option<Self::StopReason>, &'static str> {
        // The guest only runs inside wait_for_stop_reason, so it is already paused.
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT)))
    }
}

/// How a debugging session ended.
pub enum Session {
    /// The guest exited or was killed while attached.
    Exited(Exit),
    /// The debugger detached (or disconnected); the guest should keep running.
    Detached,
    /// The debugger asked to kill the guest.
    Killed,
}

/// Serve one debugger connection until it detaches or the guest ends.
pub fn serve(
    conn: Box<dyn ConnectionExt<Error = std::io::Error>>,
    process: &Process,
    thread: &mut Thread,
    host: &dyn HostOs,
) -> Result<Session, String> {
    use gdbstub::stub::{DisconnectReason, GdbStub};
    let mut target = Debuggee::new(process, thread, host);
    let reason =
        GdbStub::new(conn).run_blocking::<EventLoop<'_>>(&mut target).map_err(|e| e.to_string())?;
    Ok(match reason {
        DisconnectReason::TargetExited(_) | DisconnectReason::TargetTerminated(_) => {
            Session::Exited(target.exit.unwrap_or(Exit::Status(0)))
        }
        DisconnectReason::Disconnect => Session::Detached,
        DisconnectReason::Kill => Session::Killed,
    })
}
