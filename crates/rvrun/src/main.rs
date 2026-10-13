//! `rvrun`: run a static riscv64 Linux executable on macOS.
//!
//! Usage: `rvrun [-g PORT] <riscv64-linux-elf> [args...]`
//!
//! - `-g PORT` waits for a GDB remote connection on 127.0.0.1:PORT before running, with the
//!   guest stopped at its entry point (like `qemu-riscv64 -g`). Port 0 picks a free port;
//!   the one chosen is printed.
//!
//! Environment:
//! - `RVRUN_STRACE=1` traces every syscall to stderr.
//! - `RVRUN_STATS=1` prints instructions retired and execution speed on exit.
//! - `RVRUN_JIT=0` runs on the caching interpreter instead of the JIT.
//! - `RVRUN_PROF=FILE` samples generated code and writes a report of the hottest blocks to
//!   FILE (render it with `tools/jitprof.py FILE`).

use std::os::unix::ffi::OsStrExt;
use std::process::ExitCode;

use champollion_rv::gdb::{self, Session};
use champollion_rv::host::darwin::DarwinHost;
use champollion_rv::process::{Exit, Process};

const USAGE: &str = "usage: rvrun [-g PORT] <riscv64-linux-elf> [args...]";

fn main() -> ExitCode {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut gdb_port = None;
    if args.first().is_some_and(|a| a == "-g") {
        let port = args.get(1).and_then(|p| p.to_str()?.parse::<u16>().ok());
        let Some(port) = port else {
            eprintln!("rvrun: -g needs a port number\n{USAGE}");
            return ExitCode::from(2);
        };
        gdb_port = Some(port);
        args.drain(..2);
    }
    let Some(path) = args.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let file = match std::fs::read(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("rvrun: {}: {e}", path.to_string_lossy());
            return ExitCode::from(127);
        }
    };

    // The guest sees its own path as argv[0], and inherits our environment.
    let argv: Vec<Vec<u8>> = args.iter().map(|a| a.as_bytes().to_vec()).collect();
    let envp: Vec<Vec<u8>> =
        std::env::vars_os().map(|(k, v)| [k.as_bytes(), b"=", v.as_bytes()].concat()).collect();

    let host = DarwinHost;
    let (mut proc, mut main_thread) =
        match Process::spawn(&file, &argv, &envp, path.as_bytes(), &host) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rvrun: {}: cannot load: {e:?}", path.to_string_lossy());
                return ExitCode::from(126);
            }
        };
    proc.log.strace = std::env::var_os("RVRUN_STRACE").is_some_and(|v| v != "0");
    if std::env::var_os("RVRUN_JIT").is_none_or(|v| v != "0") {
        match champollion_rv::jit::Jit::new() {
            Ok(mut jit) => {
                jit.stats.detailed = std::env::var_os("RVRUN_STATS").is_some_and(|v| v != "0");
                if std::env::var_os("RVRUN_PROF").is_some() {
                    jit.start_profiling();
                }
                main_thread.jit = Some(jit);
            }
            Err(e) => eprintln!("rvrun: JIT unavailable ({e}); interpreting"),
        }
    }

    let start = std::time::Instant::now();
    let exit = match gdb_port {
        None => proc.run(&mut main_thread, &host),
        Some(port) => match debug(port, &proc, &mut main_thread, &host) {
            Ok(Session::Exited(exit)) => exit,
            Ok(Session::Detached) => proc.run(&mut main_thread, &host),
            Ok(Session::Killed) => return ExitCode::from(128 + 9),
            Err(e) => {
                eprintln!("rvrun: gdb: {e}");
                return ExitCode::from(1);
            }
        },
    };
    if let (Some(path), Some(p)) =
        (std::env::var_os("RVRUN_PROF"), main_thread.jit.as_ref().and_then(|j| j.profiler.as_ref()))
        && let Err(e) = std::fs::write(&path, p.report(20))
    {
        eprintln!("rvrun: cannot write profile: {e}");
    }
    if std::env::var_os("RVRUN_STATS").is_some_and(|v| v != "0") {
        let secs = start.elapsed().as_secs_f64();
        let n = main_thread.cpu.instret;
        eprintln!("[rvrun] {n} instructions in {secs:.3} s = {:.1} MIPS", n as f64 / secs / 1e6);
        if let Some(jit) = &main_thread.jit {
            eprintln!("[rvrun] {}", jit.stats.report());
        }
    }
    match exit {
        Exit::Signal(sig) => {
            eprintln!("rvrun: guest killed by signal {sig}");
            ExitCode::from(128 + sig as u8)
        }
        Exit::Status(status) => ExitCode::from(status as u8),
        exit @ Exit::Trap(trap) => {
            // Match the shell convention for a process killed by a signal.
            let sig = exit.signal().unwrap();
            eprintln!("rvrun: guest killed by signal {sig}: {trap:?}\n{:?}", main_thread.cpu);
            ExitCode::from(128 + sig as u8)
        }
    }
}

/// Wait for one debugger connection on 127.0.0.1:`port` and serve it.
fn debug(
    port: u16,
    proc: &Process,
    thread: &mut champollion_rv::process::Thread,
    host: &DarwinHost,
) -> Result<Session, String> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    eprintln!("rvrun: waiting for a debugger on 127.0.0.1:{port} (gdb: target remote :{port})");
    let (stream, _) = listener.accept().map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    gdb::serve(Box::new(stream), proc, thread, host)
}
