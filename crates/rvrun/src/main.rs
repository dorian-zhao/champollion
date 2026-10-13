//! `rvrun`: run a static riscv64 Linux executable on macOS.
//!
//! Environment:
//! - `RVRUN_STRACE=1` traces every syscall to stderr.

use std::os::unix::ffi::OsStrExt;
use std::process::ExitCode;

use champollion_rv::host::darwin::DarwinHost;
use champollion_rv::process::{Exit, Process};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: rvrun <riscv64-linux-elf> [args...]");
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

    let mut host = DarwinHost;
    let mut proc = match Process::spawn(&file, &argv, &envp, path.as_bytes(), &mut host) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("rvrun: {}: cannot load: {e:?}", path.to_string_lossy());
            return ExitCode::from(126);
        }
    };
    proc.log.strace = std::env::var_os("RVRUN_STRACE").is_some_and(|v| v != "0");

    match proc.run(&mut host) {
        Exit::Status(status) => ExitCode::from(status as u8),
        exit @ Exit::Trap(trap) => {
            // Match the shell convention for a process killed by a signal.
            let sig = exit.signal().unwrap();
            eprintln!("rvrun: guest killed by signal {sig}: {trap:?}\n{:?}", proc.cpu);
            ExitCode::from(128 + sig as u8)
        }
    }
}
