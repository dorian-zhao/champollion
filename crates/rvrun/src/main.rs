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
//! - `RVRUN_CACHE=0` disables the persistent code cache: by default, translations are saved
//!   per executable in `~/Library/Caches/champollion` (or `$RVRUN_CACHE_DIR`) and reused by
//!   the next run of the same executable.

use std::os::unix::ffi::OsStrExt;
use std::process::ExitCode;
use std::sync::Arc;

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

    let (proc, mut main_thread) =
        match Process::spawn(&file, &argv, &envp, path.as_bytes(), Arc::new(DarwinHost)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rvrun: {}: cannot load: {e:?}", path.to_string_lossy());
                return ExitCode::from(126);
            }
        };
    let strace = std::env::var_os("RVRUN_STRACE").is_some_and(|v| v != "0");
    proc.log.strace.store(strace, std::sync::atomic::Ordering::Relaxed);
    let cache = CodeCache::new(&file);
    if std::env::var_os("RVRUN_JIT").is_none_or(|v| v != "0") {
        match champollion_rv::jit::Jit::new() {
            Ok(mut jit) => {
                jit.stats.detailed = std::env::var_os("RVRUN_STATS").is_some_and(|v| v != "0");
                if std::env::var_os("RVRUN_PROF").is_some() {
                    jit.start_profiling();
                } else if let Some(cache) = &cache {
                    cache.restore(&mut jit, &proc.mem);
                }
                main_thread.jit = Some(jit);
            }
            Err(e) => eprintln!("rvrun: JIT unavailable ({e}); interpreting"),
        }
    }

    let start = std::time::Instant::now();
    let exit = match gdb_port {
        None => proc.run(&mut main_thread),
        Some(port) => match debug(port, &proc, &mut main_thread) {
            Ok(Session::Exited(exit)) => exit,
            Ok(Session::Detached) => proc.run(&mut main_thread),
            Ok(Session::Killed) => return ExitCode::from(128 + 9),
            Err(e) => {
                eprintln!("rvrun: gdb: {e}");
                return ExitCode::from(1);
            }
        },
    };
    if let (Some(cache), Some(jit)) = (&cache, &main_thread.jit)
        && jit.profiler.is_none()
    {
        cache.save(jit);
    }
    if let (Some(path), Some(p)) =
        (std::env::var_os("RVRUN_PROF"), main_thread.jit.as_mut().and_then(|j| j.profiler.as_mut()))
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
) -> Result<Session, String> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    eprintln!("rvrun: waiting for a debugger on 127.0.0.1:{port} (gdb: target remote :{port})");
    let (stream, _) = listener.accept().map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    gdb::serve(Box::new(stream), proc, thread)
}

/// Where a program's translations are kept between runs (see `champollion_rv::jit::persist`).
struct CodeCache {
    path: std::path::PathBuf,
    /// Identifies the executable and this build of rvrun, which together determine the code.
    key: u64,
}

impl CodeCache {
    fn new(file: &[u8]) -> Option<Self> {
        use std::hash::{Hash, Hasher};
        if std::env::var_os("RVRUN_CACHE").is_some_and(|v| v == "0") {
            return None;
        }
        let dir = match std::env::var_os("RVRUN_CACHE_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::PathBuf::from(std::env::var_os("HOME")?)
                .join("Library/Caches/champollion"),
        };
        // SipHash with fixed keys: stable for a given build of rvrun, which is all a key
        // must be (the build is part of it).
        let mut h = std::hash::DefaultHasher::new();
        file.hash(&mut h);
        env!("CARGO_PKG_VERSION").hash(&mut h);
        let exe = std::env::current_exe().and_then(std::fs::metadata).ok()?;
        exe.len().hash(&mut h);
        exe.modified().ok()?.hash(&mut h);
        let key = h.finish();
        Some(Self { path: dir.join(format!("{key:016x}.jit")), key })
    }

    fn restore(
        &self,
        jit: &mut champollion_rv::jit::Jit,
        mem: &champollion_rv::memory::GuestMemory,
    ) {
        let Ok(bytes) = std::fs::read(&self.path) else { return };
        if let Err(e) = jit.restore(&bytes, self.key, mem) {
            eprintln!("rvrun: ignoring code cache {}: {e}", self.path.display());
        }
    }

    /// Save the translations if this run added any. Best effort: a cache that cannot be
    /// written only costs the next run its warm start.
    fn save(&self, jit: &champollion_rv::jit::Jit) {
        if jit.stats.translations == 0 {
            return;
        }
        let Some(bytes) = jit.save(self.key) else { return };
        let Some(dir) = self.path.parent() else { return };
        // Write a private file, then rename it into place: concurrent runs never see a
        // partial snapshot.
        let tmp = self.path.with_extension(format!("tmp{}", std::process::id()));
        let written = std::fs::create_dir_all(dir)
            .and_then(|()| std::fs::write(&tmp, &bytes))
            .and_then(|()| std::fs::rename(&tmp, &self.path));
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        evict(dir, CACHE_LIMIT);
    }
}

/// Upper bound on the code cache directory's size.
const CACHE_LIMIT: u64 = 512 << 20;

/// Delete the least recently written snapshots in `dir` until the rest fit in `limit` bytes.
/// Snapshots of executables (or rvrun builds) that are gone accumulate otherwise.
fn evict(dir: &std::path::Path, limit: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<_> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            let meta = e.metadata().ok()?;
            let is_snapshot = e.path().extension().is_some_and(|x| x == "jit");
            (meta.is_file() && is_snapshot).then(|| (meta.modified().ok(), meta.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if total <= limit {
            break;
        }
        if std::fs::remove_file(path).is_ok() {
            total -= len;
        }
    }
}
