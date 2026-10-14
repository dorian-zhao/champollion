//! `rvrun`: run a riscv64 Linux executable on macOS.
//!
//! Usage: `rvrun [-g PORT] [-L SYSROOT] [-0 ARGV0] <riscv64-linux-elf or #! script> [args...]`
//!
//! - `-0 ARGV0` passes ARGV0 as the guest's `argv[0]` instead of the path (used when a guest
//!   `execve` starts a new rvrun).
//! - `-L SYSROOT` looks up the guest's absolute paths (its dynamic loader, libraries,
//!   `/etc` files) under SYSROOT first, like `qemu-riscv64 -L`; also `RVRUN_SYSROOT`.
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
//!   per executable in `~/Library/Caches/champollion` (on Linux `$XDG_CACHE_HOME/champollion`
//!   or `~/.cache/champollion`; or `$RVRUN_CACHE_DIR`) and reused by the next run of the same
//!   executable.

use std::os::unix::ffi::OsStrExt;
use std::process::ExitCode;
use std::sync::Arc;

use champollion_rv::gdb::{self, Session};
use champollion_rv::process::Process;

const USAGE: &str =
    "usage: rvrun [-g PORT] [-L SYSROOT] [-0 ARGV0] <riscv64-linux-elf or script> [args...]";

fn main() -> ExitCode {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut gdb_port = None;
    let mut argv0 = None;
    let mut sysroot = std::env::var_os("RVRUN_SYSROOT").map(std::path::PathBuf::from);
    loop {
        match args.first().and_then(|a| a.to_str()) {
            Some("-g") => {
                let port = args.get(1).and_then(|p| p.to_str()?.parse::<u16>().ok());
                let Some(port) = port else {
                    eprintln!("rvrun: -g needs a port number\n{USAGE}");
                    return ExitCode::from(2);
                };
                gdb_port = Some(port);
            }
            Some("-0") => {
                let Some(name) = args.get(1) else {
                    eprintln!("rvrun: -0 needs an argument\n{USAGE}");
                    return ExitCode::from(2);
                };
                argv0 = Some(name.clone());
            }
            Some("-L") => {
                let Some(dir) = args.get(1) else {
                    eprintln!("rvrun: -L needs a directory\n{USAGE}");
                    return ExitCode::from(2);
                };
                sysroot = Some(dir.into());
            }
            _ => break,
        }
        args.drain(..2);
    }
    if args.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    if let Some(name) = argv0 {
        args.insert(0, name); // argv[0], then the path to load
    } else {
        args.insert(0, args[0].clone());
    }
    // A script runs its `#!` interpreter (a guest path), with the script as an argument,
    // as Linux's binfmt_script does; scripts may name scripts, up to a few levels.
    let mut path = std::path::PathBuf::from(args.remove(1));
    let mut file;
    let mut depth = 0;
    loop {
        file = match std::fs::read(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("rvrun: {}: {e}", path.display());
                return ExitCode::from(127);
            }
        };
        let Some((interp, arg)) = shebang(&file) else { break };
        depth += 1;
        if depth > 4 {
            eprintln!("rvrun: {}: too many levels of #! interpreters", path.display());
            return ExitCode::from(126);
        }
        // As on Linux: the interpreter, its optional argument, the script (as given), then
        // the script's arguments; the original argv[0] is dropped.
        let mut new_args = vec![interp.clone()];
        new_args.extend(arg);
        new_args.push(path.clone().into_os_string());
        new_args.extend(args.drain(1..));
        args = new_args;
        path = champollion_rv::process::resolve(
            sysroot.as_deref(),
            std::path::Path::new(&interp),
            true,
        );
    }
    let path = path.into_os_string();

    // The guest gets `args` (argv[0] first) and inherits our environment.
    let argv: Vec<Vec<u8>> = args.iter().map(|a| a.as_bytes().to_vec()).collect();
    // (Except what an exec'ing rvrun passed us about signals.)
    let signal_set =
        |name| std::env::var(name).ok().and_then(|v| u64::from_str_radix(&v, 16).ok()).unwrap_or(0);
    let (ignored, mask) = (signal_set("RVRUN_SIGIGN"), signal_set("RVRUN_SIGMASK"));
    let envp: Vec<Vec<u8>> = std::env::vars_os()
        .filter(|(k, _)| k != "RVRUN_SIGIGN" && k != "RVRUN_SIGMASK")
        .map(|(k, v)| [k.as_bytes(), b"=", v.as_bytes()].concat())
        .collect();

    let (proc, mut main_thread) = match Process::spawn(
        &file,
        &argv,
        &envp,
        path.as_bytes(),
        Arc::new(champollion_rv::host::Native::default()),
        sysroot.clone(),
    ) {
        Ok(p) => p,
        Err(champollion_rv::process::SpawnError::Interp(interp, e)) => {
            eprintln!(
                "rvrun: {}: dynamically linked, and its loader {} is missing ({e}); point -L \
                 at a riscv64 sysroot",
                path.to_string_lossy(),
                interp.display()
            );
            return ExitCode::from(126);
        }
        Err(e) => {
            eprintln!("rvrun: {}: cannot load: {e:?}", path.to_string_lossy());
            return ExitCode::from(126);
        }
    };
    proc.set_exec(Box::new(move |req| exec(req, sysroot.as_deref())));
    for sig in (1..=champollion_rv::signal::NSIG).filter(|&s| ignored & (1 << (s - 1)) != 0) {
        let ignore = champollion_rv::signal::Action {
            handler: champollion_rv::signal::SIG_IGN,
            ..Default::default()
        };
        if champollion_rv::signal::UNBLOCKABLE & (1 << (sig - 1)) == 0 {
            proc.set_action(sig, ignore);
        }
    }
    main_thread.signals.set_mask(mask);
    champollion_rv::host::forward_signals(proc.clone());
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
                proc.jit.store(true, std::sync::atomic::Ordering::Relaxed);
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
    exit.terminate(&main_thread.cpu)
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

#[cfg(target_os = "macos")]
fn default_cache_dir() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches/champollion"))
}

#[cfg(not(target_os = "macos"))]
fn default_cache_dir() -> Option<std::path::PathBuf> {
    let base = match std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        Some(dir) => std::path::PathBuf::from(dir),
        None => std::path::PathBuf::from(std::env::var_os("HOME")?).join(".cache"),
    };
    Some(base.join("champollion"))
}

impl CodeCache {
    fn new(file: &[u8]) -> Option<Self> {
        use std::hash::{Hash, Hasher};
        if std::env::var_os("RVRUN_CACHE").is_some_and(|v| v == "0") {
            return None;
        }
        let dir = match std::env::var_os("RVRUN_CACHE_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => default_cache_dir()?,
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

/// The interpreter and optional argument of a `#!` line, if `file` starts with one.
fn shebang(file: &[u8]) -> Option<(std::ffi::OsString, Option<std::ffi::OsString>)> {
    use std::os::unix::ffi::OsStringExt;
    let line = file.strip_prefix(b"#!")?;
    let line = &line[..line.iter().position(|&b| b == b'\n').unwrap_or(line.len())];
    let line = line.trim_ascii();
    let (interp, arg) = match line.iter().position(u8::is_ascii_whitespace) {
        Some(i) => (&line[..i], Some(line[i..].trim_ascii())),
        None => (line, None),
    };
    if interp.is_empty() {
        return None;
    }
    let arg = arg.filter(|a| !a.is_empty()).map(|a| std::ffi::OsString::from_vec(a.to_vec()));
    Some((std::ffi::OsString::from_vec(interp.to_vec()), arg))
}

/// A guest `execve`: replace this process with a new rvrun running `req` (with the same
/// sysroot, and our RVRUN_* settings unless the guest's environment sets them). Returns a
/// Linux errno if the exec fails.
fn exec(req: &champollion_rv::process::Exec, sysroot: Option<&std::path::Path>) -> i32 {
    use std::ffi::OsStr;
    use std::os::unix::process::CommandExt;
    let Ok(exe) = std::env::current_exe() else { return 2 };
    let mut cmd = std::process::Command::new(&exe);
    let argv0 = req.argv.first().map_or(req.path.as_os_str(), |a| OsStr::from_bytes(a));
    cmd.arg("-0").arg(argv0);
    if let Some(root) = sysroot {
        cmd.arg("-L").arg(root);
    }
    cmd.arg(&req.path).args(req.argv.iter().skip(1).map(|a| OsStr::from_bytes(a)));
    cmd.env_clear();
    let mut keys = Vec::new();
    for var in &req.envp {
        let (k, v) = match var.iter().position(|&b| b == b'=') {
            Some(i) => (&var[..i], &var[i + 1..]),
            None => (&var[..], &[][..]),
        };
        cmd.env(OsStr::from_bytes(k), OsStr::from_bytes(v));
        keys.push(k);
    }
    for (k, v) in std::env::vars_os() {
        if k.as_bytes().starts_with(b"RVRUN_") && !keys.contains(&k.as_bytes()) {
            cmd.env(k, v);
        }
    }
    // Ignored signals and the signal mask survive execve.
    cmd.env("RVRUN_SIGIGN", format!("{:x}", req.ignored));
    cmd.env("RVRUN_SIGMASK", format!("{:x}", req.mask));
    let err = cmd.exec(); // only returns on failure
    // The common errno values agree with Linux.
    err.raw_os_error().unwrap_or(5)
}
