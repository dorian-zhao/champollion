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
    // `RVRUN_TRACE_STARTUP=1`: where a short-lived guest's time goes before its code runs.
    struct Trace(Option<std::time::Instant>);
    impl Trace {
        fn mark(&self, what: &str) {
            if let Some(t) = self.0 {
                eprintln!("[start] {:>7.0}us {what}", t.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    let trace = Trace(std::env::var_os("RVRUN_TRACE_STARTUP").map(|_| std::time::Instant::now()));
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut gdb_port = None;
    let mut argv0 = None;
    let mut sysroot = std::env::var_os("RVRUN_SYSROOT").map(std::path::PathBuf::from);
    if preserved_argv0() && args.len() >= 2 {
        // Started by binfmt_misc for a guest program: `<path> <argv[0]> [args...]`, no options.
        argv0 = Some(args.remove(1));
    } else {
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
    let mut opened = binfmt_fd();
    loop {
        // Parse from a read-only mapping: the guest image is often megabytes, and
        // reading it into a `Vec` on every exec showed up in the spawn trace.
        let read = match opened.take() {
            Some(f) => map_read_only(&f),
            None => std::fs::File::open(&path)
                .map_err(|e| e.to_string())
                .and_then(|f| map_read_only(&f)),
        };
        file = match read {
            Ok(f) => f,
            Err(e) => {
                eprintln!("rvrun: {}: {e}", path.display());
                return ExitCode::from(127);
            }
        };
        let Some((interp, arg)) = shebang(file) else { break };
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
    trace.mark("args");

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

    let image = std::fs::File::open(&path).ok();
    let (proc, mut main_thread) = match Process::spawn(
        file,
        &argv,
        &envp,
        path.as_bytes(),
        Arc::new(champollion_rv::host::Native::default()),
        sysroot.clone(),
        image.as_ref().map(std::os::unix::io::AsRawFd::as_raw_fd),
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
    trace.mark("spawn");
    // Started by binfmt_misc (VM mode), the kernel itself knows how to run guest programs.
    let native_exec = preserved_argv0() && sysroot.is_none();
    proc.set_exec(Box::new(move |req| {
        if native_exec { exec_native(req) } else { exec(req, sysroot.as_deref()) }
    }));
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
    trace.mark("signals");
    let cache = CodeCache::new(std::path::Path::new(&path));
    trace.mark("cache");
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
                trace.mark("jit");
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
    trace.mark("ran");
    if let (Some(cache), Some(jit)) = (&cache, &main_thread.jit)
        && jit.profiler.is_none()
    {
        if std::env::var_os("RVRUN_DIAG").is_some() {
            eprintln!(
                "[cache] cacheable={} translations={} restored={}",
                jit.cacheable(),
                jit.stats.translations,
                jit.stats.restored
            );
        }
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
    // The kernel reclaims the address space at exit; unmapping the guest reservation
    // explicitly costs about a millisecond per invocation on macOS.
    proc.mem.leak_at_exit();
    if let Some(jit) = main_thread.jit.as_mut() {
        jit.leak_at_exit();
    }
    trace.mark("cleanup");
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

/// The program binfmt_misc opened for us (with the `O` flag, implied by `C`), which it
/// passes as `AT_EXECFD`. Taken (and closed after reading), so the guest never sees it.
/// A read-only mapping of `file`, for the rest of the run: parsing the guest image out
/// of it faults in only the pages it reads, where reading a megabyte-large binary into a
/// `Vec` on every exec cost a few hundred microseconds in the spawn trace. The mapping
/// is the CLI's own image and deliberately never unmapped.
fn map_read_only(file: &std::fs::File) -> Result<&'static [u8], String> {
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len == 0 {
        return Ok(&[]);
    }
    // SAFETY: a fresh read-only private mapping of a file we own.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len as usize,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            std::os::fd::AsRawFd::as_raw_fd(file),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().to_string());
    }
    // SAFETY: `len` valid read-only bytes that live for the rest of the process.
    Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len as usize) })
}

#[cfg(target_os = "linux")]
fn binfmt_fd() -> Option<std::fs::File> {
    use std::os::fd::FromRawFd;
    // SAFETY: errno is thread-local; getauxval has no preconditions.
    let fd = unsafe {
        *libc::__errno_location() = 0;
        let fd = libc::getauxval(libc::AT_EXECFD);
        if fd == 0 && *libc::__errno_location() == libc::ENOENT {
            return None;
        }
        fd
    };
    // SAFETY: the kernel opened it for us and nothing else owns it.
    Some(unsafe { std::fs::File::from_raw_fd(fd as i32) })
}

#[cfg(not(target_os = "linux"))]
fn binfmt_fd() -> Option<std::fs::File> {
    None
}

/// Whether binfmt_misc started us for a guest program with the `P` (preserve-argv0) flag,
/// passing the program's path and then its original `argv[0]` (Linux 5.12+ says so in
/// `AT_FLAGS`).
#[cfg(target_os = "linux")]
fn preserved_argv0() -> bool {
    const AT_FLAGS_PRESERVE_ARGV0: u64 = 1;
    // SAFETY: getauxval has no preconditions.
    let flags = unsafe { libc::getauxval(libc::AT_FLAGS) };
    flags & AT_FLAGS_PRESERVE_ARGV0 != 0
}

#[cfg(not(target_os = "linux"))]
fn preserved_argv0() -> bool {
    false
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
    fn new(path: &std::path::Path) -> Option<Self> {
        use std::hash::{Hash, Hasher};
        let diag = std::env::var_os("RVRUN_DIAG").is_some();
        macro_rules! bail {
            ($what:expr) => {{
                if diag {
                    eprintln!("[cache] disabled: {}", $what);
                }
                return None;
            }};
        }
        if std::env::var_os("RVRUN_CACHE").is_some_and(|v| v == "0") {
            return None;
        }
        let dir = match std::env::var_os("RVRUN_CACHE_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => match default_cache_dir() {
                Some(dir) => dir,
                None => bail!("no cache directory"),
            },
        };
        // SipHash with fixed keys: stable for a given build of rvrun, which is all a key
        // must be (the build is part of it). The guest file enters through its identity -
        // length, timestamps, device and inode, plus the first and last 64 KiB - instead
        // of its full contents: hashing a 2 MiB static binary took about 0.6 ms on every
        // exec, and any rewrite of the file moves one of these.
        let mut h = std::hash::DefaultHasher::new();
        let meta = match std::fs::metadata(path) {
            Ok(meta) => meta,
            Err(e) => bail!(format!("{}: {e}", path.display())),
        };
        meta.len().hash(&mut h);
        match meta.modified() {
            Ok(m) => m.hash(&mut h),
            Err(e) => bail!(format!("modified: {e}")),
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            meta.dev().hash(&mut h);
            meta.ino().hash(&mut h);
        }
        if let Ok(mut f) = std::fs::File::open(path) {
            use std::io::{Read, Seek, SeekFrom};
            let mut head = [0u8; 1 << 16];
            let mut tail = [0u8; 1 << 16];
            if let Ok(n) = f.read(&mut head) {
                head[..n].hash(&mut h);
            }
            if meta.len() > (1 << 17)
                && f.seek(SeekFrom::End(-(1 << 16))).is_ok()
                && let Ok(n) = f.read(&mut tail)
            {
                tail[..n].hash(&mut h);
            }
        }
        env!("CARGO_PKG_VERSION").hash(&mut h);
        // binfmt_misc's `F` flag pins a snapshot of the interpreter that the container's
        // namespace cannot reach by path, so `current_exe` fails for the Linux handler;
        // `/proc/self/exe` still stats the live inode.
        let exe = match std::env::current_exe()
            .and_then(std::fs::metadata)
            .or_else(|_| std::fs::metadata("/proc/self/exe"))
        {
            Ok(exe) => exe,
            Err(e) => bail!(format!("current_exe: {e}")),
        };
        exe.len().hash(&mut h);
        match exe.modified() {
            Ok(m) => m.hash(&mut h),
            Err(e) => bail!(format!("exe modified: {e}")),
        }
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

/// A guest `execve` under binfmt_misc: hand it to the kernel as it is. It runs rvrun again
/// for riscv64 programs and `#!` interpreters, also inside a container or chroot where our
/// own path is not visible. Signal state goes along as in [`exec`]. Returns a Linux errno
/// if the exec fails.
#[cfg(target_os = "linux")]
fn exec_native(req: &champollion_rv::process::Exec) -> i32 {
    use std::ffi::CString;
    let cstr = |b: &[u8]| CString::new(b).ok();
    let mut envp: Vec<Vec<u8>> = req
        .envp
        .iter()
        .filter(|v| !v.starts_with(b"RVRUN_SIGIGN=") && !v.starts_with(b"RVRUN_SIGMASK="))
        .cloned()
        .collect();
    envp.push(format!("RVRUN_SIGIGN={:x}", req.ignored).into_bytes());
    envp.push(format!("RVRUN_SIGMASK={:x}", req.mask).into_bytes());
    let (Some(path), Some(argv), Some(envp)) = (
        cstr(req.path.as_os_str().as_bytes()),
        req.argv.iter().map(|a| cstr(a)).collect::<Option<Vec<_>>>(),
        envp.iter().map(|v| cstr(v)).collect::<Option<Vec<_>>>(),
    ) else {
        return 22; // EINVAL: a NUL inside an argument
    };
    let ptrs =
        |v: &[CString]| v.iter().map(|c| c.as_ptr()).chain([std::ptr::null()]).collect::<Vec<_>>();
    // SAFETY: NUL-terminated strings in NULL-terminated arrays; returns only on failure.
    unsafe { libc::execve(path.as_ptr(), ptrs(&argv).as_ptr(), ptrs(&envp).as_ptr()) };
    std::io::Error::last_os_error().raw_os_error().unwrap_or(5)
}

#[cfg(not(target_os = "linux"))]
fn exec_native(_: &champollion_rv::process::Exec) -> i32 {
    unreachable!("binfmt_misc exists only on Linux")
}
