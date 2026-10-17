//! End-to-end tests: run the committed riscv64 fixtures through the `rvrun` binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

fn rvrun(args: &[&std::ffi::OsStr], stdin: &[u8]) -> Output {
    rvrun_with_jit(args, stdin, true)
}

fn rvrun_with_jit(args: &[&std::ffi::OsStr], stdin: &[u8], jit: bool) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvrun"))
        .args(args)
        .env("CHAMPOLLION_TEST", "yes")
        .env("RVRUN_CACHE", "0")
        .env("RVRUN_JIT", if jit { "1" } else { "0" })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn hello() {
    let out = rvrun(&[fixture("hello").as_os_str()], b"");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello from riscv64\n");
    assert_eq!(out.stderr, b"");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn threads() {
    let out = rvrun(&[fixture("threads").as_os_str()], b"");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "atomic 800000 locked 800000 results 280\nping-pong done\n"
    );
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));

    // exit() in another thread ends the process, though main is blocked in pthread_join.
    let out = rvrun(&[fixture("threads").as_os_str(), "exit-group".as_ref()], b"");
    assert_eq!(out.status.code(), Some(7));

    // After main's pthread_exit, the process lives on until its last thread exits.
    let out = rvrun(&[fixture("threads").as_os_str(), "main-exits".as_ref()], b"");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "child outlived main\n");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn processes() {
    let out = rvrun(&[fixture("processes").as_os_str()], b"");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fork: exited 42\n\
         fork: killed by 15\n\
         pipe: through a pipe, getppid ok 1\n\
         child argv0=renamed args=4 last=y env=inherited\n\
         execve: exited 0\n\
         child argv0=spawned args=3 last=z env=inherited\n\
         posix_spawn: exited 0\n\
         execve missing: -1\n"
    );
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn vfork() {
    // A vfork (CLONE_VM | CLONE_VFORK) child that fails to exec reports the errno to its
    // waiting parent, the way glibc's posix_spawn needs it (issue #8).
    let out = rvrun(&[fixture("vfork").as_os_str()], b"");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "vfork exec missing: ENOENT\n\
         vfork exec: exited 7\n\
         vfork exec retry: exited 7\n\
         vfork exit: exited 5\n"
    );
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn signals() {
    let out = rvrun(&[fixture("signals").as_os_str()], b"");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "handler ran 2 times\n\
         siginfo ok\n\
         blocked: pending 1, ran 2\n\
         unblocked: ran 3\n\
         ignored: ran 3\n\
         caught fault 1 at 0x10\n\
         caught fault 2 at 0x20\n\
         alternate stack used\n\
         resumed after illegal instruction with 42\n\
         signal to thread delivered\n\
         signal to spinning thread delivered\n\
         sigsuspend woke for SIGCHLD\n\
         nanosleep: EINTR, time left reported\n\
         read: restarted\n"
    );
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    // Default actions: SIGTERM and abort()'s SIGABRT kill the process (SIGABRT is reported
    // as status 134 instead, so macOS files no crash report for rvrun).
    // rvrun dies by SIGTERM itself, as the guest would on Linux.
    let out = rvrun(&[fixture("signals").as_os_str(), "term".as_ref()], b"");
    assert_eq!(std::os::unix::process::ExitStatusExt::signal(&out.status), Some(15));
    let out = rvrun(&[fixture("signals").as_os_str(), "pipe".as_ref()], b"");
    assert_eq!(std::os::unix::process::ExitStatusExt::signal(&out.status), Some(13));
    let out = rvrun(&[fixture("signals").as_os_str(), "abort".as_ref()], b"");
    assert_eq!(out.status.code(), Some(128 + 6));
}

#[test]
fn syscalls() {
    let dir = std::env::temp_dir().join(format!("champollion-run-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = rvrun(
        &[fixture("syscalls").as_os_str(), dir.as_os_str(), "second arg".as_ref()],
        b"from stdin\n",
    );
    std::fs::remove_dir_all(&dir).unwrap();

    let stdout = String::from_utf8_lossy(&out.stdout);
    let failed: Vec<&str> = stdout.lines().filter(|l| !l.starts_with("ok ")).collect();
    assert!(
        failed.is_empty(),
        "failed checks: {failed:?}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let source = include_str!("../../../tests/fixtures/syscalls.c");
    let checks = source.lines().filter(|l| l.trim_start().starts_with("CHECK(\"")).count();
    assert_eq!(stdout.lines().count(), checks, "{stdout}");
    assert_eq!(out.stderr, b"", "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(7));
}

/// Calls rvrun does not emulate reach the host kernel on Linux, where guest memory is in the
/// identity layout (on macOS they fail with ENOSYS).
#[cfg(target_os = "linux")]
#[test]
fn unemulated_calls_pass_through() {
    let dir = std::env::temp_dir().join(format!("champollion-pass-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = rvrun(&[fixture("passthrough").as_os_str(), dir.as_os_str()], b"");
    std::fs::remove_dir_all(&dir).unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let source = include_str!("../../../tests/fixtures/passthrough.c");
    let checks = source.lines().filter(|l| l.trim_start().starts_with("CHECK(\"")).count();
    assert_eq!(stdout.lines().filter(|l| l.starts_with("ok ")).count(), checks, "{stdout}");
    assert_eq!(out.stderr, b"", "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn segfault_exits_like_sigsegv() {
    let out = rvrun(&[fixture("segfault").as_os_str()], b"");
    assert_eq!(out.status.code(), Some(128 + 11));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("signal 11") && stderr.contains("addr: 16"), "{stderr}");
}

#[test]
fn load_errors() {
    let out = rvrun(&["/nonexistent/champollion".as_ref()], b"");
    assert_eq!(out.status.code(), Some(127));
    // A host binary (arm64 Mach-O, or an arm64 ELF on Linux) is not a RISC-V ELF.
    let out = rvrun(&[env!("CARGO_BIN_EXE_rvrun").as_ref()], b"");
    assert_eq!(out.status.code(), Some(126));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("NotElf") || stderr.contains("WrongMachine(183)"), "{stderr}");
    let out = rvrun(&[], b"");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn floating_point_matches_native() {
    // fp.expected is the output of the same C source built natively for arm64.
    let out = rvrun(&[fixture("fp").as_os_str()], b"");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let expected = include_str!("../../../tests/fixtures/fp.expected");
    let got = String::from_utf8_lossy(&out.stdout);
    for (i, (g, e)) in got.lines().zip(expected.lines()).enumerate() {
        assert_eq!(g, e, "line {}", i + 1);
    }
    assert_eq!(got.lines().count(), expected.lines().count());
}

#[test]
fn rotate_sequences_match_the_interpreter() {
    let path = fixture("rotate");
    let args = [path.as_os_str()];
    let jit = rvrun(&args, b"");
    let interp = rvrun_with_jit(&args, b"", false);
    let stderr = String::from_utf8_lossy(&jit.stderr);
    assert_eq!(jit.status.code(), Some(0), "{stderr}");
    assert_eq!(interp.status.code(), Some(0), "{}", String::from_utf8_lossy(&interp.stderr));
    assert_eq!(jit.stdout, b"ok rotate sequences\n");
    assert_eq!(jit.stdout, interp.stdout);
}
