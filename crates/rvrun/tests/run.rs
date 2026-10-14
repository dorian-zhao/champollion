//! End-to-end tests: run the committed riscv64 fixtures through the `rvrun` binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

fn rvrun(args: &[&std::ffi::OsStr], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvrun"))
        .args(args)
        .env("CHAMPOLLION_TEST", "yes")
        .env("RVRUN_CACHE", "0")
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
    // A host (arm64 Mach-O) binary is not a RISC-V ELF.
    let out = rvrun(&[env!("CARGO_BIN_EXE_rvrun").as_ref()], b"");
    assert_eq!(out.status.code(), Some(126));
    assert!(String::from_utf8_lossy(&out.stderr).contains("NotElf"));
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
