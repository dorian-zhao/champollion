//! Dynamically linked programs, run against the Alpine riscv64 userland that
//! tests/sysroot/build.sh fetches (not committed). Tests that need it are skipped, with a
//! note, when it is absent.

use std::path::PathBuf;
use std::process::{Command, Output};

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

fn sysroot() -> Option<PathBuf> {
    let dir = tests_dir().join("sysroot/alpine");
    if dir.join("lib/ld-musl-riscv64.so.1").exists() {
        Some(dir)
    } else {
        eprintln!("skipped: no sysroot (run tests/sysroot/build.sh)");
        None
    }
}

fn rvrun(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rvrun")).env("RVRUN_CACHE", "0").args(args).output().unwrap()
}

#[test]
fn missing_loader_is_explained() {
    let out = rvrun(&[tests_dir().join("fixtures/dynamic").as_os_str()]);
    assert_eq!(out.status.code(), Some(126));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("/lib/ld-musl-riscv64.so.1") && stderr.contains("-L"), "{stderr}");
}

#[test]
fn dynamically_linked_program() {
    let Some(root) = sysroot() else { return };
    let out =
        rvrun(&["-L".as_ref(), root.as_os_str(), tests_dir().join("fixtures/dynamic").as_os_str()]);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello from a dynamically linked program, sqrt 1.414\n"
    );
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn alpine_busybox() {
    let Some(root) = sysroot() else { return };
    let busybox = root.join("bin/busybox");
    let run = |args: &[&str]| {
        let mut all = vec!["-L".as_ref(), root.as_os_str(), busybox.as_os_str()];
        all.extend(args.iter().map(std::ffi::OsStr::new));
        rvrun(&all)
    };
    let out = run(&["echo", "hello from Alpine"]);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello from Alpine\n");
    assert_eq!(out.stderr, b"", "no unsupported syscalls");
    let out = run(&["sh", "-c", "for i in 1 2 3; do echo $((i * i)); done"]);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1\n4\n9\n");
    // Subshells and pipelines fork and exec; /usr/bin/wc is an absolute symlink to
    // /bin/busybox, resolved inside the sysroot.
    let out = run(&["sh", "-c", "x=$(echo sub); echo $x $(echo a b c | wc -w)"]);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "sub 3\n");
    // /etc comes from the sysroot.
    let out = run(&["cat", "/etc/alpine-release"]);
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "3.24.2");
}
