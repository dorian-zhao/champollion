//! Real programs (built by tests/programs/build.sh): their output must match the same
//! source built natively for arm64, which produced the committed `.expected` files.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests").join(rel)
}

fn rvrun(args: &[&std::ffi::OsStr], stdin: &[u8]) -> Output {
    rvrun_with(args, stdin, &[("RVRUN_CACHE", "0".as_ref())])
}

fn rvrun_with(args: &[&std::ffi::OsStr], stdin: &[u8], env: &[(&str, &std::ffi::OsStr)]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvrun"))
        .args(args)
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}

fn assert_matches_native(out: &Output, expected: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");
    let expected = std::fs::read_to_string(path(expected)).unwrap();
    let got = String::from_utf8_lossy(&out.stdout);
    for (i, (g, e)) in got.lines().zip(expected.lines()).enumerate() {
        assert_eq!(g, e, "line {}", i + 1);
    }
    assert_eq!(got.lines().count(), expected.lines().count(), "stderr: {stderr}");
}

#[test]
fn code_cache_reuses_translations() {
    let dir = std::env::temp_dir().join(format!("rvrun-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let env = [("RVRUN_CACHE_DIR", dir.as_os_str()), ("RVRUN_STATS", "1".as_ref())];
    let lua = path("fixtures/programs/lua");
    let script = path("programs/basics.lua");
    let sqlite = path("fixtures/programs/sqlite3");
    let sql = std::fs::read(path("programs/basics.sql")).unwrap();
    for run in ["cold", "warm"] {
        let out = rvrun_with(&[lua.as_os_str(), script.as_os_str()], b"", &env);
        assert_matches_native(&out, "programs/basics.lua.expected");
        let stats = String::from_utf8_lossy(&out.stderr);
        if run == "warm" {
            // The same run again: everything comes from the cache.
            assert!(stats.contains(" 0 translations,"), "{stats}");
            assert!(!stats.contains(" 0 restored,"), "{stats}");
        }
        let out = rvrun_with(&[sqlite.as_os_str()], &sql, &env);
        assert_matches_native(&out, "programs/basics.sql.expected");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn lua() {
    let script = path("programs/basics.lua");
    let out = rvrun(&[path("fixtures/programs/lua").as_os_str(), script.as_os_str()], b"");
    assert_matches_native(&out, "programs/basics.lua.expected");
}

#[test]
fn sqlite_in_memory() {
    let sql = std::fs::read(path("programs/basics.sql")).unwrap();
    let out = rvrun(&[path("fixtures/programs/sqlite3").as_os_str(), ":memory:".as_ref()], &sql);
    assert_matches_native(&out, "programs/basics.sql.expected");
}

#[test]
fn sqlite_on_disk() {
    // Exercises file locking, fsync, and journal creation/removal across two processes.
    let dir = std::env::temp_dir().join(format!("champollion-sqlite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("test.db");
    let sqlite = path("fixtures/programs/sqlite3");
    let args = [sqlite.as_os_str(), db.as_os_str()];

    let write = b"CREATE TABLE kv(k TEXT PRIMARY KEY, v INTEGER);\n\
        BEGIN; INSERT INTO kv VALUES('a', 1), ('b', 2); COMMIT;\n\
        UPDATE kv SET v = v * 10;\n";
    assert_eq!(rvrun(&args, write).status.code(), Some(0));
    let out = rvrun(&args, b"SELECT k, v FROM kv ORDER BY k; PRAGMA integrity_check;\n");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "a|10\nb|20\nok\n");

    let leftovers: Vec<_> =
        std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(leftovers, ["test.db"], "the rollback journal is removed");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rust_unit_tests() {
    // A static-PIE Rust test binary, run the way `cargo test` does with rvrun as the
    // runner, so libtest runs the tests on threads in parallel.
    let out = rvrun(&[path("fixtures/programs/rust-tests").as_os_str()], b"");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("test result: ok. 5 passed; 0 failed"), "{stdout}");
}
