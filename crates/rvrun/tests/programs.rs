//! Real programs (built by tests/programs/build.sh): their output must match the same
//! source built natively for arm64, which produced the committed `.expected` files.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests").join(rel)
}

fn rvrun(args: &[&std::ffi::OsStr], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvrun"))
        .args(args)
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
