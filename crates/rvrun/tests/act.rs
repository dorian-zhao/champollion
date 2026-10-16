//! Runs the committed riscv-arch-test fixtures (tests/fixtures/act): each ELF writes
//! its signature region to stdout and exits 0 (see tests/act/); the committed `.sig`
//! is the qemu-riscv64 baseline, so a mismatch is a divergence from the reference
//! implementation. The tests per extension live in tests/act/dirs.txt.

use std::path::PathBuf;
use std::process::Command;

fn act_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/act")
}

#[test]
fn signatures_match_the_reference() {
    let dir = act_dir();
    let mut elfs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}; run tests/act/build.sh", dir.display()))
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            (path.extension().is_some_and(|ext| ext == "elf")).then_some(path)
        })
        .collect();
    elfs.sort();
    assert!(!elfs.is_empty(), "no ACT fixtures; run tests/act/build.sh");

    for elf in elfs {
        let name = elf.file_stem().unwrap().to_string_lossy().into_owned();
        let want = std::fs::read(elf.with_extension("sig"))
            .unwrap_or_else(|e| panic!("{name}: unreadable .sig: {e}"));
        let out = Command::new(env!("CARGO_BIN_EXE_rvrun"))
            .arg(&elf)
            .env("CHAMPOLLION_TEST", "yes")
            .env("RVRUN_CACHE", "0")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{name}: exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout == want,
            "{name}: signature of {} bytes differs from the {} byte reference",
            out.stdout.len(),
            want.len()
        );
    }
}
