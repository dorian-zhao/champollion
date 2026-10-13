//! Runs the riscv-tests suites (built by tests/riscv-tests/build.sh) on the interpreter.
//!
//! Each test is a static executable that ends with the Linux `exit` syscall: status 0 on
//! pass, otherwise the number of the first failing test case.

use std::path::{Path, PathBuf};

use champollion_rv::CpuState;
use champollion_rv::interpreter::{Trap, run};
use champollion_rv::loader;

const SYS_EXIT: u64 = 93;
const MAX_STEPS: u64 = 1_000_000;

fn fixtures(suite: &str) -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(suite);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}; run tests/riscv-tests/build.sh", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    paths
}

/// Run one test binary; `Ok(())` on pass, otherwise a description of what went wrong.
fn run_test(path: &Path) -> Result<(), String> {
    let file = std::fs::read(path).map_err(|e| e.to_string())?;
    let elf = loader::parse(&file).map_err(|e| format!("load: {e:?}"))?;
    let mut mem = loader::load_flat(&elf, 0).ok_or("load: no segments")?;
    let mut cpu = CpuState::new(elf.entry);

    match run(&mut cpu, &mut mem, MAX_STEPS) {
        (Some(Trap::Ecall { .. }), _) if cpu.x[17] == SYS_EXIT => match cpu.x[10] {
            0 => Ok(()),
            n => Err(format!("failed test case #{n}")),
        },
        (Some(trap), n) => Err(format!("trap after {n} steps: {trap:?}\n{cpu:?}")),
        (None, _) => Err(format!("no exit after {MAX_STEPS} steps\n{cpu:?}")),
    }
}

fn run_suite(suite: &str, min_tests: usize) {
    let paths = fixtures(suite);
    assert!(
        paths.len() >= min_tests,
        "{suite}: only {} tests found",
        paths.len()
    );

    let failures: Vec<String> = paths
        .iter()
        .filter_map(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            run_test(p).err().map(|e| format!("{suite}/{name}: {e}"))
        })
        .collect();

    assert!(
        failures.is_empty(),
        "{}/{} {suite} tests failed:\n{}",
        failures.len(),
        paths.len(),
        failures.join("\n")
    );
}

#[test]
fn rv64ui() {
    run_suite("rv64ui", 53);
}
