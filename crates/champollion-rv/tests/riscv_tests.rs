//! Runs the riscv-tests suites (built by tests/riscv-tests/build.sh) on the reference
//! interpreter and on the caching engine, and checks that both agree.
//!
//! Each test is a static executable that ends with the Linux `exit` syscall: status 0 on
//! pass, otherwise the number of the first failing test case.

use std::path::{Path, PathBuf};

use champollion_rv::CpuState;
use champollion_rv::engine::Engine;
use champollion_rv::interpreter::{Trap, run};
use champollion_rv::loader;
use champollion_rv::memory::GuestMemory;

const SYS_EXIT: u64 = 93;
const MAX_STEPS: u64 = 1_000_000;

fn fixtures(suite: &str) -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(suite);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}; run tests/riscv-tests/build.sh", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    paths
}

/// Run one test binary twice: on the reference interpreter (flat memory) and on the
/// caching engine (real guest address space). Both must pass and end in identical CPU
/// states, which differentially tests the engine's block cache and TLB.
fn run_test(path: &Path) -> Result<(), String> {
    let file = std::fs::read(path).map_err(|e| e.to_string())?;
    let elf = loader::parse(&file).map_err(|e| format!("load: {e:?}"))?;

    let flat = loader::load_flat(&elf, 0).ok_or("load: no segments")?;
    let mut oracle = CpuState::new(elf.entry);
    let oracle_result = run(&mut oracle, &flat, MAX_STEPS);
    check_exit("interpreter", &oracle, oracle_result)?;

    let mem = GuestMemory::new().map_err(|e| e.to_string())?;
    loader::map_elf(&mem, &elf).map_err(|e| format!("map: {e:?}"))?;
    let mut cpu = CpuState::new(elf.entry);
    let engine_result = Engine::new().run(&mut cpu, &mem, MAX_STEPS);
    check_exit("engine", &cpu, engine_result)?;

    if cpu != oracle || engine_result.1 != oracle_result.1 {
        return Err(format!(
            "engine diverged from the interpreter ({} vs {} steps)\nengine:\n{cpu:?}\ninterpreter:\n{oracle:?}",
            engine_result.1, oracle_result.1
        ));
    }
    Ok(())
}

fn check_exit(who: &str, cpu: &CpuState, result: (Option<Trap>, u64)) -> Result<(), String> {
    match result {
        (Some(Trap::Ecall { .. }), _) if cpu.x[17] == SYS_EXIT => match cpu.x[10] {
            0 => Ok(()),
            n => Err(format!("{who}: failed test case #{n}")),
        },
        (Some(trap), n) => Err(format!("{who}: trap after {n} steps: {trap:?}\n{cpu:?}")),
        (None, _) => Err(format!("{who}: no exit after {MAX_STEPS} steps\n{cpu:?}")),
    }
}

fn run_suite(suite: &str, min_tests: usize) {
    let paths = fixtures(suite);
    assert!(paths.len() >= min_tests, "{suite}: only {} tests found", paths.len());

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

#[test]
fn rv64um() {
    run_suite("rv64um", 13);
}

#[test]
fn rv64ua() {
    run_suite("rv64ua", 19);
}

#[test]
fn rv64uc() {
    run_suite("rv64uc", 1);
}

#[test]
fn rv64ui_compressed() {
    run_suite("rv64ui-c", 53);
}

#[test]
fn rv64um_compressed() {
    run_suite("rv64um-c", 13);
}

#[test]
fn rv64ua_compressed() {
    run_suite("rv64ua-c", 19);
}

#[test]
fn rv64uf() {
    run_suite("rv64uf", 11);
}

#[test]
fn rv64ud() {
    run_suite("rv64ud", 12);
}

#[test]
fn rv64ud_compressed() {
    run_suite("rv64ud-c", 12);
}
