# riscv-arch-test fixtures

The [RISC-V Architectural Certification Tests](https://github.com/riscv/riscv-arch-test)
(ACT4) are ~700 generated assembly tests per rv64i directory, self-checking through a
signature region. Ours run as ordinary static Linux executables: the environment here
makes the test write its signature to stdout and exit through the Linux syscall, so the
same ELF runs under qemu-riscv64 and rvrun.

- `dirs.txt` lists the directories built into `tests/fixtures/act/` (committed, so
  `cargo test` needs neither zig nor the submodule).
- `env/` is the overlay: a user-mode `rvmodel_macros.h`, a hand-written DUT config, and
  a `sail_macros.h` shadow (upstream's replaces the DUT halt with a tohost spin that
  only the Sail reference model watches).
- `link.ld` is upstream's qemu DUT script with `.text.rvmodel` moved to an explicit
  expression address (see the comment there).
- Each `<name>.sig` is qemu-riscv64's stdout for `<name>.elf`, i.e. the reference
  signature. The runner (`crates/rvrun/tests/act.rs`) compares rvrun's stdout against
  it, so a mismatch is a divergence from the reference implementation.

Rebuild: `git submodule update --init third_party/riscv-arch-test && tests/act/build.sh`
(needs `zig`); refresh the baselines with `tests/act/build.sh signatures` where
`qemu-riscv64` exists (see the comment at the top of `build.sh`).
