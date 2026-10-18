# Changelog

All notable changes to Champollion are documented here. Releases are tagged in
git and published on GitHub.

## [0.1.1] - 2026-10-18

### Added

- RVA23 userland extension coverage: `Zba`, `Zbb`, `Zbs`, `Zicond`, `Zcb`,
  `Zbkb`, `Zbc`, `Zbkx`, `Zcmop`, `Zfa` (including `fli`), `Zfhmin`, `Zihpm`,
  and `Zimop`.
- An experimental RVV 1.0 subset for Ubuntu 26.04 userlands: vector masks,
  indexed and segmented memory operations, whole-register moves, reductions,
  floating-point vector ops, `Zvfhmin`, rotates, shifts, `vandn`, `vslideup`,
  and e8 `vrgather`.
- The riscv-arch-test ACT suite, with committed fixtures and QEMU signature
  baselines, wired into `cargo test`.
- A daily benchmark gate for startup, Lua, SQLite, native arm64, and Rosetta 2,
  plus the `RVRUN_PROF`/`jitprof.py` generated-code profiler.

### Changed

- JIT: m2 vector groups, paired vector memory moves, load-to-op reuse, vector
  arithmetic residency, vector float scalar boxing, internal loop back edges,
  loop unrolling, fused rotate shapes, dead m2 load-store elimination, and
  correct snapshot invalidation.
- Loader and cache: map executables from the file, parse the guest image from a
  read-only mapping, key the persistent cache by identity, and support the
  cache under `binfmt_misc`.
- Syscalls and VM mode: `flock`, `openat2`, `pselect6`, Darwin
  `SOCK_SEQPACKET`, `/proc/cpuinfo`, `/proc/version`, chunked guest path reads,
  lazy `/proc/self/exe`, JIT `close`/`getpid`, read/write buffer fast paths,
  and an inline Linux `newfstatat`.
- The guest's self-description now comes from one machine table shared by
  `AT_HWCAP`, `riscv_hwprobe`, `/proc/cpuinfo`, `uname`, and `/proc/version`.

### Fixed

- The four blockers listed in v0.1.0: `pselect6` deadlocks, Darwin
  `SOCK_SEQPACKET`, missing `flock`, and `openat2` flag translation.
- A logical-immediate panic at bit 63, a stale base-register cache under vector
  accesses, `vsetvl` with neither `rd` nor `rs1`, and `vfork` exec failures.

### Performance

Representative local measurements on a MacBook Pro M2 2022, minimum of runs,
release build, October 18, 2026:

| Workload | Champollion | native arm64 | Rosetta 2 |
| --- | ---: | ---: | ---: |
| hello | 3.47 ms | 1.47 ms | 4.73 ms |
| Lua start | 4.29 ms | 1.72 ms | 5.21 ms |
| Lua compute | 34.79 ms | 15.36 ms | 40.72 ms |
| SQLite script | 23.80 ms | 10.20 ms | 17.67 ms |
| SQLite CTE | 241.98 ms | 116.93 ms | 167.47 ms |
| vector ALU | 116.4 ms | 56.2 ms | — |
| nanoGPT-shaped GEMM | 386.9 ms | 130.1 ms | 124.1 ms |

Champollion is faster than Rosetta 2 on the startup and Lua cases and still
slower on the two SQLite cases and the GEMM case. The remaining gaps are tracked
in [docs/design.md](docs/design.md#known-gaps).

### Known issues

- The RVV subset is experimental and is not advertised to guests; enable it
  with `RVRUN_V=1`.
- [#11](https://github.com/dorian-zhao/champollion/issues/11): ACT coverage
  beyond the committed 227 cases.
- [#9](https://github.com/dorian-zhao/champollion/issues/9): further RVV JIT
  coverage, including wider-element gather and more unmasked NEON paths.
- [#3](https://github.com/dorian-zhao/champollion/issues/3): `rvcontainer`
  keeps the container cgroup at 1 GiB, so highly parallel builds can be
  OOM-killed.
- VM mode unpacks the image on every run; Direct mode does not cache
  dynamically linked programs yet.

## [0.1.0] - 2026-10-14

First public snapshot: Direct mode on macOS, VM mode with `rvcontainer` and
`binfmt_misc`, an arm64 JIT with chaining, persistent code caching, threads,
signals, dynamic linking, and the initial real-program test suite.

[0.1.1]: https://github.com/dorian-zhao/champollion/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dorian-zhao/champollion/releases/tag/v0.1.0
