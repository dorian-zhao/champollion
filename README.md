# Champollion

Run RISC-V (RV64GC) Linux programs on arm64: Apple silicon Macs, and arm64 Linux hosts.

Champollion is a user-mode binary translator from riscv64 Linux to arm64. It started as a learning project, which aims at what Rosetta 2 does for x86 programs: no emulated machine, no guest kernel, just the program translated to native code and running at close to native speed. A guest runs in one of two modes, sharing one translator:

- **Direct mode, on macOS:** `rvrun ./program` runs a riscv64 Linux executable directly as a Mac process and Linux system calls are translated to Darwin ones. The daily build → test → debug loop, with no VM to boot or sync.
- **VM mode, against a real Linux kernel:** `rvcontainer run debian:trixie` runs riscv64 container images, each in a lightweight arm64 Linux VM whose kernel is the guest's — rvrun is registered as the kernel's handler for riscv64 programs and system calls pass straight through. On an arm64 Linux host the same Linux build runs directly, no VM needed: see [On arm64 Linux](#on-arm64-linux-binfmt_misc).

The name comes from Jean-François Champollion, who deciphered the Rosetta Stone 🪨.

## Status

| | |
| --- | --- |
| Instruction set | RV64GC: I, M, A, F, D, C, Zicsr, Zifencei |
| Execution | A JIT to arm64 with block chaining and an inline jump cache, a caching interpreter, and a reference interpreter |
| Programs | Static and dynamically linked (musl or glibc), multithreaded, with signals, sockets, `fork`/`execve`, and shared memory |
| Known to run | busybox, Lua, SQLite, Go programs, Rust test binaries, CPython 3.14 (Alpine), and Debian trixie's userland: bash, coreutils, apt, dpkg, perl, python3 |
| Debugging | A GDB remote stub (`rvrun -g PORT`) |

Milestones M1 through M5 (interpreter, JIT, threads, signals, dynamic linking) are complete. M6 (VM mode) is working. See [`docs/design.md`](docs/design.md) for the design, the milestones, and the [known gaps](docs/design.md#known-gaps).

### Performance

Compared with the same programs built natively for arm64 (Direct mode on a MacBook Pro M2 2022, 16G):

| Benchmark | Slowdown vs. native |
| --- | --- |
| CPU-bound C (sieve, hashing, array traffic) | 1.5× |
| n-body (floating point) | 1.37× |
| Lua 5.4 script | 2.0× |
| SQLite (300k-row insert, index, aggregates) | 1.87× |

The target is at most 2× slower than native. In the same Linux VM, rvrun runs Lua 4.8× and SQLite 4.2× faster than `qemu-riscv64`.

## Requirements

- Direct mode: a Mac with Apple silicon running macOS 14.4 or later, and Rust (stable, edition 2024).
- VM mode: on macOS, macOS 26 or later and Xcode 26 or later for `rvcontainer`, plus Docker (OrbStack recommended) to build the Linux build; on an arm64 Linux host, just Docker for that build.
- To rebuild the test programs: [`zig`](https://ziglang.org) 0.17 or later, used as a riscv64 C cross compiler.

## Direct mode: `rvrun`

```sh
cargo build --release
target/release/rvrun tests/fixtures/hello
```

Any static riscv64 Linux executable works, for example one cross-compiled with zig:

```sh
zig cc -target riscv64-linux-musl -static -O2 hello.c -o hello
target/release/rvrun ./hello
```

Dynamically linked programs need a riscv64 root file system to find their loader and libraries in. `-L` names it, like `qemu-riscv64 -L`. `tests/sysroot/build.sh` sets up Alpine's:

```sh
tests/sysroot/build.sh                  # Alpine minirootfs; `build.sh python` adds CPython
target/release/rvrun -L tests/sysroot/alpine tests/sysroot/alpine/bin/busybox sh
```

Options and environment variables:

| | |
| --- | --- |
| `-L SYSROOT` (or `RVRUN_SYSROOT`) | Look up the guest's absolute paths under SYSROOT first |
| `-g PORT` | Wait for GDB on 127.0.0.1:PORT (`0` picks a free port), then `riscv64-elf-gdb prog -ex 'target remote :PORT'` |
| `RVRUN_STRACE=1` | Trace every system call |
| `RVRUN_STATS=1` | Print instructions retired, speed, and JIT counters on exit |
| `RVRUN_JIT=0` | Use the caching interpreter instead of the JIT |
| `RVRUN_PROF=FILE` | Sample generated code; render the report with `tools/jitprof.py FILE` |
| `RVRUN_CACHE=0` | Disable the persistent code cache (`~/Library/Caches/champollion`) |

## VM mode

VM mode runs the guest against a real Linux kernel, two ways: inside an arm64 Linux VM on a Mac (`rvcontainer`), or directly on an arm64 Linux host as the kernel's `binfmt_misc` handler.

### On macOS: `rvcontainer`

`rvcontainer` is a small Swift launcher built on [apple/containerization](https://github.com/apple/containerization). It pulls an image's `linux/riscv64` variant and boots an arm64 Linux VM with Virtualization.framework and shares rvrun's Linux build into the VM and registers it with `binfmt_misc`, the same way Containerization registers Rosetta for amd64 images.

```sh
launcher/fetch-kernel.sh                # once: installs Kata Containers' kernel
launcher/build.sh                       # builds rvrun for Linux and rvcontainer
launcher/.build/release/rvcontainer run docker.io/library/debian:trixie-slim bash
```

```console
root@rv-1c2a90f3:/# uname -m
riscv64
root@rv-1c2a90f3:/# apt-get update && apt-get install -y jq
```

`rvcontainer run` accepts `-v HOST:CONTAINER` to share a directory, `-e NAME=VALUE`, `-w DIR`, `--cpus`, `--memory`, and `--no-network`. Containers have networking by default. Each run starts from a fresh copy of the image.

### On arm64 Linux: `binfmt_misc`

On an arm64 Linux host the VM is the host itself: register the Linux build as the kernel's
handler for riscv64 programs, and they run like native ones — including inside containers.
This is the same registration `rvcontainer` performs inside its VM.

```sh
tools/linux.sh build --release -p rvrun       # produces target/linux/release/rvrun
sudo install -m755 target/linux/release/rvrun /usr/local/bin/rvrun-riscv64
sudo sh -c 'printf "%s\n" ":rvrun-riscv64:M::\x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/local/bin/rvrun-riscv64:POCF" > /proc/sys/fs/binfmt_misc/register'
sudo docker run --platform linux/riscv64 alpine uname -m
riscv64
```

If `/proc/sys/fs/binfmt_misc` is not mounted yet, run
`sudo mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc` first.

On Linux, guest memory sits at the same addresses as on the host. Therefore, guest pointers are host pointers, and system calls rvrun does not emulate go to the kernel unchanged.

## How it works

- **Decoder and interpreters.** Every instruction decodes to a compact `Inst`. A reference interpreter (fetch and decode each step) is the permanent oracle. A caching interpreter adds decoded blocks and a per-thread software TLB.
- **JIT.** Basic blocks are translated to arm64 with most guest registers pinned to host registers, chained directly to each other, and joined by an inline jump cache for indirect jumps. Floating point runs on the host FPU with RISC-V rounding and NaN rules. Guest memory faults in generated code become precise guest faults through a signal handler. Translations of an executable are cached on disk between runs.
- **Memory.** The guest address space is one 256 GiB host reservation and guest addresses are masked into it, so generated code needs no bounds checks and cannot reach host memory.
- **System calls.** Linux system calls are served by a host backend: Darwin translation on macOS, and passthrough on Linux. Threads, signals, futexes, and memory mappings are emulated in the runtime itself, the same way on both hosts.

[`docs/design.md`](docs/design.md) has the details, the measurements behind each decision, and a decision log.

## Repository layout

| Path | Contents |
| --- | --- |
| `crates/champollion-rv` | The translator: decoder, interpreters, JIT, memory, loader, system calls, host backends |
| `crates/rvrun` | The `rvrun` command and its end-to-end tests |
| `crates/softfloat` | Berkeley SoftFloat 3, the reference for floating-point semantics |
| `launcher` | `rvcontainer`, VM mode's Swift launcher |
| `tests` | Fixtures, riscv-tests, real programs, and sysroot setup |
| `tools` | The profiler's report renderer and `linux.sh`, which runs cargo on arm64 Linux in a container |
| `docs/design.md` | Design brief |

## Testing

```sh
cargo test --release                    # everything on macOS
tools/linux.sh                          # the same suite on arm64 Linux, in a container
FUZZ_CASES=100000 cargo test --release -p champollion-rv fuzz
launcher/test.sh                        # VM mode smoke test (boots VMs, needs network)
```

The suite includes riscv-tests on all three execution modes, unit tests for every instruction, and a differential fuzzer. The fuzzer runs random programs on the interpreter and the JIT and requires identical traps, registers, and memory. It also includes end-to-end runs of real programs compared against native builds. Test binaries are committed. `tests/fixtures/build.sh`, `tests/programs/build.sh`, and `tests/riscv-tests/build.sh` rebuild them with zig.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Third-party code keeps its own license: riscv-tests, Berkeley SoftFloat 3, and Lua are git submodules under `third_party/`. The committed test programs are built from them, from SQLite (public domain), and from Go's and Rust's standard libraries.
