# Champollion: Design Brief for a RISC-V Translator on macOS

Oct 13, 2026 · Dorian Verlaine

## Goals and Positioning

Champollion is a user-mode binary translator that runs RISC-V Linux executables directly on macOS arm64, without a VM: `rvrun ./hello`. It is a learning project with no production use case. The goal is to implement CS61C (ISA, CPU) and CS162 (syscalls, threads, virtual memory) in reverse, then add a JIT on top.

- **Route: B-lite.** Run natively on macOS; Linux syscalls are caught by our own runtime and translated into Darwin calls. Start with static, single-threaded programs that use few syscalls.
- **Why not route A (arm64 Linux VM + binfmt\_misc):** QEMU user mode can already run RISC-V programs inside a VM, so route A would just produce "a faster QEMU." QEMU linux-user does not support macOS as a host, so route B is the open space.
- **A stays as an escape hatch:** the syscall layer is designed as a `HostOs` trait, so if M5 gets crushed by futex/clone/epoll, a VM-passthrough implementation can be plugged in.
- **Differences from Rosetta 2:** transparent execution is impossible (macOS has no binfmt\_misc), so a launcher is required; but we borrow Rosetta's strategy of AOT first, JIT to fill the gaps.

## Why RISC-V → ARM64 Is Easier Than x86 → ARM64

The shape of the ISAs makes translation in this direction inherently cheap, and the performance ceiling can be high.

| Aspect | x86 → ARM64 (Rosetta's pain points) | RISC-V → ARM64 (this project) |
| --- | --- | --- |
| Memory model | x86 is TSO, ARM is weakly ordered; Apple relies on a private TSO hardware mode in M-series chips that third parties cannot use | RVWMO is itself weakly ordered and close to ARM; no TSO emulation needed |
| Flags | Nearly every instruction updates EFLAGS | No flags; `beq a0, a1` translates directly to `cmp` + `b.eq` |
| Registers | 16 GPRs | 31 GPRs, matching ARM64's 31, allowing a near 1:1 static mapping (reserving a few for the context pointer) |
| Decoding | Variable length, 1 to 15 bytes | Fixed 4 bytes, plus 2-byte compressed instructions (C extension) |

## Scope: What B-lite Supports and What It Doesn't

What would actually sink this project is the Linux→Darwin syscall compatibility layer, not CPU translation, so the scope deliberately starts small.

**Supported first**

- RV64GC (I, M, A, F, D, C), with Zba/Zbb later
- Statically linked Linux ELF (musl), single-threaded
- About 20 to 30 syscalls: `read`, `write`, `openat`, `close`, `exit`, `exit_group`, `brk`, `mmap`, `munmap`, `mprotect`, `fstat`, `lseek`, `clock_gettime`, `getrandom`, `ioctl` (TTY-related only), etc.

**Later**

- Dynamic linking (ld.so)
- Multithreading: `clone`, `futex` (`os_sync_wait_on_address` on macOS 14.4+, or `__ulock_wait`)
- Signals and ucontext translation, `epoll`→`kqueue`
- RVV vector extension (VLEN 128 maps onto NEON, but dynamic vtype/LMUL changes are complex)

## Instruction Set Scope

The target is **RV64GC**, which is what the RVA20U64 profile requires and what mainstream riscv64 Linux distributions (Debian, Fedora, Alpine) and `zig cc` emit by default. `G` is shorthand for I + M + A + F + D + Zicsr + Zifencei.

| Extension | Contents | Approx. count | Milestone |
| --- | --- | --- | --- |
| RV64I | Integer ALU, loads/stores, branches, `jal`/`jalr`, `lui`/`auipc`, `ecall`, `fence` | 52 (40 from RV32I + 12 RV64-only) | M1 |
| Zicsr | `csrrw` and friends, for `fcsr`, `cycle`, `time`, `instret` | 6 | M1–M3 |
| C | 16-bit compressed forms; each expands to an existing 32-bit `Inst` | ~40 | M2 (common subset), M3 (complete) |
| M | Multiply/divide, including `*w` variants | 13 | M3 |
| A | `lr`/`sc` and AMOs, word and doubleword | 22 | M3 |
| F, D | Single and double precision floating point | ~60 | M3 |
| Zifencei | `fence.i` (ties into JIT invalidation later) | 1 | M3 |
| Zba, Zbb | Address generation and bit manipulation (RVA22) | ~30 | Around M4 |
| V | Vector extension (mandatory in RVA23) | Hundreds | Out of scope |

**C is pulled forward into M2.** Compilers emit compressed instructions by default, so most of the instructions in a static `hello` are 2 bytes long. Compressed instructions are decoded into the same `Inst` values as their 32-bit equivalents, so the interpreter and the JIT never see the difference.

**RV64 only; no RV32.** RV64I contains every RV32I instruction (with 64-bit registers, plus `ld`/`sd`/`lwu` and the `*w` family), so everything CS61C covers is still implemented. There is essentially no RV32 Linux userland to run, and none of the milestone targets (lua, sqlite, busybox) need it. If RV32 is ever wanted for CS61C exercises, it can be added later as an XLEN parameter on the decoder.

## System Architecture

The translation core (5 layers) is shared by both routes; only the `HostOs` implementation behind the syscall layer (part of a 4-layer runtime) changes.

Rosetta 2's strategy is AOT first, JIT to fill the gaps: on first run it translates and caches to `~/Library/Caches`, and only translates on the fly when it hits JIT-generated or self-modifying code. This project adds an AOT cache at M4.

## Languages and Toolchain

Rust at the core; ARM64 assembly and Swift at the edges.

| Part | Language / tool | Rationale |
| --- | --- | --- |
| Translator core, runtime | Rust | A JIT is full of raw pointers, so memory safety is worth a lot; `dynasm-rs` can serve as the ARM64 emitter |
| Dispatcher, trampolines, context switch | ARM64 assembly (`global_asm!` or `.s`) | Entering and leaving translated blocks needs precise register control |
| Launcher, future GUI or VM escape hatch | Swift | Smoothest path to `Virtualization.framework` and macOS APIs |
| Test generation, benchmarks | Python | Generate random instruction sequences, run differential comparisons |
| Building RISC-V test executables | Zig (`zig cc`) | `zig cc -target riscv64-linux-musl -static hello.c -o hello`, no cross toolchain to install |
| Reference oracle | Spike, QEMU (on Linux) | Compare register state instruction by instruction |

## macOS Platform Pitfalls and Translation Details

**Platform**

- **W^X:** JIT memory must be `mmap(MAP_JIT)`, toggled between writable and executable with `pthread_jit_write_protect_np()`; the executable needs the `com.apple.security.cs.allow-jit` entitlement and must be codesigned.
- **I-cache:** call `sys_icache_invalidate()` after writing machine code, or stale instructions will execute.
- **No binfmt\_misc:** ELF files can't be launched by double-clicking; only through `rvrun`.
- **Address space:** map the guest's 47-bit address space onto the host with a fixed offset or as close to identity mapping as possible, avoiding regions macOS already occupies.

**Translation**

- **LR/SC:** translate to LSE `CAS` or `LDAXR`/`STLXR`, minding the ABA semantic differences.
- **Floating point:** NaN-boxing, dynamic rounding modes, and accrued `fflags` are hidden performance costs; get it correct first, then optimize with "don't compute fflags if nobody reads them."
- **Indirect jumps:** `jalr` looks up the translated address in a hash table, plus a return address stack to predict `ret`.
- **Self-modifying code:** detect writes to translated pages and invalidate the corresponding blocks.

## Milestones

Reaching M4 makes a complete, hardcore project; M5 is a bonus. Each step uses the previous step's interpreter as its correctness baseline.

| Milestone | Content | Done when | Course |
| --- | --- | --- | --- |
| M1 | RV64I decoder + interpreter | Passes `riscv-tests` rv64ui | CS61C |
| M2 | ELF loader (auxv, initial stack) + about 10 syscalls + common C instructions | A static `hello` built with `zig cc` prints on the Mac | CS162 |
| M3 | M, A, F, D extensions, rest of C + remaining syscalls | Runs a static `lua` or `sqlite` | CS61C |
| M4 | Basic-block JIT, block chaining, indirect-jump optimization, AOT cache | Zero differences in differential tests against the interpreter; clear benchmark speedup | Extracurricular |
| M5 | Multithreading, `futex`, signals, dynamic linking | Runs multithreaded programs | CS162 |

## Testing Strategy

The interpreter comes before the JIT and is kept forever, because JIT bugs usually show up as one wrong bit in some register after millions of instructions.

1. **Compliance tests:** `riscv-tests` and `riscv-arch-test`, run as each extension lands.
2. **Differential tests:** Python generates random instruction sequences; compare register and memory state between this project's interpreter and Spike.
3. **JIT vs. interpreter:** in debug mode, compare both sides' state after every block to find the first diverging block.
4. **Real programs:** static builds of busybox, lua, and sqlite, with output compared against QEMU.

## Open Questions

- Crate name is `champollion-rv` (crates.io: no existing crate as of Oct 13, 2026); CLI is `rvrun`.
- Introduce an IR from the start, or lift directly to ARM64 until M4?

## Decision Log

- **Oct 13, 2026 — RV64 only.** RV32 is not supported; see [Instruction Set Scope](#instruction-set-scope).
- **Oct 13, 2026 — C extension moves into M2.** Real binaries are full of compressed instructions, so `hello` cannot run without them.
