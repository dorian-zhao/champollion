# Champollion: Design Brief for a RISC-V Translator on macOS

Oct 13, 2026 · Dorian Verlaine

## Goals and Positioning

Champollion is a user-mode binary translator that runs RISC-V Linux executables on macOS arm64: `rvrun ./hello`. It has two goals:

1. **A testing and emulation environment for RISC-V developers on Apple Silicon Macs.** Cross-compile on the Mac, then run and debug the result immediately: `rvrun` as the test runner for `cargo test` / `ctest`, a built-in gdbstub for debugging, no VM to boot and no files to sync.
2. **A learning project:** implement CS61C (ISA, CPU) and CS162 (syscalls, threads, virtual memory) in reverse, then add a JIT on top.

The plan is **route B first, then route A**, sharing one translation core:

- **Route B (native, primary).** Run directly on macOS; Linux syscalls are caught by our own runtime and translated into Darwin calls. This covers the developer's daily build → test → debug loop with zero startup cost and the Mac's own file system. QEMU linux-user does not support macOS as a host, so this is the open space.
- **Route A (Linux VM, full compatibility, M6).** An arm64 Linux VM (`Virtualization.framework`) runs the same translator as a binfmt\_misc handler, and syscalls pass through to a real Linux kernel. This covers what B fundamentally cannot: containers, full distribution userlands, `/proc`-heavy and `ptrace`-based tools. It is the model of Rosetta for Linux.
- **Why both:** B alone tops out at "most developer programs" (an estimated 85–90% of developers' own programs and tests, 60–70% of arbitrary distribution binaries, no containers). WSL1, which took the same approach on Windows, hit the same ceiling and was succeeded by a VM (WSL2). A alone gives up B's instant, integrated developer loop.
- **Consequence for the core:** the translator must also build and run on arm64 Linux. In particular, JIT code memory must sit behind an interface (macOS: `MAP_JIT` + `pthread_jit_write_protect_np`; Linux: `mprotect` or a dual mapping), decided before M4.
- **Differences from Rosetta 2:** transparent execution on macOS is impossible (no binfmt\_misc), so B needs a launcher; but we borrow Rosetta's strategy of AOT first, JIT to fill the gaps.

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
| Launcher, route A VM host (M6) | Swift | Smoothest path to `Virtualization.framework` and macOS APIs |
| Test generation, benchmarks | Python | Generate random instruction sequences, run differential comparisons |
| Building RISC-V test executables | Zig (`zig cc`) | `zig cc -target riscv64-linux-musl -static hello.c -o hello`, no cross toolchain to install |
| Reference oracle | Spike, QEMU (on Linux) | Compare register state instruction by instruction |

## macOS Platform Pitfalls and Translation Details

**Platform**

- **W^X:** JIT memory must be `mmap(MAP_JIT)`, toggled between writable and executable with `pthread_jit_write_protect_np()`; the executable needs the `com.apple.security.cs.allow-jit` entitlement and must be codesigned.
- **I-cache:** call `sys_icache_invalidate()` after writing machine code, or stale instructions will execute.
- **No binfmt\_misc:** ELF files can't be launched by double-clicking; only through `rvrun`.
- **Address space:** map the guest's 47-bit address space onto the host at a fixed offset. Identity mapping is not an option: static musl ELFs typically load near `0x10000`, which falls inside the 4 GiB `__PAGEZERO` that macOS reserves at the bottom of every 64-bit process.
- **Reserved registers:** Apple's arm64 ABI reserves `x18` as the platform register; it must never hold guest state.

**Translation**

- **LR/SC:** translate to LSE `CAS` or `LDAXR`/`STLXR`, minding the ABA semantic differences.
- **Floating point:** NaN-boxing, dynamic rounding modes, and accrued `fflags` are hidden performance costs; get it correct first. Most of the cost can be pushed onto hardware; see [Performance Strategy](#performance-strategy).
- **Indirect jumps:** `jalr` looks up the translated address in a hash table, plus a return address stack to predict `ret`.
- **Self-modifying code:** detect writes to translated pages and invalidate the corresponding blocks.

## Performance Strategy

Performance is protected in three layers: define what "fast" means and measure it continuously, get the architectural decisions that set the ceiling right before M4, and only then tune the JIT.

### Target

**At most 2× slower than native (≥ 50% of native speed) on CPU-bound benchmarks; stretch goal: Rosetta 2 class, 70–80% of native.** For comparison, QEMU's TCG typically runs 3–10× slower than native and Rosetta 2 about 1.2–1.5×. This is the project's main differentiator from QEMU, and it is reachable only because RISC-V → ARM64 is a friendly direction (no flags, similar memory model, matching register count; see the table above). Every design choice is checked against it:

- Anything per-instruction or per-access that is not a single host instruction must be justified (locks, permission checks, flag computation, NaN fix-ups).
- Floating point runs on the host FPU, never in software (except rare corner cases such as the RMM rounding mode, which ARM lacks).
- The interpreter is not expected to meet the target; it is the oracle and cold tier. The JIT (M4) must, and benchmarks track progress toward it from M4 on.

### Measurement

- **Primary metric: percent of native speed.** Build each benchmark from the same C source twice with `zig cc`, once for riscv64-linux-musl and once for aarch64-macos, and report the translated run as a percentage of the native one. Rosetta 2 runs at roughly 70–80% of native and serves as the reference ceiling.
- **Secondary baseline: QEMU linux-user** (run on Linux), the incumbent this project wants to beat.
- **Workloads:** CoreMark, Embench, a few lua benchmark scripts, and sqlite's `speedtest1`.
- **Regression tracking:** time with `hyperfine`, store results in the repository, and from M4 onward run the suite on every JIT-related commit.
- **JIT counters:** blocks translated, direct-chain ratio, indirect-jump lookup miss rate, and dispatcher re-entries. Performance problems usually show up here before they show up in wall-clock time.
- **Profiling:** Instruments (Time Profiler, CPU Counters) or `samply`. The JIT should emit a symbol map so profiles can show guest function names.

### Architecture That Sets the Ceiling

These must be decided before M4; changing them later means rewriting the JIT.

| Area | Approach | Why it matters |
| --- | --- | --- |
| Register mapping | Pin the ~20 most used guest registers in host registers; spill the rest to `CpuState` | The single largest factor. Not 1:1, because `x18` is reserved on macOS and `x29`, `x30`, and `sp` are spoken for; pick by usage frequency |
| Memory addressing | Keep the guest memory base in a host register and use register-offset addressing (`ldr x, [base, xN]`) with no software bounds checks | Out-of-range accesses are caught by host page protection instead of per-access checks |
| Block chaining | Once a direct branch target is translated, patch the branch to jump straight to it | Hot loops never return to the dispatcher |
| Indirect jumps | Inline hash-table lookup in generated code for `jalr`; a shadow return stack to predict `ret` | Calls and switch jump tables are typically a translator's biggest bottleneck |
| Lazy state | Write back `pc` only at block exits; compute `instret` and `cycle` only when the guest reads them | Avoids per-instruction overhead |
| Self-modifying code | Write-protect translated pages and invalidate on fault | Zero cost in the common case |
| AOT cache | Persist translations to `~/Library/Caches` | Removes translation warmup from startup |

### Floating Point Mostly Maps onto Hardware

ARM64 hardware lines up with RISC-V floating-point semantics closely enough that correctness is nearly free:

- **Canonical NaN:** with `FPCR.DN` (Default NaN) set, ARM produces `0x7FC00000`, exactly the RISC-V canonical NaN, so no software fix-up is needed.
- **`fflags`:** the cumulative exception bits in ARM's `FPSR` map one-to-one onto RISC-V `fflags` (IOC→NV, DZC→DZ, OFC→OF, UFC→UF, IXC→NX). Hardware accumulates them; read `FPSR` only when the guest reads `fflags`.
- **Rounding modes:** static rounding modes encoded in the instruction map to ARM instruction variants (e.g. `fcvtzs` / `fcvtms`); dynamic `frm` is written to `FPCR` only when it changes.
- **Exception:** NaN-boxing checks (a single-precision value requires the upper 32 bits to be all ones) have no hardware equivalent. Track the type held in each register at translation time to elide most of them.

### Peephole Optimizations (Later)

Recognize common instruction sequences and emit tighter ARM64 code:

- `lui` + `addi` → a single `mov`
- `auipc` + `jalr` → a direct `bl`
- `slli 32` + `srli 32` → `ubfx`
- fold `sext.w` into the preceding `*w` instruction
- compare-and-branch → `cbz` / `tbz`

Do these only once the JIT is correct and the benchmarks run, and pick them by profile.

### Constraints on M1–M3

- **Do not optimize the interpreter.** It is the correctness oracle; clarity beats speed.
- **Get the data structures right now:** `CpuState` stays `#[repr(C)]` for fixed-offset JIT access, and `Inst` stays small and `Copy`. The `Memory` trait is for the interpreter only; JIT code reaches guest memory through the base register, so the trait never becomes a bottleneck.
- **Fix the guest address-space layout in M2** (offset and reserved range), since it determines the M4 addressing scheme. Done: 256 GiB (Sv39 user space) reserved up front, guest address `g` at host `base + g`.

### Baseline (Oct 14, 2026)

A CPU-bound C benchmark (sieve, array traffic, integer hashing; 228 M guest instructions), Apple silicon, release build:

| Configuration | Speed | Time | vs. native |
| --- | --- | --- | --- |
| Native arm64 | — | 29.5–37 ms | 1× |
| Reference interpreter, thread-safe `GuestMemory` | 41 MIPS | 5.6 s | ~150× slower |
| Same, lock and permission checks removed (experiment only) | 147 MIPS | 1.55 s | — |
| **Engine: block cache + software TLB + chaining** | **~270 MIPS** | **0.85 s** | **~23× slower** |

Real program (Lua 5.4 running a fib + table sort + concat script, 1.94 G guest instructions): native 100 ms, engine 10.4 s (183 MIPS), **~104× slower**. Interpreters like Lua are dominated by indirect jumps (their own dispatch loop), which break into short blocks that our chaining cannot link; this is the workload M4's indirect-jump prediction and inline lookup must fix.

The per-access lock and B-tree lookup cost the reference interpreter 3.6×, mostly on instruction fetch. The engine (below) recovers that and more without giving up thread safety. Closing the remaining gap is the JIT's job (M4), as it is for QEMU's TCG.

### What we borrow from Rosetta 2

Rosetta 2 has no interpreter for ordinary code; it translates. Two of its principles still shape our interpreter:

- **Translate once, per block, and cache by guest pc.** The engine decodes each basic block once and runs the cached `Inst`s; blocks are chained to their recent successors. M4 keeps this exact cache (same key, same invalidation by mapping generation and `fence.i`) and swaps decoded `Inst`s for arm64 code.
- **No checks on the hot path.** Rosetta runs x86 code in the same address space as arm64 code, so memory accesses need no translation and faults come from the MMU. Safe Rust cannot recover from a host fault mid-interpreter, so the engine approximates this with a per-thread software TLB; the JIT will use the real thing (base register + host fault handler).

Profile of the engine (`sample`): ~63% instruction dispatch in `execute`, ~15% block lookup and loop, the rest loads/stores. Further interpreter gains would need threaded code; we stop here and invest in the JIT.

## Concurrency Model

Threads arrive in M5, but the runtime is shaped for them from M2 so that adding `clone` does not mean redesigning memory, atomics, or the syscall layer.

- **Process vs. thread.** `Process` (shared, `Sync`, used only through `&Process`) owns the address space, the program break, and syscall logging. `Thread` owns `CpuState` (including the `lr` reservation), its tid, and `clear_child_tid`. Each guest thread will run on its own host thread. `exit` ends a thread; `exit_group` ends the process.
- **Guest memory is shared, never borrowed.** All `Memory` methods take `&self`. Data accesses use relaxed host atomics, which compile to plain `ldr`/`str` on arm64 when aligned, so racing guest threads are the guest's business and never undefined behavior on the host side.
- **Atomics are host atomics.** AMOs are host compare-and-swap loops (LSE single instructions in the JIT). `lr` records `(address, value)` and `sc` is a host compare-and-swap against that value, as in QEMU; the only deviation from hardware is ABA, which a value-based reservation cannot see. `fence` is a host fence.
- **The mapping table is behind an `RwLock`.** Interpreter accesses hold the read lock for the duration of one access; `mmap`/`munmap`/`mprotect`/`brk` take the write lock, and "find a free range and map it" is a single atomic step.
- **Host I/O never holds the lock.** Syscalls validate guest buffers under the lock, release it, and hand raw `GuestBuf`s (layout-compatible with `struct iovec`) to the host kernel. If another thread unmaps the buffer meanwhile, the kernel returns `EFAULT` instead of crashing us, and a blocking `read` cannot deadlock against a concurrent `mmap`. `HostOs` is `&self + Sync`.

### Performance boundary

The locks and per-access permission checks above belong to the **interpreter only**, which is the correctness oracle and the cold-code tier. They must never reach JIT-generated code:

- **JIT loads and stores** are `ldr`/`str` relative to the guest base register, with no lock and no software permission check. Accesses to unmapped memory are caught by the host fault handler and turned into guest `SIGSEGV`, as in QEMU.
- **JIT atomics** are single ARM64 LSE instructions (`ldadd`, `swp`, `cas`, ...); `lr`/`sc` become `ldaxr`/`stlxr` or a `cas` against the reserved value.
- **JIT fences** map RISC-V `pred`/`succ` sets to the cheapest sufficient `dmb` variant (`ish`, `ishld`, `ishst`).

The engine (M2) adds a per-thread software TLB and a decoded-block cache, both invalidated by a global mapping-generation counter bumped on every mapping change, without changing any interface. With threads (M5), `munmap` must wait until every running thread has observed the new generation before it decommits host pages, since TLB hits skip the lock.

## Milestones

Each step uses the previous step's interpreter as its correctness baseline. M1–M5 are route B; M6 is route A.

| Milestone | Content | Done when | Course |
| --- | --- | --- | --- |
| M1 | RV64I decoder + interpreter | Passes `riscv-tests` rv64ui | CS61C |
| M2 | M, A, C extensions + guest address space + ELF loader (auxv, initial stack) + core syscalls | A static `hello` built with `zig cc` prints on the Mac | CS162 |
| M3 | F, D, Zicsr, Zifencei + remaining syscalls + gdbstub + test-runner integration | Runs a static `lua` or `sqlite`; `cargo test` for a riscv64 target runs through `rvrun` | CS61C |
| M4 | Basic-block JIT, block chaining, indirect-jump optimization, AOT cache | Zero differences in differential tests against the interpreter; clear benchmark speedup | Extracurricular |
| M5 | Multithreading, `futex`, signals, dynamic linking (with a riscv64 sysroot) | Runs multithreaded programs and Go binaries | CS162 |
| M6 | Route A: arm64 Linux VM via `Virtualization.framework`, translator as binfmt\_misc handler, Linux syscall passthrough | A riscv64 Docker image runs on the Mac | CS162 |

## Testing Strategy

The interpreter comes before the JIT and is kept forever, because JIT bugs usually show up as one wrong bit in some register after millions of instructions.

1. **Compliance tests:** `riscv-tests` and `riscv-arch-test`, run as each extension lands.
2. **Differential tests:** Python generates random instruction sequences; compare register and memory state between this project's interpreter and Spike.
3. **JIT vs. interpreter:** in debug mode, compare both sides' state after every block to find the first diverging block.
4. **Real programs:** static builds of busybox, lua, and sqlite, with output compared against QEMU.

## Open Questions

- Crate name is `champollion-rv` (crates.io: no existing crate as of Oct 13, 2026); CLI is `rvrun`.
- Introduce an IR from the start, or lift directly to ARM64 until M4?
- **M6 foundation:** build route A on [apple/containerization](https://github.com/apple/containerization) (the package behind [apple/container](https://github.com/apple/container))? It already provides per-container VMs on `Virtualization.framework`, a fast-booting kernel config, the `vminitd` init, ext4 and OCI image handling, and runs linux/amd64 images through Rosetta, which is structurally what we need for riscv64. Unverified: whether a custom binfmt\_misc handler can be registered without forking it. Requires macOS 26+.

## Decision Log

- **Oct 14, 2026 — Route B first, then route A as a full milestone (M6).** The project's first goal is now a test and emulation environment for RISC-V developers on Apple Silicon. B covers the daily build → test → debug loop best (instant, no VM, Mac file system); A (Linux VM + syscall passthrough) is promoted from escape hatch to M6 for containers and full distributions. The core must therefore also run on arm64 Linux; see [Goals and Positioning](#goals-and-positioning).
- **Oct 14, 2026 — Static-PIE moves into M3.** Rust's musl targets link static-PIE by default, so `cargo test` needs it; images load at `PIE_BASE` (2/3 up, like Linux) and relocate themselves.
- **Oct 14, 2026 — `cargo test` works before threads.** libtest falls back to the main thread when `clone` fails, so the M3 criterion holds without M5; parallel test execution waits for M5.
- **Oct 14, 2026 — FP reference is Berkeley SoftFloat (RISC-V specialization), vendored.** Same code as Spike; used to fuzz the host-FPU path and as the RMM fallback. Existing crates either fail to build with current CMake or are LGPL.
- **Oct 14, 2026 — Performance target: ≤ 2× native, stretch Rosetta 2 class (70–80% of native).** See [Target](#target). Consequence for M3: F/D runs on the host FPU, with the interpreter using the same `FPCR`/`FPSR` mapping the JIT will use, so the hardware-mapping claims are validated before M4.
- **Oct 14, 2026 — Fast interpreter before M3.** A decoded-block cache with chaining plus a software TLB takes the CPU benchmark from 41 to ~270 MIPS (6.6×), and is the cache structure the JIT will reuse; see [What we borrow from Rosetta 2](#what-we-borrow-from-rosetta-2).
- **Oct 14, 2026 — The runtime is thread-ready from M2.** Process/thread split, `&self` memory with atomic accesses, host atomics for AMOs and `lr`/`sc`, and lock-free host I/O on raw guest buffers; see [Concurrency Model](#concurrency-model). Locks stay interpreter-only.
- **Oct 14, 2026 — F/D data movement moves into M2.** musl's `printf` saves FP registers and moves bits with `fmv`/`fsgnj`, so every program that prints needs them; FP arithmetic stays in M3.
- **Oct 14, 2026 — M and A move into M2.** The static musl `hello` already uses `mul`, `divu`, `lr.w`, and `sc.w`.
- **Oct 13, 2026 — RV64 only.** RV32 is not supported; see [Instruction Set Scope](#instruction-set-scope).
- **Oct 13, 2026 — C extension moves into M2.** Real binaries are full of compressed instructions, so `hello` cannot run without them.
- **Oct 13, 2026 — Guest address-space layout is decided in M2.** Guest memory lives at a fixed host offset (identity mapping collides with `__PAGEZERO`), and the layout dictates the JIT's memory addressing; see [Performance Strategy](#performance-strategy).
