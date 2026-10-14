# Champollion: Design Brief for a RISC-V Translator on macOS

Oct 13, 2026 · Dorian Verlaine

## Goals and Positioning

Champollion is a user-mode binary translator that runs RISC-V Linux executables on arm64: `rvrun ./hello` on an Apple silicon Mac, or the same build natively on an arm64 Linux host. It has two goals:

1. **A testing and emulation environment for RISC-V developers on Apple Silicon Macs.** Cross-compile on the Mac, then run and debug the result immediately: `rvrun` as the test runner for `cargo test` / `ctest`, a built-in gdbstub for debugging, no VM to boot and no files to sync.
2. **A learning project:** implement CS61C (ISA, CPU) and CS162 (syscalls, threads, virtual memory) in reverse, then add a JIT on top.

The plan is **Direct mode first, then VM mode**, sharing one translation core:

- **Direct mode (native, primary).** Run directly on macOS; Linux syscalls are caught by our own runtime and translated into Darwin calls. This covers the developer's daily build → test → debug loop with zero startup cost and the Mac's own file system. QEMU linux-user does not support macOS as a host, so this is the open space.
- **VM mode (Linux VM, full compatibility, M6).** An arm64 Linux VM (`Virtualization.framework`) runs the same translator as a binfmt\_misc handler, and syscalls pass through to a real Linux kernel. This covers what Direct mode fundamentally cannot: containers, full distribution userlands, `/proc`-heavy and `ptrace`-based tools. It is the model of Rosetta for Linux.
- **Why both:** Direct mode alone tops out at "most developer programs" (an estimated 85–90% of developers' own programs and tests, 60–70% of arbitrary distribution binaries, no containers). WSL1, which took the same approach on Windows, hit the same ceiling and was succeeded by a VM (WSL2). VM mode alone gives up Direct mode's instant, integrated developer loop.
- **Consequence for the core:** the translator must also build and run on arm64 Linux. In particular, JIT code memory must sit behind an interface (macOS: `MAP_JIT` + `pthread_jit_write_protect_np`; Linux: `mprotect` or a dual mapping), decided before M4.
- **The Linux build also stands alone.** On arm64 Linux hosts (servers, CI runners, single-board computers), the Linux build runs natively as the `binfmt_misc` handler — riscv64 containers need no VM there; the launcher is how a Mac reaches the same backend.
- **Differences from Rosetta 2:** transparent execution on macOS is impossible (no binfmt\_misc), so Direct mode needs a launcher; but we borrow Rosetta's strategy of AOT first, JIT to fill the gaps.

## Why RISC-V → ARM64 Is Easier Than x86 → ARM64

The shape of the ISAs makes translation in this direction inherently cheap, and the performance ceiling can be high.

| Aspect | x86 → ARM64 (Rosetta's pain points) | RISC-V → ARM64 (this project) |
| --- | --- | --- |
| Memory model | x86 is TSO, ARM is weakly ordered; Apple relies on a private TSO hardware mode in M-series chips that third parties cannot use | RVWMO is itself weakly ordered and close to ARM; no TSO emulation needed |
| Flags | Nearly every instruction updates EFLAGS | No flags; `beq a0, a1` translates directly to `cmp` + `b.eq` |
| Registers | 16 GPRs | 31 GPRs, matching ARM64's 31, allowing a near 1:1 static mapping (reserving a few for the context pointer) |
| Decoding | Variable length, 1 to 15 bytes | Fixed 4 bytes, plus 2-byte compressed instructions (C extension) |

## Scope: What the First Version Supports and What It Doesn't

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

The translation core (5 layers) is shared by both modes; only the `HostOs` implementation behind the syscall layer (part of a 4-layer runtime) changes.

Rosetta 2's strategy is AOT first, JIT to fill the gaps: on first run it translates and caches to `~/Library/Caches`, and only translates on the fly when it hits JIT-generated or self-modifying code. This project adds an AOT cache at M4.

## Languages and Toolchain

Rust at the core; ARM64 assembly and Swift at the edges.

| Part | Language / tool | Rationale |
| --- | --- | --- |
| Translator core, runtime | Rust | A JIT is full of raw pointers, so memory safety is worth a lot; `dynasm-rs` can serve as the ARM64 emitter |
| Dispatcher, trampolines, context switch | ARM64 assembly (`global_asm!` or `.s`) | Entering and leaving translated blocks needs precise register control |
| Launcher, VM mode's host (M6) | Swift | Smoothest path to `Virtualization.framework` and macOS APIs |
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
- **Profiling:** host profilers cannot attribute samples inside generated code, so `RVRUN_PROF=FILE` runs our own sampler: a thread suspends the guest thread every 0.5 ms and records its pc (per host instruction in generated code, symbolized with `dladdr` elsewhere). `tools/jitprof.py FILE` disassembles the hottest blocks next to their guest code with per-instruction counts. `/usr/bin/time -l` gives instructions retired and cycles, which tell instruction-count problems (low IPC is not our issue) from latency problems.

### Architecture That Sets the Ceiling

These must be decided before M4; changing them later means rewriting the JIT.

| Area | Approach | Why it matters |
| --- | --- | --- |
| Register mapping | Pin the ~20 most used guest registers in host registers; spill the rest to `CpuState` | The single largest factor. Not 1:1, because `x18` is reserved on macOS and `x29`, `x30`, and `sp` are spoken for; pick by usage frequency |
| Memory addressing | Keep the guest memory base in a host register; mask the guest base register into the guest space, add the memory base once per base register, and use immediate offsets (`ldr x, [x17, #off]`), with guard pages around the guest space and no software bounds checks | Out-of-range accesses are caught by host page protection instead of per-access checks; runs of stack accesses cost one instruction each |
| Block chaining | Once a direct branch target is translated, patch the branch to jump straight to it | Hot loops never return to the dispatcher |
| Indirect jumps | Inline jump-cache lookup in generated code for `jalr`; constant targets (`auipc`/`lui` + `jalr`) chain like `jal` | Calls and switch jump tables are typically a translator's biggest bottleneck. A return-address stack was tried and dropped (see the decision log) |
| Lazy state | Write back `pc` only at block exits; count `instret` per block in a host register | Avoids per-instruction overhead; exact `instret` costs under 1% |
| Self-modifying code | Write-protect translated pages and invalidate on fault | Zero cost in the common case |
| AOT cache | Persist translations to `~/Library/Caches` | Removes translation warmup from startup |

### Floating Point Mostly Maps onto Hardware

ARM64 hardware lines up with RISC-V floating-point semantics closely enough that correctness is nearly free:

- **Canonical NaN:** with `FPCR.DN` (Default NaN) set, ARM produces `0x7FC00000`, exactly the RISC-V canonical NaN, so no software fix-up is needed.
- **`fflags`:** the cumulative exception bits in ARM's `FPSR` map one-to-one onto RISC-V `fflags` (IOC→NV, DZC→DZ, OFC→OF, UFC→UF, IXC→NX). Hardware accumulates them; read `FPSR` only when the guest reads `fflags`.
- **Rounding modes:** static rounding modes encoded in the instruction map to ARM instruction variants (e.g. `fcvtzs` / `fcvtms`); dynamic `frm` is written to `FPCR` only when it changes.
- **Exception:** NaN-boxing checks (a single-precision value requires the upper 32 bits to be all ones) have no hardware equivalent. Track the type held in each register at translation time to elide most of them.
- **JIT deviation (tininess):** ARM detects tininess before rounding, RISC-V after. The interpreter's `fpu` module recomputes the rare affected results with SoftFloat, but generated code does not: when a result rounds to exactly ±the smallest normal, the JIT may raise UF where RISC-V does not. Results are identical; only that flag can differ.

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

Real program (Lua 5.4 running a fib + table sort + concat script, 1.94 G guest instructions): native 100 ms, engine 10.4 s (183 MIPS), **~104× slower**.

### JIT progress (M4, Oct 14, 2026)

| Step | C benchmark | n-body (FP) | Lua script | SQLite (300k-row insert, index, aggregates) |
| --- | --- | --- | --- | --- |
| Native arm64 | 29.9 ms | 71.9 ms | 97.6 ms | 187 ms |
| Engine (interpreter) | 28.5× slower | — | 106× slower | — |
| JIT, no chaining | 10.2× slower | — | 113× slower | — |
| + direct block chaining | 1.6× slower | — | 75× slower | — |
| + inline jump cache, code generation | 1.5× slower | — | 69× slower | — |
| + FP moves, lr/sc, AMOs translated | 1.5× slower | 7.6× slower | 2.7× slower | 3.0× slower |
| + FP arithmetic on the host FPU, FP registers pinned | 1.55× slower | 1.52× slower | 2.7× slower | 2.9× slower |
| + `instret` in x30, t0–t2 pinned | 1.53× slower | 1.43× slower | 2.6× slower | 2.2× slower |
| + direct `auipc`/`lui` + `jalr`, probe inlined | 1.49× slower | 1.39× slower | 2.3× slower | 2.1× slower |
| + base-register address cache | 1.48× slower | 1.36× slower | 2.2× slower | 1.98× slower |
| + blocks continue past conditional branches | **1.52× slower** | **1.37× slower** | **2.0× slower** | **1.87× slower** |

(The last row is against native runs from the same session: 29.4, 69.8, 94.3, and 182 ms.)

JIT statistics (`RVRUN_STATS=1`) drove the steps up to FP: they showed Lua flushing the JIT on each data `mmap`, then 8.3M instructions bouncing through the interpreter. The sampling profiler (`RVRUN_PROF`) drove the rest:

- SQLite's hottest loops (string and memory compares) spilled t0–t2 through `CpuState` on every use, so frequency counts alone had picked the wrong registers to pin.
- The `instret` load-add-store at every block entry made a store-to-load forwarding chain through all blocks; in a register it costs one `add`.
- Far calls (`auipc` + `jalr`) went through the jump cache although their target is a constant.
- Prologues and epilogues are runs of stack accesses, each paying add, mask, and a register-offset access.

Lua and SQLite now run at a higher IPC than native (5.6 vs. 4.5–4.9 instructions per cycle) but execute 2.5× the instructions: 1.2× from RISC-V needing more instructions than arm64, the rest our expansion. Instructions off the critical path are nearly free on these wide cores (a leaner jump-cache probe and canonical `mov`s measured no change); what still costs is per-block overhead (exits, branch tails) and latency chains. Masking each new base register costs Lua 5% and SQLite 3% on pointer chasing. Ending blocks at conditional branches was such per-block overhead: the not-taken path paid a taken `b` and the next block's `instret` update. Blocks now run on past them and leave through side exits (Lua -10%, SQLite -5%, and 35% fewer translations).

Short programs are dominated by translation instead. A Rust test binary (`cargo test` through rvrun) runs only 1.7M instructions but translates 2.3k blocks, at about 2 µs each. Batching instruction-cache maintenance and W^X switches, finding bitmask immediates without a loop, and fetching through a TLB took it from 10.0 to 8.4 ms; the persistent code cache (below) takes a repeated run to 5.5 ms (hello alone takes 2.9 ms, native 1.9 ms).

#### Persistent code cache

Rosetta 2 translates ahead of time and caches the result. We cache after the fact: at exit, rvrun saves the code buffer and its tables (blocks, exits waiting to be chained, fault sites) in `~/Library/Caches/champollion`, keyed by the executable's contents and the rvrun build, and the next run of the same executable loads them at the same offsets. That needs position-independent code; the only host address generated code held was the jump cache's, which now sits in a data area at the end of the code buffer's mapping and is reached with `adrp` (macOS refuses `MAP_FIXED` for `MAP_JIT`, so it cannot be a separate mapping at a fixed distance). A snapshot is valid because our loader places an executable deterministically; only translations of that startup image are saved, so a run that changes executable mappings or executes `fence.i` saves nothing. With dynamic linking (M5) the key will have to cover the mapped libraries too.

The per-access lock and B-tree lookup cost the reference interpreter 3.6×, mostly on instruction fetch. The engine (below) recovers that and more without giving up thread safety. Closing the remaining gap is the JIT's job (M4), as it is for QEMU's TCG.

### What we borrow from Rosetta 2

Rosetta 2 has no interpreter for ordinary code; it translates. Two of its principles still shape our interpreter:

- **Translate once, per block, and cache by guest pc.** The engine decodes each basic block once and runs the cached `Inst`s; blocks are chained to their recent successors. M4 keeps this exact cache (same key, same invalidation by mapping generation and `fence.i`) and swaps decoded `Inst`s for arm64 code.
- **No checks on the hot path.** Rosetta runs x86 code in the same address space as arm64 code, so memory accesses need no translation and faults come from the MMU. Safe Rust cannot recover from a host fault mid-interpreter, so the engine approximates this with a per-thread software TLB; the JIT will use the real thing (base register + host fault handler).

Profile of the engine (`sample`): ~63% instruction dispatch in `execute`, ~15% block lookup and loop, the rest loads/stores. Further interpreter gains would need threaded code; we stop here and invest in the JIT.

## Concurrency Model

The runtime was shaped for threads from M2, so adding `clone` in M5 did not mean redesigning memory, atomics, or the syscall layer.

- **One host thread per guest thread.** Thread-style `clone` starts a host thread with its own `CpuState`, engine, and JIT (translations are per thread; only the main thread's are cached on disk). As on Linux, `exit` ends a thread and clears and wakes its `clear_child_tid` word, the process ends with its last thread, and `exit_group` or a fatal fault in any thread ends it at once (from a thread other than the main one, by exiting the host process directly, since the main thread may be blocked).
- **Futexes are a parking lot in the process.** Guest processes never share memory, so there is no need for a host facility: waiters queue per address under a bucket lock, which makes the value check and the sleep atomic with respect to wakes.
- **New processes are host processes.** A guest `fork` (and `vfork`, which copies) is a host `fork`, holding the locks other threads could hold across it; `execve` asks the embedder to start a new rvrun with the guest's `argv[0]`, sysroot, environment, ignored signals, and mask. `MAP_SHARED` memory is real host shared memory (host-page aligned), with futexes on it served by `os_sync_wait_on_address`, so semaphores work across processes.
- **Signals are delivered by the recipient's run loop.** Dispositions belong to the process; masks, pending sets, and the alternate stack to threads. A thread takes deliverable signals after each syscall and when guest code traps; a sender also sets the recipient's interrupt flag, which translated code checks before backward and indirect jumps (not returns), so even a spinning thread stops at the next loop iteration. Delivery builds the riscv64 `rt_sigframe`; handlers return through a trampoline page standing in for the vDSO. The checks cost about 2%; checking at every block entry cost 4–7%. Waits of ours (futexes, `rt_sigsuspend`) are ended by a per-thread waker, blocking host syscalls by an internal host SIGURG; interrupted calls restart under `SA_RESTART` as on Linux. Host signals are blocked in every thread and taken by a `sigwait` thread that applies the guest's disposition.

- **Process vs. thread.** `Process` (shared, `Sync`, used only through `&Process`) owns the address space, the program break, and syscall logging. `Thread` owns `CpuState` (including the `lr` reservation), its tid, and `clear_child_tid`. Each guest thread will run on its own host thread. `exit` ends a thread; `exit_group` ends the process.
- **Guest memory is shared, never borrowed.** All `Memory` methods take `&self`. Data accesses use relaxed host atomics, which compile to plain `ldr`/`str` on arm64 when aligned, so racing guest threads are the guest's business and never undefined behavior on the host side.
- **Atomics are host atomics.** AMOs are host compare-and-swap loops (LSE single instructions in the JIT). `lr` records `(address, value)` and `sc` is a host compare-and-swap against that value, as in QEMU; the only deviation from hardware is ABA, which a value-based reservation cannot see. `fence` is a host fence.
- **The mapping table is behind an `RwLock`.** Interpreter accesses hold the read lock for the duration of one access; `mmap`/`munmap`/`mprotect`/`brk` take the write lock, and "find a free range and map it" is a single atomic step.
- **Host I/O never holds the lock.** Syscalls validate guest buffers under the lock, release it, and hand raw `GuestBuf`s (layout-compatible with `struct iovec`) to the host kernel. If another thread unmaps the buffer meanwhile, the kernel returns `EFAULT` instead of crashing us, and a blocking `read` cannot deadlock against a concurrent `mmap`. `HostOs` is `&self + Sync`.

### Performance boundary

The locks and per-access permission checks above belong to the **interpreter only**, which is the correctness oracle and the cold-code tier. They must never reach JIT-generated code:

- **JIT loads and stores** are `ldr`/`str` relative to the guest base register, with no lock and no software permission check. Accesses to unmapped memory are caught by the host fault handler and turned into guest `SIGSEGV`, as in QEMU. Host protection works on 16 KiB pages, so generated code deviates from the interpreter where guest permissions are finer:
  - a guest page that is unmapped (or read-only) but shares a host page with a writable one can be read and written;
  - a base register at or above 2^38 is masked into the guest space, so its access aliases a lower address instead of faulting.

  The differential fuzzer (`jit::tests::fuzz`) keeps its programs clear of both.
- **JIT atomics** are single ARM64 LSE instructions (`ldadd`, `swp`, `cas`, ...); `lr`/`sc` become `ldaxr`/`stlxr` or a `cas` against the reserved value.
- **JIT fences** map RISC-V `pred`/`succ` sets to the cheapest sufficient `dmb` variant (`ish`, `ishld`, `ishst`).

The engine (M2) adds a per-thread software TLB and a decoded-block cache, both invalidated by a global mapping-generation counter bumped on every mapping change, without changing any interface. With threads (M5), TLB hits skip the lock, so unmapping cannot revoke host pages at once: freed host pages are replaced with fresh zero pages, and `munmap` makes them inaccessible only after every thread using a TLB has published (per block, through a `TlbReader`) that it has seen the new generation. A thread outside the engine, e.g. blocked in a syscall, publishes that it holds no entries, so it never holds anyone up.

## Milestones

Each step uses the previous step's interpreter as its correctness baseline. M1–M5 were done in Direct mode; M6 is VM mode.

| Milestone | Content | Done when | Course |
| --- | --- | --- | --- |
| M1 | RV64I decoder + interpreter | Passes `riscv-tests` rv64ui | CS61C |
| M2 | M, A, C extensions + guest address space + ELF loader (auxv, initial stack) + core syscalls | A static `hello` built with `zig cc` prints on the Mac | CS162 |
| M3 | F, D, Zicsr, Zifencei + remaining syscalls + gdbstub + test-runner integration | Runs a static `lua` or `sqlite`; `cargo test` for a riscv64 target runs through `rvrun` | CS61C |
| M4 | Basic-block JIT, block chaining, indirect-jump optimization, AOT cache | Zero differences in differential tests against the interpreter; clear benchmark speedup | Extracurricular |
| M5 | Multithreading, `futex`, signals, sockets, epoll, dynamic linking (with a riscv64 sysroot), fork/exec | Runs multithreaded programs, Go binaries, and dynamically linked programs (e.g. Python) | CS162 |
| M6 | VM mode: arm64 Linux VM via `Virtualization.framework`, translator as binfmt\_misc handler, Linux syscall passthrough | A riscv64 Docker image runs on the Mac | CS162 |

## Testing Strategy

The interpreter comes before the JIT and is kept forever, because JIT bugs usually show up as one wrong bit in some register after millions of instructions.

1. **Compliance tests:** `riscv-tests` and `riscv-arch-test`, run as each extension lands.
2. **Differential tests:** Python generates random instruction sequences; compare register and memory state between this project's interpreter and Spike.
3. **JIT vs. interpreter:** riscv-tests run on the interpreter, the engine, and the JIT, which must end in identical `CpuState`. `jit::tests::fuzz` generates random looping programs (integer, M, A, and F/D instructions, accesses through moving base registers, branches, direct and indirect jumps) and compares traps, the whole CPU state, and memory between the interpreter, the JIT, and the JIT running restored translations; `FUZZ_CASES` and `FUZZ_SEED` run longer campaigns. Its first run found a wrong `divuw` by x0.
4. **Real programs:** static builds of busybox, lua, and sqlite, with output compared against QEMU.
5. **Both hosts:** the suite runs on macOS and, through `tools/linux.sh` (a container on OrbStack or Docker Desktop), on arm64 Linux, VM mode's host.

## Known Gaps

Things that do not work yet, roughly by how likely a program is to notice.

### Direct mode

- **Dynamically linked programs get no persistent code cache.** Mapping libraries changes executable mappings, which drops translations and disables saving; CPython pays its startup translation on every run (about 0.2 s for `python3 -c pass`). The cache key would have to cover every mapped library.
- **Realtime signals do not queue.** Each signal has at most one pending instance, as standard signals do; `sigqueue` payloads are not delivered.
- **Only `ITIMER_REAL`.** The CPU-time timers (`ITIMER_VIRTUAL`, `ITIMER_PROF`) and POSIX `timer_create` are missing.
- **Startup grew by about 0.5 ms** (3.4 ms for a static hello, native 1.9 ms), mostly the host signal thread.
- **Memory protection is coarser in generated code** (16 KiB host pages, aliasing above 2^38); see [Performance boundary](#performance-boundary).

### VM mode

- **Every `run` unpacks the image again** into a fresh ext4 file (a second or two for Debian); there are no persistent containers, no `exec` into a running one, and no published ports yet.
- **The Kata kernel is 6.12**, while Containerization tests from 6.14.9.
- **Linux builds linked by GNU ld run SQLite about 12% slower than zig-linked ones** (37.9 vs. 33.5 ms in OrbStack's VM, same code and memory layout); the cause is not known yet.

## Open Questions

- Crate name is `champollion-rv` (crates.io: no existing crate as of Oct 13, 2026); CLI is `rvrun`.
- Introduce an IR from the start, or lift directly to ARM64 until M4?

## Decision Log

- **Oct 14, 2026 — VM mode's launcher is rvcontainer, built on apple/containerization.** A Swift command (`launcher/`) pulls an image's linux/riscv64 variant, unpacks it to ext4, and runs it in an arm64 Linux VM booted from Apple's published vminit image and Kata's kernel, with vmnet networking. rvrun's Linux build is shared into the VM over virtiofs and registered with binfmt\_misc through vminitd's public `SetupEmulator` call, exactly as Containerization registers Rosetta, so no fork is needed (this was the open question). Flags `POCF`: setuid programs run with their owner's credentials, and rvrun reads the program from `AT_EXECFD`. Alpine and Debian images run interactively or not and install packages; a warm run of a small command takes 0.8 s, and Lua runs as fast as in OrbStack's VM.
- **Oct 14, 2026 — On Linux, guest addresses are host addresses, and calls we do not emulate go to the kernel.** Guest memory on Linux uses an identity layout: guest address `g` is host address `g`, reserved from `mmap_min_addr` up to 2^38 plus the guard, with the control area in a separate mapping (macOS keeps the offset layout, since it reserves the low 4 GiB of every process). Guest pointers are then host pointers, and since riscv64 and arm64 share syscall numbers and structure layouts, any call the dispatcher does not handle can be passed through as it is, pointers inside structures included, which a table of pointer arguments (QEMU's way) could not do without per-call code. Excluded are calls that would undercut the emulation: thread creation and `execveat`, signals and signal-sending timers, host memory protections, seccomp, ptrace, the architecture-specific range, and everything when a sysroot is set. Generated code also gets simpler: no base addition, and the interrupt check is one load through the register that held the base. Static musl binaries must then be linked above the guest space (`.cargo/config.toml`), which zig's linker cannot do, so Linux builds use GNU ld via `tools/linux.sh`. With this, Debian trixie's riscv64 root runs `apt-get update` and installs packages.
- **Oct 14, 2026 — In VM mode, rvrun is the binfmt\_misc handler and `execve` goes to the kernel.** Registered for riscv64 ELF files with the `F` (open at registration, so it works in containers and chroots) and `P` (preserve `argv[0]`, announced in `AT_FLAGS`) flags, rvrun starts guest programs like native ones; a guest `execve` is the kernel's again, which finds rvrun for riscv64 programs and `#!` interpreters by itself. Tested in a private binfmt\_misc instance (a user namespace, so the VM's global registrations stay untouched) with Alpine's and Debian trixie's riscv64 roots: busybox, bash, coreutils, dpkg, apt, perl, and CPython run. In the same OrbStack VM, rvrun runs Lua 4.8× and SQLite 4.2× faster than the `qemu-riscv64` registered there (93 vs. 446 ms, 35 vs. 146 ms).
- **Oct 14, 2026 — VM mode starts as Direct mode on a Linux backend.** The first M6 step keeps `Process` (guest memory, threads, signals, futexes) as it is and swaps only the `HostOs` backend: `host::linux` passes syscalls to the host kernel, translating just the four `open` flags whose arm64 bits differ from asm-generic. The whole suite, including Alpine's busybox and CPython, passes on arm64 Linux with no other change, and Lua runs within about 10% of its macOS time inside OrbStack's VM. Passing through what `Process` does itself (`clone`, signals, `execve` through binfmt\_misc) is the next decision, once programs need it.
- **Oct 14, 2026 — M5 done.** Multithreaded C and Go programs, Alpine's busybox (shell pipelines, job control) and CPython 3.14 (subprocess, asyncio, multiprocessing with shared memory) run under Direct mode and match native output in the test suite. What Direct mode still lacks is listed in [Known Gaps](#known-gaps); none of it blocks M6, which reuses the translator inside a Linux VM where most of those gaps disappear.
- **Oct 14, 2026 — Dynamically linked programs run against a sysroot, resolved like a chroot.** `rvrun -L DIR` (or `RVRUN_SYSROOT`) names a riscv64 userland; the tests use Alpine's minirootfs and packages, pinned by SHA-256. Absolute guest paths resolve inside it when they exist there, following absolute symlinks within it (distributions are full of them), and on the host otherwise; host programs found that way are reported missing to `execve`, so PATH searches reach the guest's own. Alpine's busybox and CPython 3.14 (with subprocess, asyncio, and multiprocessing) run unmodified.
- **Oct 14, 2026 — epoll and eventfd are emulated on kqueue in the Darwin backend.** Go's netpoller needs them even for programs without networking. An epoll instance is a kqueue whose filters carry the guest's `epoll_data` in `udata`; regular files are refused with `EPERM` as on Linux. An eventfd is a counter kept by the backend plus a socketpair end that holds a byte while the counter is nonzero. VM mode will pass both through.
- **Oct 14, 2026 — M5 grows fork/exec and sockets; Python is a target.** Dynamically linked programs like Python need `subprocess`, which needs `fork` and `execve`. In Direct mode the whole guest lives in one host process, so `fork` can be the host's (copy-on-write duplicates guest memory and translations alike) and `execve` a fresh rvrun.
- **Oct 14, 2026 — Code cache as snapshots of the code buffer, not ahead-of-time translation.** Translating a whole executable up front would translate cold and data-in-text bytes and needs a separate code layout; saving what a run actually translated reuses the JIT unchanged and only required removing the one host address from generated code. See [Persistent code cache](#persistent-code-cache).
- **Oct 14, 2026 — Generated code keeps masking guest addresses.** The mask keeps a guest from reaching host memory outside its reservation, as a sandbox would (QEMU linux-user does not mask). It costs Lua 5% and SQLite 3%, on pointer chasing; isolation is worth more than that.
- **Oct 14, 2026 — `instret` stays exact.** Counted per block in x30, it costs under 1% (removing it entirely measured 0.6–0.8%), so there is no case for a cheaper approximate mode.
- **Oct 14, 2026 — No return-address prediction for guest returns.** We translated guest calls to `bl` into a call stub with a landing pad, and returns to `ret` through the jump cache, as Rosetta 2 does. The first version made recursive code 2× slower: overlapping blocks each had their own landing pad for the same call, so the `bl` pushed one pad while the jump cache held another and every return mispredicted. With one shared stub per call the mispredicts were gone (no profile samples on landing pads), but Lua, SQLite, and Rust tests ended within ±3% of plain `br`: Apple's indirect predictor already predicts our returns well, and the stubs cost a host register (t2 went back to memory). Dropped; revisit for call-heavy C++ code with many callers per function.
- **Oct 14, 2026 — Mask the base register, not the sum, behind guard pages.** Generated loads and stores mask a base register once into the guest space and reuse the host address across accesses with immediate offsets. Offsets can then reach up to 32 KiB past either end of the guest space, so the reservation has 64 KiB guard regions there; such accesses fault with their true address instead of wrapping.

- **Oct 14, 2026 — Direct mode first, then VM mode as a full milestone (M6).** The project's first goal is now a test and emulation environment for RISC-V developers on Apple Silicon. Direct mode covers the daily build → test → debug loop best (instant, no VM, Mac file system); VM mode (Linux VM + syscall passthrough) is promoted from escape hatch to M6 for containers and full distributions. The core must therefore also run on arm64 Linux — which also makes the Linux build usable directly on arm64 Linux hosts; see [Goals and Positioning](#goals-and-positioning).
- **Oct 14, 2026 — M3 done; gdbstub runs on the reference interpreter.** `rvrun -g PORT` serves the GDB remote protocol through the `gdbstub` crate. While attached, the guest single-steps on the oracle interpreter, so breakpoints (kept on our side, code is never patched), stepping, and Ctrl-C are exact; detaching hands the guest back to the engine. Only x0–x31 and pc are exposed for now (the stock RISC-V target description); FP registers need a custom description.
- **Oct 14, 2026 — Static-PIE moves into M3.** Rust's musl targets link static-PIE by default, so `cargo test` needs it; images load at `PIE_BASE` (2/3 up, like Linux) and relocate themselves.
- **Oct 14, 2026 — `cargo test` works before threads.** libtest falls back to the main thread when `clone` fails, so the M3 criterion holds without M5; parallel test execution waits for M5 (and works since).
- **Oct 14, 2026 — FP reference is Berkeley SoftFloat (RISC-V specialization), vendored.** Same code as Spike; used to fuzz the host-FPU path and as the RMM fallback. Existing crates either fail to build with current CMake or are LGPL.
- **Oct 14, 2026 — Performance target: ≤ 2× native, stretch Rosetta 2 class (70–80% of native).** See [Target](#target). Consequence for M3: F/D runs on the host FPU, with the interpreter using the same `FPCR`/`FPSR` mapping the JIT will use, so the hardware-mapping claims are validated before M4.
- **Oct 14, 2026 — Fast interpreter before M3.** A decoded-block cache with chaining plus a software TLB takes the CPU benchmark from 41 to ~270 MIPS (6.6×), and is the cache structure the JIT will reuse; see [What we borrow from Rosetta 2](#what-we-borrow-from-rosetta-2).
- **Oct 14, 2026 — The runtime is thread-ready from M2.** Process/thread split, `&self` memory with atomic accesses, host atomics for AMOs and `lr`/`sc`, and lock-free host I/O on raw guest buffers; see [Concurrency Model](#concurrency-model). Locks stay interpreter-only.
- **Oct 14, 2026 — F/D data movement moves into M2.** musl's `printf` saves FP registers and moves bits with `fmv`/`fsgnj`, so every program that prints needs them; FP arithmetic stays in M3.
- **Oct 14, 2026 — M and A move into M2.** The static musl `hello` already uses `mul`, `divu`, `lr.w`, and `sc.w`.
- **Oct 13, 2026 — RV64 only.** RV32 is not supported; see [Instruction Set Scope](#instruction-set-scope).
- **Oct 13, 2026 — C extension moves into M2.** Real binaries are full of compressed instructions, so `hello` cannot run without them.
- **Oct 13, 2026 — Guest address-space layout is decided in M2.** Guest memory lives at a fixed host offset (identity mapping collides with `__PAGEZERO`), and the layout dictates the JIT's memory addressing; see [Performance Strategy](#performance-strategy).
