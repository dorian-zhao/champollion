# Using Champollion

Champollion runs riscv64 Linux programs on arm64. There are two broad ways to
use it:

- **Direct mode** runs one riscv64 executable as a Mac process. It is the
  fastest path for a local build → test → debug loop.
- **VM mode** runs against a real Linux kernel. It is the right choice for
  containers, distribution userlands, `apt`, `/proc`, and tools that need
  `ptrace` or kernel state.

On Apple silicon, the recommended VM-mode setup is **OrbStack**. It keeps a
persistent arm64 Linux VM, uses the normal Docker CLI, and lets
`docker run --platform linux/riscv64 ...` work after one binfmt registration.
Apple's `containerization` launcher (`rvcontainer`) remains available for
users who want an Apple-only stack.

## Choose a mode

| What you want | Use |
| --- | --- |
| Run riscv64 containers, `apt`, or a full Ubuntu/Debian userland on a Mac | **OrbStack** (recommended) |
| Run or debug one riscv64 executable as a normal Mac process | Direct mode |
| Stay inside Apple's Virtualization.framework/containerization stack | `rvcontainer` |
| Run riscv64 binaries on an arm64 Linux host or CI runner | Linux build + `binfmt_misc` |

OrbStack is only available on Apple silicon Macs. The Linux build is arm64;
Intel Macs are not supported.

## Recommended on macOS: OrbStack

### 1. Install or start OrbStack

Install [OrbStack](https://orbstack.dev/), then make its Docker daemon the
active context:

```sh
orb start
docker context use orbstack
docker info --format '{{.OperatingSystem}} {{.Architecture}}'
# OrbStack aarch64
```

If the last line does not say `OrbStack aarch64`, fix the context before
continuing.

### 2. Get the Linux build

Download the arm64 Linux asset from the
[latest release](https://github.com/dorian-zhao/champollion/releases/latest),
or build it from the repository:

```sh
curl -fL -o /tmp/rvrun-linux.tar.gz \
  https://github.com/dorian-zhao/champollion/releases/download/v0.1.1/rvrun-v0.1.1-aarch64-unknown-linux-musl.tar.gz
tar -xzf /tmp/rvrun-linux.tar.gz -C /tmp
mkdir -p "$HOME/.local/share/champollion"
install -m755 /tmp/rvrun-v0.1.1-aarch64-unknown-linux-musl/rvrun \
  "$HOME/.local/share/champollion/rvrun"
```

From a checkout, the equivalent is:

```sh
tools/linux.sh build --release -p rvrun
install -m755 target/linux/release/rvrun "$HOME/.local/share/champollion/rvrun"
```

### 3. Register the handler

The registration has to run in a privileged container with access to
OrbStack's PID namespace. The helper script encapsulates the verified recipe:

```sh
curl -fL -o /tmp/orbstack-binfmt.sh \
  https://raw.githubusercontent.com/dorian-zhao/champollion/main/tools/orbstack-binfmt.sh
chmod +x /tmp/orbstack-binfmt.sh
/tmp/orbstack-binfmt.sh install "$HOME/.local/share/champollion/rvrun"
```

If you are already in a checkout, use `tools/orbstack-binfmt.sh` directly.
The script checks the OrbStack context, registers the handler, and smoke-tests
it with an Alpine riscv64 container. A successful run ends with:

```text
registered /Users/you/.local/share/champollion/rvrun
checking with an Alpine riscv64 container:
riscv64
```

`--privileged` and `--pid=host` are needed only for registration. Do not run
your application containers with those flags.

### 4. Run a riscv64 container

```sh
docker run --rm --platform linux/riscv64 alpine:3.22 uname -m
# riscv64

docker run --rm --platform linux/riscv64 ubuntu:26.04 sh -c \
  'cat /etc/os-release; uname -a'
```

The kernel is a real Linux kernel inside OrbStack's VM. This is VM mode, not a
Mac process wrapper: `apt`, `/proc`, threads, signals, and container images
work as they do on an arm64 Linux host.

## Common OrbStack recipes

### Run a binary from the Mac

```sh
docker run --rm --platform linux/riscv64 \
  -v "$PWD:/work:ro" -w /work \
  ubuntu:26.04 ./hello
```

The binary can be statically linked, or dynamically linked with the libraries
present in the container. This is the easiest way to exercise a cross-compiled
riscv64 program with a complete Linux userland.

### Interactive Ubuntu

```sh
docker run --rm -it --platform linux/riscv64 ubuntu:26.04 bash
```

Inside the shell:

```sh
apt-get update
apt-get install -y curl
uname -m
```

### Build or test inside a riscv64 container

```sh
docker run --rm --platform linux/riscv64 \
  -v "$PWD:/src" -w /src \
  ubuntu:26.04 sh -c './configure && make -j2 && make test'
```

Start with a small `-j` value. Parallel builds can be memory-hungry, and
OrbStack's default resources may need to be raised in its settings.

### Check which handler is active

```sh
tools/orbstack-binfmt.sh status
```

It should print `enabled`, `interpreter /usr/local/bin/rvrun-riscv64`, and the
`POCF` flags.

## Updating rvrun and re-registering

OrbStack opens and snapshots the interpreter when the registration is written.
Replacing the binary on disk does not update an existing registration.

After downloading a new release or rebuilding:

```sh
install -m755 /path/to/new/rvrun "$HOME/.local/share/champollion/rvrun"
/tmp/orbstack-binfmt.sh install "$HOME/.local/share/champollion/rvrun"
```

OrbStack's binfmt table is runtime state in its VM. Re-run `install` after
restarting OrbStack. Containers that were already running keep the old process
image; recreate them after an update.

The script tolerates OrbStack's occasional `EEXIST` return from
`/proc/sys/fs/binfmt_misc/register` and verifies the result by running a
riscv64 container. This matches the behavior documented in
[`notes/debugging-playbook.md`](../notes/debugging-playbook.md).

To remove the handler:

```sh
/tmp/orbstack-binfmt.sh uninstall
```

## Verify the handler binary

The handler is opened at registration, so `/proc/<pid>/exe` is the reliable
place to check which rvrun OrbStack is actually using. Compare its SHA-256 with
the binary you installed:

```sh
local_hash=$(shasum -a 256 "$HOME/.local/share/champollion/rvrun" | awk '{print $1}')
docker run -d --name chmp-handler-check --platform linux/riscv64 \
  alpine:3.22 sleep 30
docker exec chmp-handler-check sh -c 'sha256sum /proc/1/exe'
docker rm -f chmp-handler-check
```

The two hashes must match. A file listing elsewhere in the container is not a
reliable check: OrbStack keeps the registered interpreter in its own storage.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `exec format error` for a riscv64 image | The current OrbStack VM has no handler. Re-run `tools/orbstack-binfmt.sh install …`. |
| `docker info` shows Docker Desktop, Colima, or nothing | Run `docker context use orbstack`, then check `docker info --format '{{.OperatingSystem}} {{.Architecture}}'`. |
| `uname -m` prints `aarch64` | Add `--platform linux/riscv64`; Docker may otherwise choose a native arm64 image. |
| It worked, then stopped after an OrbStack restart | Re-register; the binfmt table is runtime state. |
| A new rvrun build is ignored | Re-run `install`; OrbStack snapshots the interpreter at registration. |
| `status` says not registered | Re-run `install` and check that the path is absolute and readable. |
| Parallel builds are killed | Raise OrbStack's memory/CPU limits or reduce `-j`. |

For a deeper debugging checklist, see
[`notes/debugging-playbook.md`](../notes/debugging-playbook.md).

## Direct mode on macOS

For a single program, use `rvrun` directly:

```sh
cargo build --release
target/release/rvrun tests/fixtures/hello
```

Direct mode translates Linux system calls to Darwin calls. It is ideal for a
local edit/build/test loop and has a built-in gdbstub:

```sh
target/release/rvrun -g 1234 ./program
riscv64-elf-gdb ./program -ex 'target remote :1234'
```

Use VM mode when the program needs containers, distribution package
management, `ptrace`, or a real Linux `/proc`. See the
[main README](../README.md#direct-mode-rvrun) for options and sysroot setup.

## Alternative on macOS: `rvcontainer`

`rvcontainer` uses Apple's `containerization` framework and boots a fresh
arm64 Linux VM for a run. It is useful if you want to stay entirely inside
Apple's stack or need to test the launcher.

```sh
launcher/fetch-kernel.sh
launcher/build.sh
launcher/.build/release/rvcontainer run docker.io/library/debian:trixie-slim bash
```

Requirements: macOS 26 or later and Xcode 26 or later. Each `run` starts a
fresh VM and unpacks the image again, so it is not a replacement for a warm
Docker/OrbStack workflow.

## On an arm64 Linux host

The Linux build is a normal `binfmt_misc` handler; no VM is involved:

```sh
tools/linux.sh build --release -p rvrun
sudo install -m755 target/linux/release/rvrun /usr/local/bin/rvrun-riscv64
sudo sh -c 'printf "%s\n" ":rvrun-riscv64:M::\x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/local/bin/rvrun-riscv64:POCF" > /proc/sys/fs/binfmt_misc/register'
docker run --rm --platform linux/riscv64 alpine:3.22 uname -m
```

If `/proc/sys/fs/binfmt_misc` is not mounted, run
`sudo mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc` first.

## What the flags mean

The handler is registered with `POCF`:

- `P` preserves `argv[0]`, so the guest sees the command it invoked.
- `O` passes the opened executable to rvrun.
- `C` computes credentials from the binary, like a normal setuid exec.
- `F` fixes the interpreter at registration time. This is why upgrading or
  replacing `rvrun` requires re-registering, and why OrbStack snapshots it.

The magic and mask match ELF64, little-endian, EM_RISCV (`e_machine = 0xf3`).
Do not edit them by hand; use the helper script.

## Limits

- The RVV subset is experimental in v0.1.1. Set `RVRUN_V=1` for experiments;
  guests are not told that V is available yet.
- Direct mode does not cache dynamically linked programs yet, and it cannot
  provide a real Linux kernel.
- VM mode's `rvcontainer` path starts a fresh VM per run. The OrbStack path
  avoids that and is the recommended macOS workflow.
- The Linux build is arm64-only. Apple silicon is required on macOS.

For the current performance numbers and remaining gaps, see the
[main README](../README.md#performance) and
[`docs/design.md`](design.md#known-gaps).
