#!/bin/sh
# Build the riscv-arch-test (ACT4) user-mode subset into tests/fixtures/act/.
#
#   git submodule update --init third_party/riscv-arch-test
#   tests/act/build.sh                  # build the fixtures (needs zig)
#   tests/act/build.sh signatures       # refresh tests/fixtures/act/*.sig (needs qemu-riscv64)
#
# The tests come from the directories listed in tests/act/dirs.txt. Each is assembled
# against the framework's env (third_party/riscv-arch-test/tests/env) overlaid with
# our overrides (tests/act/env: a user-mode rvmodel plus a hand-written DUT config, and
# a sail_macros.h shadow), linked with tests/act/link.ld, and runs as a static riscv64
# Linux executable that writes its signature region to stdout and exits 0.
#
# The committed .sig files are qemu-riscv64's stdout for the same ELF. qemu-user only
# exists on Linux, so `signatures` is meant to run in an arm64 container:
#
#   docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w ubuntu:24.04 \
#     sh -c 'apt-get update -qq && apt-get install -y -qq qemu-user bzip2 \
#            >/dev/null && tests/act/build.sh signatures'
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
act="$root/third_party/riscv-arch-test"
out="$root/tests/fixtures/act"

if [ ! -d "$act/tests/rv64i" ]; then
    echo "build.sh: third_party/riscv-arch-test is missing; run:" >&2
    echo "  git submodule update --init third_party/riscv-arch-test" >&2
    exit 1
fi

# The env the tests are compiled against: the framework's headers with our overrides
# copied over them (quoted includes find same-directory files first, so the overlay
# must be physical).
work="$root/target/act-env"
rm -rf "$work"
mkdir -p "$work"
cp "$act"/tests/env/*.h "$work"/
cp "$here"/env/*.h "$work"/

# Extensions the tests may assemble with. F/D are assembler-only here: the DUT config
# (env/rvtest_config.h) must not claim them, because the framework's boot then writes
# mstatus (an M-mode CSR) and the user-mode test dies on it.
cpu="baseline_rv64+c+m+a+f+d+zba+zbb+zbs+zbc+zbkb+zbkc+zbkx+zicond+zmmul+zca+zcb+zcmop+zifencei+zicsr+zihintpause+zihintntl"

skipped() {
    grep -v '^#' "$here/skip.txt" | grep -q "^$1" && return 0
    return 1
}

mkdir -p "$out"
built=0
for dir in $(sed 's/#.*//' "$here/dirs.txt" | tr -s '\n' ' '); do
    for src in "$act/tests/rv64i/$dir"/*.S; do
        name=$(basename "$src" .S)
        if skipped "$dir/$name"; then
            continue
        fi
        zig cc -target riscv64-linux-musl -mcpu="$cpu" -mno-relax -nostdlib -static -s \
            -Wl,--build-id=none -Wl,-T,"$here"/link.ld -Wl,-e,rvtest_entry_point \
            -I "$work" -DTEST_FILE="\"$name.S\"" -DTEST_FLEN=64 \
            "$src" -o "$out/$name.elf"
        built=$((built + 1))
    done
done
echo "built $built tests into tests/fixtures/act"

if [ "${1:-}" = signatures ]; then
    if ! command -v qemu-riscv64 >/dev/null; then
        echo "build.sh: qemu-riscv64 not found; see the comment at the top" >&2
        exit 1
    fi
    for elf in "$out"/*.elf; do
        qemu-riscv64 "$elf" > "${elf%.elf}.sig"
    done
    echo "refreshed $(ls "$out"/*.sig | wc -l) signatures from qemu-riscv64"
fi
