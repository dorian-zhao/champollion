#!/bin/sh
# Build riscv-tests suites as static riscv64 Linux executables using our user-mode env
# (riscv_test.h). Outputs go to tests/fixtures/<suite>/ and are committed so `cargo test`
# needs neither zig nor the submodule.
#
#   git submodule update --init third_party/riscv-tests
#   tests/riscv-tests/build.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
isa="$root/third_party/riscv-tests/isa"

# Tests that need extensions we don't support yet.
skip="rv64ui/fence_i" # Zifencei + self-modifying code (M3)

build_suite() {
    suite=$1 cpu=$2
    out="$root/tests/fixtures/$suite"
    rm -rf "$out" && mkdir -p "$out"
    for src in "$isa/$suite"/*.S; do
        name=$(basename "$src" .S)
        case " $skip " in *" $suite/$name "*) continue ;; esac
        zig cc -target riscv64-linux-musl -mcpu="$cpu" -mno-relax \
            -nostdlib -static -s -Wl,--build-id=none \
            -I"$here" -I"$isa/macros/scalar" \
            "$src" -o "$out/$name"
    done
}

build_suite rv64ui generic_rv64
