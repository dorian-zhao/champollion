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

# build_suite <suite> <cpu> [output-name]
build_suite() {
    suite=$1 cpu=$2 name_out=${3:-$1}
    out="$root/tests/fixtures/$name_out"
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
build_suite rv64um generic_rv64+m
build_suite rv64ua generic_rv64+m+a
build_suite rv64uc generic_rv64+c

# F/D: only the load/store tests for now; the rest need FP arithmetic and fcsr (M3).
build_suite_only() {
    suite=$1 cpu=$2 out_name=$3
    shift 3
    out="$root/tests/fixtures/$out_name"
    rm -rf "$out" && mkdir -p "$out"
    for name in "$@"; do
        zig cc -target riscv64-linux-musl -mcpu="$cpu" -mno-relax \
            -nostdlib -static -s -Wl,--build-id=none \
            -I"$here" -I"$isa/macros/scalar" \
            "$isa/$suite/$name.S" -o "$out/$name"
    done
}
build_suite_only rv64uf generic_rv64+f rv64uf ldst
build_suite_only rv64ud generic_rv64+f+d rv64ud ldst

# The same sources again with C enabled: the assembler compresses every eligible
# instruction, which exercises the C decoder on hundreds of real instruction sequences.
build_suite rv64ui generic_rv64+c rv64ui-c
build_suite rv64um generic_rv64+m+c rv64um-c
build_suite rv64ua generic_rv64+m+a+c rv64ua-c
