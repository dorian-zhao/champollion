#!/bin/sh
# Rebuild RISC-V test fixtures. Outputs are committed so `cargo test` doesn't need zig.
set -eu
cd "$(dirname "$0")"
for src in *.c; do
    zig cc -target riscv64-linux-musl -static -O2 -s "$src" -o "${src%.c}"
done
