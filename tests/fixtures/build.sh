#!/bin/sh
# Rebuild RISC-V test fixtures. Outputs are committed so `cargo test` doesn't need zig.
#   *.c -> static riscv64-linux-musl executable
#   *.s -> raw .text bytes (*.bin), used as reference encodings
set -eu
cd "$(dirname "$0")"
for src in *.c; do
    zig cc -target riscv64-linux-musl -static -O2 -s "$src" -o "${src%.c}"
done
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for src in *.s; do
    zig cc -target riscv64-linux-musl -c "$src" -o "$tmp/obj.o"
    zig objcopy -O binary -j .text "$tmp/obj.o" "${src%.s}.bin"
done
