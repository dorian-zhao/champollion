#!/bin/sh
# Rebuild RISC-V test fixtures. Outputs are committed so `cargo test` doesn't need zig.
#   *.c -> static riscv64-linux-musl executable
#   *.s -> raw .text bytes (*.bin), used as reference encodings (arm64*.s for the host)
set -eu
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for src in *.c; do
    zig cc -target riscv64-linux-musl -static -O2 -s "$src" -o "${src%.c}"
done

# fp.c only uses correctly rounded operations, so the native arm64 build defines the
# expected output for the riscv64 build.
cc -O2 -frounding-math fp.c -o "$tmp/fp-native"
"$tmp/fp-native" > fp.expected
for src in *.s; do
    # arm64*.s are host (JIT output) encodings; the rest are RISC-V.
    case $src in
    arm64*) target=aarch64-linux-musl ;;
    *) target=riscv64-linux-musl ;;
    esac
    zig cc -target "$target" -c "$src" -o "$tmp/obj.o"
    zig objcopy -O binary -j .text "$tmp/obj.o" "${src%.s}.bin"
done
