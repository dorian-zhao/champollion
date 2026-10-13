#!/bin/sh
# Linker for riscv64gc-unknown-linux-musl: zig's cross toolchain.
exec zig cc -target riscv64-linux-musl "$@"
