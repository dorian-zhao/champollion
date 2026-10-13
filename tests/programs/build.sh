#!/bin/sh
# Build real programs (lua, sqlite3) as static riscv64-linux-musl executables into
# tests/fixtures/programs/, and generate each test script's expected output by running the
# same source built natively. Outputs are committed so `cargo test` needs no network or zig.
#
#   git submodule update --init third_party/lua
#   tests/programs/build.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$root/tests/fixtures/programs"
cache="$here/.cache"
mkdir -p "$out" "$cache"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

rv() { zig cc -target riscv64-linux-musl -static -O2 -s "$@"; }

# Lua, from the pinned submodule.
lua_src="$root/third_party/lua/onelua.c"
rv -DLUA_USE_POSIX "$lua_src" -o "$out/lua" -lm
cc -O2 -DLUA_USE_POSIX "$lua_src" -o "$tmp/lua"

# SQLite amalgamation, pinned by version and SHA3-256 from sqlite.org/download.html.
sqlite=sqlite-amalgamation-3540000
sqlite_sha3=7b670a62fdfbd672b75fef004cb703c8a3e87d3a5cc7d675b4a08337004a2d93
if [ ! -d "$cache/$sqlite" ]; then
    curl -fsSL "https://www.sqlite.org/2026/$sqlite.zip" -o "$cache/$sqlite.zip"
    python3 -I -c 'import hashlib, sys
digest = hashlib.sha3_256(open(sys.argv[1], "rb").read()).hexdigest()
sys.exit(0 if digest == sys.argv[2] else f"checksum mismatch: {digest}")' "$cache/$sqlite.zip" "$sqlite_sha3"
    (cd "$cache" && unzip -q "$sqlite.zip")
fi
sqlite_flags="-DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION"
rv $sqlite_flags "$cache/$sqlite/shell.c" "$cache/$sqlite/sqlite3.c" -o "$out/sqlite3"
cc -O2 $sqlite_flags "$cache/$sqlite/shell.c" "$cache/$sqlite/sqlite3.c" -o "$tmp/sqlite3"

# Expected outputs. Scripts must be deterministic (no clocks, no randomness).
for script in "$here"/*.lua; do
    "$tmp/lua" "$script" > "$script.expected"
done
for script in "$here"/*.sql; do
    "$tmp/sqlite3" :memory: < "$script" > "$script.expected"
done
