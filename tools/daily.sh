#!/bin/sh
# The daily-use benchmark gate: everyday shapes - process startup, a script loop, SQL -
# built natively and for riscv64, timed side by side. Prints our speed as a percentage
# of native next to the Rosetta 2 band (70-80% of native), and fails when a case
# regresses past the committed baseline. Timings are the minimum of each run set:
# the startup shapes are noisy, and the floor is what a regression would move.
#
#   tools/daily.sh                    # run and compare against tools/daily.baseline
#   tools/daily.sh --update-baseline  # accept the current numbers
#
# The riscv64 fixtures are committed; the native builds go to target/daily (cc, and
# the sqlite amalgamation cache that tests/programs/build.sh downloads).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
out="$root/target/daily"
mkdir -p "$out"
update=0
[ "${1:-}" = "--update-baseline" ] && update=1

(cd "$root" && cargo build --release -p rvrun >/dev/null)
rvrun="$root/target/release/rvrun"

# Native counterparts of the committed riscv64 fixtures.
lua_src="$root/third_party/lua/onelua.c"
if [ ! -f "$lua_src" ]; then
    echo "missing $lua_src; run: git submodule update --init third_party/lua" >&2
    exit 1
fi
if [ ! -x "$out/lua-native" ]; then
    cc -O2 -DLUA_USE_POSIX "$lua_src" -o "$out/lua-native" -lm
fi

sqlite=sqlite-amalgamation-3540000
cache="$root/tests/programs/.cache"
if [ ! -d "$cache/$sqlite" ]; then
    mkdir -p "$cache"
    curl -fsSL "https://www.sqlite.org/2026/$sqlite.zip" -o "$cache/$sqlite.zip"
    (cd "$cache" && unzip -q "$sqlite.zip")
fi
if [ ! -x "$out/sqlite3-native" ]; then
    cc -O2 -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION \
        "$cache/$sqlite/shell.c" "$cache/$sqlite/sqlite3.c" -o "$out/sqlite3-native"
fi

if [ ! -x "$out/hello-native" ]; then
    printf '#include <stdio.h>\nint main(void) { puts("Hello, RISC-V!"); return 0; }\n' \
        > "$out/hello.c"
    cc -O2 "$out/hello.c" -o "$out/hello-native"
fi

# Snippet files keep every timing command free of shell quoting (hyperfine -N).
printf 'local s=0\nfor i = 1, 3000000 do s = s + i * i * 0.5 end\nprint(s)\n' \
    > "$out/lua-compute.lua"
: > "$out/lua-start.lua"
cp "$root/tests/programs/basics.sql" "$out/basics.sql"
printf 'WITH RECURSIVE c(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM c WHERE x < 1000000)\nSELECT count(*), sum(x) FROM c;\n' \
    > "$out/sqlite-cte.sql"

# name, ours, native; the short-lived cases need more runs to settle.
bench() {
    hyperfine -N --warmup 3 --runs "$1" --export-json "$out/$2.json" "$3" "$4" >/dev/null
}

bench "${DAILY_RUNS:-15}" hello "$rvrun $root/tests/fixtures/hello" "$out/hello-native"
bench "${DAILY_RUNS:-15}" lua-start "$rvrun $root/tests/fixtures/programs/lua $out/lua-start.lua" \
    "$out/lua-native $out/lua-start.lua"
bench "${DAILY_HEAVY_RUNS:-7}" lua-compute "$rvrun $root/tests/fixtures/programs/lua $out/lua-compute.lua" \
    "$out/lua-native $out/lua-compute.lua"
bench "${DAILY_HEAVY_RUNS:-7}" sqlite-script "$rvrun $root/tests/fixtures/programs/sqlite3 -init $out/basics.sql :memory:" \
    "$out/sqlite3-native -init $out/basics.sql :memory:"
bench "${DAILY_HEAVY_RUNS:-7}" sqlite-cte "$rvrun $root/tests/fixtures/programs/sqlite3 -init $out/sqlite-cte.sql :memory:" \
    "$out/sqlite3-native -init $out/sqlite-cte.sql :memory:"

python3 - "$root" "$out" "$update" <<'PY'
import json
import pathlib
import sys

root, out, update = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
cases = ["hello", "lua-start", "lua-compute", "sqlite-script", "sqlite-cte"]

def wall(entry):
    summary = entry.get("summary", entry)
    if "time_wall_clock" in summary:  # nested per-metric summaries
        return summary["time_wall_clock"]["min"]
    return summary["min"]


rows = []
for name in cases:
    data = json.loads(pathlib.Path(f"{out}/{name}.json").read_text())
    ours = wall(data["results"][0]) * 1e3
    native = wall(data["results"][1]) * 1e3
    rows.append((name, ours, native, ours / native))

print(f"\n{'case':<15}{'ours':>10}{'native':>10}{'slowdown':>11}{'of native':>12}  rosetta")
for name, ours, native, ratio in rows:
    band = "yes" if ratio <= 1.43 else ""
    print(f"{name:<15}{ours:>8.2f}ms{native:>8.2f}ms{ratio:>10.2f}x{100 / ratio:>10.0f}%   {band}")

baseline = pathlib.Path(f"{root}/tools/daily.baseline")
if update:
    baseline.write_text("".join(f"{name} {ratio:.3f}\n" for name, _, _, ratio in rows))
    print(f"\nbaseline written to {baseline}")
    sys.exit(0)
if not baseline.exists():
    print("\nno tools/daily.baseline yet; run with --update-baseline to record one")
    sys.exit(0)

known = {}
for line in baseline.read_text().splitlines():
    name, ratio = line.split()
    known[name] = float(ratio)
regressed = [
    f"{name}: {ratio:.2f}x (baseline {known[name]:.2f}x)"
    for name, _, _, ratio in rows
    if name in known and ratio > known[name] * 1.10
]
if regressed:
    print("\nREGRESSED past the baseline (tolerance 10%):")
    for line in regressed:
        print(f"  {line}")
    sys.exit(1)
print("\nno regressions against tools/daily.baseline")
PY
