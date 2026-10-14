#!/bin/sh
# Run cargo on arm64 Linux (route A's host) in a container, e.g. on OrbStack or Docker
# Desktop: `tools/linux.sh test --release`, `tools/linux.sh clippy --all-targets`.
# With no arguments, runs the test suite. Builds go to target/linux.
set -eu
cd "$(dirname "$0")/.."
[ $# -eq 0 ] && set -- test --release
exec docker run --rm -v "$PWD:/src" -v champollion-cargo:/usr/local/cargo/registry -w /src \
    -e CARGO_TARGET_DIR=/src/target/linux -e FUZZ_CASES -e FUZZ_SEED -e RVRUN_STATS \
    rust:1-alpine \
    sh -c 'rustup component add clippy >/dev/null 2>&1; exec cargo "$@"' cargo "$@"
