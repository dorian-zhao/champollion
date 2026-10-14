#!/bin/sh
# Smoke test for rvcontainer: run riscv64 Alpine and Debian images and check what comes
# back. Needs a built launcher (build.sh), the kernel (fetch-kernel.sh), and network access
# on first use. Exits nonzero on the first mismatch.
set -eu
cd "$(dirname "$0")"
rvc=.build/release/rvcontainer
fail=0
check() {
    name=$1 want=$2
    shift 2
    got=$("$rvc" run "$@" </dev/null 2>&1) || true
    if [ "$got" = "$want" ]; then
        echo "ok $name"
    else
        echo "FAIL $name"
        echo "  want: $want"
        echo "  got:  $got"
        fail=1
    fi
}
check "alpine shell" "riscv64 hello" docker.io/library/alpine:3.22 sh -c 'echo $(uname -m) hello'
if "$rvc" run docker.io/library/alpine:3.22 sh -c 'exit 7' </dev/null; then status=0; else status=$?; fi
if [ "$status" -eq 7 ]; then echo "ok exit status"; else echo "FAIL exit status: $status"; fail=1; fi
check "debian glibc" "riscv64 /root" docker.io/library/debian:trixie-slim bash -c 'echo $(uname -m) $HOME'
check "volume" "$(cat ../tests/programs/basics.lua.expected)" \
    -v "$PWD/../tests:/t" docker.io/library/alpine:3.22 /t/fixtures/programs/lua /t/programs/basics.lua
check "setuid" "0" docker.io/library/debian:trixie-slim bash -c \
    'cp /usr/bin/id /tmp/sid && chmod u+s /tmp/sid && su -s /bin/sh nobody -c "/tmp/sid -u"'
exit $fail
