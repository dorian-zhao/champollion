#!/bin/sh
# Register rvrun as OrbStack's binfmt_misc handler for riscv64 ELF programs.
#
#   tools/orbstack-binfmt.sh install [PATH-TO-RVRUN]
#   tools/orbstack-binfmt.sh status
#   tools/orbstack-binfmt.sh uninstall
#
# OrbStack snapshots the interpreter when the registration is written, and its
# binfmt table is runtime state inside the OrbStack VM. Re-run `install` after
# upgrading rvrun or restarting OrbStack.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
image=${CHAMPOLLION_ORBSTACK_IMAGE:-alpine:3.22}

usage() {
    echo "usage: $0 install [PATH-TO-RVRUN] | status | uninstall" >&2
}

require_orbstack() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "error: docker is not installed" >&2
        exit 1
    fi
    os=$(docker info --format '{{.OperatingSystem}}' 2>/dev/null || true)
    case "$os" in
        *OrbStack*) ;;
        *)
            echo "error: the active Docker context is not OrbStack (got: ${os:-unavailable})" >&2
            echo "run: docker context use orbstack" >&2
            exit 1
            ;;
    esac
}

register() {
    bin=$1
    if [ ! -f "$bin" ]; then
        echo "error: missing $bin" >&2
        echo "download the aarch64 Linux asset from the v0.1.1 release, or build it" >&2
        echo "with: tools/linux.sh build --release -p rvrun" >&2
        exit 1
    fi
    bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
    require_orbstack
    docker run --rm --privileged --pid=host \
        -v "$bin:/build/rvrun:ro" \
        "$image" sh -c '
            set -eu
            cp /build/rvrun /usr/local/bin/rvrun-riscv64
            mount -t binfmt_misc none /proc/sys/fs/binfmt_misc 2>/dev/null || true
            echo -1 > /proc/sys/fs/binfmt_misc/rvrun-riscv64 2>/dev/null || true
            printf "%s\n" ":rvrun-riscv64:M::\x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/local/bin/rvrun-riscv64:POCF" \
                > /proc/sys/fs/binfmt_misc/register || true
        '
    echo "registered $bin"
    echo "checking with an Alpine riscv64 container:"
    docker run --rm --platform linux/riscv64 "$image" uname -m
}

status() {
    require_orbstack
    docker run --rm --privileged --pid=host "$image" sh -c '
        mount -t binfmt_misc none /proc/sys/fs/binfmt_misc 2>/dev/null || true
        if [ -f /proc/sys/fs/binfmt_misc/rvrun-riscv64 ]; then
            cat /proc/sys/fs/binfmt_misc/rvrun-riscv64
        else
            echo "rvrun-riscv64 is not registered" >&2
            exit 1
        fi
    '
}

uninstall() {
    require_orbstack
    docker run --rm --privileged --pid=host "$image" sh -c '
        mount -t binfmt_misc none /proc/sys/fs/binfmt_misc 2>/dev/null || true
        echo -1 > /proc/sys/fs/binfmt_misc/rvrun-riscv64 2>/dev/null || true
    '
    echo "removed rvrun-riscv64 from OrbStack"
}

case "${1:-status}" in
    install) register "${2:-$root/target/linux/release/rvrun}" ;;
    status) status ;;
    uninstall) uninstall ;;
    *) usage; exit 2 ;;
esac
