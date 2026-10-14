#!/bin/sh
# Fetch a riscv64 Linux userland for dynamically linked guests: the Alpine minirootfs
# (musl's dynamic loader and libc, busybox, apk), pinned by version and SHA-256 from
# https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/riscv64/latest-releases.yaml.
# Unpacks into tests/sysroot/alpine (not committed); tests that need it skip without it.
# With `python`, also installs python3 and its dependencies from Alpine's main repository,
# at the versions and SHA-256 sums in python.lock (resolved from v3.24's APKINDEX).
#
#   tests/sysroot/build.sh [python]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
version=3.24.2
tarball=alpine-minirootfs-$version-riscv64.tar.gz
sha256=57132e6e4f3a4ba9ffdf24e485513ce507e45471e7fd565aeed91c055cd63f7b
mkdir -p "$here/.cache"
if [ ! -f "$here/.cache/$tarball" ]; then
    curl -fsSL "https://dl-cdn.alpinelinux.org/alpine/v${version%.*}/releases/riscv64/$tarball" \
        -o "$here/.cache/$tarball.part"
    mv "$here/.cache/$tarball.part" "$here/.cache/$tarball"
fi
echo "$sha256  $here/.cache/$tarball" | shasum -a 256 -c -
rm -rf "$here/alpine"
mkdir -p "$here/alpine"
# Device nodes cannot be created without privileges and are not needed.
tar -xzf "$here/.cache/$tarball" -C "$here/alpine" --exclude='./dev/*' 2>/dev/null || true
test -e "$here/alpine/lib/ld-musl-riscv64.so.1"

if [ "${1:-}" = python ]; then
    repo=https://dl-cdn.alpinelinux.org/alpine/v${version%.*}/main/riscv64
    mkdir -p "$here/.cache/apk"
    while read -r name ver sum; do
        apk="$here/.cache/apk/$name-$ver.apk"
        [ -f "$apk" ] || curl -fsSL "$repo/$name-$ver.apk" -o "$apk"
        echo "$sum  $apk" | shasum -a 256 -c - >/dev/null
        # An .apk is concatenated gzip streams: signature, metadata, then the files.
        tar -xzf "$apk" -C "$here/alpine" --exclude='.PKGINFO' --exclude='.SIGN.*' \
            --exclude='.pre-*' --exclude='.post-*' --exclude='.trigger' 2>/dev/null
    done < "$here/python.lock"
    test -e "$here/alpine/usr/bin/python3"
fi
echo "sysroot ready: $here/alpine"
