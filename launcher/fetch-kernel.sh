#!/bin/sh
# Install the Linux kernel rvcontainer boots its VMs with: Kata Containers' container kernel
# (the one Containerization recommends), into ~/Library/Application Support/champollion.
# The release archive is large; it is downloaded once into ~/Library/Caches/champollion.
set -eu
version=3.17.0
sha256=647c7612e6edf789d5e14698c48c99d8bac15ad139ffaa1c8bb7d229f748d181
archive="$HOME/Library/Caches/champollion/kata-static-$version-arm64.tar.xz"
dest="$HOME/Library/Application Support/champollion"
mkdir -p "$(dirname "$archive")" "$dest"
if [ ! -f "$archive" ]; then
    curl -fSL -o "$archive.part" \
        "https://github.com/kata-containers/kata-containers/releases/download/$version/kata-static-$version-arm64.tar.xz"
    mv "$archive.part" "$archive"
fi
echo "$sha256  $archive" | shasum -a 256 -c -
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
dir=./opt/kata/share/kata-containers
target=$(tar -xJf "$archive" -C "$tmp" "$dir/vmlinux.container" && readlink "$tmp/$dir/vmlinux.container")
tar -xJf "$archive" -C "$tmp" "$dir/$target"
cp "$tmp/$dir/$target" "$dest/vmlinux"
echo "installed $target as $dest/vmlinux"
