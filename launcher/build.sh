#!/bin/sh
# Build rvcontainer, sign it for Virtualization.framework, and put the Linux build of rvrun
# it shares into its VMs next to it: .build/release/{rvcontainer,rvrun-linux}.
set -eu
cd "$(dirname "$0")"
../tools/linux.sh build --release -p rvrun
swift build -c release
bin=.build/release
codesign --force --sign - --entitlements rvcontainer.entitlements "$bin/rvcontainer"
cp ../target/linux/release/rvrun "$bin/rvrun-linux"
echo "$PWD/$bin/rvcontainer"
