#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/data/install_lean.sh"
  echo "Installs elan and the pinned Lean 4 toolchain into \$ELAN_HOME (default: ~/.elan) on Linux x86_64;"
  echo "downloads are cached under \$RCI_CACHE and checked against pinned sha256 sums."
  exit 2
}

[ $# -eq 0 ] || usage
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || { echo "install_lean.sh supports Linux x86_64 only (found $(uname -s) $(uname -m))"; exit 2; }
command -v sha256sum > /dev/null || { echo "sha256sum not found"; exit 2; }

ELAN_HOME=${ELAN_HOME:-$HOME/.elan}
DL=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}/downloads
mkdir -p "$DL" "$ELAN_HOME"
export ELAN_HOME

ELAN_URL=https://github.com/leanprover/elan/releases/download/v4.2.4/elan-x86_64-unknown-linux-gnu.tar.gz
ELAN_SHA=42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63
LEAN_URL=https://github.com/leanprover/lean4/releases/download/v4.28.0/lean-4.28.0-linux.tar.zst
LEAN_SHA=ceb3a3f844f7aebf63245e2b51c28d5b0ed38942c19f93cf3febd520302160bd

fetch() {
  local url=$1 dest=$2 sha=$3
  if [ ! -f "$dest" ] || [ "$(sha256sum "$dest" | cut -d' ' -f1)" != "$sha" ]; then
    curl -fL --retry 5 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
  [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ] || { echo "sha256 mismatch for $url"; exit 1; }
}

fetch "$ELAN_URL" "$DL/elan-v4.2.4.tar.gz" "$ELAN_SHA"
fetch "$LEAN_URL" "$DL/lean-4.28.0-linux.tar.zst" "$LEAN_SHA"

if [ ! -x "$ELAN_HOME/bin/elan" ]; then
  T=$(mktemp -d)
  tar -xzf "$DL/elan-v4.2.4.tar.gz" -C "$T"
  "$T/elan-init" -y --no-modify-path --default-toolchain none
  rm -rf "$T"
fi

TC="$ELAN_HOME/toolchains/leanprover--lean4---v4.28.0"
if [ ! -x "$TC/bin/lean" ]; then
  T=$(mktemp -d -p "$ELAN_HOME")
  tar --zstd -xf "$DL/lean-4.28.0-linux.tar.zst" -C "$T"
  mkdir -p "$ELAN_HOME/toolchains"
  mv "$T/lean-4.28.0-linux" "$TC"
  rmdir "$T"
fi
export PATH="$ELAN_HOME/bin:$PATH"
elan default leanprover/lean4:v4.28.0
lean --version
lake --version
echo "add $ELAN_HOME/bin to PATH (export ELAN_HOME=$ELAN_HOME) before running prepare_fate.sh"
