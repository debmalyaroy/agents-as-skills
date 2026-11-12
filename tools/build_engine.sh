#!/usr/bin/env bash
# Build the static x86_64-musl `swe-coder` binary in Docker and pack it into build/swe_coder.txt.
# Usage: bash tools/build_engine.sh        (needs Docker and Python 3)
set -eu
cd "$(dirname "$0")/.."
IMAGE=agents-as-skills-engine
# the first interpreter that really runs (on Windows `python3` can be a Store stub)
PY=python3; "$PY" -c "" >/dev/null 2>&1 || PY=python
mkdir -p build
docker build -f tools/Dockerfile.engine ${JOBS:+--build-arg CARGO_BUILD_JOBS=$JOBS} -t "$IMAGE" .
cid=$(docker create "$IMAGE" /bin/true)
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
docker cp "$cid:/build/target/x86_64-unknown-linux-musl/release/swe-coder" build/swe-coder
"$PY" tools/pack_engine.py build/swe-coder build/swe_coder.txt
