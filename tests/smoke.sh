#!/usr/bin/env bash
# End-to-end check of the generated package without a model: mounts build/submission into a Python container
# and plays one task (tests/in_sandbox.sh). Usage: bash tests/smoke.sh      (needs Docker; pip needs network)
set -eu
cd "$(dirname "$0")/.."
[ -f build/submission/skills/swe-coder/assets/swe_coder.txt ] || { echo "run tools/generate.py first" >&2; exit 2; }
HERE=$(pwd -W 2>/dev/null || pwd)   # Windows path under Git Bash, POSIX path elsewhere
MSYS_NO_PATHCONV=1 docker run --rm --cpus 2 --memory 4g \
  -v "$HERE/build/submission:/submission:ro" -v "$HERE/tests:/tests:ro" \
  python:3.12 bash /tests/in_sandbox.sh
