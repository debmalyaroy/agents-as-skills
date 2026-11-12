"""Entry point: unpacks `/tmp/coder` if needed and executes `/tmp/coder verify`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _boot

if __name__ == "__main__":
    raise SystemExit(_boot.exec_subcommand("verify", sys.argv[1:]))
