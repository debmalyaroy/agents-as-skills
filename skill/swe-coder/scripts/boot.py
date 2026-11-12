"""Flat call shape entrypoint: installs /tmp/coder once (call it with an empty args object) and prints
how to reach every engine command through run_command with one flat string."""

from __future__ import annotations

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _boot

if __name__ == "__main__":
    if platform.system() == "Linux":
        print(_boot.install_flat_launcher())
    else:
        print(_boot.flat_usage())
