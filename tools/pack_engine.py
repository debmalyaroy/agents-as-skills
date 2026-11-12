#!/usr/bin/env python3
"""Pack a static `swe-coder` binary into the text asset the skill ships (LZMA + base85, 120-character lines).

    python tools/pack_engine.py <path/to/swe-coder> [build/swe_coder.txt]

`skill/swe-coder/scripts/_boot.py` reverses this in the sandbox: base85-decode, LZMA-decompress, chmod +x.
"""

from __future__ import annotations

import base64
import lzma
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "build" / "swe_coder.txt"
    raw = src.read_bytes()
    if raw[:4] != b"\x7fELF":
        print(f"{src} is not an ELF binary: the sandbox runs x86_64 Linux", file=sys.stderr)
        return 1
    b85 = base64.b85encode(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(b85[i : i + 120] for i in range(0, len(b85), 120)) + "\n").encode("ascii"))
    print(f"packed {len(raw)} bytes -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
