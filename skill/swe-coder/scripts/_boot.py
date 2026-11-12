"""Bootstrap and dispatcher for the native `swe-coder` (`coder`) engine.

The first call (`run_skill_script`) finds `scripts/` and `assets/swe_coder.txt` materialized in the sandbox
temp directory. `ensure_coder_binary()` decodes and unpacks the static `x86_64-unknown-linux-musl` binary
there (`coder`), so that later `run_command` calls execute it directly.
"""

from __future__ import annotations

import base64
import lzma
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _root_subdir(name: str) -> Path:
    return Path(os.path.abspath(os.sep)) / name


def _find_workspace() -> str:
    explicit = os.environ.get("SWE_WS")
    if explicit and Path(explicit).is_dir():
        return str(Path(explicit).resolve())
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / ".git").is_dir():
            return str(Path(entry).resolve())
    root_ws = _root_subdir("workspace")
    if root_ws.is_dir():
        return str(root_ws)
    probe = Path(os.environ.get("TMPDIR") or os.getcwd()).resolve()
    for _ in range(4):
        cand = probe / "workspace"
        if (cand / ".git").is_dir() or cand.is_dir():
            return str(cand)
        if probe.parent == probe:
            break
        probe = probe.parent
    return str(Path.cwd())


def _find_tmp() -> str:
    for var in ("SWE_TMP", "TMPDIR", "TEMP", "TMP"):
        t = os.environ.get(var)
        if t and Path(t).is_dir():
            return str(Path(t).resolve())
    root_tmp = _root_subdir("tmp")
    if root_tmp.is_dir():
        return str(root_tmp)
    return str(Path(tempfile.gettempdir()).resolve())


WS = _find_workspace()
TMP = _find_tmp()


def ensure_coder_binary() -> str:
    """Ensure the static `coder` binary is unpacked and executable; return binary path."""
    is_linux_x86 = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
    tmp_path = Path(TMP)
    target_bin = tmp_path / "coder"
    root_tmp = _root_subdir("tmp")
    root_coder = root_tmp / "coder"

    if is_linux_x86:
        # flat call shape (scripts/boot.py): the binary lives at /tmp/.coder.bin and /tmp/coder is a wrapper
        flat_bin = tmp_path / ".coder.bin"
        if flat_bin.is_file() and os.access(flat_bin, os.X_OK) and flat_bin.stat().st_size > 100_000:
            return str(flat_bin)
        if target_bin.is_file() and os.access(target_bin, os.X_OK) and target_bin.stat().st_size > 100_000:
            if tmp_path != root_tmp and root_tmp.is_dir() and not root_coder.is_file():
                try:
                    shutil.copyfile(target_bin, root_coder)
                    os.chmod(root_coder, 0o755)
                except OSError:
                    pass
            return str(target_bin)

        here = Path(__file__).resolve().parent
        asset_path = here.parent / "assets" / "swe_coder.txt"
        if asset_path.is_file():
            b85_data = "".join(asset_path.read_text(encoding="utf-8").split())
            raw_bytes = lzma.decompress(base64.b85decode(b85_data))
            tmp_file = tmp_path / f"coder.tmp.{os.getpid()}"
            tmp_file.write_bytes(raw_bytes)
            os.chmod(tmp_file, 0o755)
            os.replace(tmp_file, target_bin)
            if tmp_path != root_tmp and root_tmp.is_dir():
                try:
                    tmp_slash = root_tmp / f"coder.tmp.{os.getpid()}"
                    tmp_slash.write_bytes(raw_bytes)
                    os.chmod(tmp_slash, 0o755)
                    os.replace(tmp_slash, root_coder)
                except OSError:
                    pass
            return str(target_bin)

    # Non-Linux host fallback (local test runs): locate a host cargo binary
    exe_name = "swe-coder.exe" if os.name == "nt" else "swe-coder"
    search_roots = [Path(__file__).resolve().parent, Path.cwd(), Path(WS)]
    for root in search_roots:
        cur = root
        for _ in range(6):
            for prof in ("release", "debug"):
                cand = cur / "target" / prof / exe_name
                if cand.is_file():
                    return str(cand)
            if cur.parent == cur:
                break
            cur = cur.parent
    return str(target_bin)


def _is_own_script(val: str) -> bool:
    v = val.strip().replace("\\", "/").lstrip("./")
    return v.startswith("scripts/") and v.count("/") == 1 and (Path(__file__).parent / v[len("scripts/") :]).is_file()


def normalize_argv(argv: list[str]) -> tuple[list[str], str]:
    """Accept both call shapes.

    `run_command("/tmp/coder patch file=a.py symbol=f <<'EOF' ...")` reaches the binary directly; this path
    serves `run_skill_script(file_path="scripts/patch.py", args={"file": "a.py", "symbol": "f", "body": "..."})`,
    which ADK turns into `--file a.py --symbol f --body ...`. `--k v` becomes `k=v`; `--body v` (alias
    `--code`, `--new`, `--stdin`) becomes the stdin body, so multi-line text can also travel as a plain JSON
    argument.
    """
    out: list[str] = []
    body = ""
    saw_body_key = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            out.extend(argv[i + 1 :])
            break
        if tok.startswith("--") and len(tok) > 2:
            key = tok[2:].lower().replace("-", "_")
            val = argv[i + 1] if i + 1 < len(argv) else ""
            if key in (
                "body",
                "code",
                "new",
                "new_string",
                "new_str",
                "stdin",
                "replacement",
                "replace",
                "content",
                "patch",
                "diff",
            ):
                saw_body_key = True
                body = val.replace("\r\n", "\n")
            elif key in ("old", "old_string", "old_str", "snippet", "search", "target", "original"):
                if val not in ("None", "null"):
                    out.append(f"old={val.replace(chr(13) + chr(10), chr(10))}")
            elif key in ("file", "path", "file_path", "filepath", "filename"):
                # `file_path`/`skill_name` copied into `args` from the call's top level name this skill's
                # own script, not the repo file: dropping them keeps the real `file`
                if val not in ("None", "null", "") and not (key == "file_path" and _is_own_script(val)):
                    out.append(f"file={val}")
            elif key == "skill_name":
                pass
            elif key in ("symbol", "name", "func", "function", "method", "class"):
                if val not in ("None", "null", ""):
                    out.append(f"symbol={val}")
            elif key in ("start", "start_line"):
                if val not in ("None", "null", ""):
                    out.append(f"start={val}")
            elif key in ("end", "end_line"):
                if val not in ("None", "null", ""):
                    out.append(f"end={val}")
            elif key in ("query", "q") and i + 1 < len(argv):
                if val not in ("None", "null", ""):
                    out.append(f"query={val}")
            elif key in ("repro", "reproducer") and i + 1 < len(argv):
                if val not in ("None", "null", ""):
                    out.append(f"repro={val.replace(chr(13) + chr(10), chr(10))}")
            elif val not in ("None", "null"):
                out.append(f"{key}={val}")
            i += 2
            continue
        out.append(tok)
        i += 1
    if saw_body_key and body == "" and any(x.startswith("old=") and len(x) > 4 for x in out):
        out.append("delete=1")
    return out, body


# The harness compacts the conversation at COMPACTION_AT prompt tokens into a model-written summary
# that can drop the issue text. Every call's arguments and output are therefore counted here, and
# outputs shrink to fit SOFT_LIMIT. CHARS_PER_TOKEN converts raw text to tokens; the per-call
# overhead covers what never reaches this script (rejected calls, the model's own text).
COMPACTION_AT = 14336
CHARS_PER_TOKEN = 2.8
INITIAL_MEASURED = 3600


def ledger_fit(out: str, arg_chars: int) -> str:
    try:
        import _budget

        soft, initial, overhead = _budget.SOFT_LIMIT, _budget.INITIAL_TOKENS, _budget.CALL_OVERHEAD_TOKENS
    except (ImportError, AttributeError):
        soft, initial, overhead = 11500, INITIAL_MEASURED, 200
    initial = min(initial, INITIAL_MEASURED)
    path = Path(TMP) / ".agent" / "ctx_tokens"
    try:
        used = int(path.read_text().strip() or initial)
    except (OSError, ValueError):
        used = initial
    used += overhead + int(arg_chars / CHARS_PER_TOKEN)
    room = soft - used
    need = int(len(out) / CHARS_PER_TOKEN)
    if need > room:
        keep = max(300, room) * int(CHARS_PER_TOKEN)
        if keep < len(out):
            head = out[: keep * 2 // 3]
            tail = out[-(keep // 3) :]
            out = f"{head}\n... [{len(out) - keep} chars clipped: context nearly full] ...\n{tail}"
    used += int(len(out) / CHARS_PER_TOKEN)
    # warn only once the soft limit is reached, not before
    if used >= soft:
        out += (
            f"\nCONTEXT: ~{used} of {COMPACTION_AT} tokens used. At {COMPACTION_AT} this conversation is"
            f" wiped (issue text included). Make your one final edit now, then {_finish()}.\n"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(used))
    except OSError:
        pass
    return out


def _has_submit() -> bool:
    """False when the version has no submit_patch tool (the generator writes HAS_SUBMIT = False then)."""
    try:
        import _budget

        return getattr(_budget, "HAS_SUBMIT", True)
    except ImportError:
        return True


def _finish() -> str:
    return "call `submit_patch`" if _has_submit() else "stop (the workspace is graded)"


FLAT_USAGE_BODY = """
  /tmp/coder debug query='<exact identifiers and error text, one line>'
  /tmp/coder view file=pkg/mod.py symbol=Class.method
  /tmp/coder patch file=pkg/mod.py symbol=Class.method <<'EOF'
  <<<OLD
      old_line()
  ===
      new_line()
  >>>
  EOF
  /tmp/coder verify <<'EOF'
  from pkg.mod import fn
  assert fn(1) == 2
  EOF
Always pass code (patch blocks, repros) in a heredoc, never on the command line: quotes inside code break the shell."""


def flat_usage() -> str:
    tail = ", or submit_patch:" if _has_submit() else ":"
    return "swe-coder is ready. Every further call is run_command with ONE string" + tail + FLAT_USAGE_BODY


def install_flat_launcher() -> str:
    """Flat call shape: move the binary to /tmp/.coder.bin and make /tmp/coder a wrapper that sets the
    workspace, tmp and budget environment `exec_subcommand` would set, so `run_command("/tmp/coder ...")` behaves
    exactly like a skill call."""
    raw = Path(ensure_coder_binary())
    tmp_path = Path(TMP)
    flat_bin = tmp_path / ".coder.bin"
    if raw != flat_bin:
        shutil.copyfile(raw, flat_bin)
        os.chmod(flat_bin, 0o755)
    try:
        import _budget

        budget = f"export SWE_MAX_TIME_S={_budget.MAX_TIME_S} SWE_MAX_TOOL_CALLS={_budget.MAX_TOOL_CALLS}\n"
        if getattr(_budget, "ATTEMPTS", 1) != 1:
            budget += f"export SWE_ATTEMPTS={_budget.ATTEMPTS}\n"
    except (ImportError, AttributeError):
        budget = ""
    wrapper = (
        "#!/bin/sh\n"
        f"export SWE_WS='{WS}' SWE_TMP='{TMP}' SWE_CALL_SHAPE=cmd\n"
        f"{budget}"
        f"{'' if _has_submit() else 'export SWE_HAS_SUBMIT=0' + chr(10)}"
        f"cd '{WS}' 2>/dev/null\n"
        f"exec '{flat_bin}' \"$@\"\n"
    )
    for target in {tmp_path / "coder", _root_subdir("tmp") / "coder"}:
        try:
            tmp_w = target.with_name(f"coder.w.{os.getpid()}")
            tmp_w.write_text(wrapper, encoding="utf-8")
            os.chmod(tmp_w, 0o755)
            os.replace(tmp_w, target)
        except OSError:
            pass
    return flat_usage()


def exec_subcommand(subcmd: str, argv: list[str]) -> int:
    """Invoke `swe-coder <subcmd> <argv>` and print LF-normalized combined output to stdout."""
    coder_bin = ensure_coder_binary()
    env = os.environ.copy()
    env["SWE_WS"] = WS
    env["SWE_TMP"] = TMP
    # pass the session budget to the engine, whose clocks scale with it
    try:
        import _budget

        env.setdefault("SWE_MAX_TIME_S", str(_budget.MAX_TIME_S))
        env.setdefault("SWE_MAX_TOOL_CALLS", str(_budget.MAX_TOOL_CALLS))
    except (ImportError, AttributeError):
        pass

    argv, stdin_data = normalize_argv(list(argv))
    if not stdin_data and not sys.stdin.isatty() and os.name != "nt":
        try:
            import select

            rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
            if rlist:
                stdin_data = sys.stdin.read().replace("\r\n", "\n")
        except (OSError, ValueError):
            stdin_data = ""

    proc = subprocess.run(
        [coder_bin, subcmd, *argv],
        input=stdin_data,
        text=True,
        capture_output=True,
        cwd=WS if Path(WS).is_dir() else None,
        env=env,
        timeout=180,
        check=False,
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).replace("\r\n", "\n")
    out = ledger_fit(out, sum(len(a) for a in argv) + len(stdin_data))
    sys.stdout.write(out)
    if out and not out.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return 0
