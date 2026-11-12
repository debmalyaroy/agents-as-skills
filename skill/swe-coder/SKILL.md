---
name: swe-coder
description: First call scripts/boot.py (args={}) to install /tmp/coder for run_command use. Native Rust AST/Graph/Embedding localization (`scripts/debug.py`), symbol-scoped atomic code editor (`scripts/patch.py`), symbol/outline viewer (`scripts/view.py`), parallel repro+pytest+critic verifier (`scripts/verify.py`), and sandboxed probe (`scripts/probe.py`).
---
<!--skill-->
Every `run_skill_script` call has exactly 3 top-level keys, in this order: `"args": {...}`, then `"file_path": "scripts/<name>.py"`, then `"skill_name": "swe-coder"` (after `args` closes, the call is not finished) — ALL script parameters (`"file"`, `"symbol"`, `"old_string"`, `"new_string"`, `"body"`, `"repro"`, `"query"`) MUST be nested inside `"args": {...}` (top-level keys outside `"args"` are ignored!):

1. `"scripts/debug.py"` — `{"args": {"query": "<Class.method, specific identifiers, params, error text>", "repro": "<optional python snippet>"}, "file_path": "scripts/debug.py", "skill_name": "swe-coder"}`:
   Runs any issue reproducer, builds the AST + Call/Inheritance Graph + Whitened 256-d Structural Embeddings, ranks primary functions/classes (with method outlines for large classes) and `ALSO-CHANGE` counterparts, and preloads source slices.
2. `"scripts/patch.py"` — `{"args": {"file": "pkg/mod.py", "symbol": "Class.method", "old_string": "<2-8 exact lines>", "new_string": "<replacement lines>"}, "file_path": "scripts/patch.py", "skill_name": "swe-coder"}`:
   Symbol-scoped atomic code editor with CPython `ast.parse` syntax/indentation gate, accidental-deletion + docstring-splice protection, new method insertion (`"symbol": "Class.new_method"`, `"body": "def new_method(self, ...):\n    ..."`), new function insertion (`"after_symbol": "fn"`, `"body": "..."`), and new module creation (`"create": "1"`, `"body": "..."`).
3. `"scripts/view.py"` — `{"args": {"file": "pkg/mod.py", "symbol": "Class.method"}, "file_path": "scripts/view.py", "skill_name": "swe-coder"}` (or `"symbol": "imports"` for lines 1-65, `"symbol": "*"` for file outline, or `"start": 1, "end": 60`):
   Reads a Python symbol (with automatic cross-file AST symbol lookup), imports/header, pattern, or file outline (`<= 1250` tokens).
4. `"scripts/verify.py"` — `{"args": {"repro": "import pkg\ngot = ...\nassert got == expected, got"}, "file_path": "scripts/verify.py", "skill_name": "swe-coder"}`:
   Saves and runs your inline issue reproducer alongside graph-paired `pytest` and Static Critic in parallel while enforcing `/workspace` hygiene and backing up `best.patch`.
5. `"scripts/probe.py"` — `{"args": {"body": "import pkg\nprint(pkg.fn())"}, "file_path": "scripts/probe.py", "skill_name": "swe-coder"}`:
   Executes a 15-second sandboxed Python check in `/tmp/.agent/probe` (`PYTHONPATH=/workspace`).
<!--/skill-->

<!--cmd-->
Flat call shape: call `run_skill_script` ONCE with file_path `scripts/boot.py` and an EMPTY args object; it installs
`/tmp/coder`. After that every engine call is `run_command` with ONE string:
- `/tmp/coder debug query='<exact identifiers and error text, one line>'` — localization (AST + call graph + embeddings), ALSO-CHANGE counterparts, preloaded code.
- `/tmp/coder view file=pkg/mod.py symbol=Class.method` (or `symbol=imports`, `symbol=*`, `start=1 end=60`).
- `/tmp/coder patch file=pkg/mod.py symbol=Class.method <<'EOF'` then `<<<OLD`, 2-8 exact current lines, `===`, the new lines, `>>>`, `EOF`; new function: the `def ...` body as the heredoc; one-line form: `old='...' new='...'` with `
` for line breaks.
- `/tmp/coder verify <<'EOF'` then a repro ending in `assert <expected>`, `EOF` (one-line form: `repro='...'`): repro + paired pytest + static critic.
- `/tmp/coder probe <<'EOF'` then a short Python check, `EOF`.
<!--/cmd-->
