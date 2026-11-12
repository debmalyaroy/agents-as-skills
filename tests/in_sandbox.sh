#!/usr/bin/env bash
# Runs inside a Linux container that looks like the task sandbox: a git repository in /workspace, the generated
# skill mounted at /submission. It plays the calls a model would make and checks every reply.
# Each engine call goes through `bash -c "<one string>"`, the way the harness tool `run_command` runs it.
set -u
fail() { echo "SMOKE FAIL: $1"; exit 1; }
call() { # one model turn: print the command and the reply, keep the reply in $OUT
  echo; echo "\$ $1" | head -1
  OUT=$(cd /workspace && bash -c "$1" 2>&1)
  echo "$OUT"
}
expect() { echo "$OUT" | grep -qF -- "$1" || fail "expected \"$1\" in the reply above"; }

python3 -m pip install -q pytest >/dev/null 2>&1 || fail "pip install pytest"
git config --global user.email smoke@example.com && git config --global user.name smoke
git config --global --add safe.directory /workspace

# --- a small repository with one bug: Cart.total ignores the quantity -------------------------------------
rm -rf /workspace && mkdir -p /workspace/shop /workspace/tests && cd /workspace
: > shop/__init__.py
cat > shop/cart.py <<'PY'
"""A tiny shopping cart."""


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, quantity=1):
        self.items.append((name, price, quantity))

    def count(self):
        return sum(quantity for _, _, quantity in self.items)

    def total(self):
        result = 0
        for name, price, quantity in self.items:
            result += price
        return result
PY
cat > shop/report.py <<'PY'
from shop.cart import Cart


def summary(cart: Cart) -> str:
    return f"{cart.count()} items, total {cart.total()}"
PY
cat > tests/test_cart.py <<'PY'
from shop.cart import Cart


def test_single_item():
    cart = Cart()
    cart.add("pen", 2)
    assert cart.total() == 2


def test_count():
    cart = Cart()
    cart.add("pen", 2, quantity=3)
    assert cart.count() == 3
PY
git init -q && git add -A && git commit -q -m "initial" || fail "git init"

# --- call 1: the skill's boot script installs /tmp/coder (ADK: run_skill_script scripts/boot.py, args={}) ----
echo "\$ python3 /submission/skills/swe-coder/scripts/boot.py"
OUT=$(cd /workspace && python3 /submission/skills/swe-coder/scripts/boot.py 2>&1); echo "$OUT"
expect "swe-coder is ready"
[ -x /tmp/coder ] || fail "/tmp/coder was not installed"

# --- locate ---------------------------------------------------------------------------------------------
call "/tmp/coder debug query='Cart.total ignores quantity'"
expect "TOP TARGETS"
expect "shop/cart.py"
expect "Cart.total"

# --- read -----------------------------------------------------------------------------------------------
call "/tmp/coder view file=shop/cart.py symbol=Cart.total"
expect "result += price"

# --- verify before any edit: nothing to grade ---------------------------------------------------------------
call "/tmp/coder verify <<'EOF'
from shop.cart import Cart
c = Cart()
c.add('pen', 2, 3)
assert c.total() == 6, c.total()
EOF"
expect "EMPTY DIFF"

# --- a broken edit is refused and the file stays intact -------------------------------------------------------
call "/tmp/coder patch file=shop/cart.py symbol=Cart.total <<'EOF'
<<<OLD
            result += price
===
            result += price * (quantity
>>>
EOF"
python3 -c "import ast,sys; ast.parse(open('/workspace/shop/cart.py').read())" || fail "a broken edit reached the workspace"
git -C /workspace diff --quiet || fail "the refused edit changed the workspace"

# --- the fix ------------------------------------------------------------------------------------------------
call "/tmp/coder patch file=shop/cart.py symbol=Cart.total <<'EOF'
<<<OLD
            result += price
===
            result += price * quantity
>>>
EOF"
expect "PATCH APPLIED"

# --- verify: reproducer + paired tests + static critic ----------------------------------------------------------
call "/tmp/coder verify <<'EOF'
from shop.cart import Cart
c = Cart()
c.add('pen', 2, 3)
assert c.total() == 6, c.total()
EOF"
expect "REPRODUCER: PASS"
expect "VERDICT: READY"

# --- what the harness would grade ---------------------------------------------------------------------------
echo; echo "\$ git diff HEAD --stat"
git -C /workspace diff HEAD --stat
[ "$(git -C /workspace diff HEAD --name-only)" = "shop/cart.py" ] || fail "the diff must touch shop/cart.py only"
git -C /workspace diff HEAD | grep -q '^+            result += price \* quantity$' || fail "the fix is not in the diff"
[ -s /tmp/.agent/best.patch ] || fail "best.patch was not saved"
python3 -m pytest -q /workspace/tests >/dev/null 2>&1 || fail "the repository tests fail after the patch"
echo; echo "SMOKE OK: boot, debug, view, patch (refused + applied), verify READY, clean one-file diff"
