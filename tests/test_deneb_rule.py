"""tests/test_deneb_rule.py — the FORMAL Deneb-Rule assertion (QA-03).

THE DENEB RULE (LEVELUP-PLAN.md, absolute): Deneb TELLS the exact command + what it does,
and NEVER executes a setup command on its own. This file PROVES the rule holds across ALL
three v1 tell-only paths - `deneb setup`, `deneb recommend`, `deneb profile` - two ways:

  1. SOURCE PURITY - the pure advisory (deneb/setup_advisor.py) and the three CLI tell-only
     functions (cmd_setup / cmd_recommend / cmd_profile) reference NO write/exec primitive
     (no subprocess / os.system / os.popen / exec / eval, no tools executor, no engine).
  2. BEHAVIORAL SPY (the strong proof) - with every WRITE/EXEC primitive (tools.run_write,
     tools.execute, tools.execute_write, tools.write_file, os.system) monkeypatched to FAIL
     LOUD, driving setup_steps + cmd_setup + cmd_recommend + cmd_profile fires NONE of them,
     while cmd_setup still PRINTS the advice (the run command + the 127.0.0.1:8001 warning).

The read-only tools.run / tools.read_file boundary is left INTACT - profile_hardware reads
the box (diagnosis, allowed); the rule forbids executing the SETUP steps, not reading. A
positive control (test_tripwire_actually_fires) proves the spy is not a silent no-op.

Run: python3 tests/test_deneb_rule.py    (also collectable by pytest)
"""
from __future__ import annotations

import inspect
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import cli, setup_advisor, tools  # noqa: E402
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from deneb.setup_advisor import Step  # noqa: E402


# ── a constructed this-box rocm profile (mirrors tests/test_setup_advisor.py) ──
def _rocm_profile(usable=107000):
    return HardwareProfile(
        primary_backend="rocm", usable_mem_mb=usable, os="Linux", arch="x86_64",
        gpus=[GPUInfo(vendor="amd", backend="rocm", memory_kind="unified",
                      unified_mem_mb=126908, extra={"gfx": "gfx1151", "sku": "STRXLGEN"})])


# ── the fail-loud tripwire over the WRITE/EXEC surface ─────────────────────────
class _DenebRuleViolation(AssertionError):
    """Raised the instant a tell-only path touches a write/exec primitive."""


def _boom(*_a, **_k):
    raise _DenebRuleViolation("Deneb Rule violated - a tell-only path executed a command")


def _install_tripwires():
    """Patch every WRITE/EXEC primitive to fail loud; return a restore() thunk. Leaves the
    READ-ONLY tools.run / tools.read_file intact so profile_hardware can still read the box.
    (os.system, not platform.system, is the exec surface - the latter is a pure name query.)"""
    saved = []
    for obj, attr in ((tools, "run_write"), (tools, "execute"),
                      (tools, "execute_write"), (tools, "write_file"), (os, "system")):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, _boom)

    def restore():
        for obj, attr, val in saved:
            setattr(obj, attr, val)
    return restore


# ── 1. SOURCE PURITY (setup_advisor is pure) ───────────────────────────────────
_SETUP_ADVISOR_FORBIDDEN = ("subprocess", "os.system", "os.popen", "Popen",
                            "tools.run", "tools.execute", "tools.write",
                            "run_write", " exec(", " eval(")


def test_setup_advisor_source_is_pure():
    src = Path(setup_advisor.__file__).read_text()
    hits = [t for t in _SETUP_ADVISOR_FORBIDDEN if t in src]
    assert not hits, f"setup_advisor must be pure (struct in, Step out); found: {hits}"
    import_lines = [ln for ln in src.splitlines()
                    if ln.strip().startswith(("import ", "from "))]
    assert not any("tools" in ln for ln in import_lines), \
        "setup_advisor must not import the tools (executor) module"


# ── 2. SOURCE PURITY (the three CLI tell-only fns) ─────────────────────────────
# cmd_* may CALL profile_hardware (which internally uses the read-only tools.run boundary -
# diagnosis, allowed), but the fn bodies themselves must reference no executor primitive.
_CLI_FORBIDDEN = ("tools.", "subprocess", "os.system", "os.popen",
                  "client.", "loop.run", "run_write", "execute")


def test_cli_tellonly_fns_reference_no_executor():
    for fn in (cli.cmd_setup, cli.cmd_recommend, cli.cmd_profile):
        src = inspect.getsource(fn)
        hits = [t for t in _CLI_FORBIDDEN if t in src]
        assert not hits, f"{fn.__name__} references an executor token: {hits}"


# ── 3. BEHAVIORAL SPY - the strong no-execution proof ──────────────────────────
def test_setup_steps_fires_no_executor():
    restore = _install_tripwires()
    try:
        m = setup_advisor.resolve_model("Qwen3.6-35B-A3B")
        assert m is not None, "catalog top pick must resolve"
        steps = setup_advisor.setup_steps(m, _rocm_profile())   # must fire no tripwire
        assert isinstance(steps, list) and steps, "must return a non-empty Step list"
        assert all(isinstance(s, Step) for s in steps)
    finally:
        restore()


def test_cmd_setup_prints_only_and_fires_no_executor():
    restore = _install_tripwires()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli.cmd_setup(["Qwen3.6-35B-A3B"])             # must fire no tripwire
    finally:
        restore()
    assert isinstance(rc, int) and rc == 0
    out = buf.getvalue()
    assert out.strip(), "cmd_setup must print the advice"
    assert "llama-server" in out, "the run command must be shown as TEXT (advice, not run)"
    assert "127.0.0.1:8001" in out, "the run-step service warning (127.0.0.1:8001) must be shown"


def test_cmd_recommend_fires_no_executor():
    restore = _install_tripwires()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli.cmd_recommend(["--use", "coding"])         # must fire no tripwire
    finally:
        restore()
    assert rc == 0 and buf.getvalue().strip()


def test_cmd_profile_fires_no_executor():
    restore = _install_tripwires()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli.cmd_profile([])                            # must fire no tripwire
    finally:
        restore()
    assert rc == 0 and buf.getvalue().strip()


# ── 4. positive control - the tripwire is real (not a silent no-op) ────────────
def test_tripwire_actually_fires_when_executor_called():
    restore = _install_tripwires()
    try:
        fired = False
        try:
            tools.run_write("mkdir -p ~/deneb-rule-should-never-run")
        except AssertionError:
            fired = True
        assert fired, "the fail-loud tripwire MUST fire when a write primitive is called - " \
                      "otherwise the no-execution tests above prove nothing"
    finally:
        restore()
    # and the real surface is fully restored afterwards
    assert tools.run_write is not _boom, "tripwire must be restored after the test"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
