"""The APPLY loop — gating behaviour (deneb.loop).

Mocks the engine so the test is deterministic: it asserts that an apply action (fix /
write_file) is GATED (confirm is asked), executed only on approval, and that a declined
gate never touches execute_write and records a 'skipped' observation the model can see.

Run: python3 tests/test_apply_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import client, loop, tools  # noqa: E402


def _seq(*results):
    it = iter(results)
    return lambda hist, timeout=180: next(it)


def _with_mocks(agent_steps, confirm):
    """Run loop.run with a mocked engine + execute_write; return (result, applied, hist)."""
    real_step, real_exec = client.agent_step, tools.execute_write
    applied = []
    try:
        client.agent_step = _seq(*agent_steps)
        tools.execute_write = lambda action, cmd="", path="", content="": (
            applied.append((action, cmd or path)) or {"ok": True, "output": f"applied {action}"})
        captured = {}
        orig_record = loop._record
        loop._record = lambda hist, res, action, output: (captured.setdefault("obs", []).append(output),
                                                           orig_record(hist, res, action, output))[1]
        result = loop.run("please fix it", confirm=confirm)
        return result, applied, captured.get("obs", [])
    finally:
        client.agent_step, tools.execute_write = real_step, real_exec
        loop._record = orig_record


def test_apply_approved_runs_execute_write():
    result, applied, obs = _with_mocks(
        [{"type": "action", "action": "write_file", "path": "/tmp/x.service",
          "content": "ExecStart=/good\n", "why": "fix path", "apply": True},
         {"type": "final", "answer": "fixed and verified", "escalated": False}],
        confirm=lambda r: True,
    )
    assert result["answer"] == "fixed and verified"
    assert applied == [("write_file", "/tmp/x.service")], applied
    assert any("applied write_file" in o for o in obs)


def test_apply_declined_never_executes():
    result, applied, obs = _with_mocks(
        [{"type": "action", "action": "fix", "cmd": "systemctl --user restart x",
          "why": "restart", "apply": True},
         {"type": "final", "answer": "left it to you", "escalated": False}],
        confirm=lambda r: False,   # user says NO
    )
    assert applied == [], "execute_write ran despite the user declining!"
    assert any("skipped" in o.lower() for o in obs), obs


def test_no_confirm_callback_declines_by_default():
    # confirm=None (e.g. piped stdin) must NOT auto-apply — safe default.
    result, applied, obs = _with_mocks(
        [{"type": "action", "action": "fix", "cmd": "systemctl --user restart x",
          "why": "restart", "apply": True},
         {"type": "final", "answer": "done", "escalated": False}],
        confirm=None,
    )
    assert applied == [], "an apply action ran with no confirm callback!"


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
