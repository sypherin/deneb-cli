"""deneb.loop — the agent loop.

The engine decides the next READ-ONLY tool; we run it locally (deneb.tools, the
security boundary), feed the observation back, and repeat until the engine returns a
final diagnosis. The conversation lives here on the box; the moat stays server-side.
"""
from __future__ import annotations

import json

from . import client, tools

MAX_STEPS = 22
_FORCE_FINAL = (
    "You have gathered enough evidence — STOP running tools now. Reply with a FINAL "
    "answer only (action must be \"final\"): give the ✓/✗ checklist and your verdict "
    "from what you've already found. Assess against what is ACTUALLY installed on this "
    "box (it may run Qwen3.6 rather than the runbook's default Gemma), not just the "
    "runbook default."
)


def _action_repr(res: dict) -> str:
    return json.dumps({k: res[k] for k in ("action", "cmd", "path") if res.get(k)})


def _step(hist: list[dict], on_event) -> dict:
    """One engine round-trip; if it's a tool action, execute it locally and record it.
    Returns the engine result (the caller checks type)."""
    res = client.agent_step(hist)
    if res.get("type") != "action":
        return res
    action = res.get("action", "")
    cmd = res.get("cmd", "") or ""
    path = res.get("path", "") or ""
    if on_event:
        on_event("action", {"action": action, "cmd": cmd, "path": path,
                            "thought": res.get("thought", "")})
    obs = tools.execute(action, cmd=cmd, path=path)
    if on_event:
        on_event("observation", {"action": action, "output": obs.get("output", "")})
    hist.append({"role": "assistant", "content": _action_repr(res)})
    hist.append({"role": "tool", "name": action, "content": obs.get("output", "")})
    return res


def run(question: str, history: list[dict] | None = None, on_event=None) -> dict:
    """Drive the loop for one question. `on_event(kind, data)` gets 'action' /
    'observation' events for the UI. Returns the engine's final result dict."""
    hist = list(history or [])
    hist.append({"role": "user", "content": question})

    for _ in range(MAX_STEPS):
        res = _step(hist, on_event)
        if res.get("type") != "action":
            return res  # final (or a malformed-but-final result)

    # Step budget hit — force ONE concluding call so the user always gets a verdict.
    hist.append({"role": "user", "content": _FORCE_FINAL})
    try:
        res = client.agent_step(hist)
    except client.DenebError:
        res = {}
    if res.get("answer"):
        return res
    return {"type": "final", "escalated": False, "sources": [],
            "answer": "(gathered a lot of evidence but couldn't converge — paste the "
                      "specific error you're seeing, or narrow the question)"}
