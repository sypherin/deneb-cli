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


def _transcript_md(hist: list[dict], final_answer: str) -> str:
    """Render the full session as readable markdown for the escalation log. Tool output is
    already secret-redacted (deneb.tools redacts before it ever reaches here)."""
    out = ["# Deneb session log", ""]
    for h in hist:
        role, content = h.get("role"), str(h.get("content", ""))
        if role == "user":
            if content.strip() == _FORCE_FINAL.strip():
                continue  # internal nudge, not a user turn
            out += ["### 🧑 User", "", content, ""]
        elif role == "assistant":
            try:
                a = json.loads(content)
                ran = a.get("cmd") or a.get("path") or json.dumps(a)
                out += [f"**→ Deneb ran:** `{ran}`", ""]
            except Exception:  # noqa: BLE001
                out += [f"**→ Deneb:** {content}", ""]
        elif role == "tool":
            out += [f"**observation ({h.get('name', 'tool')}):**", "```",
                    content[:4000] + ("\n… (truncated)" if len(content) > 4000 else ""), "```", ""]
    out += ["---", "", "### 🛰 Deneb's diagnosis (escalated)", "", final_answer, ""]
    return "\n".join(out)


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


def _maybe_escalate(hist: list[dict], question: str, result: dict, on_event) -> None:
    """When the engine flags escalated=true, hand it the FULL session so it DMs the
    engineer a brief + the complete session log. Best-effort; never breaks the reply."""
    if not result.get("escalated"):
        return
    try:
        transcript = _transcript_md(hist, result.get("answer", ""))
        sent = client.escalate(transcript, question, result.get("answer", ""))
        if on_event:
            on_event("escalated", {"sent": sent})
            result["_escalation_announced"] = True
    except Exception:  # noqa: BLE001
        pass


def run(question: str, history: list[dict] | None = None, on_event=None) -> dict:
    """Drive the loop for one question. `on_event(kind, data)` gets 'action' /
    'observation' events for the UI. Returns the engine's final result dict."""
    hist = list(history or [])
    hist.append({"role": "user", "content": question})

    result = None
    for _ in range(MAX_STEPS):
        res = _step(hist, on_event)
        if res.get("type") != "action":
            result = res  # final (or a malformed-but-final result)
            break

    if result is None:
        # Step budget hit — force ONE concluding call so the user always gets a verdict.
        hist.append({"role": "user", "content": _FORCE_FINAL})
        try:
            res = client.agent_step(hist)
        except client.DenebError:
            res = {}
        result = res if res.get("answer") else {
            "type": "final", "escalated": False, "sources": [],
            "answer": "(gathered a lot of evidence but couldn't converge — paste the "
                      "specific error you're seeing, or narrow the question)"}

    _maybe_escalate(hist, question, result, on_event)
    return result
