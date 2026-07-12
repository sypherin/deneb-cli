"""deneb.ui — the Deneb terminal UI. 100% Altronis-branded; NO third-party names,
logos, or telemetry. Plain ANSI (zero deps) so install stays trivial.
"""
from __future__ import annotations

import sys

_C = {
    "teal": "\033[38;5;44m", "dim": "\033[2m", "b": "\033[1m", "r": "\033[0m",
    "red": "\033[31m", "grey": "\033[90m", "green": "\033[32m", "amber": "\033[33m",
}


def _p(s: str = "") -> None:
    print(s)
    sys.stdout.flush()


def banner() -> None:
    _p(f"{_C['teal']}{_C['b']}◇ Deneb{_C['r']}  {_C['dim']}Altronis · private-LLM setup for your AI box{_C['r']}")


def info(s: str) -> None:
    _p(f"{_C['grey']}{s}{_C['r']}")


def error(s: str) -> None:
    _p(f"{_C['red']}✗ {s}{_C['r']}")


def event(kind: str, d: dict) -> None:
    if kind == "action":
        a = d.get("action", "")
        label = d.get("cmd") or d.get("path") or ""
        verb = {"run": "checking", "read_file": "reading", "list_dir": "listing"}.get(a, a)
        _p(f"{_C['dim']}  ▸ {verb}: {label}{_C['r']}")
    elif kind == "escalated":
        msg = ("looping in an Altronis engineer — sent them your full session log"
               if d.get("sent") else "flagging this to an Altronis engineer")
        _p(f"{_C['amber']}  ↑ {msg}.{_C['r']}")
    # observations stay quiet by default (the model reasons on them server-side)


def final(res: dict) -> None:
    ans = (res.get("answer") or "").strip()
    _p("")
    _p(ans)
    srcs = res.get("sources") or []
    if srcs:
        _p(f"\n{_C['grey']}— grounded in your Neo runbook:{_C['r']}")
        for s in srcs[:2]:
            _p(f"{_C['grey']}  · {s.get('label', '')}{_C['r']}")
    # escalation is announced via the 'escalated' event (loop path). For the single-shot
    # image path there's no live event, so note it here as a fallback.
    if res.get("escalated") and not res.get("_escalation_announced"):
        _p(f"\n{_C['amber']}↑ also flagged to an Altronis engineer.{_C['r']}")
