"""deneb.ui — the Deneb terminal UI. 100% Altronis-branded; NO third-party names,
logos, or telemetry. Plain ANSI (zero deps) so install stays trivial.
"""
from __future__ import annotations

import re
import sys
import unicodedata

from .colour import Palette

_C = Palette(teal="teal", dim="dim", b="bold", r="reset", red="red", grey="grey",
             green="green", amber="amber")


def _p(s: str = "") -> None:
    print(s)
    sys.stdout.flush()


def banner() -> None:
    _p(f"{_C['teal']}{_C['b']}◇ Deneb{_C['r']}  {_C['dim']}Altronis · private-LLM setup for your AI box{_C['r']}")


def info(s: str) -> None:
    _p(f"{_C['grey']}{s}{_C['r']}")


def para(text: str, first: str = "   ", rest=None, width: int = 80) -> str:
    """One paragraph of prose wrapped to `width` columns: `first` before line one, `rest`
    (default: as wide as `first`, in spaces) before the others. Never use it on a command
    line: a command must stay one line so it can be copied."""
    import textwrap
    return textwrap.fill(" ".join(str(text).split()), width=width, initial_indent=first,
                         subsequent_indent=" " * len(first) if rest is None else rest,
                         break_on_hyphens=False, break_long_words=False)


def error(s: str) -> None:
    _p(f"{_C['red']}✗ {s}{_C['r']}")


# C0/C1 controls, line/paragraph separators, and every Unicode format character (category Cf:
# bidi overrides and isolates, zero-width characters, the BOM, U+061C, U+180E ...). Cf is decided
# per character from this Python's Unicode data, so it never goes stale and costs nothing at
# import (the old module-level table scanned all 1.1M code points on every start).
_NOT_PLAIN = re.compile("[^\\x20-\\x7e]")      # anything outside printable ASCII is checked


def _unsafe(ch: str) -> bool:
    o = ord(ch)
    return (o < 0x20 or 0x7f <= o <= 0x9f or o in (0x2028, 0x2029, 0x180E)
            or unicodedata.category(ch) == "Cf")


_NAMED = {"\r": "\\r", "\n": "\\n", "\t": "\\t"}


def visible(s, keep: str = "") -> str:
    """`s` with every control character shown as an escape (\\x1b, \\r, \\x07 ...) instead
    of obeyed, so nothing the model wrote can rewrite what the user is looking at. Characters
    in `keep` (newline, tab) are left alone."""
    def one(m):
        c = m.group(0)
        if c in keep or not _unsafe(c):
            return c
        return _NAMED.get(c) or (f"\\x{ord(c):02x}" if ord(c) < 256 else f"\\u{ord(c):04x}" if ord(c) < 0x10000 else f"\\U{ord(c):08x}")
    return _NOT_PLAIN.sub(one, str(s))


PREVIEW_FULL, PREVIEW_HEAD = 60, 40


def _in_exec(lines, i) -> bool:
    """True when line i is an Exec line or a backslash-continuation of one."""
    while i >= 0:
        if lines[i].lstrip().startswith("Exec"):
            return True
        if i == 0 or not lines[i - 1].rstrip().endswith("\\"):
            return False
        i -= 1
    return False


def content_preview(content: str) -> list:
    """Lines to show for a proposed file write. Up to 60 lines are shown whole. Longer content
    shows the first 40, then every later line that starts with Exec, plus the \\ continuation lines of an Exec line (each with its line number) so a
    payload cannot hide below the fold, then how many lines were not shown."""
    lines = (content or "").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) <= PREVIEW_FULL:
        return [visible(x, "\t") for x in lines]
    out = [visible(x, "\t") for x in lines[:PREVIEW_HEAD]]
    shown = 0
    cont = lines[PREVIEW_HEAD - 1].rstrip().endswith("\\") and _in_exec(lines, PREVIEW_HEAD - 1)
    for n, x in enumerate(lines[PREVIEW_HEAD:], start=PREVIEW_HEAD + 1):
        if x.lstrip().startswith("Exec") or cont:
            out.append(f"{n}: {visible(x, chr(9))}")
            shown += 1
            cont = x.rstrip().endswith("\\")
        else:
            cont = False
    out.append(f"{len(lines) - PREVIEW_HEAD - shown} more lines not shown")
    return out


def event(kind: str, d: dict) -> None:
    if kind == "action":
        a = d.get("action", "")
        label = visible(d.get("cmd") or d.get("path") or "")
        verb = {"run": "checking", "read_file": "reading", "list_dir": "listing"}.get(a, a)
        _p(f"{_C['dim']}  ▸ {verb}: {label}{_C['r']}")
    elif kind == "escalated":
        msg = ("looping in an Altronis engineer — sent them your full session log"
               if d.get("sent") else "flagging this to an Altronis engineer")
        _p(f"{_C['amber']}  ↑ {msg}.{_C['r']}")
    elif kind == "propose":
        action = d.get("action")
        _p("")
        if action == "write_file":
            _p(f"{_C['amber']}  ⚙ proposed edit:{_C['r']} write {visible(d.get('path', ''))}")
            for ln in content_preview(d.get("content", "")):
                _p(f"{_C['dim']}      {ln}{_C['r']}")
        else:
            _p(f"{_C['amber']}  ⚙ proposed fix:{_C['r']} {visible(d.get('cmd', ''))}")
        if d.get("why"):
            _p(f"{_C['dim']}    why: {visible(d['why'])}{_C['r']}")
        _p(f"{_C['dim']}    (reversible — deneb backs up any file it changes){_C['r']}")
    elif kind == "applied":
        ok = d.get("ok")
        mark, col = ("✓", _C["green"]) if ok else ("✗", _C["red"])
        _p(f"{col}  {mark} applied:{_C['r']} {visible(d.get('output', ''), chr(10))}")
    elif kind == "skipped":
        _p(f"{_C['dim']}  ▸ skipped — not applied.{_C['r']}")
    # observations stay quiet by default (the model reasons on them server-side)


def final(res: dict) -> None:
    ans = (res.get("answer") or "").strip()
    _p("")
    _p(ans)
    if res.get("hint"):
        _p(f"\n{_C['grey']}{res['hint']}{_C['r']}")
    srcs = res.get("sources") or []
    if srcs:
        _p(f"\n{_C['grey']}— grounded in the Altronis runbook:{_C['r']}")
        for s in srcs[:2]:
            _p(f"{_C['grey']}  · {s.get('label', '')}{_C['r']}")
    # escalation is announced via the 'escalated' event (loop path). For the single-shot
    # image path there's no live event, so note it here as a fallback.
    if res.get("escalated") and not res.get("_escalation_announced"):
        _p(f"\n{_C['amber']}↑ also flagged to an Altronis engineer.{_C['r']}")
