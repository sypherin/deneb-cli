"""deneb.jsonrecover — turn a local model's raw reply into one agent-step dict.

Pure: text in, dict out, no I/O. Small local models put a reasoning block first, wrap the JSON
in a code fence, or cut off mid-object. An ACTION is accepted only when the whole response (after
a leading reasoning block and one optional code fence) is a single JSON object: prose that
merely contains JSON is a final answer, shown as prose, and nothing nested inside a final is
ever promoted to an action. When the reply is cut off the user gets plain text saying so.
It never hands raw JSON to the user.

The result has the same shape the Deneb engine's agent step returns:
  {"type": "action", "action": run|read_file|list_dir|fix|write_file, "cmd", "path",
   "content", "why", "thought"}
  {"type": "final", "answer", "escalated": False, "sources": []}
"""
from __future__ import annotations

import ast
import json
import re
import shlex

ACTIONS = ("run", "read_file", "list_dir", "fix", "write_file")

_THINK_OPEN = re.compile(r"^\s*<think>", re.I)
_FENCE_CLOSED = re.compile(r"^```[A-Za-z]*[ \t]*\n?(.*?)\s*```$", re.S)
_FENCE_OPEN = re.compile(r"^```[A-Za-z]*[ \t]*\n?(.*)$", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_ANSWER_START = re.compile(r'"(?:answer|final|text|response|message|reply)"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)
_ANSWER_KEYS = ("answer", "final", "text", "response", "message", "content", "output", "result", "reply")

GARBLED = ("I couldn't turn the model's reply into an answer (it came back garbled). "
            "Try again, or use a larger model.")
EMPTY = ("The model returned nothing to show. Try again, or use a larger model.")
CUT_OFF = ("The model's reply was cut off before it finished, so Deneb did not act on it. "
           "Try again, or use a larger model.")
CUT_OFF_NOTE = "(the model's reply was cut off, so this answer may be incomplete)"


def strip_think(text: str) -> str:
    """Remove a reasoning block ONLY when the text starts with one: optional whitespace, then
    <think> up to the first </think> (an unclosed block is all reasoning, so nothing is left).
    The same tags anywhere later are content, e.g. inside a JSON string.
    One extra case: reasoning that ends in a bare </think> (templates that open the block in the
    prompt) is dropped, but ONLY when what follows the tag is one whole JSON object (bare or in a
    code fence) that recovers to a step. Prose that merely mentions </think> is returned whole."""
    t = text or ""
    if _THINK_OPEN.match(t):
        i = t.lower().find("</think>")
        return "" if i == -1 else t[i + len("</think>"):]
    j = t.lower().find("</think>")
    if j != -1 and not t.lstrip().startswith(("{", "`")) and "<think>" not in t[:j].lower():
        rest = t[j + len("</think>"):]
        if rest.lstrip().startswith(("{", "`")):
            step = parse_step(rest)
            if step["type"] == "action" or step["answer"] != rest.strip():
                return rest
    return t


def _line_text(v) -> str:
    """Text for a field a person reads (answer, content, why): lists become lines."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "\n".join(_line_text(x) for x in v)
    if isinstance(v, dict):
        return "\n".join(f"{k}: {_line_text(x)}" for k, x in v.items())
    return str(v)


def _cmd_text(v) -> str:
    """A command field: a list of words is ONE command line, never several lines."""
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float)) for x in v):
        return shlex.join(str(x) for x in v)
    return _line_text(v)


def _s(v) -> str:
    return _line_text(v)


def _final(ans: str) -> dict:
    return {"type": "final", "answer": ans.strip() or EMPTY, "escalated": False, "sources": []}


def _normalise(obj: dict):
    """A step dict, or None when the object is not a step we can honour."""
    typ = _s(obj.get("type")).lower()
    action = _s(obj.get("action")).lower()
    answer = next((obj[k] for k in _ANSWER_KEYS if obj.get(k) not in (None, "", [], {})), None)
    if action == "final" or typ == "final":
        return _final(_line_text(answer))
    if action in ACTIONS and typ in ("", "action"):
        return {"type": "action", "action": action, "cmd": _cmd_text(obj.get("cmd") or obj.get("command")),
                "path": _s(obj.get("path")), "content": _line_text(obj.get("content")),
                "why": _s(obj.get("why")), "thought": _s(obj.get("thought"))}
    if "answer" in obj and not action:
        return _final(_line_text(obj["answer"]))
    return None


def _loads(text: str):
    """json.loads, then the same with trailing commas removed, then a Python-literal parse
    (single quotes). ValueError when none of them reads it."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", text))
    except ValueError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise ValueError("not JSON") from None


def _cut_off(inner: str) -> dict:
    """The reply starts like JSON but does not parse. Show any answer text it did get out,
    with one line saying it was cut off; never the raw JSON, never an action."""
    m = _ANSWER_START.search(inner)
    if m:
        try:
            said = json.loads('"' + m.group(1) + '"')
        except ValueError:
            said = m.group(1)
        if said.strip():
            return _final(said.strip() + "\n\n" + CUT_OFF_NOTE)
    return _final(CUT_OFF)


def parse_step(text: str) -> dict:
    t = strip_think(text).strip()
    if not t:
        return _final("")
    inner = t
    m = _FENCE_CLOSED.match(t) or (_FENCE_OPEN.match(t) if t.startswith("```") else None)
    if m:
        inner = m.group(1).strip()
    if inner.startswith("{"):
        try:
            obj = _loads(inner)
        except ValueError:
            return _cut_off(inner)
        step = _normalise(obj) if isinstance(obj, dict) else None
        return step or _final(GARBLED)
    if m:                       # a fence around something that is not an object: its text
        return _final(inner)
    if t.startswith('"'):
        try:
            v = json.loads(t)
        except ValueError:
            v = None
        if isinstance(v, str):
            return _final(v)
    return _final(t)            # prose is the answer, whatever it quotes
