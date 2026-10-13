"""deneb.client — HTTPS to the Altronis / Deneb engine.

Talks to ONE endpoint (the Altronis engine) with your Bearer token. Never any third-party
service. `/agent-step` returns the next read-only tool to run, or the final diagnosis.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os

from . import config, freecmds, net

_UA = net.UA    # one User-Agent for every request (defined in net.py)


class DenebError(Exception):
    pass


_brain = None   # a deneb.brain.Brain for a local/custom model; None means the Altronis engine


def set_brain(b) -> None:
    """Choose the model the conversational calls below use. None or an `altronis` brain keeps
    them on the Altronis engine."""
    global _brain
    _brain = b if (b is not None and b.kind != "altronis") else None


def _image_data_url(path: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        raise DenebError(f"no such image: {path}")
    if os.path.getsize(p) > 8_000_000:
        raise DenebError("image too large (max 8MB)")
    mime = mimetypes.guess_type(p)[0] or "image/png"
    with open(p, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _post(pathname: str, payload: dict, timeout: int) -> dict:
    engine = config.get_engine().rstrip("/")
    token = config.get_token()
    if not token:
        raise DenebError("the Altronis engine needs a token: deneb auth --request (or deneb auth --token <token>). "
                         + freecmds.hint())
    try:
        code, raw = net.request("POST", engine + pathname, body=payload,
                                headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    except net.RedirectRefused as e:
        raise DenebError(str(e)) from e
    except net.NetError as e:
        raise DenebError(
            f"can't reach the Deneb engine ({engine}): {e}. "
            "check this box has internet and the endpoint is up.") from e
    if code == 401:
        raise DenebError("this token is no longer valid (rejected or revoked). "
                         "ask for a new one: deneb auth --request")
    if code == 429:
        msg = net.scrub(net.json_obj(raw).get("error", ""), (token,)).strip()
        raise DenebError(msg or "daily limit reached, try again later")
    if code >= 400:
        raise DenebError(f"the Deneb engine returned {code}")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise DenebError("the Deneb engine sent an answer deneb could not read") from e


def ask(question: str, image: str | None = None, timeout: int = 180) -> dict:
    """One-shot chat (used for screenshots — reuses the engine's OCR /ask path)."""
    if _brain:
        from . import brain
        return brain.ask_image(_brain, question, image) if image else \
            brain.agent_step(_brain, [{"role": "user", "content": question}])
    payload: dict = {"question": question}
    if image:
        payload["image"] = _image_data_url(image)
    data = _post("/ask", payload, timeout)
    return {"type": "final", "answer": data.get("answer", ""),
            "escalated": data.get("escalated", False), "sources": data.get("sources", [])}


def agent_step(history: list[dict], timeout: int = 180) -> dict:
    if _brain:
        from . import brain
        return brain.agent_step(_brain, history)
    return _post("/agent-step", {"history": history}, timeout)


def summarize(text: str, timeout: int = 120) -> str:
    """Ask the engine to compact a long session into a brief (auto-compaction)."""
    if _brain:
        from . import brain
        return brain.summarize(_brain, text)
    return _post("/summarize", {"text": text}, timeout).get("summary", "")


def escalate(transcript: str, question: str, answer: str, timeout: int = 60) -> bool:
    """On escalation, hand the engine the FULL (already secret-redacted) session so it can
    DM the engineer a brief + the complete session log as a .md. Best-effort — a failure
    here must never crash the user's session."""
    if _brain:
        return False        # no escalation on a local brain
    try:
        _post("/escalate", {"transcript": transcript, "question": question,
                            "answer": answer}, timeout)
        return True
    except DenebError:
        return False


def health(timeout: int = 15) -> dict:
    engine = config.get_engine().rstrip("/")
    try:
        code, raw = net.request("GET", engine + "/health", timeout=timeout)
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
