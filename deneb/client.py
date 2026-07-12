"""deneb.client — HTTPS to the Neo Altronis / Deneb engine.

Talks to ONE endpoint (your Neo cloud) with your Bearer token. Never any third-party
service. `/agent-step` returns the next read-only tool to run, or the final diagnosis.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

from . import __version__, config

# The Neo gateway/Cloudflare edge blocks the default "Python-urllib" UA as a bot
# signature — identify ourselves honestly so requests aren't 403'd at the edge.
_UA = f"deneb-cli/{__version__}"


class DenebError(Exception):
    pass


def _image_data_url(path: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        raise DenebError(f"no such image: {path}")
    if os.path.getsize(p) > 10_000_000:
        raise DenebError("image too large (max 10MB)")
    mime = mimetypes.guess_type(p)[0] or "image/png"
    with open(p, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _post(pathname: str, payload: dict, timeout: int) -> dict:
    engine = config.get_engine().rstrip("/")
    token = config.get_token()
    if not token:
        raise DenebError("not signed in — run: deneb auth --token <your-token>")
    req = urllib.request.Request(
        engine + pathname, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                 "User-Agent": _UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise DenebError("your token was rejected — re-run `deneb auth --token ...`") from e
        if e.code == 429:
            raise DenebError("daily limit reached — try again later, or ping your Altronis engineer") from e
        raise DenebError(f"Neo endpoint returned {e.code}") from e
    except urllib.error.URLError as e:
        raise DenebError(
            f"can't reach your Neo endpoint ({engine}): {e.reason}. "
            "check this box has internet and the endpoint is up."
        ) from e


def ask(question: str, image: str | None = None, timeout: int = 180) -> dict:
    """One-shot chat (used for screenshots — reuses the engine's OCR /ask path)."""
    payload: dict = {"question": question}
    if image:
        payload["image"] = _image_data_url(image)
    data = _post("/ask", payload, timeout)
    return {"type": "final", "answer": data.get("answer", ""),
            "escalated": data.get("escalated", False), "sources": data.get("sources", [])}


def agent_step(history: list[dict], timeout: int = 180) -> dict:
    return _post("/agent-step", {"history": history}, timeout)


def summarize(text: str, timeout: int = 120) -> str:
    """Ask the engine to compact a long session into a brief (auto-compaction)."""
    return _post("/summarize", {"text": text}, timeout).get("summary", "")


def escalate(transcript: str, question: str, answer: str, timeout: int = 60) -> bool:
    """On escalation, hand the engine the FULL (already secret-redacted) session so it can
    DM the engineer a brief + the complete session log as a .md. Best-effort — a failure
    here must never crash the user's session."""
    try:
        _post("/escalate", {"transcript": transcript, "question": question,
                            "answer": answer}, timeout)
        return True
    except DenebError:
        return False


def health(timeout: int = 15) -> dict:
    engine = config.get_engine().rstrip("/")
    try:
        req = urllib.request.Request(engine + "/health", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
