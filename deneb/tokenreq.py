"""deneb.tokenreq — ask Altronis for an engine token from the command line.

POST /token-request sends name, email, box, why and the sha256 of a secret that stays on this
box (~/.config/deneb/request.json, mode 600). GET /token-request/<id> with the secret in the
X-Deneb-Request-Secret header says pending, denied, expired, revoked, claimed or approved; the token
comes back once, on approval, and is saved the way `deneb auth --token` saves it.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import time

from . import config, net

_UA = net.UA
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
REVIEW_LINE = "a person at Altronis reviews each request"


class RequestError(Exception):
    pass


def valid_email(e: str) -> bool:
    return bool(_EMAIL.match(e or "")) and 3 <= len(e) <= 120


def _http(method: str, url: str, body=None, headers=None, timeout: int = net.TIMEOUT):
    """(status, parsed-json-or-{}) — HTTP errors come back as a status, a dead network raises."""
    try:
        code, raw = net.request(method, url, body=body, headers=headers, timeout=timeout)
    except net.RedirectRefused as e:
        raise RequestError(str(e)) from e
    except net.NetError as e:
        raise RequestError(f"can't reach the Deneb engine ({url.split('/token-request')[0]}): "
                           f"{e}. check this box has internet.") from e
    return code, net.json_obj(raw)


NO_ROUTE = ("this Altronis engine does not take token requests yet. you can still use your own "
            "model: deneb brain local  (or deneb brain <url> for an OpenAI-compatible server)")


def _server_text(obj: dict, code: int) -> str:
    return net.scrub(obj.get("error") or obj.get("message") or f"the engine returned {code}")


def submit(name: str, email: str, box: str, why: str, engine: str | None = None) -> dict:
    """Send the request; on 202 save request.json and return {id}. Raises RequestError."""
    engine = (engine or config.get_engine()).rstrip("/")
    secret = secrets.token_urlsafe(32)
    body = {"name": name, "email": email, "box": box, "problem": why,
            "secret_hash": hashlib.sha256(secret.encode()).hexdigest(),
            "client": _UA}
    code, obj = _http("POST", engine + "/token-request", body)
    if code != 202 or not obj.get("id"):
        if code == 404:
            raise RequestError(NO_ROUTE)
        if code == 413:
            raise RequestError("the request was too large for the engine (413)")
        raise RequestError(_server_text(obj, code))
    # the engine URL is always stored: --check must poll the engine this request went to
    rec = {"id": obj["id"], "secret": secret, "email": email, "engine": engine,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    config.save_request(rec)
    return rec


def _approved_message() -> str:
    saved = config.get_brain()
    if saved and saved.get("mode") in ("local", "url"):
        own = "your local model" if saved.get("mode") == "local" else "the model you chose"
        return ("approved. your token is saved. chats still use " + own +
                "; switch to the Altronis engine with: deneb brain altronis")
    return "approved. your token is saved; any deneb chat command can use it now."


def check_once(timeout: float = net.TIMEOUT) -> tuple:
    """Poll the pending request once. Returns (state, message) where state is one of
    none, pending, approved, denied, expired, revoked, claimed, unknown, error."""
    req = config.load_request()
    if not req:
        return "none", "no request is pending. ask for a token with: deneb auth --request"
    engine = (req.get("engine") or config.get_engine()).rstrip("/")
    try:
        code, obj = _http("GET", f"{engine}/token-request/{req['id']}",
                          headers={"X-Deneb-Request-Secret": req["secret"]}, timeout=timeout)
    except RequestError as e:
        return "error", str(e)
    if code == 404 and obj.get("status") != "unknown":     # the route itself is missing
        return "error", NO_ROUTE
    if code == 404:
        config.clear_request()
        return "unknown", ("Altronis has no record of that request (it may have expired). "
                           "ask again: deneb auth --request")
    if code != 200:
        return "error", _server_text(obj, code)
    status = obj.get("status")
    if status == "approved" and obj.get("token"):
        eng = (req.get("engine") or "").rstrip("/")
        config.save(token=str(obj["token"]).strip(),
                    engine=eng if eng and eng != config.DEFAULT_ENGINE.rstrip("/") else None)
        config.clear_request()
        return "approved", _approved_message()
    if status == "pending":
        return "pending", f"status: pending. {REVIEW_LINE}. check again later: deneb auth --check"
    if status in ("denied", "expired"):
        config.clear_request()
        return status, (f"this request was {status}. you can ask again: deneb auth --request"
                        if status == "expired" else
                        "this request was denied. the free commands and your own model still work.")
    if status == "revoked":
        config.clear_request()
        return "revoked", ("this request's token was revoked. the free commands and your own "
                           "model still work.")
    if status == "claimed":
        config.clear_request()
        return "claimed", ("this request's token was already collected (on this box or another). "
                           "if it isn't saved here, ask again: deneb auth --request")
    return "error", f"unexpected answer from the engine: {status!r}"
