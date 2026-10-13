"""deneb.config — token + endpoint config. Token stored chmod-600 in ~/.config/deneb.

Env overrides (handy for testing): DENEB_TOKEN, DENEB_ENGINE.
"""
from __future__ import annotations

import json
import os

CONFIG_DIR = os.path.expanduser("~/.config/deneb")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
REQUEST_FILE = os.path.join(CONFIG_DIR, "request.json")   # a pending token request (mode 600)
DEFAULT_ENGINE = "https://deneb-engine.altronis.sg"


def _load() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def get_token() -> str | None:
    return os.getenv("DENEB_TOKEN") or _load().get("token")


def saved_token() -> str | None:
    """The token saved by `deneb auth`, ignoring the DENEB_TOKEN environment variable."""
    return _load().get("token") or None


def get_brain() -> dict | None:
    """The brain saved by `deneb brain ...`: {mode, url, model, key_env} or None. The API key
    itself is never stored, only the NAME of the environment variable that holds it."""
    b = _load().get("brain")
    return b if isinstance(b, dict) and b.get("mode") else None


def get_engine() -> str:
    return os.getenv("DENEB_ENGINE") or _load().get("engine") or DEFAULT_ENGINE


def _write(d: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    # Write with 0600 from the start (never leave a world-readable token window).
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.chmod(CONFIG_FILE, 0o600)


def save(token: str | None = None, engine: str | None = None) -> None:
    d = _load()
    if token is not None:
        d["token"] = token
    if engine:
        d["engine"] = engine
    _write(d)


def save_brain(mode: str, url: str | None = None, model: str | None = None,
               key_env: str | None = None) -> None:
    """Save the chosen brain. `key_env` is an environment variable NAME, never a key."""
    d = _load()
    b = {"mode": mode}
    if mode == "url":
        b.update({"url": url, "model": model, "key_env": key_env})
    elif mode == "local":
        if model:
            b["model"] = model
        if key_env:
            b["key_env"] = key_env
    d["brain"] = b
    _write(d)


def load_request() -> dict | None:
    try:
        with open(REQUEST_FILE) as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return d if isinstance(d, dict) and d.get("id") and d.get("secret") else None


def save_request(d: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    fd = os.open(REQUEST_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.chmod(REQUEST_FILE, 0o600)


def clear_request() -> None:
    try:
        os.remove(REQUEST_FILE)
    except FileNotFoundError:
        pass


def clear() -> None:
    try:
        os.remove(CONFIG_FILE)
    except FileNotFoundError:
        pass
