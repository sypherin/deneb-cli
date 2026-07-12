"""deneb.config — token + endpoint config. Token stored chmod-600 in ~/.config/deneb.

Env overrides (handy for testing): DENEB_TOKEN, DENEB_ENGINE.
"""
from __future__ import annotations

import json
import os

CONFIG_DIR = os.path.expanduser("~/.config/deneb")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_ENGINE = "https://deneb-engine.altronis.sg"


def _load() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def get_token() -> str | None:
    return os.getenv("DENEB_TOKEN") or _load().get("token")


def get_engine() -> str:
    return os.getenv("DENEB_ENGINE") or _load().get("engine") or DEFAULT_ENGINE


def save(token: str | None = None, engine: str | None = None) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    d = _load()
    if token is not None:
        d["token"] = token
    if engine:
        d["engine"] = engine
    # Write with 0600 from the start (never leave a world-readable token window).
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.chmod(CONFIG_FILE, 0o600)


def clear() -> None:
    try:
        os.remove(CONFIG_FILE)
    except FileNotFoundError:
        pass
