"""deneb.tokenmark — back Deneb's advice with REAL measured benchmarks.

The curated models_catalog is honest that its numbers are approximations, not
measured facts. TokenMark (tokenmark.app) is the community tracker of MEASURED
configs (decode tok/s per model x quant x backend x hardware). This module pulls
those real numbers so Deneb can say "measured 92 tok/s on a Strix Halo (source:
kyuz0)" instead of only "should fit".

Fails SOFT: any network/parse error (or the site being down) returns None and
Deneb falls back to the static catalog — the box-setup flow never depends on an
external service being reachable. stdlib only, no new dependencies.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional

BASE = "https://tokenmark.app"
_TIMEOUT = 8

# Deneb's capability vocabulary -> TokenMark task tags.
_TASK_MAP = {
    "coding": "coding",
    "vision": "vision",
    "chat": "general",
    "general": "general",
}


def _get(path: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(f"{BASE}{path}", headers={"User-Agent": "deneb-cli"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def recommend(hardware: str, capabilities=None, prefer: str = "balanced", limit: int = 5) -> Optional[list]:
    """Ranked real-config picks for a hardware + use-case, or None on failure.

    Returns a list of dicts: {model, decode_tps, quant, backend, hardware,
    throughput_kind, trust, author, source, why}. Numbers are measured, not
    guessed. None means "TokenMark unavailable — use the static catalog".
    """
    tasks = [_TASK_MAP.get(c, c) for c in (capabilities or [])]
    q = urllib.parse.urlencode(
        {"hardware": hardware, "tasks": ",".join(tasks), "prefer": prefer, "limit": limit}
    )
    data = _get(f"/api/recommend?{q}")
    if not data or not data.get("recommendations"):
        return None
    out = []
    for r in data["recommendations"]:
        c = r.get("config", {})
        out.append({
            "model": r.get("model"),
            "vendor": r.get("vendor"),
            "decode_tps": c.get("decode_tps"),
            "quant": c.get("quant"),
            "backend": c.get("backend"),
            "variant": c.get("variant"),
            "hardware": c.get("hardwareLabel") or c.get("hardware"),
            "throughput_kind": c.get("throughput_kind"),
            "trust": c.get("trustLabel"),
            "author": c.get("author"),
            "source": c.get("url"),
            "why": r.get("why", []),
        })
    return out


def measured_for_model(model_name: str, hardware: Optional[str] = None, limit: int = 5) -> Optional[list]:
    """Real measured configs for a specific model (to enrich a catalog entry with
    an actual tok/s + source). None if TokenMark is unavailable."""
    data = _get("/tracker-summary.json")
    if not data or not data.get("configs"):
        return None
    key = (model_name or "").lower()
    rows = [c for c in data["configs"] if key in (c.get("model") or "").lower()]
    if hardware:
        h = hardware.lower()
        rows = [c for c in rows if h in f"{c.get('hardware','')} {c.get('hardwareLabel','')}".lower()]
    rows.sort(key=lambda c: c.get("decode_tps") or 0, reverse=True)
    return rows[:limit] or None
