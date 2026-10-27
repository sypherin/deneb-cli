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
import math
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
            from .net import read_capped
            return json.loads(read_capped(r).decode("utf-8"))
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


def _rows(data) -> list:
    """Every measured row in a tracker summary: `configs` (community and linked runs) plus
    `card_configs` (the per-card runs, Altronis's own cloud-GPU runs among them). Either list
    may be missing; anything that is not a dict is dropped."""
    if not isinstance(data, dict):
        return []
    out = []
    for key in ("configs", "card_configs"):
        part = data.get(key)
        if isinstance(part, list):
            out += [c for c in part if isinstance(c, dict)]
    return out


def _count(v) -> int:
    """A whole-number row field (concurrency, num_gpus, ctx) as an int: an int, a whole float or
    a digit string ("2", 2.0, " 256 "); 0 for anything else (missing, bool, text, fractions,
    zero or less). TokenMark sends ints today; a row that sends "2" must not read as no value."""
    if isinstance(v, bool):
        return 0
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            return 0
    if isinstance(v, (int, float)) and math.isfinite(v) and v == int(v) and v > 0:
        return int(v)
    return 0


def _one_stream(c: dict) -> bool:
    """One request at a time: throughput kind single-stream (or not given) and a concurrency of
    1 (or not given). Aggregate rows (many users at once) are a different measurement."""
    kind = str(c.get("throughput_kind") or "single-stream").lower()
    return kind == "single-stream" and _count(c.get("concurrency")) <= 1


def _single_user(c: dict) -> bool:
    """One request at a time on one GPU: the number a person chatting with the model sees.
    Multi-GPU rows are a different measurement too."""
    return _one_stream(c) and _count(c.get("num_gpus")) <= 1


def _unrated(c: dict) -> bool:
    """TokenMark rates a row credibility "none" when nobody has checked it, except the runs
    TokenMark marks verified (trust "zach-verified": run by TokenMark itself, with logs)."""
    return (str(c.get("credibilityLabel") or "").lower() == "none"
            and str(c.get("trust") or "").lower() != "zach-verified")


def measured_for_model(model_name: str, hardware: Optional[str] = None, limit: int = 5,
                       hardware_ids=None) -> Optional[list]:
    """Real measured single-user rows for a specific model (to enrich a catalog entry with an
    actual tok/s + source), fastest first. `hardware_ids` matches TokenMark's hardware id
    exactly (a10 never matches a100); without it, `hardware` is a loose substring hint.
    Unrated rows are left out. None if TokenMark is unavailable or has nothing."""
    data = _get("/tracker-summary.json")
    rows = _rows(data)
    if not rows:
        return None
    key = (model_name or "").lower()
    rows = [c for c in rows if key in str(c.get("model") or "").lower()
            and _single_user(c) and not _unrated(c)]
    if hardware_ids:
        ids = {str(i) for i in hardware_ids}
        rows = [c for c in rows if c.get("hardware") in ids]
    elif hardware:
        h = hardware.lower()
        rows = [c for c in rows if h in f"{c.get('hardware','')} {c.get('hardwareLabel','')}".lower()]
    rows.sort(key=lambda c: c.get("decode_tps") if isinstance(c.get("decode_tps"), (int, float)) else 0,
              reverse=True)
    return rows[:limit] or None


def tracker_rows() -> Optional[tuple]:
    """(rows, tracked hardware ids) from TokenMark's full tracker summary, `configs` and
    `card_configs` together, or None when the site is unreachable or has no rows. One GET,
    the same file measured_for_model reads. `deneb setup --measured` picks from these rows."""
    rows = _rows(_get("/tracker-summary.json"))
    if not rows:
        return None
    return rows, sorted({c.get("hardware") for c in rows if isinstance(c.get("hardware"), str)})
