"""Shape deneb.api results into agent-facing JSON.

These functions are PURE: they read attributes off the dataclasses `deneb.api` already built
and return plain dicts. They do NO ranking, NO selection and NO network of their own - every
decision was made by the engine (recommend.py / setup_advisor.py / measured.py) that `api`
called, so the MCP gives the exact same answer as `deneb recommend` / `deneb setup`. The only
transform here is formatting: the CLI renders the same `api` data for a terminal, this renders
it for a tool caller. Keep it that way - any logic added here would be a second brain.
"""
from __future__ import annotations

from typing import Optional

from deneb import cli, setup_advisor
from deneb.measured import _s as _clean, _tps as _tpsv

_UNSET = object()  # "serving was not probed" - distinct from "probed, nothing running" (None)


def _mem_gb(profile) -> Optional[int]:
    mb = getattr(profile, "usable_mem_mb", None)
    return round(mb / 1024) if mb is not None else None


def _box(profile, serving=_UNSET) -> dict:
    """The box as the engine read it. A caller that probed for a running server (box/setup)
    passes `serving` - even None - and it is reported as a stable key, so the schema does not
    change with whether a server happens to be up. recommend does not probe, so it omits it."""
    out: dict = {
        "backend": getattr(profile, "primary_backend", None),
        "memory_gb": _mem_gb(profile),
    }
    if serving is not _UNSET:
        # _serving_label(None) == "no model server running"; same label the CLI header prints
        out["serving"] = cli._serving_label(serving)
    return out


def _measured_row(row: dict) -> dict:
    """One TokenMark row, cleaned exactly as the CLI cleans a network row before it is shown."""
    tps = _tpsv(row)
    return {
        "tok_per_s": round(tps, 2) if tps else None,
        "quant": _clean(row.get("quant"), 40) or None,
        "hardware": _clean(row.get("hardwareLabel") or row.get("hardware")) or None,
        "measured_by": _clean(row.get("author"), 60) or None,   # contributor label, not a URL
    }


def recommend_payload(data) -> dict:
    """A RecommendResult -> JSON. Attribute extraction only; the ranking is the engine's."""
    picks = []
    for i, r in enumerate(data.recs, 1):
        name = getattr(r.model, "name", "") or ""
        fit = r.fit
        need = setup_advisor.download_gib(r.quant)
        est = getattr(fit, "est_tps", None)
        pick = {
            "rank": i,
            "model": name,
            "quant": getattr(r.quant, "name", "") or None,
            "fits": bool(getattr(fit, "fits", False)),
            "tight": bool(getattr(fit, "tight", False)),
            "speed_tier": getattr(fit, "expected_speed_tier", "") or None,
            # est decode is a (low, high) tok/s range and needs the box bandwidth; off the
            # catalog it means nothing. Same source the CLI formats with speed.fmt_tps.
            "est_decode_tps": ({"low": est[0], "high": est[1], "unit": "tok/s"}
                               if (data.bw_gbps and est) else None),
            "download_gib": need or None,
            "download_fits_disk": (None if (not need or data.free_gb is None) else need <= data.free_gb),
            "why": getattr(r, "why", "") or None,
        }
        m = data.measured.get(name)
        if m:
            pick["measured"] = _measured_row(m)
        picks.append(pick)
    top = data.recs[0] if data.recs else None
    box = _box(data.profile)
    box["bandwidth_gbps"] = round(data.bw_gbps, 1) if data.bw_gbps else None
    box["free_disk_gb"] = data.free_gb
    return {
        "box": box,
        "use_case": data.use_case,
        "catalog_source": data.source,
        "measured_hardware": data.hw_hint,
        "picks": picks,
        "next": (f"deneb_setup model={getattr(top.model, 'name', '')}" if top else None),
        "note": ("Deterministic, keyless ranking of Deneb's curated catalog by model size x "
                 "quant quality for this use-case on this box - not a skill score, so check a "
                 "pick's published benchmarks before a big download. Deneb runs nothing."),
    }


def setup_payload(data, requested_name: str) -> dict:
    """A SetupResult -> JSON. The steps are the engine's; this only lays them out."""
    steps = []
    for i, s in enumerate(data.steps, 1):
        steps.append({
            "n": i,
            "title": getattr(s, "title", "") or None,
            "command": getattr(s, "command", "") or None,
            "what": getattr(s, "what", "") or None,
            "warnings": list(getattr(s, "warnings", None) or []),
            "kind": getattr(s, "kind", "") or None,
        })
    ollama = next((st["n"] for st in steps if st["kind"] == "ollama"), None)
    sel = data.selection
    choice = getattr(sel, "choice", None)
    engine = getattr(choice, "engine", "") or "llama.cpp"
    unit = any(st["kind"] == "service" for st in steps)   # no systemd: no unit step, run by hand
    manual = ("llama.cpp steps" if engine != "vllm" else
              "vLLM steps (served on 127.0.0.1:8001 by the vllm-server.service user unit)" if unit
              else "vLLM steps (served on 127.0.0.1:8001)")
    out = {
        "model": getattr(data.model, "name", requested_name),
        "box": _box(data.profile, data.serving),
        "engine": engine,
        "steps": steps,
        "ollama_step": ollama,
        "note": ("Deneb shows every command and runs NOTHING (the Deneb Rule): run each step "
                 "yourself, top to bottom, and verify it before running. When an Ollama step is "
                 f"present it is an alternative route to the manual {manual}, not an extra one."),
    }
    if sel is not None:
        out["measured"] = {
            # sel.lines are built from TokenMark rows; clean again before they leave the box
            "lines": [_clean(x, 300) for x in (sel.lines or [])],
            "reproduces": bool(choice),
            "engine": engine if choice else None,
        }
        if choice is not None and engine == "vllm":
            out["measured"]["vllm"] = {"max_model_len": getattr(choice, "ctx", None),
                                       "max_num_seqs": getattr(choice, "max_seqs", None),
                                       "gpu_memory_utilization": getattr(choice, "util", None)}
    return out


def box_payload(profile, serving) -> dict:
    """hardware.profile_hardware() + check.serving_now() -> JSON. Shaping only; the read-only
    probes already ran in the caller."""
    out = _box(profile, serving)
    if serving:
        port, kind, model = serving
        out["serving_detail"] = {"port": port, "kind": kind, "model": model or None}
    return out
