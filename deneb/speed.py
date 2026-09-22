"""deneb.speed — bandwidth-based decode-speed estimate + hardware catalog lookup.

Single-stream decode is memory-bound: every generated token reads the ACTIVE weights
once, so

    seconds per token  ~=  GB read per token / (efficiency x bandwidth GB/s) + overhead
    GB read per token  ~=  active params (B) x bits-per-weight / 8

efficiency is 0.55-0.75 on real engines (kernel overhead, KV reads, sampling). This is
a first-order physical bound, not a benchmark: it is how Deneb says "about 40 tok/s"
for a box nobody has measured yet. TokenMark numbers, when present, beat it.

Pure logic (estimate, match) takes plain values and is unit-tested. The one I/O helper,
load_devices(), reads the bundled deneb/data/hardware_catalog.json (package data).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

# Fraction of peak bandwidth real engines reach on single-stream decode. Fitted to
# TokenMark measurements (see tests/test_speed.py MEASURED): dense models sit at
# 0.55-0.75; MoE models reach less (small scattered expert reads, router, shared
# layers), so they get a lower, wider band.
EFFICIENCY = {"dense": (0.55, 0.75), "moe": (0.40, 0.70)}
# Fixed per-token cost (ms) on top of the weight reads: kernel launches, sampling, and for
# MoE the router + many small expert reads. Negligible on a 256 GB/s box, dominant for a
# 3B-active MoE on a 1 TB/s GPU (without it a 4090 "does" 350 tok/s on a 30B-A3B; real
# llama.cpp runs land around 130-190). (best case, worst case).
OVERHEAD_MS = {"dense": (0.5, 1.0), "moe": (2.0, 3.5)}
EFFICIENCY_LOW, EFFICIENCY_HIGH = EFFICIENCY["dense"]

_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "hardware_catalog.json")


def load_devices(path: str = _CATALOG_PATH) -> list:
    """The bundled hardware catalog's device list. [] if unreadable (callers then fall
    back to the params-based speed tier, and say so)."""
    try:
        with open(path) as f:
            return list(json.load(f).get("devices") or [])
    except (OSError, ValueError):
        return []


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]", " ", (text or "").lower())).strip()


def match_device(names, devices: list) -> Optional[dict]:
    """First catalog device whose `match` substrings hit any of `names` (GPU name, CPU
    model, gfx id) and whose `exclude` substrings hit none. None when unknown, which is
    an honest "not in the catalog", never a guess."""
    hay = [_norm(n) for n in (names or []) if n]
    if not hay:
        return None
    for dev in devices or []:
        excl = [_norm(x) for x in dev.get("exclude") or []]
        for pat in dev.get("match") or []:
            p = _norm(pat)
            for h in hay:
                if p and p in h and not any(x and x in h for x in excl):
                    return dev
    return None


def profile_names(profile) -> list:
    """Every identifying string on a HardwareProfile: GPU names, gfx ids, CPU model."""
    out = []
    for g in getattr(profile, "gpus", None) or []:
        out.append(getattr(g, "name", "") or "")
        extra = getattr(g, "extra", None)
        if isinstance(extra, dict) and extra.get("gfx"):
            out.append(str(extra["gfx"]))
    out.append(getattr(profile, "cpu_model", "") or "")
    return [n for n in out if n]


def device_for_profile(profile, devices: Optional[list] = None) -> Optional[dict]:
    """The catalog device this box is, or None."""
    return match_device(profile_names(profile), load_devices() if devices is None else devices)


def gb_per_token(active_params_b: float, bpw: float) -> float:
    """GB of weights read per generated token."""
    return float(active_params_b) * float(bpw) / 8.0


def est_decode_tps(active_params_b, bpw, bw_gbps, moe: bool = False) -> Optional[tuple]:
    """(low, high) single-stream decode tok/s WITHOUT speculative decoding, or None when
    any input is missing/invalid. MTP / draft-model speculation typically adds 1.5-2.5x
    on top. Rounded to whole tok/s (1 decimal under 10) so the output never looks more
    precise than the model behind it."""
    try:
        gpt = gb_per_token(active_params_b, bpw)
        bw = float(bw_gbps)
    except (TypeError, ValueError):
        return None
    if gpt <= 0 or bw <= 0:
        return None

    def _r(x):
        return round(x, 1) if x < 10 else round(x)

    kind = "moe" if moe else "dense"
    eff_lo, eff_hi = EFFICIENCY[kind]
    oh_best, oh_worst = OVERHEAD_MS[kind]
    lo = 1000.0 / (gpt / (eff_lo * bw) * 1000.0 + oh_worst)
    hi = 1000.0 / (gpt / (eff_hi * bw) * 1000.0 + oh_best)
    return (_r(lo), _r(hi))


def tier_for_tps(tps_high) -> str:
    """Map an estimated tok/s to the fit module's TIER_ORDER vocabulary."""
    if tps_high is None:
        return ""
    if tps_high >= 40:
        return "fast"
    if tps_high >= 15:
        return "medium"
    if tps_high >= 6:
        return "slow"
    return "very-slow"


def fmt_tps(rng) -> str:
    if not rng:
        return "?"
    lo, hi = rng
    return f"~{lo}-{hi} tok/s"


def usable_bandwidth(profile, device: Optional[dict]) -> Optional[float]:
    """The bandwidth the model will actually read at. A discrete GPU's VRAM bandwidth only
    applies when the GPU backend is in use; on a CPU-only run the weights sit in system
    RAM, which we cannot size, so None. Unified-memory boxes read the same pool either way."""
    if not device or not device.get("bw_gbps"):
        return None
    backend = (getattr(profile, "primary_backend", "") or "").lower()
    if backend == "cpu" and device.get("class") != "unified-memory-box":
        return None
    return float(device["bw_gbps"])
