"""deneb.matrix — "what runs on which box", for boxes Deneb has never seen.

`deneb recommend` answers for the machine it runs on. The Deneb engine also has to answer
"what should I run on an RTX 5080?" or "is an M4 Max enough?" without that machine in
front of it. This module turns a hardware_catalog.json device into the same
HardwareProfile shape detect_profile() produces, then runs the normal recommend() on it,
so the engine and the CLI give the same answer for the same box.

Pure: no I/O, no network. The budget rule mirrors hardware.py exactly (dedicated VRAM =
the card's memory; unified pool = RAM minus max(8 GB, 15%)), so a catalog answer and a
live-profile answer agree.
"""
from __future__ import annotations

from typing import Optional

from .hardware import (GPUInfo, HardwareProfile, _UNIFIED_HEADROOM_FLOOR_MB,
                       _UNIFIED_HEADROOM_FRACTION)
from .recommend import recommend

# Which llama.cpp backend a vendor's GPU runs on. The backend string only feeds the
# params-based fallback tier and the CPU-only check; the estimate uses bandwidth.
_BACKEND = {"nvidia": "cuda", "amd": "rocm", "apple": "metal", "intel": "sycl"}


def device_budget_mb(device: dict) -> Optional[int]:
    """Usable model memory for a catalog device, same rule as hardware.classify_memory."""
    try:
        mem_mb = int(float(device["mem_gb"]) * 1024)
    except (KeyError, TypeError, ValueError):
        return None
    if mem_mb <= 0:
        return None
    if device.get("class") == "unified-memory-box":
        headroom = max(_UNIFIED_HEADROOM_FLOOR_MB, int(_UNIFIED_HEADROOM_FRACTION * mem_mb))
        return max(0, mem_mb - headroom)
    return mem_mb


def device_profile(device: dict) -> Optional[HardwareProfile]:
    """A synthetic HardwareProfile for a catalog device, or None when its memory is unknown."""
    budget = device_budget_mb(device)
    if budget is None:
        return None
    unified = device.get("class") == "unified-memory-box"
    mem_mb = int(float(device["mem_gb"]) * 1024)
    backend = _BACKEND.get(device.get("vendor") or "", "vulkan")
    gpu = GPUInfo(vendor=device.get("vendor") or "", name=device.get("name") or "",
                  backend=backend, vram_mb=None if unified else mem_mb,
                  unified_mem_mb=mem_mb if unified else None,
                  memory_kind="unified" if unified else "dedicated")
    return HardwareProfile(cpu_model="", ram_total_mb=mem_mb if unified else None,
                           gpus=[gpu], primary_backend=backend, usable_mem_mb=budget)


def picks_for_device(device: dict, catalog: list, use_case: str = "general",
                     top_n: int = 3) -> list:
    """recommend() for a catalog device: [] when the device has no usable memory figure.
    Only fitting picks are returned (recommend's nothing-fits placeholder is dropped, the
    caller says "nothing in the catalog fits" itself)."""
    prof = device_profile(device)
    if prof is None:
        return []
    bw = device.get("bw_gbps")
    recs = recommend(prof, use_case, top_n=top_n, catalog=catalog,
                     bw_gbps=float(bw) if bw else None)
    return [r for r in recs if getattr(r.fit, "fits", False)]
