"""deneb.hardware — the PURE, deterministic core of hardware profiling.

HW-01/HW-03/HW-04 contract. This module holds the DATA MODEL (HardwareProfile /
GPUInfo), one PURE parser per vendor tool (NVIDIA / AMD / Apple) plus CPU/RAM/PCI
parsers, and the unified-memory + usable-memory CLASSIFICATION logic. It performs
NO I/O and imports no tool runner: every function here takes an already-captured
tool-output STRING and returns structured data.

INVOCATION-vs-PARSING split (the HW-04 / QA-01 testability requirement): the live
probe layer — which actually shells out to nvidia-smi / rocm-smi / system_profiler
via the allowlisted `tools` runner and assembles a full HardwareProfile — lands in
Plan 02. It mirrors check.py's shape (a thin `_probe()` feeding pure verdict
functions). Keeping PARSING here, I/O-free, lets us unit-test every hardware fact
against captured fixtures with no live tool present.

Defensiveness is a hard requirement (T-01-01): external tool stdout is untrusted
text; a malformed / empty / refused blob must degrade to an empty/partial result
and NEVER raise. The unified-memory trap (T-01-02) is the other hard requirement:
this box's rocm-smi reports a 1 GiB BIOS VRAM carve-out while actually carrying
~124 GiB of shared system RAM — classify_memory must budget the real pool, not the
carve-out.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ── Classification constants (HW-03) ─────────────────────────────────────────
# AMD GFX versions that are unified-memory APUs (iGPU sharing system RAM), NOT
# discrete cards with their own VRAM. Strix Halo (gfx1151) is THIS box.
_UNIFIED_AMD_GFX = {
    "gfx1151",  # Strix Halo (AMD Ryzen AI Max) — this box
    "gfx1150",  # Strix Point
    "gfx1103",  # Phoenix / Hawk Point APU
    "gfx1036",  # Raphael / common desktop APU iGPU
}
# A carve-out smaller than this, on a box with far more system RAM, is a BIOS
# reservation on a unified box — not a real dedicated-VRAM budget.
_SMALL_CARVEOUT_MB = 4096
# Usable-memory headroom on a UNIFIED box. Coarse, Phase-2-refined: the fit math in
# Phase 2 replaces this with a real per-model calculation. The floor exists because
# of the "<80GB-free Strix freeze" scar — a unified box that hands its entire RAM to
# the model leaves nothing for the OS/compositor and locks up. Reserve the larger of
# a fixed floor and a percentage.
_UNIFIED_HEADROOM_FLOOR_MB = 8192
_UNIFIED_HEADROOM_FRACTION = 0.15

# PCI vendor ids for the lspci fallback path (when a vendor SMI tool is absent).
_PCI_VENDOR = {"1002": "amd", "10de": "nvidia", "8086": "intel"}
_MIB = 1024 * 1024


@dataclass
class GPUInfo:
    """One GPU as read from a vendor tool. All fields optional so a partial/degraded
    record is always constructible.

    backend     ∈ {"cuda", "rocm", "metal", "cpu", ""}
    memory_kind ∈ {"dedicated", "unified", "unknown"}
    vram_mb          — real dedicated VRAM (discrete cards).
    vram_carveout_mb — the (often tiny) BIOS VRAM reservation on a unified APU.
    unified_mem_mb   — the shared system-RAM budget on a unified box (set by
                       classify_memory, NOT the carve-out).
    extra            — vendor-specific raw fields (gfx version, sku, pci id, …).
    """
    vendor: str = ""
    name: str = ""
    backend: str = ""
    vram_mb: Optional[int] = None
    unified_mem_mb: Optional[int] = None
    vram_carveout_mb: Optional[int] = None
    driver: Optional[str] = None
    memory_kind: str = "unknown"
    extra: dict = field(default_factory=dict)


@dataclass
class HardwareProfile:
    """The whole box. Defaults make a fully-degraded ("we could read almost nothing")
    profile constructible, per PLAT-03. Plan 02's live layer fills this in."""
    os: str = ""
    arch: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    ram_total_mb: Optional[int] = None
    disk_free_mb: Optional[int] = None
    gpus: list = field(default_factory=list)
    primary_backend: str = "cpu"
    usable_mem_mb: Optional[int] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain dict (nested GPUInfo → dict) for the Plan 02 printer / JSON output."""
        return dataclasses.asdict(self)


@dataclass
class MemoryClassification:
    """Result of classify_memory: what KIND of memory this GPU has and how much of it
    is actually usable as a model budget."""
    memory_kind: str = "unknown"          # dedicated | unified | unknown
    unified_mem_mb: Optional[int] = None  # shared system-RAM budget (unified only)
    usable_mem_mb: int = 0                # the budget the fit math should use
    note: str = ""                        # human-readable why


# ── PURE parsers (stubs — Task 2 fills these test-first) ─────────────────────
def parse_nvidia_smi(text: str) -> list:
    raise NotImplementedError


def parse_rocm_smi(text: str) -> list:
    raise NotImplementedError


def parse_system_profiler(text: str) -> list:
    raise NotImplementedError


def parse_sysctl_memsize(text: str) -> int:
    raise NotImplementedError


def parse_cpuinfo(text: str) -> dict:
    raise NotImplementedError


def parse_meminfo(text: str) -> int:
    raise NotImplementedError


def parse_lspci_gpus(text: str) -> list:
    raise NotImplementedError


def classify_memory(gpu: GPUInfo, ram_total_mb: Optional[int]) -> MemoryClassification:
    raise NotImplementedError
