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


# ── helpers ──────────────────────────────────────────────────────────────────
def _to_int(v) -> Optional[int]:
    """Best-effort str/number -> int; None on anything unparseable (never raises)."""
    try:
        return int(str(v).strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _is_sentinel(text: str) -> bool:
    """True for empty / whitespace / a tools.py refusal or 'not installed' sentinel —
    i.e. there is no real tool output to parse."""
    s = (text or "").strip()
    return (not s) or s.startswith("[refused]") or "not installed on this box" in s


# ── parse_nvidia_smi ─────────────────────────────────────────────────────────
def parse_nvidia_smi(text: str) -> list:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`.
    One GPU per non-blank line: name, VRAM (strip the ' MiB' unit), driver. Returns []
    on empty / refused / not-installed / garbage input."""
    if _is_sentinel(text):
        return []
    gpus = []
    try:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[0]
            m = re.search(r"(\d+)", parts[1])  # "24564 MiB" -> 24564
            if not name or not m:
                continue
            driver = parts[2] or None
            gpus.append(GPUInfo(vendor="nvidia", name=name, backend="cuda",
                                vram_mb=int(m.group(1)), driver=driver,
                                memory_kind="dedicated"))
    except Exception:  # noqa: BLE001 — untrusted tool output must never crash the profile
        return []
    return gpus


# ── parse_rocm_smi ───────────────────────────────────────────────────────────
def _rocm_card_to_gpu(vram_b: Optional[int], series: str, sku: str, gfx: str) -> GPUInfo:
    """Build a GPUInfo from one rocm-smi card's raw fields. The reported VRAM is routed
    to vram_carveout_mb on a unified APU (gfx in the APU set OR an STRX SKU) and to
    vram_mb on a discrete card — classify_memory makes the final unified/dedicated call."""
    gfx = (gfx or "").strip()
    sku = (sku or "").strip()
    total_mb = (vram_b // _MIB) if vram_b else None
    unified_ish = gfx.lower() in _UNIFIED_AMD_GFX or sku.upper().startswith("STRX")
    g = GPUInfo(vendor="amd", name=(series or "").strip(), backend="rocm",
                extra={"gfx": gfx, "sku": sku})
    if total_mb is not None:
        if unified_ish:
            g.vram_carveout_mb = total_mb
        else:
            g.vram_mb = total_mb
    return g


_ROCM_CARD_KEYS = ("GFX Version", "Card Series", "VRAM Total Memory (B)", "Card SKU")


def _parse_rocm_text(text: str) -> list:
    """Text-table fallback: group lines by their leading GPU[N] token, extract fields."""
    blocks: dict = {}
    for line in text.splitlines():
        m = re.match(r"\s*GPU\[(\d+)\]", line)
        if m:
            blocks.setdefault(m.group(1), []).append(line)
    gpus = []
    for idx in sorted(blocks, key=int):
        b = "\n".join(blocks[idx])
        vram = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", b)
        series = re.search(r"Card Series:\s*(.+)", b)
        sku = re.search(r"Card SKU:\s*(\S+)", b)
        gfx = re.search(r"GFX Version:\s*(\S+)", b)
        if not any([vram, series, sku, gfx]):
            continue
        gpus.append(_rocm_card_to_gpu(
            int(vram.group(1)) if vram else None,
            series.group(1) if series else "",
            sku.group(1) if sku else "",
            gfx.group(1) if gfx else "",
        ))
    return gpus


def parse_rocm_smi(text: str) -> list:
    """Parse rocm-smi output — the `--json` form first (tolerating a leading banner by
    seeking the first '{'), falling back to the text table. Returns [] on any failure."""
    if _is_sentinel(text):
        return []
    brace = text.find("{")
    if brace != -1:
        try:
            data = json.loads(text[brace:])
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            gpus = []
            for card in data.values():
                if not isinstance(card, dict) or not any(k in card for k in _ROCM_CARD_KEYS):
                    continue
                gpus.append(_rocm_card_to_gpu(
                    _to_int(card.get("VRAM Total Memory (B)")),
                    card.get("Card Series", ""),
                    card.get("Card SKU", ""),
                    card.get("GFX Version", ""),
                ))
            if gpus:
                return gpus
    try:
        return _parse_rocm_text(text)
    except Exception:  # noqa: BLE001
        return []


# ── parse_system_profiler (Apple / Metal) ────────────────────────────────────
def parse_system_profiler(text: str) -> list:
    """Parse `system_profiler SPDisplaysDataType` — one GPU per 'Chipset Model:' line.
    Apple GPUs are unified-memory (the actual budget comes from sysctl hw.memsize)."""
    if _is_sentinel(text):
        return []
    gpus = []
    try:
        for m in re.finditer(r"Chipset Model:\s*(.+)", text):
            name = m.group(1).strip()
            if name:
                gpus.append(GPUInfo(vendor="apple", name=name, backend="metal"))
    except Exception:  # noqa: BLE001
        return []
    return gpus


# ── parse_sysctl_memsize ─────────────────────────────────────────────────────
def parse_sysctl_memsize(text: str) -> int:
    """`sysctl hw.memsize` -> total RAM in MB (unified budget on Apple). 0 on junk."""
    try:
        m = re.search(r"hw\.memsize:\s*(\d+)", text or "")
        return (int(m.group(1)) // _MIB) if m else 0
    except Exception:  # noqa: BLE001
        return 0


# ── parse_cpuinfo ────────────────────────────────────────────────────────────
def parse_cpuinfo(text: str) -> dict:
    """/proc/cpuinfo -> {model, cores}. cores = count of 'processor' lines (the
    logical/nproc count, NOT 'cpu cores'); model = first 'model name'. {'','0'} on junk."""
    model, cores = "", 0
    try:
        for line in (text or "").splitlines():
            if re.match(r"\s*processor\s*:", line):
                cores += 1
            elif not model:
                m = re.match(r"\s*model name\s*:\s*(.+)", line)
                if m:
                    model = m.group(1).strip()
    except Exception:  # noqa: BLE001
        return {"model": "", "cores": 0}
    return {"model": model, "cores": cores}


# ── parse_meminfo ────────────────────────────────────────────────────────────
def parse_meminfo(text: str) -> int:
    """/proc/meminfo MemTotal (kB) -> total RAM in MB. 0 on junk."""
    try:
        m = re.search(r"MemTotal:\s+(\d+)\s*kB", text or "")
        return (int(m.group(1)) // 1024) if m else 0
    except Exception:  # noqa: BLE001
        return 0


# ── parse_lspci_gpus (fallback when a vendor SMI tool is absent) ──────────────
def parse_lspci_gpus(text: str) -> list:
    """`lspci -nn` GPU lines -> named GPUs by PCI vendor id (1002 AMD / 10de NVIDIA /
    8086 Intel), so the profile can still say 'GPU present, SMI tool not installed'."""
    if _is_sentinel(text):
        return []
    gpus = []
    try:
        for line in text.splitlines():
            if not re.search(r"(VGA compatible controller|3D controller|Display controller)", line):
                continue
            m = re.search(r"\[([0-9a-fA-F]{4}):[0-9a-fA-F]{4}\]", line)
            vendor = _PCI_VENDOR.get(m.group(1).lower(), "") if m else ""
            after = line.split("]:", 1)[1].strip() if "]:" in line else line.strip()
            name = re.sub(r"\s*\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\].*$", "", after).strip()
            backend = {"amd": "rocm", "nvidia": "cuda"}.get(vendor, "")
            gpus.append(GPUInfo(vendor=vendor, name=name, backend=backend,
                                extra={"pci": m.group(0) if m else ""}))
    except Exception:  # noqa: BLE001
        return []
    return gpus


# ── classify_memory (HW-03) ──────────────────────────────────────────────────
def classify_memory(gpu: GPUInfo, ram_total_mb: Optional[int]) -> MemoryClassification:
    """Decide unified vs dedicated and compute the usable memory budget.

    Rules (coarse; Phase-2 fit math refines the exact numbers):
      - Apple  -> always unified.
      - AMD    -> unified when the GFX version is a known APU, OR the SKU starts 'STRX',
                  OR the VRAM carve-out is implausibly small vs system RAM (a BIOS
                  reservation on a shared-memory box). Otherwise dedicated.
      - else   -> dedicated.
    Unified budget = system RAM minus a headroom margin (max of a fixed floor and 15%) —
    the floor exists because of the '<80GB-free Strix freeze' scar: hand the model the
    whole pool and the box locks up. Dedicated budget = the GPU's own VRAM.
    """
    try:
        vendor = (gpu.vendor or "").lower()
        gfx = str(gpu.extra.get("gfx", "")).lower() if gpu.extra else ""
        sku = str(gpu.extra.get("sku", "")).upper() if gpu.extra else ""
        carve = gpu.vram_carveout_mb

        unified = False
        if vendor == "apple":
            unified = True
        elif vendor == "amd":
            if gfx in _UNIFIED_AMD_GFX or sku.startswith("STRX"):
                unified = True
            elif (carve is not None and carve < _SMALL_CARVEOUT_MB
                  and ram_total_mb and ram_total_mb > 2 * carve):
                unified = True

        if unified:
            gpu.memory_kind = "unified"
            if not ram_total_mb or ram_total_mb <= 0:
                # Known unified, but system RAM was unreadable — be honest, don't guess.
                res = MemoryClassification(
                    memory_kind="unified", unified_mem_mb=None, usable_mem_mb=0,
                    note="unified-memory GPU; system RAM unreadable — usable budget unknown")
                gpu.unified_mem_mb = None
                return res
            headroom = max(_UNIFIED_HEADROOM_FLOOR_MB, int(_UNIFIED_HEADROOM_FRACTION * ram_total_mb))
            usable = max(0, ram_total_mb - headroom)
            gpu.unified_mem_mb = ram_total_mb
            return MemoryClassification(
                memory_kind="unified", unified_mem_mb=ram_total_mb, usable_mem_mb=usable,
                note=(f"unified memory: the GPU shares the ~{ram_total_mb} MB system-RAM pool "
                      f"(NOT the small VRAM carve-out); reserved {headroom} MB headroom "
                      f"(coarse, Phase-2-refined) so the box does not freeze."))

        # dedicated
        gpu.memory_kind = "dedicated"
        vram = gpu.vram_mb
        return MemoryClassification(
            memory_kind="dedicated", unified_mem_mb=None,
            usable_mem_mb=(vram if vram else 0),
            note=(f"dedicated VRAM: the GPU's own {vram} MB is the budget."
                  if vram else "dedicated GPU; VRAM size unknown."))
    except Exception:  # noqa: BLE001 — classification must never crash the profile
        return MemoryClassification(memory_kind="unknown", usable_mem_mb=0,
                                    note="could not classify memory")
