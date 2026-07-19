"""deneb.fit — the PURE fit-math + speed-tier heuristic + caveat rules (FIT-02, FIT-02b).

fits(model, quant, profile) -> FitResult answers "will THIS model at THIS quant run on
THIS box?" deterministically, from a Phase-1 HardwareProfile. This is a PURE layer:
str/struct in, struct out — NO I/O, NO network, NO LLM, no tools import — so every fit
fact is unit-testable against constructed profiles (QA-01). recommend() (Plan 02) filters
the catalog with this predicate and ranks over its FitResults.

Two hard requirements:
  - HONEST DEGRADATION (T-02-01): the profile is Phase-1-derived and may be degraded —
    usable_mem_mb None (memory unreadable) or gpus []. fits() treats None as "unknown
    budget" -> fits False + a caveat, and its whole body is guarded so malformed input
    degrades rather than raises (PLAT-03).
  - NO DOUBLE-COUNTING (see <fit_math_spec>): profile.usable_mem_mb ALREADY has the OS /
    unified-pool headroom subtracted in Phase 1. The RUNTIME_RESERVE_MB added here is a
    SEPARATE, coarse KV/context/compute allowance on top of the raw weights — not a second
    OS margin.

All the numeric constants are COARSE first-pass heuristics (a later phase refines them
with real measurement); each is named and documented so the guess is visible, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hardware import GPUInfo, HardwareProfile  # noqa: F401  (typing / interface anchor)
from .models_catalog import Model, Quant  # noqa: F401  (typing / interface anchor)

# ── speed-tier ladder (worst -> best); index math clamps into this list ───────
TIER_ORDER = ["very-slow", "slow", "medium", "fast"]

# AMD GFX versions with the known Strix-Halo/gfx1151 ROCm-kernel caveat. Kept local to
# the caveat layer (the hardware module owns the broader unified-APU set).
_CAVEAT_GFX = {"gfx1151", "gfx1150"}

# The CPU-only speed penalty: no accelerator -> drop the tier this many steps (floored).
_CPU_TIER_PENALTY = 2

# tight-fit threshold: warn when spare memory is below the larger of a fixed floor and
# 10% of the need. Coarse — a later phase ties it to real context/KV growth.
_TIGHT_FLOOR_MB = 2000
_TIGHT_FRACTION = 0.10


@dataclass
class FitResult:
    """The FIT-02 contract {fits, headroom_mb, expected_speed_tier, caveats} plus need_mb
    (extra context Plan 02's ranking / why-strings use). headroom_mb is 0 when the budget
    was unreadable; it can be negative when the model overflows the budget."""
    fits: bool
    headroom_mb: int
    expected_speed_tier: str
    caveats: list = field(default_factory=list)
    need_mb: int = 0


def RUNTIME_RESERVE_MB(size_mb: int) -> int:
    """Coarse KV-cache / context / compute allowance ON TOP of the raw GGUF weights:
    max(512 MB, 10% of the weight size). Separate from the profile's OS headroom (already
    in usable_mem_mb) — no double counting. Refined by a later phase."""
    try:
        return max(512, int(0.10 * int(size_mb)))
    except (TypeError, ValueError):
        return 512


def _effective_params(model) -> float:
    """The params that actually gate tok/s: active experts for a MoE, total for a dense
    model. This is why a 35B-A3B (3B active) runs at ~3B speed, not 35B."""
    try:
        arch = (getattr(model, "architecture", "") or "").lower()
        active = getattr(model, "active_params_b", None)
        if arch == "moe" and active:
            return float(active)
        return float(getattr(model, "params_b", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _base_tier_index(effective_params: float) -> int:
    """effective_params -> TIER_ORDER index (before any backend adjustment)."""
    if effective_params <= 8:
        return 3   # fast
    if effective_params <= 20:
        return 2   # medium
    if effective_params <= 40:
        return 1   # slow
    return 0       # very-slow


def _speed_tier(model, backend: str) -> str:
    """Label the expected speed from effective params + backend. cuda/rocm/metal are all
    'accelerated' and share a tier at this coarse granularity (ROCm-specific risk is
    surfaced via caveats, not the tier); cpu-only drops the tier by _CPU_TIER_PENALTY."""
    idx = _base_tier_index(_effective_params(model))
    if (backend or "").lower() == "cpu":
        idx = max(0, idx - _CPU_TIER_PENALTY)
    return TIER_ORDER[idx]


def _primary_gpu(profile):
    """The GPU whose backend matches primary_backend (fall back to the first GPU, else
    None). Used for the unified-pool signal. Never raises."""
    gpus = getattr(profile, "gpus", None) or []
    backend = (getattr(profile, "primary_backend", "") or "").lower()
    for g in gpus:
        if (getattr(g, "backend", "") or "").lower() == backend:
            return g
    return gpus[0] if gpus else None


def _any_caveat_gfx(profile) -> str:
    """Return the first caveat-flagged gfx version found on ANY GPU, else '' (per the
    FIT-02b rule 'if any gpu.extra.get(gfx) in {gfx1151, gfx1150}')."""
    for g in (getattr(profile, "gpus", None) or []):
        extra = getattr(g, "extra", None)
        if isinstance(extra, dict):
            gfx = str(extra.get("gfx", "")).lower()
            if gfx in _CAVEAT_GFX:
                return gfx
    return ""


def _caveats(model, quant, profile, fits_ok: bool, headroom_mb: int,
             need_mb: int, budget_unknown: bool) -> list:
    """Collect the FIT-02b caveats that apply, in the fixed deterministic order:
    gfx1151 -> unified pool -> tight fit -> cpu-only -> unreadable budget."""
    caveats: list = []
    backend = (getattr(profile, "primary_backend", "") or "").lower()

    # 1. gfx1151 / ROCm kernel note (Strix Halo instability on older kernels).
    gfx = _any_caveat_gfx(profile)
    if gfx or (backend == "rocm" and _any_caveat_gfx(profile)):
        caveats.append(
            f"{gfx or 'gfx1151'} (Strix Halo): needs a recent kernel with the ROCm "
            "fixes — older kernels have known gfx1151 instability.")

    # 2. unified shared pool (primary GPU shares one pool with the OS/apps).
    gpu = _primary_gpu(profile)
    if gpu is not None and (getattr(gpu, "memory_kind", "") or "").lower() == "unified":
        caveats.append(
            "unified memory: the model shares the single memory pool with the OS and "
            "other apps — close other heavy apps before loading.")

    # 3. tight fit (fits, but little spare — reduced context / OOM risk under load).
    if fits_ok:
        threshold = max(_TIGHT_FLOOR_MB, int(_TIGHT_FRACTION * need_mb))
        if headroom_mb < threshold:
            caveats.append(
                f"tight fit: only ~{headroom_mb} MB spare — expect reduced context "
                "length and possible OOM under load.")

    # 4. cpu-only (no accelerator -> low tok/s).
    if backend == "cpu":
        caveats.append(
            "CPU-only: expect low tok/s — prefer a smaller model for interactive use.")

    # 5. unreadable budget (honest degradation — could not compute fit).
    if budget_unknown:
        caveats.append(
            "the box's memory budget was unreadable, so fit could not be computed — "
            "treating as does-not-fit.")

    return caveats


def fits(model, quant, profile) -> FitResult:
    """Does `model` at `quant` fit `profile`'s usable memory?

    need_mb = quant.size_mb + RUNTIME_RESERVE_MB(quant.size_mb).
    headroom_mb = usable_mem_mb - need_mb (only when usable_mem_mb is a real int).
    fits = usable_mem_mb is not None AND headroom_mb >= 0.
    expected_speed_tier from _speed_tier; caveats from _caveats.

    Degrades honestly and NEVER raises (T-02-01): a None budget -> fits False + the
    unreadable-budget caveat; any unexpected malformed input -> a safe not-fits result."""
    try:
        try:
            size_mb = int(getattr(quant, "size_mb", 0) or 0)
        except (TypeError, ValueError):
            size_mb = 0
        need_mb = size_mb + RUNTIME_RESERVE_MB(size_mb)

        backend = getattr(profile, "primary_backend", "cpu") or "cpu"
        tier = _speed_tier(model, backend)

        usable = getattr(profile, "usable_mem_mb", None)
        usable_int: Optional[int]
        try:
            usable_int = int(usable) if usable is not None else None
        except (TypeError, ValueError):
            usable_int = None

        if usable_int is None:
            # Unknown budget: cannot compute fit -> False, headroom 0, honest caveat.
            caveats = _caveats(model, quant, profile, False, 0, need_mb,
                               budget_unknown=True)
            return FitResult(fits=False, headroom_mb=0, expected_speed_tier=tier,
                             caveats=caveats, need_mb=need_mb)

        headroom_mb = usable_int - need_mb
        fits_ok = headroom_mb >= 0
        caveats = _caveats(model, quant, profile, fits_ok, headroom_mb, need_mb,
                           budget_unknown=False)
        return FitResult(fits=fits_ok, headroom_mb=headroom_mb,
                         expected_speed_tier=tier, caveats=caveats, need_mb=need_mb)
    except Exception:  # noqa: BLE001 — the fit math must never crash the caller (PLAT-03)
        return FitResult(fits=False, headroom_mb=0, expected_speed_tier="very-slow",
                         caveats=["fit could not be computed for this model/box."],
                         need_mb=0)
