"""Unit tests for deneb.fit — the pure fit-math + speed tier + caveat rules (FIT-02, FIT-02b).

fits(model, quant, profile) answers "will THIS model at THIS quant run on THIS box?"
deterministically from a Phase-1 HardwareProfile — no I/O, no LLM. Profiles are
CONSTRUCTED directly (QA-01 style), so every fit fact is pinned against a known box.

Load-bearing cases (the plan's canonical set + every caveat rule + graceful degradation):
  - 7B-Q4 fits a 16GB box; 70B-Q4 does NOT; 70B-Q4 DOES fit a 128GB unified box;
  - a 35B-A3B MoE is "fast" (low active params) even though 35B total is large;
  - each caveat rule (gfx1151, unified pool, tight, cpu-only, unreadable budget) fires on
    its trigger;
  - a None / empty-gpu profile degrades to a valid FitResult and NEVER raises (T-02-01).

Run: python3 tests/test_fit.py    (also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb.fit import FitResult, fits  # noqa: E402
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from deneb.models_catalog import CATALOG  # noqa: E402


# ── helpers: pull a specific model + quant from the real catalog ──────────────
def model(name: str):
    for m in CATALOG:
        if m.name == name:
            return m
    raise KeyError(name)


def quant(m, qname: str):
    for q in m.quants:
        if q.name == qname:
            return q
    raise KeyError(qname)


# ── constructed test boxes (see the plan's <interfaces>) ──────────────────────
def cuda_box(usable):
    return HardwareProfile(
        primary_backend="cuda", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=usable,
                      memory_kind="dedicated")])


def strix_unified_box(usable=107000):
    return HardwareProfile(
        primary_backend="rocm", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="amd", backend="rocm", memory_kind="unified",
                      unified_mem_mb=126908,
                      extra={"gfx": "gfx1151", "sku": "STRXLGEN"})])


def cpu_box(usable=120000):
    return HardwareProfile(primary_backend="cpu", usable_mem_mb=usable, gpus=[])


# ── CANONICAL fit / no-fit ────────────────────────────────────────────────────
def test_canonical_7b_q4_fits_16gb():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(16000))
    assert r.fits is True
    assert r.headroom_mb > 0


def test_canonical_70b_q4_does_not_fit_16gb():
    m = model("Llama-3.3-70B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(16000))
    assert r.fits is False
    assert r.headroom_mb < 0


def test_canonical_70b_q4_fits_128gb_unified():
    m = model("Llama-3.3-70B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), strix_unified_box(107000))
    assert r.fits is True


def test_canonical_moe_35b_a3b_is_fast():
    m = model("Qwen3.6-35B-A3B")
    r = fits(m, quant(m, "Q4_K_M"), strix_unified_box(107000))
    assert r.expected_speed_tier == "fast"     # effective params = 3, not 35


# ── speed tiers ───────────────────────────────────────────────────────────────
def test_speed_tier_70b_dense_is_very_slow():
    m = model("Llama-3.3-70B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(200000))
    assert r.expected_speed_tier == "very-slow"


def test_speed_tier_14b_dense_is_medium():
    m = model("Qwen2.5-Coder-14B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(200000))
    assert r.expected_speed_tier == "medium"


def test_speed_tier_7b_dense_is_fast_on_gpu():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(200000))
    assert r.expected_speed_tier == "fast"


def test_speed_tier_7b_on_cpu_is_downgraded():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cpu_box(120000))
    assert r.expected_speed_tier != "fast"     # cpu downgrades by 2 -> "slow"
    assert r.expected_speed_tier == "slow"


# ── caveat rules (FIT-02b) ────────────────────────────────────────────────────
def test_caveat_gfx1151_fires_on_strix():
    m = model("Qwen3.6-35B-A3B")
    r = fits(m, quant(m, "Q4_K_M"), strix_unified_box(107000))
    assert any("gfx1151" in c.lower() for c in r.caveats)
    assert any("kernel" in c.lower() for c in r.caveats)


def test_caveat_unified_shared_pool():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), strix_unified_box(107000))
    assert any("pool" in c.lower() for c in r.caveats)


def test_caveat_tight_fit_still_fits():
    # 7B Q4: size 4700, reserve max(512, 470)=512 -> need 5212; threshold max(2000,...)=2000.
    # usable 6712 -> headroom 1500 (< 2000) -> tight, but still fits.
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(6712))
    assert r.fits is True
    assert 0 <= r.headroom_mb < 2000
    assert any("tight" in c.lower() for c in r.caveats)


def test_caveat_cpu_only():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cpu_box(120000))
    assert any("cpu-only" in c.lower() for c in r.caveats)


# ── graceful degradation (T-02-01 — never raises) ─────────────────────────────
def test_none_budget_degrades_not_raises():
    m = model("Qwen2.5-Coder-7B-Instruct")
    prof = HardwareProfile(primary_backend="cuda", usable_mem_mb=None,
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda")])
    r = fits(m, quant(m, "Q4_K_M"), prof)          # must not raise
    assert r.fits is False
    assert r.headroom_mb == 0
    assert any("unreadable" in c.lower() for c in r.caveats)


def test_empty_gpus_does_not_raise():
    m = model("Qwen2.5-Coder-7B-Instruct")
    prof = HardwareProfile(primary_backend="cpu", usable_mem_mb=32000, gpus=[])
    r = fits(m, quant(m, "Q4_K_M"), prof)          # must not raise
    assert isinstance(r, FitResult)
    assert r.fits is True


# ── FitResult contract ────────────────────────────────────────────────────────
def test_fitresult_exposes_the_contract_fields():
    m = model("Qwen2.5-Coder-7B-Instruct")
    r = fits(m, quant(m, "Q4_K_M"), cuda_box(16000))
    assert isinstance(r.fits, bool)
    assert isinstance(r.headroom_mb, int)
    assert isinstance(r.expected_speed_tier, str)
    assert isinstance(r.caveats, list)
    assert isinstance(r.need_mb, int)
    assert r.need_mb > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
