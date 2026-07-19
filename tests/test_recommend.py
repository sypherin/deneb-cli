"""Unit tests for deneb.recommend — the pure, deterministic recommendation ranking
(REC-01, REC-03).

recommend(profile, use_case) filters the curated CATALOG by capability, keeps each model's
best FITTING quant, ranks deterministically (specialist > general; the most capable that
still runs acceptably; a comfortable fit; faster), and returns the top-N each with a plain
"why" string. When nothing fits it returns ONE honest under-spec pick instead of an empty
list. Profiles are CONSTRUCTED directly (QA-01 style) so every ranking fact is pinned
against a known box — no live tool, no I/O, no LLM.

Load-bearing cases (the plan's canonical set):
  - determinism (same order twice); the Recommendation contract (model/quant/fit/why);
  - BIG-BOX coding: the fast Qwen3.6-35B-A3B MoE tops a 107GB unified box at a high-bpw quant;
  - SMALL-BOX coding: a 16GB box excludes the 32B/33B/35B/70B and returns a mid model;
  - specialist > general of the same size; best-quant = highest bpw that still fits;
  - vision filtering; the nothing-fits single under-spec result; the plain no-em-dash why;
  - a None-budget profile degrades to nothing-fits and NEVER raises (T-03-02).

Run: python3 tests/test_recommend.py    (also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from deneb.models_catalog import CATALOG  # noqa: E402
from deneb.recommend import Recommendation, recommend  # noqa: E402


# ── constructed test boxes (see the plan's <interfaces>) ──────────────────────
def strix_box(usable=107000):
    """This-box shape: 128GB unified Strix Halo (gfx1151), rocm."""
    return HardwareProfile(
        primary_backend="rocm", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="amd", backend="rocm", memory_kind="unified",
                      unified_mem_mb=126908,
                      extra={"gfx": "gfx1151", "sku": "STRXLGEN"})])


def cuda_box(usable):
    return HardwareProfile(
        primary_backend="cuda", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=usable,
                      memory_kind="dedicated")])


def tiny_box(usable=1500):
    return HardwareProfile(
        primary_backend="cuda", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=usable,
                      memory_kind="dedicated")])


def _names(recs):
    return [(r.model.name, r.quant.name) for r in recs]


def _pick(recs, name):
    for r in recs:
        if r.model.name == name:
            return r
    return None


# ── determinism + contract ────────────────────────────────────────────────────
def test_determinism_same_order_twice():
    a = recommend(strix_box(), "coding")
    b = recommend(strix_box(), "coding")
    assert _names(a) == _names(b)
    assert len(a) >= 1


def test_recommendation_exposes_contract_fields():
    r = recommend(strix_box(), "coding")[0]
    assert isinstance(r, Recommendation)
    assert hasattr(r, "model") and hasattr(r, "quant")
    assert hasattr(r, "fit") and hasattr(r, "why")


# ── the canonical big-box / small-box coding picks ────────────────────────────
def test_big_box_coding_top_is_qwen_moe():
    recs = recommend(strix_box(107000), "coding")
    top = recs[0]
    assert top.model.name == "Qwen3.6-35B-A3B"       # fast MoE + highest total params
    assert top.fit.fits is True
    assert "coding" in top.model.capabilities
    assert top.quant.name == "Q8_0"                  # highest-bpw quant that fits 107GB


def test_small_box_coding_excludes_giants():
    recs = recommend(cuda_box(16000), "coding")
    top = recs[0]
    assert top.fit.fits is True
    assert top.model.params_b < 20                   # not the 32B/33B/35B/70B
    assert top.model.name == "Qwen2.5-Coder-14B-Instruct"


# ── specialist > general ; best-quant selection ───────────────────────────────
def test_specialized_outranks_general_same_size():
    recs = recommend(strix_box(107000), "coding", top_n=len(CATALOG))
    order = [r.model.name for r in recs]
    assert "Qwen2.5-Coder-14B-Instruct" in order     # 14B coding specialist
    assert "Phi-4" in order                          # 14B general-only (chat+general)
    assert order.index("Qwen2.5-Coder-14B-Instruct") < order.index("Phi-4")


def test_best_quant_is_highest_bpw_that_fits():
    big = recommend(strix_box(107000), "coding", top_n=len(CATALOG))
    tight = recommend(cuda_box(10000), "coding", top_n=len(CATALOG))
    big14 = _pick(big, "Qwen2.5-Coder-14B-Instruct")
    tight14 = _pick(tight, "Qwen2.5-Coder-14B-Instruct")
    assert big14 is not None and tight14 is not None
    assert big14.fit.fits and tight14.fit.fits
    assert big14.quant.bpw > tight14.quant.bpw       # more budget -> higher-quality quant


# ── vision filtering ──────────────────────────────────────────────────────────
def test_vision_returns_vision_capable():
    recs = recommend(strix_box(107000), "vision")
    for r in recs:
        caps = r.model.capabilities
        assert ("vision" in caps) or ("general" in caps)
    assert "vision" in recs[0].model.capabilities    # a vision-tagged model tops it
    assert any("vision" in r.model.capabilities for r in recs)


# ── nothing-fits (REC-03) ─────────────────────────────────────────────────────
def test_nothing_fits_returns_single_underspec():
    recs = recommend(tiny_box(1500), "coding")
    assert len(recs) == 1
    r = recs[0]
    assert r.fit.fits is False
    low = r.why.lower()
    assert "under-spec" in low
    assert "smallest" in low
    assert "short by" in low


# ── why-string is plain, names model + tier, no em-dash ───────────────────────
def test_why_is_plain_with_name_and_tier_no_emdash():
    recs = recommend(strix_box(107000), "coding")
    for r in recs:
        assert isinstance(r.why, str) and r.why.strip()
        assert r.model.name in r.why
        assert r.fit.expected_speed_tier in r.why
        assert "—" not in r.why                 # no em-dash (house style)


# ── graceful degradation (T-03-02 — never raises) ─────────────────────────────
def test_none_budget_degrades_not_raises():
    prof = HardwareProfile(primary_backend="cuda", usable_mem_mb=None,
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda")])
    recs = recommend(prof, "coding")                 # must not raise
    assert len(recs) == 1
    assert recs[0].fit.fits is False


def test_unknown_use_case_falls_back_to_general():
    recs = recommend(strix_box(107000), "bogus")     # normalized to "general"
    assert len(recs) >= 1
    assert recs[0].fit.fits is True


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
