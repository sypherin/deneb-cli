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


# ── #38: a tight pick is never #1 and never gets the green tick ───────────────────────────
def _model(name, params, quants, caps=("coding", "general")):
    from deneb.models_catalog import Model, Quant
    return Model(name=name, params_b=params, architecture="dense", active_params_b=None,
                 capabilities=list(caps), quants=[Quant(*q) for q in quants], gguf_repo="x/" + name,
                 notes="")


def test_tight_pick_is_never_first_when_a_comfortable_one_exists():
    # Budget 100000 MB. The big model needs 100000-4820: headroom 4820 < 10% of need (the
    # documented KV allowance), so it is tight. The smaller one leaves plenty.
    big = _model("Big-70B", 70, [("Q4_K_M", 86300, 4.85)])
    small = _model("Small-8B", 8, [("Q4_K_M", 5000, 4.85)])
    recs = recommend(cuda_box(100000), "coding", top_n=2, catalog=[big, small])
    assert recs[0].model.name == "Small-8B"
    assert recs[1].model.name == "Big-70B" and recs[1].fit.tight is True
    assert recs[0].fit.tight is False


def test_a_model_prefers_a_smaller_quant_that_leaves_headroom():
    m = _model("M-30B", 30, [("Q4_K_M", 30000, 4.85), ("Q8_0", 55000, 8.5)])
    recs = recommend(cuda_box(60000), "coding", top_n=1, catalog=[m])
    assert recs[0].quant.name == "Q4_K_M" and recs[0].fit.tight is False


def test_a_tight_pick_that_would_otherwise_rank_first_is_demoted():
    # Both fit and neither is very-slow, and the tight one is the MORE capable (14B vs 7B), so
    # only the comfortable-first key keeps it from #1. Guards recommend._rank_key.
    big = _model("Big-14B", 14, [("Q4_K_M", 18000, 4.85)])
    small = _model("Small-7B", 7, [("Q4_K_M", 4500, 4.85)])
    box = cuda_box(20000)
    from deneb.fit import fits
    fb, fs = fits(big, big.quants[0], box), fits(small, small.quants[0], box)
    assert fb.fits and fb.tight and fb.expected_speed_tier != "very-slow"
    assert fs.fits and not fs.tight and fs.expected_speed_tier != "very-slow"
    recs = recommend(box, "coding", top_n=2, catalog=[big, small])
    assert [r.model.name for r in recs] == ["Small-7B", "Big-14B"]


def test_best_quant_prefers_a_roomy_quant_over_a_higher_bpw_tight_one():
    # Q8_0 fits but is tight; Q4_K_M fits with room. Guards recommend.best_fitting_quant.
    m = _model("M-14B", 14, [("Q4_K_M", 9000, 4.85), ("Q8_0", 18000, 8.5)])
    box = cuda_box(20000)
    from deneb.fit import fits
    assert fits(m, m.quants[1], box).fits and fits(m, m.quants[1], box).tight
    assert fits(m, m.quants[0], box).fits and not fits(m, m.quants[0], box).tight
    recs = recommend(box, "coding", top_n=1, catalog=[m])
    assert recs[0].quant.name == "Q4_K_M" and recs[0].fit.tight is False


def test_cli_fit_cell_marks_tight_with_a_warning_not_a_tick():
    from deneb import cli
    from deneb.fit import fits
    big = _model("Big-70B", 70, [("Q4_K_M", 86300, 4.85)])
    fr = fits(big, big.quants[0], cuda_box(100000))
    assert fr.fits and fr.tight and fr.headroom_mb > 2000       # the old fixed 2000 MB test said fine
    cell = cli._fit_cell(fr)
    assert "✓" not in cell and "⚠" in cell and "tight" in cell


def test_why_line_gives_the_tight_reason_even_when_another_caveat_comes_first():
    big = _model("Big-70B", 70, [("Q4_K_M", 86300, 4.85)])
    r = recommend(strix_box(100000), "coding", top_n=1, catalog=[big])[0]
    assert "tight fit" in r.why
