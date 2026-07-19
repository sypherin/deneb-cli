"""Integrity tests for deneb.models_catalog — the curated model matrix (FIT-01).

These lock the CATALOG's SHAPE, not network facts (the sizes/repo-ids are curated
approximations, verified online only in a later phase). The load-bearing invariants
the FIT-02 fit math and REC-01 ranking rely on:

  - every model is well-formed (name, params, a known architecture, >=1 quant, a repo);
  - capability tags come only from the fixed vocabulary;
  - MoE models carry active_params_b, dense models leave it None (the speed tier keys
    off this);
  - within each model, size_mb rises with bpw (quant monotonicity — fits() assumes the
    smallest quant is the cheapest to fit);
  - the Qwen3.6-35B-A3B MoE is present and correctly described (the canonical MoE case).

Run: python3 tests/test_catalog.py    (also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb.models_catalog import (  # noqa: E402
    CAPABILITIES,
    CATALOG,
    Model,
    Quant,
    by_capability,
)


# ── whole-catalog shape ───────────────────────────────────────────────────────
def test_catalog_non_empty_and_covers_the_families():
    assert isinstance(CATALOG, list)
    assert len(CATALOG) >= 16                       # the curated starter set
    names = " ".join(m.name for m in CATALOG)
    for family in ("Qwen2.5-Coder", "Qwen3.6-35B-A3B", "Llama-3", "Gemma",
                   "Mistral", "Mixtral", "Phi", "DeepSeek-Coder"):
        assert family in names, f"missing family: {family}"


def test_model_names_are_unique():
    names = [m.name for m in CATALOG]
    assert len(names) == len(set(names)), "duplicate model name in CATALOG"


def test_every_model_is_wellformed():
    for m in CATALOG:
        assert isinstance(m, Model)
        assert m.name and isinstance(m.name, str)
        assert m.params_b and m.params_b > 0
        assert m.architecture in ("dense", "moe")
        assert m.gguf_repo and "/" in m.gguf_repo   # looks like an org/repo id
        assert isinstance(m.quants, list) and len(m.quants) >= 1
        for q in m.quants:
            assert isinstance(q, Quant)
            assert q.size_mb and q.size_mb > 0
            assert q.bpw and q.bpw > 0


def test_capability_tags_are_in_the_allowed_set():
    for m in CATALOG:
        assert m.capabilities, f"{m.name} has no capability tags"
        for tag in m.capabilities:
            assert tag in CAPABILITIES, f"{m.name}: bad tag {tag!r}"


def test_moe_has_active_params_dense_leaves_it_none():
    for m in CATALOG:
        if m.architecture == "moe":
            assert m.active_params_b and m.active_params_b > 0, \
                f"{m.name} is MoE but has no active_params_b"
            assert m.active_params_b < m.params_b, \
                f"{m.name}: active params should be < total"
        else:
            assert m.active_params_b is None, \
                f"{m.name} is dense but sets active_params_b"


def test_quant_size_is_monotonic_in_bpw():
    # The fit math assumes size_mb rises with bits/weight — lock it per model.
    for m in CATALOG:
        ordered = sorted(m.quants, key=lambda q: q.bpw)
        sizes = [q.size_mb for q in ordered]
        assert sizes == sorted(sizes), f"{m.name}: size_mb not monotonic in bpw {sizes}"
        assert len(set(q.bpw for q in ordered)) == len(ordered), \
            f"{m.name}: duplicate bpw among quants"


# ── the canonical MoE row ─────────────────────────────────────────────────────
def test_qwen36_moe_row_is_correct():
    match = [m for m in CATALOG if m.name == "Qwen3.6-35B-A3B"]
    assert len(match) == 1, "Qwen3.6-35B-A3B must exist exactly once"
    m = match[0]
    assert m.architecture == "moe"
    assert m.active_params_b == 3
    assert "coding" in m.capabilities


# ── by_capability helper ──────────────────────────────────────────────────────
def test_by_capability_filters():
    coders = by_capability("coding")
    vision = by_capability("vision")
    assert len(coders) >= 1
    assert len(vision) >= 1
    assert all("coding" in m.capabilities for m in coders)
    assert all("vision" in m.capabilities for m in vision)
    assert by_capability("does-not-exist") == []   # unknown tag: empty, no raise


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
