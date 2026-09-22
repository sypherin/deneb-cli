"""deneb.matrix: catalog device -> synthetic profile -> recommend(), same rules as a live box."""
from deneb import matrix
from deneb.models_catalog import Model, Quant

GPU24 = {"id": "g", "name": "Test GPU 24", "vendor": "nvidia", "class": "consumer-gpu",
         "mem_gb": 24, "bw_gbps": 1000}
BOX128 = {"id": "u", "name": "Test Box 128", "vendor": "amd", "class": "unified-memory-box",
          "mem_gb": 128, "bw_gbps": 256}

SMALL = Model("Small-8B", 8.0, "dense", None, ["general"],
              [Quant("Q4_K_M", 5000, 4.8), Quant("Q8_0", 8600, 8.5)], "x/small")
BIG = Model("Big-120B", 120.0, "moe", 5.0, ["general"],
            [Quant("Q4_K_M", 70000, 4.6)], "x/big")


def test_budget_dedicated_is_full_vram():
    assert matrix.device_budget_mb(GPU24) == 24 * 1024


def test_budget_unified_reserves_15pct_or_8gb():
    mem = 128 * 1024
    assert matrix.device_budget_mb(BOX128) == mem - int(0.15 * mem)
    small = dict(BOX128, mem_gb=32)
    assert matrix.device_budget_mb(small) == 32 * 1024 - 8192  # floor wins below ~53 GB


def test_budget_unknown_memory_is_none():
    assert matrix.device_budget_mb({"mem_gb": None}) is None
    assert matrix.device_profile({"name": "x"}) is None
    assert matrix.picks_for_device({"name": "x"}, [SMALL]) == []


def test_picks_respect_memory_and_carry_estimate():
    picks = matrix.picks_for_device(GPU24, [SMALL, BIG])
    assert [p.model.name for p in picks] == ["Small-8B"]  # 70 GB model cannot fit 24 GB
    assert picks[0].quant.name == "Q8_0"                   # best fitting quant
    assert picks[0].fit.est_tps is not None
    big = [p.model.name for p in matrix.picks_for_device(BOX128, [SMALL, BIG])]
    assert big[0] == "Big-120B"


def test_nothing_fits_returns_empty_not_placeholder():
    tiny = dict(GPU24, mem_gb=2)
    assert matrix.picks_for_device(tiny, [SMALL, BIG]) == []
