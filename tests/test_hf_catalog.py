"""deneb.hf_catalog — pure parser/builder tests (no network)."""
import json
import os

from deneb import hf_catalog as hc
from deneb.models_catalog import Model

GB = 10**9


def _f(path, size):
    return {"type": "file", "path": path, "size": size}


# Shaped like the real unsloth Qwen3.8-Flash-Next listing: quant subfolders with shards,
# an MTP/ folder of draft heads, an mmproj projector, and a top-level UD file.
FLASH_TREE = [
    {"type": "directory", "path": "BF16"},
    _f("BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf", 50 * GB),
    _f("Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00004.gguf", 49 * GB),
    _f("Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00004.gguf", 49 * GB),
    _f("Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00004.gguf", 49 * GB),
    _f("Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00004.gguf", 48 * GB),
    _f("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf", 50 * GB),
    _f("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00003.gguf", 50 * GB),
    _f("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00003.gguf", 11 * GB),
    _f("MTP/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf", 2 * GB),
    _f("mmproj-F16.gguf", 1 * GB),
    _f("Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf", 79 * GB),
    _f("Qwen3.8-Flash-Next-UD-IQ1_M.gguf", 45 * GB),
    _f("README.md", 5000),
    _f("imatrix_unsloth.gguf_file", 1000),
]


def test_quant_of_exact_tokens():
    assert hc.quant_of("Q6_K/x-Q6_K-00001-of-00002.gguf") == "Q6_K"
    # the old naive regex folded UD-Q6_K_XL into Q6_K: must stay distinct
    assert hc.quant_of("UD-Q6_K_XL/x-UD-Q6_K_XL-00001-of-00002.gguf") == "UD-Q6_K_XL"
    assert hc.quant_of("gpt-oss-120b-MXFP4_MOE.gguf") == "MXFP4"
    assert hc.quant_of("x-IQ4_XS.gguf") == "IQ4_XS"


def test_quant_of_rejects_non_weights():
    assert hc.quant_of("MTP/x-MTP-Q4_K_M.gguf") is None
    assert hc.quant_of("mmproj-F16.gguf") is None
    assert hc.quant_of("README.md") is None
    assert hc.quant_of("imatrix.gguf") is None


def test_parse_sums_shards_and_excludes_mtp():
    sizes, mmproj = hc.parse_gguf_tree(FLASH_TREE)
    assert mmproj is True
    assert sizes["Q8_0"] == 195 * GB
    assert sizes["UD-Q4_K_XL"] == 111 * GB
    assert sizes["UD-Q2_K_XL"] == 79 * GB
    # MTP draft head must NOT be counted as a Q4_K_M quant
    assert "Q4_K_M" not in sizes


def test_parse_junk_is_safe():
    assert hc.parse_gguf_tree(None) == ({}, False)
    assert hc.parse_gguf_tree([None, 3, {"type": "file"}]) == ({}, False)


def test_build_entry_drops_1bit_and_bf16_and_derives_bpw():
    sizes, mmproj = hc.parse_gguf_tree(FLASH_TREE)
    w = {"name": "Flash", "gguf_repo": "u/F", "architecture": "moe", "active_params_b": 6,
         "capabilities": ["coding"]}
    e = hc.build_entry(w, 176_900_000_000, sizes, mmproj, "2026-09-23")
    names = [q["name"] for q in e["quants"]]
    assert names == ["UD-Q2_K_XL", "UD-Q4_K_XL", "Q8_0"]  # sorted by bpw, no IQ1/BF16
    q8 = e["quants"][-1]
    assert 8.5 < q8["bpw"] < 9.2 and q8["size_mb"] == 195000
    assert "vision" in e["capabilities"] and e["params_b"] == 176.9


def test_build_entry_none_without_params_or_quants():
    w = {"name": "x", "gguf_repo": "u/x"}
    assert hc.build_entry(w, None, {"Q4_K_M": GB}, False, "") is None
    assert hc.build_entry(w, 10**9, {"BF16": GB}, False, "") is None


def test_entry_roundtrips_to_model():
    sizes, mmproj = hc.parse_gguf_tree(FLASH_TREE)
    w = {"name": "Flash", "gguf_repo": "u/F", "architecture": "moe", "active_params_b": 6}
    m = hc.entry_to_model(hc.build_entry(w, 176_900_000_000, sizes, mmproj, "t"))
    assert isinstance(m, Model) and m.active_params_b == 6 and "mmproj" in m.notes
    assert hc.entry_to_model({"name": "broken"}) is None


def test_bundled_watchlist_and_snapshot_are_valid():
    wl = hc.load_watchlist()
    assert len(wl) >= 10
    for w in wl:
        assert w["name"] and "/" in w["gguf_repo"]
        assert w["architecture"] in ("dense", "moe")
        if w["architecture"] == "moe":
            assert w["active_params_b"] and w["active_params_b"] > 0
    with open(hc.SNAPSHOT_PATH) as f:
        snap = json.load(f)
    models = [hc.entry_to_model(e) for e in snap["models"]]
    assert len(models) >= 10 and all(models)
    for m in models:  # size must rise with bpw inside every model
        sz = [q.size_mb for q in m.quants]
        assert sz == sorted(sz), m.name


def test_package_data_declared():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "pyproject.toml")) as f:
        assert "data/*.json" in f.read()
