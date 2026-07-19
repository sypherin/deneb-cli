"""Unit tests for deneb.hardware — the PURE hardware parsers + memory classification.

Every parser takes an already-captured tool-output STRING (loaded from
tests/fixtures/hardware/) and returns structured data, with NO live tool present —
that is the HW-04 / QA-01 testability contract. Two facts are load-bearing enough to
lock with tests:

  - HW-03 unified-memory trap: this box's rocm-smi reports a 1 GiB VRAM CARVE-OUT on
    a ~124 GiB unified box. parse_rocm_smi must surface it as vram_carveout_mb (not a
    real 1 GB card), and classify_memory must budget the ~124 GiB system RAM, not the
    carve-out. The discrete fixture proves not-all-AMD-is-unified.
  - T-01-01 defensiveness: every parser must degrade to []/0/{} on empty / whitespace
    / refused / garbage input and NEVER raise.

Run: python3 tests/test_hardware.py    (also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import hardware  # noqa: E402
from deneb.hardware import (  # noqa: E402
    GPUInfo,
    classify_memory,
    parse_cpuinfo,
    parse_lspci_gpus,
    parse_meminfo,
    parse_nvidia_smi,
    parse_rocm_smi,
    parse_sysctl_memsize,
    parse_system_profiler,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "hardware"


def load(name: str) -> str:
    return (FIX / name).read_text()


# Inputs every parser must survive without raising (T-01-01 / PLAT-03).
JUNK = ["", "   ", "\n\t ", "[refused] not on deneb's allowlist",
        "(not installed on this box: nvidia-smi)", "{not valid json",
        "\x00\xff garbage \x01", "random unrelated text with, commas, here"]


# ── parse_rocm_smi ───────────────────────────────────────────────────────────
def test_rocm_strix_json_is_unified_carveout_not_a_1gb_card():
    gpus = parse_rocm_smi(load("rocm-smi_strix_gfx1151.json"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "amd"
    assert "8060S" in g.name
    assert g.backend == "rocm"
    assert g.extra.get("gfx") == "gfx1151"
    assert g.extra.get("sku") == "STRXLGEN"
    # THE HW-03 TRAP: the 1 GiB is a carve-out, not a dedicated-VRAM budget.
    assert g.vram_carveout_mb == 1024
    assert g.vram_mb is None


def test_rocm_strix_text_variant_matches_json():
    gpus = parse_rocm_smi(load("rocm-smi_strix_gfx1151.txt"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "amd"
    assert "8060S" in g.name
    assert g.extra.get("gfx") == "gfx1151"
    assert g.extra.get("sku") == "STRXLGEN"
    assert g.vram_carveout_mb == 1024
    assert g.backend == "rocm"


def test_rocm_discrete_is_real_dedicated_vram():
    gpus = parse_rocm_smi(load("rocm-smi_discrete_gfx1100.json"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "amd"
    assert g.extra.get("gfx") == "gfx1100"
    assert g.extra.get("sku") == "NAVI31"
    # 25757220864 B / 1024 / 1024 == 24564 MiB of REAL dedicated VRAM.
    assert g.vram_mb == 24564
    assert g.vram_carveout_mb is None


def test_rocm_smi_never_raises_on_junk():
    for j in JUNK:
        assert parse_rocm_smi(j) == []


# ── parse_nvidia_smi ─────────────────────────────────────────────────────────
def test_nvidia_single():
    gpus = parse_nvidia_smi(load("nvidia-smi_single.csv"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA GeForce RTX 4090"
    assert g.vram_mb == 24564          # " MiB" unit stripped
    assert g.driver == "550.54.14"
    assert g.vendor == "nvidia"
    assert g.backend == "cuda"


def test_nvidia_multi():
    gpus = parse_nvidia_smi(load("nvidia-smi_multi.csv"))
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[1].name == "NVIDIA GeForce RTX 3090"
    assert gpus[1].vram_mb == 24576


def test_nvidia_smi_never_raises_on_junk():
    for j in JUNK:
        assert parse_nvidia_smi(j) == []


# ── parse_system_profiler (Apple / Metal) ────────────────────────────────────
def test_system_profiler_apple():
    gpus = parse_system_profiler(load("system_profiler_m2max.txt"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "apple"
    assert g.name == "Apple M2 Max"
    assert g.backend == "metal"


def test_system_profiler_never_raises_on_junk():
    for j in JUNK:
        assert parse_system_profiler(j) == []


# ── parse_sysctl_memsize ─────────────────────────────────────────────────────
def test_sysctl_memsize():
    assert parse_sysctl_memsize(load("sysctl_hw_memsize.txt")) == 65536  # 64 GiB
    assert parse_sysctl_memsize("hw.memsize: 68719476736") == 65536


def test_sysctl_memsize_never_raises_on_junk():
    for j in JUNK:
        assert parse_sysctl_memsize(j) == 0


# ── parse_cpuinfo ────────────────────────────────────────────────────────────
def test_cpuinfo():
    info = parse_cpuinfo(load("proc_cpuinfo_amd.txt"))
    assert info["model"] == "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"
    assert info["cores"] == 2          # count of "processor" lines in the trimmed fixture


def test_cpuinfo_never_raises_on_junk():
    for j in JUNK:
        info = parse_cpuinfo(j)
        assert info["cores"] == 0 and info["model"] == ""


# ── parse_meminfo ────────────────────────────────────────────────────────────
def test_meminfo():
    assert parse_meminfo(load("proc_meminfo.txt")) == 126908  # 129953980 kB // 1024


def test_meminfo_never_raises_on_junk():
    for j in JUNK:
        assert parse_meminfo(j) == 0


# ── parse_lspci_gpus (fallback when SMI tool absent) ─────────────────────────
def test_lspci_strix():
    gpus = parse_lspci_gpus(load("lspci_strix.txt"))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "amd"           # from PCI id 1002
    assert "Strix Halo" in g.name


def test_lspci_never_raises_on_junk():
    for j in JUNK:
        assert parse_lspci_gpus(j) == []


# ── classify_memory (HW-03 — the whole point) ────────────────────────────────
def test_classify_strix_is_unified_from_system_ram_not_carveout():
    g = parse_rocm_smi(load("rocm-smi_strix_gfx1151.json"))[0]
    res = classify_memory(g, ram_total_mb=126908)
    assert res.memory_kind == "unified"
    assert res.unified_mem_mb == 126908               # system RAM, NOT the 1024 carve-out
    assert 0 < res.usable_mem_mb < 126908             # documented headroom applied
    assert res.note                                   # a human-readable why is recorded
    assert "1024" not in str(res.unified_mem_mb)      # never the carve-out as the budget
    # side effect: the GPU record is updated too
    assert g.memory_kind == "unified"


def test_classify_discrete_is_dedicated_from_vram():
    g = parse_rocm_smi(load("rocm-smi_discrete_gfx1100.json"))[0]
    res = classify_memory(g, ram_total_mb=126908)
    assert res.memory_kind == "dedicated"
    assert res.usable_mem_mb == g.vram_mb == 24564    # GPU VRAM is the budget, not system RAM
    assert res.unified_mem_mb is None


def test_classify_apple_is_unified():
    g = parse_system_profiler(load("system_profiler_m2max.txt"))[0]
    res = classify_memory(g, ram_total_mb=65536)
    assert res.memory_kind == "unified"
    assert res.unified_mem_mb == 65536
    assert 0 < res.usable_mem_mb < 65536


def test_classify_never_raises_on_missing_ram():
    g = parse_rocm_smi(load("rocm-smi_strix_gfx1151.json"))[0]
    res = classify_memory(g, ram_total_mb=None)       # couldn't read system RAM
    assert res.memory_kind == "unified"               # still known-unified from gfx
    assert res.usable_mem_mb == 0                      # honest: budget unknown, never a crash


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
