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


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ORCHESTRATION — graceful multi-vendor degradation (Plan 02, HW-02)
#
# profile_hardware() drives the live probe layer. We monkeypatch the `tools` boundary
# (mirroring test_check.py's _install router) to simulate four boxes that each have
# EXACTLY ONE vendor tool — the real-world case (HW-02): most boxes have one of
# nvidia-smi / rocm-smi / system_profiler, never all three. Absent tools return the
# "(not installed on this box: X)" sentinel and MUST degrade to not-present with NO
# crash. Plus a live assertion on THIS real Strix Halo box.
# ═══════════════════════════════════════════════════════════════════════════════
import contextlib  # noqa: E402
import platform    # noqa: E402
import shutil      # noqa: E402

_NOT_INSTALLED = "(not installed on this box: x)"
# rocm-smi --showdriverversion sample (optional metadata).
_ROCM_DRIVER = ("Driver version: 6.10.5\n")


def _run_router(mapping: dict):
    """Fake tools.run: first value whose key is a substring of cmd; absent tools return
    the not-installed sentinel (so the parser sees absence)."""
    def fake_run(cmd, *a, **k):
        for key, val in mapping.items():
            if key in cmd:
                return {"ok": True, "output": val}
        return {"ok": True, "output": _NOT_INSTALLED}
    return fake_run


def _read_router(mapping: dict):
    def fake_read(path, *a, **k):
        for key, val in mapping.items():
            if key in path:
                return {"ok": True, "output": val}
        return {"ok": True, "output": f"(does not exist: {path})"}
    return fake_read


class _FakeTools:
    def __init__(self, run_fn, read_fn):
        self.run, self.read_file = run_fn, read_fn


class _FakePlatform:
    def __init__(self, system):
        self._s = system

    def system(self):
        return self._s

    def machine(self):
        return "arm64" if self._s == "Darwin" else "x86_64"

    def release(self):
        return "23.0.0"

    def processor(self):
        return "Apple" if self._s == "Darwin" else "x86_64"


@contextlib.contextmanager
def simulated_box(run_map, read_map=None, system="Linux"):
    """Swap hardware.tools (and platform, for the Apple box) for fakes, then restore —
    leaves the real tools module untouched so the live test still reads the real box."""
    real_tools, real_platform = hardware.tools, hardware.platform
    hardware.tools = _FakeTools(_run_router(run_map), _read_router(read_map or {}))
    if system != real_platform.system():
        hardware.platform = _FakePlatform(system)
    try:
        yield
    finally:
        hardware.tools = real_tools
        hardware.platform = real_platform


# Deterministic CPU/RAM fixtures (Linux boxes read /proc via the patched read_file).
_LINUX_READS = {
    "/proc/cpuinfo": load("proc_cpuinfo_amd.txt"),
    "/proc/meminfo": load("proc_meminfo.txt"),
}


def test_nvidia_only_box_one_cuda_gpu_no_crash():
    run_map = {"nvidia-smi --query-gpu": load("nvidia-smi_single.csv")}  # rocm/apple/lspci absent
    with simulated_box(run_map, _LINUX_READS):
        p = hardware.profile_hardware()
    assert isinstance(p, hardware.HardwareProfile)
    assert len(p.gpus) == 1
    assert p.gpus[0].vendor == "nvidia" and p.gpus[0].backend == "cuda"
    assert p.primary_backend == "cuda"
    assert p.gpus[0].vram_mb == 24564 and p.gpus[0].memory_kind == "dedicated"
    # absent vendors are simply not represented (HW-02)
    assert not any(g.vendor in ("amd", "apple") for g in p.gpus)
    assert p.ram_total_mb == 126908 and p.cpu_cores == 2


def test_amd_only_strix_box_unified_from_ram_not_carveout():
    run_map = {
        "rocm-smi --showproductname": load("rocm-smi_strix_gfx1151.json"),
        "rocm-smi --showdriverversion": _ROCM_DRIVER,
    }  # nvidia/apple/lspci absent
    with simulated_box(run_map, _LINUX_READS):
        p = hardware.profile_hardware()
    assert len(p.gpus) == 1
    g = p.gpus[0]
    assert g.vendor == "amd" and g.backend == "rocm"
    assert g.memory_kind == "unified"
    assert g.unified_mem_mb == 126908          # system RAM, NOT the 1024 carve-out
    assert g.vram_carveout_mb == 1024
    assert 0 < p.usable_mem_mb < 126908         # usable from the RAM pool, minus headroom
    assert p.primary_backend == "rocm"
    assert not any(x.vendor in ("nvidia", "apple") for x in p.gpus)


def test_apple_only_box_one_metal_gpu_unified():
    run_map = {
        "system_profiler SPDisplaysDataType": load("system_profiler_m2max.txt"),
        "sysctl hw.memsize": load("sysctl_hw_memsize.txt"),
    }  # nvidia/rocm/lspci absent; platform patched to Darwin
    with simulated_box(run_map, read_map={}, system="Darwin"):
        p = hardware.profile_hardware()
    assert p.os == "Darwin"
    assert len(p.gpus) == 1
    g = p.gpus[0]
    assert g.vendor == "apple" and g.backend == "metal"
    assert g.memory_kind == "unified"
    assert g.unified_mem_mb == 65536            # unified pool from sysctl hw.memsize
    assert p.primary_backend == "metal"
    assert not any(x.vendor in ("nvidia", "amd") for x in p.gpus)


def test_cpu_only_box_returns_profile_no_gpu_backend_cpu():
    with simulated_box({}, _LINUX_READS):        # ALL vendor tools absent
        p = hardware.profile_hardware()
    assert isinstance(p, hardware.HardwareProfile)
    assert p.gpus == []                          # no vendor represented
    assert p.primary_backend == "cpu"
    assert p.ram_total_mb == 126908
    assert 0 < p.usable_mem_mb < 126908          # budget from RAM, not a GPU
    assert p.cpu_model                           # CPU still read


def test_no_vendor_tool_never_raises_across_all_four_boxes():
    # The HW-02 contract in one shot: none of the four degraded boxes raises.
    boxes = [
        ({"nvidia-smi --query-gpu": load("nvidia-smi_single.csv")}, _LINUX_READS, "Linux"),
        ({"rocm-smi --showproductname": load("rocm-smi_strix_gfx1151.json")}, _LINUX_READS, "Linux"),
        ({"system_profiler SPDisplaysDataType": load("system_profiler_m2max.txt"),
          "sysctl hw.memsize": load("sysctl_hw_memsize.txt")}, {}, "Darwin"),
        ({}, _LINUX_READS, "Linux"),
    ]
    for run_map, read_map, system in boxes:
        with simulated_box(run_map, read_map, system):
            p = hardware.profile_hardware()          # must not raise
        assert isinstance(p, hardware.HardwareProfile)


def test_live_this_box_amd_unified_no_nvidia_no_apple():
    """LIVE e2e gate on THIS real Strix Halo box (skipped elsewhere): one AMD/rocm/gfx1151
    GPU with unified ~124GB (NOT the 1GB carve-out), NVIDIA + Apple absent, no crash."""
    if platform.system() != "Linux" or shutil.which("rocm-smi") is None:
        try:
            import pytest
            pytest.skip("live AMD/ROCm assertion is only meaningful on the Strix Halo box")
        except ImportError:
            return
    p = hardware.profile_hardware()                  # real tools, real box
    assert isinstance(p, hardware.HardwareProfile)
    assert p.os == "Linux" and p.arch == "x86_64"
    amd = [g for g in p.gpus if g.vendor == "amd"]
    assert len(amd) == 1
    g = amd[0]
    assert g.backend == "rocm"
    assert g.extra.get("gfx") == "gfx1151"
    assert g.memory_kind == "unified"
    assert g.unified_mem_mb and g.unified_mem_mb > 100_000   # ~124GB pool
    assert g.unified_mem_mb != 1024                          # NOT the carve-out
    assert g.vram_carveout_mb == 1024                        # the carve-out is recorded, not budgeted
    assert not any(x.vendor in ("nvidia", "apple") for x in p.gpus)  # absent + unmentioned
    assert p.primary_backend == "rocm"
    assert "RYZEN AI MAX+ 395" in p.cpu_model
    assert p.ram_total_mb and p.ram_total_mb > 100_000
    assert p.disk_free_mb is not None
    assert 0 < p.usable_mem_mb < p.ram_total_mb


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
