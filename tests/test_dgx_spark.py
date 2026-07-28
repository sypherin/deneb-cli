"""End-to-end: Deneb's advice AS IF running on a DGX Spark.

Everything else in this suite was written on, and largely verified against, a Strix Halo.
That is a real blind spot: the DGX Spark is the other box Deneb is sold against, nobody here
has one, and code that has only ever met one machine tends to encode it.

Driving a captured-shape DGX fixture through the whole chain - nvidia-smi text, memory
classification, platform detection, guide selection - is what found the bug these tests now
pin: `classify_memory` fell through to "else -> dedicated" for every NVIDIA GPU, so a DGX
Spark's 128 GB unified pool was treated as private VRAM. The usable-budget floor was skipped
entirely and the model was offered the whole machine, on a box where the OS lives in that
same RAM. That is exactly the failure the floor exists to prevent.

⚠ HONESTY CAVEAT: the fixture's nvidia-smi line is CONSTRUCTED to the documented
   --query-gpu CSV shape, not captured from real DGX hardware. It pins Deneb's behaviour
   given that input; it cannot prove the real device reports that exact product string.
   The marker match is deliberately substring-based to survive naming variations, and the
   dedicated-GPU tests below prove the widened rule did not swallow ordinary cards.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deneb import hardware as hw, platform_guide as pg  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "hardware"
DGX_RAM_MB = 131072          # 128 GB unified


def _dgx_gpu():
    gpus = hw.parse_nvidia_smi((FIX / "nvidia-smi_dgx_spark.csv").read_text())
    assert gpus, "the DGX fixture did not parse at all"
    return gpus[0]


class _Profile:
    def __init__(self, gpus):
        self.gpus = gpus


# ── the bug this file was written to catch ────────────────────────────────────
def test_dgx_unified_memory_is_not_treated_as_dedicated_vram():
    cls = hw.classify_memory(_dgx_gpu(), DGX_RAM_MB)
    assert cls.memory_kind == "unified", (
        "DGX Spark shares one memory pool between CPU and GPU. Calling it dedicated makes "
        "the whole 128 GB look like private VRAM.")


def test_dgx_budget_withholds_headroom_for_the_os():
    cls = hw.classify_memory(_dgx_gpu(), DGX_RAM_MB)
    assert cls.usable_mem_mb < DGX_RAM_MB, (
        "the model was offered the entire pool; on a unified box the OS lives there too")
    withheld = DGX_RAM_MB - cls.usable_mem_mb
    # Same floor the Strix scar produced: at least 8 GB, and 15% on a box this size.
    assert withheld >= hw._UNIFIED_HEADROOM_FLOOR_MB
    assert withheld >= int(hw._UNIFIED_HEADROOM_FRACTION * DGX_RAM_MB)


def test_dgx_and_strix_are_budgeted_on_the_same_rule():
    # Two unified 128 GB boxes should not disagree about how much of themselves is usable
    # just because one is AMD and one is NVIDIA.
    dgx = hw.classify_memory(_dgx_gpu(), DGX_RAM_MB)
    strix = hw.GPUInfo(vendor="amd", name="AMD Radeon 8060S Graphics", backend="rocm",
                       extra={"gfx": "gfx1151"})
    strix_cls = hw.classify_memory(strix, DGX_RAM_MB)
    assert dgx.usable_mem_mb == strix_cls.usable_mem_mb


# ── the widened rule must not swallow ordinary cards ──────────────────────────
def test_a_discrete_geforce_is_still_dedicated():
    gpus = hw.parse_nvidia_smi((FIX / "nvidia-smi_single.csv").read_text())
    cls = hw.classify_memory(gpus[0], DGX_RAM_MB)
    assert cls.memory_kind == "dedicated", (
        "an RTX 4090 has its own VRAM; widening the unified rule must not catch it")
    assert cls.usable_mem_mb == gpus[0].vram_mb


def test_multi_discrete_gpus_unaffected():
    for g in hw.parse_nvidia_smi((FIX / "nvidia-smi_multi.csv").read_text()):
        assert hw.classify_memory(g, DGX_RAM_MB).memory_kind == "dedicated"


# ── platform detection + guide selection ──────────────────────────────────────
def test_dgx_is_detected_as_its_own_platform():
    assert pg.detect_platform(_Profile([_dgx_gpu()])) == "dgx-spark"


def test_a_dgx_box_gets_the_dgx_guide_not_the_strix_one():
    key = pg.detect_platform(_Profile([_dgx_gpu()]))
    guide = pg.guide_for(key)
    assert guide is not None
    text = " ".join(f"{s.title} {s.what} {s.command}" for s in guide.steps).lower()
    # The Strix advice is actively wrong here: no BIOS UMA carveout, no amdgpu params.
    assert "amdgpu" not in text
    assert "uma frame buffer" not in text
    assert "nvidia-driver" in text


def test_dgx_guide_targets_ubuntu_not_fedora():
    guide = pg.guide_for("dgx-spark")
    assert "22.04" in guide.recommended_os
    text = " ".join(s.command for s in guide.steps)
    # Package commands must be apt, since NVIDIA's reference platform is Ubuntu. Telling a
    # DGX operator to run dnf is advice that cannot execute.
    assert "apt" in text
    assert "dnf" not in text


def test_ubuntu_fixture_resolves_to_the_debian_family():
    # The box the DGX guide is written for, so its package advice has to land correctly.
    fam = pg.distro_family((FIX / "os-release_ubuntu2204.txt").read_text())
    assert fam == "debian"
    assert pg.install_command(fam, "vulkan-tools").startswith("sudo apt-get")


def test_strix_advice_is_not_offered_to_a_dgx_operator():
    # The inverse guard: the two guides must not blur into one.
    strix = pg.guide_for("strix-halo")
    strix_text = " ".join(f"{s.title} {s.command}" for s in strix.steps).lower()
    assert "nvidia" not in strix_text and "cuda" not in strix_text
