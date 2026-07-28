"""Tests for deneb.platform_guide — the pre-flight runbook data + its pure helpers.

Two things are being defended here.

The first is ordinary: distro_family and detect_platform are parsing functions, and parsing
functions get pinned against captured real-world shapes rather than idealised ones.

The second matters more. The values in these guides are not stylistic - each one was arrived
at by something breaking. A UMA carveout of 512 MB rather than 64 GB, a gttsize that must
equal its ttm.pages_limit, a kernel floor of 6.18.4: change any of them and the advice still
reads plausibly while describing a box that hangs under load. These tests exist so a future
edit to that data is a deliberate act rather than a typo nobody noticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deneb import platform_guide as pg  # noqa: E402


# ── distro detection ──────────────────────────────────────────────────────────
_FEDORA = 'NAME="Fedora Linux"\nID=fedora\nVERSION_ID=43\n'
_UBUNTU = 'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\nVERSION_ID="22.04"\n'
_NOBARA = 'NAME="Nobara"\nID=nobara\nID_LIKE="fedora"\n'
_CACHY = 'NAME="CachyOS"\nID=cachyos\nID_LIKE=arch\n'


def test_distro_family_reads_plain_ids():
    assert pg.distro_family(_FEDORA) == "fedora"
    assert pg.distro_family(_UBUNTU) == "debian"


def test_distro_family_honours_id_like_for_derivatives():
    # A derivative that ID_LIKEs a known family must resolve, or its user is handed a
    # package command that cannot run on their box.
    assert pg.distro_family(_NOBARA) == "fedora"
    assert pg.distro_family(_CACHY) == "arch"


def test_distro_family_tolerates_quotes_and_junk():
    assert pg.distro_family('ID="fedora"') == "fedora"
    assert pg.distro_family("") == ""
    assert pg.distro_family("nonsense\n\n") == ""


def test_install_command_is_distro_correct():
    assert pg.install_command("fedora", "vulkan-tools") == "sudo dnf -y install vulkan-tools"
    assert pg.install_command("debian", "vulkan-tools") == "sudo apt-get install -y vulkan-tools"


def test_unknown_distro_says_so_rather_than_guessing_apt():
    # Guessing apt on an unknown distro produces a command that silently cannot run.
    # Saying "I do not know" is the honest failure and the useful one.
    out = pg.install_command("plan9", "vulkan-tools")
    assert "apt" not in out and "dnf" not in out
    assert "vulkan-tools" in out


# ── platform detection ────────────────────────────────────────────────────────
class _GPU:
    def __init__(self, name="", vendor="", extra=None):
        self.name, self.vendor, self.extra = name, vendor, extra or {}


class _Profile:
    def __init__(self, gpus=None):
        self.gpus = gpus or []


def test_detects_strix_from_gfx_version():
    assert pg.detect_platform(_Profile([_GPU(extra={"gfx": "gfx1151"})])) == "strix-halo"
    assert pg.detect_platform(_Profile([_GPU(extra={"gfx": "gfx1150"})])) == "strix-halo"


def test_detects_dgx_from_name():
    assert pg.detect_platform(_Profile([_GPU(name="NVIDIA GB10", vendor="NVIDIA")])) == "dgx-spark"


def test_unknown_hardware_detects_nothing_rather_than_guessing():
    assert pg.detect_platform(_Profile([_GPU(name="GeForce RTX 4090", vendor="NVIDIA")])) == ""
    assert pg.detect_platform(_Profile([])) == ""


def test_detection_never_raises_on_a_degraded_probe():
    # A probe can come back malformed; advice must still print.
    class Broken:
        @property
        def gpus(self):
            raise RuntimeError("probe failed")
    assert pg.detect_platform(Broken()) == ""
    assert pg.detect_platform(None) == ""


# ── guide lookup ──────────────────────────────────────────────────────────────
def test_guide_lookup_is_forgiving_about_separators():
    for key in ("strix-halo", "Strix-Halo", "strix_halo", "  STRIX-HALO  "):
        assert pg.guide_for(key) is not None, key
    assert pg.guide_for("nope") is None
    assert pg.guide_for("") is None


# ── the hard-won values (regression pins) ─────────────────────────────────────
def _strix_text() -> str:
    g = pg.guide_for("strix-halo")
    return "\n".join(f"{s.title}\n{s.what}\n{s.command}\n{s.expect}" for s in g.steps)


def test_strix_boot_params_are_paired_and_exact():
    text = _strix_text()
    # gttsize and ttm.pages_limit must appear together and unchanged. A mismatch between
    # them is the single most common cause of "device lost under load", and the two numbers
    # are only correct as a pair.
    assert "amdgpu.gttsize=126976" in text
    assert "ttm.pages_limit=32505856" in text
    assert "amdgpu.cwsr_enable=0" in text
    assert "iommu=pt" in text


def test_strix_uma_advice_is_the_small_carveout():
    # The intuitive answer (a big VRAM carveout) is the wrong one. If this ever flips to
    # advising a large UMA, the guide is actively harmful.
    text = _strix_text().lower()
    assert "512 mb" in text
    assert "gtt" in text


def test_strix_kernel_floor_is_stated():
    assert "6.18.4" in _strix_text()


def test_strix_guide_warns_before_touching_the_bootloader():
    g = pg.guide_for("strix-halo")
    boot = [s for s in g.steps if "gttsize" in s.command]
    assert boot, "the kernel command-line step vanished"
    joined = " ".join(boot[0].warnings).lower()
    assert "bootloader" in joined and "unbootable" in joined


def test_bios_step_carries_a_warning_and_no_command():
    g = pg.guide_for("strix-halo")
    bios = [s for s in g.steps if s.title.lower().startswith("bios")]
    assert bios, "the BIOS step vanished"
    # Nothing to type: a command here would imply Deneb could do it, and it cannot.
    assert bios[0].command == ""
    assert bios[0].warnings


def test_dgx_guide_does_not_tell_you_to_install_cuda_12():
    # This is a regression pin for advice that WAS shipped and was wrong. CUDA 12.x does not
    # support GB10 Blackwell, the box is aarch64 so an x86_64 repo is the wrong
    # architecture, and the machine ships with CUDA 13 already installed. Any of the three
    # is enough to leave an operator with a toolkit that cannot see the GPU.
    # Checked against the COMMANDS, not the prose: the guide deliberately names these in a
    # warning telling you not to use them, and a test that banned the words outright would
    # forbid the very explanation that prevents the mistake.
    g = pg.guide_for("dgx-spark")
    commands = " ".join(s.command for s in g.steps)
    for bad in ("cuda-toolkit-12", "x86_64", "ubuntu2204", "nvidia-driver-570"):
        assert bad not in commands, f"DGX guide still tells you to run {bad!r}"

    # And the warning against them must survive, since the stale runbook is still in
    # circulation and someone will otherwise follow it.
    prose = " ".join(f"{s.what} {' '.join(s.warnings)}" for s in g.steps).lower()
    assert "12.4" in prose or "cuda 12" in prose


def test_dgx_guide_says_verify_rather_than_install():
    g = pg.guide_for("dgx-spark")
    text = " ".join(f"{s.title} {s.what}" for s in g.steps).lower()
    assert "do not install" in text or "do NOT reinstall".lower() in g.recommended_os.lower()
    assert "24.04" in g.recommended_os
    joined = " ".join(s.command for s in g.steps)
    assert "nvidia-smi" in joined and "nvcc --version" in joined
    # The headers-mismatch recovery is the one non-obvious step; losing it costs an hour.
    assert "linux-headers" in joined


def test_dgx_build_uses_native_arch_not_a_pinned_capability():
    g = pg.guide_for("dgx-spark")
    joined = " ".join(s.command for s in g.steps)
    assert "CMAKE_CUDA_ARCHITECTURES=native" in joined


def test_dgx_guide_states_the_memory_is_unified():
    g = pg.guide_for("dgx-spark")
    blob = f"{g.applies_to} " + " ".join(f"{s.title} {s.what}" for s in g.steps)
    assert "unified" in blob.lower()
    assert "aarch64" in blob.lower() or "arm" in blob.lower()


def test_every_sudo_step_carries_a_warning():
    # SET-03's contract, applied to this module: anything running as root says so inline.
    for key, guide in pg.GUIDES.items():
        for step in guide.steps:
            if step.command.startswith("sudo ") or " sudo " in step.command:
                assert step.warnings, f"{key}: '{step.title}' runs sudo with no warning"


def test_no_em_dashes_in_generated_copy():
    # House style, and it is enforced rather than trusted because the source runbooks are
    # full of them.
    for key, guide in pg.GUIDES.items():
        blob = " ".join(
            f"{guide.label}{guide.applies_to}{guide.recommended_os}"
            + f"{s.title}{s.what}{s.command}{s.expect}{' '.join(s.warnings)}"
            for s in guide.steps
        )
        assert "—" not in blob and "–" not in blob, f"{key} contains an em/en dash"


def test_module_is_pure():
    # Same contract as setup_advisor: this module advises, it never acts. It is the module
    # whose printed commands touch firmware, so purity here is not a formality.
    src = Path(pg.__file__).read_text(encoding="utf-8")
    for token in ("subprocess", "os.system", "os.popen", "Popen", " exec(", " eval("):
        assert token not in src, f"platform_guide must stay pure; found {token!r}"
    imports = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    assert not any("tools" in ln for ln in imports), "must not import the executor module"
