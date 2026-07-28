"""deneb.platform_guide — the PRE-FLIGHT runbook: how to make the BOX ready, before any model.

`setup_advisor` answers "how do I run THIS model", and assumes the machine underneath is
already sane. On the two boxes Deneb actually targets that assumption is wrong, and wrong in
ways that do not announce themselves: a Strix Halo with the factory BIOS carveout will run,
then die under sustained load; a DGX Spark without matching kernel headers will install the
driver and then fail at `nvidia-smi`. Both look like model problems and are not.

This module carries that platform layer as DATA, distilled from two runbooks that were
written against real hardware rather than documentation:

  * the on-premise private-LLM runbook (Strix Halo OR DGX Spark, written for a client's IT
    team, hardware-verified end to end)
  * the maintained Strix Halo host guide (github.com/sypherin/strix-halo-setup)

It is PURE, in the same sense as the rest of the advisory core: strings and structs in,
structs out. It imports no shell-out layer, spawns nothing, prints nothing. THE DENEB RULE
holds here exactly as it does in setup_advisor - Deneb TELLS you the step and never runs it.
That matters more here than anywhere else in the codebase, because these steps edit the
bootloader and the BIOS.

⚠ HONESTY CAVEAT: these are curated, best-effort instructions verified on specific hardware
   and specific distro versions, stated inline per step. Firmware menus differ between board
   vendors and revisions. Read each step before running it, and never paste a GRUB line you
   have not read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── house style: no em/en dashes in generated copy (recommend.py's rule) ──────
_DASHES = ("—", "–")


def _plain(text: str) -> str:
    """Replace em/en dashes with a spaced hyphen (house style)."""
    for d in _DASHES:
        text = text.replace(d, "-")
    return text


# ── distro awareness ──────────────────────────────────────────────────────────
# The existing PLAYBOOK in setup_advisor is Debian/Ubuntu only, but the runbook's
# recommended OS for Strix Halo is Fedora. Telling a Fedora user to `apt-get install
# rocm-hip-sdk` is advice that cannot work, so the family has to be a known fact rather
# than an assumption.

_FAMILY_BY_ID = {
    "fedora": "fedora", "rhel": "fedora", "centos": "fedora", "rocky": "fedora",
    "almalinux": "fedora", "nobara": "fedora",
    "ubuntu": "debian", "debian": "debian", "pop": "debian", "linuxmint": "debian",
    "arch": "arch", "cachyos": "arch", "endeavouros": "arch", "manjaro": "arch",
}

_INSTALL_BY_FAMILY = {
    "fedora": "sudo dnf -y install",
    "debian": "sudo apt-get install -y",
    "arch": "sudo pacman -S --noconfirm",
}


def distro_family(os_release_text: str) -> str:
    """Family from the contents of /etc/os-release. "" when it cannot be determined.

    PURE: takes the already-read file CONTENTS, never reads the file itself, so the mapping
    is unit-testable against captured fixtures the way hardware.py parses tool output.

    ID_LIKE is honoured, because derivatives ("ID=nobara", "ID_LIKE=fedora") are common and
    a miss here produces a package command that silently cannot run.
    """
    ident, like = "", ""
    for raw in (os_release_text or "").splitlines():
        line = raw.strip()
        if line.startswith("ID="):
            ident = line[3:].strip().strip('"\'').lower()
        elif line.startswith("ID_LIKE="):
            like = line[8:].strip().strip('"\'').lower()
    if ident in _FAMILY_BY_ID:
        return _FAMILY_BY_ID[ident]
    for token in like.split():
        if token in _FAMILY_BY_ID:
            return _FAMILY_BY_ID[token]
    return ""


def install_command(family: str, packages: str) -> str:
    """The install command for a family. Falls back to a named-but-unknown form rather than
    guessing apt, so a user on an unrecognised distro sees that Deneb does not know instead
    of being handed a command that will not run."""
    prefix = _INSTALL_BY_FAMILY.get((family or "").lower())
    if not prefix:
        return f"# install with your distro's package manager: {packages}"
    return f"{prefix} {packages}"


# ── the guide unit ────────────────────────────────────────────────────────────
@dataclass
class GuideStep:
    """One ordered pre-flight instruction. `command` may be empty for steps that happen in
    firmware, where there is nothing to type and the instruction IS the text."""
    title: str
    what: str
    command: str = ""
    warnings: list = field(default_factory=list)
    verify: str = ""            # command that proves the step worked
    expect: str = ""            # what a correct verify looks like


@dataclass
class PlatformGuide:
    label: str
    applies_to: str
    recommended_os: str
    steps: list                 # list[GuideStep]
    source: str = ""


_BIOS_WARNING = "changes firmware settings - reboot into BIOS; menu names differ by vendor."
_BOOT_WARNING = ("edits the bootloader - a malformed line can leave the box unbootable. "
                 "Read the line before writing it and keep the existing parameters.")
_REBOOT_WARNING = "requires a reboot before it takes effect."
_SUDO_WARNING = "needs sudo - changes system configuration with root privilege."

# ── Strix Halo ────────────────────────────────────────────────────────────────
# Every value here is one that was arrived at by something breaking first.
_STRIX = PlatformGuide(
    label="AMD Strix Halo (Ryzen AI Max, gfx1151)",
    applies_to="gfx1151 / gfx1150 iGPU, 128 GB unified memory",
    recommended_os="Fedora 43 (Fedora 44 carries more RDNA 3.5 fixes in mesa 26.0)",
    source="on-prem private LLM runbook + github.com/sypherin/strix-halo-setup",
    steps=[
        GuideStep(
            title="BIOS: set UMA Frame Buffer Size to 512 MB",
            what=_plain(
                "AMD section -> Integrated Graphics Configuration -> UMA Frame Buffer Size "
                "(sometimes 'GPU Memory' or 'VRAM Size'). Counter-intuitive but correct: a "
                "LARGE carveout is what causes trouble. The kernel statically reserves it as "
                "fake VRAM and cannot reclaim it, which produces TTM eviction storms during "
                "long inference. At 512 MB the weights live in GTT - dynamically reclaimable "
                "system memory - which is the path the driver is actually tuned for."),
            warnings=[_BIOS_WARNING],
        ),
        GuideStep(
            title="Check the kernel is new enough",
            what=_plain(
                "gfx1151 stability fixes landed in 6.18.4. Below that the box has known "
                "instability that can hang it hard, with nothing useful in the logs."),
            command="uname -r",
            verify="uname -r",
            expect="6.18.4 or newer (Fedora 43 ships 7.0.x by default)",
        ),
        GuideStep(
            title="Confirm the GPU is actually recognised",
            what=_plain(
                "Proves the kernel has gfx1151 support before anything is built against it. "
                "If the expected name does not appear, no amount of model tuning will help."),
            command="vulkaninfo | grep deviceName | head -1",
            verify="vulkaninfo | grep deviceName | head -1",
            expect="AMD Radeon Graphics (RADV STRIX_HALO)",
        ),
        GuideStep(
            title="Set the kernel command line",
            what=_plain(
                "Add these to GRUB_CMDLINE_LINUX in /etc/default/grub, KEEPING what is "
                "already there. amdgpu.cwsr_enable=0 disables Compute Wave Save/Resume, "
                "which triggers MES firmware hangs on this chip. iommu=pt cuts DMA overhead. "
                "gttsize and ttm.pages_limit MUST be set together and must agree - a "
                "mismatch between them is the single most common cause of 'device lost "
                "under load'."),
            command="amdgpu.cwsr_enable=0 iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856",
            warnings=[_BOOT_WARNING, _REBOOT_WARNING],
        ),
        GuideStep(
            title="Regenerate the bootloader config",
            what="Writes the new command line into the boot configuration.",
            command="sudo grub2-mkconfig -o /boot/grub2/grub.cfg",
            warnings=[_SUDO_WARNING, _REBOOT_WARNING],
        ),
        GuideStep(
            title="Cap dirty writeback so heavy jobs cannot freeze the box",
            what=_plain(
                "tuned's throughput-performance profile sets dirty_ratio=40, which on a "
                "124 GiB box permits roughly 50 GiB of dirty pages before the kernel "
                "throttles writers. When that ceiling is hit every process touching the "
                "filesystem blocks, and the desktop stops responding. Worse, this kernel has "
                "hung-task detection compiled out, so the freeze logs absolutely nothing. "
                "Set absolute byte limits instead, via a tuned profile so the setting is not "
                "reverted by a power-profile change. Diagnostic signature: load average very "
                "high while CPU pressure is near zero and io pressure is high."),
            command="cat /proc/sys/vm/dirty_bytes /proc/sys/vm/dirty_background_bytes",
            verify="cat /proc/sys/vm/dirty_bytes",
            expect="8589934592 (8 GiB), not 0 - a 0 means the ratio-based default is still active",
            warnings=["use dirty_bytes, not dirty_ratio: the ratio scales with RAM and is "
                      "deprecated because it does not support profile inheritance."],
        ),
    ],
)

# ── DGX Spark ─────────────────────────────────────────────────────────────────
_DGX = PlatformGuide(
    label="NVIDIA DGX Spark (GB10 Blackwell)",
    applies_to="GB10 Blackwell, 128 GB unified memory",
    recommended_os="Ubuntu 22.04 LTS (NVIDIA's reference platform)",
    source="on-prem private LLM runbook",
    steps=[
        GuideStep(
            title="Install Ubuntu 22.04 LTS Server",
            what=_plain(
                "NVIDIA's reference platform for this hardware. DGX OS bundles drivers and "
                "tools but is heavier and tied to NVIDIA's release cadence; plain Ubuntu LTS "
                "stays more flexible."),
        ),
        GuideStep(
            title="Add NVIDIA's CUDA repository",
            what="Points apt at NVIDIA's packages rather than the distro's older ones.",
            command=(
                "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/"
                "ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && "
                "sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt update"),
            warnings=[_SUDO_WARNING],
        ),
        GuideStep(
            title="Install the driver and CUDA toolkit",
            what="Installs the proprietary driver plus the CUDA toolkit llama.cpp builds against.",
            command="sudo apt -y install nvidia-driver-570 cuda-toolkit-12-4",
            warnings=[_SUDO_WARNING, _REBOOT_WARNING, "downloads several GB."],
            verify="nvidia-smi",
            expect="a table listing the GB10 and a driver version",
        ),
        GuideStep(
            title="If nvidia-smi fails, match the kernel headers",
            what=_plain(
                "The usual cause is headers that do not match the running kernel, so the "
                "driver module never built. Install the matching headers and reinstall the "
                "driver rather than reinstalling CUDA."),
            command="sudo apt -y install linux-headers-$(uname -r)",
            warnings=[_SUDO_WARNING],
        ),
    ],
)

GUIDES = {"strix-halo": _STRIX, "dgx-spark": _DGX}


def detect_platform(profile) -> str:
    """Best-effort platform key from a HardwareProfile. "" when it is neither.

    Never raises: a probe that came back degraded must produce "unknown", not a traceback,
    because the caller's job is to print advice either way.
    """
    try:
        for gpu in (getattr(profile, "gpus", None) or []):
            gfx = str((getattr(gpu, "extra", None) or {}).get("gfx", "")).lower()
            if gfx.startswith("gfx115"):
                return "strix-halo"
            name = f"{getattr(gpu, 'name', '')} {getattr(gpu, 'vendor', '')}".lower()
            if "gb10" in name or "dgx" in name:
                return "dgx-spark"
            if "strix" in name:
                return "strix-halo"
        return ""
    except Exception:  # noqa: BLE001 - advice must never crash the caller
        return ""


def guide_for(key: str) -> "PlatformGuide | None":
    """Look up a guide by key, case/separator-insensitively. None when unknown."""
    k = str(key or "").strip().lower().replace("_", "-").replace(" ", "-")
    return GUIDES.get(k)
