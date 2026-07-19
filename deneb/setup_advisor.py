"""deneb.setup_advisor — the PURE, tell-only setup advisory core (SET-01, SET-02, SET-03).

`setup_steps(model, profile)` answers "how do I actually set up THIS model on MY box?"
as ORDERED ADVICE: (1) stand up a runtime (build/prebuilt llama.cpp per accelerator),
(2) download the chosen GGUF, (3) run llama-server with platform-correct flags. Each Step
carries the exact command, a one-line plain-English "what it does", and an inline warning on
anything that needs sudo / downloads N GB / starts a service (SET-03).

This is THE DENEB RULE at the source: Deneb TELLS the steps and NEVER executes them. The
module is PURE — string/struct in, Step structs out. It imports no shell-out layer, no
process-spawning, no system-exec and no LLM; it prints nothing and runs nothing. Plan 03-02
wires this into `deneb setup <model>` (the display + the formal no-execution QA gate).

⚠ HONESTY CAVEAT (mirrors models_catalog): the PLAYBOOK commands are CURATED BEST-EFFORT
   advice, NOT machine-verified for every box (distro, driver and kernel differ). The Deneb
   Rule means Deneb only TELLS them - the user (or the later paid executor tier) runs them.
   Verify before running. Repo ids + sizes come from the catalog, itself a static v1 guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fit import fits
from .models_catalog import CATALOG, Model, Quant  # noqa: F401  (typing / interface anchor)

# ── house style: no em/en dashes in generated copy (recommend.py's rule) ──────
_DASHES = ("—", "–")


def _plain(text: str) -> str:
    """Replace em/en dashes with a spaced hyphen (house style; catalog notes carry them)."""
    for d in _DASHES:
        text = text.replace(d, "-")
    return text


# ── the advice unit (SET-01) ──────────────────────────────────────────────────
@dataclass
class Step:
    """One ordered setup instruction shown to the user (never executed here).

    kind ∈ {"runtime", "download", "run", "note"}. `warnings` is every applicable inline
    risk (sudo / N GB / starts a service); empty when the step carries no risk.
    """
    title: str
    command: str
    what: str
    warnings: list = field(default_factory=list)
    kind: str = "note"


# ── the curated PLAYBOOK data model (SET-02) ──────────────────────────────────
@dataclass
class RuntimeStep:
    """One ordered runtime-provisioning step template in a Playbook. `needs_sudo` and
    `is_build` drive the SET-03 warning stamping in setup_steps (kept as flags in the DATA
    so the warning text lives in one place, not sprinkled through the playbook)."""
    title: str
    command: str
    what: str
    needs_sudo: bool = False
    is_build: bool = False


@dataclass
class Playbook:
    """The per-accelerator branch (SET-02): a human label, the ORDERED runtime steps that
    stand up llama.cpp for that accelerator, the llama-server run flags for it, and any
    backend run-warning (e.g. the gfx1151/Strix kernel caveat on rocm, applied conditionally
    by setup_steps when that GPU is actually present)."""
    label: str
    runtime: list          # list[RuntimeStep]
    run_flags: str         # backend llama-server flags ("-ngl 999" on an accelerator, "" on cpu)
    run_warning: str = ""  # extra backend run-warning text (conditional in setup_steps)


# The Strix Halo / gfx1151 ROCm-kernel caveat (kept as DATA on the rocm branch; applied only
# when the box actually reports a gfx1151/gfx1150 GPU). Mirrors the fit.py caveat wording.
_GFX_CAVEAT = ("gfx1151 (Strix Halo): needs a recent kernel with the ROCm fixes - older "
               "kernels have known gfx1151 instability that can hang the box.")

# Debian/Ubuntu build prerequisites (curated best-effort; adapt for your distro). Kept once.
_APT_BASE = "sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev"
_CLONE = "git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"

# ── THE CURATED PLAYBOOK (SET-02) — one branch per accelerator, templates not per-model ──
PLAYBOOK = {
    "cuda": Playbook(
        label="CUDA (NVIDIA)",
        runtime=[
            RuntimeStep(
                "Install build prerequisites (CUDA)",
                _APT_BASE,
                "installs the compiler, CMake and build prerequisites (Debian/Ubuntu; adapt "
                "for your distro). You also need the NVIDIA CUDA toolkit installed.",
                needs_sudo=True),
            RuntimeStep(
                "Build llama.cpp (CUDA)",
                f"{_CLONE} && cmake -B build -DGGML_CUDA=ON && "
                "cmake --build build --config Release -j",
                "clones and compiles llama.cpp with CUDA GPU acceleration",
                is_build=True),
        ],
        run_flags="-ngl 999"),
    "rocm": Playbook(
        label="ROCm (AMD)",
        runtime=[
            RuntimeStep(
                "Install build prerequisites (ROCm)",
                f"{_APT_BASE} rocm-hip-sdk",
                "installs the compiler, CMake and the ROCm/HIP SDK build prerequisites "
                "(Debian/Ubuntu; adapt for your distro).",
                needs_sudo=True),
            RuntimeStep(
                "Build llama.cpp (ROCm/HIP)",
                f"{_CLONE} && HIPCXX=$(hipconfig -l)/clang cmake -B build -DGGML_HIP=ON && "
                "cmake --build build --config Release -j",
                "clones and compiles llama.cpp with ROCm/HIP acceleration for your AMD GPU",
                is_build=True),
        ],
        run_flags="-ngl 999",
        run_warning=_GFX_CAVEAT),
    "metal": Playbook(
        label="Metal (Apple)",
        runtime=[
            RuntimeStep(
                "Install llama.cpp (Metal, prebuilt via Homebrew)",
                "brew install llama.cpp",
                "installs a prebuilt llama.cpp with Metal (Apple GPU) acceleration via "
                "Homebrew - no compile needed on macOS."),
        ],
        run_flags="-ngl 999"),
    "cpu": Playbook(
        label="CPU (generic)",
        runtime=[
            RuntimeStep(
                "Install build prerequisites (CPU)",
                _APT_BASE,
                "installs the compiler and CMake build prerequisites (Debian/Ubuntu; adapt "
                "for your distro).",
                needs_sudo=True),
            RuntimeStep(
                "Build llama.cpp (CPU)",
                f"{_CLONE} && cmake -B build && cmake --build build --config Release -j",
                "clones and compiles llama.cpp for CPU-only inference",
                is_build=True),
        ],
        run_flags=""),
}


# ── resolve_model — case/separator-insensitive catalog lookup (never raises) ──
def _norm_key(name) -> str:
    """Comparison key: lower-case, stripped, with spaces / '-' / '.' / '_' dropped, so
    'qwen3.6-35b-a3b' and 'Qwen3.6-35B-A3B' collapse to the same key."""
    key = str(name).strip().lower() if name is not None else ""
    for ch in (" ", "-", ".", "_"):
        key = key.replace(ch, "")
    return key


def resolve_model(name) -> "Model | None":
    """Match `name` against CATALOG case- and separator-insensitively; return the Model on a
    match, None on no match (or empty/None input). PURE, never raises. (Plan 03-02's CLI turns
    None into a helpful error + a `deneb recommend` pointer.)"""
    try:
        target = _norm_key(name)
        if not target:
            return None
        for m in CATALOG:
            if _norm_key(getattr(m, "name", "")) == target:
                return m
        return None
    except Exception:  # noqa: BLE001 — a lookup must never crash the caller
        return None
