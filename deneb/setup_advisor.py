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


# ── SET-03 warning text (one place each; stamped inline by setup_steps) ────────
_SUDO_WARNING = "needs sudo - installs system packages / drivers with root privilege."
_BUILD_WARNING = ("compiles llama.cpp from source - can take several minutes and needs the "
                  "build prerequisites above.")
_SERVICE_WARNING = ("starts a local server on 127.0.0.1:8001 - it stays running until you "
                    "stop it (Ctrl-C).")
_UNDERSPEC_WARNING = ("may exceed your box's memory budget - expect a tight fit or OOM; "
                      "consider a smaller model or CPU offload.")

# AMD GFX versions that carry the gfx1151/Strix ROCm-kernel caveat (mirrors fit.py).
_CAVEAT_GFX = {"gfx1151", "gfx1150"}


# ── small pure helpers for setup_steps ─────────────────────────────────────────
def _repo_basename(repo) -> str:
    """The last path segment of an HF repo id -> the local-dir name (e.g.
    'Qwen/Qwen3-30B-A3B-GGUF' -> 'Qwen3-30B-A3B-GGUF'). 'model' when empty."""
    r = str(repo or "").strip().rstrip("/")
    return r.split("/")[-1] if r else "model"


def _size_gb(size_mb) -> int:
    """quant.size_mb -> whole GB (round(size_mb/1024)); 0 when unreadable."""
    try:
        return int(round(int(size_mb) / 1024))
    except (TypeError, ValueError):
        return 0


def _is_vision(model) -> bool:
    """True when the model is a vision model (a 'vision' capability, or its notes mention the
    mmproj projector). Vision models need the mmproj GGUF alongside the weights."""
    try:
        if "vision" in (getattr(model, "capabilities", None) or []):
            return True
        return "mmproj" in (getattr(model, "notes", "") or "").lower()
    except Exception:  # noqa: BLE001
        return False


def _has_caveat_gfx(profile) -> bool:
    """True when ANY GPU reports a gfx1151/gfx1150 (the Strix Halo ROCm-kernel caveat)."""
    try:
        for g in (getattr(profile, "gpus", None) or []):
            extra = getattr(g, "extra", None)
            if isinstance(extra, dict) and str(extra.get("gfx", "")).lower() in _CAVEAT_GFX:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _advised_quant(model, profile, quant):
    """The advised quant: the caller's explicit `quant`, else the HIGHEST-bpw quant that
    fits() this box (same rule recommend uses). If NONE fit, the LOWEST-bpw quant with an
    under-spec flag. Returns (quant | None, under_spec: bool). Never raises."""
    quants = list(getattr(model, "quants", None) or [])
    if quant is not None:
        return quant, False
    if not quants:
        return None, True
    best = None  # (bpw, quant)
    for q in quants:
        try:
            if fits(model, q, profile).fits:
                bpw = float(getattr(q, "bpw", 0) or 0)
                if best is None or bpw > best[0]:
                    best = (bpw, q)
        except Exception:  # noqa: BLE001
            continue
    if best is not None:
        return best[1], False
    low = min(quants, key=lambda q: float(getattr(q, "bpw", 0) or 0))
    return low, True


def _runtime_steps(pb: "Playbook") -> list:
    """Build the ordered runtime Steps from a Playbook branch, stamping the SET-03 sudo
    warning on any sudo/system-install step and a compile note on any build step."""
    steps = []
    for rs in pb.runtime:
        warnings = []
        if rs.needs_sudo or "sudo" in rs.command:
            warnings.append(_SUDO_WARNING)
        if rs.is_build:
            warnings.append(_BUILD_WARNING)
        steps.append(Step(title=rs.title, command=rs.command, what=_plain(rs.what),
                          warnings=[_plain(w) for w in warnings], kind="runtime"))
    return steps


def _run_step(pb: "Playbook", profile, basename: str, qname: str,
              vision: bool, under_spec: bool) -> Step:
    """The llama-server RUN step: platform-branched flags, mmproj on vision, always bound to
    127.0.0.1 (never 0.0.0.0 - exposure is out of scope for v1). SET-03 service warning +
    the gfx1151 Strix caveat on rocm + the under-spec warning when it applies."""
    parts = ["llama-server", "-m", f"~/models/{basename}/*{qname}*.gguf"]
    if vision:
        parts += ["--mmproj", f"~/models/{basename}/mmproj-*.gguf"]
    if pb.run_flags:
        parts.append(pb.run_flags)
    parts += ["--host", "127.0.0.1", "--port", "8001"]
    warnings = [_plain(_SERVICE_WARNING)]
    if pb.run_warning and _has_caveat_gfx(profile):
        warnings.append(_plain(pb.run_warning))
    if under_spec:
        warnings.append(_plain(_UNDERSPEC_WARNING))
    return Step(title="Run the model (llama-server)", command=" ".join(parts),
                what=_plain("starts the model server locally"), warnings=warnings, kind="run")


# ── the public entry point (SET-01, SET-02, SET-03; PURE, never raises) ────────
def setup_steps(model, profile, quant=None) -> list:
    """The ORDERED tell-only setup advice for `model` on `profile`: runtime(s) -> download ->
    run, platform-branched from the PLAYBOOK, every sudo/download/service step warned inline.

    Backend resolves from profile.primary_backend; an unknown/empty backend degrades to the
    generic 'cpu' branch plus an honest NOTE (PLAT-03). The advised quant is the highest-bpw
    quant that fits (else the lowest, with an under-spec warning). Vision models get an extra
    mmproj projector note. PURE - builds and returns Step structs, executes NOTHING. The whole
    body is guarded so any malformed input degrades to a single honest note, never raises."""
    try:
        backend = (getattr(profile, "primary_backend", "") or "").strip().lower()
        degraded = backend not in PLAYBOOK
        if degraded:
            backend = "cpu"
        pb = PLAYBOOK[backend]

        steps: list = []
        if degraded:
            steps.append(Step(
                title="Platform not confidently detected",
                command="",
                what=_plain("could not read your platform confidently - showing generic "
                            "CPU-safe steps; verify the accelerator flags for your box."),
                warnings=[], kind="note"))

        chosen, under_spec = _advised_quant(model, profile, quant)
        qname = getattr(chosen, "name", "") or ""
        size_mb = getattr(chosen, "size_mb", 0) or 0
        repo = getattr(model, "gguf_repo", "") or ""
        basename = _repo_basename(repo)
        vision = _is_vision(model)

        # (1) RUNTIME — stand up llama.cpp for this accelerator.
        steps += _runtime_steps(pb)

        # (2) DOWNLOAD — the chosen GGUF, warning naming the size in GB (SET-03).
        dl_warnings = [_plain(f"downloads ~{_size_gb(size_mb)} GB to ~/models/{basename} - "
                              "check you have the disk space.")]
        if under_spec:
            dl_warnings.append(_plain(_UNDERSPEC_WARNING))
        steps.append(Step(
            title=f"Download the {qname} GGUF" if qname else "Download the GGUF weights",
            command=f'hf download {repo} --include "*{qname}*.gguf" '
                    f"--local-dir ~/models/{basename}",
            what=_plain(f"downloads the {qname} GGUF weights"),
            warnings=dl_warnings, kind="download"))

        # (2b) VISION — the mmproj projector alongside the weights (an extra note, not a 2nd
        #      download, so the pipeline stays runtime -> one download -> one run).
        if vision:
            steps.append(Step(
                title="Download the vision projector (mmproj)",
                command=f'hf download {repo} --include "*mmproj*.gguf" '
                        f"--local-dir ~/models/{basename}",
                what=_plain("downloads the mmproj projector GGUF - vision models need it "
                            "alongside the weights to process images"),
                warnings=[_plain("downloads an extra projector file (usually a few "
                                 "hundred MB).")],
                kind="note"))

        # (3) RUN — llama-server with platform flags, bound to 127.0.0.1 (SET-03 service).
        steps.append(_run_step(pb, profile, basename, qname, vision, under_spec))

        return steps
    except Exception:  # noqa: BLE001 — the advisory must degrade, never crash the caller (PLAT-03)
        return [Step(
            title="Could not build setup steps",
            command="",
            what=_plain("could not build setup steps for this model/box - verify the model "
                        "name and re-run; the Deneb Rule means nothing was executed."),
            warnings=[], kind="note")]
