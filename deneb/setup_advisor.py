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

import re
import sys
from dataclasses import dataclass, field, replace

from . import platform_guide
from .recommend import best_fitting_quant
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

    kind ∈ {"runtime", "download", "run", "service", "ollama", "note"}; "ollama" is the
    route through an Ollama already serving, taken INSTEAD of the others. `warnings` is every applicable inline
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
    pkgs: str = ""   # "" | "base" | "rocm": a package-manager step, rendered per distro family


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

# Build prerequisites per distro family (curated best-effort). Kept once, rendered by
# _runtime_steps; an unrecognised distro gets an honest note instead of a guessed command.
_PKGS = {
    "apt": {"base": "build-essential cmake git libcurl4-openssl-dev", "rocm": "rocm-hip-sdk",
            "vulkan": "libvulkan-dev glslc"},
    "dnf": {"base": "gcc-c++ make cmake git libcurl-devel", "rocm": "rocm-hip-devel hipblas-devel rocblas-devel",
            "vulkan": "vulkan-headers vulkan-loader-devel glslc"},
    "pacman": {"base": "base-devel cmake git curl", "rocm": "rocm-hip-sdk",
               "vulkan": "vulkan-headers vulkan-icd-loader shaderc"},
}
_LLAMA_DIR = "~/llama.cpp"
_LLAMA_SERVER = f"{_LLAMA_DIR}/build/bin/llama-server"   # where the build step leaves it
_CLONE = f"git clone https://github.com/ggml-org/llama.cpp {_LLAMA_DIR} && cd {_LLAMA_DIR}"
_HF_INSTALL = "pipx install huggingface_hub"
_SERVICE_NAME = "llama-server.service"

# ── THE CURATED PLAYBOOK (SET-02) — one branch per accelerator, templates not per-model ──
PLAYBOOK = {
    "cuda": Playbook(
        label="CUDA (NVIDIA)",
        runtime=[
            RuntimeStep(
                "Install build prerequisites (CUDA)",
                "",
                "installs the compiler, CMake and build prerequisites. You also need the "
                "NVIDIA CUDA toolkit installed.",
                needs_sudo=True, pkgs="base"),
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
                "",
                "installs the compiler, CMake and the ROCm/HIP SDK build prerequisites.",
                needs_sudo=True, pkgs="rocm"),
            RuntimeStep(
                "Build llama.cpp (ROCm/HIP)",
                f"{_CLONE} && HIPCXX=$(hipconfig -l)/clang cmake -B build -DGGML_HIP=ON && "
                "cmake --build build --config Release -j",
                "clones and compiles llama.cpp with ROCm/HIP acceleration for your AMD GPU",
                is_build=True),
        ],
        run_flags="-ngl 999",
        run_warning=_GFX_CAVEAT),
    # Not a default for any box: only `deneb setup --measured` selects it, when the best measured
    # TokenMark row for an AMD GPU is a llama.cpp Vulkan build.
    "vulkan": Playbook(
        label="Vulkan (AMD, measured pick)",
        runtime=[
            RuntimeStep(
                "Install build prerequisites (Vulkan)",
                "",
                "installs the compiler, CMake, the Vulkan headers and the glslc shader compiler. "
                "On Ubuntu older than 24.04 there is no glslc package: install the LunarG Vulkan "
                "SDK or Google's shaderc instead.",
                needs_sudo=True, pkgs="vulkan"),
            RuntimeStep(
                "Build llama.cpp (Vulkan)",
                f"{_CLONE} && cmake -B build -DGGML_VULKAN=ON && "
                "cmake --build build --config Release -j",
                "clones and compiles llama.cpp with the Vulkan backend, the build the measured "
                "TokenMark row used",
                is_build=True),
        ],
        run_flags="-ngl 999"),
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
                "",
                "installs the compiler and CMake build prerequisites.",
                needs_sudo=True, pkgs="base"),
            RuntimeStep(
                "Build llama.cpp (CPU)",
                f"{_CLONE} && cmake -B build && cmake --build build --config Release -j",
                "clones and compiles llama.cpp for CPU-only inference",
                is_build=True),
        ],
        run_flags=""),
}


# ── distro family from /etc/os-release text (pure; the caller reads the file) ──
_PM_BY_FAMILY = {"debian": "apt", "fedora": "dnf", "arch": "pacman"}
_FAMILY_BY_PM = {v: k for k, v in _PM_BY_FAMILY.items()}


def distro_family(os_release) -> str:
    """'apt' | 'dnf' | 'pacman' | 'unknown' from the text of /etc/os-release. The mapping
    itself lives in platform_guide (one place); anything it does not recognise is 'unknown'
    and nothing is guessed."""
    from . import platform_guide
    return _PM_BY_FAMILY.get(platform_guide.distro_family(str(os_release or "")), "unknown")


# ── resolve_model — case/separator-insensitive catalog lookup (never raises) ──
def _norm_key(name) -> str:
    """Comparison key: lower-case, stripped, with spaces / '-' / '.' / '_' dropped, so
    'qwen3.6-35b-a3b' and 'Qwen3.6-35B-A3B' collapse to the same key."""
    key = str(name).strip().lower() if name is not None else ""
    for ch in (" ", "-", ".", "_"):
        key = key.replace(ch, "")
    return key


def resolve_model(name, catalog=None) -> "Model | None":
    """Match `name` against the catalog (default: the static CATALOG; the CLI passes
    hf_catalog.all_models()) case- and separator-insensitively; return the Model on a
    match, None on no match (or empty/None input). Never raises."""
    try:
        target = _norm_key(name)
        if not target:
            return None
        for m in (CATALOG if catalog is None else catalog):
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
_UNIT_WARNING = ("starts a local server on 127.0.0.1:8001 - it stays running until you stop "
                 f"it: `systemctl --user stop {_SERVICE_NAME}`.")
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
    """quant.size_mb (decimal MB, bytes / 1e6) -> whole GiB, rounded UP: the same unit as the
    free-space reading (bytes // 2**30), so the two numbers in the disk note compare directly.
    0 when unreadable."""
    try:
        return -(-int(size_mb) * 1_000_000 // 1024 ** 3)
    except (TypeError, ValueError):
        return 0


def download_gib(quant) -> int:
    """One quant's download in whole GiB, by the same maths as setup's disk note, so
    `deneb recommend` and `deneb setup` print one number for one file. 0 when unreadable."""
    return _size_gb(getattr(quant, "size_mb", 0) or 0)


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
    """The advised quant: the caller's explicit `quant`, else the quant `recommend` picks for
    this model (recommend.best_fitting_quant: one rule, shared, so the two commands never
    disagree - a roomy fit beats a tight one, then the highest bpw). If NONE fit, the
    LOWEST-bpw quant with an under-spec flag. Returns (quant | None, under_spec: bool).
    Never raises."""
    quants = list(getattr(model, "quants", None) or [])
    if quant is not None:
        return quant, False
    if not quants:
        return None, True
    try:
        best, _fr = best_fitting_quant(model, profile)
    except Exception as e:  # noqa: BLE001 - keep the fallback, but not silently
        print(f"deneb: couldn't pick a quant for this box ({type(e).__name__}); "
              "showing the lowest-bpw one", file=sys.stderr)
        best = None
    if best is not None:
        return best, False
    low = min(quants, key=lambda q: float(getattr(q, "bpw", 0) or 0))
    return low, True


# A Grace Hopper box (a GH200 on Lambda): the toolkit packages are a different build from x86's.
# Only Hopper GPUs on Arm get this; DGX Spark (GB10) and Jetson ship their own CUDA stack and
# keep their existing steps.
_ARM_CUDA_WARNING = ("aarch64 (Arm) Grace Hopper box: use a CUDA 12 toolkit built for arm64/sbsa (NVIDIA's "
                     "cuda-ubuntu2204-sbsa apt repository, or the one Lambda Stack installs), not "
                     "x86_64 packages. Ubuntu 22.04's own nvidia-cuda-toolkit is CUDA 11.5, too "
                     "old for Hopper (sm_90). Check `nvcc --version`; if nvcc is not found, run "
                     "`export PATH=/usr/local/cuda/bin:$PATH` first.")


def is_grace_hopper(profile) -> bool:
    """An Arm (aarch64) box with a Hopper GPU: a GH200. False for GB10 (DGX Spark), Jetson and
    Graviton boxes, whose CUDA setup is unchanged."""
    if (getattr(profile, "arch", "") or "").lower() not in ("aarch64", "arm64"):
        return False
    return any(re.search(r"gh200|grace.?hopper|\bh[12]00\b", (getattr(g, "name", "") or "").lower())
               for g in getattr(profile, "gpus", None) or [])


def _runtime_steps(pb: "Playbook", family: str = "unknown", grace_hopper: bool = False) -> list:
    """Build the ordered runtime Steps from a Playbook branch, stamping the SET-03 sudo
    warning on any sudo/system-install step and a compile note on any build step. A package
    step is rendered for the distro family; an unrecognised family gets a note, no command."""
    steps = []
    for rs in pb.runtime:
        command, what, kind = rs.command, rs.what, "runtime"
        if rs.pkgs:
            fam = _PKGS.get(family)
            if fam is None:
                steps.append(Step(
                    title=rs.title, command="", kind="note", warnings=[],
                    what=_plain("your Linux distribution was not recognised, so no package "
                                "command is guessed. Install a C++ compiler, make, CMake, git "
                                "and the libcurl development headers"
                                + (" and the ROCm/HIP SDK" if rs.pkgs == "rocm" else "")
                                + (" and the Vulkan headers and glslc" if rs.pkgs == "vulkan" else "")
                                + " with your package manager, then continue.")))
                continue
            pkgs = fam["base"] + (f" {fam[rs.pkgs]}" if rs.pkgs in ("rocm", "vulkan") else "")
            command = platform_guide.install_command(_FAMILY_BY_PM[family], pkgs)
        if rs.pkgs == "rocm" and family == "apt":
            what += (" On Debian/Ubuntu rocm-hip-sdk comes from AMD's own apt repository, "
                     "not the distro's: add that repo first, following the install guide at "
                     "https://rocm.docs.amd.com")
        warnings = []
        if rs.needs_sudo or "sudo" in command:
            warnings.append(_SUDO_WARNING)
        if rs.is_build:
            warnings.append(_BUILD_WARNING)
            if pb.label.startswith("CUDA") and grace_hopper:
                warnings.append(_ARM_CUDA_WARNING)
        steps.append(Step(title=rs.title, command=command, what=_plain(what),
                          warnings=[_plain(w) for w in warnings], kind=kind))
    return steps


def _weights_glob(basename: str, labels: tuple) -> str:
    """Shell expression for the FIRST weights file of exactly this quant: the single file or
    shard 00001 (llama.cpp loads the other shards itself), in the repo root or its quant
    subfolder. Exact-token match so Q6_K never picks up UD-Q6_K_XL, and MTP draft heads
    are skipped. `labels` are the quant's file labels (Quant.file_labels)."""
    names = " -o ".join(f"-name '*-{lb}.gguf' -o -name '*-{lb}-00001-of-*.gguf'" for lb in labels)
    return (f'"$(find ~/models/{basename} -not -path \'*/MTP/*\' \\( {names} \\) '
            f'| sort | head -1)"')


def _mmproj_expr(basename: str) -> str:
    """Shell expression for the ONE projector file: the first mmproj GGUF in the model folder.
    The download step fetches exactly one, so there is one to find; the run step and the
    service step both use this expression so they can never pick differently."""
    return f'"$(ls ~/models/{basename}/*mmproj*.gguf | head -1)"'


def _run_step(pb: "Playbook", profile, basename: str, labels: tuple,
              vision: bool, under_spec: bool) -> Step:
    """The llama-server RUN step: platform-branched flags, mmproj on vision, always bound to
    127.0.0.1 (never 0.0.0.0 - exposure is out of scope for v1). SET-03 service warning +
    the gfx1151 Strix caveat on rocm + the under-spec warning when it applies."""
    exe = "llama-server" if _on_path(pb) else _LLAMA_SERVER   # brew puts it on PATH; a build does not
    parts = [exe, "-m", _weights_glob(basename, labels)]
    if vision:
        parts += ["--mmproj", _mmproj_expr(basename)]
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


def _on_path(pb: "Playbook") -> bool:
    """True when the runtime step installs a package (llama-server lands on PATH) rather than
    building llama.cpp into _LLAMA_DIR."""
    return not any(rs.is_build for rs in pb.runtime)


def _need_file(var: str, message: str) -> str:
    """Shell guard: continue only when $var is non-empty; otherwise print `message` on stderr and
    fail, so the steps after it (writing and starting a unit) do not run."""
    return f'{{ [ -n "${var}" ] || {{ echo "{message}" >&2; false; }}; }}'


def _projector_landed(repo: str, basename: str) -> str:
    """Shell guard after the projector download: hf prints "Downloaded" and exits 0 when its
    --include filter matches nothing in the repo, so check that a projector is on disk where the
    run and service steps look (~/models/<dir>/*mmproj*.gguf). If not, say so plainly and print
    the command that lists the repo's projector files and the BF16 download. A `false` inside
    braces, never `exit`: it must not close the shell the user pasted it into."""
    msg = (f"no vision projector (mmproj) landed in ~/models/{basename}: the F16 filter matched nothing "
           f"in {repo}. List the repo's projector files with: "
           f"curl -s 'https://huggingface.co/api/models/{repo}/tree/main?recursive=1' "
           "| grep -o 'mmproj[A-Za-z0-9._-]*gguf' | sort -u   "
           f"Or download the BF16 one: hf download {repo} "
           f"--include '*mmproj*BF16.gguf' --local-dir ~/models/{basename}")
    return (f'{{ ls ~/models/{basename}/*mmproj*.gguf >/dev/null 2>&1 || '
            f'{{ echo "{msg}" >&2; false; }}; }}')


def _service_step(pb: "Playbook", basename: str, labels: tuple, vision: bool) -> "Step | None":
    """A systemd USER unit that runs the built server, so `deneb check` and `deneb watch` can
    find it and it survives a reboot. None when llama-server comes from a package manager on
    macOS (no systemd there). The weights path is resolved by the shell first, because a unit
    cannot expand a glob; the unit file itself gets absolute paths (%h = the home directory)."""
    if _on_path(pb):
        return None
    exe = _LLAMA_SERVER.replace("~", "%h", 1)
    exec_parts = [exe, "-m", "$MODEL"]
    pre = (f"MODEL={_weights_glob(basename, labels)} && "
           + _need_file("MODEL", f"no model file found under ~/models/{basename}: "
                                 "run the download step first"))
    if vision:
        pre += (f" && MMPROJ={_mmproj_expr(basename)} && "
                + _need_file("MMPROJ", f"no vision projector (mmproj) found under ~/models/{basename}: "
                                       "run the vision projector download step first"))
        exec_parts += ["--mmproj", "$MMPROJ"]
    if pb.run_flags:
        exec_parts.append(pb.run_flags)
    exec_parts += ["--host", "127.0.0.1", "--port", "8001"]
    unit = ("[Unit]\nDescription=llama.cpp model server (127.0.0.1 only)\n\n[Service]\n"
            f"ExecStart={' '.join(exec_parts)}\nRestart=on-failure\n\n"
            "[Install]\nWantedBy=default.target\n")
    path = f"~/.config/systemd/user/{_SERVICE_NAME}"
    # one command line: the heredoc body follows it, so a failed guard also skips systemctl.
    # `restart`, not `enable --now`: --now only starts an inactive unit, so re-running this step
    # for a new model would leave the old one serving.
    command = (f"{pre} && mkdir -p ~/.config/systemd/user && cat > {path} <<EOF && "
               f"systemctl --user daemon-reload && systemctl --user enable {_SERVICE_NAME} && "
               f"systemctl --user restart {_SERVICE_NAME}\n"
               f"{unit}EOF")
    return Step(
        title="Keep it running (systemd user service)", command=command, kind="service",
        what=_plain("writes a user service for the server above and starts it, so it restarts "
                    "after a crash and `deneb check` / `deneb watch` can find and name it."),
        warnings=[_plain(_UNIT_WARNING)])


# A user service lives only while its owner is logged in: over SSH the server stops at logout and
# starts again at the next login (QC on a cloud GPU, 20 Oct). A command, not a warning line, so
# following the steps top to bottom keeps it up.
_LINGER_STEP = Step(
    title="Keep it running after you log out", command="loginctl enable-linger \"$(id -un)\"", kind="linger",
    what=_plain("lets your user services run without a login session. Without it the model server "
                "stops when you log out (an SSH session ending counts) and only starts again at your "
                "next login."),
    warnings=[_plain("if it asks for a password or says access denied, run it with sudo in front: "
                     "sudo loginctl enable-linger \"$(id -un)\".")])


_OLLAMA_SPLIT_ISSUE = "https://github.com/ollama/ollama/issues/5245"


def _one_file_quant(model, profile):
    """(quant, any): the quant setup would advise if only one-file quants existed (Ollama can
    pull those), or None when none fits this box; `any` is whether the model has one at all."""
    ones = [q for q in getattr(model, "quants", None) or [] if getattr(q, "parts", 0) == 1]
    if not ones:
        return None, False
    try:
        best, _fr = best_fitting_quant(replace(model, quants=ones), profile)
    except Exception:  # noqa: BLE001 - no alternative is the honest fallback; the step says so
        return None, True
    return best, True


def _ollama_step(port: int, repo: str, quant, one_file, any_one_file: bool, vision: bool,
                 store=None) -> Step:
    """The route through an Ollama that already serves: pull the advised quant into it.
    Ollama can't pull a GGUF split into several files, so a split quant points at the best
    one-file quant that fits (`one_file`), or says plainly that Ollama can't take it. A
    catalog that never recorded the split gets the command and the way out. `store` is
    (free GiB, path) for where Ollama keeps models, read by the caller, or None."""
    parts = getattr(quant, "parts", 0) or 0
    qname = getattr(quant, "name", "") or ""
    # the ollama command talks to :11434 unless OLLAMA_HOST says otherwise
    ollama = "ollama" if port == 11434 else f"OLLAMA_HOST=127.0.0.1:{port} ollama"
    if port == 8001:
        second = _plain("this Ollama serves on :8001, the port the llama.cpp steps below use: "
                        "their server can't start until Ollama stops or moves to another port.")
    else:
        second = _plain(f"the llama.cpp steps below start a second server on :8001, and both hold "
                        f"a model in memory: `{ollama} ps` shows what Ollama has loaded, and "
                        f"`{ollama} stop <name>` frees it.")
    lead = ""
    if parts > 1:
        lead = (f"the {qname} quant the llama.cpp steps below use is split into {parts} files on "
                f"Hugging Face, and Ollama can't pull a split GGUF yet ({_OLLAMA_SPLIT_ISSUE}). ")
        if one_file is None:
            # deneb speaks for the quants its catalog lists: the repo may hold others
            why = ("No one-file quant of it fits this box" if any_one_file
                   else "Every quant deneb lists for it is split")
            return Step(title=f"Ollama on :{port} can't take this model's quants", command="",
                        what=_plain(lead + f"{why}, so use the llama.cpp steps below."),
                        warnings=[second], kind="note")
        quant, parts = one_file, 1
    name = getattr(quant, "name", "") or ""
    need_gb = _size_gb(getattr(quant, "size_mb", 0) or 0)
    warnings = []
    if store and store[0] is not None and need_gb > store[0]:
        warnings.append(_plain(
            f"this needs about {need_gb} GiB but only {store[0]} GiB is free on {store[1]}, "
            "which deneb takes to be Ollama's model store. If this Ollama keeps models "
            "somewhere else (OLLAMA_MODELS, or a per-user ollama serve), check that disk "
            "instead. If not, free some space before you pull."))
    warnings += [_plain(f"downloads ~{need_gb} GiB into "
                        "Ollama's own model store, not ~/models."),
                _plain("Ollama runs only the model architectures its build supports: if the "
                       "pull or the first request fails, use the llama.cpp steps below.")]
    if parts == 0:
        warnings.append(_plain(
            "deneb's catalog has not recorded whether this quant is one file. If Ollama says "
            "the tag is a sharded GGUF or can't find it, use the llama.cpp steps below; "
            "`deneb catalog --refresh` records it."))
    if vision:
        warnings.append(_plain("the pull also brings the model's vision projector, up to about "
                               "1 GiB more. deneb has not checked vision through Ollama."))
    warnings.append(second)
    which = (f"{name}, the best one-file quant that fits this box (smaller and less exact "
             f"than {qname})," if lead else f"{name}, the same quant the llama.cpp steps below download,")
    return Step(
        title=f"Ollama route: pull it into the Ollama already serving on :{port}",
        # the last label: a catalog that never recorded the tag still gets MXFP4_MOE for MXFP4
        command=f"{ollama} pull hf.co/{repo}:{quant.file_labels()[-1] if isinstance(quant, Quant) else name}",
        what=_plain(lead + f"This pulls {which} into the Ollama you already run. Take this "
                    "route or the llama.cpp steps below, not both. Ollama loads the model on "
                    f"the first request, and `deneb check` finds it on :{port}."),
        warnings=warnings, kind="ollama")


def one_way_to_start(steps: list) -> list:
    """The steps as a person should follow them top to bottom: when a user service starts the
    server, the foreground run step is dropped (running both starts the server twice, on the same
    port) and its warnings move to the service step. Pure; setup_steps keeps both for callers."""
    svc = next((s for s in steps if s.kind == "service"), None)
    run = next((s for s in steps if s.kind == "run"), None)
    if svc is None or run is None:
        return list(steps)
    # the foreground run's "stop it (Ctrl-C)" does not apply to a unit: the unit's own stop line
    # replaces it (the run step still carries it for callers that show the run step alone)
    own = [_plain(_SERVICE_WARNING)]
    merged = [w for w in run.warnings if w not in own] + [w for w in svc.warnings if w not in run.warnings]
    merged = [w for w in svc.warnings if w == _plain(_UNIT_WARNING)] + [w for w in merged if w != _plain(_UNIT_WARNING)]
    new_svc = Step(title="Run the model and keep it running (systemd user service)",
                   command=svc.command, what=svc.what, warnings=merged, kind=svc.kind)
    return [new_svc if s is svc else s for s in steps if s is not run]


# ── the public entry point (SET-01, SET-02, SET-03; PURE, never raises) ────────
def setup_steps(model, profile, quant=None, family: str = "unknown", free_gb=None,
                models_dir: str = "~/models", serving=None, ollama_free=None,
                measured=None) -> list:
    """The ORDERED tell-only setup advice for `model` on `profile`: runtime(s) -> download ->
    run, platform-branched from the PLAYBOOK, every sudo/download/service step warned inline.

    Backend resolves from profile.primary_backend; an unknown/empty backend degrades to the
    generic 'cpu' branch plus an honest NOTE (PLAT-03). The advised quant is the highest-bpw
    quant that fits (else the lowest, with an under-spec warning). Vision models get an extra
    mmproj projector note. PURE - builds and returns Step structs, executes NOTHING. The whole
    body is guarded so any malformed input degrades to a single honest note, never raises.

    `family` is the distro family from `distro_family()` (default 'unknown': no package command
    is guessed). `free_gb` is the free space on the download directory, read by the caller;
    when it is below the download size a note saying so comes first. `serving` is
    check.serving_now()'s (port, kind, model) or None: an Ollama serving gets its own step
    before the llama.cpp ones. `ollama_free` is (free GiB, path) where that Ollama keeps
    models, read by the caller: its step warns when the pull would not fit there.
    `measured` is a `deneb setup --measured` pick (measured.Choice): its .quant, .playbook (the
    llama.cpp build the row used) and .flags (extra llama-server flags) replace the defaults.
    None keeps the catalog default, byte for byte."""
    try:
        backend = (getattr(profile, "primary_backend", "") or "").strip().lower()
        degraded = backend not in PLAYBOOK
        if degraded:
            backend = "cpu"
        if measured is not None:
            quant = measured.quant
            if measured.playbook in PLAYBOOK and not degraded:
                backend = measured.playbook
        pb = PLAYBOOK[backend]
        if measured is not None and measured.flags:
            pb = replace(pb, run_flags=f"{pb.run_flags} {measured.flags}".strip())

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
        labels = chosen.file_labels() if isinstance(chosen, Quant) else (qname,)

        # (0) DISK — say first if the download cannot fit where it will be written.
        need_gb = _size_gb(size_mb)
        if free_gb is not None and need_gb > free_gb:
            steps.append(Step(
                title="Not enough free disk space",
                command="",
                what=_plain(f"the download needs about {need_gb} GiB but only {free_gb} GiB is "
                            f"free on {models_dir}. Free some space or point the download at "
                            "another drive before you start."),
                warnings=[], kind="note"))

        # (0b) OLLAMA — one already serving can take this quant itself, or say why it can't.
        if serving and serving[1] == "ollama":
            steps.append(_ollama_step(serving[0], repo, chosen, *_one_file_quant(model, profile),
                                      vision=vision, store=ollama_free))

        # (1) RUNTIME — stand up llama.cpp for this accelerator.
        steps += _runtime_steps(pb, family, is_grace_hopper(profile))
        steps.append(Step(
            title="Install the Hugging Face CLI (hf)", command=_HF_INSTALL, kind="runtime",
            what=_plain("installs the `hf` command the download step uses, with pipx (the same "
                        "tool that installs deneb; see https://pipx.pypa.io if you do not have "
                        "it). Make sure ~/.local/bin is on PATH."),
            warnings=[]))

        # (2) DOWNLOAD — the chosen GGUF, warning naming the size in GB (SET-03).
        dl_warnings = [_plain(f"downloads ~{_size_gb(size_mb)} GiB to ~/models/{basename} - "
                              "check you have the disk space.")]
        if under_spec:
            dl_warnings.append(_plain(_UNDERSPEC_WARNING))
        includes = " ".join(f'--include "*-{lb}.gguf" --include "*-{lb}-0*.gguf"' for lb in labels)
        steps.append(Step(
            title=f"Download the {qname} GGUF" if qname else "Download the GGUF weights",
            command=f'hf download {repo} {includes} --exclude "MTP/*" '
                    f"--local-dir ~/models/{basename}",
            what=_plain(f"downloads the {qname} GGUF weights"),
            warnings=dl_warnings, kind="download"))

        # (2b) VISION — the mmproj projector alongside the weights (an extra note, not a 2nd
        #      download, so the pipeline stays runtime -> one download -> one run).
        if vision:
            steps.append(Step(
                title="Download the vision projector (mmproj)",
                command=f'hf download {repo} --include "*mmproj*-F16.gguf" --include "*mmproj*-f16.gguf" '
                        f"--local-dir ~/models/{basename} && " + _projector_landed(repo, basename),
                what=_plain("downloads the F16 mmproj projector GGUF (one file, not the BF16 and F32 "
                            "copies) - vision models need it alongside the weights to "
                            "process images"),
                warnings=[_plain("downloads an extra projector file (a few hundred MB to "
                                 "over 1 GiB).")],
                kind="note"))

        # (3) RUN — llama-server with platform flags, bound to 127.0.0.1 (SET-03 service).
        steps.append(_run_step(pb, profile, basename, labels, vision, under_spec))

        # (4) SERVICE — a user unit running that same binary, the one check/watch look for.
        svc = _service_step(pb, basename, labels, vision)
        if svc is not None:
            steps += [svc, _LINGER_STEP]

        return steps
    except Exception:  # noqa: BLE001 — the advisory must degrade, never crash the caller (PLAT-03)
        return [Step(
            title="Could not build setup steps",
            command="",
            what=_plain("could not build setup steps for this model/box - verify the model "
                        "name and re-run; the Deneb Rule means nothing was executed."),
            warnings=[], kind="note")]
