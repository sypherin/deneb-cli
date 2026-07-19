"""Unit tests for deneb.setup_advisor — the pure, tell-only setup step generation
(SET-01, SET-02, SET-03, QA-01).

setup_steps(model, profile) returns the ORDERED advisory steps for THIS model on THIS box:
runtime(s) -> download -> run, platform-branched from the curated PLAYBOOK, with every
sudo / N-GB / service step flagged inline. It is PURE (struct in, Step structs out — no I/O,
no tools, no subprocess, no LLM), so every step and warning is pinned against a CONSTRUCTED
HardwareProfile with no live tool present.

Load-bearing cases (the plan's canonical set):
  - ordered runtime->download->run pipeline; per-backend branching (rocm/cuda/cpu differ);
  - SET-03: every sudo/download/service step carries its warning inline;
  - best-quant = highest bpw that fits; under-spec + degraded-platform honest fallbacks;
  - vision mmproj surfaced; no em-dash in generated copy; resolve_model matching.

Run: python3 tests/test_setup_advisor.py   (also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from deneb.setup_advisor import Step, resolve_model, setup_steps  # noqa: E402


# ── constructed test boxes (mirror tests/test_recommend.py) ───────────────────
def strix_box(usable=107000):
    """128GB unified Strix Halo (gfx1151), rocm — this-box shape."""
    return HardwareProfile(
        primary_backend="rocm", usable_mem_mb=usable, os="Linux", arch="x86_64",
        gpus=[GPUInfo(vendor="amd", backend="rocm", memory_kind="unified",
                      unified_mem_mb=126908, extra={"gfx": "gfx1151", "sku": "STRXLGEN"})])


def rocm_plain_box(usable=48000):
    """A rocm box WITHOUT a gfx1151/gfx1150 GPU (no Strix caveat should fire)."""
    return HardwareProfile(
        primary_backend="rocm", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="amd", backend="rocm", memory_kind="dedicated",
                      vram_mb=usable, extra={"gfx": "gfx1100"})])


def cuda_box(usable):
    return HardwareProfile(
        primary_backend="cuda", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=usable,
                      memory_kind="dedicated")])


def cpu_box(usable=32000):
    return HardwareProfile(primary_backend="cpu", usable_mem_mb=usable, gpus=[])


def metal_box(usable=24000):
    return HardwareProfile(
        primary_backend="metal", usable_mem_mb=usable, os="Darwin", arch="arm64",
        gpus=[GPUInfo(vendor="apple", backend="metal", memory_kind="unified")])


def tiny_box(usable=1500):
    return HardwareProfile(
        primary_backend="cuda", usable_mem_mb=usable,
        gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=usable,
                      memory_kind="dedicated")])


def degraded_box():
    """A live profile that could read almost nothing: unknown backend + no budget (PLAT-03)."""
    return HardwareProfile(primary_backend="", usable_mem_mb=None, gpus=[])


# ── helpers ───────────────────────────────────────────────────────────────────
def _kinds(steps):
    return [s.kind for s in steps]


def _only(steps, kind):
    hits = [s for s in steps if s.kind == kind]
    assert len(hits) == 1, f"expected exactly one {kind} step, got {len(hits)}"
    return hits[0]


def _model(name="Qwen3.6-35B-A3B"):
    m = resolve_model(name)
    assert m is not None, f"catalog model {name} must resolve"
    return m


# ── ORDERED PIPELINE (SET-01) ──────────────────────────────────────────────────
def test_pipeline_is_ordered_runtime_download_run():
    steps = setup_steps(_model(), strix_box())
    assert steps, "must return a non-empty ordered list"
    kinds = _kinds(steps)
    assert "runtime" in kinds
    assert kinds.count("download") == 1
    assert kinds.count("run") == 1
    di, ri = kinds.index("download"), kinds.index("run")
    runtime_idxs = [i for i, k in enumerate(kinds) if k == "runtime"]
    assert runtime_idxs, "at least one runtime step"
    assert max(runtime_idxs) < di < ri, f"order must be runtime(s) -> download -> run: {kinds}"


def test_every_step_has_command_and_what():
    # strix + non-vision model => no note steps; every step must be fully populated.
    for s in setup_steps(_model(), strix_box()):
        assert isinstance(s, Step)
        assert s.what.strip(), f"empty what on {s.title}"
        if s.kind != "note":
            assert s.command.strip(), f"empty command on {s.title}"


# ── PLATFORM BRANCHING (SET-02) ─────────────────────────────────────────────────
def test_backends_branch_and_differ():
    m = _model()
    rocm = setup_steps(m, strix_box())
    cuda = setup_steps(m, cuda_box(48000))
    cpu = setup_steps(m, cpu_box(64000))

    rocm_run = _only(rocm, "run").command
    cuda_run = _only(cuda, "run").command
    cpu_run = _only(cpu, "run").command

    assert "-ngl 999" in rocm_run
    assert "-ngl 999" in cuda_run
    assert "-ngl" not in cpu_run                       # cpu has no GPU-offload flag

    assert any(("HIP" in s.command or "ROCm" in s.command or "HIP" in s.what
                or "ROCm" in s.what) for s in rocm if s.kind == "runtime")
    assert any("CUDA" in s.command or "CUDA" in s.what
               for s in cuda if s.kind == "runtime")

    # the whole step sets must not be identical across backends
    assert [s.command for s in rocm] != [s.command for s in cuda]
    assert [s.command for s in cuda] != [s.command for s in cpu]


def test_metal_uses_prebuilt_brew():
    steps = setup_steps(_model("Llama-3.1-8B-Instruct"), metal_box())
    runtime = [s for s in steps if s.kind == "runtime"]
    assert any("brew" in s.command for s in runtime)   # Metal = prebuilt via Homebrew


# ── SET-03: sudo / download-GB / service warnings inline ───────────────────────
def test_sudo_steps_carry_sudo_warning():
    seen_sudo = False
    for box in (strix_box(), cuda_box(48000), cpu_box(64000)):
        for s in setup_steps(_model(), box):
            if "sudo" in s.command:
                seen_sudo = True
                assert s.warnings, f"sudo step {s.title} has no warning"
                assert any("sudo" in w.lower() for w in s.warnings)
    assert seen_sudo, "at least one backend must produce a sudo step to exercise the rule"


def test_download_step_warns_gb_and_names_repo():
    m = _model()
    dl = _only(setup_steps(m, strix_box()), "download")
    assert m.gguf_repo in dl.command                   # names the real HF repo
    assert dl.warnings, "download step must warn"
    assert any("GB" in w for w in dl.warnings), "download warning must name the size in GB"


def test_run_step_flags_service_and_binds_localhost():
    run = _only(setup_steps(_model(), strix_box()), "run")
    assert run.warnings, "run step must warn"
    assert any(("127.0.0.1:8001" in w or "server" in w.lower()) for w in run.warnings)
    assert "127.0.0.1" in run.command                  # never 0.0.0.0 (T-03-12)
    assert "0.0.0.0" not in run.command


def test_gfx1151_caveat_only_on_strix_rocm():
    strix_run = _only(setup_steps(_model(), strix_box()), "run")
    assert any("gfx1151" in w for w in strix_run.warnings)     # Strix caveat fires
    plain_run = _only(setup_steps(_model(), rocm_plain_box()), "run")
    assert not any("gfx1151" in w for w in plain_run.warnings)  # non-Strix rocm: no caveat


# ── BEST-QUANT selection (consistent with recommend) ───────────────────────────
def test_best_quant_is_highest_bpw_that_fits():
    big = _only(setup_steps(_model(), strix_box(107000)), "download")
    assert "Q8_0" in big.command                       # highest-bpw quant fits 107GB
    tight = _only(setup_steps(_model("Qwen2.5-Coder-14B-Instruct"), cuda_box(10000)),
                  "download")
    assert "Q4_K_M" in tight.command                   # only the low quant fits 10GB
    assert "Q8_0" not in tight.command


def test_explicit_quant_override_is_honored():
    m = _model()
    q = m.quants[0]                                    # force the lowest quant
    dl = _only(setup_steps(m, strix_box(107000), quant=q), "download")
    assert q.name in dl.command


# ── UNDER-SPEC (never empty; honest warning) ───────────────────────────────────
def test_underspec_tiny_box_still_returns_steps_with_warning():
    steps = setup_steps(_model(), tiny_box(1500))
    assert steps, "under-spec box must still get advisory steps, never empty"
    dl = _only(steps, "download")
    run = _only(steps, "run")
    blob = " ".join(dl.warnings + run.warnings).lower()
    assert ("exceed" in blob or "tight" in blob or "oom" in blob or "budget" in blob)


# ── VISION mmproj (surface the projector) ──────────────────────────────────────
def test_vision_model_surfaces_mmproj():
    steps = setup_steps(_model("Qwen2.5-VL-7B-Instruct"), strix_box())
    text = " ".join(s.command + " " + s.what for s in steps).lower()
    assert "mmproj" in text, "vision model must surface the mmproj projector requirement"
    # still exactly one download + one run (the mmproj is an extra note, not a 2nd download)
    assert _kinds(steps).count("download") == 1
    assert _kinds(steps).count("run") == 1


# ── DEGRADED PLATFORM (PLAT-03 — never raises, never empty) ─────────────────────
def test_degraded_platform_falls_back_to_cpu_note():
    steps = setup_steps(_model(), degraded_box())      # must not raise
    assert steps, "degraded profile must still yield generic steps"
    notes = [s for s in steps if s.kind == "note"]
    assert notes, "degraded profile must include an honest fallback note"
    blob = " ".join(s.what.lower() for s in notes)
    assert ("generic" in blob or "cpu" in blob or "could not read" in blob)
    # cpu fallback => the run step has no -ngl flag
    assert "-ngl" not in _only(steps, "run").command


def test_setup_steps_never_raises_on_junk():
    # a totally malformed model object must degrade to a single honest note, not raise.
    class _Junk:
        pass
    steps = setup_steps(_Junk(), degraded_box())
    assert isinstance(steps, list) and steps


# ── NO EM-DASH (house style) ────────────────────────────────────────────────────
def test_no_emdash_in_generated_copy():
    for box in (strix_box(), cuda_box(48000), cpu_box(64000), tiny_box(1500),
                metal_box(), degraded_box()):
        for name in ("Qwen3.6-35B-A3B", "Qwen2.5-VL-7B-Instruct"):
            for s in setup_steps(_model(name), box):
                assert "—" not in s.what, f"em-dash in what: {s.what}"
                for w in s.warnings:
                    assert "—" not in w, f"em-dash in warning: {w}"


# ── resolve_model (exact / case / separator insensitive) ───────────────────────
def test_resolve_model_matching():
    assert resolve_model("Qwen3.6-35B-A3B") is not None
    assert resolve_model("qwen3.6-35b-a3b") is not None          # case-insensitive
    assert resolve_model("  QWEN3 6 35B A3B  ") is not None      # separator/space-insensitive
    assert resolve_model("qwen3635ba3b") is not None             # all separators dropped
    assert resolve_model("no-such-model") is None
    assert resolve_model("") is None
    assert resolve_model(None) is None


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
