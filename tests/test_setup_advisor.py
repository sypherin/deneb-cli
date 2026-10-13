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
from deneb import platform_guide, setup_advisor as _sa  # noqa: E402
from deneb.setup_advisor import Step, resolve_model, distro_family  # noqa: E402


def setup_steps(model, profile, quant=None, **kw):
    """Existing tests assume a Debian-family box; the distro is now an explicit input."""
    kw.setdefault("family", "apt")
    return _sa.setup_steps(model, profile, quant, **kw)


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
    assert any("GiB" in w for w in dl.warnings), "download warning must name the size in GiB"


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


# ── #40: package commands branch by distro family; the hf CLI is installed first ──
def test_distro_family_from_os_release():
    assert distro_family("ID=fedora\nVERSION_ID=44\n") == "dnf"
    assert distro_family('ID="rocky"\nID_LIKE="rhel centos fedora"\n') == "dnf"
    assert distro_family("ID=ubuntu\nID_LIKE=debian\n") == "apt"
    assert distro_family("ID=linuxmint\nID_LIKE=\"ubuntu debian\"\n") == "apt"
    assert distro_family("ID=arch\n") == "pacman"
    assert distro_family("ID=gentoo\n") == "unknown"
    assert distro_family("") == "unknown"
    assert distro_family(None) == "unknown"


def test_prereq_command_follows_the_distro_family():
    m = _model()
    def prereq(fam, box=None):
        steps = _sa.setup_steps(m, box or cpu_box(64000), family=fam)
        return [s for s in steps if s.kind == "runtime" and "prerequisites" in s.title][0]
    assert "dnf -y install" in prereq("dnf").command and "apt-get" not in prereq("dnf").command
    assert "gcc-c++" in prereq("dnf").command
    assert "pacman -S" in prereq("pacman").command and "apt-get" not in prereq("pacman").command
    assert "apt-get install" in prereq("apt").command
    assert "rocm-hip" in prereq("dnf", strix_box()).command


def test_unknown_distro_says_so_and_guesses_no_command():
    steps = _sa.setup_steps(_model(), cpu_box(64000), family="unknown")
    note = [s for s in steps if "prerequisites" in s.title][0]
    assert note.command == "" and note.kind == "note"
    assert "not recognised" in note.what and "package manager" in note.what
    assert not any("apt-get" in s.command or "dnf" in s.command or "pacman" in s.command
                   for s in steps)
    omitted = _sa.setup_steps(_model(), cpu_box(64000))
    assert not any("apt-get" in s.command for s in omitted)


def test_hf_cli_is_installed_before_the_download_step():
    for fam in ("apt", "dnf", "pacman", "unknown"):
        steps = _sa.setup_steps(_model(), cpu_box(64000), family=fam)
        di = _kinds(steps).index("download")
        hf = [i for i, s in enumerate(steps) if "huggingface_hub" in s.command]
        assert hf and hf[0] < di, f"{fam}: hf CLI install must precede the download"
        assert steps[hf[0]].kind == "runtime"


def test_hf_install_is_pipx_for_every_family_and_never_pip_or_the_cli_extra():
    """PEP 668 distros refuse `pip install --user`, and huggingface_hub has no [cli] extra now."""
    for fam in ("apt", "dnf", "pacman", "unknown"):
        for box in (cpu_box(64000), strix_box(), cuda_box(24000), metal_box()):
            steps = _sa.setup_steps(_model(), box, family=fam)
            text = "\n".join(f"{s.command}\n{s.what}\n" + "\n".join(s.warnings) for s in steps)
            assert "pip install" not in text, f"{fam}: {text}"
            assert "[cli]" not in text
            hf = [s for s in steps if "huggingface_hub" in s.command]
            assert [s.command for s in hf] == ["pipx install huggingface_hub"]


def test_setup_and_recommend_advise_the_same_quant_for_every_catalog_model():
    from deneb import recommend as _rec
    from deneb.models_catalog import CATALOG
    for usable in (24000, 64000, 107000):
        box = strix_box(usable)
        for m in CATALOG:
            uc = "general" if "general" in m.capabilities else m.capabilities[0]
            recs = _rec.recommend(box, uc, top_n=999, catalog=[m])
            assert len(recs) == 1
            steps = _sa.setup_steps(m, box, family="apt")
            dl = _only(steps, "download")
            want = recs[0].quant.name
            assert f'"*-{want}.gguf"' in dl.command, (usable, m.name, want, dl.command)


# ── vision projector: exactly one file downloaded, one path passed, same pick everywhere ──
def _vision_steps():
    m = _model("Qwen3.8-27B")
    assert _sa._is_vision(m)
    return setup_steps(m, strix_box())


def _sh(script, home, timeout=20):
    """Run printed shell text with HOME=<scratch dir>. Callers pass text that only expands
    (printf / find / ls) or writes under the scratch HOME; never systemctl or llama-server."""
    import subprocess
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin"})


def test_the_vision_download_fetches_exactly_one_projector():
    import fnmatch, shlex
    steps = _vision_steps()
    dl = [s for s in steps if s.kind == "note" and "projector" in s.title][0]
    pats = [shlex.split(dl.command)[i + 1] for i, t in enumerate(shlex.split(dl.command)) if t == "--include"]
    for repo_files in (["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"],
                       ["mmproj-Qwen-f16.gguf", "mmproj-Qwen-Q8_0.gguf", "mmproj-Qwen-bf16.gguf"]):
        hit = [f for f in repo_files if any(fnmatch.fnmatchcase(f, p) for p in pats)]
        assert len(hit) == 1, (pats, repo_files, hit)


def test_with_three_projectors_on_disk_the_run_step_passes_exactly_one(tmp_path):
    steps = _vision_steps()
    run = _only(steps, "run")
    base = _sa._repo_basename(_model("Qwen3.8-27B").gguf_repo)
    d = tmp_path / "models" / base
    d.mkdir(parents=True)
    for n in ("mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"):
        (d / n).write_text("x")
    qname = _sa._advised_quant(_model("Qwen3.8-27B"), strix_box(), None)[0].name
    (d / f"Qwen3.8-27B-{qname}.gguf").write_text("x")
    assert run.command.startswith(_sa._LLAMA_SERVER)
    script = run.command.replace(_sa._LLAMA_SERVER, "printf '<%s>\\n'", 1)
    r = _sh(script, tmp_path)
    assert r.returncode == 0, r.stderr
    toks = r.stdout.split()
    i = toks.index("<--mmproj>")
    after = []
    for t in toks[i + 1:]:
        if t.startswith("<-"):
            break
        after.append(t)
    assert len(after) == 1 and "mmproj" in after[0], toks


def test_run_and_service_steps_pick_the_projector_with_one_shared_expression():
    steps = _vision_steps()
    expr = _sa._mmproj_expr(_sa._repo_basename(_model("Qwen3.8-27B").gguf_repo))
    assert expr in _only(steps, "run").command
    assert f"MMPROJ={expr}" in _only(steps, "service").command


# ── service step: stops with a plain message when there is no model file; right stop command ──
def _unit_script(svc):
    """The service step's shell text with only the trailing systemctl call cut off, so the
    test can run it against a scratch HOME and never reach systemd."""
    tail = (" && systemctl --user daemon-reload && systemctl --user enable llama-server.service"
            " && systemctl --user restart llama-server.service")
    assert tail in svc.command, svc.command
    return svc.command.replace(tail, "")


def test_service_step_with_no_model_file_writes_no_unit_and_says_why(tmp_path):
    svc = _only(setup_steps(_model(), strix_box()), "service")
    r = _sh(_unit_script(svc), tmp_path)
    unit = tmp_path / ".config" / "systemd" / "user" / "llama-server.service"
    assert r.returncode != 0, "must fail loudly, not write a unit with an empty -m"
    assert not unit.exists()
    assert "no model file" in r.stderr.lower() and "download" in r.stderr.lower(), r.stderr


def test_service_step_with_the_model_present_writes_a_unit_with_real_paths(tmp_path):
    m = _model()
    svc = _only(setup_steps(m, strix_box()), "service")
    base = _sa._repo_basename(m.gguf_repo)
    qname = _sa._advised_quant(m, strix_box(), None)[0].name
    d = tmp_path / "models" / base
    d.mkdir(parents=True)
    (d / f"X-{qname}.gguf").write_text("x")
    r = _sh(_unit_script(svc), tmp_path)
    assert r.returncode == 0, r.stderr
    text = (tmp_path / ".config" / "systemd" / "user" / "llama-server.service").read_text()
    exec_line = [ln for ln in text.splitlines() if ln.startswith("ExecStart=")][0]
    assert f"-m {tmp_path}/models/{base}/X-{qname}.gguf" in exec_line and "-m  " not in exec_line


def test_a_unit_that_starts_the_server_is_stopped_with_systemctl_not_ctrl_c():
    steps = setup_steps(_model(), strix_box())
    svc = _only(steps, "service")
    assert any("systemctl --user stop llama-server.service" in w for w in svc.warnings)
    assert not any("Ctrl-C" in w for w in svc.warnings)
    assert any("Ctrl-C" in w for w in _only(steps, "run").warnings)       # a foreground run IS stopped with Ctrl-C
    merged = _sa.one_way_to_start(steps)
    only = _only(merged, "service")
    assert any("systemctl --user stop llama-server.service" in w for w in only.warnings)
    assert not any("Ctrl-C" in w for w in only.warnings)


# ── #41: compare the download with free disk on the target directory ──
def test_short_disk_is_the_first_thing_said_with_both_numbers_in_gib():
    m = _model()
    steps = _sa.setup_steps(m, strix_box(), family="apt", free_gb=3)
    first = steps[0]
    assert first.kind == "note" and "disk" in first.title.lower()
    assert "needs about 35 GiB" in first.what and "only 3 GiB is free" in first.what
    assert first.command == ""


def test_enough_disk_adds_no_warning_step():
    steps = _sa.setup_steps(_model(), strix_box(), family="apt", free_gb=4000)
    assert not any("disk" in s.title.lower() for s in steps)
    unknown = _sa.setup_steps(_model(), strix_box(), family="apt", free_gb=None)
    assert not any("disk" in s.title.lower() for s in unknown)


# ── #42: the serve step uses the path the build made; a user unit keeps it running ──
def test_run_step_uses_the_binary_the_build_step_produced():
    for box in (strix_box(), cuda_box(48000), cpu_box(64000)):
        steps = setup_steps(_model(), box)
        build = [s for s in steps if s.kind == "runtime" and "Build" in s.title][0]
        run = _only(steps, "run")
        assert "git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp" in build.command
        assert run.command.startswith("~/llama.cpp/build/bin/llama-server ")


def test_metal_run_uses_the_brew_binary_on_path():
    run = _only(setup_steps(_model("Llama-3.1-8B-Instruct"), metal_box()), "run")
    assert run.command.startswith("llama-server ")


def test_a_user_unit_step_creates_the_service_check_and_watch_expect():
    steps = setup_steps(_model(), strix_box())
    kinds = _kinds(steps)
    assert kinds.index("run") < kinds.index("service")
    svc = _only(steps, "service")
    assert "~/.config/systemd/user/llama-server.service" in svc.command
    assert "ExecStart=~/llama.cpp" not in svc.command          # systemd needs an absolute path
    assert "ExecStart=%h/llama.cpp/build/bin/llama-server" in svc.command
    assert "systemctl --user enable llama-server.service" in svc.command
    assert "WantedBy=default.target" in svc.command
    assert "0.0.0.0" not in svc.command and "--host 127.0.0.1" in svc.command
    assert "~/models" not in svc.command.split("ExecStart=")[1]  # %h, not ~, inside a unit


def test_cli_reads_distro_and_free_disk_without_touching_the_real_machine(monkeypatch, tmp_path):
    import builtins, io, shutil
    from deneb import cli
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: io.StringIO("ID=fedora\n")
                        if p == "/etc/os-release" else real_open(p, *a, **k))
    seen = {}
    class _U:
        free = 7 * 1024 ** 3
    monkeypatch.setattr(shutil, "disk_usage", lambda path: seen.setdefault("p", path) and _U())
    monkeypatch.setenv("HOME", str(tmp_path))           # ~/models does not exist here
    assert cli._setup_inputs() == ("dnf", 7)
    assert seen["p"] == str(tmp_path)                    # nearest existing parent of ~/models


def _rocm_prereq(fam):
    steps = _sa.setup_steps(_model(), strix_box(), family=fam)
    return [s for s in steps if s.kind == "runtime" and "prerequisites" in s.title][0]


def test_fedora_rocm_prereq_names_the_hip_blas_libraries_the_build_needs():
    # llama.cpp's ggml-hip CMake does find_package(hip), find_package(hipblas) and
    # find_package(rocblas) as REQUIRED, so rocm-hip-devel alone fails the configure step.
    cmd = _rocm_prereq("dnf").command
    for pkg in ("rocm-hip-devel", "hipblas-devel", "rocblas-devel"):
        assert pkg in cmd.split()


def test_apt_rocm_prereq_says_the_sdk_needs_amds_repo():
    step = _rocm_prereq("apt")
    text = step.what + " ".join(step.warnings)
    assert "rocm-hip-sdk" in step.command
    assert "https://rocm.docs.amd.com" in text and "repo" in text.lower()


def test_pacman_install_uses_needed():
    assert "pacman -S --needed" in _rocm_prereq("pacman").command
    assert "--needed" in platform_guide.install_command("arch", "git")


def test_dnf_and_pacman_rocm_notes_do_not_claim_an_amd_repo():
    assert "rocm.docs.amd.com" not in _rocm_prereq("dnf").what
    assert "rocm.docs.amd.com" not in _rocm_prereq("pacman").what


def test_disk_note_and_download_note_use_gib_with_one_set_of_maths():
    """catalog size_mb is decimal MB (bytes / 1e6); free space is read in GiB (bytes / 2**30).
    Both numbers are labelled GiB, and the need is converted from bytes, not divided by 1024."""
    class Q:
        name, size_mb, bpw = "Q4_K_M", 21474, 4.5     # 21,474,000,000 bytes = 19.9998 GiB
    steps = _sa.setup_steps(_model(), strix_box(), quant=Q(), family="apt", free_gb=19)
    note = steps[0]
    assert "needs about 20 GiB" in note.what and "only 19 GiB is free" in note.what, note.what
    assert " GB" not in note.what
    dl = [s for s in steps if s.kind == "download"][0]
    assert any("~20 GiB" in w for w in dl.warnings), dl.warnings
    assert not any(" GB " in w for w in dl.warnings)


# ── r4/R3: a projector download that matched nothing must not look like success ───────────
def _projector_step():
    return [s for s in _vision_steps() if s.kind == "note" and "projector" in s.title][0]


def _fake_hf_path(tmp_path, creates=None):
    """A fake PATH holding only a stub `hf` (and `ls`, `true` from the system's coreutils via
    the shell): the stub touches `creates` (relative to $HOME) when given, then exits 0, so the
    run proves what the printed text does AFTER an hf that 'succeeded'. Nothing is fetched."""
    import os, stat
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    hf = bindir / "hf"
    body = "#!/bin/sh\n"
    if creates:
        body += f'mkdir -p "$(dirname "$HOME/{creates}")" && : > "$HOME/{creates}"\n'
    body += "exit 0\n"
    hf.write_text(body)
    hf.chmod(hf.stat().st_mode | stat.S_IXUSR)
    return str(bindir)


def _run_step_text(cmd, tmp_path, creates=None):
    import subprocess
    base = _sa._repo_basename(_model("Qwen3.8-27B").gguf_repo)
    path = _fake_hf_path(tmp_path, creates.format(base=base) if creates else None)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    # `echo still-here` after the step proves the step did not exit the (interactive) shell
    return subprocess.run(["bash", "-c", cmd + "\necho still-here"], capture_output=True, text=True,
                          timeout=20, env={"HOME": str(home), "PATH": f"{path}:/usr/bin:/bin"})


def test_projector_download_that_matched_nothing_says_so_and_lists_the_repo_files(tmp_path):
    step = _projector_step()
    r = _run_step_text(step.command, tmp_path)           # stub hf exits 0 and writes nothing
    assert "still-here" in r.stdout, "the step must never exit the user's shell"
    assert "no vision projector" in r.stderr.lower(), (r.stdout, r.stderr)
    assert "tree/main" in r.stderr and "mmproj" in r.stderr         # the listing command
    assert "BF16" in r.stderr                                       # the alternative download
    assert "exit" not in step.command.split()


def test_projector_download_that_landed_a_file_is_quiet(tmp_path):
    r = _run_step_text(_projector_step().command, tmp_path, creates="models/{base}/mmproj-F16.gguf")
    assert "still-here" in r.stdout and r.stderr.strip() == "", r.stderr


def test_projector_check_looks_where_the_run_step_looks():
    base = _sa._repo_basename(_model("Qwen3.8-27B").gguf_repo)
    assert f"~/models/{base}/*mmproj*.gguf" in _projector_step().command


# ── r4/R4: re-running the service step for a new model must serve the NEW model ───────────
def test_service_step_restarts_the_unit_after_enabling_it():
    cmd = _only(setup_steps(_model(), strix_box()), "service").command
    # `enable --now` only starts an INACTIVE unit: a running one would keep serving the old model
    assert "enable --now" not in cmd
    enable = "systemctl --user enable llama-server.service"
    restart = "systemctl --user restart llama-server.service"
    assert enable in cmd and restart in cmd
    assert "systemctl --user daemon-reload" in cmd
    assert cmd.index("daemon-reload") < cmd.index(enable) < cmd.index(restart)


# ── r4/R7: the quant-pick fallback is kept but no longer silent ───────────────────────────
def test_a_failing_quant_pick_falls_back_and_names_the_exception_on_stderr(monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("kaput")
    monkeypatch.setattr(_sa, "best_fitting_quant", boom)
    chosen, under = _sa._advised_quant(_model(), strix_box(), None)
    err = capsys.readouterr().err
    assert chosen is not None and under is True         # the fallback still happens
    assert "ValueError" in err and err.count("\n") == 1, err
    assert "kaput" not in err                            # the type only: no message text echoed


def test_a_normal_quant_pick_writes_nothing_to_stderr(capsys):
    _sa._advised_quant(_model(), strix_box(), None)
    assert capsys.readouterr().err == ""


def test_best_fitting_quant_docstring_describes_the_real_rule():
    from deneb import recommend
    doc = " ".join(recommend.best_fitting_quant.__doc__.split())
    assert "headroom" in doc and "then" in doc and "highest-bpw one (best" not in doc
