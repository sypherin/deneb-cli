"""`deneb setup --measured`, the `deneb check` speed row, and GH200 / aarch64 detection.

The pick logic (deneb.measured.select) is pure, so it is pinned against tests/fixtures/
tokenmark_rows.json and constructed boxes. Nothing here touches the network or the real
~/.config/deneb (conftest isolates CONFIG_DIR) and nothing is installed or started.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deneb import check, cli, config, hardware, measured, setup_advisor, speed, tokenmark  # noqa: E402
from deneb import hf_catalog  # noqa: E402
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from fakes import FakeOpenAI  # noqa: E402
from test_setup_advisor import strix_box  # noqa: E402

FIX = json.loads((Path(__file__).parent / "fixtures" / "tokenmark_rows.json").read_text())
ROWS, TRACKED = FIX["configs"], FIX["tracked"]


def model(name="Qwen3.8-27B"):
    return setup_advisor.resolve_model(name, hf_catalog.all_models())


def a10_box():
    return HardwareProfile(primary_backend="cuda", usable_mem_mb=23000, arch="x86_64",
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda", name="NVIDIA A10",
                                         vram_mb=23028, memory_kind="dedicated")])


def gh200_box():
    return HardwareProfile(primary_backend="cuda", usable_mem_mb=95000, arch="aarch64",
                           cpu_model="Arm Neoverse-V2 (NVIDIA Grace)",
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda",
                                         name="NVIDIA GH200 480GB", vram_mb=97871,
                                         memory_kind="dedicated")])


# ── pick logic ────────────────────────────────────────────────────────────────
def test_strix_skips_fp4_and_takes_installable_vulkan_mtp_row():
    sel = measured.select(ROWS, TRACKED, model(), strix_box())
    c = sel.choice
    assert c is not None
    assert (c.quant.name, c.playbook) == ("UD-Q4_K_XL", "vulkan")
    assert "--spec-type draft-mtp" in c.flags and "-c 131072" in c.flags
    text = "\n".join(sel.lines)
    assert "24.6 tok/s" in text and "36.8 tok/s" in text          # both numbers named
    assert "FP4" in text and "https://tokenmark.app/c/udq4-vulkan-mtp" in text
    assert "measured on AMD Strix Halo (Ryzen AI Max+ 395), 2026-09-22" in text
    assert sel.expected["tok_s"] == 24.6 and sel.expected["quant"] == "UD-Q4_K_XL"


def test_aggregate_and_multi_stream_rows_never_picked():
    rows = [r for r in ROWS if r["hardware"] == "strix-halo" and r["throughput_kind"] == "single-stream"]
    cands, bad = measured._candidates(ROWS, "strix-halo", model())
    assert bad == 0
    assert {r["decode_tps"] for r in cands} == {r["decode_tps"] for r in rows}
    assert 120.0 not in {r["decode_tps"] for r in cands}


def test_a10_picks_best_llama_cpp_row_and_names_vllm_number():
    sel = measured.select(ROWS, TRACKED, model(), a10_box())
    c = sel.choice
    assert c.playbook == "cuda" and c.quant.name == "UD-Q3_K_XL"
    # the 41 tok/s row is vllm with GGUF weights deneb has no vLLM recipe for: named, then the
    # best llama.cpp row used
    text = "\n".join(sel.lines)
    assert "41 tok/s" in text and "vllm" in text and "no vLLM recipe" in text and "UD-Q4_K_XL" in text
    assert sel.expected["hardware"] == "a10"


def test_small_bench_ctx_is_not_served_with_dash_c():
    sel = measured.select(ROWS, TRACKED, model(), a10_box())
    assert sel.choice.quant.name == "UD-Q3_K_XL"
    # never below deneb's default context, and the user is told it differs from the row's
    assert "-c 32768" in sel.choice.flags
    assert any("2048 token context" in w and "32768" in w for w in sel.choice.warnings)


def test_mode_flags():
    assert measured._mode_flags({"mode": "mtp-4"}) == ("--spec-type draft-mtp --spec-draft-n-max 4", "")
    assert measured._mode_flags({"mode": "mtp"}) == ("--spec-type draft-mtp", "")
    assert measured._mode_flags({"mode": None}) == ("", "")
    assert measured._mode_flags({"mode": "baseline"}) == ("", "")
    assert measured._mode_flags({"mode": "dflash2"})[1]


def test_no_rows_for_hardware_one_line_default():
    sel = measured.select(ROWS, TRACKED, model(), gh200_box())
    assert sel.choice is None and sel.expected is None
    assert len(sel.lines) == 1
    assert "no rows for NVIDIA GH200" in sel.lines[0] and "catalog default" in sel.lines[0]


def test_no_rows_for_model_on_tracked_hardware():
    other = next(m for m in hf_catalog.all_models() if m.name != "Qwen3.8-27B")
    sel = measured.select(ROWS, TRACKED, other, strix_box())
    assert sel.choice is None and len(sel.lines) == 1
    assert f"no single-user rows for {other.name}" in sel.lines[0] and "catalog default" in sel.lines[0]


def test_nothing_installable_names_fastest_row():
    only_vllm = [r for r in ROWS if r["hardware"] == "a10" and r["backend"] == "vllm"]
    sel = measured.select(only_vllm, TRACKED, model(), a10_box())
    assert sel.choice is None
    assert "41 tok/s" in sel.lines[0] and "vllm" in sel.lines[0] and "catalog default" in sel.lines[0]


def test_row_that_does_not_fit_is_skipped():
    tiny = HardwareProfile(primary_backend="cuda", usable_mem_mb=11000, arch="x86_64",
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda", name="NVIDIA A10",
                                         vram_mb=11000, memory_kind="dedicated")])
    sel = measured.select(ROWS, TRACKED, model(), tiny)
    # every 27B quant above ~10 GB is out; the fixture's rows are Q4/Q3 (13+ GB) so none fits
    assert sel.choice is None and "no other row" in sel.lines[0]


def test_hostile_strings_are_cleaned():
    row = dict(ROWS[5], url="http://evil\nEOF", author="x\x1b[31mred", hardwareLabel="A\nB")
    sel = measured.select([row], TRACKED, model(), a10_box())
    text = "\n".join(sel.lines)
    assert "\x1b" not in text and "http://evil" not in text
    step = measured.record_step(sel.expected)
    assert step.command.count("\nEOF") == 1 and step.command.endswith("\nEOF")


def test_fetch_failures_fail_soft():
    assert measured.choose_for_box(model(), strix_box(), fetch=lambda: None).choice is None

    def boom():
        raise RuntimeError("boom")
    sel = measured.choose_for_box(model(), strix_box(), fetch=boom)
    assert sel.choice is None and "RuntimeError" in sel.lines[0] and "catalog default" in sel.lines[0]
    assert "could not be reached" in measured.choose_for_box(model(), strix_box(), fetch=lambda: None).lines[0]


def test_tracker_rows_fails_soft(monkeypatch):
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: None)
    assert tokenmark.tracker_rows() is None
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: {"configs": ROWS})
    rows, tracked = tokenmark.tracker_rows()
    assert "strix-halo" in tracked and len(rows) == len(ROWS)


# ── setup steps follow the pick ──────────────────────────────────────────────
def test_setup_steps_use_measured_quant_flags_and_vulkan_build():
    sel = measured.select(ROWS, TRACKED, model(), strix_box())
    steps = setup_advisor.setup_steps(model(), strix_box(), family="apt", measured=sel.choice)
    cmds = "\n".join(s.command for s in steps)
    assert "GGML_VULKAN=ON" in cmds
    assert "UD-Q4_K_XL" in cmds and "--spec-type draft-mtp" in cmds and "-c 131072" in cmds


def test_setup_steps_default_unchanged_without_measured():
    a = setup_advisor.setup_steps(model(), strix_box(), family="apt")
    b = setup_advisor.setup_steps(model(), strix_box(), family="apt", measured=None)
    assert a == b
    assert "GGML_VULKAN" not in "\n".join(s.command for s in a)


def _run_setup(monkeypatch, capsys, argv, fetch=None, box=None):
    monkeypatch.setattr(hardware, "profile_hardware", lambda: box or strix_box())
    monkeypatch.setattr(check, "serving_now", lambda: None)
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("apt", 500))
    calls = []
    monkeypatch.setattr(tokenmark, "tracker_rows", lambda: calls.append(1) or fetch)
    rc = cli.cmd_setup(argv)
    return rc, capsys.readouterr().out, calls


def test_default_setup_never_touches_tokenmark(monkeypatch, capsys):
    rc, out, calls = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B"], fetch=(ROWS, TRACKED))
    assert rc == 0 and calls == [] and "--measured" not in out and "expected.json" not in out


def test_setup_measured_prints_pick_and_record_step(monkeypatch, capsys):
    rc, out, calls = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B", "--measured"], fetch=(ROWS, TRACKED))
    assert rc == 0 and calls == [1]
    assert "24.6 tok/s" in out and "https://tokenmark.app/c/udq4-vulkan-mtp" in out
    assert "measured on AMD Strix Halo" in out and "expected.json" in out


def test_setup_measured_network_down_falls_back(monkeypatch, capsys):
    rc, out, _ = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B", "--measured"], fetch=None)
    assert rc == 0 and "could not be reached" in out and "expected.json" not in out
    _, plain, _ = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B"], fetch=None)
    assert "setup steps for" in plain


def test_setup_measured_unknown_hardware(monkeypatch, capsys):
    rc, out, _ = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B", "--measured"],
                            fetch=(ROWS, TRACKED), box=gh200_box())
    assert rc == 0 and "no rows for NVIDIA GH200" in out and "expected.json" not in out


# ── deneb check speed row ─────────────────────────────────────────────────────
def _write_expected(**kw):
    d = {"tok_s": 45.0, "model": "Qwen3.8-27B"}
    d.update(kw)
    Path(config.expected_path()).parent.mkdir(parents=True, exist_ok=True)
    Path(config.expected_path()).write_text(json.dumps(d))


def _serve(monkeypatch, tps, served="qwen3.8-27b"):
    monkeypatch.setattr(check, "serving_now", lambda: (8001, "llama.cpp", served))
    monkeypatch.setattr(check, "_decode_probe", lambda port, model=None: (tps, "") if tps else (None, "no answer"))


def test_speed_row_absent_without_record(monkeypatch):
    _serve(monkeypatch, 41.0)
    assert check._speed_row() is None


def test_speed_row_ok(monkeypatch):
    _write_expected()
    _serve(monkeypatch, 41.0)
    label, ok, ev, fix = check._speed_row()
    assert ok is True and "measured 45 tok/s on TokenMark, this box now 41 tok/s, ok" in ev


def test_speed_row_warns_below_70_percent(monkeypatch):
    _write_expected()
    _serve(monkeypatch, 20.0)
    label, ok, ev, fix = check._speed_row()
    assert ok is True and "⚠" in ev and "20 tok/s" in ev and fix


def test_speed_row_edge_exactly_70_percent_ok(monkeypatch):
    _write_expected(tok_s=100)
    _serve(monkeypatch, 70.0)
    assert "⚠" not in check._speed_row()[2]


def test_speed_row_other_model_not_compared(monkeypatch):
    _write_expected()
    _serve(monkeypatch, 41.0, served="llama-3.1-8b-instruct.gguf")
    label, ok, ev, fix = check._speed_row()
    assert ok is None and "not compared" in ev


def test_speed_row_gguf_filename_matches(monkeypatch):
    _write_expected()
    _serve(monkeypatch, 41.0, served="Qwen3.8-27B-UD-Q4_K_XL.gguf")
    assert check._speed_row()[1] is True


def test_speed_row_probe_failure_is_unknown(monkeypatch):
    _write_expected()
    _serve(monkeypatch, None)
    label, ok, ev, fix = check._speed_row()
    assert ok == check.UNKNOWN and "could not time" in ev


def test_speed_row_corrupt_record(monkeypatch):
    Path(config.expected_path()).parent.mkdir(parents=True, exist_ok=True)
    Path(config.expected_path()).write_text("{not json")
    _serve(monkeypatch, 41.0)
    label, ok, ev, fix = check._speed_row()
    assert ok is None and "not a usable record" in ev


def test_speed_row_nothing_serving(monkeypatch):
    _write_expected()
    monkeypatch.setattr(check, "serving_now", lambda: None)
    assert check._speed_row() is None


def test_speed_row_not_in_scan_so_watch_never_probes():
    assert all("speed" not in fn.__name__ for fn in check._CHECKS)


def test_decode_probe_real_http_path():
    # FakeOpenAI reports neither timings nor usage, so the probe reaches a 200 and says so
    with FakeOpenAI(["one, two, three"]) as srv:
        tps, why = check._decode_probe(srv.port)
        body = srv.chats()[-1][2]
    assert tps is None and "too short" in why
    assert body["max_tokens"] == check._PROBE_TOKENS and body["temperature"] == 0


def test_decode_probe_uses_server_timings(monkeypatch):
    monkeypatch.setattr(check.net, "local_post_json",
                        lambda *a, **k: (200, {"timings": {"predicted_per_second": 43.5}}))
    assert check._decode_probe(8001) == (43.5, "")
    monkeypatch.setattr(check.net, "local_post_json", lambda *a, **k: (500, {}))
    assert check._decode_probe(8001)[0] is None
    monkeypatch.setattr(check.net, "local_post_json", lambda *a, **k: (None, "refused"))
    assert "refused" in check._decode_probe(8001)[1]
    monkeypatch.setattr(check.net, "local_post_json",
                        lambda *a, **k: (200, {"usage": {"completion_tokens": 3}}))
    assert "too short" in check._decode_probe(8001)[1]


def test_run_appends_speed_row(monkeypatch, capsys):
    _write_expected()
    monkeypatch.setattr(check, "scan", lambda: [("endpoint", "LLM endpoint", True, "serving", "")])
    _serve(monkeypatch, 41.0)
    assert check.run() == 0
    assert "Decode speed vs TokenMark" in capsys.readouterr().out


# ── aarch64 / GH200 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("gpu_name", ["NVIDIA GH200 480GB", "NVIDIA GH200 Grace Hopper Superchip"])
def test_gh200_is_not_h200(gpu_name):
    p = gh200_box()
    p.gpus[0].name = gpu_name
    dev = speed.device_for_profile(p)
    assert dev["id"] == "gh200" and dev["mem_gb"] == 96


def test_real_h200_still_h200():
    p = gh200_box()
    p.gpus[0].name = "NVIDIA H200"
    assert speed.device_for_profile(p)["id"] == "h200"


def test_gh200_uses_hbm_not_lpddr_as_budget():
    assert "gh200" not in hardware._UNIFIED_NVIDIA_MARKERS
    assert "gb200" not in hardware._UNIFIED_NVIDIA_MARKERS


def test_arm_cpu_part_names_grace():
    txt = "processor\t: 0\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU part\t: 0xd4f\n"
    info = hardware.parse_cpuinfo(txt)
    assert "Neoverse-V2" in str(info)


def test_arm_cuda_warning_only_on_aarch64():
    m = model()
    arm = "\n".join(f"{s.command} {' '.join(s.warnings)}"
                    for s in setup_advisor.setup_steps(m, gh200_box(), family="apt"))
    x86 = "\n".join(f"{s.command} {' '.join(s.warnings)}"
                    for s in setup_advisor.setup_steps(m, a10_box(), family="apt"))
    assert "aarch64" in arm and "aarch64" not in x86


def test_gh200_hint_is_not_dgx_spark():
    assert cli._tokenmark_hw_hint(gh200_box()) != "DGX Spark"


# ── review fixes (0.4.2) ──────────────────────────────────────────────────────
def test_serve_ctx_never_below_default_but_raised_to_row():
    c32 = measured.select([dict(ROWS[5], ctx=8192)], TRACKED, model(), a10_box()).choice
    assert "-c 32768" in c32.flags and "8192" in " ".join(c32.warnings)
    c128 = measured.select([dict(ROWS[5], ctx=65536)], TRACKED, model(), a10_box()).choice
    assert "-c 65536" in c128.flags


def test_tight_memory_holds_ctx_to_row():
    row = dict(ROWS[5], ctx=8192)
    box = a10_box()
    box.usable_mem_mb = 20000            # UD-Q4_K_XL (17.5 GB) fits with little spare (19.5-21 GB is tight)
    sel = measured.select([row], TRACKED, model(), box)
    assert sel.choice is not None
    # the fixture must reach the tight path, or this test checks nothing
    assert measured.fit.fits(model(), sel.choice.quant, box).tight
    assert "-c 8192" in sel.choice.flags and "memory is tight" in " ".join(sel.choice.warnings)


def test_mtp_flag_only_for_family_with_nextn_layers():
    other = next(m for m in hf_catalog.all_models() if not measured._MTP_FAMILY.search(m.name))
    assert measured._mode_flags({"mode": "mtp-4"}, other)[0] == ""
    assert "cannot confirm" in measured._mode_flags({"mode": "mtp-4"}, other)[1]
    assert measured._mode_flags({"mode": "mtp-4"}, model())[0].startswith("--spec-type draft-mtp")
    for n in ("Qwen3.6-35B-A3B", "Qwen3.8-Flash-Next", "Qwen3.5-9B"):
        assert measured._MTP_FAMILY.search(n)
    assert not measured._MTP_FAMILY.search("Qwen3-Coder-Next")


def test_mtp_row_on_unknown_family_is_not_picked():
    other = next(m for m in hf_catalog.all_models() if m.name == "gpt-oss-20b")
    q = other.quants[0].name
    rows = [dict(ROWS[5], model=other.name, quant=q, mode="mtp", decode_tps=30.0, ctx=16384),
            dict(ROWS[5], model=other.name, quant=q, mode=None, decode_tps=10.0, ctx=16384)]
    sel = measured.select(rows, TRACKED, other, a10_box())
    assert sel.choice is not None and "--spec-type" not in sel.choice.flags
    assert "cannot confirm" in "\n".join(sel.lines)


def test_arm_warning_only_for_grace_hopper():
    def steps(box):
        return "\n".join(" ".join(s.warnings) for s in setup_advisor.setup_steps(model(), box, family="apt"))
    spark = HardwareProfile(primary_backend="cuda", usable_mem_mb=100000, arch="aarch64",
                            gpus=[GPUInfo(vendor="nvidia", backend="cuda", name="NVIDIA GB10",
                                          memory_kind="unified", unified_mem_mb=120000)])
    jetson = HardwareProfile(primary_backend="cuda", usable_mem_mb=7000, arch="aarch64",
                             gpus=[GPUInfo(vendor="nvidia", backend="cuda", name="Orin (nvgpu)",
                                           memory_kind="unified", unified_mem_mb=8000)])
    assert "Grace Hopper" in steps(gh200_box())
    assert "Grace Hopper" not in steps(spark) and "Grace Hopper" not in steps(jetson)
    assert setup_advisor.is_grace_hopper(gh200_box()) and not setup_advisor.is_grace_hopper(spark)


def test_neoverse_v2_named_grace_only_beside_grace_gpu():
    info = hardware.parse_cpuinfo("processor : 0\nCPU part\t: 0xd4f\n")
    assert info["model"] == "Arm Neoverse-V2"
    assert hardware.grace_cpu_name("Arm Neoverse-V2", gh200_box().gpus) == "Arm Neoverse-V2 (NVIDIA Grace)"
    assert hardware.grace_cpu_name("Arm Neoverse-V2", []) == "Arm Neoverse-V2"       # Graviton4


def test_hw_hint_spark_is_gb10_not_grace():
    spark = HardwareProfile(cpu_model="Cortex-X925", gpus=[GPUInfo(vendor="nvidia", name="NVIDIA GB10")])
    assert cli._tokenmark_hw_hint(spark) == "DGX Spark"
    grace_only = HardwareProfile(cpu_model="Arm Neoverse-V2 (NVIDIA Grace)", gpus=[])
    assert cli._tokenmark_hw_hint(grace_only) is None
    assert cli._tokenmark_hw_hint(gh200_box()) is None


def test_unrated_rows_skipped_and_only_unrated_falls_back():
    bad = dict(ROWS[5], decode_tps=35.0, credibilityLabel="none", url="https://tokenmark.app/c/x")
    good = dict(ROWS[5], decode_tps=20.0, credibilityLabel="high")
    sel = measured.select([bad, good], TRACKED, model(), a10_box())
    assert sel.expected["tok_s"] == 20.0 and "credibility none" in "\n".join(sel.lines)
    only = measured.select([bad], TRACKED, model(), a10_box())
    assert only.choice is None and "only unrated rows" in only.lines[0]


def test_over_bound_row_is_not_trusted():
    # A10 600 GB/s, 17.5 GB weights: ceiling about 34 tok/s without speculation
    absurd = dict(ROWS[5], decode_tps=500.0, mode=None)
    sel = measured.select([absurd, dict(ROWS[5], decode_tps=25.0)], TRACKED, model(), a10_box())
    assert sel.expected["tok_s"] == 25.0
    assert "memory bandwidth" in "\n".join(sel.lines)
    assert measured.select([absurd], TRACKED, model(), a10_box()).choice is None


def test_malformed_rows_skipped_individually_and_counted():
    junk = [None, "x", 7, dict(ROWS[5], decode_tps="fast"), dict(ROWS[5], num_gpus="two", decode_tps=99.0),
            dict(ROWS[5], ctx=[1])]
    sel = measured.select(junk + [ROWS[5]], TRACKED, model(), a10_box())
    assert sel.choice is not None
    assert any("skipped" in ln and "malformed" in ln for ln in sel.lines)


def test_check_warms_up_before_timing():
    calls = []

    def fake(port, path, body, timeout=30):
        calls.append(body["max_tokens"])
        return 200, {"timings": {"predicted_per_second": 40.0}}
    import deneb.check as ck
    orig = ck.net.local_post_json
    ck.net.local_post_json = fake
    try:
        assert ck._decode_probe(8001) == (40.0, "")
    finally:
        ck.net.local_post_json = orig
    assert calls == [8, check._PROBE_TOKENS]


def test_speed_row_other_hardware_or_quant_not_comparable(monkeypatch):
    _write_expected(hardware="strix-halo", hardware_label="AMD Strix Halo", quant="UD-Q4_K_XL")
    _serve(monkeypatch, 41.0, served="Qwen3.8-27B-UD-Q4_K_XL.gguf")
    monkeypatch.setattr(hardware, "profile_hardware", lambda: a10_box())
    ok = check._speed_row()
    assert ok[1] is None and "expected is for AMD Strix Halo" in ok[2] and "not comparable" in ok[2]
    monkeypatch.setattr(hardware, "profile_hardware", lambda: strix_box())
    assert check._speed_row()[1] is True                     # same box, same quant
    _serve(monkeypatch, 41.0, served="Qwen3.8-27B-UD-Q2_K_XL.gguf")
    row = check._speed_row()
    assert row[1] is None and "UD-Q4_K_XL" in row[2] and "UD-Q2_K_XL" in row[2]
