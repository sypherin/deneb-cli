"""`deneb setup --measured` on cloud Nvidia GPUs: TokenMark's card_configs rows (Altronis's own
H100 / A100 / A10 runs), the vLLM steps that reproduce them, the catalog entries those GPUs
need, and the coverage check that flags a GPU TokenMark measures but deneb cannot set up.

Rows come from tests/fixtures/tokenmark_card_rows.json, copied verbatim from the live
tracker-summary.json. Nothing here touches the network or installs or starts anything.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deneb import api, check, cli, coverage, measured, setup_advisor, speed, tokenmark, vllm_setup, watch  # noqa: E402
from deneb import hf_catalog  # noqa: E402
from deneb.check import _own_llama_unit_exists as REAL_OWN_UNIT  # noqa: E402  (conftest stubs it)
from deneb.cli import _has_systemd as REAL_HAS_SYSTEMD  # noqa: E402  (conftest stubs it)
from deneb.hardware import GPUInfo, HardwareProfile  # noqa: E402
from test_measured import _run_setup  # noqa: E402

FIX = json.loads((Path(__file__).parent / "fixtures" / "tokenmark_card_rows.json").read_text())
ROWS = FIX["configs"] + FIX["card_configs"]
TRACKED = sorted({r["hardware"] for r in ROWS})


def model(name="Qwen3.8-27B"):
    return setup_advisor.resolve_model(name, hf_catalog.all_models())


def nv_box(name, vram, arch="x86_64"):
    return HardwareProfile(primary_backend="cuda", usable_mem_mb=vram, arch=arch,
                           gpus=[GPUInfo(vendor="nvidia", backend="cuda", name=name, vram_mb=vram,
                                         memory_kind="dedicated")])


# the names and memory nvidia-smi printed on the boxes the TokenMark rows were measured on
def h100_pcie(arch="x86_64"):
    return nv_box("NVIDIA H100 PCIe", 81559, arch)


def h100_sxm():
    return nv_box("NVIDIA H100 80GB HBM3", 81559)


def a100_40():
    return nv_box("NVIDIA A100-SXM4-40GB", 40960)


def a10():
    return nv_box("NVIDIA A10", 23028)


def row(hw, quant, mode=None, kind="single-stream", backend="vllm"):
    return next(r for r in ROWS if r["hardware"] == hw and r["quant"] == quant and r["mode"] == mode
                and r["throughput_kind"] == kind and r["backend"] == backend
                and r["model"] == "Qwen3.8-27B")


def cmds(steps):
    return "\n".join(s.command for s in steps)


# ── TokenMark rows ────────────────────────────────────────────────────────────
def test_tracker_rows_merges_card_configs(monkeypatch):
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: FIX)
    rows, tracked = tokenmark.tracker_rows()
    assert len(rows) == len(ROWS)
    assert {"h100-pcie", "a100-40gb", "h100-sxm", "a10", "dgx-spark"} <= set(tracked)


def test_tracker_rows_card_configs_alone_and_both_empty(monkeypatch):
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: {"card_configs": FIX["card_configs"]})
    rows, tracked = tokenmark.tracker_rows()
    assert len(rows) == len(FIX["card_configs"]) and "h100-pcie" in tracked
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: {"configs": [], "card_configs": []})
    assert tokenmark.tracker_rows() is None
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: {"configs": "x", "card_configs": None})
    assert tokenmark.tracker_rows() is None


def test_zach_verified_rows_are_not_unrated():
    r = row("h100-pcie", "INT4", "mtp-4")
    assert r["credibilityLabel"] == "none" and r["trust"] == "zach-verified"
    assert not measured._unrated(r)
    assert measured._unrated(dict(r, trust="community"))


def test_measured_for_model_exact_hardware_single_stream(monkeypatch):
    monkeypatch.setattr(tokenmark, "_get", lambda *a, **k: FIX)
    got = tokenmark.measured_for_model("Qwen3.8-27B", hardware_ids=["h100-pcie"])
    assert got[0]["decode_tps"] == 103.7                     # not the 1877.7 aggregate
    assert all(r["throughput_kind"] == "single-stream" for r in got)
    on_a10 = tokenmark.measured_for_model("Qwen3.8-27B", hardware_ids=["a10"])
    assert {r["hardware"] for r in on_a10} == {"a10"}         # a10 is not a prefix match for a100
    assert on_a10[0]["decode_tps"] == 28.4
    loose = tokenmark.measured_for_model("Qwen3.8-27B")
    assert all(r["throughput_kind"] == "single-stream" for r in loose)


def test_api_recommend_passes_exact_hardware_ids(monkeypatch):
    seen = []
    monkeypatch.setattr(tokenmark, "measured_for_model",
                        lambda name, **k: seen.append(k) or [])
    data = api.recommend("coding", measured=True, profile=h100_pcie())
    assert seen and all(k.get("hardware_ids") == ["h100-pcie"] for k in seen)
    assert data.hw_hint == "NVIDIA H100 PCIe"


# ── catalog ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("gpu,dev_id", [
    ("NVIDIA H100 PCIe", "h100-pcie"),
    ("NVIDIA H100 NVL", "h100-nvl"),
    ("NVIDIA H100 80GB HBM3", "h100-sxm"),
    ("NVIDIA A100-SXM4-40GB", "a100-40gb"),
    ("NVIDIA A100-PCIE-40GB", "a100-40gb"),
    ("NVIDIA A100-SXM4-80GB", "a100-80gb"),
    ("NVIDIA A100 80GB PCIe", "a100-80gb"),
    ("NVIDIA A10", "a10"),
    ("NVIDIA A10G", "a10g"),
])
def test_cloud_gpu_names_match_catalog(gpu, dev_id):
    assert speed.match_device([gpu], speed.load_devices())["id"] == dev_id


def test_new_catalog_entries_carry_sources_and_bandwidth():
    devs = {d["id"]: d for d in speed.load_devices()}
    assert devs["h100-pcie"]["bw_gbps"] == 2000 and devs["h100-pcie"]["mem_gb"] == 80
    assert devs["h100-nvl"]["bw_gbps"] == 3900 and devs["h100-nvl"]["mem_gb"] == 94
    assert devs["a100-40gb"]["bw_gbps"] == 1555 and devs["a100-40gb"]["mem_gb"] == 40
    for i in ("h100-pcie", "h100-nvl", "a100-40gb"):
        assert devs[i]["source"].startswith("https://www.nvidia.com/")
        assert devs[i]["class"] == "datacenter-gpu"
    assert len(devs) == len(speed.load_devices())             # ids stay unique


# ── the pick ─────────────────────────────────────────────────────────────────
def test_h100_pcie_picks_vllm_int4_mtp4():
    sel = measured.select(ROWS, TRACKED, model(), h100_pcie())
    c = sel.choice
    assert c is not None and c.engine == "vllm"
    assert c.quant.name == "INT4" and c.quant.repo == "RedHatAI/Qwen3.8-27B-INT4"
    assert (c.ctx, c.max_seqs, c.util) == (65536, 512, 0.92)
    assert '{"method":"mtp","num_speculative_tokens":4}' in c.flags
    text = "\n".join(sel.lines)
    assert "103.7 tok/s" in text and "#mtp4-c1" in text
    assert "1190 tok/s combined at 256 users" in text
    assert "1878 tok/s combined at 256 users" in text         # plain INT4 peaks higher for many users
    assert "drop --speculative-config in the service step" in text   # the one edit that gets there
    assert sel.expected["quant"] == "INT4" and sel.expected["tok_s"] == 103.7
    assert sel.expected["hardware"] == "h100-pcie" and "vllm" in sel.expected["engine"]


def test_h100_sxm_picks_vllm_int4_mtp4():
    sel = measured.select(ROWS, TRACKED, model(), h100_sxm())
    assert sel.choice.engine == "vllm" and sel.expected["tok_s"] == 161.8
    assert sel.expected["hardware"] == "h100-sxm"


def test_a100_40gb_picks_fp8_mtp4_at_its_measured_limits():
    sel = measured.select(ROWS, TRACKED, model(), a100_40())
    c = sel.choice
    assert c.engine == "vllm" and c.quant.name == "FP8" and sel.expected["tok_s"] == 81.2
    assert (c.ctx, c.max_seqs, c.util) == (32768, 64, 0.95)


def test_a10_picks_vllm_int4_over_llama_cpp():
    sel = measured.select(ROWS, TRACKED, model(), a10())
    c = sel.choice
    assert c.engine == "vllm" and c.quant.name == "INT4" and sel.expected["tok_s"] == 28.4
    assert (c.ctx, c.max_seqs, c.util) == (32768, 16, 0.95)
    assert "--speculative-config" not in c.flags


def test_multi_gpu_row_never_picked():
    sel = measured.select(ROWS, TRACKED, model("Qwen3.8-Flash-Next"), h100_sxm())
    assert sel.choice is None and "no single-user rows" in sel.lines[0]


def test_aarch64_box_skips_vllm_and_falls_back_to_llama_cpp():
    sel = measured.select(ROWS, TRACKED, model(), h100_pcie(arch="aarch64"))
    c = sel.choice
    assert c.engine == "llama.cpp" and c.quant.name == "UD-Q4_K_XL" and sel.expected["tok_s"] == 54.4
    text = "\n".join(sel.lines)
    assert "103.7 tok/s" in text and "x86_64" in text and "vllm" in text


def test_unknown_vllm_weights_are_rejected_with_reason():
    r = dict(row("h100-pcie", "INT4", "mtp-4"), quant="AWQ")
    c, why = measured._classify(r, model(), h100_pcie())
    assert c is None and "vllm" in why and "AWQ" in why


def test_vllm_weights_that_do_not_fit_are_rejected():
    r = dict(row("a10", "INT4"), quant="FP8")
    c, why = measured._classify(r, model(), a10())
    assert c is None and "does not fit" in why and "vllm" in why


def test_vllm_row_above_bandwidth_is_not_trusted():
    r = dict(row("a10", "INT4"), decode_tps=900.0)
    c, why = measured._classify(r, model(), a10())
    assert c is None and "memory bandwidth" in why


def test_vllm_needs_cuda():
    rocm = HardwareProfile(primary_backend="rocm", usable_mem_mb=196608, arch="x86_64",
                           gpus=[GPUInfo(vendor="amd", backend="rocm", name="AMD Instinct MI300X",
                                         vram_mb=196608, memory_kind="dedicated")])
    c, why = vllm_setup.classify(row("h100-pcie", "INT4"), model(), rocm)
    assert c is None and "vllm" in why and "CUDA" in why


def test_small_row_ctx_is_raised_and_said():
    r = dict(row("h100-pcie", "INT4", "mtp-4"), ctx=8192)
    c, _ = measured._classify(r, model(), h100_pcie())
    assert c.ctx == 32768 and any("8192" in w for w in c.warnings)


def test_no_multi_user_runs_falls_back_to_16_seqs():
    only = [row("h100-pcie", "INT4", "mtp-4")]
    sel = measured.select(only, ["h100-pcie"], model(), h100_pcie())
    assert sel.choice.max_seqs == 16
    assert any("--max-num-seqs 16" in ln for ln in sel.lines)


def test_count_reads_whole_numbers_only():
    for v, want in [(2, 2), (2.0, 2), ("2", 2), (" 65536 ", 65536), ("2.0", 2), (None, 0), (True, 0),
                    (False, 0), ("two", 0), (2.5, 0), ("2.5", 0), (0, 0), (-4, 0), (float("nan"), 0),
                    (float("inf"), 0), ([2], 0)]:
        assert tokenmark._count(v) == want, v


def test_numeric_strings_in_rows_are_read_not_dropped():
    vrow = row("h100-pcie", "INT4", "mtp-4")
    lrow = row("h100-pcie", "UD-Q4_K_XL", backend="llama.cpp")
    for r in (vrow, lrow):                       # a multi-GPU row stays out whether 2 or "2"
        c, why = measured._classify(dict(r, num_gpus="2"), model(), h100_pcie())
        assert c is None and "2 GPUs" in why
    c, _ = measured._classify(dict(vrow, ctx="65536"), model(), h100_pcie())
    assert c.ctx == 65536 and not any("context" in w for w in c.warnings)
    assert tokenmark._single_user(dict(vrow, concurrency="1", num_gpus="1"))
    assert not tokenmark._single_user(dict(vrow, concurrency="4"))
    cands, _ = measured._candidates([dict(vrow, concurrency="4")], "h100-pcie", model())
    assert cands == []
    agg = dict(row("h100-pcie", "INT4", "mtp-4", kind="aggregate"), concurrency="64")
    assert vllm_setup.peak(vrow, [agg])[1] == 64


def test_mtp_draft_count_outside_1_to_16_is_not_repeated():
    vrow = row("h100-pcie", "INT4", "mtp-4")
    for mode in ("mtp-0", "mtp-17", "mtp-99"):
        c, why = vllm_setup.classify(dict(vrow, mode=mode), model(), h100_pcie())
        assert c is None and "outside the 1 to 16" in why, mode
        flags, why = measured._mode_flags({"mode": mode}, model())
        assert flags == "" and "outside the 1 to 16" in why, mode
    assert measured._mode_flags({"mode": "mtp-16"}, model())[0].endswith("--spec-draft-n-max 16")
    c, _ = vllm_setup.classify(dict(vrow, mode="mtp-04"), model(), h100_pcie())
    assert '"num_speculative_tokens":4}' in c.flags


def test_unreadable_gpu_memory_is_said_not_shown_as_zero():
    c, why = vllm_setup.classify(row("h100-pcie", "INT4", "mtp-4"), model(), nv_box("NVIDIA H100 PCIe", 0))
    assert c is None and "could not read this GPU's memory" in why and "0 GiB" not in why


def test_raised_ctx_says_to_lower_seqs_if_vllm_will_not_start():
    c, _ = measured._classify(dict(row("h100-pcie", "INT4", "mtp-4"), ctx=8192), model(), h100_pcie())
    assert any("lower --max-num-seqs" in w for w in c.warnings)
    c, _ = measured._classify(dict(row("h100-pcie", "INT4", "mtp-4"), ctx=524288), model(), h100_pcie())
    assert c.ctx == 262144 and not any("lower --max-num-seqs" in w for w in c.warnings)


# ── the steps ────────────────────────────────────────────────────────────────
def _vllm_steps(box=None, serving=None, free_gb=None):
    box = box or h100_pcie()
    sel = measured.select(ROWS, TRACKED, model(), box)
    return sel, setup_advisor.setup_steps(model(), box, family="apt", measured=sel.choice,
                                          serving=serving, free_gb=free_gb)


def test_h100_steps_install_vllm_and_serve_the_measured_settings():
    _, steps = _vllm_steps()
    text = cmds(steps)
    assert vllm_setup.WHEEL in text and "--torch-backend=cu129" in text
    assert "hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir ~/models/Qwen3.8-27B-INT4" in text
    run = next(s for s in steps if s.kind == "run")
    for part in ("VLLM_USE_DEEP_GEMM=0 ~/vllm-venv/bin/vllm serve ~/models/Qwen3.8-27B-INT4",
                 "--served-model-name Qwen3.8-27B", "--host 127.0.0.1", "--port 8001",
                 "--max-model-len 65536", "--gpu-memory-utilization 0.92", "--max-num-seqs 512",
                 "--no-enable-prefix-caching", "--language-model-only",
                 """--speculative-config '{"method":"mtp","num_speculative_tokens":4}'"""):
        assert part in run.command, part
    # nothing of the llama.cpp route
    assert "cmake" not in text and "unsloth" not in text and "llama-server" not in run.command
    assert "pipx install huggingface_hub" not in text
    kinds = [s.kind for s in steps]
    assert kinds.count("run") == 1 and kinds.count("service") == 1 and kinds[-1] == "linger"
    assert "ollama" not in kinds


def test_vllm_service_unit_and_one_way_to_start():
    _, steps = _vllm_steps()
    svc = next(s for s in steps if s.kind == "service")
    assert "vllm-server.service" in svc.command and "<<'EOF'" in svc.command
    assert "Environment=VLLM_USE_DEEP_GEMM=0" in svc.command
    assert "ExecStart=%h/vllm-venv/bin/vllm serve %h/models/Qwen3.8-27B-INT4" in svc.command
    assert "[ -f ~/models/Qwen3.8-27B-INT4/config.json ]" in svc.command
    one = setup_advisor.one_way_to_start(steps)
    assert not any(s.kind == "run" for s in one)
    merged = next(s for s in one if s.kind == "service")
    assert "systemctl --user stop vllm-server.service" in merged.warnings[0]
    assert not any("Ctrl-C" in w for w in merged.warnings)
    assert any("--language-model-only" in w for w in merged.warnings)


def test_no_systemd_container_keeps_the_run_step_and_says_how_to_keep_it_up(monkeypatch):
    monkeypatch.setattr(cli, "_has_systemd", lambda: False)
    monkeypatch.setattr(measured, "choose_for_box",
                        lambda m, p: measured.select(ROWS, TRACKED, m, p))
    res = api.setup("Qwen3.8-27B", measured=True, profile=h100_pcie())
    kinds = [s.kind for s in res.steps]
    assert "service" not in kinds and "linger" not in kinds
    run = kinds.index("run")
    note = res.steps[run + 1]
    assert "systemd" in note.title and "tmux" in note.what and "nohup" in note.what
    assert "vllm serve" in res.steps[run].command and "systemctl" not in cmds(res.steps)
    # llama.cpp steps on a container: same shape
    _, steps = _vllm_steps(box=h100_pcie(arch="aarch64"))
    one = setup_advisor.one_way_to_start(steps, systemd=False)
    assert [s.kind for s in one].count("run") == 1 and "systemctl" not in cmds(one)
    assert one[[s.kind for s in one].index("run") + 1].title == note.title


def test_systemd_box_keeps_the_service_route():
    _, steps = _vllm_steps()
    one = setup_advisor.one_way_to_start(steps, systemd=True)
    kinds = [s.kind for s in one]
    assert "service" in kinds and "linger" in kinds and "run" not in kinds
    assert not any("No systemd" in s.title for s in one)


def test_has_systemd_reads_the_boot_marker(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os.path, "isdir", lambda p: p != "/run/systemd/system")
    assert REAL_HAS_SYSTEMD() is False
    monkeypatch.setattr(os.path, "isdir", lambda p: p == "/run/systemd/system")
    assert REAL_HAS_SYSTEMD() is True
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    assert REAL_HAS_SYSTEMD() is True


def test_vllm_steps_carry_the_cloud_container_advice():
    _, steps = _vllm_steps(free_gb=10)
    text = cmds(steps) + "\n" + "\n".join(s.what + " " + " ".join(s.warnings) for s in steps)
    assert "/workspace" in text and "HF_TOKEN" in text and "Python 3.12" in text
    uv = next(s for s in steps if s.title == "Install uv")
    assert "astral.sh/uv/install.sh" in uv.command and "--torch-backend" in uv.command
    svc = next(s for s in steps if s.kind == "service")
    assert "StartLimitIntervalSec=1800\nStartLimitBurst=4\n\n[Service]" in svc.command


def test_every_vllm_step_is_valid_bash():
    _, steps = _vllm_steps()
    for s in steps:
        if s.command:
            r = subprocess.run(["bash", "-n", "-c", s.command], capture_output=True, text=True)
            assert r.returncode == 0, (s.title, r.stderr)


def test_vllm_steps_note_a_server_already_on_8001_and_disk():
    _, steps = _vllm_steps(serving=(8001, "llama-server", "x"), free_gb=5)
    notes = " ".join(s.what for s in steps if s.kind == "note")
    assert "8001" in notes and "llama-server.service" in notes
    assert "18.2 GiB" in notes or "19 GiB" in notes
    _, steps = _vllm_steps(serving=(11434, "ollama", "x"))
    notes = " ".join(s.what for s in steps if s.kind == "note")
    assert "Ollama" in notes and not any(s.kind == "ollama" for s in steps)


def test_llama_steps_note_a_vllm_already_serving():
    box = h100_pcie(arch="aarch64")                    # llama.cpp route
    _, steps = _vllm_steps(box=box, serving=(8001, "vllm", "Qwen3.8-27B"))
    note = next(s for s in steps if s.title == "vLLM is serving on port 8001")
    assert "the port llama-server uses here" in note.what and "vllm-server.service" in note.what
    _, steps = _vllm_steps(box=box, serving=(8000, "vllm", "x"))
    note = next(s for s in steps if s.title == "vLLM is serving on port 8000")
    assert "the port llama-server uses" not in note.what and "GPU's memory" in note.what
    _, steps = _vllm_steps(box=box, serving=(8001, "llama", "x"))
    assert not any("vLLM is serving" in s.title for s in steps)


def test_llama_cpp_choice_keeps_llama_cpp_steps():
    sel = measured.select(ROWS, TRACKED, model(), h100_pcie(arch="aarch64"))
    steps = setup_advisor.setup_steps(model(), h100_pcie(arch="aarch64"), family="apt",
                                      measured=sel.choice)
    text = cmds(steps)
    # no vLLM install or server; the one mention is the llama unit turning a vLLM unit off (same port)
    assert "vllm serve" not in text and "vllm-venv" not in text and "UD-Q4_K_XL" in text
    assert "systemctl --user disable --now vllm-server.service" in text


def test_record_step_for_vllm_pick_is_one_line_json():
    sel, _ = _vllm_steps()
    st = measured.record_step(sel.expected)
    body = st.command.split("\n")[1]
    assert json.loads(body)["quant"] == "INT4"


# ── check / cli / watch ──────────────────────────────────────────────────────
def test_own_unit_detection_sees_vllm_unit(monkeypatch):
    monkeypatch.setattr(os.path, "isfile", lambda p: str(p).endswith("vllm-server.service"))
    assert REAL_OWN_UNIT() is True
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    assert REAL_OWN_UNIT() is False


def test_decode_probe_names_the_served_model():
    seen = []

    def fake(port, path, body, timeout=30):
        seen.append(body.get("model"))
        return 200, {"usage": {"completion_tokens": 64}}
    orig = check.net.local_post_json
    check.net.local_post_json = fake
    try:
        tps, why = check._decode_probe(8001, "Qwen3.8-27B")
    finally:
        check.net.local_post_json = orig
    assert tps and not why and seen == ["Qwen3.8-27B", "Qwen3.8-27B"]


VLLM_UNIT = ("[Unit]\nDescription=vLLM model server (127.0.0.1 only)\n\n[Service]\n"
             "ExecStart=%h/vllm-venv/bin/vllm serve %h/models/Qwen3.8-27B-INT4 --port 8001\n")
LLAMA_UNIT = ("[Service]\nExecStart=%h/llama.cpp/build/bin/llama-server -m "
              "/home/u/models/q/q.gguf --port 8001\n")


def _unit_dir(monkeypatch, tmp_path, **files):
    d = tmp_path / "user"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    monkeypatch.setattr(check, "_WRITTEN_UNIT_DIRS", ((str(d), "user"),))
    monkeypatch.setattr(check, "_probe", lambda cmd: "")       # nothing listening, nothing found
    monkeypatch.setattr(check, "_read", lambda path: "")
    return d


def test_stopped_vllm_unit_is_not_a_missing_model_file(monkeypatch, tmp_path):
    _unit_dir(monkeypatch, tmp_path, **{"vllm-server.service": VLLM_UNIT})
    label, ok, ev, fix = check._chk_model_file()
    assert ok is None and "vLLM manages its own model files" in ev and not fix


def test_stopped_llama_unit_still_looks_for_its_gguf(monkeypatch, tmp_path):
    _unit_dir(monkeypatch, tmp_path, **{"llama-server.service": LLAMA_UNIT})
    label, ok, ev, fix = check._chk_model_file()
    assert ok is False and "deneb setup" in fix


def test_enabled_server_unit_wins_then_the_newest(monkeypatch, tmp_path):
    d = _unit_dir(monkeypatch, tmp_path, **{"llama-server.service": LLAMA_UNIT,
                                            "vllm-server.service": VLLM_UNIT})
    os.utime(d / "llama-server.service", (2_000_000_000, 2_000_000_000))   # llama newer
    os.utime(d / "vllm-server.service", (1_000_000_000, 1_000_000_000))
    assert check._server_unit_on_disk() == ("llama-server.service", "user")
    (d / "default.target.wants").mkdir()
    (d / "default.target.wants" / "vllm-server.service").symlink_to(d / "vllm-server.service")
    assert check._server_unit_on_disk() == ("vllm-server.service", "user")
    assert "systemctl --user start vllm-server.service" in check._stopped_fix()


def test_unit_split_over_lines_is_still_read_as_vllm(monkeypatch, tmp_path):
    # a hand-written unit can continue ExecStart= with a backslash; systemd joins the lines
    split = ("[Service]\nExecStart=%h/vllm-venv/bin/vllm \\\n    serve %h/models/Qwen3.8-27B-INT4 "
             "\\\n    --port 8001\n")
    _unit_dir(monkeypatch, tmp_path, **{"vllm-server.service": split})
    label, ok, ev, fix = check._chk_model_file()
    assert ok is None and "vLLM manages its own model files" in ev
    # `python -m vllm.entrypoints...` split the same way is found as the server unit at all
    mod = ("[Service]\nExecStart=/opt/v/bin/python -m \\\n  vllm.entrypoints.openai.api_server "
           "--port 8001\n")
    (tmp_path / "b").mkdir()
    _unit_dir(monkeypatch, tmp_path / "b", **{"api.service": mod})
    assert check._server_unit_on_disk() == ("api.service", "user")
    assert check._VLLM_EXEC.search(check._unit_text()[2])


def test_every_uv_step_puts_uv_on_path_for_a_new_terminal():
    _, steps = _vllm_steps()
    install, venv, weights = [s for s in steps if "uv " in s.command or "uvx " in s.command]
    assert install.title == "Install uv"
    assert install.command.index(vllm_setup._UV_PATH) < install.command.index("uv pip")
    for s in (venv, weights):   # each pastes on its own, into a terminal that never ran the install
        assert s.command.startswith(vllm_setup._UV_PATH), s.command


def test_speed_probe_sends_the_full_served_id(monkeypatch, tmp_path):
    body = '{"object":"list","data":[{"id":"RedHatAI/Qwen3.8-27B-INT4","object":"model"}]}'
    monkeypatch.setattr(check, "_curl", lambda port, path: body if path == "/v1/models" else "")
    ok, model, _mv, _h = check._endpoint_state(8001)
    assert ok and model == "Qwen3.8-27B-INT4"
    assert check._model_id(body) == "RedHatAI/Qwen3.8-27B-INT4" and check._model_id("") == ""
    sent = []
    monkeypatch.setattr(check, "_decode_probe", lambda port, m: (sent.append(m), (100.0, ""))[1])
    monkeypatch.setattr(check, "serving_now", lambda: (8001, "vllm", "Qwen3.8-27B-INT4"))
    monkeypatch.setattr(check, "_record_mismatch", lambda exp, served: "")
    rec = tmp_path / "expected.json"
    rec.write_text(json.dumps({"model": "Qwen3.8-27B", "tok_s": 103.7}))
    monkeypatch.setattr(check.config, "expected_path", lambda: str(rec))
    label, ok, ev, _ = check._speed_row()
    assert sent == ["RedHatAI/Qwen3.8-27B-INT4"] and "103.7" in ev


def test_cli_measured_on_h100_prints_vllm_route(monkeypatch, capsys):
    rc, out, calls = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B", "--measured"],
                                fetch=(ROWS, TRACKED), box=h100_pcie())
    assert rc == 0 and calls == [1]
    assert "103.7 tok/s" in out and "vllm serve" in out and "expected.json" in out
    assert "llama.cpp route" not in out


def test_cli_without_measured_on_datacenter_gpu_hints_it(monkeypatch, capsys):
    rc, out, calls = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B"], fetch=(ROWS, TRACKED),
                                box=h100_pcie())
    assert rc == 0 and calls == []
    assert "deneb setup Qwen3.8-27B --measured" in out and "vLLM" in out


def test_cli_without_measured_on_consumer_box_no_hint(monkeypatch, capsys):
    rc, out, _ = _run_setup(monkeypatch, capsys, ["Qwen3.8-27B"], fetch=(ROWS, TRACKED))
    assert "--measured" not in out


def test_watch_unschedule_names_both_server_units():
    t = watch.unschedule_text()
    assert "llama-server.service" in t and "vllm-server.service" in t


# ── coverage ─────────────────────────────────────────────────────────────────
def test_coverage_clean_on_tokenmark_rows():
    rep = coverage.report(ROWS)
    assert rep["unknown_hardware"] == [] and rep["no_setup"] == []
    assert {"h100-pcie", "h100-sxm", "a100-40gb", "a10"} <= set(rep["reproducible"])
    assert "dgx-spark" in rep["catalog_only"]                  # unified boxes: level 1 only, said


def test_coverage_stand_in_reads_memory_like_nvidia_smi():
    dev = {d["id"]: d for d in speed.load_devices()}
    assert coverage.stand_in(dev["h100-pcie"]).gpus[0].vram_mb == 81510   # nvidia-smi: 81559
    assert coverage.stand_in(dev["a10"]).gpus[0].vram_mb == 23101         # nvidia-smi: 23028


def test_coverage_flags_unknown_gpu():
    extra = dict(row("a10", "INT4"), hardware="tpu-v9x", hardwareLabel="TPU v9x")
    rep = coverage.report(ROWS + [extra])
    assert rep["unknown_hardware"] == ["tpu-v9x"]


def test_coverage_flags_gpu_missing_from_catalog():
    devices = [d for d in speed.load_devices() if d["id"] != "h100-pcie"]
    rep = coverage.report(ROWS, devices=devices)
    assert rep["unknown_hardware"] == ["h100-pcie"]


def test_coverage_flags_gpu_with_nothing_deneb_can_set_up():
    only = [dict(r, backend="sglang") for r in ROWS if r["hardware"] == "h100-sxm"]
    rep = coverage.report(only)
    assert [g["hardware"] for g in rep["no_setup"]] == ["h100-sxm"]
    assert "sglang" in " ".join(rep["no_setup"][0]["why"])


def test_coverage_main_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(tokenmark, "tracker_rows", lambda: (ROWS, TRACKED))
    assert coverage.main([]) == 0
    extra = dict(row("a10", "INT4"), hardware="tpu-v9x")
    monkeypatch.setattr(tokenmark, "tracker_rows", lambda: (ROWS + [extra], TRACKED + ["tpu-v9x"]))
    assert coverage.main([]) == 1 and "tpu-v9x" in capsys.readouterr().out
    monkeypatch.setattr(tokenmark, "tracker_rows", lambda: None)
    assert coverage.main([]) == 3
    monkeypatch.setattr(tokenmark, "tracker_rows", lambda: (ROWS, TRACKED))
    assert coverage.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["unknown_hardware"] == []


# ── the hint on a plain `deneb setup` ────────────────────────────────────────
def test_datacenter_hint_points_at_measured_without_claiming_a_run():
    l40s = nv_box("NVIDIA L40S", 46068)                       # a cloud card TokenMark has no row for
    for box in (h100_pcie(), l40s):
        hint = cli._datacenter_hint(model(), box)
        assert "`deneb setup Qwen3.8-27B --measured`" in hint and "falls back" in hint
        assert "has measured" not in hint                      # never claims a run on this card
    plain = cli._datacenter_hint(model("Qwen3.6-27B"), h100_pcie())   # no vLLM weights deneb knows
    assert "--measured" in plain and "vLLM" not in plain


def test_datacenter_hint_is_silent_off_datacenter_cards():
    assert cli._datacenter_hint(model(), nv_box("NVIDIA GeForce RTX 5090", 32607)) == ""
    assert cli._datacenter_hint(model(), HardwareProfile(primary_backend="cpu", usable_mem_mb=65536)) == ""
