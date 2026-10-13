"""14 Oct stranger QC, the small wrong-output rows: F18, F19, F20, F21, F23."""
from __future__ import annotations

import io
import os
import re
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deneb import cli, config, freecmds, watch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _main(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, _plain(buf.getvalue())


# F18: the no-model message listed 8 free commands and left out selfupdate
def test_selfupdate_is_a_free_command_everywhere_the_list_is_used():
    assert "deneb selfupdate" in freecmds.NAMES
    assert "deneb selfupdate" in freecmds.hint()
    real = config.get_token
    config.get_token = lambda: None
    try:
        _rc, out = _main([])
    finally:
        config.get_token = real
    assert "deneb selfupdate" in out, out


def test_help_names_selfupdate_once():
    rc, out = _main(["--help"])
    assert rc == 0 and out.count("deneb selfupdate") == 1, out


# F19: the last --unschedule step errored ("Unit deneb-watch.service not loaded", exit 1)
# because the unit files were already removed and reloaded away
def test_unschedule_resets_the_failed_state_before_the_units_are_removed():
    lines = _plain(watch.unschedule_text()).splitlines()
    reset = lines.index("systemctl --user reset-failed deneb-watch.service")
    rm = next(i for i, ln in enumerate(lines) if ln.startswith("rm ~/.config/systemd/user/deneb-watch"))
    disable = lines.index("systemctl --user disable --now deneb-watch.timer")
    assert disable < reset < rm, lines


# F20: with nothing listening, watch said "the :8001 server isn't a systemd unit"
def _box(monkeypatch, **attrs):
    for k, v in attrs.items():
        monkeypatch.setattr(watch.check, k, v)


def test_no_server_listening_is_not_called_a_server_outside_systemd(monkeypatch):
    _box(monkeypatch, _port_owner=lambda p: ("", ""), hinted_unit=lambda: ("", ""),
         serving_port=lambda: 8001)
    f, n, _u = watch.probe_restarts(3)
    assert f["status"] == "unknown" and n == 3
    assert "isn't a systemd unit" not in f["evidence"], f["evidence"]
    assert "no model server" in f["evidence"], f["evidence"]


def test_a_listening_server_outside_systemd_still_says_so(monkeypatch):
    _box(monkeypatch, _port_owner=lambda p: ("42", "llama-server"), _unit_of_pid=lambda pid: "",
         _system_unit_of_pid=lambda pid: "", serving_port=lambda: 8001)
    f, _n, _u = watch.probe_restarts(3)
    assert "isn't a systemd unit" in f["evidence"], f["evidence"]


# F21: copy promised an alert "when something changes"; a problem that only eases sends none
def test_alert_copy_says_which_changes_send_one():
    sched = _plain(watch.schedule_text("15m"))
    texts = {"install.sh": (ROOT / "install.sh").read_text(), "README.md": (ROOT / "README.md").read_text(),
             "schedule": sched}
    for name, t in texts.items():
        assert "when something changes" not in t and "alert on change" not in t, name
    assert watch.alert_text([{"id": "x", "label": "L", "kind": "better", "from": "fail", "to": "warn",
                              "evidence": "e", "fix": ""}]) is None
    readme = " ".join(texts["README.md"].split())
    assert "breaks, gets worse or recovers" in readme


# F23: `deneb auth --help` printed "✗ usage" and exited 2
def test_auth_help_is_help():
    for flag in ("--help", "-h"):
        rc, out = _main(["auth", flag])
        assert rc == 0, out
        for part in ("deneb auth --request", "deneb auth --check", "deneb auth --token"):
            assert part in out, out
        assert "✗" not in out


def test_auth_with_nothing_is_still_a_usage_error():
    rc, _out = _main(["auth"])
    assert rc == 2


# F11: `deneb stack` printed one owner's own box: OCR on :8093, Surya 1's private wrapper
# contract, a "before cutting over from Surya 1" step, "a working deployment runs this",
# and on an Ubuntu Strix Halo it said "could not identify this box"
_OWNER_TRACES = ("surya 1", "cutting over", "working deployment", "8093", "8090", "/layout",
                 "/healthz", "podman", "convert_hf_to_gguf", "config.json")


def _strix_profile():
    from deneb import hardware
    gpu = hardware.GPUInfo(vendor="amd", name="Advanced Micro Devices, Inc. [AMD/ATI] Device",
                           backend="rocm")
    return hardware.HardwareProfile(cpu_model="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
                                    ram_total_mb=126908, gpus=[gpu])


def _stack(monkeypatch, argv):
    from deneb import hardware
    monkeypatch.setattr(hardware, "profile_hardware", _strix_profile)
    return _main(["stack", *argv])


def test_stack_carries_no_trace_of_one_owners_box(monkeypatch):
    for argv in ([], ["ocr"]):
        rc, out = _stack(monkeypatch, argv)
        assert rc == 0, out
        for trace in _OWNER_TRACES:
            assert trace not in out.lower(), (argv, trace)


def test_ocr_is_one_service_whatever_it_is_called():
    from deneb import services
    for typed in ("ocr", "surya", "surya2", "surya 2", "surya-2"):
        assert services.resolve_service(typed).key == "ocr", typed
    assert not {"surya", "surya2"} & set(services.SERVICES)
    assert "ocr" in services.STACK_ORDER


def test_ocr_fetches_the_published_gguf_rather_than_converting():
    from deneb import services
    cmds = " ".join(s.command for s in services.SERVICES["ocr"].steps)
    assert "hf download datalab-to/surya-ocr-2-gguf" in cmds, cmds


def test_stack_names_an_ubuntu_strix_halo(monkeypatch):
    _rc, out = _stack(monkeypatch, [])
    assert "On this box: AMD Strix Halo" in out and "could not identify" not in out, out


def test_stack_prose_fits_80_columns(monkeypatch):
    from deneb import services
    cmds = {ln for svc in services.stack_services() for s in svc.steps
            for ln in s.command.split("\n") + [f"verify: {s.verify}"] if ln.strip()}
    _rc, out = _stack(monkeypatch, [])
    wide = [ln for ln in out.splitlines() if len(ln) > 80 and ln.strip() not in cmds]
    assert not wide, wide


# F4: recommend's fit column said "✓ ~36 GB free", which is spare MEMORY but reads as disk,
# and nothing checked the download against the disk it goes to (26 GiB free, every pick
# bigger), so the top pick sent people into a setup that could not finish
def _recommend(monkeypatch, free_gb, argv=()):
    import test_setup_advisor as tsa
    from deneb import hardware
    monkeypatch.setattr(hardware, "profile_hardware", lambda: tsa.strix_box(107871))
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("dnf", free_gb))
    return _main(["recommend", *argv])


def _rows(out):
    """(model name, download GiB) per ranked row, from the 'download:' line under it."""
    rows, name = [], None
    for ln in out.splitlines():
        m = re.match(r"\s{2}(\d+)\s+(\S+)\s", ln)
        if m:
            name = m.group(2)
        d = re.match(r"\s+download: ~(\d+) GiB", ln)
        if d and name:
            rows.append((name, int(d.group(1))))
            name = None
    return rows


def test_recommend_names_memory_as_memory(monkeypatch):
    _rc, out = _recommend(monkeypatch, 500)
    assert "GB free" not in out, out
    assert "memory budget ~105 GB" in out and "107871 MB" not in out, out


def test_every_pick_states_its_download_against_free_disk(monkeypatch):
    rc, out = _recommend(monkeypatch, 500)
    rows = _rows(out)
    assert rc == 0 and len(rows) >= 3, out
    assert out.count("fits the 500 GiB free on ~/models") == len(rows), out
    assert "does not fit on disk" not in out


def test_a_top_pick_too_big_for_the_disk_is_called_out(monkeypatch):
    _rc, out = _recommend(monkeypatch, 26)
    rows = _rows(out)
    too_big = [r for r in rows if r[1] > 26]
    assert too_big and out.count("more than the 26 GiB free on ~/models") == len(too_big), out
    assert rows[0][1] > 26 and "#1 does not fit on disk" in out, out


def test_unreadable_free_disk_claims_nothing(monkeypatch):
    _rc, out = _recommend(monkeypatch, None)
    assert "could not read the free disk on ~/models" in out, out
    assert "fits the" not in out and "more than the" not in out


def test_recommend_and_setup_give_the_same_download_size(monkeypatch):
    _rc, out = _recommend(monkeypatch, 1)
    name, gib = _rows(out)[0]
    _rc, setup_out = _main(["setup", name])
    assert f"needs about {gib} GiB but only 1 GiB is free" in setup_out, setup_out


# F17: with http_proxy set and no_proxy empty, check's 5 loopback GETs went through the proxy,
# and check then reported "runs as another user", "?" rows and exit 1
def _server(hits):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'{"object":"list","data":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _probe_through(monkeypatch, tmp_path, via_env, via_curlrc):
    from deneb import check
    real, proxied = [], []
    box, proxy = _server(real), _server(proxied)
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    try:
        for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(var, proxy_url) if via_env else monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        (tmp_path / ".curlrc").write_text(f"proxy = {proxy_url}\n" if via_curlrc else "")
        monkeypatch.setenv("CURL_HOME", str(tmp_path))
        out = check._curl(box.server_port, "/v1/models")
    finally:
        box.shutdown()
        proxy.shutdown()
    return out, real, proxied


def test_loopback_probes_skip_a_proxy_in_the_environment(monkeypatch, tmp_path):
    out, real, proxied = _probe_through(monkeypatch, tmp_path, via_env=True, via_curlrc=False)
    assert proxied == [] and real == ["/v1/models"], (real, proxied)
    assert '"object":"list"' in out, out


def test_loopback_probes_skip_a_proxy_in_curlrc(monkeypatch, tmp_path):
    out, real, proxied = _probe_through(monkeypatch, tmp_path, via_env=False, via_curlrc=True)
    assert proxied == [] and real == ["/v1/models"], (real, proxied)
    assert '"object":"list"' in out, out


def test_the_proxy_test_can_see_a_proxied_request(monkeypatch, tmp_path):
    """The harness check: without deneb in the way, curl does use that proxy, so the two
    tests above would fail if the fix were deleted."""
    import subprocess
    real, proxied = [], []
    box, proxy = _server(real), _server(proxied)
    try:
        env = {"PATH": os.environ.get("PATH", ""), "CURL_HOME": str(tmp_path),
               "http_proxy": f"http://127.0.0.1:{proxy.server_port}", "no_proxy": ""}
        subprocess.run(["curl", "-s", f"localhost:{box.server_port}/v1/models", "--max-time", "3"],
                       env=env, capture_output=True, timeout=10)
    finally:
        box.shutdown()
        proxy.shutdown()
    assert proxied and not real, (real, proxied)


# F24: #1's why said "most capable coding model", but the order is model size x quant. On the
# vendors' own SWE-bench Verified, two models that fit a Strix Halo (Qwen3.6-35B-A3B 73.4,
# Qwen3-Coder-Next 70.6) beat that #1 (gpt-oss-120b, 62.4 at high reasoning)
def test_recommend_claims_no_skill_it_does_not_measure(monkeypatch):
    for argv in ((), ("--use", "coding"), ("--use", "chat"), ("--use", "vision")):
        _rc, out = _recommend(monkeypatch, 500, argv)
        flat = " ".join(out.split())
        assert "most capable" not in flat, (argv, out)
        assert "pick by size and quant" in flat, (argv, out)
        assert "It is not a skill score" in flat, (argv, out)


def test_the_order_note_names_the_use(monkeypatch):
    _rc, out = _recommend(monkeypatch, 500, ("--use", "coding"))
    assert "among models tagged for coding" in " ".join(out.split()), out


def test_the_top_why_names_its_basis():
    import test_setup_advisor as tsa
    from deneb import recommend as rec
    top = rec.recommend(tsa.strix_box(107871), "coding")[0]
    assert "top coding pick by size and quant" in top.why, top.why


# F5: with Ollama serving, recommend said "rocm, no engine" ("no engine" meant no call to the
# hosted Deneb engine, which a stranger cannot know), and setup gave only a llama.cpp build and
# a second server on :8001, with no `ollama pull` route
from deneb import check, hf_catalog, setup_advisor  # noqa: E402
from deneb.models_catalog import Model, Quant  # noqa: E402

_REAL_SERVING_NOW = getattr(check, "serving_now", None)   # taken before conftest stubs it
_OLLAMA = (11434, "ollama", "qwen2.5:0.5b")


def _api(bodies):
    return lambda port, path: bodies.get(path, "")


def test_serving_now_names_a_serving_ollama(monkeypatch):
    monkeypatch.setattr(check, "_kind_cache", {})
    monkeypatch.setattr(check, "serving_port", lambda: 11434)
    monkeypatch.setattr(check, "_curl", _api({
        "/v1/models": '{"object":"list","data":[{"id":"qwen2.5:0.5b","object":"model"}]}',
        "/api/version": '{"version":"0.12.3"}'}))
    assert _REAL_SERVING_NOW() == _OLLAMA


def test_serving_now_is_none_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(check, "_kind_cache", {})
    monkeypatch.setattr(check, "serving_port", lambda: 8001)
    monkeypatch.setattr(check, "_curl", _api({}))
    assert _REAL_SERVING_NOW() is None


def _setup(monkeypatch, serving, name="gpt-oss-120b"):
    import test_setup_advisor as tsa
    from deneb import hardware
    monkeypatch.setattr(hardware, "profile_hardware", lambda: tsa.strix_box(107871))
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("dnf", 500))
    monkeypatch.setattr(check, "serving_now", lambda: serving)
    return _main(["setup", name])


def test_no_header_says_no_engine(monkeypatch):
    monkeypatch.setattr(check, "serving_now", lambda: None)
    _rc, rec_out = _recommend(monkeypatch, 500)
    _rc, set_out = _setup(monkeypatch, None)
    for out in (rec_out, set_out):
        assert "no engine" not in out, out
        assert "no model server running" in out, out


def test_headers_name_a_serving_ollama(monkeypatch):
    monkeypatch.setattr(check, "serving_now", lambda: _OLLAMA)
    _rc, rec_out = _recommend(monkeypatch, 500)
    _rc, set_out = _setup(monkeypatch, _OLLAMA)
    for out in (rec_out, set_out):
        assert "Ollama serving qwen2.5:0.5b on :11434" in " ".join(out.split()), out


def _ollama_steps(parts, caps=("coding", "general"), serving=_OLLAMA):
    import test_setup_advisor as tsa
    q = Quant("UD-Q4_K_XL", 20000, 4.8, parts=parts)
    m = Model("Demo-30B", 30, "moe", 3, list(caps), [q], "unsloth/Demo-30B-GGUF")
    steps = setup_advisor.setup_steps(m, tsa.strix_box(107871), quant=q, family="dnf",
                                      free_gb=500, serving=serving)
    return steps, [s for s in steps if "Ollama" in s.title]


def test_a_one_file_quant_gets_an_ollama_pull():
    steps, ours = _ollama_steps(parts=1)
    assert len(ours) == 1, [s.title for s in steps]
    assert ours[0].command == "ollama pull hf.co/unsloth/Demo-30B-GGUF:UD-Q4_K_XL"
    assert "sharded" not in " ".join(ours[0].warnings)
    assert steps.index(ours[0]) < next(i for i, s in enumerate(steps) if s.kind == "runtime")


def test_a_split_quant_gets_no_ollama_command():
    _steps, ours = _ollama_steps(parts=3)
    assert len(ours) == 1 and ours[0].command == ""
    assert "split into 3 files" in ours[0].what and "issues/5245" in ours[0].what, ours[0].what


def test_unknown_split_gives_the_command_and_the_way_out():
    _steps, ours = _ollama_steps(parts=0)
    assert ours and ours[0].command.startswith("ollama pull hf.co/")
    warn = " ".join(ours[0].warnings)
    assert "sharded GGUF" in warn and "deneb catalog --refresh" in warn, warn


def test_vision_through_ollama_is_not_promised():
    _steps, ours = _ollama_steps(parts=1, caps=("vision", "chat", "general"))
    assert any("vision" in w for w in ours[0].warnings), ours[0].warnings


def test_the_ollama_step_says_how_to_free_its_memory():
    _steps, ours = _ollama_steps(parts=1)
    warn = " ".join(ours[0].warnings)
    assert "ollama ps" in warn and "ollama stop" in warn, warn


def test_no_ollama_serving_means_no_ollama_step():
    for serving in (None, (8001, "llama", "qwen3.8-27b"), (8000, "vllm", "x")):
        _steps, ours = _ollama_steps(parts=1, serving=serving)
        assert not ours, serving


_TREE = [{"type": "file", "path": "UD-Q8_K_XL/M-UD-Q8_K_XL-00001-of-00002.gguf", "size": 40e9},
         {"type": "file", "path": "UD-Q8_K_XL/M-UD-Q8_K_XL-00002-of-00002.gguf", "size": 20e9},
         {"type": "file", "path": "M-Q4_K_M.gguf", "size": 18e9},
         {"type": "file", "path": "M-MXFP4_MOE.gguf", "size": 17e9},
         {"type": "file", "path": "mmproj-F16.gguf", "size": 1e9}]


def test_the_catalog_records_each_quants_file_layout():
    layout = hf_catalog.gguf_layout(_TREE)
    assert layout == {"UD-Q8_K_XL": {"parts": 2, "tag": "UD-Q8_K_XL"},
                      "Q4_K_M": {"parts": 1, "tag": "Q4_K_M"},
                      "MXFP4": {"parts": 1, "tag": "MXFP4_MOE"}}, layout
    sizes, mmproj = hf_catalog.parse_gguf_tree(_TREE)
    e = hf_catalog.build_entry({"name": "M", "gguf_repo": "u/M-GGUF"}, int(30e9), sizes, mmproj,
                               "now", layout=layout)
    got = {q["name"]: (q["parts"], q["tag"]) for q in e["quants"]}
    assert got == {"UD-Q8_K_XL": (2, "UD-Q8_K_XL"), "Q4_K_M": (1, "Q4_K_M"),
                   "MXFP4": (1, "MXFP4_MOE")}, got
    m = hf_catalog.entry_to_model(e)
    assert {q.name: (q.parts, q.tag) for q in m.quants} == got


def test_an_older_catalog_entry_has_an_unknown_layout():
    e = {"name": "M", "params_b": 30, "gguf_repo": "u/M-GGUF",
         "quants": [{"name": "Q4_K_M", "size_mb": 18000, "bpw": 4.8}]}
    q = hf_catalog.entry_to_model(e).quants[0]
    assert (q.parts, q.tag) == (0, "")


# the catalog calls unsloth's MXFP4_MOE files "MXFP4": the download filter, the run step's find
# and the service unit all matched *-MXFP4.gguf, so an MXFP4 setup downloaded nothing
def _mxfp4_steps(tag, serving=None):
    import test_setup_advisor as tsa
    q = Quant("MXFP4", 18000, 4.25, parts=1 if tag else 0, tag=tag)
    m = Model("Demo-35B-A3B", 35, "moe", 3, ["coding", "general"], [q], "unsloth/Demo-35B-A3B-GGUF")
    return setup_advisor.setup_steps(m, tsa.strix_box(107871), quant=q, family="dnf",
                                     free_gb=500, serving=serving)


def _cmd(steps, kind):
    return next(s.command for s in steps if s.kind == kind)


def test_mxfp4_steps_use_the_file_names_own_label():
    steps = _mxfp4_steps("MXFP4_MOE", serving=_OLLAMA)
    assert '--include "*-MXFP4_MOE.gguf"' in _cmd(steps, "download")
    assert "'*-MXFP4_MOE.gguf'" in _cmd(steps, "run") and "'*-MXFP4_MOE.gguf'" in _cmd(steps, "service")
    pull = next(s.command for s in steps if "Ollama" in s.title)
    assert pull == "ollama pull hf.co/unsloth/Demo-35B-A3B-GGUF:MXFP4_MOE", pull


def test_mxfp4_from_an_older_catalog_matches_either_label():
    steps = _mxfp4_steps("")
    dl = _cmd(steps, "download")
    assert '--include "*-MXFP4.gguf"' in dl and '--include "*-MXFP4_MOE.gguf"' in dl, dl
    assert "'*-MXFP4_MOE.gguf'" in _cmd(steps, "run") and "'*-MXFP4.gguf'" in _cmd(steps, "run")


def test_a_quant_without_a_recorded_tag_keeps_todays_command():
    steps, _ours = _ollama_steps(parts=0, serving=None)
    assert _cmd(steps, "download") == ('hf download unsloth/Demo-30B-GGUF --include "*-UD-Q4_K_XL.gguf" '
                                       '--include "*-UD-Q4_K_XL-0*.gguf" --exclude "MTP/*" '
                                       '--local-dir ~/models/Demo-30B-GGUF')


# a split advised quant is not the end of the Ollama route: Qwen3-Coder-Next's Q5 and up are
# split on Hugging Face, its Q4 and below are one file each
def _mixed_steps(one_file_mb):
    import test_setup_advisor as tsa
    qs = [Quant("UD-Q4_K_XL", one_file_mb, 4.8, parts=1, tag="UD-Q4_K_XL"),
          Quant("Q8_0", 60000, 8.5, parts=3, tag="Q8_0")]
    m = Model("Demo-80B-A3B", 80, "moe", 3, ["coding", "general"], qs, "unsloth/Demo-80B-A3B-GGUF")
    steps = setup_advisor.setup_steps(m, tsa.strix_box(107871), family="dnf", free_gb=500,
                                      serving=_OLLAMA)
    return steps, [s for s in steps if "Ollama" in s.title]


def test_a_split_quant_points_ollama_at_a_one_file_quant_that_fits():
    steps, ours = _mixed_steps(30000)
    assert '"*-Q8_0.gguf"' in _cmd(steps, "download")          # llama.cpp keeps the advised quant
    assert len(ours) == 1 and ours[0].command == "ollama pull hf.co/unsloth/Demo-80B-A3B-GGUF:UD-Q4_K_XL"
    what = " ".join(ours[0].what.split())
    assert "split into 3 files" in what and "issues/5245" in what and "UD-Q4_K_XL" in what, what


def test_no_one_file_quant_that_fits_means_no_command():
    _steps, ours = _mixed_steps(900000)
    assert len(ours) == 1 and ours[0].command == "", ours[0]


def test_a_tag_that_could_break_a_shell_command_is_dropped():
    e = {"name": "M", "params_b": 30, "gguf_repo": "u/M-GGUF",
         "quants": [{"name": "Q4_K_M", "size_mb": 18000, "bpw": 4.8, "parts": 1,
                     "tag": "Q4_K_M.gguf' ; touch ~/pwned ; '"}]}
    q = hf_catalog.entry_to_model(e).quants[0]
    assert q.tag == "" and q.file_labels() == ("Q4_K_M",)


def test_the_ollama_route_is_an_alternative_not_step_one_of_llama_cpp(monkeypatch):
    rc, out = _setup(monkeypatch, _OLLAMA, name="Qwen3.6-35B-A3B")
    flat = " ".join(out.split())
    assert rc == 0 and "top to bottom" in flat
    assert "either run step 1 (the Ollama route)" in flat, flat[-400:]
    assert "not both" in flat
    assert "Or pull" not in flat


def test_no_ollama_route_keeps_the_plain_footer(monkeypatch):
    _rc, out = _setup(monkeypatch, None, name="Qwen3.6-35B-A3B")
    assert "either run step" not in out and "run these yourself, top to bottom" in out


def test_a_model_with_only_split_quants_says_so():
    _steps, ours = _ollama_steps(parts=3)
    what = " ".join(ours[0].what.split())
    assert "Every quant deneb lists for it is split" in what and "fits this box" not in what, what


def test_a_one_file_quant_that_does_not_fit_is_named_as_too_big():
    _steps, ours = _mixed_steps(900000)
    assert "No one-file quant of it fits this box" in " ".join(ours[0].what.split())


def test_the_one_file_fallback_says_it_is_the_smaller_quant():
    _steps, ours = _mixed_steps(30000)
    assert "smaller and less exact than Q8_0" in " ".join(ours[0].what.split()), ours[0].what


# review 14 Oct: an Ollama on another port, an Ollama on llama.cpp's own port, and an older
# catalog's MXFP4 (Hugging Face answers 400 to `:MXFP4`; the files are *-MXFP4_MOE.gguf)
def test_an_ollama_on_another_port_gets_commands_that_reach_it():
    _steps, ours = _ollama_steps(parts=1, serving=(8080, "ollama", "x"))
    assert ours[0].command == ("OLLAMA_HOST=127.0.0.1:8080 ollama pull "
                               "hf.co/unsloth/Demo-30B-GGUF:UD-Q4_K_XL"), ours[0].command
    warn = " ".join(" ".join(ours[0].warnings).split())
    assert "OLLAMA_HOST=127.0.0.1:8080 ollama ps" in warn, warn


def test_the_usual_ollama_port_needs_no_host():
    _steps, ours = _ollama_steps(parts=1)
    assert "OLLAMA_HOST" not in ours[0].command + " ".join(ours[0].warnings)


def test_an_ollama_on_8001_says_the_llama_cpp_steps_clash_with_it():
    for parts in (1, 3):
        _steps, ours = _ollama_steps(parts=parts, serving=(8001, "ollama", "x"))
        warn = " ".join(" ".join(ours[0].warnings).split())
        assert "second server on :8001" not in warn, warn
        assert "can't start" in warn and ":8001" in warn, warn


def test_mxfp4_from_an_older_catalog_pulls_the_label_hugging_face_serves():
    steps = _mxfp4_steps("", serving=_OLLAMA)
    pull = next(s.command for s in steps if "Ollama" in s.title)
    assert pull == "ollama pull hf.co/unsloth/Demo-35B-A3B-GGUF:MXFP4_MOE", pull


def test_a_vision_pull_says_the_projector_comes_too():
    _steps, ours = _ollama_steps(parts=1, caps=("vision", "chat", "general"))
    warn = " ".join(" ".join(ours[0].warnings).split())
    assert "projector" in warn and "1 GiB" in warn, warn
    _steps, ours = _ollama_steps(parts=1)
    assert "projector" not in " ".join(ours[0].warnings)
