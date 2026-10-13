"""Regression tests for the code review of 38bef2c (items 1-10; item 11 lives in test_deneb_rule.py).

Every test feeds canned command output through check._probe / tools.run, the same way
test_check.py and test_watch.py do. Each one failed on 38bef2c.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deneb import check, cli, tools, watch  # noqa: E402
from test_check import _install, _router, _ss, teardown_function  # noqa: E402,F401
from test_watch import _GW_GONE, _GW_UP, _scan_run, patched, tmp_env  # noqa: E402

OLLAMA_MODELS = '{"object":"list","data":[{"id":"llama3:8b","object":"model","owned_by":"library"}]}'
VLLM_MODELS = '{"object":"list","data":[{"id":"org/Qwen","object":"model","owned_by":"vllm"}]}'
LLAMA_MODELS = '{"object":"list","data":[{"id":"m","object":"model","owned_by":"llamacpp"}]}'
OTHER_MODELS = '{"object":"list","data":[{"id":"x","object":"model","owned_by":"somebody"}]}'
HIDDEN_11434 = "LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:*\n"
OLLAMA_LOADED = "loaded\nactive\nenabled\n"


def _recording(mapping: dict):
    """_probe that records every command and answers from `mapping`."""
    seen = []
    base = _router(mapping)
    check._probe = lambda c: seen.append(c) or base(c)
    check._read = _router({})
    return seen


# ── 1. _unit_of_pid reads the heading line only ──────────────────────────────
TMUX_STATUS = """● tmux-spawn-6c1f.scope - tmux child pane 3 launched by process 1200
     Loaded: loaded (/run/user/1000/systemd/transient/tmux-spawn-6c1f.scope; transient)
     Active: active (running) since Tue 2026-10-13 09:00:00 +08; 2h ago
      Tasks: 4
     CGroup: /user.slice/user-1000.slice/user@1000.service/app.slice/tmux-spawn-6c1f.scope
             └─5 llama-server -m /m/a.gguf --port 8001
"""


def test_unit_of_pid_ignores_the_cgroup_line_of_a_terminal_process():
    _install({"status 5": TMUX_STATUS})
    assert check._unit_of_pid("5") == ""
    _install({"status 5": "● user@1000.service - User Manager for UID 1000\n  CGroup: x"})
    assert check._unit_of_pid("5") == ""
    _install({"status 5": "Failed to get unit for PID 5: PID 5 does not belong to any loaded unit."})
    assert check._unit_of_pid("5") == ""


def test_hand_started_server_is_not_named_after_the_login_session():
    _install({"ss -ltnp": _ss((8001, "llama-server", 5)), "systemctl --user status 5": TMUX_STATUS,
              "curl -s localhost:8001/v1/models": OLLAMA_MODELS.replace("library", "llamacpp")})
    _, ok, ev, _ = check._chk_service()
    assert ok is True and "user@1000.service" not in ev and "llama-server (pid 5)" in ev, ev


# ── 2. hidden socket owner: kind from the HTTP API, never a pgrep guess ──────
def _hidden_ollama(**extra):
    m = {"ss -ltnp": HIDDEN_11434,
         "curl -s localhost:11434/v1/models": OLLAMA_MODELS,
         "curl -s localhost:11434/api/version": '{"version":"0.5.1"}',
         "systemctl show ollama.service": OLLAMA_LOADED,
         "pgrep -af llama": "500 /usr/local/bin/ollama serve",
         "ollama ps": "NAME ID SIZE PROCESSOR UNTIL\nllama3:8b abc 5 GB 100% GPU 4 minutes from now"}
    m.update(extra)
    return m


def test_api_kind_reads_the_http_api():
    _install({"curl -s localhost:11434/api/version": '{"version":"0.5.1"}'})
    assert check._api_kind(11434) == "ollama"
    check._reset_port_cache()
    _install({"curl -s localhost:8000/v1/models": VLLM_MODELS})
    assert check._api_kind(8000) == "vllm"
    check._reset_port_cache()
    _install({"curl -s localhost:8080/v1/models": LLAMA_MODELS})
    assert check._api_kind(8080) == "llama"
    check._reset_port_cache()
    _install({"curl -s localhost:1234/v1/models": OTHER_MODELS})
    assert check._api_kind(1234) == "unknown"
    check._reset_port_cache()
    _install({"curl -s localhost:1234/api/version": "404 page not found"})
    assert check._api_kind(1234) == "unknown"


def test_hidden_ollama_service_row_names_the_system_unit_and_never_guesses_a_process():
    seen = _recording(_hidden_ollama())
    _, ok, ev, _ = check._chk_service()
    assert ok is True and ev == "ollama.service (system) serving :11434", ev
    assert not any("pgrep" in c for c in seen), seen


def test_hidden_ollama_beside_a_user_llama_server_reads_ollama_ps_not_the_wrong_process():
    ss = HIDDEN_11434 + 'LISTEN 0 1 0.0.0.0:8005 0.0.0.0:* users:(("llama-server",pid=40,fd=3))\n'
    m = _hidden_ollama(**{"ss -ltnp": ss,
                          "pgrep -af llama": "40 /x/llama-server -m /m/a.gguf --port 8005 -ngl 99\n"
                                             "500 /usr/local/bin/ollama serve"})
    _install(m)
    check._port_cache = (11434, "")
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True and "100% GPU" in ev, ev
    _, ok, ev, _ = check._chk_model_file()
    assert ok is None and "Ollama" in ev, ev


def test_hidden_ollama_cpu_is_still_a_real_failure_from_ollama_ps():
    _install(_hidden_ollama(**{"ollama ps": "NAME ID SIZE PROCESSOR UNTIL\nm:1 a 5 GB 100% CPU 4 minutes",
                               "pgrep -af llama": "40 /x/llama-server -m /m/a.gguf --port 8005 -ngl 99\n"
                                                  "500 /usr/local/bin/ollama serve"}))
    _, ok, ev, _ = check._chk_gpu()
    assert ok is False and "100% CPU" in ev


def test_hidden_owner_with_no_ollama_unit_is_a_warning_that_says_why():
    _install(_hidden_ollama(**{"systemctl show ollama.service": "not-found\ninactive\n"}))
    label, ok, ev, fix = check._chk_service()
    assert ok is True and "⚠" in ev and watch.status_of(ok, ev) == "warn", ev
    assert "another user" in ev and "can't see" in ev and "ollama.service (system) serving" not in ev
    assert fix


def test_hidden_owner_that_is_unknown_or_vllm_has_no_gpu_or_model_file_row_and_no_failure():
    for models in (OTHER_MODELS, VLLM_MODELS):
        check._reset_port_cache()
        _install({"ss -ltnp": 'LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*\n',
                  "curl -s localhost:8000/v1/models": models,
                  "pgrep -af llama": "40 /x/llama-server -m /m/a.gguf --port 8005 -ngl 99"})
        assert check.serving_port() == 8000
        for fn in (check._chk_gpu, check._chk_model_file):
            _, ok, ev, _ = fn()
            assert ok in (None, check.UNKNOWN), (models, fn.__name__, ev)   # never a pass or fail
        _, ok, ev, _ = check._chk_gpu()
        assert "can't see the server's process (it runs as another user)" in ev
        assert check._chk_endpoint()[1] is True
        assert check._chk_service()[1] is True            # a hidden owner is a warning, never a fail


# ── 3. _server_kind: an unrecognised process is "unknown" ────────────────────
def test_server_kind_unrecognised_process_is_unknown():
    assert check._server_kind("node", "9 node app.js --port 8080") == "unknown"
    assert check._server_kind("python3", "9 python3 -m http.server 8000") == "unknown"
    assert check._server_kind("llama-server", "") == "llama"
    assert check._server_kind("python3", "9 python3 -m llama_cpp.server --model /m/a.gguf") == "llama"
    assert check._server_kind("ollama", "") == "ollama"


def test_unknown_server_gets_no_gpu_or_model_file_verdict():
    body = '{"data":[{"id":"x","object":"model"}]}'
    _install({"ss -ltnp": _ss((8080, "node", 9)), "curl -s localhost:8080/v1/models": body,
              "pgrep -af node": "9 node app.js --port 8080"})
    assert check.serving_port() == 8080
    for fn in (check._chk_gpu, check._chk_model_file):
        _, ok, ev, _ = fn()
        assert ok in (None, check.UNKNOWN), (fn.__name__, ev)


# ── 4. port choice ───────────────────────────────────────────────────────────
def test_port_skips_an_unrelated_app_that_returns_a_data_list():
    _install({"ss -ltnp": _ss((8080, "node", 9)), "curl -s localhost:8080/v1/models": '{"data":[{"a":1}],"meta":1}'})
    assert check.serving_port() == 8001                     # not 8080
    _, ok, ev, _ = check._chk_endpoint()
    assert ok is False and ":8001" in ev                    # the dead model is not masked
    # with an object:"model" entry it does qualify
    check._reset_port_cache()
    _install({"ss -ltnp": _ss((8080, "node", 9)),
              "curl -s localhost:8080/v1/models": '{"data":[{"id":"a","object":"model"}]}'})
    assert check.serving_port() == 8080


def test_port_needs_a_data_list_even_for_a_known_server_that_answers():
    # llama-server on 8080 answers {"data": null}: not a models body. 8000 (hidden) has one.
    ss = _ss((8080, "llama-server", 3)) + "LISTEN 0 1 0.0.0.0:8000 0.0.0.0:*\n"
    _install({"ss -ltnp": ss, "curl -s localhost:8080/v1/models": '{"data":"nope"}',
              "curl -s localhost:8000/v1/models": '{"data":[{"id":"a"}]}'})
    assert check.serving_port() == 8000


def test_port_hidden_owner_qualifies_on_a_data_list_alone():
    _install({"ss -ltnp": "LISTEN 0 1 0.0.0.0:8000 0.0.0.0:*\n",
              "curl -s localhost:8000/v1/models": '{"data":[{"id":"a"}]}'})
    assert check.serving_port() == 8000


def _state_file(state, **kw):
    os.makedirs(state, exist_ok=True)
    data = {"ts": "t", "restarts": None, "restart_unit": None, "unsent": None, "seen": [],
            "findings": {"endpoint": {"status": "ok", "evidence": "serving"}}}
    data.update(kw)
    Path(state, "watch-state.json").write_text(json.dumps(data))


def test_watch_stores_the_port_and_warns_new_when_it_moves():
    with tmp_env() as (state, _h):
        with patched(watch.check, resolved_port=lambda: 8001):
            rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and watch.load_state()["port"] == 8001
        with patched(watch.check, resolved_port=lambda: 8001):
            rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and "moved" not in out
        with patched(watch.check, resolved_port=lambda: 8080):
            rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 1 and "model server moved from :8001 to :8080" in out
        assert "if :8001 was your model, it is down" in out and "NEW" in out
        assert watch.load_state()["port"] == 8080
        with patched(watch.check, resolved_port=lambda: 8080):
            rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and "moved" not in out


def test_reset_clears_the_stored_port():
    with tmp_env() as (state, _h):
        with patched(watch.check, resolved_port=lambda: 8001):
            _scan_run(_GW_GONE)
        assert watch.load_state()["port"] == 8001
        _scan_run(_GW_GONE, ["--reset"])
        assert watch.load_state()["port"] is None
        with patched(watch.check, resolved_port=lambda: 8080):
            rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and "moved" not in out                  # fresh baseline: nothing to compare


def test_a_bad_stored_port_is_dropped():
    with tmp_env() as (state, _h):
        for bad in ("8001", 0, 70000, True, None):
            _state_file(state, port=bad)
            assert watch.load_state()["port"] is None, bad


# ── 5. no `ss`: probe the candidate ports with curl alone ────────────────────
def test_without_ss_the_candidate_ports_are_probed_with_curl():
    for ss_out in ("", "(not installed on this box: ss)"):
        check._reset_port_cache()
        _install({"ss -ltnp": ss_out,
                  "curl -s localhost:11434/v1/models": OLLAMA_MODELS,
                  "curl -s localhost:11434/api/version": '{"version":"0.5.1"}'})
        assert check.serving_port() == 11434, ss_out
        assert check._port_owner(11434) == ("?", "")


def test_without_ss_candidates_go_in_order_and_nothing_answering_is_8001():
    check._reset_port_cache()
    _install({"curl -s localhost:1234/v1/models": OTHER_MODELS,
              "curl -s localhost:8080/v1/models": OTHER_MODELS})
    assert check.serving_port() == 8080
    check._reset_port_cache()
    _install({})
    assert check.serving_port() == 8001
    assert check._port_owner(8001) == ("", "")              # nothing answers: not "hidden"


def test_without_ss_a_whole_scan_finds_ollama_and_its_unit():
    _install(_hidden_ollama(**{"ss -ltnp": "(not installed on this box: ss)"}))
    rows = {r[0]: r for r in check.scan()}
    assert rows["endpoint"][2] is True and ":11434" in rows["endpoint"][1]
    assert rows["service"][3] == "ollama.service (system) serving :11434"
    assert rows["gpu"][2] is True and rows["model_file"][2] is None


# ── 6. every localhost curl has a timeout ────────────────────────────────────
def _all_curls(mapping):
    seen = _recording(mapping)
    check._reset_port_cache()
    check.scan()
    return [c for c in seen if c.startswith("curl")]


def test_no_localhost_curl_runs_without_max_time():
    boxes = [
        _hidden_ollama(),
        _hidden_ollama(**{"ss -ltnp": ""}),
        {"ss -ltnp": _ss((8001, "llama-server", 5), (8002, "bun", 9)),
         "curl -s localhost:8001/v1/models": OTHER_MODELS, "curl -s localhost:8002/v1/models": "unauthorized"},
    ]
    n = 0
    for b in boxes:
        curls = _all_curls(b)
        n += len(curls)
        assert curls, "the scan ran no curl at all"
        for c in curls:
            assert "--max-time 3" in c and "localhost" in c, c
    assert n >= 6


def test_source_has_no_curl_without_max_time_and_the_tool_allows_the_form():
    for mod in (check, watch):
        for i, ln in enumerate(Path(mod.__file__).read_text().splitlines(), 1):
            if "curl -s" in ln and not ln.lstrip().startswith(("#", '"""')):
                assert "--max-time 3" in ln, f"{Path(mod.__file__).name}:{i}: {ln.strip()}"
    ran = []
    with patched(tools.subprocess, run=lambda argv, **k: ran.append(argv) or type(
            "P", (), {"stdout": "x", "stderr": "", "returncode": 0})()):
        out = tools.run("curl -s localhost:8001/v1/models --max-time 3")["output"]
    assert not out.startswith("[refused]") and ran and ran[0][-2:] == ["--max-time", "3"]


# ── 8. watch state migration ─────────────────────────────────────────────────
def test_old_state_seeds_seen_from_optional_rows_that_were_running():
    with tmp_env() as (state, _h):
        _state_file(state, findings={"gateway": {"status": "ok", "evidence": "llm-gateway.service, auth enforced"},
                                     "tunnel": {"status": "warn", "evidence": "cloudflared running ⚠"}}, seen=None)
        d = json.loads(Path(state, "watch-state.json").read_text())
        d.pop("seen")
        Path(state, "watch-state.json").write_text(json.dumps(d))
        assert watch.load_state()["seen"] == ["gateway", "tunnel"]
        rc, out, _ = _scan_run(_GW_GONE)                    # gateway died before the first new run
        assert rc == 2 and "was running before, now not found (:8002)" in out


def test_old_state_that_never_had_a_gateway_does_not_seed_it():
    with tmp_env() as (state, _h):
        _state_file(state, findings={"gateway": {"status": "ok", "evidence": "not set up (optional: only needed)"}})
        d = json.loads(Path(state, "watch-state.json").read_text())
        d.pop("seen")
        Path(state, "watch-state.json").write_text(json.dumps(d))
        assert watch.load_state()["seen"] == []
        rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and "was running before" not in out


def test_old_failing_optional_row_that_is_none_now_is_not_reported_as_recovered():
    with tmp_env() as (state, _h):
        _state_file(state, findings={"gateway": {"status": "fail", "evidence": "AUTH NOT ENFORCED"}})
        d = json.loads(Path(state, "watch-state.json").read_text())
        d.pop("seen")
        Path(state, "watch-state.json").write_text(json.dumps(d))
        assert watch.load_state()["seen"] == []
        rc, out, _ = _scan_run(_GW_GONE)
        assert rc == 0 and "RECOVERED" not in out
        assert not Path(state, "history.jsonl").exists() or "recovered" not in Path(state, "history.jsonl").read_text()
        # and a state that already has `seen` still reports a real recovery
        _state_file(state, findings={"gateway": {"status": "fail", "evidence": "x"}}, seen=["gateway"])
        rc, out, _ = _scan_run(_GW_UP)
        assert "RECOVERED" in out


# ── 9. tunnel: only a process whose argv[0] is cloudflared ───────────────────
def test_tunnel_ignores_editors_and_tails_that_mention_cloudflared():
    _install({"pgrep -af cloudflared": "123 vim cloudflared.yml\n124 tail -f /var/log/cloudflared.log\n"
                                       "125 grep cloudflared"})
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is None and "not set up" in ev


def test_tunnel_counts_the_real_binary_and_asks_about_its_pid():
    seen = _recording({"pgrep -af cloudflared": "123 vim cloudflared.yml\n999 /usr/bin/cloudflared tunnel run my-tunnel",
                       "--user status 999": "● cloudflared-tunnel.service - tunnel"})
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is True and "cloudflared-tunnel.service" in ev
    assert any("status 999" in c for c in seen) and not any("status 123" in c for c in seen)


# ── 10. --reset while another watch holds the lock ───────────────────────────
def test_reset_refused_under_the_lock_says_so_and_exits_3():
    import fcntl
    with tmp_env() as (state, _h):
        _scan_run(_GW_UP)
        fd = os.open(os.path.join(state, "watch.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rc, out, err = _scan_run(_GW_UP, ["--reset"])
        finally:
            os.close(fd)
        assert rc == 3
        assert "reset NOT done: another deneb watch is running; try again in a minute" in out + err
        assert watch.load_state()["seen"] == ["gateway"]                # untouched
        # a normal run under the lock still skips with exit 0
        fd = os.open(os.path.join(state, "watch.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rc, out, err = _scan_run(_GW_UP)
        finally:
            os.close(fd)
        assert rc == 0 and "skipping" in out


# ── watch's restart probe for a hidden ollama ────────────────────────────────
def test_restart_probe_uses_the_system_unit_when_the_owner_is_hidden_ollama():
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        if cmd.startswith("systemctl show ollama.service -p NRestarts"):
            return {"ok": True, "output": "2\n"}
        for key, val in _hidden_ollama().items():
            if key in cmd:
                return {"ok": True, "output": val}
        return {"ok": True, "output": ""}
    with tmp_env(), patched(tools, run=fake_run):
        f, n, unit = watch.probe_restarts(None)
    assert unit == "system:ollama.service" and n == 2 and f["status"] == "ok", (f, calls)
    assert "ollama.service (system)" in f["evidence"]
