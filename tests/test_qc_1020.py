"""20 Oct QC: deneb run as a customer on a fresh cloud GPU (one A10, Ubuntu), from `pipx install`
to a served model and a fault. Each test failed on 0.4.0, or pins a fix made after that run."""
from __future__ import annotations

import io
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deneb import check, platform_guide, setup_advisor, watch  # noqa: E402
from test_check import _install, _ss, teardown_function  # noqa: E402,F401
from test_setup_advisor import cuda_box  # noqa: E402

GONE = "(does not exist: x)"                  # what tools.read_file says for a missing file
LAMBDA_CF = "4242 /usr/bin/cloudflared --no-autoupdate --config /etc/cloudflared/config.yml tunnel run"
NOTEBOOK_INGRESS = ("tunnel: abc\ningress:\n  - hostname: nb.example.com\n"
                    "    service: http://localhost:7000\n  - service: http_status:404\n")
UNIT = ("[Unit]\nDescription=llama-server\n\n[Service]\n"
        "ExecStart=/home/u/llama.cpp/build/bin/llama-server -m {model} --port 8001 -ngl 999\n"
        "Restart=on-failure\n\n[Install]\nWantedBy=default.target\n")


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _reads(mapping: dict):
    """A fake _read: the mapped text, else "does not exist" (an empty string would read as
    an existing, empty config)."""
    def fake(path: str) -> str:
        for key, val in mapping.items():
            if key in path:
                return val
        return GONE
    return fake


def _tunnel(args: str, reads: dict, serving=()):
    _install({"ss -ltnp": _ss(*serving), "pgrep -af cloudflared": f"4242 {args}",
              "ps -p 4242 -o args=": args})
    check._read = _reads(reads)
    return check._chk_tunnel()


# ── A. a running cloudflared is not yet a tunnel to this model ──────────────────
def test_a_cloud_images_own_notebook_tunnel_is_not_counted_as_ours():
    _, ok, ev, _ = _tunnel(LAMBDA_CF.split(" ", 1)[1], {"/etc/cloudflared/config.yml": NOTEBOOK_INGRESS},
                           serving=[(8001, "llama-server", 5)])
    assert ok is None, ev
    assert ":7000" in ev and "not a tunnel to this model" in ev


def test_a_token_tunnel_says_its_routes_live_in_the_dashboard():
    _, ok, ev, _ = _tunnel("/usr/bin/cloudflared tunnel run --token abc", {})
    assert ok is None and "Cloudflare dashboard" in ev


def test_a_tunnel_straight_to_the_model_port_is_a_warning():
    _, ok, ev, fix = _tunnel("/usr/bin/cloudflared tunnel --url http://127.0.0.1:8001", {},
                             serving=[(8001, "llama-server", 5)])
    assert ok is True and "⚠" in ev and ":8001" in ev
    assert ":8002" in fix


def test_tunnel_ports_reads_url_config_and_default_paths():
    check._read = _reads({"/srv/cf.yml": "ingress:\n  - service: http://[::1]:8002\n  - service: http_status:404\n",
                          "/etc/cloudflared/config.yml": "ingress:\n  - hostname: a\n    service: http://0.0.0.0:9000\n"})
    assert check._tunnel_ports("cloudflared tunnel --url http://localhost:8002") == {8002}
    assert check._tunnel_ports("cloudflared --config=/srv/cf.yml tunnel run") == {8002}
    assert check._tunnel_ports("cloudflared tunnel run") == {9000}          # first default that exists
    check._read = _reads({"/etc/cloudflared/config.yml": "tunnel: abc\ncredentials-file: /x.json\n"})
    assert check._tunnel_ports("cloudflared tunnel run abc") is None        # no ingress in the file
    check._read = _reads({})
    assert check._tunnel_ports("cloudflared tunnel run abc") is None        # no config readable
    check._read = _reads({"/etc/cloudflared/config.yml": "(permission denied: /etc/cloudflared/config.yml)"})
    assert check._tunnel_ports("cloudflared tunnel run abc") is None        # unreadable config


def test_tunnel_ports_read_bare_and_quoted_spellings():
    assert check._tunnel_ports("cloudflared tunnel --url localhost:8002") == {8002}
    assert check._tunnel_ports("cloudflared tunnel --url=127.0.0.1:8002") == {8002}
    check._read = _reads({"/c.yml": 'ingress:\n  - hostname: a\n    service: "http://localhost:8002"\n'
                                     "  - hostname: b\n    service: 'localhost:8001'\n"})
    assert check._tunnel_ports("cloudflared --config /c.yml tunnel run") == {8002, 8001}
    assert check._tunnel_ports("cloudflared tunnel --url http://localhost:80021") == {80021}   # not 8002


def test_every_cloudflared_is_read_not_just_the_first():
    nb = "/usr/bin/cloudflared --config /nb.yml tunnel run"
    ours = "/usr/bin/cloudflared --config /ours.yml tunnel run"
    _install({"ss -ltnp": _ss((8001, "llama-server", 5)), "pgrep -af cloudflared": f"100 {nb}\n200 {ours}",
              "ps -p 100 -o args=": nb, "ps -p 200 -o args=": ours})
    check._read = _reads({"/nb.yml": NOTEBOOK_INGRESS,
                          "/ours.yml": "ingress:\n  - hostname: m\n    service: http://localhost:8002\n"})
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is True and "routes to the gateway (:8002)" in ev, ev


# ── I. plain `deneb check` names the stopped unit instead of "no server found" ──
def _units(monkeypatch, tmp_path, scope="user", **files):
    d = tmp_path / scope
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    monkeypatch.setattr(check, "_WRITTEN_UNIT_DIRS", ((str(d), scope),))
    return d


def test_a_stopped_user_unit_is_named_from_the_unit_files(monkeypatch, tmp_path):
    _units(monkeypatch, tmp_path, **{"backup.service": "[Service]\nExecStart=/usr/bin/rsync -a x y\n",
                                     "llama-server.service": UNIT.format(model="/home/u/models/g.gguf")})
    _install({"ss -ltnp": ""})
    assert check._server_unit_on_disk() == ("llama-server.service", "user")
    fix = check._stopped_fix()
    assert fix.startswith("start it again") and "systemctl --user start llama-server.service" in fix


def test_a_stopped_system_unit_gets_sudo(monkeypatch, tmp_path):
    _units(monkeypatch, tmp_path, scope="system",
           **{"llm.service": "[Service]\nExecStart=/opt/venv/bin/vllm serve org/m --port 8001\n"})
    _install({"ss -ltnp": ""})
    assert "`sudo systemctl start llm.service`" in check._stopped_fix()


def test_no_unit_files_still_says_none_was_found():
    _install({"ss -ltnp": ""})
    assert check._stopped_fix().startswith("no model server was found")


def test_check_prints_a_repeated_fix_once_and_says_the_server_is_stopped(monkeypatch, tmp_path):
    _units(monkeypatch, tmp_path, **{"llama-server.service": UNIT.format(model="/home/u/models/g.gguf")})
    _install({"ss -ltnp": ""})
    fix = check._stopped_fix()
    rows = [("endpoint", "LLM endpoint", False, "nothing is listening on :8001", fix),
            ("service", "Model service", False, "nothing is listening on :8001", fix),
            ("gpu", "GPU", None, "n/a", "")]
    monkeypatch.setattr(check, "scan", lambda: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = check.run()
    out = _plain(buf.getvalue())
    assert rc == 2
    assert out.count("systemctl --user start llama-server.service") == 1, out
    assert "fix: same as above" in out
    assert "set up but stopped" in out and "No model server was found" not in out


# ── J. the model file, not the vision projector next to it ─────────────────────
def test_model_file_skips_the_projector():
    _install({"ss -ltnp": "", "models -maxdepth 2": "/home/u/models/g/mmproj-F16.gguf\n"
              "/home/u/models/g/gemma-4-26B-A4B-it-Q4_K_M.gguf"})
    _, ok, ev, _ = check._chk_model_file()
    assert ok is True and ev == "gemma-4-26B-A4B-it-Q4_K_M.gguf"


def test_a_projector_alone_is_not_a_model_file():
    _install({"ss -ltnp": "", "models -maxdepth 2": "/home/u/models/g/mmproj-F16.gguf"})
    _, ok, ev, _ = check._chk_model_file()
    assert ok is False and ev == "no .gguf found"


def test_a_unit_split_over_lines_still_names_its_model(monkeypatch, tmp_path):
    _units(monkeypatch, tmp_path, **{"llama-server.service":
           '[Service]\nExecStart=/opt/llama-server \\\n    -m "/data/m.gguf" \\\n    --port 8001\n'})
    assert check._unit_model_path() == "/data/m.gguf"


def test_a_stopped_servers_model_comes_from_its_unit(monkeypatch, tmp_path):
    model = "/data/gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf"         # outside ~/models: only the unit knows
    _units(monkeypatch, tmp_path, **{"llama-server.service": UNIT.format(model=model)})
    _install({"ss -ltnp": "", f"stat -L -c %s {model}": str(17 * 2**30)})
    _, ok, ev, _ = check._chk_model_file()
    assert ok is True and ev.startswith("gemma-4-26B-A4B-it-Q4_K_M.gguf")


# ── K. a user service stops at logout unless lingering is on ───────────────────
_USER_SVC = {"ss -ltnp": _ss((8001, "llama-server", 5)), "--user status 5": "● llama-server.service - llama",
             "is-enabled": "enabled"}


def test_a_user_service_without_linger_is_a_warning(monkeypatch):
    monkeypatch.setattr(check, "_linger", lambda: False)
    _install(_USER_SVC)
    _, ok, ev, fix = check._chk_service()
    assert ok is True and "⚠" in ev and "log out" in ev
    assert 'loginctl enable-linger "$(id -un)"' in fix


def test_a_user_service_with_linger_is_clean(monkeypatch):
    for linger in (True, None):                     # None: can't tell, so no warning
        monkeypatch.setattr(check, "_linger", lambda v=linger: v)
        _install(_USER_SVC)
        check._reset_port_cache()
        _, ok, ev, fix = check._chk_service()
        assert ok is True and "⚠" not in ev and fix == "", (linger, ev)


def test_setup_turns_on_linger_right_after_the_service():
    m = setup_advisor.resolve_model("Llama-3.1-8B-Instruct")
    kinds = [s.kind for s in setup_advisor.setup_steps(m, cuda_box(23000), family="apt")]
    assert "service" in kinds, kinds
    assert kinds[kinds.index("service") + 1] == "linger"
    linger = [s for s in setup_advisor.setup_steps(m, cuda_box(23000), family="apt") if s.kind == "linger"][0]
    assert linger.command == 'loginctl enable-linger "$(id -un)"' and "log out" in linger.what


# ── G. a fresh image's package index is stale ──────────────────────────────────
def test_apt_installs_refresh_the_index_first():
    assert platform_guide.install_command("debian", "cmake").startswith("sudo apt-get update && ")


# ── B. watch says a shared fix once, in the terminal and in the alert ──────────
def test_watch_alert_lists_a_shared_fix_once():
    f = "start it again: `systemctl --user start llama-server.service`"
    changes = [{"id": i, "label": i, "kind": "new-problem", "from": "ok", "to": "fail",
                "evidence": "nothing is listening on :8001", "fix": f} for i in ("endpoint", "service")]
    txt = watch.alert_text(changes)
    assert txt.count(f) == 1 and txt.count("NEW") == 2, txt


# ── C. a cloud A10 gets tok/s estimates; a box off the catalog gets no dash column ──
def _recommend_on(monkeypatch, gpu_name):
    from deneb import cli, hardware
    from deneb.hardware import GPUInfo, HardwareProfile
    box = HardwareProfile(primary_backend="cuda", usable_mem_mb=22000, os="Linux", arch="x86_64",
                          gpus=[GPUInfo(vendor="nvidia", backend="cuda", vram_mb=23028,
                                        memory_kind="dedicated", name=gpu_name)])
    monkeypatch.setattr(hardware, "profile_hardware", lambda: box)
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("apt", 400))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.cmd_recommend([])
    return rc, _plain(buf.getvalue())


def test_an_a10_is_in_the_catalog_and_gets_estimates(monkeypatch):
    from deneb import speed
    assert speed.match_device(["NVIDIA A10"], speed.load_devices())["bw_gbps"] == 600
    rc, out = _recommend_on(monkeypatch, "NVIDIA A10")
    assert rc == 0 and "est decode" in out and "tok/s" in out, out


def test_an_unknown_gpu_gets_no_column_of_dashes(monkeypatch):
    rc, out = _recommend_on(monkeypatch, "NVIDIA Mystery 9000")
    assert rc == 0 and "not in Deneb's hardware catalog" in out
    assert "est decode" not in out, out


def test_similar_names_still_match_their_own_part():
    from deneb import speed
    devs = speed.load_devices()
    got = {n: (speed.match_device([n], devs) or {}).get("id") for n in
           ("NVIDIA A10G", "NVIDIA A100-SXM4-80GB", "NVIDIA L4", "NVIDIA L40S", "NVIDIA L40", "Tesla T4",
            "NVIDIA T400 4GB", "NVIDIA T4000")}
    assert got == {"NVIDIA A10G": "a10g", "NVIDIA A100-SXM4-80GB": "a100-80gb", "NVIDIA L4": "l4",
                   "NVIDIA L40S": "l40s", "NVIDIA L40": None, "Tesla T4": "t4",
                   "NVIDIA T400 4GB": None, "NVIDIA T4000": None}


# ── D. the model list is named as a dated list, not a live feed ────────────────
def test_the_model_list_is_not_called_live(monkeypatch, tmp_path):
    from deneb import hf_catalog
    monkeypatch.setattr(hf_catalog, "CACHE_PATH", str(tmp_path / "none.json"))
    _models, src = hf_catalog.ranking_catalog()
    assert re.fullmatch(r"model list of \d{4}-\d{2}-\d{2}, shipped with deneb", src), src
