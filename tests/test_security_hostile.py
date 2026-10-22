"""Security review 2026-10-22: remote data must never become a command, terminal control
characters, or an unbounded read. Hostile TokenMark rows and catalog entries go through the real
code; the one shell step that carries network text is actually run in bash to prove nothing in
it executes. Nothing here touches the network."""
from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from deneb import cli, config, hf_catalog, measured, net, setup_advisor, tokenmark, watch  # noqa: E402
from test_measured import ROWS, TRACKED, a10_box, model, strix_box  # noqa: E402

PAYLOADS = [
    "$(touch CANARY)", "`touch CANARY`", "x; touch CANARY", "x && touch CANARY", "x | touch CANARY",
    "x\ntouch CANARY", "x\nEOF\ntouch CANARY\ncat <<'EOF'", "x' ; touch CANARY ; '",
    'x" ; touch CANARY ; "', "x\x1b]0;pwned\x07\x1b[31m", "‮evil", "mtp-٣",
]


def _hostile(row, payload):
    r = dict(row)
    for k in ("model", "hardware", "hardwareLabel", "author", "url", "source", "mode", "quant",
              "backend", "engine", "last_updated", "first_seen"):
        if k in r:
            r[k] = payload
    r["ctx"] = payload
    r["flags"] = payload
    return r


def _all_commands(box, rows):
    m = model()
    sel = measured.select(rows, TRACKED, m, box)
    steps = setup_advisor.setup_steps(m, box, measured=sel.choice)
    cmds = [s.command for s in steps]
    if sel.expected is not None:
        cmds.append(measured.record_step(sel.expected).command)
    return sel, cmds


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("box", [a10_box, strix_box])
def test_poisoned_rows_never_reach_a_command(payload, box):
    rows = [_hostile(r, payload) for r in ROWS] + list(ROWS)
    sel, cmds = _all_commands(box(), rows)
    text = "\n".join(cmds + sel.lines)
    assert "CANARY" not in text and "pwned" not in text and "\x1b" not in text
    assert "‮" not in text


def test_record_step_executes_to_exact_json_and_runs_nothing(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    row = _hostile(ROWS[5], "x\nEOF\ntouch CANARY\n$(touch CANARY) `touch CANARY` '\"")
    sel = measured.select([row], TRACKED, model(), a10_box())
    if sel.expected is None:
        sel = measured.select([ROWS[5]], TRACKED, model(), a10_box())
    sel.expected["hardware_label"] = "x\nEOF\ntouch CANARY\n$(touch CANARY) `touch CANARY` '\"\\"
    cmd = measured.record_step(sel.expected).command
    home = tmp_path / "h"
    home.mkdir()
    r = subprocess.run([bash, "-c", cmd], cwd=tmp_path, env={"HOME": str(home), "PATH": os.environ["PATH"]},
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    got = json.loads((home / ".config" / "deneb" / "expected.json").read_text())
    assert got == sel.expected
    assert not list(tmp_path.rglob("CANARY")) and not (home / "CANARY").exists()


def test_unicode_digits_are_not_an_mtp_draft_count():
    assert measured._MTP.fullmatch("mtp-3")
    assert not measured._MTP.fullmatch("mtp-٣")


# ── catalog (the cache file is plain, user-writable JSON) ────────────────────────────
def _entry(**kw):
    e = {"name": "Qwen3.8-27B", "params_b": 27, "gguf_repo": "unsloth/Qwen3.8-27B-GGUF",
         "quants": [{"name": "Q4_K_M", "size_mb": 1000, "bpw": 4.8}], "capabilities": ["general"]}
    e.update(kw)
    return e


def test_catalog_entry_ok():
    assert hf_catalog.entry_to_model(_entry()) is not None


@pytest.mark.parametrize("bad", ["a/b; touch CANARY", "a/$(id)", "a/b\nEOF", "../../etc", "a b/c", "a/b`id`", "a/b'x"])
def test_catalog_repo_must_be_a_plain_repo_id(bad):
    assert hf_catalog.entry_to_model(_entry(gguf_repo=bad)) is None


@pytest.mark.parametrize("bad", ["x; id", "x$(id)", "x`id`", "x\ny", "x'y", 'x"y', "x|y", "x&y"])
def test_catalog_name_and_quant_must_be_plain(bad):
    assert hf_catalog.entry_to_model(_entry(name=bad)) is None
    assert hf_catalog.entry_to_model(_entry(quants=[{"name": bad, "size_mb": 1, "bpw": 4.0}])) is None


def test_bundled_snapshot_still_loads_whole():
    snap = json.loads(Path(hf_catalog.SNAPSHOT_PATH).read_text())["models"]
    assert len(snap) > 5 and all(hf_catalog.entry_to_model(e) is not None for e in snap)


# ── terminal output: TokenMark text is cleaned before it is printed ──────────────────
def test_recommend_measured_strips_terminal_escapes(monkeypatch, capsys):
    bad = {"decode_tps": "\x1b[2J", "quant": "Q4\x1b]0;pwn\x07", "hardwareLabel": "A\x1b[31m\u202e",
           "author": "me\x1b[0m"}
    monkeypatch.setattr(tokenmark, "measured_for_model", lambda *a, **k: [bad])
    from deneb import check, hardware
    monkeypatch.setattr(hardware, "profile_hardware", lambda: strix_box())
    monkeypatch.setattr(check, "serving_now", lambda: None)
    cli.cmd_recommend(["--measured"])
    out = capsys.readouterr().out
    assert "measured on TokenMark" in out and "\x1b]0" not in out and "\u202e" not in out
    assert "\x1b[2J" not in out and "pwn" in out  # text kept, control characters gone


def test_measured_line_helpers_drop_control_characters():
    assert "\x1b" not in measured._s("a\x1b]0;x\x07b")
    assert measured._tps({"decode_tps": "\x1b[2J"}) == 0.0


# ── size caps, cleartext engine, config perms, webhook ───────────────────────────────
class _Big:
    def __init__(self, n):
        self.n = n

    def read(self, k=-1):
        k = self.n if k < 0 else min(k, self.n)
        return b"x" * k


def test_read_capped_refuses_an_oversized_answer():
    assert net.read_capped(_Big(10), cap=10) == b"x" * 10
    with pytest.raises(net.NetError):
        net.read_capped(_Big(11), cap=10)


@pytest.mark.parametrize("url", ["http://deneb.example.com", "ftp://x/y", "file:///etc/passwd", "http://8.8.8.8/"])
def test_engine_calls_refuse_cleartext_to_the_internet(url):
    with pytest.raises(net.NetError):
        net.request("POST", url + "/ask", body={"a": 1}, headers={"Authorization": "Bearer t"})


@pytest.mark.parametrize("url", ["http://127.0.0.1:8014", "http://localhost:8014", "http://192.168.1.5:8014", "https://x.example.com"])
def test_engine_calls_allow_https_and_private_http(url):
    net.require_secure(url)


def test_config_dir_is_owner_only(tmp_path, monkeypatch):
    d = tmp_path / "newcfg"
    monkeypatch.setattr(config, "CONFIG_DIR", str(d))
    monkeypatch.setattr(config, "CONFIG_FILE", str(d / "config.json"))
    monkeypatch.setattr(config, "REQUEST_FILE", str(d / "request.json"))
    config._write({"token": "t"})
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(d / "config.json").st_mode) == 0o600


def test_webhook_refuses_cleartext_to_the_internet(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a request was made")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    err = watch.send_notify("http://hooks.example.com/abc/secret-token", "hi")
    assert err and "https" in err and "secret-token" not in err
