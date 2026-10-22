"""Unit tests for deneb.watch — the maintain mode (`deneb watch`).

The logic (status mapping, diff, alert text, payloads, exit code) is pure, so it is tested
with plain values. The probes and the notify call are monkeypatched: no real box, no real
network. State goes to a tmp DENEB_STATE_DIR, and HOME is pointed at a tmp dir so a test
that wrongly writes under ~/.config shows up as a file.

Run: python3 tests/test_watch.py    (also collectable by pytest)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import check, cli, colour, tools, watch  # noqa: E402

GIB = 2 ** 30


@contextlib.contextmanager
def patched(obj, **attrs):
    saved = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def tmp_env(**env):
    """A tmp dir used as DENEB_STATE_DIR and HOME; yields (state_dir, home)."""
    with tempfile.TemporaryDirectory() as d:
        home = os.path.join(d, "home")
        state = os.path.join(d, "state")
        os.makedirs(home)
        keys = ["DENEB_STATE_DIR", "HOME", "XDG_STATE_HOME", "DENEB_WATCH_NOTIFY"] + list(env)
        old = {k: os.environ.get(k) for k in keys}
        os.environ["DENEB_STATE_DIR"], os.environ["HOME"] = state, home
        os.environ.pop("XDG_STATE_HOME", None)
        os.environ.pop("DENEB_WATCH_NOTIFY", None)
        os.environ.update(env)
        check._reset_port_cache()
        try:
            yield state, home
        finally:
            check._reset_port_cache()
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def F(id, status, ev="", fix="fix it"):
    return {"id": id, "label": id, "status": status, "evidence": ev, "fix": fix}


# ── status mapping ───────────────────────────────────────────────────────────
def test_status_mapping():
    assert watch.status_of(True, "serving x") == "ok"
    assert watch.status_of(True, "unit ⚠ (not enabled)") == "warn"
    assert watch.status_of(False, "down") == "fail"
    assert watch.status_of(False, "⚠ AUTH NOT ENFORCED") == "fail"


def test_findings_from_rows():
    rows = [("endpoint", "LLM", True, "serving", ""), ("service", "Svc", True, "u ⚠ x", "fix s"),
            ("gpu", "GPU", False, "cpu", "fix g")]
    got = watch.findings_from_rows(rows)
    assert [(f["id"], f["status"]) for f in got] == [("endpoint", "ok"), ("service", "warn"), ("gpu", "fail")]
    assert got[2]["fix"] == "fix g" and got[2]["label"] == "GPU"


# ── diff ─────────────────────────────────────────────────────────────────────
def _kinds(prev, cur):
    return {c["id"]: c["kind"] for c in watch.diff(prev, cur)}


def test_diff_every_kind():
    prev = {"a": "ok", "b": "warn", "c": "fail", "d": "warn", "e": "fail", "f": "ok"}
    cur = [F("a", "warn"), F("b", "fail"), F("c", "warn"), F("d", "ok"), F("e", "ok"),
           F("f", "ok"), F("g", "fail"), F("h", "ok")]
    k = _kinds(prev, cur)
    assert k == {"a": "new-problem", "b": "worse", "c": "better", "d": "recovered",
                 "e": "recovered", "f": "unchanged", "g": "new-problem", "h": "unchanged"}


def test_diff_same_problem_is_unchanged():
    assert _kinds({"a": "warn", "b": "fail"}, [F("a", "warn"), F("b", "fail")]) == \
        {"a": "unchanged", "b": "unchanged"}


def test_diff_unknown_keeps_previous_status():
    assert _kinds({"a": "fail", "b": "ok"}, [F("a", "unknown"), F("b", "unknown")]) == \
        {"a": "unchanged", "b": "unchanged"}
    merged = watch.merge_unknown({"a": "fail"}, [F("a", "unknown"), F("z", "unknown")])
    assert {f["id"]: f["status"] for f in merged} == {"a": "fail", "z": "unknown"}
    # and the next real reading is compared with the kept status, not with "unknown"
    assert _kinds({"a": "fail"}, [F("a", "ok")]) == {"a": "recovered"}


def test_diff_first_run_reports_problems_as_new():
    k = _kinds(None, [F("a", "ok"), F("b", "warn"), F("c", "fail"), F("d", "unknown")])
    assert k == {"a": "unchanged", "b": "new-problem", "c": "new-problem", "d": "unchanged"}


# ── restarts ─────────────────────────────────────────────────────────────────
def _fake_restarts(count_text, unit="llm.service"):
    def fake_run(cmd, *a, **k):
        assert cmd.startswith("systemctl --user show llm.service"), cmd
        return {"ok": True, "output": count_text}
    return patched(tools, run=fake_run), patched(watch.check, _port_owner=lambda p: ("42", "llama-server"),
                                                 _unit_of_pid=lambda pid: unit, serving_port=lambda: 8001)


def _restarts(count_text, prev):
    a, b = _fake_restarts(count_text)
    with a, b:
        return watch.probe_restarts(prev)


def test_restarts_deltas():
    f, n, _u = _restarts("5\n", 4)
    assert f["status"] == "warn" and n == 5 and "restarted 1 time since the last check" in f["evidence"]
    assert "1 times" not in f["evidence"]
    assert "journalctl --user -u llm.service -e" in f["fix"]
    assert _restarts("6", 4)[0]["status"] == "warn"
    f, n, _u = _restarts("7", 4)
    assert f["status"] == "fail" and "crash-looping" in f["evidence"] and n == 7
    assert "restarted 3 times" in f["evidence"]
    assert _restarts("4", 4)[0]["status"] == "ok"


def test_restarts_first_run_and_unreadable():
    f, n, _u = _restarts("9", None)
    assert f["status"] == "ok" and n == 9
    f, n, _u = _restarts("[not a number]", 3)
    assert f["status"] == "unknown" and n == 3   # keep the old count
    f, n, _u = _restarts("1", 5)                  # counter reset (reinstall/reboot)
    assert f["status"] == "ok" and n == 1 and "counter reset" in f["evidence"] \
        and "rebooted" in f["evidence"]


def test_restarts_no_unit_is_unknown():
    with patched(watch.check, _port_owner=lambda p: ("42", "x"), _unit_of_pid=lambda pid: "",
                 _system_unit_of_pid=lambda pid: "", serving_port=lambda: 8001):
        f, n, _u = watch.probe_restarts(3)
    assert f["status"] == "unknown" and n == 3
    with patched(watch.check, _port_owner=lambda p: ("", ""), _unit_of_pid=lambda pid: "",
                 _system_unit_of_pid=lambda pid: "", serving_port=lambda: 8001):
        assert watch.probe_restarts(None)[0]["status"] == "unknown"


# ── disk / memory / catalog thresholds ───────────────────────────────────────
def test_disk_thresholds():
    def st(free_gib, total_gib):
        return watch.disk_status(int(free_gib * GIB), int(total_gib * GIB))
    assert st(150, 1000) == "ok"           # exactly 15% free: not below
    assert st(149, 1000) == "warn"         # 14.9% < 15%
    assert st(20, 100) == "ok"             # 20% free, exactly 20 GiB: not below
    assert st(19.9, 100) == "warn"         # 19.9 GiB < 20 GiB
    assert st(50, 1000) == "warn"          # exactly 5% free (50 GiB): not below the fail limit
    assert st(49, 1000) == "fail"          # 4.9% < 5%
    assert st(5, 100) == "warn"            # exactly 5% and 5 GiB: not below
    assert st(4.9, 100) == "fail"          # < 5 GiB and < 5%
    assert st(4.99, 1000) == "fail"


def test_memory_thresholds():
    assert watch.memory_status(8 * GIB) == "ok"
    assert watch.memory_status(8 * GIB - 1) == "warn"
    assert watch.memory_status(3 * GIB) == "warn"
    assert watch.memory_status(3 * GIB - 1) == "fail"


def test_probe_memory_reads_meminfo():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "meminfo")
        Path(p).write_text("MemTotal: 100 kB\nMemAvailable:    2097152 kB\n")  # 2 GiB
        with patched(watch, MEMINFO=p):
            f = watch.probe_memory()
        assert f["status"] == "fail" and "2.0 GiB" in f["evidence"] and "ps aux --sort=-rss" in f["fix"]
        with patched(watch, MEMINFO=os.path.join(d, "nope")):
            assert watch.probe_memory()["status"] == "unknown"
        Path(p).write_text("MemTotal: 1 kB\n")
        with patched(watch, MEMINFO=p):
            assert watch.probe_memory()["status"] == "unknown"


def test_probe_disk_uses_model_dir_and_home_fallback():
    seen = []

    def fake_usage(path):
        seen.append(path)
        return type("U", (), {"total": 100 * GIB, "used": 98 * GIB, "free": 2 * GIB})()
    with tempfile.TemporaryDirectory() as d:
        model = os.path.join(d, "m.gguf")
        Path(model).write_text("x")
        with patched(watch.shutil, disk_usage=fake_usage):
            f = watch.probe_disk(model)
            assert f["status"] == "fail" and "du -sh" in f["fix"]
            watch.probe_disk("/models/gone.gguf")
            watch.probe_disk(None)
    assert seen[0] == d
    assert seen[1] == seen[2] == os.path.expanduser("~")

    def boom(path):
        raise OSError("nope")
    with patched(watch.shutil, disk_usage=boom):
        assert watch.probe_disk(None)["status"] == "unknown"


def test_catalog_age():
    day = 86400
    assert watch.catalog_status(None) == "ok"
    assert watch.catalog_status(30 * day) == "ok"
    assert watch.catalog_status(31 * day) == "warn"
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "models.json")
        with patched(watch.hf_catalog, CACHE_PATH=missing):
            f = watch.probe_catalog()
        assert f["status"] == "ok" and "bundled" in f["evidence"]
        Path(missing).write_text("{}")
        old = 1_000_000_000
        os.utime(missing, (old, old))
        with patched(watch.hf_catalog, CACHE_PATH=missing):
            f = watch.probe_catalog(now=old + 31 * day)
        assert f["status"] == "warn" and "deneb catalog --refresh" in f["fix"]


# ── state + history ──────────────────────────────────────────────────────────
def test_state_dir_resolution():
    with tmp_env() as (state, home):
        assert watch.state_dir() == state
        del os.environ["DENEB_STATE_DIR"]
        os.environ["XDG_STATE_HOME"] = "/xdg"
        assert watch.state_dir() == "/xdg/deneb"
        del os.environ["XDG_STATE_HOME"]
        assert watch.state_dir() == os.path.join(home, ".local", "state", "deneb")


def test_state_round_trip_and_atomic():
    with tmp_env() as (state, _home):
        assert watch.load_state() is None
        fs = [F("a", "warn", "evidence a"), F("b", "ok", "fine")]
        watch.save_state(fs, restarts=7, ts="2026-01-01T00:00:00+08:00")
        got = watch.load_state()
        assert got["ts"] == "2026-01-01T00:00:00+08:00" and got["restarts"] == 7
        assert got["findings"]["a"] == {"status": "warn", "evidence": "evidence a"}
        assert sorted(os.listdir(state)) == ["watch-state.json"]      # no temp file left behind
        # a failed write must leave the previous state intact
        with patched(os, replace=lambda *a: (_ for _ in ()).throw(OSError("disk full"))):
            try:
                watch.save_state([F("a", "fail")], restarts=9, ts="t")
            except OSError:
                pass
        assert watch.load_state()["restarts"] == 7
        Path(state, "watch-state.json").write_text("{not json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert watch.load_state() is None                          # corrupt = first run
        assert "unreadable" in err.getvalue() and "reset" in err.getvalue()


def test_history_one_line_per_change_none_when_unchanged():
    with tmp_env() as (state, _h):
        cur = [F("a", "warn", "ev"), F("b", "ok")]
        changes = watch.diff({"a": "ok", "b": "ok"}, cur)
        watch.append_history(changes, ts="2026-01-01T00:00:00+08:00")
        watch.append_history(watch.diff({"a": "warn", "b": "ok"}, cur), ts="t2")
        lines = Path(state, "history.jsonl").read_text().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev == {"ts": "2026-01-01T00:00:00+08:00", "id": "a", "from": "ok", "to": "warn", "evidence": "ev"}


def test_timestamp_has_utc_offset():
    ts = watch.now_iso()
    assert ts[-6] in "+-" and ts[-3] == ":" and "T" in ts


# ── alert text ───────────────────────────────────────────────────────────────
def test_alert_none_when_nothing_to_say():
    cur = [F("a", "warn"), F("b", "ok")]
    assert watch.alert_text(watch.diff({"a": "warn", "b": "ok"}, cur)) is None
    assert watch.alert_text(watch.diff({"a": "fail"}, [F("a", "warn")])) is None   # only "better"


def test_alert_text_has_fix_and_host():
    ch = watch.diff({"a": "ok"}, [F("a", "fail", "port down", "start the server"), F("b", "ok")])
    with patched(watch.socket, gethostname=lambda: "box1"):
        txt = watch.alert_text(ch)
    assert "box1" in txt and "port down" in txt and "start the server" in txt
    assert "NEW" in txt
    rec = watch.alert_text(watch.diff({"a": "fail"}, [F("a", "ok", "back")]))
    assert "RECOVERED" in rec and "fix:" not in rec


def test_alert_redacts_secrets():
    ch = watch.diff({}, [F("a", "fail", "cmd had api_key=abcdef1234567890 in it", "run x")])
    txt = watch.alert_text(ch)
    assert "[REDACTED]" in txt and "abcdef1234567890" not in txt


# ── notify ───────────────────────────────────────────────────────────────────
def test_payload_per_url_type():
    body, ct = watch.build_payload("https://hooks.slack.com/services/T/B/x", "hi")
    assert json.loads(body) == {"text": "hi"} and ct == "application/json"
    for u in ("https://discord.com/api/webhooks/1/abc", "https://discordapp.com/api/webhooks/1/abc"):
        body, ct = watch.build_payload(u, "hi")
        assert json.loads(body) == {"content": "hi"} and ct == "application/json"
    body, ct = watch.build_payload("https://ntfy.example.org/topic", "hi")
    assert body == b"hi" and ct.startswith("text/plain")


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return b""


def test_send_notify_request_shape():
    seen = {}

    def fake(req, timeout=None):
        seen.update(url=req.full_url, timeout=timeout, ua=req.get_header("User-agent"),
                    ct=req.get_header("Content-type"), data=req.data)
        return _Resp()
    with patched(urllib.request, urlopen=fake):
        assert watch.send_notify("https://ntfy.example.org/t", "msg") is None
    assert seen["timeout"] == 10 and seen["ua"].startswith("deneb-cli/") and seen["data"] == b"msg"
    with patched(urllib.request, urlopen=fake):
        assert watch.send_notify("file:///etc/passwd", "x") is not None   # only http(s)


def _run_watch(argv, findings, restarts=0, **env):
    """cmd_watch with every probe replaced by canned findings."""
    out, err = io.StringIO(), io.StringIO()
    keep = [f for f in findings]
    with patched(watch, gather=lambda *a: (keep, restarts, None)), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.cmd_watch(argv)
    return rc, out.getvalue(), err.getvalue()


def test_notify_not_called_when_nothing_changed():
    calls = []
    with tmp_env(), patched(urllib.request, urlopen=lambda *a, **k: calls.append(a) or _Resp()):
        url = "https://ntfy.example.org/secret-topic"
        _run_watch(["--notify", url], [F("a", "fail", "down")])
        assert len(calls) == 1                       # first run: new problem, sent
        _run_watch(["--notify", url], [F("a", "fail", "down")])
        assert len(calls) == 1                       # unchanged: silent


def test_notify_failure_is_visible_and_never_prints_url():
    url = "https://hooks.slack.com/services/T000/B000/SECRETSECRETSECRET"

    def boom(*a, **k):
        raise OSError(f"cannot reach {url}")
    with tmp_env() as (state, _h), patched(urllib.request, urlopen=boom):
        rc, out, err = _run_watch(["--notify", url], [F("a", "fail", "down")])
        assert rc == 2
        assert "SECRETSECRET" not in out + err and url not in out + err
        assert "hooks.slack.com" in err and "notify" in err.lower()
        lines = [json.loads(ln) for ln in Path(state, "history.jsonl").read_text().splitlines()]
        n = [x for x in lines if x["id"] == "notify"]
        assert len(n) == 1 and "SECRETSECRET" not in json.dumps(n[0])


def test_notify_from_env():
    calls = []
    with tmp_env(DENEB_WATCH_NOTIFY="https://ntfy.example.org/t"), \
            patched(urllib.request, urlopen=lambda *a, **k: calls.append(a) or _Resp()):
        _run_watch([], [F("a", "warn", "x")])
    assert len(calls) == 1


# ── exit code ────────────────────────────────────────────────────────────────
def test_exit_codes():
    # an unreadable row scores 1, as in `deneb check` (it is neither a pass nor a fail)
    assert watch.exit_code([F("a", "ok"), F("b", "unknown")]) == 1
    assert watch.exit_code([F("a", "fail"), F("b", "unknown")]) == 2
    assert watch.exit_code([]) == 0
    assert watch.exit_code([F("a", "ok"), F("b", "warn")]) == 1
    assert watch.exit_code([F("a", "warn"), F("b", "fail"), F("c", "ok")]) == 2


# ── the command ──────────────────────────────────────────────────────────────
def test_run_marks_changes_and_quiet_mode():
    with tmp_env():
        rc, out, _ = _run_watch([], [F("a", "ok", "fine"), F("b", "fail", "broken", "do the thing")])
        assert rc == 2 and "NEW" in out and "fix:" in out and "do the thing" in out
        rc, out, _ = _run_watch([], [F("a", "ok", "fine"), F("b", "fail", "broken", "do the thing")])
        assert "NEW" not in out and "WORSE" not in out
        rc, out, _ = _run_watch(["--quiet"], [F("a", "ok", "fine"), F("b", "ok", "healed")])
        assert "RECOVERED" in out and "fine" not in out
        rc, out, _ = _run_watch(["--quiet"], [F("a", "ok"), F("b", "ok")])
        assert rc == 0 and len(out.strip().splitlines()) == 1


def test_history_command():
    with tmp_env():
        _run_watch([], [F("a", "fail", "down")])
        _run_watch([], [F("a", "ok", "up")])
        rc, out, _ = _run_watch(["--history"], [])
        assert rc == 0 and out.index("fail") < out.index("ok", out.index("fail"))
        rc, out, _ = _run_watch(["--history", "1"], [])
        assert len(out.strip().splitlines()) == 1


def test_run_stores_restart_count():
    with tmp_env():
        _run_watch([], [F("a", "ok")], restarts=4)
        assert watch.load_state()["restarts"] == 4


def test_gather_collects_all_ids():
    rows = [("endpoint", "E", True, "ok", "")]
    ok = lambda i: {"id": i, "label": i, "status": "ok", "evidence": "", "fix": ""}  # noqa: E731
    with patched(watch.check, scan=lambda: rows, _serving_cmdline=lambda: ""), \
            patched(watch, probe_disk=lambda p: ok("disk"), probe_memory=lambda: ok("memory"),
                    probe_restarts=lambda n, u=None: (ok("restarts"), 2, "u.service"), probe_catalog=lambda: ok("catalog")):
        fs, n, u = watch.gather(1)
    assert [f["id"] for f in fs] == ["endpoint", "disk", "memory", "restarts", "catalog"] and n == 2


def test_a_raising_probe_becomes_unknown_not_a_crash():
    def boom(*a):
        raise RuntimeError("x")
    with patched(watch.check, scan=lambda: [], _serving_cmdline=lambda: ""), \
            patched(watch, probe_disk=boom, probe_memory=boom, probe_restarts=boom, probe_catalog=boom):
        fs, n, u = watch.gather(3)
    assert {f["status"] for f in fs} == {"unknown"} and len(fs) == 4 and n == 3


# ── --schedule: tell-only ────────────────────────────────────────────────────
def _schedule(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = cli.cmd_watch(argv)
    return rc, out.getvalue()


def test_schedule_prints_units_and_writes_nothing():
    with tmp_env() as (state, home):
        rc, out = _schedule(["--schedule"])
        assert rc == 0
        for needle in ("[Service]", "Type=oneshot", "watch --quiet", "EnvironmentFile=-%h/.config/deneb/watch.env",
                       "[Timer]", "OnBootSec=5min", "OnUnitActiveSec=15m", "SuccessExitStatus=1 2",
                       "deneb-watch.service", "deneb-watch.timer", "systemctl --user daemon-reload",
                       "systemctl --user enable --now deneb-watch.timer", "chmod 600",
                       "loginctl enable-linger"):
            assert needle in out, needle
        assert "Persistent" not in out
        assert "ExecStart=/" in out or "ExecStart=%h/" in out
        assert os.listdir(home) == [] and not os.path.exists(state)
        rc, out = _schedule(["--schedule", "--every", "2h"])
        assert rc == 0 and "OnUnitActiveSec=2h" in out
        rc, out = _schedule(["--schedule", "--every=90s"])
        assert rc == 0 and "OnUnitActiveSec=90s" in out


def test_schedule_rejects_bad_every():
    with tmp_env():
        for bad in ("15", "m", "0m", "1d", "5 m", "-5m", "abc"):
            rc, _out = _schedule(["--schedule", "--every", bad])
            assert rc == 2, bad


# ── check.scan / check.run ───────────────────────────────────────────────────
def test_scan_ids_and_run_output_shape():
    rows = {"_chk_endpoint": ("L1", True, "e1", ""), "_chk_service": ("L2", False, "e2", "fx2")}

    def mk(name):
        def fn():
            if name in rows:
                return rows[name]
            raise RuntimeError("probe broke")
        fn.__name__ = name
        return fn
    names = list(check._IDS)
    fns = [mk(n) for n in names]
    with patched(check, _CHECKS=fns):
        got = check.scan()
        assert [r[0] for r in got] == ["endpoint", "service", "gpu", "model_file", "gateway", "tunnel"]
        assert got[0] == ("endpoint", "L1", True, "e1", "")
        assert got[2][2] is False and "probe broke" in got[2][3]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert check.run() == 2          # a failed row: exit 2
    text = out.getvalue()
    assert "✓ L1" in text.replace("\033[32m", "").replace("\033[0m", "")
    assert "fix:" in text and "fx2" in text and "item(s) left above" in text
    assert "(check error: probe broke)" in text


# ═══ review fixes ═══════════════════════════════════════════════════════════
def _main_run(argv, findings, restarts=0):
    return _run_watch(argv, findings, restarts)


# ── 1/2. notify errors never echo the URL; a bad URL cannot crash ────────────
def test_send_notify_error_never_leaks_url_parts():
    url = "https://user:pw@h.example/x?token=SECRET123"

    def boom(*a, **k):
        raise ValueError("nonnumeric port: 'pw@h.example'")
    with patched(urllib.request, urlopen=boom):
        err = watch.send_notify(url, "hi")
    assert err == "ValueError reaching h.example"
    for leak in ("pw", "SECRET123", "user", "token", "nonnumeric"):
        assert leak not in err

    def http404(*a, **k):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    with patched(urllib.request, urlopen=http404):
        err = watch.send_notify(url, "hi")
    assert err == "HTTP 404 from h.example"


def test_bad_notify_url_does_not_crash_the_run():
    with tmp_env() as (state, _h):
        rc, out, err = _run_watch(["--notify", "http://[bad"], [F("a", "fail", "down")])
        assert rc == 2 and "notify" in err.lower() and "[bad" not in err + out
        lines = [json.loads(x) for x in Path(state, "history.jsonl").read_text().splitlines()]
        assert [x["id"] for x in lines].count("notify") == 1
        assert "[bad" not in json.dumps(lines)
    assert watch._safe_host("http://[bad") == "the notify URL"
    assert watch._safe_host("https://user:pw@h.example:99/x") == "h.example"
    with patched(watch, send_notify=lambda *a: (_ for _ in ()).throw(RuntimeError("SECRET"))):
        assert "SECRET" not in watch._notify_step("https://h.example/x", "m")


# ── 3. state shape validation ────────────────────────────────────────────────
def test_malformed_state_is_dropped_not_fatal():
    with tmp_env() as (state, _h):
        os.makedirs(state)
        sp = Path(state, "watch-state.json")
        sp.write_text('{"findings": {"a": "ok"}}')
        rc, out, err = _run_watch([], [F("a", "ok")])
        assert rc == 0 and "ignored 1 bad" in err
        assert watch.load_state()["findings"]["a"]["status"] == "ok"      # healed by the save
        sp.write_text(json.dumps({"findings": {"a": {"status": "bogus"}, "b": {"status": "fail", "evidence": 5}},
                                  "restarts": "7", "restart_unit": 3, "unsent": 9}))
        err_io = io.StringIO()
        with contextlib.redirect_stderr(err_io):
            st = watch.load_state()
        assert list(st["findings"]) == ["b"] and st["restarts"] is None
        assert st["restart_unit"] is None and st["unsent"] is None
        sp.write_text('["not", "a", "dict"]')
        rc, out, err = _run_watch([], [F("a", "ok")])
        assert rc == 0 and "unreadable" in err
        sp.write_text(json.dumps({"findings": {}, "restarts": True}))
        with contextlib.redirect_stderr(io.StringIO()):
            assert watch.load_state()["restarts"] is None                  # bool is not a count


# ── 4. unwritable state dir ──────────────────────────────────────────────────
def test_unwritable_state_dir_warns_and_still_notifies():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        blocker = os.path.join(d, "file")
        Path(blocker).write_text("x")
        env_old = os.environ.get("DENEB_STATE_DIR")
        os.environ["DENEB_STATE_DIR"] = os.path.join(blocker, "sub")   # cannot be created
        try:
            with patched(urllib.request, urlopen=lambda *a, **k: calls.append(1) or _Resp()):
                rc, out, err = _run_watch(["--notify", "https://ntfy.example.org/t"], [F("a", "fail", "down")])
        finally:
            if env_old is None:
                os.environ.pop("DENEB_STATE_DIR", None)
            else:
                os.environ["DENEB_STATE_DIR"] = env_old
    assert rc == 2 and len(calls) == 1
    assert "couldn't save the watch state" in err and "couldn't write the watch history" in err


# ── 5. failed alerts are retried ─────────────────────────────────────────────
def test_unsent_alert_is_retried_then_cleared():
    url = "https://ntfy.example.org/t"
    bodies = []

    def fail(*a, **k):
        raise OSError("down")

    def ok(req, timeout=None):
        bodies.append(req.data.decode())
        return _Resp()
    with tmp_env():
        with patched(urllib.request, urlopen=fail):
            _run_watch(["--notify", url], [F("a", "fail", "disk exploded", "run du")])
        st = watch.load_state()
        assert "disk exploded" in st["unsent"]
        with patched(urllib.request, urlopen=ok):
            rc, _o, _e = _run_watch(["--notify", url], [F("a", "fail", "disk exploded", "run du")])
        assert len(bodies) == 1 and bodies[0].startswith("earlier alert not delivered:")
        assert "disk exploded" in bodies[0]                      # nothing new changed, still sent
        assert watch.load_state()["unsent"] is None
        with patched(urllib.request, urlopen=ok):
            _run_watch(["--notify", url], [F("a", "fail", "disk exploded", "run du")])
        assert len(bodies) == 1                                  # cleared: silent again


def test_unsent_is_capped_combined_and_kept_without_notify():
    with tmp_env():
        big = F("a", "fail", "x" * 9000, "fix")
        with patched(urllib.request, urlopen=lambda *a, **k: (_ for _ in ()).throw(OSError("d"))):
            _run_watch(["--notify", "https://ntfy.example.org/t"], [big])
            assert len(watch.load_state()["unsent"]) <= watch.UNSENT_CAP
            _run_watch(["--notify", "https://ntfy.example.org/t"], [F("b", "fail", "second", "f2")])
            assert "second" in watch.load_state()["unsent"]
        _run_watch([], [big, F("b", "fail", "second", "f2")])      # no notify configured: kept
        assert watch.load_state()["unsent"]


# ── 6/7. exit codes ──────────────────────────────────────────────────────────
def test_unknown_over_stored_fail_stays_red():
    with tmp_env():
        assert _run_watch([], [F("a", "fail", "down")])[0] == 2
        assert _run_watch([], [F("a", "unknown", "can't read")])[0] == 2
        assert watch.load_state()["findings"]["a"]["status"] == "fail"


def test_watch_breaking_returns_3():
    def boom(*a):
        raise RuntimeError("kaput")
    with tmp_env(), patched(watch, gather=boom):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.cmd_watch([])
    assert rc == 3 and "internal error" in err.getvalue() and "Traceback" in err.getvalue()


def test_schedule_unit_text_matches_the_exit_codes():
    with tmp_env():
        rc, out = _schedule(["--schedule"])
    assert "SuccessExitStatus=1 2" in out and "Persistent" not in out
    assert "OnBootSec=5min" in out and "OnUnitActiveSec=15m" in out
    assert "systemctl --user --failed" in out and "3" in out


# ── 8. the Deneb Rule against the REAL probes ────────────────────────────────
_SS = ('LISTEN 0 512  127.0.0.1:8002  0.0.0.0:*  users:(("bun",pid=2892,fd=11))\n'
       'LISTEN 0 512    0.0.0.0:8001  0.0.0.0:*  users:(("llama-server",pid=876334,fd=10))\n')
_CANNED = {
    "ss -ltnp": _SS,
    "systemctl --user status": "● llama.service - llama\n  Loaded: loaded",
    "systemctl --user is-enabled": "enabled\n",
    "systemctl --user show": "3\n",
    "curl -s localhost:8001/health": '{"status":"ok"}',
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"m"}]}',
    "curl -s localhost:8002/v1/models": '{"error":"unauthorized"}',
    "pgrep -af llama-server": "876334 /usr/bin/llama-server -m /m/x.gguf -ngl 99 --port 8001",
    "stat -L -c %s": "13000000000",
    "ldd": "libamdhip64.so => not found",
    "grep -o": "libamdhip64",
}
_ALLOWED_CMDS = {"ss", "pgrep", "stat", "curl", "ldd", "find", "grep", "which", "readlink", "systemctl",
                 "ollama", "ps"}
_BAD_ARGS = {"-X", "-o", "-w", "-d", "-O", "-exec", "-delete", "-ok", "-fprint", "--output", "--data"}


def _assert_read_only(cmd):
    import shlex
    argv = shlex.split(cmd)
    assert argv and argv[0] in _ALLOWED_CMDS, f"command not on the read-only list: {cmd}"
    assert not (_BAD_ARGS & set(argv)), f"write-capable flag in: {cmd}"
    if argv[0] == "systemctl":
        verb = argv[2] if argv[1] == "--user" else argv[1]       # system scope: read verbs only
        assert verb in ("status", "is-enabled", "show"), cmd
    if argv[0] == "ollama":
        assert argv[1:] == ["ps"], cmd
    if argv[0] == "curl":
        assert argv[1] == "-s" and argv[2].startswith("localhost:"), cmd


def _snapshot(root):
    import hashlib
    snap = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(dp, fn)
            snap[os.path.relpath(fp, root)] = (os.stat(fp).st_mtime_ns,
                                               hashlib.sha256(Path(fp).read_bytes()).hexdigest())
    return snap


def test_real_gather_and_main_only_read_and_write_only_own_files():
    import importlib
    rule = importlib.import_module("test_deneb_rule") if "test_deneb_rule" in sys.modules else None
    if rule is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        rule = importlib.import_module("test_deneb_rule")
    calls = []

    def recorder(cmd, *a, **k):
        _assert_read_only(cmd)
        calls.append(cmd)
        for key, val in _CANNED.items():
            if cmd.startswith(key) or key in cmd:
                return {"ok": True, "output": val}
        return {"ok": True, "output": ""}
    with tmp_env() as (state, home):
        Path(home, "keep.txt").write_text("untouched")
        before_home = _snapshot(home)
        restore = rule._install_tripwires()
        try:
            with patched(tools, run=recorder, read_file=lambda p, **k: {"ok": True, "output": ""}):
                fs, n, unit = watch.gather(None)
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = cli.cmd_watch([])
        finally:
            restore()
        assert {f["id"] for f in fs} >= {"endpoint", "service", "gpu", "model_file", "gateway", "tunnel",
                                         "disk", "memory", "restarts", "catalog"}
        assert n == 3 and unit == "llama.service"
        assert rc in (1, 2), "the tunnel row is red in the canned box, so this must not be 0"
        assert any(c.startswith("systemctl --user show") for c in calls) and len(calls) > 10
        assert _snapshot(home) == before_home, "watch touched the home dir"
        assert sorted(os.listdir(state)) == ["history.jsonl", "watch-state.json", "watch.lock"]


_FORBIDDEN_TOKENS = ("subprocess", "os.system", "os.popen", "Popen", "check_output", "run_write",
                     "execute", " exec(", " eval(", "write_file", "os.remove", "os.unlink",
                     "shutil.rmtree", "shutil.move", "os.exec", "os.spawn", "ctypes", "__import__",
                     ".write_text", ".write_bytes")
_OWN_PATHS = {"_history_path", "_state_path", "_tmp_path", "_lock_path"}


def source_violations(src: str) -> list:
    """Read-only guard over a module's source: forbidden tokens, plus every open()/os.open()
    that could write, which may only target deneb's own state, history, temp or lock path."""
    import ast
    out = [t for t in _FORBIDDEN_TOKENS if t in src]
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
        if name != "open" or owner not in ("", "io", "os", "Path"):
            continue
        args = [ast.unparse(a) for a in node.args[1:]] + [
            ast.unparse(k.value) for k in node.keywords if k.arg in ("mode", "flags")]
        writes = any(any(m in a for m in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"))
                     for a in args) if owner == "os" else \
            any(not (a.strip("'\"") and set(a.strip("'\"")) <= set("rbt")) for a in args if a)
        if writes:
            first = node.args[0] if node.args else None
            ok = isinstance(first, ast.Call) and isinstance(first.func, ast.Name) \
                and first.func.id in _OWN_PATHS
            if not ok:
                out.append(f"write-capable open: {ast.unparse(node)}")
        if owner == "os" and not node.args:
            out.append("os.open without a path")
    return out


def test_watch_source_is_read_only():
    src = Path(watch.__file__).read_text()
    assert not source_violations(src), source_violations(src)


def test_source_guard_catches_planted_violations():
    src = Path(watch.__file__).read_text()
    plants = ["os.remove(p)", "os.unlink(p)", "shutil.rmtree(p)", "shutil.move(a, b)", "os.execv(a, b)",
              "os.spawnl(0, a)", "os.popen(c)", "import ctypes", "__import__('os')", "Popen(c)",
              "check_output(c)", "subprocess.run(c)",
              "open('/etc/passwd', 'w')", "open(p, 'a')", "open(p, mode='w')", "open(p, 'r+')",
              "os.open(p, os.O_WRONLY)", "open(_other_path(), 'w')", "Path(p).write_text('x')"]
    assert not source_violations(src)
    for plant in plants:
        bad = source_violations(src + "\n" + plant + "\n") if "import" not in plant else \
            source_violations(src + "\n" + plant + "\n")
        assert bad, f"guard missed: {plant}"
    # and the legitimate forms are not flagged
    ok = src + "\nopen(_history_path(), 'a')\nopen(p)\nopen(p, errors='replace')\nopen(p, 'rb')\n"
    assert not source_violations(ok)


# ── low-severity items ───────────────────────────────────────────────────────
def test_file_and_dir_modes():
    mode = lambda p: os.stat(p).st_mode & 0o777  # noqa: E731
    with tmp_env() as (_s, home):
        xdg = os.path.join(home, "xdg")
        own = os.path.join(xdg, "deneb")
        os.makedirs(own, mode=0o755)
        os.chmod(own, 0o755)
        os.environ.pop("DENEB_STATE_DIR")
        os.environ["XDG_STATE_HOME"] = xdg
        _run_watch([], [F("a", "fail", "down")])
        assert mode(own) == 0o700                                    # deneb's own dir: tightened
        for name in ("history.jsonl", "watch-state.json", "watch.lock"):
            assert mode(os.path.join(own, name)) == 0o600, name
    with tmp_env() as (state, _h):
        os.makedirs(state, mode=0o755)
        os.chmod(state, 0o755)
        _run_watch([], [F("a", "fail", "down")])
        assert mode(state) == 0o755                  # a dir the user chose via DENEB_STATE_DIR: left alone
        for name in ("history.jsonl", "watch-state.json", "watch.lock"):
            assert mode(os.path.join(state, name)) == 0o600, name     # the files are still private
    with tmp_env() as (state, _h):
        watch._ensure_dir()
        assert (os.stat(state).st_mode & 0o777) == 0o700             # created private


def test_second_run_while_locked_skips():
    import fcntl
    with tmp_env() as (state, _h):
        watch._ensure_dir()
        fd = os.open(os.path.join(state, "watch.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            calls = []
            with patched(watch, gather=lambda *a: calls.append(1)):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = cli.cmd_watch([])
            assert rc == 0 and "another deneb watch is running, skipping" in out.getvalue() and not calls
        finally:
            os.close(fd)
        assert _run_watch([], [F("a", "ok")])[0] == 0                # lock released: runs again
    with tmp_env(), patched(watch, fcntl=None):
        assert _run_watch([], [F("a", "ok")])[0] == 0                # no fcntl: still runs


def test_history_survives_bad_lines():
    with tmp_env() as (state, _h):
        os.makedirs(state)
        Path(state, "history.jsonl").write_bytes(
            b'{"ts":"t1","id":"a","from":null,"to":"fail","evidence":"e1"}\n\xff\xfe junk\n[]\n"str"\n'
            b'{"ts":"t2","id":"a","from":"fail","to":"ok","evidence":"e2"}\n')
        rc, out, _e = _run_watch(["--history"], [])
        assert rc == 0 and "t1" in out and "t2" in out
        assert len(out.strip().splitlines()) == 2


def test_every_validation():
    with tmp_env():
        for bad in ("15m\n", "59s", "0s", "30s", "1.5h", "15M"):
            assert _schedule(["--schedule", "--every", bad])[0] == 2, repr(bad)
        assert _schedule(["--schedule", "--every"])[0] == 2
        assert _schedule(["--schedule", "--every="])[0] == 2
        assert _schedule(["--schedule", "--every", "60s"])[0] == 0
        assert _schedule(["--schedule", "--every", "1m"])[0] == 0
        assert "minimum is 1m" in watch.check_every("30s")
        assert watch.check_every("15m\n") is not None


def test_exec_start_escaping():
    with patched(watch.shutil, which=lambda n: "/opt/my dir/100%/deneb"):
        text = watch.schedule_text("15m")
    assert 'ExecStart="/opt/my dir/100%%/deneb" watch --quiet' in text
    with patched(watch.shutil, which=lambda n: "/opt/50%/deneb"):
        assert "ExecStart=/opt/50%%/deneb watch --quiet" in watch.schedule_text("15m")
    with patched(watch.shutil, which=lambda n: None):
        assert "ExecStart=%h/.local/bin/deneb watch --quiet" in watch.schedule_text("15m")


def test_schedule_env_steps_do_not_wipe_and_prefer_env_file():
    text = watch.schedule_text("15m")
    assert "install -m 600 /dev/null" not in text
    lines = text.splitlines()
    i = lines.index("mkdir -p ~/.config/deneb")
    assert lines[i + 1] == "touch ~/.config/deneb/watch.env" and lines[i + 2].startswith("chmod 600")
    assert "<your-webhook-url>" in text and "argv" in text


def test_schedule_has_no_escape_codes_in_commands_even_with_colour():
    with patched(colour, enabled=lambda: True):
        text = watch.schedule_text("15m")
    assert "\033" in text                                    # headings do use colour here
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("mkdir -p ~/.config/systemd"))
    stop = max(i for i, ln in enumerate(lines) if ln == "EOF")
    block = lines[start:stop + 1]
    assert block and not any("\033" in ln for ln in block)
    assert [ln for ln in lines if "EOF" in ln and "<<" not in ln] == ["EOF", "EOF"]
    for ln in lines:
        if ln.startswith(("systemctl", "loginctl", "journalctl", "touch", "chmod", "echo")):
            assert "\033" not in ln


def test_flag_parsing_is_strict():
    with tmp_env() as (state, _h):
        for bad in (["--quite"], ["--notify"], ["--notify="], ["--quiet", "extra"], ["--history", "abc"],
                    ["--history", "0"], ["--history", "-3"], ["--every", "15m"]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = cli.cmd_watch(bad)
            assert rc == 2 and "usage" in err.getvalue(), bad
        assert not os.path.exists(state)                     # none of them ran a scan
        assert _run_watch(["--history", "5"], [])[0] == 0


def test_restart_unit_change_is_a_first_check():
    a, b = _fake_restarts("50")
    with a, b:
        f, n, u = watch.probe_restarts(2, "other.service")   # new unit, small count -> not +48
    assert f["status"] == "ok" and "first check" in f["evidence"] and u == "llm.service" and n == 50
    with tmp_env():
        a, b = _fake_restarts("5")
        with a, b:
            fs, n, u = watch.gather(None)
        watch.save_state(fs, n, "t", u)
        assert watch.load_state()["restart_unit"] == "llm.service"


def test_colour_only_on_a_tty():
    class T(io.StringIO):
        def isatty(self):
            return True
    f = [F("a", "fail", "down")]
    with tmp_env():
        out = T()
        with patched(watch.sys, stdout=out), contextlib.redirect_stderr(io.StringIO()), \
                patched(watch, gather=lambda *a: (f, 0, None)):
            old = os.environ.pop("NO_COLOR", None)
            try:
                watch.main([])
            finally:
                if old is not None:
                    os.environ["NO_COLOR"] = old
        assert "\033[" in out.getvalue()
        out = T()
        os.environ["NO_COLOR"] = "1"
        try:
            with patched(watch.sys, stdout=out), contextlib.redirect_stderr(io.StringIO()), \
                    patched(watch, gather=lambda *a: (f, 0, None)):
                watch.main([])
        finally:
            del os.environ["NO_COLOR"]
        assert "\033[" not in out.getvalue()
        rc, plain, _e = _run_watch([], f)                    # StringIO: not a tty
        assert "\033[" not in plain


def test_quiet_piped_has_no_escape_codes():
    with tmp_env():
        rc, out, err = _run_watch(["--quiet"], [F("a", "fail", "down", "fix it")])
        assert "\033" not in out + err and "NEW" in out


def test_scrub_patterns():
    cases = {
        "slack": ("posted to https://hooks.slack.com/services/T000/B000/XXXXSECRET ok", "XXXXSECRET"),
        "discord": ("hook https://discord.com/api/webhooks/123456/TOKENTOKEN done", "TOKENTOKEN"),
        "discordapp": ("hook https://ptb.discordapp.com/api/webhooks/123456/TOKENTOKEN done", "TOKENTOKEN"),
        "userinfo": ("pulling https://alice:hunter2@git.example/x.git", "hunter2"),
        "api-key": ("llama-server --api-key s3cr3tvalue --port 1", "s3cr3tvalue"),
        "token": ("cmd --token abc123xyz --x", "abc123xyz"),
        "key": ("cmd --key=short1 --x", "short1"),
        "env-style": ("api_key=abcdef1234567890", "abcdef1234567890"),
    }
    for name, (text, secret) in cases.items():
        got = watch.scrub(text)
        assert secret not in got and "[REDACTED]" in got, (name, got)
    assert "hunter2" not in watch.scrub("https://alice:hunter2@git.example/x.git")
    assert watch.scrub("plain text, 12 GiB free") == "plain text, 12 GiB free"


def test_scrub_applies_to_state_and_history():
    with tmp_env() as (state, _h):
        _run_watch([], [F("a", "fail", "started with --api-key s3cr3tvalue and https://u:pw@h.example/")])
        blob = Path(state, "watch-state.json").read_text() + Path(state, "history.jsonl").read_text()
        assert "s3cr3tvalue" not in blob and "u:pw@" not in blob and "[REDACTED]" in blob
    txt = watch.alert_text(watch.diff({}, [F("a", "fail", "--token abc123xyz", "f")]))
    assert "abc123xyz" not in txt


def test_a_mistyped_secret_argument_is_never_echoed():
    for argv in (["https://hooks.slack.com/services/T1/B2/SECRETXYZ"],
                 ["--quiet", "https://ntfy.sh/mysecrettopic"],
                 ["--history", "https://ntfy.sh/mysecrettopic"],
                 ["--notfy=https://ntfy.sh/mysecrettopic"]):
        opts, err = watch.parse_args(argv)
        assert opts is None and err, argv
        assert "SECRETXYZ" not in err and "mysecrettopic" not in err and "https" not in err, (argv, err)
    with tmp_env():
        e = io.StringIO()
        with contextlib.redirect_stderr(e), contextlib.redirect_stdout(io.StringIO()):
            rc = cli.cmd_watch(["https://hooks.slack.com/services/T1/B2/SECRETXYZ"])
        assert rc == 2 and "SECRETXYZ" not in e.getvalue() and "--notify" in e.getvalue()
    assert "--quite" in watch.parse_args(["--quite"])[1]          # an option name is still shown


def test_alert_body_fits_the_smallest_target():
    unsent = "OLD-START " + "old line " * 600 + " OLD-END"
    new = "deneb watch on box:\n- NEW memory [fail]: 1.2 GiB available\n  fix: free some memory"
    body = watch.compose_alert(unsent, new)
    assert len(body) <= watch.MAX_ALERT <= 2000                  # Discord rejects content over 2000
    assert body.startswith("earlier alert not delivered:") and body.endswith(new)
    assert "OLD-END" in body and "OLD-START" not in body           # the oldest part is cut first
    assert watch.compose_alert(None, new) == new
    assert watch.compose_alert("short old", None) == "earlier alert not delivered:\nshort old"
    huge = watch.compose_alert("old", "N" * 9000)
    assert len(huge) <= watch.MAX_ALERT and huge.startswith("N")   # the new alert alone is cut to fit


def test_a_long_backlog_still_goes_out_in_one_body_that_fits():
    url = "https://discord.com/api/webhooks/1/abc"
    sent = []

    def ok(req, timeout=None):
        sent.append(json.loads(req.data.decode())["content"])
        return _Resp()
    with tmp_env():
        with patched(urllib.request, urlopen=lambda *a, **k: (_ for _ in ()).throw(OSError("d"))):
            for i in range(6):
                _run_watch(["--notify", url], [F(f"p{i}", "fail", "disk is full " * 60, "fix")])
            assert len(watch.load_state()["unsent"]) > 2000              # a real backlog built up
        with patched(urllib.request, urlopen=ok):
            _run_watch(["--notify", url], [F("last", "fail", "the newest problem", "fix")])
        assert len(sent) == 1 and len(sent[0]) <= 2000 and "the newest problem" in sent[0]
        assert watch.load_state()["unsent"] is None


# ═══ optional rows, seen -> absent -> fail, --reset, old state files, system units ═══
def test_status_of_none_is_ok():
    assert watch.status_of(None, "not set up (optional)") == "ok"


def test_none_row_is_ok_with_its_evidence():
    rows = [("gateway", "GW", None, "not set up (optional: only needed to reach the model from other machines)", "fx"),
            ("gpu", "GPU", None, "no model loaded right now", "")]
    got = watch.findings_from_rows(rows)
    assert [f["status"] for f in got] == ["ok", "ok"]
    assert "not set up" in got[0]["evidence"] and not got[0].get("seen")


def test_optional_row_seen_then_absent_is_a_failure_with_the_reset_hint():
    absent = [("gateway", "GW", None, "not set up (optional)", ""),
              ("tunnel", "Tun", None, "not set up (optional)", "")]
    first = watch.findings_from_rows(absent, seen=["gateway", "tunnel"])
    gw, tun = first
    assert gw["status"] == tun["status"] == "fail"
    assert gw["evidence"] == "was running before, now not found (:8002)"
    assert tun["evidence"] == "was running before, now not found (cloudflared)"
    assert gw["fix"] == "start it again; if you removed it on purpose, run deneb watch --reset"
    # a non-optional None row is never turned into a failure by the seen list
    assert watch.findings_from_rows([("gpu", "GPU", None, "n/a", "")], seen=["gpu"])[0]["status"] == "ok"
    # present rows (running, or up but failing) are marked as seen
    up = watch.findings_from_rows([("gateway", "GW", True, "up", ""), ("tunnel", "T", False, "x", "f")])
    assert all(f.get("seen") for f in up)


def _scan_run(rows, argv=()):
    """cmd_watch with the REAL gather/findings logic, only the box probes canned."""
    ok = lambda i: {"id": i, "label": i, "status": "ok", "evidence": "", "fix": ""}  # noqa: E731
    out, err = io.StringIO(), io.StringIO()
    with patched(watch.check, scan=lambda: rows, _serving_cmdline=lambda: ""), \
            patched(watch, probe_disk=lambda p: ok("disk"), probe_memory=lambda: ok("memory"),
                    probe_restarts=lambda n, u=None: (ok("restarts"), 1, None), probe_catalog=lambda: ok("catalog")), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.cmd_watch(list(argv))
    return rc, out.getvalue(), err.getvalue()


_GW_UP = [("endpoint", "E", True, "serving", ""), ("gateway", "GW", True, "gw up", ""),
          ("tunnel", "Tun", None, "not set up (optional)", "")]
_GW_GONE = [("endpoint", "E", True, "serving", ""), ("gateway", "GW", None, "not set up (optional)", ""),
            ("tunnel", "Tun", None, "not set up (optional)", "")]


def test_watch_remembers_a_seen_component_and_fails_when_it_vanishes():
    with tmp_env() as (state, _h):
        rc, out, _ = _scan_run(_GW_GONE)                       # never set up: healthy
        assert rc == 0 and "0 fail" in out
        assert watch.load_state()["seen"] == []
        rc, out, _ = _scan_run(_GW_UP)                          # now it runs
        assert rc == 0 and watch.load_state()["seen"] == ["gateway"]
        rc, out, _ = _scan_run(_GW_GONE)                        # and goes away
        assert rc == 2 and "was running before, now not found (:8002)" in out and "NEW" in out
        assert "deneb watch --reset" in out
        assert watch.load_state()["seen"] == ["gateway"]       # remembered until reset
        rc, out, _ = _scan_run(_GW_GONE)                        # stays red, no repeat alert tag
        assert rc == 2 and "NEW" not in out
        rc, out, _ = _scan_run(_GW_UP)                          # back: recovered
        assert rc == 0 and "RECOVERED" in out


def test_reset_forgets_baseline_keeps_history_and_ends_the_failure():
    with tmp_env() as (state, _h):
        _scan_run(_GW_UP)
        _scan_run(_GW_GONE)
        hist_before = Path(state, "history.jsonl").read_text().splitlines()
        assert watch.load_state()["seen"] == ["gateway"]
        rc, out, err = _scan_run(_GW_GONE, ["--reset"])
        assert rc == 0 and "reset" in out.lower() and "history" in out.lower()
        st = watch.load_state()
        assert st["seen"] == [] and st["findings"] == {}
        assert not os.path.exists(os.path.join(state, "watch-state.json.tmp"))     # atomic write
        lines = Path(state, "history.jsonl").read_text().splitlines()
        assert lines[:len(hist_before)] == hist_before and len(lines) == len(hist_before) + 1
        ev = json.loads(lines[-1])
        assert ev["id"] == "reset" and "reset" in ev["evidence"]
        rc, out, _ = _scan_run(_GW_GONE)                        # the removed component is now fine
        assert rc == 0 and "0 fail" in out


def test_reset_with_nothing_saved_is_a_harmless_no_op():
    with tmp_env() as (state, _h):
        rc, out, _ = _scan_run(_GW_UP, ["--reset"])
        assert rc == 0 and "nothing to reset" in out
        assert not os.path.exists(os.path.join(state, "watch-state.json"))   # it saved nothing
        assert not os.path.exists(os.path.join(state, "history.jsonl"))


def test_reset_takes_the_run_lock():
    import fcntl
    with tmp_env() as (state, _h):
        _scan_run(_GW_UP)
        fd = os.open(os.path.join(state, "watch.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rc, out, _ = _scan_run(_GW_UP, ["--reset"])
        finally:
            os.close(fd)
        assert rc == 3 and "reset NOT done: another deneb watch is running" in out   # was rc 0 + "skipping" (item 10)
        assert watch.load_state()["seen"] == ["gateway"]       # untouched


def test_reset_flag_is_strict_and_documented():
    for bad in (["--reset", "--quiet"], ["--reset", "--history"], ["--reset", "--schedule"],
                ["--reset", "--notify", "https://x.example/t"], ["--reset", "extra"], ["--rest"]):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cli.cmd_watch(bad)
        assert rc == 2 and "--reset" in err.getvalue() and "usage" in err.getvalue(), bad
    o, e = watch.parse_args(["--reset"])
    assert e is None and o["reset"] is True
    assert watch.parse_args([])[0]["reset"] is False
    assert "--reset" in cli._HELP


def test_old_state_file_without_seen_still_loads():
    with tmp_env() as (state, _h):
        os.makedirs(state)
        Path(state, "watch-state.json").write_text(json.dumps(
            {"ts": "t", "restarts": 2, "restart_unit": "u.service", "unsent": None,
             "findings": {"endpoint": {"status": "ok", "evidence": "serving"}}}))
        st = watch.load_state()
        assert st["seen"] == [] and st["restarts"] == 2 and st["findings"]["endpoint"]["status"] == "ok"
        rc, out, _ = _scan_run(_GW_UP)                          # and a normal run upgrades it
        assert rc == 0 and watch.load_state()["seen"] == ["gateway"]
        Path(state, "watch-state.json").write_text(json.dumps({"findings": {}, "seen": "nope"}))
        assert watch.load_state()["seen"] == []                 # a wrong-typed seen is dropped


def test_save_state_writes_seen_through_the_atomic_path():
    with tmp_env() as (state, _h):
        watch.save_state([F("a", "ok")], 1, "t", None, None, seen=["tunnel", "gateway"])
        assert watch.load_state()["seen"] == ["gateway", "tunnel"]
        assert sorted(os.listdir(state)) == ["watch-state.json"]


# ── restarts for a system unit ──
def _sys_restarts(count_text, prev, prev_unit=None, unit="ollama.service"):
    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(cmd)
        return {"ok": True, "output": count_text}
    with patched(tools, run=fake_run), \
            patched(watch.check, _port_owner=lambda p: ("77", "ollama"), _unit_of_pid=lambda pid: "",
                    _system_unit_of_pid=lambda pid: unit, serving_port=lambda: 11434):
        return watch.probe_restarts(prev, prev_unit), cmds


def test_restarts_reads_a_system_unit_without_user_scope():
    (f, n, u), cmds = _sys_restarts("4\n", 3)
    assert cmds == ["systemctl show ollama.service -p NRestarts --value"]
    assert u == "system:ollama.service" and n == 4
    assert f["status"] == "warn" and "ollama.service (system)" in f["evidence"]
    assert "journalctl -u ollama.service -e" in f["fix"] and "--user" not in f["fix"]


def test_a_user_unit_and_a_system_unit_of_the_same_name_are_not_confused():
    # prev was the --user "ollama.service" at count 2; now the system one at 50: a first check.
    (f, n, u), _ = _sys_restarts("50", 2, prev_unit="ollama.service")
    assert f["status"] == "ok" and "first check" in f["evidence"] and u == "system:ollama.service"
    # and the stored scoped name is compared with itself on the next run
    (f, n, u), _ = _sys_restarts("51", 50, prev_unit="system:ollama.service")
    assert f["status"] == "warn" and "restarted 1 time" in f["evidence"]


def test_restarts_user_unit_keeps_its_plain_stored_name():
    f, n, u = _restarts("5", None)
    assert u == "llm.service"                  # unchanged for state files written before scopes existed


def test_restarts_message_names_the_resolved_port():
    with patched(watch.check, _port_owner=lambda p: ("42", "x"), _unit_of_pid=lambda pid: "",
                 _system_unit_of_pid=lambda pid: "", serving_port=lambda: 11434):
        f, _n, _u = watch.probe_restarts(None)
    assert f["status"] == "unknown" and ":11434" in f["evidence"] and "8001" not in f["evidence"]


def test_restarts_uses_the_stored_unit_when_nothing_listens():
    # The endpoint row names ollama.service from the stored unit; the restarts row must not
    # say "the :11434 server isn't a systemd unit" about the very same server.
    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(cmd)
        return {"ok": True, "output": "2\n"}
    check.set_hint(11434, "system:ollama.service")
    try:
        with patched(tools, run=fake_run), patched(watch.check, _port_owner=lambda p: ("", ""),
                                                   serving_port=lambda: 11434):
            f, n, u = watch.probe_restarts(None)
    finally:
        check.clear_hint()
    assert cmds == ["systemctl show ollama.service -p NRestarts --value"]
    assert f["status"] == "ok" and "ollama.service (system)" in f["evidence"] and "isn't a systemd unit" not in f["evidence"]
    assert n == 2 and u == "system:ollama.service"
    # a stored --user unit keeps its plain name
    check.set_hint(8001, "llm.service")
    try:
        with patched(tools, run=lambda cmd, *a, **k: {"ok": True, "output": "0"}), \
                patched(watch.check, _port_owner=lambda p: ("", ""), serving_port=lambda: 8001):
            f, n, u = watch.probe_restarts(None)
    finally:
        check.clear_hint()
    assert u == "llm.service" and "llm.service" in f["evidence"]


# ── the real probes on an Ollama box with a SYSTEM unit, no gateway, no tunnel ──
_OLLAMA_CANNED = {
    "ss -ltnp": 'LISTEN 0 4096  127.0.0.1:11434  0.0.0.0:*  users:(("ollama",pid=77,fd=3))\n',
    "systemctl --user status": "Failed to get unit for PID 77",
    "systemctl status 77": "● ollama.service - Ollama Service\n  Loaded: loaded",
    "systemctl is-enabled ollama.service": "enabled\n",
    "systemctl show ollama.service -p NRestarts --value": "0\n",
    "curl -s localhost:11434/v1/models": '{"object":"list","data":[{"id":"llama3:8b"}]}',
    "pgrep -af ollama": "77 /usr/local/bin/ollama serve",
    "ollama ps": "NAME ID SIZE PROCESSOR UNTIL\nllama3:8b abc 5 GB 100% GPU 4 minutes from now",
}


def test_real_probes_on_an_ollama_system_unit_box_are_read_only_and_healthy():
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rule = importlib.import_module("test_deneb_rule")
    calls = []

    def recorder(cmd, *a, **k):
        _assert_read_only(cmd)
        calls.append(cmd)
        for key, val in _OLLAMA_CANNED.items():
            if cmd.startswith(key):
                return {"ok": True, "output": val}
        return {"ok": True, "output": ""}
    with tmp_env() as (state, home):
        restore = rule._install_tripwires()
        try:
            with patched(tools, run=recorder, read_file=lambda p, **k: {"ok": True, "output": ""}):
                fs, n, unit = watch.gather(None)
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    rc = cli.cmd_watch([])
        finally:
            restore()
    by = {f["id"]: f for f in fs}
    assert unit == "system:ollama.service" and n == 0
    assert by["endpoint"]["status"] == "ok" and ":11434" in by["endpoint"]["label"]
    assert by["service"]["status"] == "ok" and "ollama.service (system)" in by["service"]["evidence"]
    assert by["gpu"]["status"] == "ok" and "100% GPU" in by["gpu"]["evidence"]
    assert by["gateway"]["status"] == by["tunnel"]["status"] == by["model_file"]["status"] == "ok"
    assert "systemctl show ollama.service -p NRestarts --value" in calls and "ollama ps" in calls
    assert not any("8001" in c for c in calls), "nothing may still probe the hard-coded port"
    assert rc in (0, 1) and "0 fail" in out.getvalue()


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


# ── #29 / #31 / #62: not-applicable rows, unreadable rows, and a stopped server's port/unit ──
_UNREAD = [("endpoint", "E", True, "serving", ""),
           ("gpu", "GPU", False, "running on CPU: m 100% CPU", "give it the GPU")]


def test_not_applicable_rows_show_a_dash_and_are_not_counted_as_ok():
    with tmp_env():
        rc, out, _ = _scan_run(_GW_GONE)
    plain = out.replace("\033[2m", "").replace("\033[0m", "")
    assert "– GW" in plain and "– Tun" in plain and "✓ GW" not in plain
    assert "5 ok, 0 warn, 0 fail, 0 unknown" in out     # 7 rows shown, the 2 optional ones are not counted


def test_an_unreadable_row_is_unknown_never_recovered_and_keeps_the_old_status():
    unread = [("endpoint", "E", True, "serving", ""),
              ("gpu", "GPU", check.UNKNOWN, "couldn't run `ollama ps` here, so the GPU use can't be read", "")]
    with tmp_env():
        rc, out, _ = _scan_run(_UNREAD)
        assert rc == 2
        rc, out, _ = _scan_run(unread)
        plain = out.replace("\033[2m", "").replace("\033[0m", "")
        assert "? GPU" in plain
        assert "RECOVERED" not in out
        assert rc == 2                                  # the stored failure stands
        assert "1 unknown" in out


def test_stopped_server_is_reported_on_its_last_seen_port_and_unit_in_screen_and_alert():
    sent = []
    nothing = lambda *a: ""  # noqa: E731
    ok = lambda i: {"id": i, "label": i, "status": "ok", "evidence": "", "fix": ""}  # noqa: E731
    with tmp_env() as (state, _h):
        watch.save_state([], 3, "2026-01-01T00:00:00+08:00", "system:ollama.service", None, [], 11434)
        with patched(watch.check, _probe=nothing, _read=nothing, _curl=nothing), \
                patched(watch, probe_disk=lambda p: ok("disk"), probe_memory=lambda: ok("memory"),
                        probe_catalog=lambda: ok("catalog"),
                        send_notify=lambda url, msg: sent.append(msg)):
            watch.check._reset_port_cache()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.cmd_watch(["--notify", "https://ntfy.example.org/t"])
        watch.check._reset_port_cache()
    text = out.getvalue()
    assert ":11434" in text and "not responding on :11434" in text
    assert "ollama.service" in text and "<its-unit>" not in text
    assert sent and ":11434" in sent[0] and "ollama.service" in sent[0] and "<its-unit>" not in sent[0]
    gpu_line = next(ln for ln in text.splitlines() if "GPU accelerator" in ln)
    assert "RECOVERED" not in gpu_line and "CPU (no GPU-offload flag)" not in text


def test_restarts_reads_the_system_unit_of_a_server_owned_by_another_user():
    # ollama.service runs as a system unit under its own user: ss hides the pid, the API says Ollama.
    ss = 'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:*\n'
    models = '{"object":"list","data":[{"id":"m","object":"model"}]}'
    table = {"ss -ltnp": ss, "curl -s localhost:11434/api/version": '{"version":"0.1"}',
             "curl -s localhost:11434/v1/models": models,
             "systemctl show ollama.service -p LoadState": "loaded\nactive\nenabled\n",
             "systemctl show ollama.service -p NRestarts": "2\n"}

    def fake(cmd, *a):
        for k, v in table.items():
            if k in cmd:
                return v
        return ""
    real = (check._probe, check._curl)
    check._probe = fake
    check._curl = lambda port, path: fake(f"curl -s localhost:{port}{path}")
    check._reset_port_cache()
    try:
        with patched(watch.tools, run=lambda c, *a, **k: {"output": fake(c)}):
            f, n, unit = watch.probe_restarts(None)
    finally:
        check._probe, check._curl = real
        check._reset_port_cache()
    assert unit == "system:ollama.service" and n == 2 and f["status"] == "ok", f


def test_unschedule_prints_the_undo_for_exactly_what_schedule_creates_and_runs_nothing():
    import subprocess
    calls = []
    real = subprocess.run
    subprocess.run = lambda *a, **k: calls.append(a) or real(*a, **k)
    try:
        with tmp_env() as (state, home):
            rc, out = _schedule(["--unschedule"])
            assert rc == 0
            for needle in ("systemctl --user disable --now deneb-watch.timer",
                           "~/.config/systemd/user/deneb-watch.timer",
                           "~/.config/systemd/user/deneb-watch.service",
                           "systemctl --user daemon-reload", "loginctl disable-linger",
                           "ran nothing"):
                assert needle in out, needle
            sched = _schedule(["--schedule"])[1]
            for unit in ("deneb-watch.service", "deneb-watch.timer"):
                assert unit in sched and unit in out         # the same two units, both ways
            assert os.listdir(home) == [] and not os.path.exists(state)
    finally:
        subprocess.run = real
    assert calls == []                                        # tell-only: nothing was executed


def test_unschedule_takes_no_other_option():
    with tmp_env():
        for bad in (["--unschedule", "--quiet"], ["--unschedule", "--schedule"],
                    ["--unschedule", "--every", "5m"], ["--unschedule", "--reset"]):
            rc, _ = _schedule(bad)
            assert rc == 2, bad


def test_readme_uninstall_runs_unschedule_first():
    readme = open(os.path.join(os.path.dirname(__file__), "..", "README.md"), encoding="utf-8").read()
    sec = readme[readme.index("## Uninstall"):]
    assert sec.index("deneb watch --unschedule") < sec.index("pipx uninstall")


def test_unschedule_also_prints_reset_failed_and_names_the_model_server_unit_for_linger():
    with tmp_env() as (state, home):
        rc, out = _schedule(["--unschedule"])
    assert rc == 0
    assert "systemctl --user reset-failed deneb-watch.service" in out
    # the optional disable-linger step says the model server unit setup creates would stop too
    tail = out[out.index("loginctl disable-linger") - 700:out.index("loginctl disable-linger")]
    assert "llama-server.service" in tail, tail


def test_top_level_help_mentions_watch_unschedule():
    import io
    from contextlib import redirect_stdout
    from deneb import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(["--help"])
    assert "--unschedule" in buf.getvalue()


# ── r4/R1: an unreadable CURRENT row is never an exit 0, whatever the previous run said ──
def test_unknown_after_an_ok_run_exits_1_not_0_also_quiet():
    for argv in ([], ["--quiet"]):
        with tmp_env():
            assert _run_watch([], [F("x", "ok", "fine")])[0] == 0
            rc, out, _ = _run_watch(argv, [F("x", "unknown", "can't read")])
            assert rc == 1, (argv, out)
            # what is SAVED still carries the old status (a blind probe must not rewrite history)
            assert watch.load_state()["findings"]["x"]["status"] == "ok"


def test_unknown_over_a_stored_fail_exits_2_and_all_ok_exits_0():
    with tmp_env():
        assert _run_watch([], [F("x", "fail", "down")])[0] == 2
        assert _run_watch([], [F("x", "unknown", "blind")])[0] == 2
        assert _run_watch([], [F("x", "ok", "fine")])[0] == 0


def test_watch_exit_code_text_says_a_row_it_could_not_read():
    want = "1 a warning or a row it could not read"
    assert want in watch.__doc__
    with tmp_env():
        _, out = _schedule(["--schedule"])
    assert "1 a warning or a row it could not read" in out
    assert want in watch.main.__doc__
