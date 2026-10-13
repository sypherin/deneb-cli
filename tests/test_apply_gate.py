"""Part 3 review, group E: the apply gate. --auto never auto-approves a file write (S3), a
write_file that would plant a unit running a shell or interpreter is refused (S3), the write
preview cannot hide a payload below the fold (S3), and every proposal display shows control
characters instead of obeying them (S4). subprocess is never run here; files go under tmp."""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pytest

from deneb import cli, client, loop, tools, ui

AUTO_LINE = "--auto applies fix commands only; file writes always ask"


def _say(fn, *a):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*a)
    return out, buf.getvalue()


def _no_input(monkeypatch):
    def boom(prompt=""):
        raise AssertionError("asked a question where none was expected")
    monkeypatch.setattr("builtins.input", boom)


# ── S3: --auto and write_file ─────────────────────────────────────────────────────────
def test_auto_still_approves_a_fix_without_asking(monkeypatch):
    _no_input(monkeypatch)
    ok, out = _say(cli._make_confirm(True), {"action": "fix", "cmd": "systemctl --user restart x"})
    assert ok is True and "--auto: applying" in out


@pytest.mark.parametrize("answer,expected", [("n", False), ("", False), ("y", True)])
def test_auto_always_asks_before_a_file_write(monkeypatch, answer, expected):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or answer)
    ok, out = _say(cli._make_confirm(True), {"action": "write_file", "path": "/tmp/x", "content": "c"})
    assert ok is expected and len(asked) == 1
    assert AUTO_LINE in out and "--auto: applying" not in out


def test_auto_file_write_with_no_terminal_declines(monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    ok, _ = _say(cli._make_confirm(True), {"action": "write_file", "path": "/tmp/x", "content": "c"})
    assert ok is False


def test_without_auto_both_kinds_ask(monkeypatch):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(1) or "n")
    cli._make_confirm(False)({"action": "fix", "cmd": "x"})
    cli._make_confirm(False)({"action": "write_file", "path": "/tmp/x"})
    assert len(asked) == 2


def test_loop_under_auto_does_not_write_a_file_when_the_user_says_no(monkeypatch):
    steps = iter([{"type": "action", "action": "write_file", "path": "/tmp/x", "content": "c", "why": "w"},
                  {"type": "final", "answer": "done", "escalated": False, "sources": []}])
    monkeypatch.setattr(client, "agent_step", lambda hist, timeout=180: next(steps))
    writes = []
    monkeypatch.setattr(tools, "execute_write", lambda *a, **k: writes.append((a, k)) or {"ok": True, "output": ""})
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with redirect_stdout(io.StringIO()):
        res = loop.run("q", on_event=ui.event, confirm=cli._make_confirm(True))
    assert res["type"] == "final" and writes == []


def test_the_auto_banner_and_help_say_file_writes_still_ask(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    monkeypatch.setattr(cli, "_prepare", lambda *a, **k: True)
    _, out = _say(cli._interactive, True)
    assert "file writes" in out and "still ask" in out
    assert AUTO_LINE in cli._HELP


def test_readme_states_the_auto_limit_truthfully():
    text = " ".join(open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read().split())
    # the README says both halves once, in plain words (the --help line is the terse form)
    assert "`--auto` applies fix commands without asking" in text
    assert text.count("file writes always ask") == 1, "said once, not repeated"
    assert "`--auto` applies fix commands without asking, but file writes always ask." in text


# ── S3: write_file and systemd units ───────────────────────────────────────────────────
@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".config" / "systemd" / "user").mkdir(parents=True)
    return tmp_path


def _unit(exec_line, extra=""):
    return f"[Unit]\nDescription=x\n\n[Service]\n{extra}{exec_line}\n\n[Install]\nWantedBy=default.target\n"


BAD_EXECS = [
    "ExecStart=/bin/sh -c 'echo hi'", "ExecStart=/usr/bin/bash -c x", "ExecStart=/bin/dash -c x",
    "ExecStart=/usr/bin/zsh -c x", "ExecStart=/usr/bin/env FOO=1 /usr/bin/x",
    "ExecStart=/usr/bin/python3 -c 'print(1)'", "ExecStart=/usr/bin/python3.12 x.py",
    "ExecStart=/usr/bin/perl -e 1", "ExecStart=/usr/bin/ruby x.rb", "ExecStart=/usr/bin/node x.js",
    "ExecStart=/usr/bin/php x.php", "ExecStart=/usr/bin/lua x.lua",
    "ExecStart=relative/llama-server", "ExecStart=llama-server --port 1",
    "ExecStart=%h/bin/llama-server", "ExecStartPre=/bin/sh -c x", "ExecStartPost=-/bin/sh -c x",
    "ExecStop=/usr/bin/env x", "ExecReload=+/bin/sh -c x", "ExecStart=@/bin/sh sh -c x",
    "ExecStart=-/bin/sh -c x", "ExecStart=!/bin/sh -c x", 'ExecStart="/bin/sh" -c x',
    "ExecStart=/usr/local/bin/ok \\\n  && x\nExecStartPost=/bin/sh -c y",
]


@pytest.mark.parametrize("line", BAD_EXECS)
def test_unit_whose_exec_runs_a_shell_or_interpreter_or_non_absolute_is_refused(home, line):
    p = home / ".config" / "systemd" / "user" / "x.service"
    r = tools.write_file(str(p), _unit(line))
    assert r.get("refused") or r.get("ok") is False, (line, r)
    assert "write that unit yourself" in r["output"], r
    assert not p.exists()


@pytest.mark.parametrize("name", ["x.service", "x.timer", "x.socket", "x.path", "x.conf", "override.conf"])
def test_every_unit_kind_and_every_file_under_systemd_user_is_checked(home, name):
    p = home / ".config" / "systemd" / "user" / name
    r = tools.write_file(str(p), _unit("ExecStart=/bin/sh -c x"))
    assert "write that unit yourself" in r["output"] and not p.exists()


def test_a_unit_name_outside_the_systemd_dir_is_checked_too(home):
    p = home / "elsewhere.timer"
    r = tools.write_file(str(p), _unit("ExecStart=/bin/sh -c x"))
    assert "write that unit yourself" in r["output"] and not p.exists()


@pytest.mark.parametrize("line", [
    "ExecStart=/usr/local/bin/llama-server --port 8001 -m /models/x.gguf",
    "ExecStart=", "ExecStart=-/usr/local/bin/helper", "ExecStart=/opt/node_exporter/node_exporter",
    "ExecStart=/usr/bin/pythonista", "ExecStart=/usr/bin/shellcheck x",
    "Environment=SH=/bin/sh",
])
def test_an_ordinary_unit_is_still_written(home, line):
    p = home / ".config" / "systemd" / "user" / "ok.service"
    r = tools.write_file(str(p), _unit(line))
    assert r["ok"] is True and p.exists(), r


def test_a_non_unit_file_with_exec_text_is_not_touched_by_the_unit_rule(home):
    p = home / "notes.txt"
    r = tools.write_file(str(p), "ExecStart=/bin/sh -c x\n")
    assert r["ok"] is True


# ── S3: the preview cannot hide a payload ───────────────────────────────────────────────
def _lines(n, exec_at=()):
    return "\n".join(f"ExecStartPre=/payload{i}" if i in exec_at else f"line{i}" for i in range(1, n + 1))


def test_preview_shows_the_whole_content_up_to_60_lines():
    out = ui.content_preview(_lines(60))
    assert len(out) == 60 and not any("more lines" in x for x in out)


def test_preview_past_60_shows_40_then_only_exec_lines_with_numbers_and_a_count():
    out = ui.content_preview(_lines(100, exec_at=(70, 90)))
    text = "\n".join(out)
    assert "line40" in text and "line41" not in text and "line50" not in text
    assert "70: ExecStartPre=/payload70" in text and "90: ExecStartPre=/payload90" in text
    assert "58 more lines not shown" in text


def test_preview_with_no_later_exec_line_counts_all_the_hidden_lines():
    out = ui.content_preview(_lines(61, exec_at=(5,)))
    assert "21 more lines not shown" in "\n".join(out)      # 61 lines, 40 shown, no later Exec line


# ── S4: control characters are shown, not obeyed ────────────────────────────────────────
SPOOF = "systemctl --user stop llama-server.service\r\x1b[2Kmkdir -p ~/models"


def _event(kind, d):
    return _say(ui.event, kind, d)[1]


def _raw_controls(s):
    return [c for c in s if (ord(c) < 32 and c not in "\n\t") or ord(c) == 0x7f or 0x80 <= ord(c) <= 0x9f]


def test_fix_cmd_with_a_carriage_return_and_escape_is_shown_visibly():
    out = _event("propose", {"action": "fix", "cmd": SPOOF, "why": "w"})
    assert _raw_controls(out) == []
    assert "\\r" in out and "\\x1b" in out
    assert "systemctl --user stop llama-server.service" in out and "mkdir -p ~/models" in out


@pytest.mark.parametrize("field", ["cmd", "why", "path"])
def test_every_proposal_field_is_sanitised(field):
    d = {"action": "write_file" if field == "path" else "fix", "cmd": "c", "why": "w", "path": "/p"}
    d[field] = "a\x07b\x1b[31mc\rd\x00e"
    out = _event("propose", d)
    assert _raw_controls(out) == [] and "\\x07" in out and "\\x1b" in out


def test_write_file_path_and_content_are_sanitised_but_content_keeps_newline_and_tab():
    out = _event("propose", {"action": "write_file", "path": "/a\x1b[2K/b",
                             "content": "k=v\tx\n\x1b[31mred\rback\n", "why": "w"})
    assert _raw_controls(out) == []
    assert "\\x1b" in out and "\\r" in out
    assert "k=v\tx" in out                      # the tab survived


def test_the_read_only_action_line_is_sanitised_too():
    out = _event("action", {"action": "run", "cmd": SPOOF})
    assert _raw_controls(out) == []
