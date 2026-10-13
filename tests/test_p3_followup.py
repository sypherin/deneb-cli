"""Part 3 follow-up fixes. Nothing here runs a proposed command; subprocess, input() and the
network are stubbed wherever a path could reach them."""
from __future__ import annotations

import io
import unicodedata
from contextlib import redirect_stdout

import pytest

from deneb import ui


def _say(fn, *a):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*a)
    return out, buf.getvalue()


# ── 1: Unicode format characters are shown escaped ──────────────────────────────────────
def _fmt_points():
    pts = list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A)) + list(range(0x200B, 0x2010))
    pts += list(range(0x2060, 0x2065)) + [0xFEFF, 0x061C, 0x180E]
    return sorted(set(pts))


def _has_format_char(s):
    return any(unicodedata.category(c) == "Cf" or c == "᠎" for c in s)


@pytest.mark.parametrize("cp", _fmt_points())
def test_format_characters_are_escaped_in_cmd_path_and_content(cp):
    ch = chr(cp)
    esc = f"\\u{cp:04x}"
    cmd = _say(ui.event, "propose", {"action": "fix", "cmd": f"rm ~/notes{ch}txt.sh", "why": "w"})[1]
    assert not _has_format_char(cmd) and esc in cmd
    wr = _say(ui.event, "propose", {"action": "write_file", "path": f"~/notes{ch}txt.sh",
                                    "content": f"echo a{ch}b\n", "why": "w"})[1]
    assert not _has_format_char(wr)
    assert wr.count(esc) == 2                      # once in the path, once in the content line


def test_visible_escapes_a_right_to_left_override_in_a_path():
    assert ui.visible("~/notes‮txt.sh") == "~/notes\\u202etxt.sh"


def test_ordinary_text_survives_in_content():
    txt = "中文 வணக்கம் தமிழ் café naïve\tend"
    out = ui.content_preview(txt + "\n")
    assert out == [txt]


# ── 2: prose before a bare </think> is kept ─────────────────────────────────────────────
def test_prose_that_mentions_a_bare_close_think_tag_is_kept_whole():
    from deneb import jsonrecover as jr
    text = ("Qwen templates put reasoning before </think>, so set --reasoning-format to hide it "
            "in your client.")
    assert jr.strip_think(text) == text
    assert jr.parse_step(text)["answer"] == text


def test_reasoning_then_a_bare_close_tag_then_an_action_is_still_stripped():
    from deneb import jsonrecover as jr
    r = jr.parse_step('I should look at the port.</think>{"type":"action","action":"run","cmd":"ss -ltn"}')
    assert r["type"] == "action" and r["cmd"] == "ss -ltn"
    r = jr.parse_step('hmm</think>\n```json\n{"type":"action","action":"run","cmd":"ss -ltn"}\n```')
    assert r["type"] == "action" and r["cmd"] == "ss -ltn"


# ── 3: token-only userinfo is redacted ──────────────────────────────────────────────────
@pytest.mark.parametrize("url", ["postgres://fake-not-a-real-secret-123@db.example.com/x",
                                 "https://fake-not-a-real-secret-123@example.com/"])
def test_token_only_userinfo_is_redacted(url):
    from deneb import tools
    out = tools._redact(f"connecting to {url} now")
    assert "fake-not-a-real-secret-123" not in out
    assert "[REDACTED]@" in out and out.endswith(" now")


def test_a_url_with_no_userinfo_is_unchanged():
    from deneb import tools
    assert tools._redact("http://localhost:8001/") == "http://localhost:8001/"


# ── 4: continuation lines of a later Exec line are listed with it ───────────────────────
def test_preview_lists_the_continuation_lines_of_a_later_exec_line():
    lines = [f"line{i}" for i in range(1, 71)]
    lines[69 - 1] = "ExecStart=/bin/true \\"          # line 69
    lines[70 - 1] = "  --first-extra \\"             # line 70
    lines.append("  --second-extra")                 # line 71
    lines += [f"tail{i}" for i in range(72, 81)]     # lines 71.. total 80 -> fix below
    out = ui.content_preview("\n".join(lines) + "\n")
    text = "\n".join(out)
    assert "69: ExecStart=/bin/true \\" in text
    assert "70:   --first-extra \\" in text
    assert "71:   --second-extra" in text
    assert "tail72" not in text
    total = len(lines)
    assert f"{total - 40 - 3} more lines not shown" in text


def test_preview_continuation_that_starts_before_the_fold_is_not_double_counted():
    lines = [f"line{i}" for i in range(1, 71)]
    lines[39 - 1] = "ExecStart=/bin/true \\"          # line 39, shown in the head
    lines[40 - 1] = "  --head-cont \\"
    lines[41 - 1] = "  --after-fold"                 # line 41: continuation of a head Exec line
    out = ui.content_preview("\n".join(lines) + "\n")
    text = "\n".join(out)
    assert "41:   --after-fold" in text
    assert "29 more lines not shown" in text         # 70 lines, 40 head, 1 listed


# ── 6: setup prints exactly one way to start the server ─────────────────────────────────
def _setup_output(monkeypatch, box_name):
    import test_setup_advisor as tsa
    from deneb import cli, hardware, hf_catalog
    box = getattr(tsa, box_name)()
    monkeypatch.setattr(hardware, "profile_hardware", lambda: box)
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("apt", 500))
    name = hf_catalog.all_models()[0].name
    return _say(cli.cmd_setup, [name])[1]


def _server_starts(out):
    """Printed commands that start the server: the foreground llama-server line or the
    systemd `restart`."""
    n, cmd, in_cmd = 0, [], False
    blocks = []
    for ln in out.splitlines():
        if ln.strip().startswith("$ "):
            in_cmd, cmd = True, [ln.strip()[2:]]
        elif in_cmd and ln.strip().startswith(("what:", "⚠")):
            blocks.append("\n".join(cmd))
            in_cmd = False
        elif in_cmd:
            cmd.append(ln)
    for c in blocks:
        if "systemctl --user restart llama-server" in c or c.split(" ", 1)[0].endswith("llama-server"):
            n += 1
    return n


def test_following_every_printed_setup_step_in_order_starts_the_server_once(monkeypatch):
    out = _setup_output(monkeypatch, "strix_box")
    assert "systemctl --user restart llama-server" in out                       # this box shape gets the user service
    assert _server_starts(out) == 1, out


def test_the_footer_and_the_service_step_do_not_contradict(monkeypatch):
    out = _setup_output(monkeypatch, "strix_box")
    assert "top to bottom" in out
    assert "instead of" not in out and "not as well" not in out


def test_a_box_with_no_user_service_still_starts_the_server_once(monkeypatch):
    out = _setup_output(monkeypatch, "metal_box")
    assert "enable --now" not in out
    assert _server_starts(out) == 1, out


# ── 7: an engine with no /token-request route says so plainly ───────────────────────────
RAW_404 = b'{"error": "Cannot POST /token-request RAW-SERVER-TEXT-XYZ"}'


def _run_cli(argv):
    from deneb import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def _assert_plain_no_route(out):
    assert "RAW-SERVER-TEXT-XYZ" not in out and "Cannot POST" not in out
    assert "does not take token requests yet" in out
    assert "deneb brain local" in out


def test_auth_request_against_an_engine_without_the_route_is_plain(monkeypatch):
    from deneb import net
    monkeypatch.setenv("DENEB_ENGINE", "http://engine.invalid")
    monkeypatch.setattr(net, "request", lambda *a, **k: (404, RAW_404))
    rc, out = _run_cli(["auth", "--request", "--name", "Ada", "--email", "ada@example.com",
                        "--box", "Strix Halo", "--why", "x"])
    assert rc == 1
    _assert_plain_no_route(out)


def test_auth_check_against_an_engine_without_the_route_is_plain(monkeypatch):
    from deneb import config, net
    monkeypatch.setenv("DENEB_ENGINE", "http://engine.invalid")
    config.save_request({"id": "fake-id-1", "secret": "fake-not-a-real-secret-123",
                         "email": "ada@example.com", "engine": "http://engine.invalid",
                         "created": "2026-01-01T00:00:00+0800"})
    monkeypatch.setattr(net, "request", lambda *a, **k: (404, RAW_404))
    rc, out = _run_cli(["auth", "--check"])
    assert rc != 0
    _assert_plain_no_route(out)


# ── 8: the private deny-list is applied to everything that ships ────────────────────────
def test_the_whole_tree_is_free_of_the_private_names():
    import os
    import private_names
    pats = private_names.require()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skip_dirs = {"__pycache__"}
    me = os.path.abspath(__file__)
    seen = 0
    for dirpath, dirs, files in (w for top in ("deneb",) for w in os.walk(os.path.join(root, top))):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            path = os.path.join(dirpath, f)
            if path == me or not f.endswith((".py", ".md", ".txt", ".toml", ".json", ".sh", ".cfg")):
                continue
            seen += 1
            text = open(path, encoding="utf-8", errors="replace").read()
            for n, rx in enumerate(pats, 1):
                assert not rx.search(text), f"private name (pattern #{n}) in {os.path.relpath(path, root)}"
    assert seen > 10
