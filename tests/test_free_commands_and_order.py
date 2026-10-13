"""#49 one list of free commands, #63 logout with nothing saved, #66 one next-step order."""
from __future__ import annotations

import io
import os
import re
from contextlib import redirect_stdout

from deneb import cli, client, config, freecmds

ROOT = os.path.join(os.path.dirname(__file__), "..")
ORDER = ["deneb check", "deneb recommend", "deneb setup", "deneb watch"]


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_one_list_feeds_the_no_token_message_the_engine_error_and_help():
    names = freecmds.NAMES
    assert "deneb catalog" in names and "deneb check" in names
    assert cli._FREE_HINT == freecmds.hint()
    config_token = config.get_token
    config.get_token = lambda: None
    try:
        try:
            client._post("/x", {}, timeout=1)
        except client.DenebError as e:
            msg = str(e)
    finally:
        config.get_token = config_token
    _, helptext = _run(["--help"])
    for n in names:
        assert n in freecmds.hint() and n in msg and n in helptext, n


def test_readme_names_every_free_command():
    readme = _read("README.md")
    for n in freecmds.NAMES:
        assert n in readme, n


def test_next_step_order_is_the_same_in_help_installer_and_readme():
    def positions(text):
        return [text.index(c) for c in ORDER]
    _, helptext = _run(["--help"])
    for text in (helptext, _read("install.sh"), _read("README.md")[:1400]):
        p = positions(text)
        assert p == sorted(p), (text[:60], p)


def test_logout_with_nothing_saved_says_no_token_saved(monkeypatch):
    monkeypatch.delenv("DENEB_TOKEN", raising=False)
    rc, out = _run(["logout"])
    assert rc == 0 and "no token saved" in out and "signed out" not in out


def test_logout_with_a_token_still_signs_out():
    config.save(token="t")
    rc, out = _run(["logout"])
    assert rc == 0 and "signed out" in out and config.get_token() is None


def test_readme_check_sample_is_the_clean_box_capture_with_no_internal_service():
    readme = _read("README.md")
    assert "qwen27-fpx" not in readme and "llm-gateway.service" not in readme
    a = readme.index("## `deneb check`")
    block = readme[readme.index("```\n", a) + 4:]
    block = block[:block.index("```")]
    rows = [r for r in block.splitlines() if r.strip()]
    assert rows[-1] == "✓ all checks pass: the model is served and ready."
    assert rows[0].startswith("  ✓ LLM endpoint serving (:11434)")
    assert sum(r.startswith("  –") for r in rows) == 3        # model file + the 2 optional rows
    assert not any(r.startswith("  ?") for r in rows)          # "all checks pass" only with no '?'
