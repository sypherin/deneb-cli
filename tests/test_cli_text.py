"""What a newcomer sees: the help, and the conversational mode run without a token.

The free commands are the way in, so a bare `deneb` with no token must point at them
instead of dead-ending at "sign in", and none of it may lean on an internal name or the word "runbook".
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deneb import cli, client, config  # noqa: E402
import private_names  # noqa: E402

FREE = ("deneb check", "deneb watch", "deneb profile", "deneb recommend", "deneb setup")


def _no_token_run(argv):
    real = config.get_token
    config.get_token = lambda: None
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli.main(argv)
    finally:
        config.get_token = real
    return rc, buf.getvalue()


def test_bare_deneb_without_a_token_points_at_the_free_commands():
    rc, out = _no_token_run([])
    assert rc != 0
    assert "deneb auth --request" in out
    for c in FREE:
        assert c in out, c
    assert "runbook" not in out.lower()
    private_names.assert_clean(out, "output")


def test_a_one_shot_question_without_a_token_points_at_the_free_commands():
    rc, out = _no_token_run(["llama-server", "won't", "start"])
    assert rc != 0
    for c in FREE:
        assert c in out, c


def test_the_engine_client_without_a_token_names_the_free_commands():
    real = config.get_token
    config.get_token = lambda: None
    try:
        try:
            client._post("/x", {}, timeout=1)
        except client.DenebError as e:
            msg = str(e)
        else:
            raise AssertionError("no error without a token")
    finally:
        config.get_token = real
    assert "deneb check" in msg and "runbook" not in msg.lower()
    private_names.assert_clean(msg, "message")


def test_help_lists_the_free_commands_before_the_token_mode_and_says_what_is_sent():
    h = cli._HELP
    private_names.assert_clean(h, "help")
    free_at = h.index("free, no token")
    token_at = h.index("conversational troubleshooting")
    assert free_at < token_at
    for c in FREE:
        assert free_at < h.index(c) < token_at, c
    assert "sent to the Altronis engine" in h
    assert "nothing is sent to Altronis" in h
    assert "--auto" in h[token_at:]


def test_help_header_does_not_claim_nothing_is_sent_anywhere():
    h = cli._HELP
    assert "anywhere" not in h
    line = next(ln for ln in h.splitlines() if ln.startswith("free, no token"))
    assert "nothing sent unless you ask" in line and len(line) < 80, (len(line), line)
    for sends in ("alerts", "--refresh", "--measured"):
        assert sends in line


def test_readme_does_not_claim_nothing_is_sent_anywhere():
    readme = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read()
    assert "nothing anywhere" not in readme
    first = next(ln for ln in readme.splitlines() if "need no account or token" in ln)
    assert "unless you ask" in first


def test_help_mentions_no_internal_name_in_any_user_string():
    from deneb import ui
    import inspect
    src = inspect.getsource(ui)
    private_names.assert_clean(src, "ui source")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("ok  ", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, type(e).__name__, e)
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
