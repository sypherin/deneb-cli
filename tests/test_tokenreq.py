"""deneb auth --request / --check against a fake engine on 127.0.0.1."""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from contextlib import redirect_stdout

import pytest

from deneb import cli, config
from fakes import FakeEngine

ARGS = ["auth", "--request", "--name", "Ada Lovelace", "--email", "ada@example.com",
        "--box", "Strix Halo 128GB, Ollama", "--why", "slow model"]


def _run(argv, monkeypatch=None, stdin=None):
    buf = io.StringIO()
    if stdin is not None and monkeypatch is not None:
        it = iter(stdin)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


@pytest.fixture
def engine(monkeypatch):
    with FakeEngine() as e:
        monkeypatch.setenv("DENEB_ENGINE", e.url)
        yield e


def test_request_sends_only_the_hash_and_stores_the_secret_mode_600(engine):
    rc, out = _run(ARGS)
    assert rc == 0
    body = engine.posts[0]
    assert set(body) == {"name", "email", "box", "problem", "secret_hash", "client"}
    assert body["name"] == "Ada Lovelace" and body["email"] == "ada@example.com"
    assert body["problem"] == "slow model" and body["client"].startswith("deneb-cli/")
    saved = json.load(open(config.REQUEST_FILE))
    assert saved["id"] == "abc123XYZ9" and len(saved["secret"]) >= 43
    assert body["secret_hash"] == hashlib.sha256(saved["secret"].encode()).hexdigest()
    assert saved["secret"] not in json.dumps(body)
    assert stat.S_IMODE(os.stat(config.REQUEST_FILE).st_mode) == 0o600
    assert "a person at Altronis reviews each request" in out
    assert "request sent. once it's approved, run deneb auth --check" in out
    assert "sending this to Altronis" in out


def test_request_prompts_when_flags_are_missing(engine, monkeypatch):
    rc, out = _run(["auth", "--request"], monkeypatch,
                   stdin=["Grace Hopper", "grace@example.org", "DGX Spark", ""])
    assert rc == 0
    b = engine.posts[0]
    assert b["name"] == "Grace Hopper" and b["box"] == "DGX Spark" and b["problem"] == ""


def test_request_rejects_a_bad_email_before_sending(engine):
    rc, out = _run(["auth", "--request", "--name", "A", "--email", "nope", "--box", "x"])
    assert rc != 0 and engine.posts == [] and not os.path.exists(config.REQUEST_FILE)
    assert "email" in out


def test_check_pending_then_approved_saves_the_token_and_removes_request_json(engine):
    _run(ARGS)
    rc, out = _run(["auth", "--check"])
    assert rc == 0 and "pending" in out.lower() and os.path.exists(config.REQUEST_FILE)
    secret = json.load(open(config.REQUEST_FILE))["secret"]
    assert engine.poll_headers[-1].get("X-Deneb-Request-Secret") == secret
    engine.status = "approved"
    rc, out = _run(["auth", "--check"])
    assert rc == 0 and "approved" in out.lower()
    assert config.get_token() == "dnb_faketoken"
    assert not os.path.exists(config.REQUEST_FILE)
    assert "dnb_faketoken" not in out        # never echoed


def test_check_denied_and_expired_clear_the_request(engine):
    for status in ("denied", "expired"):
        _run(ARGS)
        engine.status = status
        rc, out = _run(["auth", "--check"])
        assert status in out and not os.path.exists(config.REQUEST_FILE)
        assert config.get_token() is None


def test_check_claimed_says_so(engine):
    _run(ARGS)
    engine.status = "claimed"
    rc, out = _run(["auth", "--check"])
    assert "already" in out and config.get_token() is None


def test_check_unknown_404_is_plain(engine):
    _run(ARGS)
    engine.poll_code = 404
    rc, out = _run(["auth", "--check"])
    assert rc != 0 and "deneb auth --request" in out


def test_check_with_no_request_says_so(engine):
    rc, out = _run(["auth", "--check"])
    assert rc != 0 and "no request" in out.lower()


def test_429_on_post_shows_the_servers_message_and_keeps_no_file(engine):
    engine.post_code = 429
    engine.post_body = {"error": "a request for this email is already waiting"}
    rc, out = _run(ARGS)
    assert rc != 0 and "a request for this email is already waiting" in out
    assert not os.path.exists(config.REQUEST_FILE)


@pytest.mark.parametrize("code,msg", [(413, "too large"), (400, "bad field")])
def test_413_and_400_are_plain_messages(engine, code, msg):
    engine.post_code = code
    engine.post_body = {"error": msg}
    rc, out = _run(ARGS)
    assert rc != 0 and not os.path.exists(config.REQUEST_FILE)
    assert str(code) in out or msg in out


def test_429_on_poll_shows_the_servers_message(engine):
    _run(ARGS)
    engine.poll_code = 429
    rc, out = _run(["auth", "--check"])
    assert "too many requests" in out and os.path.exists(config.REQUEST_FILE)


def test_unreachable_engine_is_plain_and_does_not_crash(monkeypatch):
    monkeypatch.setenv("DENEB_ENGINE", "http://127.0.0.1:1")
    rc, out = _run(ARGS)
    assert rc != 0 and "can't reach" in out and not os.path.exists(config.REQUEST_FILE)
    # and polling with a request on disk
    config.save_request({"id": "x", "secret": "s", "created": "t"})
    rc, out = _run(["auth", "--check"])
    assert rc != 0 and "can't reach" in out and os.path.exists(config.REQUEST_FILE)


def test_a_pending_request_is_polled_once_before_a_one_shot(engine, monkeypatch):
    from deneb import brain
    _run(ARGS)
    engine.status = "approved"
    seen = {}

    def fake_run(question, history=None, on_event=None, confirm=None):
        seen["token"] = config.get_token()
        return {"type": "final", "answer": "ok", "escalated": False, "sources": []}
    monkeypatch.setattr(cli.loop, "run", fake_run)
    rc, out = _run(["why is it slow"])
    assert seen["token"] == "dnb_faketoken" and not os.path.exists(config.REQUEST_FILE)
    assert len([h for h in engine.poll_headers]) == 1


def test_no_poll_without_a_pending_request(engine, monkeypatch):
    monkeypatch.setattr(cli.loop, "run", lambda *a, **k: {"type": "final", "answer": "ok",
                                                          "escalated": False, "sources": []})
    config.save(token="t")
    _run(["hello"])
    assert engine.poll_headers == []


def test_auth_token_still_works(engine):
    rc, out = _run(["auth", "--token", "abc"])
    assert rc == 0 and config.get_token() == "abc"


def test_daily_cap_429_on_an_authed_route_prints_the_engine_text_plainly(engine):
    config.save(token="dnb_x")
    engine.authed_code = 429
    engine.authed_body = {"error": "daily limit for this token reached, resets at 00:00 SGT"}
    rc, out = _run(["hello"])
    assert rc != 0
    assert "daily limit for this token reached, resets at 00:00 SGT" in out
    assert "engine returned" not in out and "Traceback" not in out


def test_revoked_token_401_points_at_a_new_request(engine):
    config.save(token="dnb_x")
    engine.authed_code = 401
    engine.authed_body = {"error": "unauthorized"}
    rc, out = _run(["hello"])
    assert rc != 0 and "no longer valid" in out and "deneb auth --request" in out


def test_check_revoked_is_handled_like_denied(engine, monkeypatch):
    """The engine answers a poll with {"status":"revoked"}: clear the stored request, say so
    in plain words, and never print "unexpected answer"."""
    from deneb import net
    _run(ARGS)
    assert os.path.exists(config.REQUEST_FILE)
    seen = []

    def stub(method, url, body=None, headers=None, timeout=None):
        seen.append((method, url))
        return 200, json.dumps({"status": "revoked"}).encode()

    monkeypatch.setattr(net, "request", stub)
    rc, out = _run(["auth", "--check"])
    assert seen and seen[0][0] == "GET"          # the stub, not the fake engine, answered
    assert "this request's token was revoked. the free commands and your own model still work." in out
    assert "unexpected" not in out
    assert not os.path.exists(config.REQUEST_FILE)
    assert config.get_token() is None
