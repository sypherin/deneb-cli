"""Part 3 review, group C: logout and token requests (S2, S6, S7, S10, N8, N12), and which brain
gets picked and what is said when none answers (S8, S9, S11, N7). Fake servers on 127.0.0.1 only."""
from __future__ import annotations

import io
import json
import os
import time
from contextlib import redirect_stdout

import pytest

from deneb import brain, check, cli, config, tokenreq, ui
from deneb.client import DenebError
from fakes import FakeBlackHole, FakeEngine, FakeOpenAI

ARGS = ["auth", "--request", "--name", "Ada Lovelace", "--email", "ada@example.com",
        "--box", "Strix Halo 128GB, Ollama", "--why", "slow model"]
FAKE = "fake-not-a-real-secret-123"


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


@pytest.fixture
def engine(monkeypatch):
    with FakeEngine() as e:
        monkeypatch.setenv("DENEB_ENGINE", e.url)
        yield e


# ── S2 ────────────────────────────────────────────────────────────────────────────
def test_logout_removes_the_pending_request_and_a_later_approval_saves_nothing(engine):
    _run(ARGS)
    config.save(token="t")
    assert os.path.exists(config.REQUEST_FILE)
    rc, out = _run(["logout"])
    assert rc == 0 and not os.path.exists(config.REQUEST_FILE)
    engine.status = "approved"
    cli._poll_pending()                                  # what every chat does first
    assert config.get_token() is None
    assert engine.poll_headers == []                     # nothing was even asked


# ── S6 ────────────────────────────────────────────────────────────────────────────
def test_poll_during_a_chat_to_a_black_holed_engine_adds_under_4_seconds_and_prints_nothing():
    with FakeBlackHole() as hole:
        config.save_request({"id": "abc123XYZ9", "secret": FAKE, "engine": hole.url})
        t0 = time.monotonic()
        rc, out = _run_poll()
        took = time.monotonic() - t0
    assert took < 4, took
    assert out == ""                                       # no network-failure line in a chat
    assert os.path.exists(config.REQUEST_FILE)             # still pending, try again next time


def _run_poll():
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._poll_pending()
    return 0, buf.getvalue()


def test_check_still_reports_a_network_failure(monkeypatch):
    monkeypatch.setenv("DENEB_ENGINE", "http://127.0.0.1:1")
    config.save_request({"id": "abc123XYZ9", "secret": FAKE, "engine": "http://127.0.0.1:1"})
    rc, out = _run(["auth", "--check"])
    assert rc != 0 and "can't reach" in out


def test_404_removes_the_request_and_says_so_once(engine):
    _run(ARGS)
    engine.poll_code = 404
    _, first = _run_poll()
    assert "no record of that request" in first and "it may have expired" in first
    assert "deneb auth --request" in first
    assert not os.path.exists(config.REQUEST_FILE)
    _, second = _run_poll()
    assert second == ""


# ── S7 ────────────────────────────────────────────────────────────────────────────
def test_second_request_while_pending_refuses_and_keeps_the_first(engine):
    _run(ARGS)
    first = json.load(open(config.REQUEST_FILE))
    rc, out = _run(["auth", "--request", "--name", "Bob", "--email", "bob@example.com",
                    "--box", "x", "--why", "y"])
    assert rc != 0
    assert "already pending" in out and "ada@example.com" in out and first["created"][:10] in out
    assert "deneb auth --check" in out and "--replace" in out
    assert len(engine.posts) == 1
    assert json.load(open(config.REQUEST_FILE)) == first


def test_replace_overwrites_with_a_warning(engine):
    _run(ARGS)
    first = json.load(open(config.REQUEST_FILE))
    engine.post_body = {"id": "second99ID", "status": "pending"}
    rc, out = _run(ARGS + ["--replace"])
    assert rc == 0 and len(engine.posts) == 2
    assert "first request can no longer be claimed" in out
    assert json.load(open(config.REQUEST_FILE))["id"] == "second99ID" != first["id"]


def test_request_json_always_stores_the_engine_url(engine):
    _run(ARGS)
    assert json.load(open(config.REQUEST_FILE))["engine"] == engine.url


def test_request_to_the_default_engine_url_is_stored_too(monkeypatch):
    with FakeEngine() as e:
        monkeypatch.setattr(config, "DEFAULT_ENGINE", e.url)
        monkeypatch.delenv("DENEB_ENGINE", raising=False)
        config.save(engine="http://127.0.0.1:2")           # a different saved custom engine
        _run(ARGS + ["--engine", e.url])
        assert json.load(open(config.REQUEST_FILE))["engine"] == e.url
        engine_seen = len(e.poll_headers)
        _run(["auth", "--check"])
        assert len(e.poll_headers) == engine_seen + 1       # polled the stored URL, not the saved one


def test_check_with_a_different_engine_refuses_in_one_line(engine):
    _run(ARGS)
    rc, out = _run(["auth", "--check", "--engine", "http://127.0.0.1:2"])
    assert rc != 0 and len(out.strip().splitlines()) == 1
    assert engine.url in out and engine.poll_headers == []


# ── S10 ───────────────────────────────────────────────────────────────────────────
def test_approval_with_a_saved_local_brain_says_chats_stay_local(engine):
    _run(ARGS)
    config.save_brain(mode="local")
    engine.status = "approved"
    state, msg = tokenreq.check_once()
    assert state == "approved" and config.get_token()
    assert "still use your local model" in msg and "deneb brain altronis" in msg
    assert "any deneb chat command can use it now" not in msg


def test_approval_with_no_saved_brain_keeps_the_plain_message(engine):
    _run(ARGS)
    engine.status = "approved"
    state, msg = tokenreq.check_once()
    assert "any deneb chat command can use it now" in msg


# ── N12 ───────────────────────────────────────────────────────────────────────────
def test_request_prints_the_actual_fields_before_sending(engine, monkeypatch):
    seen = {}
    real = tokenreq.submit

    def spy(*a, **k):
        seen["before_send"] = buf.getvalue()
        return real(*a, **k)

    monkeypatch.setattr(tokenreq, "submit", spy)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(ARGS)
    shown = seen["before_send"]
    for v in ("Ada Lovelace", "ada@example.com", "Strix Halo 128GB, Ollama", "slow model", engine.url):
        assert v in shown, v


# ── N8 ────────────────────────────────────────────────────────────────────────────
def test_preflight_does_not_claim_nothing_goes_to_altronis_while_a_request_is_pending(engine, monkeypatch):
    _run(ARGS)
    with FakeOpenAI(["x"]) as f:
        monkeypatch.setattr(brain, "discover_local", lambda: f.base)
        b = brain.resolve()
        lines = brain.preflight(b, pending=True)
    assert lines and "nothing is sent to Altronis" not in lines[0]
    assert "pending token request" in lines[0]


# ── S8 ────────────────────────────────────────────────────────────────────────────
def test_flags_beat_the_saved_brain_settings():
    config.save_brain(mode="url", url="http://127.0.0.1:9/v1", model="saved-model", key_env="SAVED_KEY")
    b = brain.resolve(None, "flag-model", "FLAG_KEY")
    assert (b.model, b.key_env) == ("flag-model", "FLAG_KEY")
    b = brain.resolve()
    assert (b.model, b.key_env) == ("saved-model", "SAVED_KEY")


def test_flags_beat_the_saved_settings_for_a_saved_local_brain(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", lambda: "http://localhost:8001/v1")
    config.save_brain(mode="local", model="saved-model", key_env="SAVED_KEY")
    b = brain.resolve(None, "flag-model", "FLAG_KEY")
    assert (b.model, b.key_env) == ("flag-model", "FLAG_KEY")


# ── S9 ────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def local_down(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", lambda: None)
    monkeypatch.setattr(check, "serving_port", lambda: 8001)


def test_saved_local_brain_down_with_a_token_points_at_the_brain_command(local_down):
    config.save(token="t")
    config.save_brain(mode="local")
    rc, out = _run(["why is it slow"])
    assert rc != 0
    assert "not answering at http://localhost:8001/v1" in out
    assert "deneb brain altronis" in out
    assert "deneb auth --request" not in out
    assert "Your box isn't serving one" not in out


def test_saved_local_brain_down_without_a_token_points_at_the_request(local_down):
    config.save_brain(mode="local")
    rc, out = _run(["why is it slow"])
    assert rc != 0 and "not answering at http://localhost:8001/v1" in out
    assert "deneb auth --request" in out and "deneb brain altronis" not in out


# ── S11 ───────────────────────────────────────────────────────────────────────────
def _keyed(key):
    """A model server that answers 401 on /v1/models unless the right Bearer key is sent."""
    f = FakeOpenAI(["x"])
    from fakes import _Quiet, _serve
    import threading

    class H(_Quiet):
        def do_GET(self):
            f.requests.append((self.path, dict(self.headers), None))
            if self.headers.get("Authorization") != f"Bearer {key}":
                return self._send(401, {"error": "key needed"})
            self._send(200, {"data": [{"id": "m1"}]})

    f.httpd.server_close()
    f.httpd = _serve("127.0.0.1", H)
    f.port = f.httpd.server_address[1]
    f.thread = threading.Thread(target=f.httpd.serve_forever, daemon=True)
    return f


def test_discovery_sends_the_configured_key_and_finds_a_protected_server(monkeypatch):
    monkeypatch.setenv("MY_KEY_5", FAKE)
    f = _keyed(FAKE)
    with f:
        monkeypatch.setattr(check, "serving_port", lambda: f.port)
        monkeypatch.setattr(brain, "discover_local", brain._discover_local)
        b = brain.resolve("local", None, "MY_KEY_5")
    assert b.base_url == f"http://localhost:{f.port}/v1"
    assert f.requests[0][1].get("Authorization") == f"Bearer {FAKE}"


def test_discovery_401_says_the_server_needs_a_key(monkeypatch):
    f = _keyed(FAKE)
    with f:
        monkeypatch.setattr(check, "serving_port", lambda: f.port)
        monkeypatch.setattr(brain, "discover_local", brain._discover_local)
        with pytest.raises(DenebError) as e:
            brain.resolve("local")
    assert "needs an API key" in str(e.value) and "--key-env NAME" in str(e.value)


def test_discovery_failures_name_the_reason(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", brain._discover_local)
    monkeypatch.setattr(check, "serving_port", lambda: 1)
    with pytest.raises(DenebError) as e:
        brain.resolve("local")
    assert "refused" in str(e.value)
    with FakeOpenAI(["x"]) as f:                                  # answers, but not /v1/models
        monkeypatch.setattr(check, "serving_port", lambda: f.port)
        f.httpd.RequestHandlerClass.do_GET = lambda self: self._send(404, {"error": "no"})
        with pytest.raises(DenebError) as e2:
            brain.resolve("local")
    assert "OpenAI-style" in str(e2.value)


def test_discovery_timeout_is_named(monkeypatch):
    with FakeBlackHole() as hole:
        monkeypatch.setattr(brain, "discover_local", brain._discover_local)
        monkeypatch.setattr(check, "serving_port", lambda: int(hole.url.rsplit(":", 1)[1]))
        monkeypatch.setattr(brain, "_DISCOVERY_TIMEOUT", 1)
        with pytest.raises(DenebError) as e:
            brain.resolve("local")
    assert "did not answer" in str(e.value)


# ── N7 ────────────────────────────────────────────────────────────────────────────
def test_remote_notice_prints_before_any_request_carrying_the_key(monkeypatch):
    events = []
    monkeypatch.setattr(ui, "info", lambda s: events.append(("say", s)))
    monkeypatch.setattr(brain, "_model", lambda b: events.append(("request", "")) or "m1")
    with FakeOpenAI(["x"], host="127.0.0.2") as f:
        b = brain.Brain(kind="url", base_url=f.base, key_env="MY_KEY_6")
        monkeypatch.setenv("MY_KEY_6", FAKE)
        monkeypatch.setattr(cli.brain, "resolve", lambda *a, **k: b)
        cli._prepare("whatever")
    kinds = [k for k, _ in events]
    notice = next(i for i, (k, s) in enumerate(events) if k == "say" and "go to 127.0.0.2" in s)
    assert notice < kinds.index("request")
