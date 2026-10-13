"""Part 3 review, group B: a key value can never be saved or printed (B4, N4), a URL's userinfo
is never shown (N3), and one HTTP helper refuses redirects to another origin (S5, M1).
Only fake servers on 127.0.0.1 are used; the "key" is always a fake value."""
from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout

import pytest

from deneb import brain, cli, client, config, tokenreq
from deneb.client import DenebError
from fakes import FakeOpenAI, FakeRecorder, FakeRedirector

FAKE = "fake-not-a-real-secret-123"
REFUSAL = ("--key-env takes the NAME of an environment variable (like MY_KEY), not the key "
           "itself. nothing was saved.")


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue() + err.getvalue()


# ── B4 ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [FAKE, "$MY_KEY", "1ABC", "A B", "KEY=1", "sk-abc", ""])
def test_brain_command_refuses_a_key_value_as_the_variable_name(bad):
    config.save(token="t")
    before = open(config.CONFIG_FILE).read()
    rc, out = _run(["brain", "http://127.0.0.1:9/v1", "--key-env", bad])
    assert rc != 0
    assert REFUSAL in out
    assert open(config.CONFIG_FILE).read() == before          # nothing saved
    if bad:
        assert bad not in out.replace(REFUSAL, "")


def test_brain_command_refusal_for_the_equals_form_and_for_local():
    for argv in (["brain", "local", f"--key-env={FAKE}"], ["brain", "altronis", "--key-env", FAKE]):
        rc, out = _run(argv)
        assert rc != 0 and FAKE not in out.replace(REFUSAL, "") and REFUSAL in out
        assert not os.path.exists(config.CONFIG_FILE)


def test_a_good_name_is_still_saved_and_named():
    rc, out = _run(["brain", "http://127.0.0.1:9/v1", "--key-env", "MY_KEY_2"])
    assert rc == 0 and "$MY_KEY_2" in out
    assert json.load(open(config.CONFIG_FILE))["brain"]["key_env"] == "MY_KEY_2"


def test_chat_with_a_key_value_as_key_env_is_refused_and_sends_nothing(monkeypatch):
    with FakeRecorder() as srv:
        rc, out = _run(["--brain", srv.url + "/v1", "--key-env", FAKE, "hello"])
        assert srv.seen == []
    assert rc != 0 and REFUSAL in out
    assert FAKE not in out.replace(REFUSAL, "")


def test_key_error_never_prints_a_saved_value_that_is_not_a_name(monkeypatch):
    # a config written by an older version, or by hand, holds a key value in key_env
    os.makedirs(config.CONFIG_DIR, exist_ok=True)
    with open(config.CONFIG_FILE, "w") as f:
        json.dump({"brain": {"mode": "url", "url": "http://127.0.0.1:9/v1", "key_env": FAKE}}, f)
    b = brain.resolve()
    with pytest.raises(DenebError) as e:
        brain.agent_step(b, [{"role": "user", "content": "q"}])
    assert FAKE not in str(e.value)
    assert "deneb brain" in str(e.value)


def test_key_error_for_an_unset_name_still_names_the_variable(monkeypatch):
    monkeypatch.delenv("NOT_SET_KEY", raising=False)
    b = brain.Brain(kind="url", base_url="http://127.0.0.1:9/v1", key_env="NOT_SET_KEY")
    with pytest.raises(DenebError) as e:
        brain.agent_step(b, [{"role": "user", "content": "q"}])
    assert "NOT_SET_KEY" in str(e.value)


# ── N4 ────────────────────────────────────────────────────────────────────────────
def test_server_error_text_is_scrubbed_of_the_key_and_bearer_tokens(monkeypatch):
    monkeypatch.setenv("MY_KEY_3", FAKE)
    leak = f"bad header Authorization: Bearer {FAKE} and Bearer abc.def-123 and key {FAKE}"
    with FakeRecorder({"error": {"message": leak}}, code=500) as srv:
        b = brain.Brain(kind="url", base_url=srv.url + "/v1", key_env="MY_KEY_3", model="m")
        with pytest.raises(DenebError) as e:
            brain.agent_step(b, [{"role": "user", "content": "q"}])
    msg = str(e.value)
    assert FAKE not in msg and "abc.def-123" not in msg
    assert "500" in msg


def test_server_error_text_is_capped_at_200_chars(monkeypatch):
    with FakeRecorder({"error": "x" * 900}, code=500) as srv:
        b = brain.Brain(kind="url", base_url=srv.url + "/v1", model="m")
        with pytest.raises(DenebError) as e:
            brain.agent_step(b, [{"role": "user", "content": "q"}])
    assert str(e.value).count("x") == 200


# ── N3 ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url,shown", [
    ("http://localhost@evil.example/v1", "evil.example"),
    ("https://alice:fake-not-a-real-secret-123@host.example:8443/v1", "host.example:8443"),
    ("http://127.0.0.2:8001/v1", "127.0.0.2:8001"),
    ("http://[::1]:8080/v1", "[::1]:8080"),
])
def test_host_never_shows_userinfo(url, shown):
    assert brain._host(url) == shown


def test_privacy_notice_names_the_real_host_not_the_userinfo():
    b = brain.Brain(kind="url", base_url="http://localhost@evil.example/v1")
    assert brain.privacy_notice(b).endswith("go to evil.example")


# ── S5 + M1: one helper, no cross-origin redirects ─────────────────────────────────────
def _brain_for(url, key_env=None):
    return brain.Brain(kind="url", base_url=url + "/v1", model="m", key_env=key_env)


def test_brain_chat_does_not_follow_a_redirect_to_another_port_and_sends_no_key(monkeypatch):
    monkeypatch.setenv("MY_KEY_4", FAKE)
    with FakeRecorder() as target, FakeRedirector(target.url) as hop:
        with pytest.raises(DenebError) as e:
            brain.agent_step(_brain_for(hop.url, "MY_KEY_4"), [{"role": "user", "content": "q"}])
        assert hop.seen and hop.seen[0][2].get("Authorization") == f"Bearer {FAKE}"
        assert target.seen == []                                # the second server got nothing
    msg = str(e.value)
    assert "redirected to another address" in msg and "does not send your key there" in msg
    assert "final URL" in msg and FAKE not in msg


def test_brain_model_listing_get_is_also_refused(monkeypatch):
    monkeypatch.setenv("MY_KEY_4", FAKE)
    with FakeRecorder() as target, FakeRedirector(target.url) as hop:
        b = brain.Brain(kind="url", base_url=hop.url + "/v1", key_env="MY_KEY_4")   # no model: GET /models
        with pytest.raises(DenebError):
            brain._model(b)
        assert target.seen == []


def test_https_to_http_and_scheme_change_is_refused():
    with FakeRecorder() as target, FakeRedirector(f"https://127.0.0.1:{target.port}") as hop:
        with pytest.raises(DenebError) as e:
            brain.agent_step(_brain_for(hop.url), [{"role": "user", "content": "q"}])
        assert "redirected to another address" in str(e.value)
        assert target.seen == []


def test_a_redirect_to_another_host_name_is_refused():
    with FakeRecorder() as target, FakeRedirector(f"http://localhost:{target.port}") as hop:
        with pytest.raises(DenebError):
            brain.agent_step(_brain_for(hop.url), [{"role": "user", "content": "q"}])
        assert target.seen == []


def test_same_origin_redirect_is_followed():
    with FakeRedirector(same_origin=True) as hop:
        b = brain.Brain(kind="url", base_url=hop.url + "/v1")
        assert brain._model(b) == "m1"
        assert [p for _m, p, _h in hop.seen] == ["/v1/models", "/final/v1/models"]


def test_engine_calls_refuse_a_cross_origin_redirect_and_send_no_token():
    config.save(token=FAKE)
    with FakeRecorder() as target, FakeRedirector(target.url) as hop:
        os.environ["DENEB_ENGINE"] = hop.url
        try:
            with pytest.raises(DenebError) as e:
                client._post("/agent-step", {"history": []}, 5)
        finally:
            del os.environ["DENEB_ENGINE"]
        assert target.seen == []
    assert "redirected to another address" in str(e.value) and FAKE not in str(e.value)


def test_token_request_refuses_a_cross_origin_redirect_and_sends_nothing():
    with FakeRecorder() as target, FakeRedirector(target.url, code=307) as hop:
        with pytest.raises(tokenreq.RequestError) as e:
            tokenreq.submit("n", "a@b.example", "box", "why", engine=hop.url)
        assert target.seen == []
    assert "redirected to another address" in str(e.value)


def test_token_poll_refuses_a_cross_origin_redirect_and_never_forwards_the_secret():
    config.save_request({"id": "abc123XYZ9", "secret": FAKE, "engine": None})
    with FakeRecorder() as target, FakeRedirector(target.url) as hop:
        config.save_request({"id": "abc123XYZ9", "secret": FAKE, "engine": hop.url})
        state, msg = tokenreq.check_once()
        assert target.seen == []
    assert state == "error" and "redirected to another address" in msg and FAKE not in msg


def test_the_three_modules_share_one_user_agent_and_helper():
    from deneb import net
    assert client._UA == net.UA == brain._UA == tokenreq._UA
    import inspect
    for mod in (client, brain, tokenreq):
        assert "urlopen" not in inspect.getsource(mod), mod.__name__
