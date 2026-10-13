"""The conversational mode from the command line: brain choice, messages, warnings."""
from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout

from deneb import brain, check, cli, client, config
from fakes import FakeOpenAI
import private_names


def _run(argv, monkeypatch=None, stdin=None):
    buf = io.StringIO()
    if stdin is not None:
        it = iter(stdin)

        def fake_input(prompt=""):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        monkeypatch.setattr("builtins.input", fake_input)
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def _final(text):
    return json.dumps({"type": "final", "answer": text})


def test_no_brain_message_points_at_check_and_the_token_request():
    rc, out = _run(["why is it slow"])
    assert rc != 0
    assert "Deneb needs a model to think with" in out
    assert "deneb check" in out and "deneb auth --request" in out
    assert "runbook" not in out.lower()
    private_names.assert_clean(out, "output")


def test_bare_deneb_with_no_brain_gives_the_same_message():
    rc, out = _run([])
    assert rc != 0 and "Deneb needs a model to think with" in out


def test_one_shot_on_a_discovered_local_server(monkeypatch):
    with FakeOpenAI([_final("It is on the CPU. Run: deneb check")]) as f:
        monkeypatch.setattr(brain, "discover_local", lambda: f.base)
        rc, out = _run(["why", "is", "my", "model", "slow"])
    assert rc == 0 and "It is on the CPU" in out
    assert "nothing is sent to Altronis" in out


def test_brain_flag_url_wins_and_remote_url_prints_the_privacy_notice_once(monkeypatch):
    with FakeOpenAI([_final("answer-one-zq"), _final("answer-two-zq")], host="127.0.0.2") as f:
        rc, out = _run(["--brain", f.base, "hello"])
        assert rc == 0
        assert f"your question and the log lines Deneb reads go to 127.0.0.2" in out
        assert out.index("go to 127.0.0.2") < out.index("answer-one-zq")    # notice BEFORE the answer
        # interactive: notice once, two questions
        rc, out = _run(["--brain", f.base], monkeypatch, stdin=["a", "b", "exit"])
    assert out.count("go to 127.0.0.2") == 1


def test_tiny_model_warns_once(monkeypatch):
    with FakeOpenAI([_final("x"), _final("y")], models=("qwen2.5-0.5b-instruct",)) as f:
        monkeypatch.setattr(brain, "discover_local", lambda: f.base)
        rc, out = _run([], monkeypatch, stdin=["q1", "q2", "exit"])
    assert out.count("answers will be weak") == 1


def test_localhost_brain_has_no_privacy_notice(monkeypatch):
    with FakeOpenAI([_final("fine")]) as f:
        monkeypatch.setattr(brain, "discover_local", lambda: f.base)
        rc, out = _run(["hello"])
    assert "go to" not in out


def test_deneb_brain_shows_current_brain_and_reason(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", lambda: "http://localhost:8001/v1")
    rc, out = _run(["brain"])
    assert rc == 0 and "local" in out and "http://localhost:8001/v1" in out and "deneb check" in out
    monkeypatch.setattr(brain, "discover_local", lambda: None)
    rc, out = _run(["brain"])
    assert "no brain" in out.lower() and "deneb auth --request" in out


def test_deneb_brain_saves_local_altronis_and_url():
    rc, out = _run(["brain", "http://127.0.0.1:9/v1", "--model", "m1", "--key-env", "SOME_KEY"])
    assert rc == 0
    d = json.load(open(config.CONFIG_FILE))["brain"]
    assert d == {"mode": "url", "url": "http://127.0.0.1:9/v1", "model": "m1", "key_env": "SOME_KEY"}
    assert "SOME_KEY" in out
    rc, out = _run(["brain"])
    assert "http://127.0.0.1:9/v1" in out and "deneb brain" in out
    _run(["brain", "local"])
    assert json.load(open(config.CONFIG_FILE))["brain"]["mode"] == "local"
    _run(["brain", "altronis"])
    assert json.load(open(config.CONFIG_FILE))["brain"]["mode"] == "altronis"


def test_deneb_brain_rejects_nonsense():
    rc, out = _run(["brain", "banana"])
    assert rc != 0 and "usage" in out.lower()


def test_logout_keeps_working_and_removes_the_token_and_a_pending_request():
    config.save(token="t")
    config.save_request({"id": "abc123XYZ9", "secret": "fake-not-a-real-secret-123"})
    _run(["logout"])
    assert config.get_token() is None
    assert config.load_request() is None and not os.path.exists(config.REQUEST_FILE)


def test_image_on_a_non_vision_local_brain_says_so(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG" + b"0" * 10)
    with FakeOpenAI(["x"], models=("llama3.1:8b",)) as f:
        monkeypatch.setattr(brain, "discover_local", lambda: f.base)
        rc, out = _run(["--image", str(img), "what is this"])
    assert rc != 0 and "--image needs a vision model or the Altronis engine" in out
    assert f.chats() == []


def test_altronis_brain_without_token_is_told_how_to_get_one():
    config.save_brain(mode="altronis")
    rc, out = _run(["hello"])
    assert rc != 0 and "deneb auth --request" in out


def test_saved_token_uses_the_engine_path(monkeypatch):
    config.save(token="t")
    got = {}
    monkeypatch.setattr(client, "agent_step", lambda h, timeout=180: got.setdefault("h", h) and
                        {"type": "final", "answer": "from engine", "escalated": False, "sources": []})
    rc, out = _run(["hello"])
    assert rc == 0 and "from engine" in out
