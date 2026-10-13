"""Bring your own model: brain selection, the OpenAI-compatible agent step, and the safety
boundary staying where it is (deneb/tools.py + the confirm gate)."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from deneb import brain, check, client, config, loop
from fakes import FakeOpenAI
import private_names


def _json(**kw):
    return json.dumps(kw)


@pytest.fixture(autouse=True)
def _canned_scan(monkeypatch):
    monkeypatch.setattr(check, "scan", lambda: [
        ("endpoint", "Model endpoint", True, "serving on :8001", ""),
        ("gpu", "Runs on the GPU", False, "CPU only", "install the GPU runtime"),
    ])


# ── precedence ────────────────────────────────────────────────────────────────────
def test_no_brain_at_all_resolves_to_none():
    assert brain.resolve() is None


def test_discovered_local_server_is_used_last(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", lambda: "http://localhost:8001/v1")
    b = brain.resolve()
    assert b.kind == "local" and b.base_url == "http://localhost:8001/v1"
    assert "deneb check" in b.reason


def test_saved_token_beats_a_discovered_server(monkeypatch):
    monkeypatch.setattr(brain, "discover_local", lambda: "http://localhost:8001/v1")
    config.save(token="t0k")
    assert brain.resolve().kind == "altronis"


def test_saved_config_beats_token(monkeypatch):
    config.save(token="t0k")
    config.save_brain(mode="url", url="http://127.0.0.1:9/v1", model="m", key_env=None)
    b = brain.resolve()
    assert b.kind == "url" and b.base_url == "http://127.0.0.1:9/v1" and b.model == "m"
    assert "deneb brain" in b.reason


def test_flag_beats_saved_config(monkeypatch):
    config.save_brain(mode="altronis")
    monkeypatch.setattr(brain, "discover_local", lambda: "http://localhost:8001/v1")
    assert brain.resolve(flag="local").kind == "local"
    assert brain.resolve(flag="http://127.0.0.1:5/v1").base_url == "http://127.0.0.1:5/v1"
    assert brain.resolve(flag="altronis").kind == "altronis"


def test_local_saved_but_nothing_serving_raises_with_a_reason():
    config.save_brain(mode="local")
    with pytest.raises(brain.BrainError) as e:
        brain.resolve()
    assert "saved brain is local" in str(e.value)


def test_bad_brain_flag_is_an_error():
    with pytest.raises(brain.BrainError):
        brain.resolve(flag="banana")


def test_discover_local_uses_the_same_port_finder_as_deneb_check(monkeypatch):
    with FakeOpenAI(["x"]) as f:
        monkeypatch.setattr(check, "serving_port", lambda: f.port)
        assert brain._discover_local() == f"http://localhost:{f.port}/v1"


def test_discover_local_none_when_nothing_answers(monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(check, "serving_port", lambda: port)
    assert brain._discover_local() is None


# ── config: the key is never stored ───────────────────────────────────────────────
def test_save_brain_stores_the_env_var_name_only(monkeypatch):
    monkeypatch.setenv("MY_BRAIN_KEY", "sk-super-secret-value")
    config.save_brain(mode="url", url="http://127.0.0.1:9/v1", model="m", key_env="MY_BRAIN_KEY")
    on_disk = open(config.CONFIG_FILE).read()
    assert "MY_BRAIN_KEY" in on_disk and "sk-super-secret-value" not in on_disk
    assert oct(os.stat(config.CONFIG_FILE).st_mode & 0o777) == "0o600"


def test_key_is_sent_as_bearer_but_never_written_to_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_BRAIN_KEY", "sk-super-secret-value")
    with FakeOpenAI([_json(type="final", answer="fine")]) as f:
        config.save_brain(mode="url", url=f.base, model=None, key_env="MY_BRAIN_KEY")
        b = brain.resolve()
        res = brain.agent_step(b, [{"role": "user", "content": "hi"}])
        assert res["type"] == "final"
        auth = [h for p, h, body in f.chats()][0].get("Authorization")
    assert auth == "Bearer sk-super-secret-value"
    for root, _d, files in os.walk(tmp_path):
        for name in files:
            assert "sk-super-secret-value" not in open(os.path.join(root, name), errors="ignore").read()


def test_missing_key_env_is_a_plain_error(monkeypatch):
    monkeypatch.delenv("NOT_SET_KEY", raising=False)
    b = brain.Brain(kind="url", base_url="http://127.0.0.1:9/v1", model="m", key_env="NOT_SET_KEY")
    with pytest.raises(client.DenebError) as e:
        brain.agent_step(b, [{"role": "user", "content": "hi"}])
    assert "NOT_SET_KEY" in str(e.value)


# ── model choice and warnings ─────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "Llama-3.2-1B", "smollm-135m",
                                  "SmolLM2-360M", "qwen2.5:1.5b", "phi-3b-q4", "gemma-2-3B-it"])
def test_tiny_model_names(name):
    assert brain.is_tiny_model(name)


@pytest.mark.parametrize("name", ["qwen3.8-27b", "llama3.1:8b", "mistral-7b", "gpt-oss-120b",
                                  "qwen3-30b-a3b", "llama-13b", "phi-4", "model-13b-v3", ""])
def test_not_tiny_model_names(name):
    assert not brain.is_tiny_model(name)


def test_first_model_from_v1_models_is_used_and_model_flag_overrides():
    with FakeOpenAI([_json(type="final", answer="a")], models=("first-model", "second")) as f:
        brain.agent_step(brain.Brain(kind="url", base_url=f.base), [{"role": "user", "content": "q"}])
        assert f.chats()[0][2]["model"] == "first-model"
    with FakeOpenAI([_json(type="final", answer="a")], models=("first-model", "second")) as f:
        brain.agent_step(brain.Brain(kind="url", base_url=f.base, model="second"),
                         [{"role": "user", "content": "q"}])
        assert f.chats()[0][2]["model"] == "second"


def test_is_local_url():
    for u in ("http://localhost:8001/v1", "http://127.0.0.1:1/v1", "http://LOCALHOST/v1"):
        assert brain.is_local_url(u)
    for u in ("https://api.example.com/v1", "http://192.168.1.5:8000/v1", "http://127.0.0.2/v1"):
        assert not brain.is_local_url(u)


def test_privacy_notice_names_the_host_for_a_remote_brain_only():
    remote = brain.Brain(kind="url", base_url="https://models.example.com/v1")
    note = brain.privacy_notice(remote)
    assert "your question and the log lines Deneb reads go to models.example.com" in note
    assert brain.privacy_notice(brain.Brain(kind="local", base_url="http://localhost:8001/v1")) == ""
    assert brain.privacy_notice(brain.Brain(kind="altronis")) == ""


# ── the agent step ────────────────────────────────────────────────────────────────
def test_read_action_then_final_through_the_existing_loop(monkeypatch):
    ran = []
    from deneb import tools
    real = tools.execute
    monkeypatch.setattr(tools, "execute", lambda a, cmd="", path="": (ran.append((a, cmd, path)),
                        {"ok": True, "output": "llama-server: active (running)"})[1])
    replies = [_json(type="action", action="run", cmd="systemctl --user status llama", thought="check"),
               _json(type="final", answer="The service is running. See runbook: strix-guard.")]
    events = []
    with FakeOpenAI(replies) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        res = loop.run("is my model service running?", on_event=lambda k, d: events.append(k))
        chats = f.chats()
    assert ran == [("run", "systemctl --user status llama", "")]
    assert res["type"] == "final" and "running" in res["answer"]
    assert "action" in events and "observation" in events
    # the second call carried the observation back
    second = json.dumps(chats[1][2]["messages"])
    assert "llama-server: active (running)" in second
    assert real  # keep reference


def test_first_call_carries_system_prompt_scan_rows_and_retrieved_docs():
    with FakeOpenAI([_json(type="final", answer="ok")]) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        loop.run("vulkan pinned build of llama.cpp is failing")
        msgs = f.chats()[0][2]["messages"]
    assert msgs[0]["role"] == "system"
    sys_text = msgs[0]["content"]
    assert "read-only" in sys_text.lower() and "I'm not sure" in sys_text
    assert "runbook" not in sys_text.lower()
    private_names.assert_clean(sys_text, "system prompt")
    joined = json.dumps(msgs)
    assert "Runs on the GPU" in joined and "CPU only" in joined           # the deneb check scan
    assert "vulkan-pinned-build" in joined                                # retrieved doc label


def test_scan_runs_once_per_question_not_per_step(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "scan", lambda: calls.append(1) or [])
    replies = [_json(type="action", action="list_dir", path="/tmp"), _json(type="final", answer="x")]
    with FakeOpenAI(replies) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        loop.run("what is in /tmp")
    assert len(calls) == 1


def test_malformed_model_output_gives_a_plain_final_not_raw_json():
    with FakeOpenAI(['{"type":"action","action":"run","cmd":"ollama ps"']) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        res = loop.run("why slow")
    assert res["type"] == "final" and "{" not in res["answer"]


def test_think_tags_are_stripped():
    with FakeOpenAI(["<think>hmm {\"a\":1}</think>" + _json(type="final", answer="clean")]) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        assert loop.run("q")["answer"].startswith("clean")


def test_local_final_points_to_a_person_without_escalating(monkeypatch):
    sent = []
    monkeypatch.setattr(client, "escalate", lambda *a, **k: sent.append(a) or True)
    with FakeOpenAI([_json(type="final", answer="not sure", escalated=True)]) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        res = loop.run("q")
    assert sent == [] and not res.get("escalated")
    assert "deneb auth --request" in res.get("hint", "")


def test_server_down_is_a_plain_error():
    b = brain.Brain(kind="url", base_url="http://127.0.0.1:1/v1", model="m")
    with pytest.raises(client.DenebError) as e:
        brain.agent_step(b, [{"role": "user", "content": "q"}])
    assert "can't reach" in str(e.value)


# ── the client-side security boundary is unchanged ────────────────────────────────
def test_local_brain_asking_for_rm_rf_is_refused_exactly_as_with_the_engine(monkeypatch):
    replies = [_json(type="action", action="run", cmd="rm -rf /"),
               _json(type="final", answer="could not")]
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or pytest.fail("ran a command"))
    with FakeOpenAI(replies) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        loop.run("clean up")
        obs = json.dumps(f.chats()[1][2]["messages"])
    assert "[refused]" in obs
    assert calls == []                       # nothing was ever handed to subprocess


def test_local_brain_asking_for_sudo_fix_is_refused_even_when_the_user_says_yes():
    replies = [_json(type="action", action="fix", cmd="sudo systemctl restart llama", why="x"),
               _json(type="final", answer="could not")]
    shown = []
    with FakeOpenAI(replies) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        loop.run("fix it", on_event=lambda k, d: shown.append(k), confirm=lambda r: True)
        obs = json.dumps(f.chats()[1][2]["messages"])
    assert "propose" in shown and "[refused]" in obs


def test_apply_actions_are_declined_without_a_confirm_gate():
    replies = [_json(type="action", action="write_file", path="/tmp/x-deneb-test", content="y"),
               _json(type="final", answer="skipped")]
    with FakeOpenAI(replies) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        loop.run("write it")
        obs = json.dumps(f.chats()[1][2]["messages"])
    assert "skipped by user" in obs and not os.path.exists("/tmp/x-deneb-test")


# ── summarize and vision ──────────────────────────────────────────────────────────
def test_summarize_uses_the_same_brain():
    with FakeOpenAI(["a short brief"]) as f:
        client.set_brain(brain.Brain(kind="local", base_url=f.base))
        assert client.summarize("user: a\nassistant: b") == "a short brief"
        assert f.chats()


def test_image_needs_a_vision_model_on_a_local_brain(tmp_path):
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    with FakeOpenAI(["unused"], vision=False, models=("llama3.1:8b",)) as f:
        b = brain.Brain(kind="local", base_url=f.base)
        assert not brain.supports_vision(b)
        client.set_brain(b)
        with pytest.raises(client.DenebError) as e:
            client.ask("what is this", image=str(img))
    assert "--image needs a vision model or the Altronis engine" in str(e.value)


def test_image_goes_to_a_vision_model(tmp_path):
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    with FakeOpenAI(["it is an error screen"], vision=True, models=("qwen2.5-vl-7b",)) as f:
        b = brain.Brain(kind="local", base_url=f.base)
        assert brain.supports_vision(b)
        client.set_brain(b)
        res = client.ask("what is this", image=str(img))
        body = f.chats()[0][2]
    assert res["answer"] == "it is an error screen"
    assert "image_url" in json.dumps(body)


def test_system_prompt_describes_the_real_run_limits():
    p = brain.SYSTEM_PROMPT
    assert "curl is a plain GET to localhost only" in p and "no -w, -o, -d, -L" in p
    assert "refuse secret files" in p and "printenv needs explicit" in p
    assert "one version flag only" in p
    assert "curl to localhost," not in p


def test_image_cap_is_8_million_bytes_so_the_base64_body_fits_the_engine(tmp_path):
    """base64 inflates by 4/3: 8,000,000 bytes is ~10.7 MB, under the engine's 12,000,000-byte
    body cap. The old 10,000,000-byte cap made ~13.3 MB and got a 413."""
    from deneb import client
    from deneb.client import DenebError
    ok = tmp_path / "ok.png"
    with open(ok, "wb") as f:
        f.truncate(8_000_000)                     # sparse: no real disk used
    assert client._image_data_url(str(ok)).startswith("data:image/png;base64,")
    big = tmp_path / "big.png"
    with open(big, "wb") as f:
        f.truncate(8_000_001)
    with pytest.raises(DenebError) as e:
        client._image_data_url(str(big))
    assert str(e.value) == "image too large (max 8MB)"
