"""Test isolation: no test may read or write the real ~/.config/deneb, or find the real model
server on the machine running the suite. Every test gets a scratch config dir, no DENEB_*
environment, and "no local model server" unless the test sets one up itself."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _isolated_deneb(tmp_path, monkeypatch):
    from deneb import brain, check, client, config

    cfg = tmp_path / "deneb-config"
    monkeypatch.setattr(config, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(config, "CONFIG_FILE", str(cfg / "config.json"))
    monkeypatch.setattr(config, "REQUEST_FILE", str(cfg / "request.json"))
    for var in ("DENEB_TOKEN", "DENEB_ENGINE", "DENEB_PORT"):
        monkeypatch.delenv(var, raising=False)
    # the real box may be serving a model: tests must not find it
    monkeypatch.setattr(brain, "discover_local", lambda: None)
    monkeypatch.setattr(check, "_port_cache", None)
    # the real box may have Ollama installed: tests start from "not installed" and say otherwise
    monkeypatch.setattr(check, "_ollama_install", lambda: "", raising=False)
    monkeypatch.setattr(check, "_own_llama_unit_exists", lambda: False, raising=False)
    # nor its unit files or its logind linger flag
    monkeypatch.setattr(check, "_WRITTEN_UNIT_DIRS", (), raising=False)
    monkeypatch.setattr(check, "_linger", lambda: None, raising=False)
    # recommend and setup name what serves: nothing, unless the test sets one up
    monkeypatch.setattr(check, "serving_now", lambda: None, raising=False)
    client.set_brain(None)
    brain.reset()
    yield
    client.set_brain(None)
