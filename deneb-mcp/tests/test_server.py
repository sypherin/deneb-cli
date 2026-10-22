"""The Deneb MCP tools are thin wrappers over deneb.api; the shaping (deneb_mcp.shape) is pure.
These tests pin the JSON shape the tools return, that the server registers exactly the keyless
verbs, and that the error branches answer instead of raising.

Keyless by default, so these run offline. Direct tool calls work because FastMCP's @tool
decorator returns the underlying function.
"""
import asyncio

from deneb import api, check, hardware
from deneb_mcp import server, shape


def test_recommend_payload_shape():
    out = shape.recommend_payload(api.recommend("coding"))
    assert out["use_case"] == "coding"
    assert isinstance(out["box"], dict) and "backend" in out["box"]
    assert "bandwidth_gbps" in out["box"] and "free_disk_gb" in out["box"]
    assert isinstance(out["picks"], list) and out["picks"]
    top = out["picks"][0]
    for k in ("rank", "model", "quant", "fits", "tight", "speed_tier",
              "est_decode_tps", "download_gib", "download_fits_disk", "why"):
        assert k in top, k
    assert top["rank"] == 1
    assert out["next"] and top["model"] in out["next"]
    assert all("measured" not in p for p in out["picks"])   # keyless default


def test_setup_payload_shape():
    name = shape.recommend_payload(api.recommend("coding"))["picks"][0]["model"]
    out = shape.setup_payload(api.setup(name), name)
    assert out["model"] == name
    assert isinstance(out["box"], dict) and "serving" in out["box"]
    assert isinstance(out["steps"], list) and out["steps"]
    s0 = out["steps"][0]
    for k in ("n", "title", "command", "what", "warnings", "kind"):
        assert k in s0, k
    assert s0["n"] == 1
    assert isinstance(s0["warnings"], list)
    assert "measured" not in out                              # keyless default


def test_recommend_payload_measured_branch(monkeypatch):
    from deneb import tokenmark
    row = {"decode_tps": 42.5, "quant": "Q4_K_M", "hardwareLabel": "Strix Halo", "author": "tester"}
    monkeypatch.setattr(tokenmark, "measured_for_model", lambda *a, **k: [row])
    out = shape.recommend_payload(api.recommend("coding", measured=True))
    with_m = [p for p in out["picks"] if "measured" in p]
    assert with_m, "measured rows should attach to at least one pick"
    m = with_m[0]["measured"]
    assert m["tok_per_s"] == 42.5
    assert m["quant"] == "Q4_K_M" and m["hardware"] == "Strix Halo" and m["measured_by"] == "tester"


def test_setup_payload_measured_branch():
    from deneb.api import SetupResult
    from deneb.measured import Selection
    name = shape.recommend_payload(api.recommend("coding"))["picks"][0]["model"]
    base = api.setup(name)
    sel = Selection(choice=object(),  # truthy choice -> reproduces True
                    lines=["llama.cpp vulkan, Q4_K_M, ctx 131072", "https://tokenmark.app/x"],
                    expected={"decode_tps": 40})
    data = SetupResult(base.model, base.profile, base.serving, base.steps, sel)
    out = shape.setup_payload(data, name)
    assert "measured" in out
    assert out["measured"]["reproduces"] is True
    assert out["measured"]["lines"] and all(isinstance(x, str) for x in out["measured"]["lines"])


def test_box_payload_shape():
    out = shape.box_payload(hardware.profile_hardware(), check.serving_now())
    assert "backend" in out and "memory_gb" in out and "serving" in out


def test_box_payload_no_server_still_reports_serving():
    # schema stability: with nothing serving, deneb_box must SAY so, not drop the field
    out = shape.box_payload(hardware.profile_hardware(), None)
    assert out["serving"] == "no model server running"
    assert "serving_detail" not in out


def test_box_payload_with_server_adds_detail():
    out = shape.box_payload(hardware.profile_hardware(), (8001, "llama.cpp", "qwen3.8-27b"))
    assert out["serving"] and out["serving"] != "no model server running"
    assert out["serving_detail"] == {"port": 8001, "kind": "llama.cpp", "model": "qwen3.8-27b"}


def test_server_registers_exactly_the_keyless_verbs():
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {"deneb_recommend", "deneb_setup", "deneb_box"}
    # each tool has a description (the agent reads it to choose)
    assert all((t.description or "").strip() for t in tools)


def test_recommend_tool_runs_and_matches_shape():
    out = server.deneb_recommend(use_case="general")
    assert out["use_case"] == "general" and out["picks"]


def test_recommend_tool_rejects_bad_use_case():
    out = server.deneb_recommend(use_case="nonsense")
    assert "error" in out and "valid" in out
    assert "picks" not in out


def test_setup_tool_unknown_model_suggests():
    out = server.deneb_setup(model="qwen3-coder-typo-xyz")
    assert "error" in out and "did_you_mean" in out


def test_setup_tool_empty_model():
    out = server.deneb_setup(model="  ")
    assert "error" in out


def test_box_tool_runs():
    out = server.deneb_box()
    assert "backend" in out
