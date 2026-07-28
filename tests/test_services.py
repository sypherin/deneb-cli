"""Tests for deneb.services — the multi-service stack advice.

These pin two classes of thing.

The mundane one: the prompts a real user actually types must resolve. "qwen vlm", "surya2",
"caddy" are the words in someone's head, not the internal keys, and every one of them
returned "unknown model" before this module existed.

The one that matters: the stack has a security boundary, and it is the only part of Deneb's
advice where being wrong is dangerous rather than merely unhelpful. Every model server here
is unauthenticated by construction - llama-server answers anyone who can reach it - so the
gateway ordering, the loopback binds and the tunnel target are load-bearing. A tunnel
pointed at :8001 instead of :8002 publishes an unauthenticated model to the internet and
looks like it is working.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deneb import services as sv  # noqa: E402


# ── the words people actually type ────────────────────────────────────────────
def test_real_user_phrasings_resolve():
    cases = {
        "gemma": "llm", "gemma 4": "llm", "llm": "llm", "chat": "llm",
        "qwen vlm": "vlm", "vlm": "vlm", "vision": "vlm", "qwen3-vl": "vlm",
        "surya": "surya", "ocr": "surya",
        "surya 2": "surya2", "surya2": "surya2",
        "gateway": "gateway", "caddy": "gateway", "traefik": "gateway", "endpoint": "gateway",
        "cloudflared": "tunnel", "tunnel": "tunnel",
    }
    for typed, expected in cases.items():
        svc = sv.resolve_service(typed)
        assert svc is not None, f"{typed!r} resolved to nothing"
        assert svc.key == expected, f"{typed!r} -> {svc.key}, expected {expected}"


def test_unknown_input_returns_none_rather_than_a_wrong_service():
    for junk in ("", "   ", "postgres", None):
        assert sv.resolve_service(junk) is None


# ── the security boundary ─────────────────────────────────────────────────────
def test_gateway_comes_before_the_tunnel():
    order = list(sv.STACK_ORDER)
    assert order.index("gateway") < order.index("tunnel"), (
        "standing the tunnel up before the gateway publishes an unauthenticated model")


def test_backends_come_before_the_gateway():
    order = list(sv.STACK_ORDER)
    for backend in ("llm", "vlm", "surya2"):
        assert order.index(backend) < order.index("gateway")


def test_tunnel_targets_the_gateway_not_a_model_port():
    tunnel = sv.SERVICES["tunnel"]
    commands = " ".join(s.command for s in tunnel.steps)
    assert "8002" in commands, "the tunnel must point at the gateway"
    for model_port in ("8001", "8080", "8093"):
        assert f":{model_port}" not in commands, (
            f"tunnel points at {model_port} - that exposes an unauthenticated model")


def test_gateway_step_states_the_401_contract():
    gw = sv.SERVICES["gateway"]
    blob = " ".join(f"{s.what} {s.expect} {' '.join(s.warnings)}" for s in gw.steps)
    assert "401" in blob, "the gateway's whole job is rejecting unauthenticated calls"
    assert "bearer" in blob.lower()


def test_gateway_tells_you_to_verify_from_another_machine():
    # Testing exposure from the box itself proves nothing: localhost always answers.
    gw = sv.SERVICES["gateway"]
    blob = " ".join(f"{s.what} {s.command} {' '.join(s.warnings)}" for s in gw.steps).lower()
    assert "another machine" in blob or "different machine" in blob


def test_vision_and_ocr_backends_bind_loopback_only():
    # Only the LLM binds 0.0.0.0, and only because the gateway fronts it - which its own
    # warning says. Everything else must stay on loopback.
    for key in ("vlm", "surya2"):
        commands = " ".join(s.command for s in sv.SERVICES[key].steps)
        assert "0.0.0.0" not in commands, f"{key} must not bind every interface"
        assert "127.0.0.1" in commands


def test_the_one_public_bind_is_flagged():
    llm = sv.SERVICES["llm"]
    exposed = [s for s in llm.steps if "0.0.0.0" in s.command]
    assert exposed, "the LLM serve step vanished"
    joined = " ".join(exposed[0].warnings).lower()
    assert "gateway" in joined, "binding 0.0.0.0 must say why it is acceptable"


# ── correctness traps specific to these models ────────────────────────────────
def test_vision_services_require_the_projector():
    # Without mmproj a vision model loads, answers text, and silently cannot see images.
    for key in ("vlm", "surya2"):
        blob = " ".join(f"{s.command} {s.what}" for s in sv.SERVICES[key].steps)
        assert "mmproj" in blob, f"{key} must fetch and pass the mmproj projector"


def test_ocr_runs_at_temperature_zero():
    # OCR is transcription. Any sampling temperature invents characters.
    commands = " ".join(s.command for s in sv.SERVICES["surya2"].steps)
    assert "--temp 0" in commands


def test_llm_keeps_its_speculative_draft_model():
    commands = " ".join(s.command for s in sv.SERVICES["llm"].steps)
    assert "--spec-type draft-mtp" in commands
    assert "-md " in commands, "the draft model is what makes speculative decoding work"


def test_surya2_build_keeps_the_patch_step():
    # Recovered from a working build's own GGUF metadata: it reports architecture "qwen35",
    # a "qwen3vl_merger" projector and the model name "_Patched_Ckpt". That name is the
    # fingerprint of a patched config. Without the patch the converter rejects the
    # checkpoint outright, so this step is the whole build - and it is the one no
    # documentation mentions.
    blob = " ".join(f"{s.title} {s.what}" for s in sv.SERVICES["surya2"].steps).lower()
    assert "patch" in blob
    assert "qwen35" in blob and "qwen3vl_merger" in blob


def test_surya2_is_not_quantised():
    # The deployed model is f16 (GGUF file_type 1). An earlier version of this advice told
    # people to quantise to Q4_K_M, which is not what runs - and on a 630M OCR model the
    # saving is trivial while the accuracy loss is silent.
    steps = sv.SERVICES["surya2"].steps
    commands = " ".join(s.command for s in steps)
    assert "llama-quantize" not in commands
    assert "f16" in commands.lower()
    blob = " ".join(f"{s.what} {s.expect}" for s in steps).lower()
    assert "unquantised" in blob or "f16" in blob


def test_surya2_conversion_writes_partial_then_moves():
    # A truncated GGUF still loads and then misbehaves, which is the worst failure shape.
    commands = " ".join(s.command for s in sv.SERVICES["surya2"].steps)
    assert ".partial" in commands and "mv " in commands


def test_surya_cutover_requires_a_comparison_first():
    # A newer OCR model is not automatically better on someone's own documents, and wrong
    # characters look exactly like right ones.
    blob = " ".join(f"{s.title} {s.what} {' '.join(s.warnings)}"
                    for s in sv.SERVICES["surya2"].steps).lower()
    assert "surya 1" in blob and ("compare" in blob or "comparison" in blob)


def test_surya1_is_described_as_a_container_not_a_model_file():
    # The point is that Surya 1 cannot be served by llama-server at all, which is why the
    # model catalog could never have represented it. Asserted on what it IS rather than
    # banning the word "gguf", since the text legitimately says "not a GGUF".
    surya = sv.SERVICES["surya"]
    blob = f"{surya.label} {surya.notes}".lower()
    assert "container" in blob
    assert "not a gguf" in blob or "not llama.cpp" in blob
    commands = " ".join(s.command for s in surya.steps)
    assert "podman" in commands or "docker" in commands
    assert "llama-server" not in commands


# ── purity ────────────────────────────────────────────────────────────────────
def test_module_is_pure():
    src = Path(sv.__file__).read_text(encoding="utf-8")
    for token in ("subprocess", "os.system", "os.popen", "Popen", " exec(", " eval("):
        assert token not in src, f"services must stay pure; found {token!r}"
    imports = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    assert not any("tools" in ln for ln in imports)


def test_no_em_dashes_in_generated_copy():
    for key, svc in sv.SERVICES.items():
        blob = f"{svc.label}{svc.role}{svc.notes}" + " ".join(
            f"{s.title}{s.what}{s.expect}{' '.join(s.warnings)}" for s in svc.steps)
        assert "—" not in blob and "–" not in blob, f"{key} contains an em/en dash"


# ── platform caveats ──────────────────────────────────────────────────────────
def test_dgx_is_warned_that_container_images_are_arch_specific():
    # GGUFs carry over to aarch64 unchanged; container images do not. An x86-only image
    # either refuses to run or drops into emulation and is unusably slow - and that is only
    # discovered at the machine.
    notes = " ".join(sv.platform_notes("dgx-spark")).lower()
    assert "aarch64" in notes
    assert "container" in notes
    assert "surya 2" in notes, "the GGUF OCR path is the sound one on this arch"


def test_dgx_notes_say_build_cuda_first():
    notes = " ".join(sv.platform_notes("dgx-spark")).lower()
    assert "cuda" in notes
    # And must not contradict the corrected guide by implying a toolkit install.
    assert "do not install a toolkit" in notes


def test_both_platforms_warn_about_shared_memory_across_all_three_services():
    # The three services are resident simultaneously. Sizing them one at a time is how a
    # box ends up thrashing with every individual model "fitting".
    for key in ("dgx-spark", "strix-halo"):
        notes = " ".join(sv.platform_notes(key)).lower()
        assert "unified" in notes
        assert "together" in notes or "combined" in notes


def test_unknown_platform_yields_no_invented_caveats():
    assert sv.platform_notes("") == []
    assert sv.platform_notes("some-other-box") == []


def test_surya1_says_it_ships_no_server():
    # The trap: Surya 1 is a library, not a service. There is no upstream image to pull and
    # no `surya serve`, so the HTTP wrapper is a component someone has to build. Planning
    # around finding a ready-made one is how "just run Surya" becomes a lost day.
    blob = " ".join(f"{s.title} {s.what} {' '.join(s.warnings)}"
                    for s in sv.SERVICES["surya"].steps).lower()
    assert "library" in blob
    assert "no upstream image" in blob or "no official ready-made server image" in blob


def test_surya1_wrapper_loads_models_once_not_per_request():
    # Per-request loading passes a one-page test and then adds full model-load time to
    # every call in production.
    blob = " ".join(f"{s.what} {s.command} {' '.join(s.warnings)}"
                    for s in sv.SERVICES["surya"].steps).lower()
    assert "per request" in blob
    assert "startup" in blob


def test_surya1_health_must_not_go_green_early():
    # A health endpoint that returns 200 before models are resident makes every orchestrator
    # declare the service ready while the first real request times out.
    surya = sv.SERVICES["surya"]
    blob = " ".join(f"{s.expect} {' '.join(s.warnings)}" for s in surya.steps).lower()
    assert "resident" in blob


def test_surya1_contract_is_described_without_naming_any_deployment():
    # Deneb ships the CONTRACT, never anyone's source. Nothing here may carry a customer,
    # product or project name.
    blob = " ".join(f"{s.title} {s.what} {s.command}" for s in sv.SERVICES["surya"].steps).lower()
    assert "/healthz" in blob and "/layout" in blob
    for forbidden in ("cf-platform", "cf_platform", "docflow", "chong"):
        assert forbidden not in blob
