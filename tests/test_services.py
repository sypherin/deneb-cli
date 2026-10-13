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
        "surya": "ocr", "ocr": "ocr", "surya 2": "ocr", "surya2": "ocr",
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
    for backend in ("llm", "vlm", "ocr"):
        assert order.index(backend) < order.index("gateway")


def test_tunnel_targets_the_gateway_not_a_model_port():
    tunnel = sv.SERVICES["tunnel"]
    commands = " ".join(s.command for s in tunnel.steps)
    assert "8002" in commands, "the tunnel must point at the gateway"
    for model_port in ("8001", "8080", "8082"):
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


def test_every_backend_binds_loopback_only():
    # The gateway runs on the same box and reaches loopback, so no command needs another bind.
    for key in ("llm", "vlm", "ocr"):
        commands = " ".join(s.command for s in sv.SERVICES[key].steps)
        assert "0.0.0.0" not in commands, f"{key} must not bind every interface"
        assert "127.0.0.1" in commands


def test_stack_output_never_binds_all_interfaces(capsys):
    import re
    from deneb import cli
    assert cli.cmd_stack([]) == 0
    out = capsys.readouterr().out
    assert "0.0.0.0" not in out
    assert "is deliberate here" not in out
    for host in re.findall(r"--host\s+(\S+)", out):
        assert host == "127.0.0.1", host


def test_stack_ports_are_labelled_as_examples(capsys):
    from deneb import cli
    cli.cmd_stack([])
    assert "example" in capsys.readouterr().out.lower()


# ── correctness traps specific to these models ────────────────────────────────
def test_vision_services_require_the_projector():
    # Without mmproj a vision model loads, answers text, and silently cannot see images.
    for key in ("vlm", "ocr"):
        blob = " ".join(f"{s.command} {s.what}" for s in sv.SERVICES[key].steps)
        assert "mmproj" in blob, f"{key} must fetch and pass the mmproj projector"


def test_ocr_runs_at_temperature_zero():
    # OCR is transcription. Any sampling temperature invents characters.
    commands = " ".join(s.command for s in sv.SERVICES["ocr"].steps)
    assert "--temp 0" in commands


def test_llm_keeps_its_speculative_draft_model():
    commands = " ".join(s.command for s in sv.SERVICES["llm"].steps)
    assert "--spec-type draft-mtp" in commands
    assert "-md " in commands, "the draft model is what makes speculative decoding work"


def test_ocr_uses_the_published_gguf_not_a_hand_patched_checkpoint():
    # Datalab publishes Surya 2 as a GGUF (datalab-to/surya-ocr-2-gguf). An earlier version
    # of this advice had people patch the checkpoint's config and convert it themselves,
    # which was one machine's history, not the way to get it.
    ocr = sv.SERVICES["ocr"]
    commands = " ".join(s.command for s in ocr.steps)
    assert "hf download datalab-to/surya-ocr-2-gguf" in commands
    blob = " ".join(f"{s.title} {s.what} {s.command}" for s in ocr.steps).lower()
    for gone in ("patch", "convert_hf_to_gguf", "config.json", "qwen3vl_merger"):
        assert gone not in blob, gone


def test_ocr_is_not_quantised():
    # The published GGUF is 16-bit. On a 650M OCR model the saving is trivial while the
    # accuracy loss is silent.
    steps = sv.SERVICES["ocr"].steps
    commands = " ".join(s.command for s in steps)
    assert "llama-quantize" not in commands and "q4_k" not in commands.lower()
    blob = " ".join(f"{s.what} {s.expect}" for s in steps).lower()
    assert "16-bit" in blob


def test_ocr_is_checked_on_your_own_pages_first():
    # A wrong character looks exactly like a right one, so nothing downstream catches it.
    blob = " ".join(f"{s.title} {s.what} {' '.join(s.warnings)}"
                    for s in sv.SERVICES["ocr"].steps).lower()
    assert "your own pages" in blob and "silent" in blob
    assert "surya 1" not in blob


def test_the_stack_is_five_services_with_no_private_wrapper():
    # Surya 1 needed an HTTP wrapper someone writes themselves, and the advice described one
    # machine's wrapper contract. A recipe for strangers ships services they can fetch.
    assert set(sv.SERVICES) == {"llm", "vlm", "ocr", "gateway", "tunnel"}
    assert list(sv.STACK_ORDER) == ["llm", "vlm", "ocr", "gateway", "tunnel"]


def test_no_service_text_names_a_deployment():
    # Deneb ships recipes, never anyone's deployment. The real client / product / project
    # terms to guard live in an env var so they are NOT published in this open repo (the
    # public default below is generic placeholders). Locally, set
    # DENEB_FORBIDDEN_TERMS="term1,term2,..." to assert against the real ones.
    import os
    forbidden = os.environ.get(
        "DENEB_FORBIDDEN_TERMS", "acme-corp,example_project,widgetflow"
    ).split(",")
    for key, svc in sv.SERVICES.items():
        blob = f"{svc.label} {svc.role} {svc.notes} " + " ".join(
            f"{s.title} {s.what} {s.command} {s.expect} {' '.join(s.warnings)}"
            for s in svc.steps)
        for term in forbidden:
            term = term.strip().lower()
            if term:
                assert term not in blob.lower(), (key, term)


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
    assert "gguf" in notes and "surya 1" not in notes


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


# ── download and serve must agree ─────────────────────────────────────────────
def _gguf_names(text):
    import re
    return set(re.findall(r"[\w.\-/]+\.gguf", text))


def test_every_file_the_server_opens_is_one_the_download_fetches():
    """The defect this catches actually shipped.

    The download commands were written from the model repository's contents and the serve
    commands from a working machine, where the files had been renamed by hand. Both halves
    were individually correct and together they were unrunnable: following the steps in
    order downloaded one set of filenames and then started a server pointing at another.
    Nothing in the text looked wrong, because each line was true in isolation.
    """
    for key in ("llm", "vlm", "ocr"):
        svc = sv.SERVICES[key]
        fetched, opened = set(), set()
        for step in svc.steps:
            if "hf download" in step.command:
                fetched |= _gguf_names(step.command)
            if "llama-server" in step.command:
                opened |= {n.split("/")[-1] for n in _gguf_names(step.command)}
        assert fetched, f"{key}: no download step found"
        assert opened, f"{key}: no serve step found"
        fetched_base = {n.split("/")[-1] for n in fetched}
        missing = opened - fetched_base
        assert not missing, (
            f"{key}: the serve command opens {sorted(missing)}, which the download step "
            f"never fetches (it fetches {sorted(fetched_base)})")


def test_the_speculative_draft_model_is_actually_downloaded():
    # -md pointing at a file that was never fetched means the server refuses to start, and
    # the error names a path rather than the missing --include.
    llm = sv.SERVICES["llm"]
    dl = " ".join(s.command for s in llm.steps if "hf download" in s.command)
    assert "mtp" in dl.lower(), "the MTP draft model is referenced but never downloaded"
