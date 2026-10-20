"""Unit tests for deneb.check — the DETERMINISTIC scan.

The parsers (`_port_owner`, `_unit_of_pid`, `_has`) and each `_chk_*` verdict are pure
functions of shell output, so we feed canned `ss` / `systemctl` / `which` / `curl` output
and assert the checklist. These tests exist because two real bugs shipped without them:
  - `_has`: `which nvidia-smi` prints "no nvidia-smi in ..." on a miss → a substring test
    false-matched CUDA on an AMD box. (test_has_rejects_which_miss)
  - detection by service *filename* named the wrong unit. Now we detect by PORT-owner +
    PID→unit and assert the resolved name. (test_gateway_names_real_unit)

Run: python3 tests/test_check.py    (also collectable by pytest)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import check  # noqa: E402
import private_names  # noqa: E402

SS = (
    'LISTEN 0 512  127.0.0.1:8002  0.0.0.0:*  users:(("bun",pid=2892,fd=11))\n'
    'LISTEN 0 512    0.0.0.0:8001  0.0.0.0:*  users:(("llama-server",pid=876334,fd=10))\n'
)
STATUS_GATEWAY = "● llm-gateway.service - llm.example.com secure inference gateway (Bun)\n  Loaded: loaded"
STATUS_LLAMA = "● llama-server.service - llama-server (Vulkan - Strix Halo - Qwen 3.6)\n  Loaded: loaded"


_REAL = {"_probe": check._probe, "_read": check._read}
_real_ollama_install = getattr(check, "_ollama_install", None)
_real_own_unit = getattr(check, "_own_llama_unit_exists", None)


def teardown_function(_fn=None):
    """Put the real probes back after every test (pytest calls this by name), so a fake
    never leaks into another test file: test_watch.py runs the real scan."""
    for name, fn in _REAL.items():
        setattr(check, name, fn)
    check._reset_port_cache()
    os.environ.pop("DENEB_PORT", None)


def _router(mapping: dict):
    """Return a fake _probe that yields the first value whose key is a substring of cmd."""
    def fake(cmd: str) -> str:
        for key, val in mapping.items():
            if key in cmd:
                return val
        return ""
    return fake


def _install(mapping: dict, reads: dict | None = None):
    check._probe = _router(mapping)
    check._read = _router(reads or {})


# ── pure parsers ─────────────────────────────────────────────────────────────
def test_port_owner_parses_ss():
    _install({"ss -ltnp": SS})
    assert check._port_owner(8002) == ("2892", "bun")
    assert check._port_owner(8001) == ("876334", "llama-server")
    assert check._port_owner(9999) == ("", "")  # nothing listening


def test_unit_of_pid_resolves_name():
    _install({"status 2892": STATUS_GATEWAY})
    assert check._unit_of_pid("2892") == "llm-gateway.service"
    assert check._unit_of_pid("") == ""     # no pid
    assert check._unit_of_pid("?") == ""    # owner hidden


def test_has_rejects_which_miss():
    # THE cuda bug: a miss must be False even though the text contains the command name.
    check._probe = lambda c: "[stderr]\nwhich: no nvidia-smi in (/usr/bin:/bin)"
    assert check._has("nvidia-smi") is False


def test_has_accepts_real_path():
    check._probe = lambda c: "/usr/bin/rocminfo"
    assert check._has("rocminfo") is True


# ── whole checks: a healthy AMD (Strix) box ──────────────────────────────────
HEALTHY = {
    "ss -ltnp": SS,
    "status 2892": STATUS_GATEWAY,
    "status 876334": STATUS_LLAMA,
    "is-enabled": "enabled",
    "which nvidia-smi": "[stderr]\nwhich: no nvidia-smi in (/usr/bin)",  # AMD box: no CUDA
    "which rocminfo": "/usr/bin/rocminfo",
    "pgrep -af llama-server": "876334 /usr/bin/llama-server -ngl 99 --port 8001 --jinja",
    "curl -s localhost:8002/v1/models": "unauthorized: missing api key",
    "pgrep -af cloudflared": "999 cloudflared tunnel run my-tunnel",
    "ps -p 999 -o args=": "/usr/bin/cloudflared --config /etc/cloudflared/config.yml tunnel run",
    "curl -s localhost:8001/health": '{"status":"ok"}',
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"/models/Qwen3.6.gguf"}]}',
    "models -maxdepth 2": "/home/x/models/Qwen3.6.gguf",
}
# the tunnel's config file, read (not run): its ingress routes a hostname to the key gateway
HEALTHY_READS = {"/etc/cloudflared/config.yml": "tunnel: my-tunnel\ningress:\n"
                 "  - hostname: llm.example.com\n    service: http://localhost:8002\n"
                 "  - service: http_status:404\n"}


def test_gpu_is_rocm_not_cuda_on_amd():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_gpu()
    assert ok is True
    assert "vulkan/rocm" in ev and "cuda" not in ev  # the regression this test guards


def test_gateway_names_real_unit_and_auth_enforced():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_gateway()
    assert ok is True
    assert "llm-gateway.service" in ev        # resolved via PID, not a filename glob
    assert "auth enforced" in ev
    assert "goose" not in ev                   # must not grab an unrelated *gateway* unit


def test_service_reports_running_and_enabled():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_service()
    assert ok is True
    assert "llama-server.service" in ev and "enabled" in ev


def test_tunnel_up():
    _install(HEALTHY, HEALTHY_READS)
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is True and "routes to the gateway (:8002)" in ev


# ── whole checks: a broken box (CPU-only, no gateway, disabled service) ───────
CPU_ONLY_SS = 'LISTEN 0 512  0.0.0.0:8001  0.0.0.0:*  users:(("llama-server",pid=5,fd=10))\n'
BROKEN = {
    "ss -ltnp": CPU_ONLY_SS,              # nothing on :8002
    "status 5": STATUS_LLAMA,
    "is-enabled": "disabled",             # won't survive reboot
    "which nvidia-smi": "which: no nvidia-smi in (/usr/bin)",
    "which rocminfo": "which: no rocminfo in (/usr/bin)",  # no GPU toolkit at all
    "pgrep -af llama-server": "5 /usr/bin/llama-server --port 8001",  # NO -ngl → CPU
    "curl -s localhost:8002/v1/models": "",
    "pgrep -af cloudflared": "",
    "curl -s localhost:8001/health": '{"status":"ok"}',
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"/models/Qwen3.6.gguf"}]}',
    "models -maxdepth 2": "/home/x/models/Qwen3.6.gguf",
}


def test_broken_gpu_flags_cpu():
    _install(BROKEN)
    _, ok, ev, fix = check._chk_gpu()
    assert ok is False and "CPU" in ev and fix


def test_broken_gateway_absent():
    _install(BROKEN)
    _, ok, ev, _ = check._chk_gateway()
    assert ok is None and "not set up" in ev       # optional: nothing there is not a failure


def test_broken_tunnel_absent():
    _install(BROKEN)
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is None and "not set up" in ev


def test_disabled_service_still_running_but_warned():
    _install(BROKEN)
    _, ok, ev, fix = check._chk_service()
    assert ok is True                      # it IS serving :8001
    assert "won't auto-start" in ev and "enable" in fix  # but flag the reboot risk


# ── a real Strix box (2026-10-10): bundled ROCm build, no toolkit on PATH, model alias ──
STRIX_SS = 'LISTEN 0 512  127.0.0.1:8001  0.0.0.0:*  users:(("llama-server",pid=1925615,fd=11))\n'
STRIX_MODEL = "/mnt/m/qwen3.8-27b-fp4/Qwen3.8-27B-ROCmFP4-FAST.gguf"
STRIX_PGREP = (
    "889803 /home/x/llama.cpp/build/bin/llama-server -m /home/x/models/judge-Q8_0.gguf --port 8101 -ngl 99\n"
    f"1925591 /bin/bash /home/x/bin/guard.sh --need 30 /mnt/q/bin/llama-server -m {STRIX_MODEL}\n"
    f"1925615 /mnt/q/bin/llama-server -m {STRIX_MODEL} -dev ROCm0 -ngl 99 -np 1 -c 131072\n"
    "1958540 /home/x/llama.cpp-alt/build/bin/llama-server -m /mnt/m/alt.gguf --port 8102 -ngl 999\n"
)
STRIX = {
    "ss -ltnp": STRIX_SS,
    "pgrep -af llama-server": STRIX_PGREP,
    "which nvidia-smi": "[stderr]\nwhich: no nvidia-smi in (/usr/bin)",
    "which rocminfo": "[stderr]\nwhich: no rocminfo in (/usr/bin)",  # no toolkit on PATH
    "ldd /mnt/q/bin/llama-server": ("\tlibhipblas.so.3 => not found\n\tlibamdhip64.so.7 => not found\n"
                                    "\tlibvulkan.so.1 => /lib64/libvulkan.so.1 (0x00007f3f)"),
    "-name libggml-*": "(no output)",
    f"stat -L -c %s {STRIX_MODEL}": "14562236384",
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"qwen3.8-27b"}]}',  # an alias, not a path
    "models -maxdepth 2": "",
    "/ns/mnt": "mnt:[4026531840]",          # a native server: same mount namespace as deneb
}


def test_serving_cmdline_is_the_8001_pid_not_the_first_llama_server():
    _install(STRIX)
    assert check._serving_cmdline().startswith("1925615 /mnt/q/bin/llama-server")


def test_gpu_read_from_the_server_build_without_a_toolkit():
    _install(STRIX)
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True, ev                   # was "no GPU toolkit found" on a GPU box
    assert "rocm" in ev and "vulkan" in ev and "cuda" not in ev


def test_gpu_judged_on_the_serving_process_only():
    # A judge on :8101 has -ngl; the :8001 server doesn't. The old any-line test passed it.
    pgrep = STRIX_PGREP.replace("-dev ROCm0 -ngl 99 ", "")
    _install({**STRIX, "pgrep -af llama-server": pgrep})
    _, ok, ev, _ = check._chk_gpu()
    assert ok is False and "CPU" in ev


def test_llama_without_ngl_is_not_passed_on_the_build_default():
    # Current llama.cpp defaults -ngl to auto, older builds to 0: with the GPU runtime
    # loaded and no flag, the honest answer is "pin it", not "on CPU" and not a pass.
    pgrep = STRIX_PGREP.replace("-dev ROCm0 -ngl 99 ", "")
    _install({**STRIX, "pgrep -af llama-server": pgrep, "/proc/1925615/maps": "libamdhip64\n"})
    _, ok, ev, fix = check._chk_gpu()
    assert ok is False and "build default" in ev and "-ngl 99" in fix


def test_ngl_zero_is_cpu():
    _install({**STRIX, "pgrep -af llama-server": STRIX_PGREP.replace("-ngl 99 -np", "-ngl 0 -np")})
    _, ok, ev, _ = check._chk_gpu()
    assert ok is False and "-ngl 0" in ev


def test_draft_model_flag_is_not_the_offload_flag():
    pgrep = STRIX_PGREP.replace("-dev ROCm0 -ngl 99 ", "-ngld 99 ")
    _install({**STRIX, "pgrep -af llama-server": pgrep})
    _, ok, _, _ = check._chk_gpu()
    assert ok is False


def test_cpu_only_build_with_offload_flag_still_fails():
    _install({**STRIX, "ldd /mnt/q/bin/llama-server": "\tlibc.so.6 => /lib64/libc.so.6"})
    _, ok, _, fix = check._chk_gpu()
    assert ok is False and fix


def test_model_file_from_the_serving_cmdline_when_models_reports_an_alias():
    _install(STRIX)
    _, ok, ev, _ = check._chk_model_file()
    assert ok is True, ev                   # was "no .gguf found" while serving it
    assert ev == "Qwen3.8-27B-ROCmFP4-FAST.gguf (13.6 GiB)"


def test_model_file_deleted_under_a_running_server():
    _install({**STRIX, f"stat -L -c %s {STRIX_MODEL}":
              f"[stderr]\nstat: cannot statx '{STRIX_MODEL}': No such file or directory"})
    _, ok, ev, fix = check._chk_model_file()
    assert ok is False and "next restart" in ev and STRIX_MODEL in fix


def test_model_file_unreadable_path_falls_back_to_models_dir():
    _install({**STRIX, f"stat -L -c %s {STRIX_MODEL}": "[stderr]\nstat: Permission denied",
              "models -maxdepth 2": "/home/x/models/other.gguf"})
    _, ok, ev, _ = check._chk_model_file()
    assert ok is True and ev == "other.gguf"


def test_model_file_missing_in_an_unknown_namespace_is_not_called_deleted():
    # Another user's process: its /proc entries can't be read, so a path that is missing
    # from deneb's view may still exist in the server's. Fall through, don't accuse.
    mapping = {k: v for k, v in STRIX.items() if k != "/ns/mnt"}
    mapping[f"stat -L -c %s {STRIX_MODEL}"] = "[stderr]\nstat: No such file or directory"
    _install(mapping)
    _, ok, ev, _ = check._chk_model_file()
    assert ok is False and "next restart" not in ev


# ── the model in a rootless container: the port belongs to the runtime's forwarder ──
HOST_NS, CT_NS = "mnt:[4026531832]", "mnt:[4026533311]"
CT_MODEL = "/models/model-Q4.gguf"
CT = {
    "ss -ltnp": ('LISTEN 0 4096  127.0.0.1:8001  0.0.0.0:*  users:(("rootlessport",pid=5001,fd=10))\n'
                 'LISTEN 0 512   127.0.0.1:8005  0.0.0.0:*  users:(("llama-server",pid=5300,fd=8))\n'),
    "pgrep -af '\\.gguf'": (
        f"5010 /bin/bash /home/x/bin/load-guard.sh podman run --rm -p 127.0.0.1:8001:9000 img --weights {CT_MODEL}\n"
        f"5020 podman run --rm -p 127.0.0.1:8001:9000 img --weights {CT_MODEL}\n"
        f"5090 /bin/sh /entrypoint.sh --weights {CT_MODEL}\n"
        f"5100 /usr/local/bin/model_serve --weights {CT_MODEL} --port 9000 --ctx 131072\n"
        "5300 /home/x/llama.cpp/build/bin/llama-server -m /home/x/models/judge.gguf --port 8005 -ngl 99\n"),
    "readlink /proc/self/ns/mnt": HOST_NS,
    "readlink /proc/5020/ns/mnt": "mnt:[4026533100]",   # rootless podman: a namespace of its own
    "readlink /proc/5090/ns/mnt": CT_NS,
    "readlink /proc/5100/ns/mnt": CT_NS,
    "/ns/mnt": HOST_NS,                                   # the wrappers and the judge: host side
    "/proc/5100/maps": "libamdhip64\nlibhipblas\nlibhipblas\n",
    f"stat -L -c %s /proc/5100/root{CT_MODEL}": "93682583296",
    "which nvidia-smi": "[stderr]\nwhich: no nvidia-smi in (/usr/bin)",
    "which rocminfo": "[stderr]\nwhich: no rocminfo in (/usr/bin)",
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"local-model"}]}',
    "models -maxdepth 2": "",
}


def test_container_server_found_behind_the_port_forwarder():
    # Not the forwarder, not the host-side wrapper, not `podman run` (which sits in a mount
    # namespace of its own), not the judge, not the entrypoint shell.
    _install(CT)
    assert check._serving_cmdline().startswith("5100 /usr/local/bin/model_serve")


def test_container_gpu_read_from_the_runtime_it_loaded():
    _install(CT)
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True, ev                   # was "running on CPU (no GPU-offload flag)"
    assert "rocm" in ev and "cuda" not in ev


def test_container_gpu_without_a_runtime_is_cpu():
    _install({**CT, "/proc/5100/maps": ""})
    _, ok, ev, _ = check._chk_gpu()
    assert ok is False and "CPU" in ev


def test_container_model_file_read_through_the_process_root():
    _install(CT)
    _, ok, ev, _ = check._chk_model_file()
    assert ok is True, ev                   # was "no .gguf found"
    assert ev == "model-Q4.gguf (87.2 GiB), in a container"


def test_container_model_file_deleted_under_a_running_server():
    _install({**CT, f"stat -L -c %s /proc/5100/root{CT_MODEL}":
              "[stderr]\nstat: No such file or directory"})
    _, ok, ev, fix = check._chk_model_file()
    assert ok is False and "next restart" in ev and CT_MODEL in fix


def test_container_server_not_visible_says_so():
    _install({**CT, "pgrep -af '\\.gguf'": ""})
    assert check._serving_cmdline() == ""
    _, ok, ev, _ = check._chk_gpu()
    assert ok == check.UNKNOWN and "couldn't read the command line" in ev   # unread, never a verdict


def test_models_dir_fallback_expands_home():
    seen = []
    check._probe = lambda c: seen.append(c) or ""
    check._read = lambda p: ""
    check._chk_model_file()
    assert any(c.startswith("find /") and c.endswith("models -maxdepth 2 -name '*.gguf'") for c in seen), seen


# ═══ any server on any port: serving_port(), endpoint, system units, GPU, optional rows ═══
def _ss(*listeners):
    """Canned `ss -ltnp` for (port, process, pid) listeners."""
    return "".join(f'LISTEN 0 512  0.0.0.0:{p}  0.0.0.0:*  users:(("{n}",pid={pid},fd=10))\n'
                   for p, n, pid in listeners)


_MODELS = '{"object":"list","data":[{"id":"%s"}]}'


def _box(port, proc, pid=77, model="m1", **extra):
    """A box with one server on `port` that answers /v1/models."""
    m = {"ss -ltnp": _ss((port, proc, pid)),
         f"curl -s localhost:{port}/v1/models": _MODELS % model}
    m.update(extra)
    return m


def test_port_default_layout_still_8001():
    _install(HEALTHY)
    assert check.serving_port() == 8001


def test_port_each_server_type_is_found():
    # python3 is vLLM or any app: it is a model server only if it says "object": "model".
    vllm_body = '{"object":"list","data":[{"id":"m1","object":"model"}]}'
    for port, proc in ((8080, "llama-server"), (8000, "python3"), (11434, "ollama"), (1234, "LM Studio")):
        check._reset_port_cache()
        box = _box(port, proc)
        if proc == "python3":
            box[f"curl -s localhost:{port}/v1/models"] = vllm_body
        _install(box)
        assert check.serving_port() == port, (port, proc)


def test_port_order_prefers_8001_when_several_answer():
    _install({"ss -ltnp": _ss((11434, "ollama", 5), (8001, "llama-server", 6), (8080, "llama-server", 7)),
              "curl -s localhost:11434/v1/models": _MODELS % "a",
              "curl -s localhost:8001/v1/models": _MODELS % "b",
              "curl -s localhost:8080/v1/models": _MODELS % "c"})
    assert check.serving_port() == 8001


def test_port_needs_a_listener_even_if_curl_would_answer():
    # curl is canned to answer on :8080, but ss shows nothing there: not a candidate.
    _install({"ss -ltnp": _ss((9999, "other", 1)), "curl -s localhost:8080/v1/models": _MODELS % "x"})
    assert check.serving_port() == 8001


def test_port_listener_that_does_not_answer_json_is_skipped():
    _install({"ss -ltnp": _ss((8001, "nginx", 1), (8080, "llama-server", 2)),
              "curl -s localhost:8001/v1/models": "<html>welcome</html>",
              "curl -s localhost:8080/v1/models": _MODELS % "x"})
    assert check.serving_port() == 8080


def test_port_falls_back_to_known_server_owner_when_nothing_answers():
    # Models still loading: nothing answers, but :11434 is owned by ollama and :8080 by bash.
    _install({"ss -ltnp": _ss((8080, "bash", 1), (11434, "ollama", 2))})
    assert check.serving_port() == 11434


def test_port_fallback_knows_vllm_under_python_and_lm_studio():
    _install({"ss -ltnp": _ss((8000, "python3", 31)),
              "ps -p 31": "python3 -m vllm.entrypoints.openai.api_server --model x"})
    assert check.serving_port() == 8000
    check._reset_port_cache()
    _install({"ss -ltnp": _ss((8000, "python3", 31)), "ps -p 31": "python3 -m http.server 8000"})
    assert check.serving_port() == 8001          # a plain python server is not an LLM server
    check._reset_port_cache()
    _install({"ss -ltnp": _ss((1234, "LM Studio", 9))})
    assert check.serving_port() == 1234
    check._reset_port_cache()
    _install({"ss -ltnp": _ss((1234, "lms", 9))})
    assert check.serving_port() == 1234


def test_port_nothing_matches_uses_8001():
    _install({"ss -ltnp": _ss((22, "sshd", 1))})
    assert check.serving_port() == 8001
    check._reset_port_cache()
    _install({})
    assert check.serving_port() == 8001


def test_port_env_var_wins_even_when_something_else_answers():
    os.environ["DENEB_PORT"] = "9090"
    _install(_box(11434, "ollama"))
    assert check.serving_port() == 9090


def test_port_env_var_invalid_uses_default_and_warns_on_the_endpoint_row():
    for bad in ("abc", "0", "65536", "-5", "80.5", ""):
        check._reset_port_cache()
        os.environ["DENEB_PORT"] = bad
        _install(_box(8001, "llama-server", **{"curl -s localhost:8001/health": '{"status":"ok"}'}))
        if bad == "":
            assert check.serving_port() == 8001      # empty = unset, no warning
            _, ok, ev, _ = check._chk_endpoint()
            assert "⚠" not in ev
            continue
        assert check.serving_port() == 8001, bad
        _, ok, ev, _ = check._chk_endpoint()
        assert "⚠" in ev and "DENEB_PORT" in ev and bad in ev, (bad, ev)
        assert ok is True


def test_port_is_computed_once_per_process():
    calls = []
    check._probe = lambda c: calls.append(c) or ""
    check.serving_port(); n = len(calls)
    check.serving_port(); check.serving_port()
    assert len(calls) == n and n >= 1


def test_labels_and_texts_carry_the_resolved_port():
    _install(_box(11434, "ollama", model="llama3:8b"))
    label, ok, ev, _ = check._chk_endpoint()
    assert label == "LLM endpoint serving (:11434)" and ok and "llama3:8b" in ev
    _, ok, ev, _ = check._chk_service()
    assert ":11434" in ev and ":8001" not in ev
    _install({"ss -ltnp": ""})
    check._reset_port_cache()
    _, ok, ev, _ = check._chk_service()
    assert ok is False and ":8001" in ev


# ── endpoint: every server type ──
def test_endpoint_vllm_empty_health_with_models_data_is_ok():
    _install(_box(8000, "python3", **{"curl -s localhost:8000/health": "(no output)",
                                      "ps -p 77": "python3 -m vllm.entrypoints.openai.api_server"},
                  model="org/Qwen-7B"))
    label, ok, ev, _ = check._chk_endpoint()
    assert ok is True and label == "LLM endpoint serving (:8000)" and ev == "serving Qwen-7B"


def test_endpoint_ollama_has_no_health_but_models_data_is_ok():
    _install(_box(11434, "ollama", **{"curl -s localhost:11434/health": "404 page not found"}))
    assert check._chk_endpoint()[1] is True


def test_endpoint_health_only_still_ok():
    _install({"ss -ltnp": _ss((8001, "llama-server", 5)), "curl -s localhost:8001/health": '{"status":"ok"}'})
    _, ok, ev, _ = check._chk_endpoint()
    assert ok is True and ev == "serving"


def test_endpoint_down_is_a_failure_naming_the_port():
    _install({"ss -ltnp": _ss((8080, "llama-server", 5)),
              "curl -s localhost:8080/health": "[stderr]\ncurl: (7) Failed to connect"})
    label, ok, ev, fix = check._chk_endpoint()
    assert ok is False and ":8080" in ev and ":8080" in label and fix


# ── service row: system units ──
STATUS_OLLAMA = "● ollama.service - Ollama Service\n     Loaded: loaded (/etc/systemd/system/ollama.service; enabled)"


def test_service_system_unit_is_reported_with_its_scope():
    _install(_box(11434, "ollama", pid=77, **{"systemctl status 77": STATUS_OLLAMA,
                                              "systemctl is-enabled ollama.service": "enabled"}))
    _, ok, ev, _ = check._chk_service()
    assert ok is True and "ollama.service (system)" in ev and "enabled at boot" in ev and ":11434" in ev


def test_service_system_unit_disabled_is_a_warning_with_a_system_fix():
    _install(_box(11434, "ollama", pid=77, **{"systemctl status 77": STATUS_OLLAMA,
                                              "systemctl is-enabled ollama.service": "disabled"}))
    _, ok, ev, fix = check._chk_service()
    assert ok is True and "⚠" in ev and "systemctl enable ollama.service" in fix and "--user" not in fix


def test_scoped_unit_keeps_user_and_system_apart():
    _install({"systemctl --user status 5": "● same.service - a user unit"})
    assert check._scoped_unit_of_pid("5") == ("same.service", "user")
    _install({"systemctl status 5": "● same.service - a system unit"})
    assert check._scoped_unit_of_pid("5") == ("same.service", "system")
    _install({})
    assert check._scoped_unit_of_pid("5") == ("", "")
    assert check._scoped_unit_of_pid("?") == ("", "") and check._scoped_unit_of_pid("") == ("", "")


def test_system_scope_ignores_the_user_manager_and_scopes():
    # A bare process started from a terminal or under the user manager is NOT a model service.
    _install({"systemctl status 5": "● user@1000.service - User Manager for UID 1000\n  CGroup: ..."})
    assert check._scoped_unit_of_pid("5") == ("", "")
    _install({"systemctl status 5": "● session-3.scope - Session 3 of User x\n  CGroup: ... foo.service"})
    assert check._scoped_unit_of_pid("5") == ("", "")


def test_user_unit_wins_over_system_and_system_probe_is_not_run():
    seen = []
    base = _router({"ss -ltnp": _ss((8001, "llama-server", 5)), "systemctl --user status 5": STATUS_LLAMA,
                    "systemctl --user is-enabled": "enabled"})
    check._probe = lambda c: seen.append(c) or base(c)
    check._read = _router({})
    _, ok, ev, _ = check._chk_service()
    assert "llama-server.service" in ev and "(system)" not in ev
    assert not any(c.startswith("systemctl status") for c in seen)


# ── GPU row per server type ──
OLLAMA_PS = "NAME ID SIZE PROCESSOR UNTIL\n{row}"


def _ollama(row, **extra):
    return _box(11434, "ollama", pid=77, **{"pgrep -af ollama": "77 /usr/local/bin/ollama serve",
                                            "ollama ps": OLLAMA_PS.format(row=row), **extra})


def test_gpu_ollama_full_gpu_is_ok():
    _install(_ollama("llama3:8b  365c0bd3c000  5.4 GB  100% GPU  4 minutes from now"))
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True and "100% GPU" in ev and "⚠" not in ev


def test_gpu_ollama_cpu_split_is_a_warning():
    _install(_ollama("big:70b  abc  40 GB  48%/52% CPU/GPU  4 minutes from now"))
    _, ok, ev, fix = check._chk_gpu()
    assert ok is True and "⚠" in ev and "48%" in ev and fix


def test_gpu_ollama_cpu_only_fails():
    _install(_ollama("llama3:8b  365c0bd3c000  5.4 GB  100% CPU  4 minutes from now"))
    _, ok, ev, fix = check._chk_gpu()
    assert ok is False and "CPU" in ev and fix


def test_gpu_ollama_no_model_loaded_is_unread_until_a_model_loads():
    # 14 Oct QC F3: an idle Ollama can't say where a model will run, so this row is '?',
    # never a quiet '-' under "all checks pass"
    _install(_ollama(""))
    _, ok, ev, _ = check._chk_gpu()
    assert ok == check.UNKNOWN and "no model loaded" in ev and "one request" in ev


def test_idle_ollama_is_not_all_checks_pass(monkeypatch):
    _install(_ollama(""))
    rows = [("endpoint", "LLM endpoint serving", True, "serving :11434", ""), ("gpu",) + check._chk_gpu()]
    rc, out = _run_with_rows(monkeypatch, rows)
    assert rc == 1 and "all checks pass" not in out and "could not be read" in out, out


def test_readme_sample_is_not_the_idle_false_green():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "– Built with GPU accelerator (not CPU)  — no model loaded right now" not in readme


def test_gpu_ollama_cli_missing_is_unreadable_not_a_failure():
    _install(_ollama("", **{"ollama ps": "(not installed on this box: ollama)"}))
    _, ok, ev, _ = check._chk_gpu()
    assert ok == check.UNKNOWN and "ollama ps" in ev


def test_gpu_vllm_ok_unless_device_cpu():
    base = _box(8000, "python3", pid=31, **{"ps -p 31": "python3 -m vllm.entrypoints.openai.api_server --model x",
                                            "pgrep -af python3": "31 python3 -m vllm.entrypoints.openai.api_server --model x"})
    _install(base)
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True and "vLLM" in ev
    cpu = "31 python3 -m vllm.entrypoints.openai.api_server --model x --device cpu"
    _install({**base, "pgrep -af python3": cpu})
    _, ok, ev, fix = check._chk_gpu()
    assert ok is False and "--device cpu" in ev and fix
    _install({**base, "pgrep -af python3": cpu.replace("--device cpu", "--device=cpu")})
    assert check._chk_gpu()[1] is False


def test_gpu_lm_studio_is_not_applicable():
    _install(_box(1234, "LM Studio", pid=9, **{"pgrep -af": "9 /opt/LM-Studio/lm-studio"}))
    _, ok, ev, _ = check._chk_gpu()
    assert ok is None and ev == "LM Studio doesn't expose GPU offload; check its settings"


def test_gpu_llama_server_on_8080_uses_its_own_cmdline():
    _install({**_box(8080, "llama-server", pid=5),
              "pgrep -af llama-server": "5 /usr/bin/llama-server -ngl 99 --port 8080 --jinja",
              "which nvidia-smi": "/usr/bin/nvidia-smi"})
    _, ok, ev, _ = check._chk_gpu()
    assert ok is True and "GPU offload on" in ev


def test_serving_cmdline_matches_the_resolved_port_flag():
    # Owner seen but its pid is not in the pgrep output: fall back to the line whose --port
    # is the resolved port. (A hidden owner gets no pgrep guess at all: item 2.)
    _install({"ss -ltnp": _ss((8080, "llama-server", 99)),
              "curl -s localhost:8080/v1/models": _MODELS % "x",
              "pgrep -af llama": "1 llama-server --port 8005\n2 llama-server --port 8080 -ngl 99"})
    assert check._serving_cmdline().startswith("2 llama-server")


# ── optional rows: gateway + tunnel ──
PLAIN = {"ss -ltnp": _ss((8001, "llama-server", 5)), "pgrep -af cloudflared": "(no output)",
         "curl -s localhost:8001/v1/models": _MODELS % "m"}


def test_gateway_and_tunnel_not_set_up_are_none_not_failures():
    _install(PLAIN)
    for fn in (check._chk_gateway, check._chk_tunnel):
        _, ok, ev, _ = fn()
        assert ok is None, fn.__name__
        assert ev == "not set up (optional: only needed to reach the model from other machines)"


def test_gateway_up_without_auth_is_still_a_failure():
    _install({**PLAIN, "ss -ltnp": _ss((8001, "llama-server", 5), (8002, "bun", 9)),
              "curl -s localhost:8002/v1/models": _MODELS % "m"})
    _, ok, ev, _ = check._chk_gateway()
    assert ok is False and "AUTH NOT ENFORCED" in ev


def test_scan_marks_optional_rows_in_ids():
    assert check._OPTIONAL == ("gateway", "tunnel")


def _run_check(mapping):
    import contextlib, io
    _install(mapping)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = check.run()
    return rc, out.getvalue()


def test_run_prints_dash_for_none_rows_and_does_not_count_them():
    rc, out = _run_check({**PLAIN, "pgrep -af llama-server": "5 /usr/bin/llama-server -ngl 99 --port 8001",
                          "which nvidia-smi": "/usr/bin/nvidia-smi", "curl -s localhost:8001/health": '{"status":"ok"}',
                          "models -maxdepth 2": "/home/x/models/m.gguf", "systemctl --user status 5": STATUS_LLAMA,
                          "systemctl --user is-enabled": "enabled"})
    plain = out.replace("\033[2m", "").replace("\033[0m", "").replace("\033[32m", "").replace("\033[31m", "")
    assert rc == 0
    assert "– Auth gateway" in plain and "– Cloudflare tunnel" in plain
    assert "left above" not in out and "all checks pass: the model is served and ready" in out
    assert "runbook" not in out.lower()
    private_names.assert_clean(out, "check output")


def test_run_summary_counts_only_real_failures():
    rc, out = _run_check({**PLAIN, "pgrep -af llama-server": "5 /usr/bin/llama-server --port 8001",
                          "curl -s localhost:8001/health": '{"status":"ok"}'})
    assert "Not done yet — 1 item(s) left above" in out or "Not done yet" in out
    n = out.count("fix:")
    assert f"{n} item(s) left above" in out
    assert "runbook" not in out.lower()
    private_names.assert_clean(out, "check output")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {e!r}")
        finally:
            teardown_function()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


# ── exit code: 0 only when every row passed, 1 when only warnings, 2 on any fail ──────────
def _run_with_rows(monkeypatch, rows):
    import contextlib, io
    monkeypatch.setattr(check, "scan", lambda: rows)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = check.run()
    return rc, out.getvalue()


def test_run_exit_codes_follow_the_readme(monkeypatch):
    ok = ("endpoint", "LLM endpoint serving", True, "serving :8001", "")
    warn = ("service", "Model service", True, "running ⚠ won't start after reboot", "enable it")
    bad = ("gpu", "Built with GPU accelerator", False, "running on CPU", "rebuild")
    none = ("gateway", "Auth gateway", None, "not set up", "")
    assert _run_with_rows(monkeypatch, [ok, none])[0] == 0
    assert _run_with_rows(monkeypatch, [ok, warn])[0] == 1
    assert _run_with_rows(monkeypatch, [ok, warn, bad])[0] == 2
    assert _run_with_rows(monkeypatch, [bad])[0] == 2


def test_warning_rows_show_their_fix_and_the_summary_counts_them_not_all_pass(monkeypatch):
    ok = ("endpoint", "LLM endpoint serving", True, "serving :8001", "")
    warn = ("gpu", "Built with GPU accelerator", True, "part of the model is on the CPU: m 48% CPU / 52% GPU ⚠",
            "use a smaller or more heavily quantised model")
    rc, out = _run_with_rows(monkeypatch, [ok, warn])
    assert rc == 1
    assert "use a smaller or more heavily quantised model" in out
    assert "all checks pass" not in out
    assert "1 warning" in out and "left above" not in out
    rc, out = _run_with_rows(monkeypatch, [ok, warn, warn])
    assert "2 warnings" in out


def test_clean_rows_still_say_all_checks_pass(monkeypatch):
    ok = ("endpoint", "LLM endpoint serving", True, "serving :8001", "")
    rc, out = _run_with_rows(monkeypatch, [ok])
    assert rc == 0 and "all checks pass" in out and "warning" not in out


def test_a_real_ollama_split_prints_its_fix_line(monkeypatch):
    _install(_ollama("big:70b  abc  40 GB  48%/52% CPU/GPU  4 minutes from now"))
    rows = [("gpu",) + check._chk_gpu()]
    rc, out = _run_with_rows(monkeypatch, rows)
    assert "48%" in out and "use a smaller or more heavily" in out and "all checks pass" not in out


def test_readme_documents_the_unreadable_row_for_check_and_watch():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    lines = [ln for ln in readme.splitlines() if ln.startswith("Exit codes:")]
    assert len(lines) == 2, lines
    for ln in lines:
        assert "?" in ln and "could not" in ln, ln


def test_check_summary_line_says_an_unreadable_row_exits_1(monkeypatch):
    ok = ("endpoint", "LLM endpoint serving", True, "serving :8001", "")
    unread = ("gpu", "Built with GPU accelerator", check.UNKNOWN, "can't read", "")
    rc, out = _run_with_rows(monkeypatch, [ok, unread])
    assert rc == 1 and "shown as ?" in out and "exit" in out.lower()


def test_a_stopped_ollama_gets_no_gguf_advice_by_hint_or_by_port(monkeypatch):
    # Ollama keeps its own model store, so "no .gguf found" is wrong advice even when it is
    # stopped. The stored unit (deneb watch's hint) or Ollama's default port says what it was.
    stopped = {"ss -ltnp": "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"}
    for port, unit in ((11434, "system:ollama.service"), (11434, ""), (9999, "ollama.service")):
        _install(stopped)
        check.set_hint(port, unit)
        try:
            label, ok, ev, fix = check._chk_model_file()
            assert ok is None and "Ollama manages its own model files" in ev, (port, unit, ev)
            assert fix == "" and ".gguf" not in ev
        finally:
            check.clear_hint()
            check._reset_port_cache()
    # nothing known about the old server: the .gguf advice stays
    _install(stopped)
    assert check._chk_model_file()[1] is False


def test_a_stopped_ollama_alert_text_has_no_gguf_fix(monkeypatch):
    _install({"ss -ltnp": "LISTEN\n"})
    check.set_hint(11434, "ollama.service")
    try:
        row = check._chk_model_file()
    finally:
        check.clear_hint()
        check._reset_port_cache()
    assert "deneb setup" not in " ".join(str(x) for x in row)


LOADING = '{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}'


def test_a_listening_server_still_loading_is_up_and_loading_not_not_found():
    _install({"ss -ltnp": _ss((8001, "llama-server", 5)),
              "curl -s localhost:8001/v1/models": LOADING, "curl -s localhost:8001/health": LOADING})
    label, ok, ev, fix = check._chk_endpoint()
    assert ok is False
    assert "loading" in ev.lower() and "not responding" not in ev
    assert "wait" in fix and "deneb check" in fix
    assert "no model server was found" not in fix and "could not find which service" not in fix


def test_a_dead_port_with_a_loading_looking_body_elsewhere_still_gets_the_stopped_fix():
    # nothing listens: no "still loading" claim, the stopped advice stays
    _install({"ss -ltnp": "LISTEN\n", "curl -s localhost:8001/v1/models": LOADING})
    _, ok, ev, fix = check._chk_endpoint()
    assert ok is False and "loading" not in ev.lower() and "no model server was found" in fix


# ── an empty box: no guess about another user, next steps are recommend then setup ────────
EMPTY = {"ss -ltnp": "State Recv-Q Send-Q Local Address:Port Peer Address:Port\nLISTEN 0 4096 127.0.0.1:631 0.0.0.0:*\n"}


def test_empty_box_has_no_another_users_guess_no_token_or_runbook_advice(monkeypatch):
    import contextlib, io
    _install(EMPTY)
    rows = check.scan()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = check.run()
    text = out.getvalue()
    assert rc == 2
    assert "another user" not in text
    assert "runbook" not in text
    assert 'ask deneb' not in text and 'deneb "' not in text      # a token-only command
    assert "<its-unit>" not in text
    assert "deneb recommend" in text and "deneb setup <model>" in text
    gpu = next(r for r in rows if r[0] == "gpu")
    assert gpu[2] == check.UNKNOWN and "no model server is running" in gpu[3]


def test_stopped_server_is_named_by_its_last_seen_port_and_unit():
    _install(EMPTY)
    check.set_hint(11434, "system:ollama.service")
    assert check.serving_port() == 11434
    _, ok, ev, fix = check._chk_endpoint()
    assert ok is False and ":11434" in ev
    assert "ollama.service" in fix and "<its-unit>" not in fix
    assert "sudo systemctl start ollama.service" in fix
    check.set_hint(None, "")


def test_stopped_server_with_no_known_unit_says_it_could_not_be_found():
    _install(EMPTY)
    check.set_hint(None, "")
    _, _, _, fix = check._chk_endpoint()
    assert "<its-unit>" not in fix and "could not find which service" in fix


def _readme_text():
    return (Path(__file__).resolve().parent.parent / "README.md").read_text()


def _code_blocks(text):
    parts = text.split("```")
    return [p for i, p in enumerate(parts) if i % 2 == 1]


def test_readme_check_sample_is_what_check_prints_for_that_box(monkeypatch):
    """The README sample is a stubbed Ollama box (a model loaded on the GPU, enabled system
    unit); every row in it, and its last line, must be printed by the real code for that box.
    (Idle, the GPU row is '?' and the run is not "all checks pass": 14 Oct QC F3.)"""
    import re
    box = _box(11434, "ollama", pid=77, model="qwen2.5:0.5b", **{
        "systemctl status 77": STATUS_OLLAMA, "systemctl is-enabled ollama.service": "enabled",
        "pgrep -af ollama": "77 /usr/local/bin/ollama serve",
        "ollama ps": "NAME ID SIZE PROCESSOR UNTIL\n"
                     "qwen2.5:0.5b  a8b0c5157701  397 MB  100% GPU  4 minutes from now\n"})
    rc, out = _run_check(box)
    assert rc == 0 and "all checks pass" in out
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    sample = next(b for b in _code_blocks(_readme_text()) if "LLM endpoint serving" in b)
    rows = [ln.rstrip() for ln in sample.splitlines() if ln.strip()[:1] in ("✓", "–", "✗", "⚠", "?")]
    assert len(rows) >= 6, rows
    printed = [ln.rstrip() for ln in plain.splitlines()]
    for ln in rows:
        assert ln in printed, f"README sample row not printed by check: {ln!r}"


def test_no_readme_code_block_both_prints_the_timer_undo_and_uninstalls():
    for block in _code_blocks(_readme_text()):
        assert not ("watch --unschedule" in block and "pipx uninstall" in block), block


# ── r4/R2: Ollama installed but stopped is not a llama.cpp build ─────────────────────────
def _run_check_text():
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = check.run()
    return rc, out.getvalue()


def _ollama_installed_as(monkeypatch, how):
    monkeypatch.setattr(check, "_ollama_install", lambda: how)
    check._reset_port_cache()


def test_stopped_ollama_install_names_11434_and_says_start_it_not_build_llama_cpp(monkeypatch):
    for how, start in (("system", "sudo systemctl start ollama"), ("binary", "ollama serve")):
        _install(EMPTY)
        _ollama_installed_as(monkeypatch, how)
        rc, text = _run_check_text()
        assert rc == 2
        assert "LLM endpoint serving (:11434)" in text and ":8001" not in text
        assert "installed but not running" in text and start in text, text
        assert "Ollama manages its own model files; not checked" in text
        assert ".gguf" not in text
        assert "deneb recommend" not in text and "deneb setup" not in text
        assert "No model server was found" not in text


def test_a_user_unit_for_ollama_is_started_with_systemctl_user(monkeypatch):
    _install(EMPTY)
    _ollama_installed_as(monkeypatch, "user")
    assert "systemctl --user start ollama" in check._chk_endpoint()[3]


def test_no_ollama_and_nothing_listening_keeps_todays_output(monkeypatch):
    _install(EMPTY)
    _ollama_installed_as(monkeypatch, "")
    rc, text = _run_check_text()
    assert rc == 2
    assert "LLM endpoint serving (:8001)" in text
    assert "no .gguf found" in text
    assert "No model server was found. Next: deneb recommend, then deneb setup <model>." in text


def test_a_running_ollama_is_still_found_on_11434_when_installed(monkeypatch):
    _install({"ss -ltnp": 'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=77,fd=3))\n',
              "curl -s localhost:11434/v1/models": '{"object":"list","data":[{"id":"llama3","object":"model"}]}'})
    _ollama_installed_as(monkeypatch, "system")
    assert check.serving_port() == 11434
    assert check._chk_endpoint()[1] is True


def test_our_own_llama_server_unit_wins_over_an_ollama_install(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_own_llama_unit_exists", _real_own_unit)
    unit = tmp_path / ".config" / "systemd" / "user" / "llama-server.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(EMPTY)
    _ollama_installed_as(monkeypatch, "system")
    assert check.serving_port() == 8001


def test_a_stored_hint_still_beats_an_ollama_install(monkeypatch):
    _install(EMPTY)
    _ollama_installed_as(monkeypatch, "system")
    check.set_hint(8080, "llm.service")
    try:
        assert check.serving_port() == 8080
    finally:
        check.clear_hint()


def test_ollama_install_detection_reads_unit_dirs_and_path_only(monkeypatch, tmp_path):
    import shutil
    check_real = _real_ollama_install
    monkeypatch.setattr(shutil, "which", lambda n: None)
    sysdir, userdir = tmp_path / "sys", tmp_path / "usr"
    monkeypatch.setattr(check, "_SYSTEM_UNIT_DIRS", (str(sysdir),), raising=False)
    monkeypatch.setattr(check, "_USER_UNIT_DIRS", (str(userdir),), raising=False)
    assert check_real() == ""
    userdir.mkdir(); (userdir / "ollama.service").write_text("x")
    assert check_real() == "user"
    sysdir.mkdir(); (sysdir / "ollama.service").write_text("x")
    assert check_real() == "system"
    (sysdir / "ollama.service").unlink(); (userdir / "ollama.service").unlink()
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/ollama" if n == "ollama" else None)
    assert check_real() == "binary"
