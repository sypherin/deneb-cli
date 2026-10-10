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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import check  # noqa: E402

SS = (
    'LISTEN 0 512  127.0.0.1:8002  0.0.0.0:*  users:(("bun",pid=2892,fd=11))\n'
    'LISTEN 0 512    0.0.0.0:8001  0.0.0.0:*  users:(("llama-server",pid=876334,fd=10))\n'
)
STATUS_GATEWAY = "● neo-gateway.service - neo.altronis.sg secure inference gateway (Bun)\n  Loaded: loaded"
STATUS_LLAMA = "● llama-server.service - llama-server (Vulkan - Strix Halo - Qwen 3.6)\n  Loaded: loaded"


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
    assert check._unit_of_pid("2892") == "neo-gateway.service"
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
    "pgrep -af cloudflared": "999 cloudflared tunnel run neo",
    "curl -s localhost:8001/health": '{"status":"ok"}',
    "curl -s localhost:8001/v1/models": '{"data":[{"id":"/models/Qwen3.6.gguf"}]}',
    "models -maxdepth 2": "/home/x/models/Qwen3.6.gguf",
}


def test_gpu_is_rocm_not_cuda_on_amd():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_gpu()
    assert ok is True
    assert "vulkan/rocm" in ev and "cuda" not in ev  # the regression this test guards


def test_gateway_names_real_unit_and_auth_enforced():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_gateway()
    assert ok is True
    assert "neo-gateway.service" in ev        # resolved via PID, not a filename glob
    assert "auth enforced" in ev
    assert "goose" not in ev                   # must not grab an unrelated *gateway* unit


def test_service_reports_running_and_enabled():
    _install(HEALTHY)
    label, ok, ev, _ = check._chk_service()
    assert ok is True
    assert "llama-server.service" in ev and "enabled" in ev


def test_tunnel_up():
    _install(HEALTHY)
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is True and "running" in ev


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
    assert ok is False and "8002" in ev


def test_broken_tunnel_absent():
    _install(BROKEN)
    _, ok, ev, _ = check._chk_tunnel()
    assert ok is False and "no tunnel" in ev


def test_disabled_service_still_running_but_warned():
    _install(BROKEN)
    _, ok, ev, fix = check._chk_service()
    assert ok is True                      # it IS serving :8001
    assert "won't auto-start" in ev and "enable" in fix  # but flag the reboot risk


# ── a real Strix box (2026-10-10): bundled ROCm build, no toolkit on PATH, model alias ──
STRIX_SS = 'LISTEN 0 512  127.0.0.1:8001  0.0.0.0:*  users:(("llama-server",pid=1925615,fd=11))\n'
STRIX_MODEL = "/mnt/m/qwen3.8-27b-fp4/Qwen3.8-27B-ROCmFP4-FAST.gguf"
STRIX_PGREP = (
    "889803 /home/x/llama.cpp/build/bin/llama-server -m /home/x/models/judge-Q8_0.gguf --port 8005 -ngl 99\n"
    f"1925591 /bin/bash /home/x/bin/model-load-guard.sh --need 30 /mnt/q/bin/llama-server -m {STRIX_MODEL}\n"
    f"1925615 /mnt/q/bin/llama-server -m {STRIX_MODEL} -dev ROCm0 -ngl 99 -np 1 -c 131072\n"
    "1958540 /home/x/llama.cpp-llmjp/build/bin/llama-server -m /mnt/m/llm-jp.gguf --port 8056 -ngl 999\n"
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
    # A judge on :8005 has -ngl; the :8001 server doesn't. The old any-line test passed it.
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
    assert ok is False and "couldn't see" in ev


def test_models_dir_fallback_expands_home():
    seen = []
    check._probe = lambda c: seen.append(c) or ""
    check._read = lambda p: ""
    check._chk_model_file()
    assert any(c.startswith("find /") and c.endswith("models -maxdepth 2 -name '*.gguf'") for c in seen), seen


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
