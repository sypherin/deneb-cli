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
    "find ~/models": "/home/x/models/Qwen3.6.gguf",
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
    "find ~/models": "/home/x/models/Qwen3.6.gguf",
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
