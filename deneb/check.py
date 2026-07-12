"""deneb.check — the DETERMINISTIC "am I done?" scan.

Not everything needs the LLM. This runs a FIXED sequence of read-only probes (via the
same allowlisted tools), evaluates each Neo component IN CODE by PORT + unit-PATTERN (not
guessed names), and prints a ✓/✗ checklist + verdict. Fast (~seconds), reliable, and
immune to the agentic-loop failure modes (no JSON leaks, no hallucinated state).

Hybrid: for a component that doesn't match the deterministic probes, or for a deeper
"why is X failing", the user runs `deneb "<question>"` which uses the LLM. So: code for the
routine 90%, LLM on demand for the weird 10%.
"""
from __future__ import annotations

import re

from . import tools, ui

_C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "d": "\033[2m", "z": "\033[0m",
      "b": "\033[1m", "teal": "\033[38;5;44m"}


def _probe(cmd: str) -> str:
    out = tools.run(cmd).get("output", "")
    return "" if out.startswith("[refused]") else out


def _read(path: str) -> str:
    return tools.read_file(path).get("output", "")


def _has(cmd: str) -> bool:
    """True only if `cmd` resolves to a real path — `which` prints 'no <cmd> in ...' on a
    miss, so a substring test would false-match; require a leading '/'."""
    return _probe(f"which {cmd}").strip().startswith("/")


def _port_owner(port: int) -> tuple[str, str]:
    """(pid, procname) of whatever is LISTENING on `port`, from `ss -ltnp`. Detection is by
    PORT + owning PROCESS — never a guessed service filename (unit names vary per box).
    Returns ('','') if nothing listens; ('?','') if something listens but the owner is hidden
    (a process we don't own — e.g. a root service seen by a non-root user)."""
    for line in _probe("ss -ltnp").splitlines():
        if f":{port} " not in line:
            continue
        m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', line)
        return (m.group(2), m.group(1)) if m else ("?", "")
    return "", ""


def _unit_of_pid(pid: str) -> str:
    """Resolve the systemd --user unit owning `pid` — from the PID, not a name guess. '' if
    it can't be mapped (e.g. a bare process launched outside systemd)."""
    if not pid or pid == "?":
        return ""
    m = re.search(r"([\w@.\-]+\.service)", _probe(f"systemctl --user status {pid} --no-pager"))
    return m.group(1) if m else ""


# Each check returns (label, ok, evidence, fix_if_not_ok)
def _chk_endpoint():
    h = _probe("curl -s localhost:8001/health")
    ok = bool(h) and ("ok" in h.lower() or "status" in h.lower())
    model = ""
    if ok:
        mv = _probe("curl -s localhost:8001/v1/models")
        m = re.search(r'"id"\s*:\s*"([^"]+)"', mv)
        if m:
            model = m.group(1).split("/")[-1]
    ev = (f"serving {model}" if model else "serving") if ok else "not responding on :8001"
    return ("LLM endpoint serving (:8001)", ok, ev,
            "the model server isn't up — start it and, if it fails, run "
            "`journalctl --user -u <llama-unit> -e` for the reason")


def _chk_service():
    # Detect by who owns :8001 (the serving process), then resolve its unit FROM the pid.
    pid, proc = _port_owner(8001)
    if not pid:
        return ("Model service enabled + running", False, "nothing is listening on :8001",
                "start the model server (its systemd unit or launch script); if it dies at "
                "boot, `journalctl --user -u <its-unit> -e` shows why")
    unit = _unit_of_pid(pid)
    who = unit or (f"{proc} (pid {pid})" if proc and proc != "?" else f"pid {pid}")
    at_boot = _probe(f"systemctl --user is-enabled {unit}").strip() if unit else ""
    if at_boot == "disabled":
        return ("Model service enabled + running", True,
                f"{who} serving :8001 ⚠ (not enabled — won't auto-start after reboot)",
                f"`systemctl --user enable {unit}` so it survives a reboot")
    boot = f", {at_boot} at boot" if at_boot in ("enabled", "static", "indirect") else ""
    return ("Model service enabled + running", True, f"{who} serving :8001{boot}", "")


def _chk_gpu():
    # Inspect the ACTUAL serving process's cmdline for GPU-offload flags — using the proc
    # name that owns :8001, not a hardcoded "llama-server".
    _, proc = _port_owner(8001)
    ps = _probe(f"pgrep -af {proc}") if proc and proc != "?" else (
        _probe("pgrep -af llama") or _probe("pgrep -af ollama") or _probe("pgrep -af vllm"))
    ngl = any(f in ps for f in ("-ngl", "--gpu-layers", "n_gpu_layers", "--n-gpu-layers"))
    accel = "cuda" if _has("nvidia-smi") else ("vulkan/rocm" if _has("rocminfo") else "")
    ok = ngl and bool(accel)
    ev = (f"GPU offload on, {accel}") if ok else ("running on CPU (no GPU-offload flag)" if not ngl else "no GPU toolkit found")
    return ("Built with GPU accelerator (not CPU)", ok, ev,
            "rebuild llama.cpp with the accelerator (CUDA: -DGGML_CUDA=ON; ROCm/Vulkan on "
            "Strix) and run with -ngl 99")


def _chk_model_file():
    mv = _probe("curl -s localhost:8001/v1/models")
    m = re.search(r'"id"\s*:\s*"(/[^"]+\.gguf)"', mv)
    if m and "does not exist" not in _read(m.group(1)):
        return ("Model file present", True, m.group(1).split("/")[-1], "")
    found = _probe("find ~/models -maxdepth 2 -name '*.gguf'")
    ok = ".gguf" in found
    return ("Model file present", ok, (found.splitlines()[0].split("/")[-1] if ok else "no .gguf found"),
            "download the model .gguf into your models directory (see runbook §Download Model)")


def _chk_gateway():
    # Who owns :8002 (behavioural), then its unit from the pid — no filename guessing.
    pid, proc = _port_owner(8002)
    up = bool(pid)
    who = _unit_of_pid(pid) or (f"{proc} on :8002" if proc and proc != "?" else ":8002")
    # Auth: an UNauthenticated request must be rejected. (Plain GET only — the allowlist
    # blocks -o/-w/-X, so read the body and look for a rejection instead of a status code.)
    body = _probe("curl -s localhost:8002/v1/models").lower()
    rejected = any(w in body for w in ("unauthorized", "forbidden", "missing", "invalid",
                                       "api key", "api_key", "authenticat", "401", "403"))
    served = '"id"' in body or '"data"' in body
    if not up:
        return ("Auth gateway up (:8002, key + rate-limit)", False, "nothing listening on :8002",
                "stand up the bearer-key gateway on :8002 in front of :8001 (never expose :8001)")
    if served and not rejected:
        return ("Auth gateway up (:8002, key + rate-limit)", False,
                f"{who} up but ⚠ AUTH NOT ENFORCED (served without a key)",
                "the gateway is not requiring an API key — enable bearer-key auth before use")
    ev = f"{who}, auth enforced" if rejected else f"{who} up"
    return ("Auth gateway up (:8002, key + rate-limit)", True, ev,
            "stand up the bearer-key gateway on :8002 in front of :8001 (never expose :8001)")


def _chk_tunnel():
    # cloudflared is a real binary — detect the RUNNING PROCESS, not a guessed unit name.
    procs = _probe("pgrep -af cloudflared").strip()
    ok = bool(procs)
    m = re.match(r"\s*(\d+)", procs)
    who = (_unit_of_pid(m.group(1)) if m else "") or "cloudflared"
    return ("Cloudflare tunnel live", ok, (f"{who} running" if ok else "no tunnel (cloudflared not running)"),
            "install + run cloudflared to expose the gateway outbound-only (no inbound ports)")


_CHECKS = [_chk_endpoint, _chk_service, _chk_gpu, _chk_model_file, _chk_gateway, _chk_tunnel]


def run() -> int:
    ui.banner()
    print(f"{_C['d']}running a read-only scan of your Neo stack…{_C['z']}")
    rows = []
    for fn in _CHECKS:
        try:
            rows.append(fn())
        except Exception as e:  # noqa: BLE001 — a probe error is a ✗, never a crash
            rows.append((fn.__name__, False, f"(check error: {e})", "re-run, or ask deneb directly"))
    print()
    all_ok = True
    for label, ok, ev, fix in rows:
        mark = f"{_C['g']}✓{_C['z']}" if ok else f"{_C['r']}✗{_C['z']}"
        print(f"  {mark} {label}  {_C['d']}— {ev}{_C['z']}")
        if not ok:
            all_ok = False
            print(f"      {_C['y']}fix:{_C['z']} {fix}")
    print()
    if all_ok:
        print(f"{_C['teal']}{_C['b']}✓ Done — this box has reached the working Neo Altronis state.{_C['z']}")
    else:
        n = sum(1 for r in rows if not r[1])
        print(f"{_C['y']}{_C['b']}Not done yet — {n} item(s) left above.{_C['z']} "
              f"{_C['d']}for a deeper look at any one, ask: deneb \"why is <that> failing\"{_C['z']}")
    return 0
