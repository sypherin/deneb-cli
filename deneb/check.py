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

import os
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


def _mnt_ns(pid: str) -> str:
    """The mount namespace of `pid` ('self' = deneb's own), or '' when it can't be read
    (another user's process)."""
    ns = _probe(f"readlink /proc/{pid}/ns/mnt").strip()
    return ns if ns.startswith("mnt:") else ""


def _where(pid: str) -> str:
    """'host' or 'container' for a process, '' when unknown. A container's paths are its
    own: /models/x.gguf inside it is not /models/x.gguf on the host."""
    me, it = _mnt_ns("self"), _mnt_ns(pid) if pid else ""
    if not (me and it):
        return ""
    return "host" if me == it else "container"


# What owns a host port when the model runs in Docker or podman: the runtime's port
# forwarder, not the model server. The server is another process, inside the container.
_PORT_PROXIES = {"rootlessport", "rootlesskit", "docker-proxy", "conmon", "pasta", "slirp4netns"}
_SHELLS = {"bash", "sh", "dash", "zsh", "env", "timeout"}
# The runtime's own processes: rootless podman runs in a namespace of its own, so
# "not deneb's mount namespace" alone would pick `podman run` itself.
_RUNTIMES = {"podman", "docker", "nerdctl", "conmon", "crun", "runc", "catatonit",
             "containerd-shim", "containerd-shim-runc-v2"}


def _container_server() -> str:
    """The model server inside a container: a process whose args name a .gguf (or a vllm /
    ollama server) and that runs in a different mount namespace from deneb. Targeted
    pgreps, not one `ps` of every process: tool output is capped at 8000 chars."""
    found = []
    for pat in (r"\.gguf", "vllm", "ollama"):
        for ln in _probe(f"pgrep -af '{pat}'").splitlines():
            ln = ln.strip()
            if (re.match(r"\d+\s", ln) and ln not in found
                    and os.path.basename(ln.split()[1]) not in _RUNTIMES | _PORT_PROXIES
                    and _where(ln.split()[0]) == "container"):
                found.append(ln)
        if found:
            break
    found.sort(key=lambda ln: os.path.basename(ln.split()[1]) in _SHELLS)  # wrapper scripts last
    return found[0] if found else ""


def _serving_cmdline() -> str:
    """The command line of the process that owns :8001, picked by PID. A box often runs
    several llama-servers (a judge on :8005, a second model on :8056, a wrapper script
    whose args name the binary), so the first pgrep line is not reliably the one serving."""
    pid, proc = _port_owner(8001)
    if proc in _PORT_PROXIES:
        return _container_server()
    ps = _probe(f"pgrep -af {proc}") if proc and proc != "?" else (
        _probe("pgrep -af llama") or _probe("pgrep -af ollama") or _probe("pgrep -af vllm"))
    lines = [ln.strip() for ln in ps.splitlines() if re.match(r"\s*\d+\s", ln)]
    for ln in lines:
        if pid and ln.split(None, 1)[0] == pid:
            return ln
    for ln in lines:
        if re.search(r"--port[ =]8001(?!\d)", ln):
            return ln
    return lines[0] if lines else ""


# Libraries that show which accelerator a llama.cpp build uses. A box can serve from a
# bundled build with no rocminfo / nvidia-smi on PATH, so the toolkit test alone gave a
# false "CPU" on a GPU box (2026-10-10, Strix serving Qwen3.8-27B on ROCm).
_ACCEL_LIBS = (("cuda", ("libcudart", "libcublas", "libggml-cuda")),
               ("rocm", ("libamdhip64", "libhipblas", "librocblas", "libggml-hip")),
               ("vulkan", ("libvulkan", "libggml-vulkan")))


def _accel_of_binary(cmdline: str) -> str:
    """Accelerators the serving binary is built with: its linked libraries (ldd lists them
    even when they resolve "not found" outside the service's environment), plus the ggml
    backend libraries beside it, which GGML_BACKEND_DL builds load at runtime."""
    parts = cmdline.split()
    binary = parts[1] if len(parts) > 1 else ""
    if not binary.startswith("/"):
        return ""
    libs = _probe(f"ldd {binary}") + "\n" + _probe(
        f"find {binary.rsplit('/', 1)[0]} -maxdepth 1 -name libggml-*")
    return "+".join(name for name, keys in _ACCEL_LIBS if any(k in libs for k in keys))


def _accel_loaded(pid: str) -> str:
    """Accelerator runtimes the serving process has actually loaded, from its memory map.
    Works for a server inside a container, where the binary path is not a host path."""
    if not pid:
        return ""
    keys = " ".join(f"-e {k}" for _, ks in _ACCEL_LIBS for k in ks)
    libs = _probe(f"grep -o {keys} /proc/{pid}/maps")
    return "+".join(name for name, ks in _ACCEL_LIBS if any(k in libs for k in ks))


# The offload flag (not the draft-model ones: -ngld, --spec-draft-ngl, --gpu-layers-draft).
_NGL = re.compile(r"(?<![\w-])(?:-ngl|--gpu-layers|--n-gpu-layers|n_gpu_layers)(?![\w-])"
                  r"(?:[ =:]+(-?\d+|auto|all))?")


def _chk_gpu():
    # Inspect the ACTUAL serving process (the one behind :8001, not a hardcoded
    # "llama-server"): its offload flag, and the accelerator it loaded or links.
    label = "Built with GPU accelerator (not CPU)"
    ps = _serving_cmdline()
    if not ps:
        return (label, False, "couldn't see the process serving :8001 (another user's?)",
                "ask deneb \"which process serves :8001 and is it on the GPU\" "
                "(it can't read a process another user owns)")
    pid, binary = ps.split()[0], (ps.split() + [""])[1]
    loaded = _accel_loaded(pid)
    accel = loaded or _accel_of_binary(ps) or (
        "cuda" if _has("nvidia-smi") else ("vulkan/rocm" if _has("rocminfo") else ""))
    m = _NGL.search(ps)
    llama = os.path.basename(binary).startswith("llama")
    if m:
        offload = m.group(1) not in ("0", "-0")          # -ngl 0 = every layer on CPU
    else:
        # vLLM and custom engines have no flag: on the GPU when their process has the GPU
        # runtime loaded. llama.cpp without -ngl is left to the build's default, which is
        # "auto" (offload) on current builds and 0 (CPU) on older ones, so it can't be
        # read from here.
        offload = bool(loaded) and not llama
    ok = offload and bool(accel)
    if ok:
        ev = f"GPU offload on, {accel}" if m else f"{os.path.basename(binary)} has the {accel} runtime loaded"
    elif m and not offload:
        ev = "running on CPU (-ngl 0)"
    elif not m and loaded and llama:
        ev = "no -ngl flag: offload is left to the build default (auto on current llama.cpp, CPU on older)"
    else:
        ev = "running on CPU (no GPU-offload flag)" if not offload else "no GPU toolkit found"
    if accel and not offload:
        fix = "start the server with -ngl 99 (all layers on the GPU), not the build's default"
    else:
        fix = ("rebuild llama.cpp with the accelerator (CUDA: -DGGML_CUDA=ON; ROCm/Vulkan on "
               "Strix) and run with -ngl 99")
    return (label, ok, ev, fix)


def _gib(nbytes: int) -> str:
    return f"{nbytes / 2**30:.1f} GiB"


def _chk_model_file():
    # 1. The path the serving process was started with (-m / --model). /v1/models often
    #    reports an alias ("qwen3.8-27b"), not a path, so it can't be the only source.
    #    Servers that don't take -m (custom engines) still name the .gguf somewhere.
    ps = _serving_cmdline()
    m = (re.search(r"\s(?:-m|--model)[ =](/\S+)", " " + ps)
         or re.search(r"\s(/\S+\.gguf)(?!\S)", " " + ps))
    if m:
        path, pid = m.group(1), ps.split()[0]
        where = _where(pid)
        # A container's /models/x.gguf is only reachable from the host through the
        # process's own root.
        st = _probe(f"stat -L -c %s {f'/proc/{pid}/root{path}' if where == 'container' else path}").strip()
        if st.isdigit():
            return ("Model file present", True, f"{path.split('/')[-1]} ({_gib(int(st))})"
                    + (", in a container" if where == "container" else ""), "")
        # "Deleted" only when we know which filesystem the path lives in.
        if "No such file" in st and where:
            return ("Model file present", False,
                    f"{path.split('/')[-1]} is loaded but no longer on disk — the server "
                    "will fail on its next restart",
                    f"put the model back at {path} (or point -m at where it is now)")
    # 2. /v1/models naming an absolute .gguf path.
    mv = _probe("curl -s localhost:8001/v1/models")
    m = re.search(r'"id"\s*:\s*"(/[^"]+\.gguf)"', mv)
    if m and "does not exist" not in _read(m.group(1)):
        return ("Model file present", True, m.group(1).split("/")[-1], "")
    # 3. The conventional models directory. Expanded here: commands run without a shell,
    #    so a literal "~/models" was never found.
    found = _probe(f"find {os.path.expanduser('~/models')} -maxdepth 2 -name '*.gguf'")
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
