"""deneb.check — the DETERMINISTIC "am I done?" scan.

Not everything needs the LLM. This runs a FIXED sequence of read-only probes (via the
same allowlisted tools), evaluates each part of the local-LLM stack IN CODE by PORT + unit-PATTERN (not
guessed names), and prints a ✓/✗ checklist + verdict. Fast (~seconds), reliable, and
immune to the agentic-loop failure modes (no JSON leaks, no hallucinated state).

Hybrid: for a component that doesn't match the deterministic probes, or for a deeper
"why is X failing", the user runs `deneb "<question>"` which uses the LLM. So: code for the
routine 90%, LLM on demand for the weird 10%.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil

from . import config, net, tools, ui
from .colour import Palette

_C = Palette(g="green", r="red", y="amber", d="dim", z="reset", b="bold", teal="teal")


def _probe(cmd: str) -> str:
    out = tools.run(cmd).get("output", "")
    return "" if out.startswith("[refused]") else out


def _read(path: str) -> str:
    return tools.read_file(path).get("output", "")


def _curl(port: int, path: str) -> str:
    """A plain GET to localhost. --max-time 3: a listener that accepts and never answers must
    cost seconds, not the tool's 25 s wait on every probe. (The option goes after the URL so
    the command still starts with the plain GET.) A box without curl (a fresh Ubuntu desktop)
    gets the same GET from Python, with the same output: "(no output)" when nothing answers."""
    out = _probe(f"curl -s localhost:{port}{path} --max-time 3")
    if out.startswith("(not installed on this box: curl"):
        body = (net.local_get(port, path) or "").strip()
        out = tools._redact(body) if body else "(no output)"
    return out


def _has(cmd: str) -> bool:
    """True only if `cmd` resolves to a real path — `which` prints 'no <cmd> in ...' on a
    miss, so a substring test would false-match; require a leading '/'."""
    return _probe(f"which {cmd}").strip().startswith("/")


def _owner_in(ss_text: str, port: int) -> tuple[str, str]:
    """(pid, procname) of whatever LISTENS on `port` in `ss -ltnp` output."""
    for line in ss_text.splitlines():
        if f":{port} " not in line:
            continue
        m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', line)
        return (m.group(2), m.group(1)) if m else ("?", "")
    return "", ""


def _port_owner(port: int) -> tuple[str, str]:
    """(pid, procname) of whatever is LISTENING on `port`, from `ss -ltnp`. Detection is by
    PORT + owning PROCESS — never a guessed service filename (unit names vary per box).
    Returns ('','') if nothing listens; ('?','') if something listens but the owner is hidden
    (a process we don't own — e.g. a root service seen by a non-root user). Without `ss`
    (macOS, a minimal container) the owner can't be read at all: a port that answers
    /v1/models is then reported as ('?', '') and one that doesn't as nothing."""
    ss = _probe("ss -ltnp")
    if "LISTEN" in ss:
        return _owner_in(ss, port)
    return ("?", "") if _models_json(_curl(port, "/v1/models")) else ("", "")


# ── which port is the model served on? ──────────────────────────────────────────
# 8001 is the Altronis layout and what `deneb setup` prints. The others are the defaults of
# llama-server (8080), vLLM (8000), Ollama (11434) and LM Studio (1234).
DEFAULT_PORT = 8001
_PORT_CANDIDATES = (8001, 8080, 8000, 11434, 1234)
_port_cache = None          # (port, warning), filled once per process
UNKNOWN = "?"               # a row's ok value when the thing exists but could not be read
_hint = {"port": None, "unit": ""}   # where a server was last seen (deneb watch's state)
_kind_cache = {}            # port -> server kind read from its HTTP API
_inferred = {"ollama": ""}  # set when nothing answered and Ollama is installed: how (see _ollama_install)


def _reset_port_cache() -> None:
    """Forget the resolved port (for tests; a real process resolves it once)."""
    global _port_cache
    _port_cache = None
    _kind_cache.clear()
    _inferred["ollama"] = ""


def set_hint(port=None, unit="") -> None:
    """Remember where the model server was last seen (a port, and a unit such as
    "ollama.service" or "system:ollama.service"). Used only when nothing answers now, so a
    stopped server is reported on ITS port and unit, not on the default."""
    ok = isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
    _hint["port"] = port if ok else None
    _hint["unit"] = unit if isinstance(unit, str) else ""
    _reset_port_cache()


def clear_hint() -> None:
    """Forget the hint without touching the resolved port (watch has already read it)."""
    _hint["port"], _hint["unit"] = None, ""


def hinted_unit() -> tuple:
    """(name, scope) of the unit the server was last seen under ('', '' when none is stored):
    scope 'system' for a stored "system:<name>", else 'user'."""
    unit = _hint["unit"]
    if not unit:
        return "", ""
    return (unit.split(":", 1)[1], "system") if unit.startswith("system:") else (unit, "user")


_SYSTEM_UNIT_DIRS = ("/etc/systemd/system", "/usr/lib/systemd/system", "/lib/systemd/system")
_USER_UNIT_DIRS = ("~/.config/systemd/user", "/etc/systemd/user", "/usr/lib/systemd/user")


def _ollama_install() -> str:
    """How Ollama is installed on this box, read-only (no process is run): "system" or "user"
    when an ollama.service unit file is in the standard unit directories, "binary" when only
    the `ollama` program is on PATH, "" when neither."""
    def has(dirs):
        return any(os.path.isfile(os.path.join(os.path.expanduser(d), "ollama.service")) for d in dirs)
    if has(_SYSTEM_UNIT_DIRS):
        return "system"
    if has(_USER_UNIT_DIRS):
        return "user"
    return "binary" if shutil.which("ollama") else ""


def _own_llama_unit_exists() -> bool:
    """True when a user unit that `deneb setup` writes is on disk: llama-server.service, or
    vllm-server.service from the vLLM route."""
    return any(os.path.isfile(os.path.expanduser(f"~/.config/systemd/user/{u}"))
               for u in ("llama-server.service", "vllm-server.service"))


# Unit files a person wrote (not the ones a package ships), and the programs that make one a
# model server. Read as files: no process is run.
_WRITTEN_UNIT_DIRS = (("~/.config/systemd/user", "user"), ("/etc/systemd/system", "system"))
_SERVER_EXEC = re.compile(r"^ExecStart=.*(llama-server|llama_cpp\.server|vllm|ollama serve)", re.M)


_VLLM_EXEC = re.compile(r"^ExecStart=.*\bvllm(?:\s+serve|\.entrypoints)", re.M)


def _enabled(unit_dir: str, name: str) -> bool:
    """`systemctl enable` left a link to `name` in one of the unit dir's *.wants directories."""
    try:
        wants = [w for w in os.listdir(unit_dir) if w.endswith(".wants")]
    except OSError:
        return False
    return any(os.path.lexists(os.path.join(unit_dir, w, name)) for w in wants)


def _server_unit_on_disk() -> tuple:
    """(name, scope) of a unit file whose ExecStart runs a model server, ('', '') when none.
    How plain `deneb check` names a stopped server: it has no record of where one was seen
    (only `deneb watch` keeps one), so it looks for the unit that would start it. With more than
    one (a llama.cpp and a vLLM unit, both from deneb setup), the enabled one wins, then the
    newest: deneb's setup disables the other when it switches engines."""
    found = []
    for order, (d, scope) in enumerate(_WRITTEN_UNIT_DIRS):
        unit_dir = os.path.expanduser(d)
        try:
            names = sorted(os.listdir(unit_dir))
        except OSError:
            continue
        for n in names:
            if not n.endswith(".service"):
                continue
            path = os.path.join(unit_dir, n)
            try:
                with open(path, errors="replace") as f:
                    if _SERVER_EXEC.search(_joined(f.read(65536))):
                        found.append((not _enabled(unit_dir, n), -os.path.getmtime(path), order, n, scope))
            except OSError:
                continue
    if not found:
        return "", ""
    best = min(found)
    return best[3], best[4]


def _joined(text: str) -> str:
    """Unit text with backslash-continued lines joined, so an ExecStart= split over lines reads
    as the one command systemd runs."""
    return re.sub(r"\\\n\s*", " ", text)


def _unit_text() -> tuple:
    """(name, scope, text) of the unit that starts the model server: the one `deneb watch` saw,
    else one on disk. Text is "" when there is none or it can't be read; continued lines are
    joined."""
    name, scope = hinted_unit()
    if not name:
        name, scope = _server_unit_on_disk()
    for d in ([d for d, s in _WRITTEN_UNIT_DIRS if s == scope] if name else []):
        try:
            with open(os.path.join(os.path.expanduser(d), name), errors="replace") as f:
                return name, scope, _joined(f.read(65536))
        except OSError:
            continue
    return name, scope, ""


def _stopped_fix() -> str:
    """What to do when no server answers: name the last-seen unit, or say it was not found.
    Never a placeholder."""
    serving_port()
    unit = _hint["unit"]
    if not unit and not _inferred["ollama"]:
        name, scope = _server_unit_on_disk()
        unit = (f"system:{name}" if scope == "system" else name) if name else ""
    if not unit and _inferred["ollama"]:
        how = _inferred["ollama"]
        start = {"system": "`sudo systemctl start ollama`",
                 "user": "`systemctl --user start ollama`"}.get(how, "`ollama serve`")
        return (f"Ollama is installed but not running: start it with {start}, then re-run "
                "`deneb check`")
    if unit:
        system = unit.startswith("system:")
        name = unit.split(":", 1)[-1]
        scope = "" if system else "--user "
        sudo = "sudo " if system else ""
        return (f"start it again: `{sudo}systemctl {scope}start {name}`; if it fails, read its "
                f"log: `journalctl {scope}-u {name} -e`")
    return ("no model server was found. If none is set up yet, run `deneb recommend`, then "
            "`deneb setup <model>`; if one was running, start it and read its log "
            "(deneb could not find which service runs it)")


def resolved_port():
    """The port serving_port() has resolved in this process, or None if it hasn't run yet.
    Never probes: `deneb watch` stores it after the scan to notice a server that moved."""
    return _port_cache[0] if _port_cache else None


def _models_json(text: str) -> bool:
    """True when `text` is the JSON body of /v1/models: an object with a "data" key. The tool
    output is capped at 8000 characters, so a long model list can arrive cut off: a body that
    still opens like the JSON object and names "data" counts too."""
    t = text.strip()
    try:
        obj = json.loads(t)
        return isinstance(obj, dict) and "data" in obj
    except ValueError:
        return t.startswith("{") and '"data"' in t


def _models_list(text: str):
    """(is_models_list, has_model_entry) for a /v1/models body: an object whose "data" is a
    LIST, and whether any entry says "object": "model" (what real servers send). A body cut
    off by the tool's 8000-character cap still counts if it opens that way."""
    t = text.strip()
    try:
        obj = json.loads(t)
    except ValueError:
        if t.startswith("{") and re.search(r'"data"\s*:\s*\[', t):
            return True, bool(re.search(r'"object"\s*:\s*"model"', t))
        return False, False
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, list):
        return False, False
    return True, any(isinstance(e, dict) and e.get("object") == "model" for e in data)


def _health_ok(text: str) -> bool:
    t = text.strip()
    if not t or t.startswith(("[stderr]", "(")):    # nothing came back, or a tool message
        return False
    return "ok" in t.lower() or "status" in t.lower()


def _is_known_server(proc: str, pid: str) -> bool:
    """llama-server, vLLM (often `python -m vllm...`), Ollama or LM Studio, by process name."""
    n = proc.lower()
    if n == "lms" or any(k in n for k in ("llama-server", "vllm", "ollama", "lm studio",
                                          "lm-studio", "lmstudio")):
        return True
    if n.startswith("python") and pid.isdigit():
        return "vllm" in _probe(f"ps -p {pid} -o args=").lower()
    return False


def _env_port():
    """(port, bad_text): DENEB_PORT as a valid 1-65535 int, or (None, the bad value), or
    (None, '') when it is unset or empty."""
    raw = os.environ.get("DENEB_PORT", "").strip()
    if not raw:
        return None, ""
    if re.fullmatch(r"[0-9]{1,5}", raw) and 1 <= int(raw) <= 65535:
        return int(raw), ""
    return None, raw[:20]


def serving_port() -> int:
    """The port the model is served on, resolved once per process.
    1. DENEB_PORT, when it is a valid port (invalid: ignored, the endpoint row says so).
    2. The first of 8001, 8080, 8000, 11434, 1234 with a listener whose /v1/models answers
       with JSON that has a "data" LIST, and that is a model server: its owner is a known
       one (llama-server, vllm, ollama, LM Studio), or the owner is hidden, or an entry says
       "object": "model". Any other app that happens to answer JSON is skipped, so a dead
       model is not masked by it.
    3. Else the first of those whose listener is a known model server: it may still be loading.
    4. Else 8001.
    Without `ss` the candidates are probed with curl alone and the owner is unknown."""
    global _port_cache
    if _port_cache is not None:
        return _port_cache[0]
    port, bad = _env_port()
    warn = f"⚠ DENEB_PORT={bad} is not a port (1-65535), ignored" if bad else ""
    if port is None:
        ss = _probe("ss -ltnp")
        have_ss = "LISTEN" in ss
        if have_ss:
            listening = [(p,) + _owner_in(ss, p) for p in _PORT_CANDIDATES if _owner_in(ss, p) != ("", "")]
        else:
            listening = [(p, "?", "") for p in _PORT_CANDIDATES]

        def qualifies(p, pid, proc):
            is_list, has_model = _models_list(_curl(p, "/v1/models"))
            return is_list and (pid == "?" or has_model or _is_known_server(proc, pid))
        port = next((p for p, pid, proc in listening if qualifies(p, pid, proc)), None)
        if port is None and have_ss:
            port = next((p for p, pid, proc in listening if _is_known_server(proc, pid)), None)
        if port is None and _hint["port"]:
            port = _hint["port"]
        if port is None:
            # Nothing answers and nothing is remembered. Prefer what this box has: the unit
            # `deneb setup` writes (the 8001 default), then an installed Ollama, then 8001.
            how = "" if _own_llama_unit_exists() else _ollama_install()
            if how:
                port, _inferred["ollama"] = 11434, how
            else:
                port = DEFAULT_PORT
    _port_cache = (port, warn)
    return port


def _port_warning() -> str:
    serving_port()
    return _port_cache[1]


def _service_in_heading(status_text: str) -> str:
    """The *.service named on the heading (first) line of `systemctl status`, else ''. Only
    the heading counts: a CGroup line names every ancestor, so a process started from a
    terminal shows user@1000.service there, and its heading is a *.scope. The user manager
    (user@1000.service) is not a model service either way."""
    lines = status_text.strip().splitlines()
    m = re.match(r"^\W*([\w@.\-]+\.service)\b", lines[0]) if lines else None
    return m.group(1) if m and not m.group(1).startswith("user@") else ""


def _unit_of_pid(pid: str) -> str:
    """Resolve the systemd --user unit owning `pid` — from the PID, not a name guess. '' if
    it can't be mapped (e.g. a bare process launched outside systemd)."""
    if not pid or pid == "?":
        return ""
    return _service_in_heading(_probe(f"systemctl --user status {pid} --no-pager"))


def _system_unit_of_pid(pid: str) -> str:
    """The system (not --user) service owning `pid`, read-only: `systemctl status <pid>`."""
    if not pid or pid == "?":
        return ""
    return _service_in_heading(_probe(f"systemctl status {pid} --no-pager"))


def _scoped_unit_of_pid(pid: str) -> tuple[str, str]:
    """(unit, scope) with scope 'user' or 'system', or ('', '') when the process is not a
    service. The scope travels with the name: a --user unit and a system unit can share one."""
    unit = _unit_of_pid(pid)
    if unit:
        return unit, "user"
    unit = _system_unit_of_pid(pid)
    return (unit, "system") if unit else ("", "")


def _is_enabled(unit: str, scope: str) -> str:
    return _probe(f"systemctl {'--user ' if scope == 'user' else ''}is-enabled {unit}").strip()


def _api_kind(port: int) -> str:
    """'ollama', 'vllm', 'llama' or 'unknown', from the server's own HTTP API: read-only GETs,
    for when the process behind the port can't be seen (another user's). Ollama answers
    /api/version with a "version"; vLLM and llama-server say who they are in /v1/models
    ("owned_by": "vllm" / "llamacpp"). Anything else is 'unknown': never guessed."""
    if port in _kind_cache:
        return _kind_cache[port]
    kind = "unknown"
    try:
        obj = json.loads(_curl(port, "/api/version"))
        if isinstance(obj, dict) and "version" in obj:
            kind = "ollama"
    except ValueError:
        pass
    if kind == "unknown":
        owners = set(re.findall(r'"owned_by"\s*:\s*"([^"]*)"', _curl(port, "/v1/models")))
        kind = "vllm" if "vllm" in owners else ("llama" if "llamacpp" in owners else "unknown")
    _kind_cache[port] = kind
    return kind


def _ollama_system_unit():
    """('ollama.service', unit-file state) when that system unit is loaded and active, else
    ('', ''). Read-only. Only asked once the API has confirmed the server IS Ollama."""
    vals = [v.strip() for v in _probe(
        "systemctl show ollama.service -p LoadState -p ActiveState -p UnitFileState --value").splitlines()]
    if "loaded" in vals and "active" in vals:
        return "ollama.service", next((v for v in vals if v in (
            "enabled", "disabled", "static", "indirect", "generated", "masked", "alias")), "")
    return "", ""


def _hidden_unit(port: int):
    """(unit, 'system') for a server whose process can't be seen, when its API says it is
    Ollama and ollama.service is loaded and active; else ('', '')."""
    if _api_kind(port) == "ollama":
        unit, _boot = _ollama_system_unit()
        if unit:
            return unit, "system"
    return "", ""


SERVER_NAMES = {"ollama": "Ollama", "vllm": "vLLM", "llama": "llama.cpp"}   # _api_kind -> name


def _endpoint_state(port: int):
    """(answers, model id or "", /v1/models body, /health body) for the server on `port`.
    Read-only GETs. /v1/models works on every server; /health is llama.cpp's (vLLM's is an
    empty 200, Ollama has none)."""
    mv = _curl(port, "/v1/models")
    ok = _models_json(mv)
    health = ""
    if not ok:
        health = _curl(port, "/health")
        ok = _health_ok(health)
    model = _model_id(mv).split("/")[-1] if ok else ""
    return ok, model, mv, health


def _model_id(mv: str) -> str:
    """The first model id in a /v1/models body, exactly as served ("" when none): the name a
    request must send. vLLM serving a repo id (org/name) and Ollama's hf.co/... names answer
    404 to the short form."""
    m = re.search(r'"id"\s*:\s*"([^"]+)"', mv or "")
    return m.group(1) if m else ""


def serving_now():
    """(port, kind, model) for the model server answering now, or None when none answers.
    kind is _api_kind's; model is "" when the server lists none. Read-only, so recommend and
    setup can say what already serves instead of planning a second server blind."""
    port = serving_port()
    ok, model, _mv, _health = _endpoint_state(port)
    return (port, _api_kind(port), model) if ok else None


# Each check returns (label, ok, evidence, fix_if_not_ok). ok is True, False, or None for
# "doesn't apply here" (an optional part that isn't set up, or a fact the server can't give).
def _chk_endpoint():
    port = serving_port()
    ok, model, mv, health = _endpoint_state(port)
    ev = (f"serving {model}" if model else "serving") if ok else f"not responding on :{port}"
    fix = _stopped_fix()
    if not ok and _port_owner(port)[0] and "loading" in f"{mv} {health}".lower():
        # llama.cpp answers 503 "Loading model" on both routes until the weights are in memory
        ev = f"listening on :{port}, still loading the model"
        fix = "the server is up and still loading the model: wait a minute, then re-run `deneb check`"
    warn = _port_warning()
    if warn:
        ev += f" {warn}"
    return (f"LLM endpoint serving (:{port})", ok, ev, fix)


def _service_hidden(port: int):
    """The service row when the socket's owner can't be seen (it runs as another user). The
    server's own API says what it is; Ollama's system unit is then asked for by name, which
    is no longer a guess. Anything else is a warning, never a failure by itself."""
    label = "Model service enabled + running"
    kind = _api_kind(port)
    if kind == "ollama":
        unit, boot = _ollama_system_unit()
        if unit and boot == "disabled":
            return (label, True, f"{unit} (system) serving :{port} ⚠ (not enabled — won't auto-start "
                    "after reboot)", f"`sudo systemctl enable {unit}` so it survives a reboot")
        if unit:
            return (label, True, f"{unit} (system) serving :{port}", "")
    name = SERVER_NAMES.get(kind, "A server")
    return (label, True, f"{name} answers on :{port} but runs as another user, and check can't "
            "see which service owns it ⚠",
            "nothing to fix if that is intended. To see the owner, run `systemctl status <pid>` "
            f"as that user or root (`sudo ss -ltnp | grep :{port}` shows the pid)")


def _chk_service():
    # Detect by who owns the serving port, then resolve its unit FROM the pid.
    port = serving_port()
    pid, proc = _port_owner(port)
    if not pid:
        return ("Model service enabled + running", False, f"nothing is listening on :{port}",
                _stopped_fix())
    if pid == "?":
        return _service_hidden(port)
    unit, scope = _scoped_unit_of_pid(pid)
    who = (f"{unit} (system)" if scope == "system" else unit) or (
        f"{proc} (pid {pid})" if proc and proc != "?" else f"pid {pid}")
    at_boot = _is_enabled(unit, scope) if unit else ""
    if at_boot == "disabled":
        cmd = f"systemctl --user enable {unit}" if scope == "user" else f"sudo systemctl enable {unit}"
        return ("Model service enabled + running", True,
                f"{who} serving :{port} ⚠ (not enabled — won't auto-start after reboot)",
                f"`{cmd}` so it survives a reboot")
    if scope == "user" and at_boot in ("enabled", "static", "indirect") and _linger() is False:
        # a user manager only runs while its owner is logged in, unless lingering is on
        return ("Model service enabled + running", True,
                f"{who} serving :{port} ⚠ (a user service: it stops when you log out and only "
                "starts at your next login, not at boot)",
                "`loginctl enable-linger \"$(id -un)\"` keeps it running (if it is refused: `sudo loginctl enable-linger \"$(id -un)\"`)")
    boot = f", {at_boot} at boot" if at_boot in ("enabled", "static", "indirect") else ""
    return ("Model service enabled + running", True, f"{who} serving :{port}{boot}", "")


def _linger():
    """True / False: whether systemd keeps this user's services running without a login
    (logind's flag file); None when it can't be read (no logind, no user name)."""
    try:
        import pwd
        user = pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        return None
    if not os.path.isdir("/run/systemd/system"):   # not booted with systemd
        return None
    return os.path.exists(f"/var/lib/systemd/linger/{user}")


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
    """The command line of the process that owns the serving port, picked by PID. A box often
    runs several llama-servers (a judge on one port, a second model on another, a wrapper script
    whose args name the binary), so the first pgrep line is not reliably the one serving."""
    port = serving_port()
    pid, proc = _port_owner(port)
    if proc in _PORT_PROXIES:
        return _container_server()
    if not proc or proc == "?":
        # A hidden owner (another user's socket) or nothing listening: there is no process to
        # find, and a `pgrep -af llama|ollama|vllm` would pick some unrelated one. The caller
        # reads the server's HTTP API instead.
        return ""
    ps = _probe(f"pgrep -af {shlex.quote(proc)}")
    lines = [ln.strip() for ln in ps.splitlines() if re.match(r"\s*\d+\s", ln)]
    for ln in lines:
        if pid and ln.split(None, 1)[0] == pid:
            return ln
    for ln in lines:
        if re.search(rf"--port[ =]{port}(?!\d)", ln):
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


def _server_kind(proc: str, ps: str = "") -> str:
    """'ollama', 'vllm', 'lmstudio', 'llama' (llama.cpp, llama-cpp-python, anything that names
    a .gguf) or 'unknown', from the port owner's name and the serving command line."""
    n, a = proc.lower(), ps.lower()
    if n.startswith("llama"):
        return "llama"
    if "ollama" in n or (not n.startswith("python") and "ollama" in a):
        return "ollama"
    if "vllm" in n or "vllm" in a:
        return "vllm"
    if n == "lms" or any(k in n for k in ("lm studio", "lm-studio", "lmstudio")):
        return "lmstudio"
    if ".gguf" in a or "llama_cpp" in a or "llama-server" in a:
        return "llama"
    return "unknown"


_OLLAMA_SPLIT = re.compile(r"(\d+)%/(\d+)%\s*CPU/GPU")
_OLLAMA_ONE = re.compile(r"(\d+)%\s*(GPU|CPU)\b")


def _gpu_ollama(label: str):
    """`ollama ps` lists the loaded models and where each runs: '100% GPU', '100% CPU' or a
    split like '48%/52% CPU/GPU'. Nothing loaded means there is nothing to judge yet."""
    out = _probe("ollama ps").strip()
    if out.startswith(("(not installed", "(could not run", "(‘ollama’")) or out.startswith("[stderr]"):
        return (label, UNKNOWN, "couldn't run `ollama ps` here, so the GPU use can't be read", "")
    seen = []                                   # (model name, cpu %, gpu %)
    for ln in out.splitlines():
        if ln.lstrip().upper().startswith("NAME") or not ln.strip():
            continue
        name = ln.split()[0]
        m = _OLLAMA_SPLIT.search(ln)
        if m:
            seen.append((name, int(m.group(1)), int(m.group(2))))
            continue
        m = _OLLAMA_ONE.search(ln)
        if m:
            pct = int(m.group(1))
            seen.append((name, pct, 100 - pct) if m.group(2) == "CPU" else (name, 100 - pct, pct))
    if not seen:
        # Ollama loads a model on the first request and unloads it when idle, so with none
        # loaded where it WILL run is unknown: '?', never a quiet '-' under "all checks pass".
        return (label, UNKNOWN, "no model loaded, so where it runs can't be read yet: send it "
                                "one request, then re-run deneb check", "")
    desc = ", ".join(f"{n} {'100% GPU' if c == 0 else ('100% CPU' if g == 0 else f'{c}% CPU / {g}% GPU')}"
                     for n, c, g in seen)
    if any(g == 0 for _n, _c, g in seen):
        return (label, False, f"running on CPU: {desc}",
                "give Ollama the GPU: check the driver is visible (`nvidia-smi` or `rocm-smi`), "
                "or use a smaller model that fits in GPU memory")
    if any(c > 0 for _n, c, _g in seen):
        return (label, True, f"part of the model is on the CPU: {desc} ⚠",
                "the model doesn't fully fit in GPU memory: use a smaller or more heavily "
                "quantised model, or a shorter context")
    return (label, True, f"{desc}", "")


_HIDDEN_WHY = "can't see the server's process (it runs as another user)"


def _chk_gpu():
    # Inspect the ACTUAL serving process (the one behind the serving port, not a hardcoded
    # "llama-server"). llama.cpp: its offload flag and the accelerator it loaded or links;
    # Ollama: `ollama ps`; vLLM: its --device; LM Studio: nothing it exposes.
    label = "Built with GPU accelerator (not CPU)"
    port = serving_port()
    pid, proc = _port_owner(port)
    hidden = pid == "?"
    ps = "" if hidden else _serving_cmdline()
    kind = _api_kind(port) if hidden else _server_kind(proc, ps)
    if kind == "ollama":
        return _gpu_ollama(label)
    if kind == "lmstudio":
        return (label, None, "LM Studio doesn't expose GPU offload; check its settings", "")
    if hidden:
        return (label, UNKNOWN, _HIDDEN_WHY, "")
    if not pid:
        return (label, UNKNOWN, "no model server is running, so the GPU use can't be read", "")
    if kind == "vllm":
        if re.search(r"--device[ =]cpu(?![\w-])", ps):
            return (label, False, "vLLM is running on CPU (--device cpu)",
                    "start vLLM without --device cpu, on a machine with a supported GPU")
        return (label, True, "vLLM serving (no --device cpu)", "")
    if not ps:
        return (label, UNKNOWN, f"couldn't read the command line of the process serving :{port}, "
                "so the GPU use can't be read", "")
    if kind == "unknown":
        return (label, UNKNOWN, "can't tell what kind of server this is, so the GPU use isn't checked", "")
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
    port = serving_port()
    # Ollama, vLLM and LM Studio keep models in stores of their own (blobs, the Hugging Face
    # cache, a models folder), not behind a -m path: there is no single file to look for.
    pid, proc = _port_owner(port)
    hidden = pid == "?"
    kind = _api_kind(port) if hidden else _server_kind(proc, _serving_cmdline())
    if not pid and (port == 11434 or "ollama" in _hint["unit"].lower()):
        kind = "ollama"      # stopped, but the stored unit or Ollama's own port says what it was
    elif not pid and _VLLM_EXEC.search(_unit_text()[2]):
        kind = "vllm"        # stopped, and the unit that starts it runs vLLM
    name = {"ollama": "Ollama", "vllm": "vLLM", "lmstudio": "LM Studio"}.get(kind)
    if name:
        return ("Model file present", None, f"{name} manages its own model files; not checked", "")
    if hidden:
        return ("Model file present", UNKNOWN, _HIDDEN_WHY, "")
    if pid and kind != "llama":
        return ("Model file present", UNKNOWN, "can't tell what kind of server this is, so no model "
                "file is looked for", "")
    # (nothing listening: the server is down, but its model file can still be looked for below)
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
    mv = _curl(port, "/v1/models")
    m = re.search(r'"id"\s*:\s*"(/[^"]+\.gguf)"', mv)
    if m and "does not exist" not in _read(m.group(1)):
        return ("Model file present", True, m.group(1).split("/")[-1], "")
    # 2b. A stopped server: the -m path in the unit that starts it (deneb setup writes the
    #     resolved absolute path there).
    path = _unit_model_path()
    if path:
        st = _probe(f"stat -L -c %s {path}").strip()
        if st.isdigit():
            return ("Model file present", True, f"{path.split('/')[-1]} ({_gib(int(st))})", "")
    # 3. The conventional models directory. Expanded here: commands run without a shell,
    #    so a literal "~/models" was never found. A vision projector (mmproj) is not the model.
    found = [ln for ln in _probe(f"find {os.path.expanduser('~/models')} -maxdepth 2 -name '*.gguf'").splitlines()
             if ln.strip().endswith(".gguf") and "mmproj" not in ln.split("/")[-1].lower()]
    ok = bool(found)
    return ("Model file present", ok, (sorted(found)[0].split("/")[-1] if ok else "no .gguf found"),
            "no model file yet: run `deneb recommend`, then `deneb setup <model>`")


def _unit_model_path() -> str:
    """The absolute -m / --model path in the unit file that starts the model server, '' when
    there is none (a unit can't be read, or names its model some other way)."""
    text = _unit_text()[2]
    m = re.search(r"""^ExecStart=.*?\s(?:-m|--model)[ =]["']?(/[^\s"']+)""", text, re.M)
    return m.group(1) if m else ""


_NOT_SET_UP = "not set up (optional: only needed to reach the model from other machines)"
_GATEWAY_LABEL = "Auth gateway up (:8002, key + rate-limit)"


def _chk_gateway():
    # Optional: only for reaching the model from other machines. Who owns :8002
    # (behavioural), then its unit from the pid — no filename guessing.
    port = serving_port()
    fix = f"stand up the bearer-key gateway on :8002 in front of :{port} (never expose :{port} directly)"
    pid, proc = _port_owner(8002)
    if not pid:
        return (_GATEWAY_LABEL, None, _NOT_SET_UP, fix)
    unit, scope = _scoped_unit_of_pid(pid)
    who = (f"{unit} (system)" if scope == "system" else unit) or (
        f"{proc} on :8002" if proc and proc != "?" else ":8002")
    # Auth: an UNauthenticated request must be rejected. (Plain GET only — the allowlist
    # blocks -o/-w/-X, so read the body and look for a rejection instead of a status code.)
    body = _curl(8002, "/v1/models").lower()
    rejected = any(w in body for w in ("unauthorized", "forbidden", "missing", "invalid",
                                       "api key", "api_key", "authenticat", "401", "403"))
    served = '"id"' in body or '"data"' in body
    if served and not rejected:
        return (_GATEWAY_LABEL, False,
                f"{who} up but ⚠ AUTH NOT ENFORCED (served without a key)",
                "the gateway is not requiring an API key — enable bearer-key auth before use")
    ev = f"{who}, auth enforced" if rejected else f"{who} up"
    return (_GATEWAY_LABEL, True, ev, fix)


def _chk_tunnel():
    # Optional. cloudflared is a real binary — detect the RUNNING PROCESS, not a guessed unit name.
    # Only a process whose program IS cloudflared counts: `vim cloudflared.yml` and
    # `tail -f cloudflared.log` match the pattern too.
    pids = [m.group(1) for ln in _probe("pgrep -af cloudflared").splitlines()
            for m in [re.match(r"\s*(\d+)\s+(\S+)", ln)]
            if m and os.path.basename(m.group(2)) == "cloudflared"]
    label = "Cloudflare tunnel live"
    fix = "install + run cloudflared to expose the gateway outbound-only (no inbound ports)"
    if not pids:
        return (label, None, _NOT_SET_UP, fix)
    # A running cloudflared is not yet this model's tunnel: cloud GPU images ship one for
    # their own notebook (QC on a cloud GPU, 20 Oct). Count it only when it routes here, and
    # read every cloudflared: the notebook's may be listed before the user's own.
    port = serving_port()
    found = [(_tunnel_rank(ports, port), pid, ports) for pid in pids
             for ports in [_tunnel_ports(_probe(f"ps -p {pid} -o args=").strip())]]
    _, pid, ports = max(found, key=lambda f: f[0])
    unit, scope = _scoped_unit_of_pid(pid)
    who = (f"{unit} (system)" if scope == "system" else unit) or "cloudflared"
    if ports is None:
        return (label, None, f"{who} runs, but its routes are kept in the Cloudflare dashboard, so "
                "deneb can't see whether one reaches the gateway (:8002)", fix)
    if port in ports:
        return (label, True, f"{who} running ⚠ routes straight to the model on :{port}, with no key "
                "in front", "route the tunnel to the key gateway on :8002 instead of the model port")
    if 8002 in ports:
        return (label, True, f"{who} running, routes to the gateway (:8002)", "")
    where = ", ".join(f":{p}" for p in sorted(ports)) or "nothing local"
    return (label, None, f"{who} runs, but routes to {where}, not to the gateway (:8002): it is "
            "not a tunnel to this model", fix)


def _tunnel_rank(ports, port) -> int:
    # which cloudflared speaks for the row: one exposing the model port without a key is the
    # finding that matters most, then one routing to the gateway, then one deneb can't read
    if ports is None:
        return 1
    return 3 if port in ports else 2 if 8002 in ports else 0


_CF_CONFIGS = ("~/.cloudflared/config.yml", "~/.cloudflared/config.yaml",
               "/etc/cloudflared/config.yml", "/etc/cloudflared/config.yaml")
# the scheme is optional (`--url localhost:8002` works) and an ingress value may be quoted
_CF_LOCAL = r"""["']?(?:https?://)?(?:localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0):(\d+)(?!\d)"""


def _tunnel_ports(args: str):
    """The local ports a cloudflared routes to, from its command line (`--url`) or its config
    file's ingress rules (`service:`). None when the routes can't be read here: a token tunnel
    keeps them in the Cloudflare dashboard, and a config file may be unreadable or missing."""
    m = re.search(r"--url[= ](\S+)", args)
    if m:
        return {int(p) for p in re.findall(_CF_LOCAL, m.group(1))}
    m = re.search(r"--config[= ](\S+)", args)
    paths = [m.group(1)] if m else list(_CF_CONFIGS)
    for path in paths:
        text = _read(path)
        if text.startswith("("):            # does not exist / permission denied / not a file
            continue
        if not re.search(r"^\s*-?\s*service:", text, re.M):
            return None                     # no ingress here: the dashboard holds the routes
        return {int(p) for p in re.findall(r"^\s*-?\s*service:\s*" + _CF_LOCAL, text, re.M)}
    return None


_CHECKS = [_chk_endpoint, _chk_service, _chk_gpu, _chk_model_file, _chk_gateway, _chk_tunnel]


# Stable ids for each check, so callers that track a row over time (deneb watch) don't have
# to key on a label that might be reworded.
_IDS = {"_chk_endpoint": "endpoint", "_chk_service": "service", "_chk_gpu": "gpu",
        "_chk_model_file": "model_file", "_chk_gateway": "gateway", "_chk_tunnel": "tunnel"}


# Rows that are optional: a box that never set them up is healthy (ok is None). `deneb watch`
# remembers whether it has seen each one running, so a later disappearance is a failure.
_OPTIONAL = ("gateway", "tunnel")


SPEED_WARN_RATIO = 0.7      # below this share of the measured speed the row warns
_PROBE_PROMPT = "Count from 1 to 40 in words, separated by commas."
_PROBE_TOKENS = 64


def _decode_probe(port: int, model=None):
    """(tok/s, "") from one short greedy completion on the serving port, or (None, why). Uses
    the server's own decode timing when it reports one (llama.cpp does), else completion tokens
    over wall time, which includes the prompt, so it reads a little low. `model` is the served
    name, sent in both requests (vLLM answers only to the name it serves)."""
    import time
    named = {"model": model} if model else {}
    # One warm-up first: a model still loading (or not yet paged in) answers its first request
    # slowly, which would read as a false "under 70%". Only the second request is timed.
    wstatus, wbody = net.local_post_json(
        port, "/v1/chat/completions",
        {**named, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
         "temperature": 0, "stream": False}, timeout=120)
    if wstatus is None:
        return None, f"no answer ({str(wbody)[:60]})"
    t0 = time.monotonic()
    status, body = net.local_post_json(
        port, "/v1/chat/completions",
        {**named, "messages": [{"role": "user", "content": _PROBE_PROMPT}],
         "max_tokens": _PROBE_TOKENS, "temperature": 0, "stream": False}, timeout=60)
    dt = time.monotonic() - t0
    if status is None:
        return None, f"no answer ({str(body)[:60]})"
    if status != 200:
        return None, f"the server answered HTTP {status}"
    per = (body.get("timings") or {}).get("predicted_per_second")
    if isinstance(per, (int, float)) and not isinstance(per, bool) and per > 0:
        return float(per), ""
    n = (body.get("usage") or {}).get("completion_tokens")
    if isinstance(n, int) and not isinstance(n, bool) and n >= 16 and dt > 0:
        return n / dt, ""
    return None, "the answer was too short to time"


_QUANT_IN_NAME = re.compile(r"(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)+|Q8_0|BF16|F16|F32)", re.I)


def _record_mismatch(exp: dict, served: str) -> str:
    """'<what the record is for>' when the record's hardware or quant differs from this box or
    the quant visible in the served model's name; "" when they match or cannot be told."""
    hw = exp.get("hardware")
    if isinstance(hw, str) and hw:
        try:
            from . import hardware, measured
            ids = measured.hardware_candidates(hardware.profile_hardware())
        except Exception:  # noqa: BLE001 - an unreadable box is "cannot tell", not a mismatch
            ids = []
        if ids and hw not in ids:
            return f"{exp.get('hardware_label') or hw}, this box is {ids[-1]}"
    q = exp.get("quant")
    seen = _QUANT_IN_NAME.findall(served or "")
    if isinstance(q, str) and q and seen and not any(q.lower() in x.lower() or x.lower() in q.lower()
                                                     for x in seen):
        return f"{q}, this server serves {seen[0]}"
    return ""


def _speed_row():
    """The decode-speed row for `deneb check`: a quick live probe against the speed that
    `deneb setup --measured` recorded. None when nothing was recorded or nothing is serving
    (the endpoint row already says that). Kept out of scan(): it asks the model for tokens, so
    `deneb watch` must not run it every tick."""
    label = "Decode speed vs TokenMark"
    path = config.expected_path()
    if not os.path.exists(path):
        return None
    exp = config.load_expected()
    if exp is None:
        return (label, None, f"{path} is not a usable record (no positive tok_s)",
                "delete it, or re-run `deneb setup <model> --measured` and run its record step")
    now = serving_now()
    if not now:
        return None
    port, _kind, served = now
    from . import setup_advisor
    want = setup_advisor._norm_key(exp.get("model") or "")
    have = setup_advisor._norm_key(served or "")
    if want and have and want not in have and have not in want:
        return (label, None, f"not compared: the record is for {exp.get('model')}, this server "
                f"serves {served}", "")
    # a record for other hardware or another quant is not a yardstick for this box
    mism = _record_mismatch(exp, served)
    if mism:
        return (label, None, f"expected is for {mism}, not comparable", "")
    tps, why = _decode_probe(port, _model_id(_curl(port, "/v1/models")) or served)
    shown = f"{exp['tok_s']:.1f}".rstrip("0").rstrip(".")
    if tps is None:
        return (label, UNKNOWN, f"measured {shown} tok/s on TokenMark; could not time this box: {why}",
                "re-run `deneb check` when the server is idle")
    mine = f"{tps:.1f}".rstrip("0").rstrip(".")
    ev = f"measured {shown} tok/s on TokenMark, this box now {mine} tok/s"
    if tps >= exp["tok_s"] * SPEED_WARN_RATIO:
        return (label, True, f"{ev}, ok", "")
    return (label, True, f"{ev} ⚠ (under {int(SPEED_WARN_RATIO * 100)}% of the measured speed)",
            "another job may be using the GPU, the model may be partly on CPU, or the settings "
            "differ from the measured row; compare with the setup steps and `deneb profile`")


def scan() -> list:
    """Run every check and return the rows WITHOUT printing: (id, label, ok, evidence, fix).
    ok is True, False, None (doesn't apply: shown as "–", never a failure) or UNKNOWN "?" (could
    not be read: shown as "?", never a pass or a fail).
    A probe that raises becomes a ✗ row, never a crash. run() prints these; `deneb watch`
    tracks them."""
    rows = []
    for fn in _CHECKS:
        rid = _IDS.get(fn.__name__, fn.__name__.removeprefix("_chk_"))
        try:
            rows.append((rid,) + tuple(fn()))
        except Exception as e:  # noqa: BLE001 — a probe error is a ✗, never a crash
            rows.append((rid, fn.__name__, False, f"(check error: {e})", "re-run, or ask deneb directly"))
    return rows


def run() -> int:
    ui.banner()
    print(f"{_C['d']}running a read-only scan of this box…{_C['z']}")
    rows = [r[1:] for r in scan()]
    speed_row = _speed_row()
    if speed_row:
        rows.append(speed_row)
    print()
    shown = set()                           # one fault often fails two rows with the same fix
    for label, ok, ev, fix in rows:
        if ok is None:                      # not applicable or not set up: not a failure
            print(f"  {_C['d']}–{_C['z']} {label}  {_C['d']}— {ev}{_C['z']}")
            continue
        if ok == UNKNOWN:                   # exists but could not be read: neither pass nor fail
            print(f"  {_C['d']}?{_C['z']} {label}  {_C['d']}— {ev}{_C['z']}")
            continue
        mark = f"{_C['g']}✓{_C['z']}" if ok else f"{_C['r']}✗{_C['z']}"
        print(f"  {mark} {label}  {_C['d']}— {ev}{_C['z']}")
        if not ok or ("⚠" in ev and fix):
            print(f"      {_C['y']}fix:{_C['z']} {'same as above' if fix in shown else fix}")
            shown.add(fix)
    print()
    n = sum(1 for r in rows if r[1] is False)
    unread = sum(1 for r in rows if r[1] == UNKNOWN)
    warned = sum(1 for r in rows if r[1] is True and "⚠" in r[2])
    if n:
        print(f"{_C['y']}{_C['b']}Not done yet — {n} item(s) left above.{_C['z']}")
        down = any(r[1] is False and "nothing is listening" in r[2] for r in rows)
        if down and _inferred["ollama"]:
            print(f"{_C['d']}Ollama is installed but not running. Start it as shown above, then "
                  f"re-run deneb check.{_C['z']}")
        elif down and _stopped_fix().startswith("start it again"):
            print(f"{_C['d']}The model server is set up but stopped. Start it as shown above, then "
                  f"re-run deneb check.{_C['z']}")
        elif down:
            print(f"{_C['d']}No model server was found. Next: deneb recommend, then "
                  f"deneb setup <model>.{_C['z']}")
    elif unread or warned:
        parts = ([f"{warned} warning{'s' if warned != 1 else ''} above (marked ⚠)"] if warned else []) + (
            [f"{unread} row(s) could not be read (shown as ?, which exits 1)"] if unread else [])
        print(f"{_C['y']}{_C['b']}No failures, but {' and '.join(parts)}.{_C['z']}")
    else:
        print(f"{_C['teal']}{_C['b']}✓ all checks pass: the model is served and ready.{_C['z']}")
    if n:
        return 2
    return 1 if unread or warned else 0
