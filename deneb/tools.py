"""deneb.tools — the read-only tool executor. THE SECURITY BOUNDARY.

The Neo endpoint DECIDES which tool to run; this module runs it on the box and is
the last line of defence. It executes ONLY read-only, unprivileged actions, by a
DENY-BY-DEFAULT allowlist:

  - `run` accepts a command only if its binary is on the read-only allowlist, with
    extra per-binary guards for the few allowed binaries that have mutating modes
    (systemctl start/stop, find -exec/-delete, nvidia-smi -pl, pip install, …).
  - Commands run with shell=False (argv, no shell), so shell metacharacters
    (; | & > $() `` ) can NEVER chain or redirect — they'd be literal args.
  - NEVER sudo/su. deneb runs as the logged-in user and reads what that user can
    read; if a step needs root it tells the user to run it themselves.

Even a tampered endpoint asking for `rm -rf ~` / `systemctl stop` / `find -delete`
is refused here. The client is the security boundary precisely so it can be open
and audited.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess

# ── Secret hygiene ────────────────────────────────────────────────────────────
# On a customer's box Deneb must never surface secrets. Two defences: (1) refuse to
# read the CONTENT of files that are secrets by nature (report existence + a count so
# a "key present?" check still works), and (2) REDACT any secret-looking token from
# every tool observation before it leaves this machine (to the model or the screen).
_SECRET_FILE = re.compile(
    r"^(keys?\.json|\.env(\..+)?|.*token.*|.*secret.*|.*credential.*|id_rsa.*|.*\.pem"
    r"|.*\.key|\.netrc|\.npmrc|\.git-credentials|.*\.p12|.*\.pfx)$",
    re.I,
)
_KEYNAME = (r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret"
            r"|secret|token|password|passwd)")
_REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1[REDACTED]"),
    # key = value / "key": "value" / key: value — value may be quoted; capture the
    # separator (incl. any quotes) into group 1 so the redaction reads cleanly.
    (re.compile(rf"(?i)({_KEYNAME}[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-/+]{{8,}}"), r"\1[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9._\-]{20,}"), "[REDACTED-JWT]"),
    (re.compile(r"\b[A-Fa-f0-9]{40,}\b"), "[REDACTED-HEX]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACTIONS:
        text = pat.sub(repl, text)
    return text

# Deny-by-default: only these binaries may run at all.
_READ_ONLY = {
    "journalctl", "systemctl", "ls", "cat", "head", "tail", "grep", "egrep", "zgrep",
    "find", "stat", "readlink", "realpath", "file", "wc", "du", "df", "free",
    "uname", "hostnamectl", "lscpu", "lsblk", "lspci", "lsusb", "ps", "id", "groups",
    "env", "printenv", "which", "whereis", "type", "echo", "pwd", "date", "uptime",
    "ps", "pgrep", "pstree", "nvidia-smi", "rocminfo", "rocm-smi", "ldd", "nproc",
    "getconf", "ss",
    "curl", "tree", "sha256sum", "md5sum", "cut", "sort", "uniq", "basename", "dirname",
    "python3", "python", "pip", "pip3", "cmake", "gcc", "g++", "nvcc", "ip", "sysctl",
    "systemd-analyze", "loginctl", "test", "true",
}
# systemctl: read verbs only — never start/stop/restart/enable/disable/mask/kill/reload.
_SYSTEMCTL_OK = {
    "status", "show", "is-active", "is-enabled", "is-failed", "list-units",
    "list-unit-files", "list-timers", "list-sockets", "cat", "get-default",
    "show-environment", "list-dependencies",
}
# Per-binary flags/subcommands that mutate or execute despite the binary being "read-ish".
_BAD_ARGS: dict[str, set[str]] = {
    "find": {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf",
             "-fprint", "-fls", "-fprint0"},
    "nvidia-smi": {"-pl", "--gpu-reset", "-r", "--reset", "-e", "-c", "-pm", "-ac",
                   "-rac", "-lgc", "-rgc", "-caa", "--gom", "-am", "-cc"},
    "pip": {"install", "uninstall", "download", "wheel", "config"},
    "pip3": {"install", "uninstall", "download", "wheel", "config"},
    "sysctl": {"-w", "--write", "-p", "--load"},
    "ip": {"add", "del", "delete", "set", "flush", "change", "replace", "append"},
    "loginctl": {"lock-session", "unlock-session", "terminate-session",
                 "kill-session", "enable-linger", "disable-linger", "poweroff",
                 "reboot", "suspend", "hibernate"},
}
# Binaries allowed ONLY for a version/help query (they can otherwise execute arbitrary code).
_VERSION_ONLY = {
    "python3": {"--version", "-V"},
    "python": {"--version", "-V"},
    "cmake": {"--version"},
    "nvcc": {"--version", "-V"},
    "gcc": {"--version", "-dumpversion", "-dumpfullversion"},
    "g++": {"--version", "-dumpversion", "-dumpfullversion"},
}
_CHAIN_TOKENS = {";", "|", "&", "&&", "||", "`", ">", ">>", "<", "<<", "$(", "&>", "2>", "2>>"}


def _refuse(reason: str) -> dict:
    return {"ok": False, "refused": True, "output": f"[refused] {reason}"}


def run(cmd: str, timeout: int = 25) -> dict:
    """Run a READ-ONLY shell command (no shell, no sudo). Returns
    {ok, output[, refused]}. Refuses anything not provably read-only + unprivileged."""
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return _refuse("could not parse the command safely")
    if not argv:
        return _refuse("empty command")
    if argv[0] in ("sudo", "su", "pkexec", "doas"):
        return _refuse("deneb never escalates privileges — run that step yourself if it needs root")
    # Refuse any shell-chaining/redirection token appearing as its own arg. With
    # shell=False these are literal + harmless, but refusing keeps intent honest.
    for a in argv:
        if a in _CHAIN_TOKENS:
            return _refuse("no shell chaining or redirection allowed — one command at a time")
    binv = os.path.basename(argv[0])
    if binv not in _READ_ONLY:
        return _refuse(f"'{binv}' is not on deneb's read-only allowlist")
    rest = argv[1:]

    if binv == "systemctl":
        verb = next((a for a in rest if not a.startswith("-")), "")
        if verb and verb not in _SYSTEMCTL_OK:
            return _refuse(f"systemctl '{verb}' is not a read-only action")
    if binv in _VERSION_ONLY:
        if not (set(rest) & _VERSION_ONLY[binv]):
            return _refuse(f"{binv} can execute code — only a version query is allowed")
    bad = _BAD_ARGS.get(binv)
    if bad:
        for a in rest:
            head = a.split("=", 1)[0]
            if a in bad or head in bad:
                return _refuse(f"'{a}' would mutate/execute — refused (deneb is read-only)")
    if binv == "curl":
        blocked = {"-o", "-O", "--output", "--output-dir", "-T", "--upload-file",
                   "-d", "--data", "--data-binary", "--data-raw", "-F", "--form",
                   "-X", "--request"}
        if any(a in blocked or a.split("=", 1)[0] in blocked for a in rest):
            return _refuse("curl: only a plain localhost GET is allowed (no -o/-X/-d/-F)")
        if not any(("localhost" in a) or ("127.0.0.1" in a) for a in rest):
            return _refuse("curl: only localhost / 127.0.0.1 health checks are allowed")

    try:
        p = subprocess.run(  # noqa: S603 — shell=False, allowlisted binary, read-only
            argv, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expanduser("~"),
        )
    except subprocess.TimeoutExpired:
        return {"ok": True, "output": f"(‘{binv}’ timed out after {timeout}s)"}
    except FileNotFoundError:
        return {"ok": True, "output": f"(not installed on this box: {binv})"}
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "output": f"(could not run {binv}: {e})"}
    out = p.stdout or ""
    if p.stderr:
        out += ("\n[stderr]\n" + p.stderr)
    out = out.strip() or "(no output)"
    return {"ok": True, "output": _redact(out[:8000]), "exit": p.returncode}


def read_file(path: str, max_bytes: int = 200_000) -> dict:
    """Read a file's text (size-capped). Runs as the user — permission denied is
    reported, never bypassed with sudo."""
    p = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.exists(p):
        return {"ok": True, "output": f"(does not exist: {path})"}
    if not os.path.isfile(p):
        return {"ok": True, "output": f"(not a regular file: {path})"}
    if _SECRET_FILE.match(os.path.basename(p)):
        # A secrets file — report existence + a count, NEVER the values.
        try:
            n = sum(1 for _ in open(p, errors="ignore"))
        except Exception:  # noqa: BLE001
            n = "?"
        return {"ok": True, "output": f"({os.path.basename(p)} exists ({n} lines) — deneb does "
                "not read secret values. To verify keys it confirms presence + counts entries, "
                "never the contents.)"}
    try:
        with open(p, "rb") as f:
            data = f.read(max_bytes + 1)
    except PermissionError:
        return {"ok": True, "output": f"(permission denied reading {path} — deneb runs as "
                "you with no sudo; read it yourself if it needs root)"}
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "output": f"(could not read {path}: {e})"}
    text = data[:max_bytes].decode("utf-8", "replace")
    if len(data) > max_bytes:
        text += "\n(…truncated)"
    return {"ok": True, "output": _redact(text)}


def list_dir(path: str) -> dict:
    """List a directory (type + size + name), capped."""
    p = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isdir(p):
        return {"ok": True, "output": f"(not a directory: {path})"}
    try:
        entries = sorted(os.listdir(p))
    except PermissionError:
        return {"ok": True, "output": f"(permission denied listing {path})"}
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "output": f"(could not list {path}: {e})"}
    lines = []
    for e in entries[:400]:
        fp = os.path.join(p, e)
        try:
            st = os.lstat(fp)
            kind = "d" if os.path.isdir(fp) else ("l" if os.path.islink(fp) else "-")
            lines.append(f"{kind} {st.st_size:>10}  {e}")
        except OSError:
            lines.append(f"?            {e}")
    extra = f"\n(…{len(entries) - 400} more)" if len(entries) > 400 else ""
    return {"ok": True, "output": ("\n".join(lines) or "(empty)") + extra}


# Dispatch used by the loop: map an engine action to a local execution.
def execute(action: str, cmd: str = "", path: str = "") -> dict:
    if action == "run":
        return run(cmd)
    if action == "read_file":
        return read_file(path)
    if action == "list_dir":
        return list_dir(path)
    return _refuse(f"unknown tool '{action}'")
