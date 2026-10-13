"""deneb.tools — the read-only tool executor. THE SECURITY BOUNDARY.

The model (or the engine) DECIDES which tool to run; this module runs it on the box and is
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
import shutil
import subprocess

# ── Secret hygiene ────────────────────────────────────────────────────────────
# On a customer's box Deneb must never surface secrets. Two defences: (1) refuse to
# read the CONTENT of files that are secrets by nature (report existence + a count so
# a "key present?" check still works), and (2) REDACT any secret-looking token from
# every tool observation before it leaves this machine (to the model or the screen).
_SECRET_FILE = re.compile(
    r"^(keys?\.json|\.env(\..+)?|.*token.*|.*secret.*|.*credential.*|id_rsa.*|id_(?!.*\.pub$).*|.*\.pem"
    r"|.*\.key|\.netrc|\.npmrc|\.pgpass|\.git-credentials|.*\.p12|.*\.pfx)$",
    re.I,
)
# Directories (relative to the home directory) whose CONTENT is never read or written.
_SECRET_HOME_DIRS = (".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
                     ".config/gcloud", ".password-store", ".local/share/keyrings")
# A variable NAME containing one of these is secret: printenv refuses it, _redact blanks it.
_SECRET_WORDS = "KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH|PRIVATE|COOKIE|SESSION|DSN"
_SECRET_NAME = re.compile(_SECRET_WORDS, re.I)
_KEYNAME = (r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret"
            r"|secret|token|password|passwd)")
_ENV_SECRET = re.compile(
    rf"(?i)(?P<q>[\"']?)(?<![A-Za-z0-9_])(?P<name>[A-Za-z0-9_]*(?:{_SECRET_WORDS})[A-Za-z0-9_]*)"
    r"=(?P<val>[^\n]*)")


def _redact_env(m):
    q, val = m.group("q"), m.group("val")
    rest = val[val.index(q):] if q and q in val else ""
    return f"{q}{m.group('name')}=[REDACTED]{rest}"


_REDACTIONS = [
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END[^\n]*?-----|\Z)", re.S),
     "[REDACTED-PRIVATE-KEY]"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+):[^\s/@]+@"), r"\1:[REDACTED]@"),
    # token-only userinfo (scheme://TOKEN@host, no colon)
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s:/@]+@"), r"\1[REDACTED]@"),
    (_ENV_SECRET, _redact_env),
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

# Deny-by-default: only these binaries may run at all. (`env` is NOT here: it runs any command.)
_READ_ONLY = {
    "journalctl", "systemctl", "ls", "cat", "head", "tail", "grep", "egrep", "zgrep",
    "find", "stat", "readlink", "realpath", "file", "wc", "du", "df", "free",
    "uname", "hostnamectl", "lscpu", "lsblk", "lspci", "lsusb", "ps", "id", "groups",
    "printenv", "which", "whereis", "type", "echo", "pwd", "date", "uptime",
    "pgrep", "pstree", "nvidia-smi", "rocminfo", "rocm-smi", "ldd", "nproc",
    "getconf", "ss", "system_profiler",  # macOS read-only hardware/GPU info probe (Apple path)
    "curl", "tree", "sha256sum", "md5sum", "cut", "sort", "uniq", "basename", "dirname",
    "python3", "python", "pip", "pip3", "cmake", "gcc", "g++", "nvcc", "ip", "sysctl",
    "systemd-analyze", "loginctl", "test", "true", "ollama",
}
# ollama: only the two listing verbs (and a version query), each with NO further argument.
# Everything else (run, pull, rm, stop, serve, create, cp, push, ...) changes the box.
_OLLAMA_OK = {"ps", "list", "--version", "-v", "version"}
# systemctl: read verbs only — never start/stop/restart/enable/disable/mask/kill/reload.
# (show-environment is NOT here: the manager environment holds secrets.)
_SYSTEMCTL_OK = {
    "status", "show", "is-active", "is-enabled", "is-failed", "list-units",
    "list-unit-files", "list-timers", "list-sockets", "cat", "get-default",
    "list-dependencies",
}
# Every other systemctl verb we know of. Used as a backstop so a flag VALUE that happens to
# spell an allowed verb can never hide the real (mutating) verb.
_SYSTEMCTL_MUTATING = {
    "start", "stop", "restart", "reload", "reload-or-restart", "try-restart",
    "try-reload-or-restart", "condrestart", "kill", "clean", "freeze", "thaw", "enable",
    "disable", "reenable", "preset", "preset-all", "mask", "unmask", "link", "revoke",
    "add-wants", "add-requires", "edit", "set-property", "service-log-level",
    "service-log-target", "reset-failed", "daemon-reload", "daemon-reexec", "isolate",
    "default", "rescue", "emergency", "halt", "poweroff", "reboot", "kexec", "exit",
    "switch-root", "suspend", "hibernate", "hybrid-sleep", "suspend-then-hibernate",
    "set-default", "set-environment", "unset-environment", "import-environment",
    "show-environment", "cancel", "bind", "mount-image", "revert", "attach",
}
_SYSTEMD_VAL_SHORT = "ptnosMH"
_SYSTEMD_VAL_LONG = ("property", "type", "lines", "output", "state", "signal", "kill-whom",
                     "kill-who", "root", "image", "job-mode", "host", "machine", "what")
# Per-binary flags/subcommands that mutate or execute despite the binary being "read-ish".
_BAD_ARGS: dict[str, set[str]] = {
    "find": {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf",
             "-fprint", "-fls", "-fprint0", "-files0-from"},
    "nvidia-smi": {"-pl", "--gpu-reset", "-r", "--reset", "-e", "-c", "-pm", "-ac",
                   "-rac", "-lgc", "-rgc", "-caa", "--gom", "-am", "-cc",
                   "-f", "--filename",  # -f FILE writes the report to a file
                   "--power-limit", "--persistence-mode", "--applications-clocks",
                   "--reset-applications-clocks", "--lock-gpu-clocks", "--reset-gpu-clocks",
                   "--lock-memory-clocks", "--reset-memory-clocks", "-lmc", "-rmc",
                   "--compute-mode", "--ecc-config", "--accounting-mode", "--driver-model",
                   "-dm", "--gpu-target-temp", "-gtt", "--auto-boost-default", "-ab",
                   "--mig", "-mig", "-fdm"},
    "ip": {"add", "del", "delete", "set", "flush", "change", "replace", "append"},
}
# Binaries allowed ONLY as ONE version flag (they can otherwise execute arbitrary code).
_VERSION_ONLY = {
    "python3": {"--version", "-V"},
    "python": {"--version", "-V"},
    "cmake": {"--version"},
    "nvcc": {"--version", "-V"},
    "gcc": {"--version", "-dumpversion", "-dumpfullversion"},
    "g++": {"--version", "-dumpversion", "-dumpfullversion"},
}
_CHAIN_TOKENS = {";", "|", "&", "&&", "||", "`", ">", ">>", "<", "<<", "$(", "&>", "2>", "2>>"}


# Control characters (CR, ESC, NUL, DEL, newline...) could spoof what the user sees at a y/N
# prompt or in the log. Tab is fine.
_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _refuse(reason: str) -> dict:
    return {"ok": False, "refused": True, "output": f"[refused] {reason}"}


def _deny(reason: str) -> dict:
    """A refusal of a read-tier command: one short line, and the user can run it themselves."""
    return _refuse(f"{reason}; deneb's `run` is read-only, so run it yourself if you really need it.")


# ── Argument parsing helpers ─────────────────────────────────────────────────
def _canon(name: str, table):
    """GNU-style long option: the full name in `table` that `name` (maybe abbreviated) means."""
    nm = name[2:]
    if nm in table:
        return nm
    hits = [t for t in table if t.startswith(nm)]
    return hits[0] if len(hits) == 1 else None


def _lhit(name: str, banned) -> bool:
    """True when long option `name` (maybe an unambiguous abbreviation, as getopt allows) is
    one of the `banned` full names or a prefix of one."""
    name = name.split("=", 1)[0]
    return name.startswith("--") and len(name) >= 3 and any(b.startswith(name) for b in banned)


def _parse(rest, short_val: str = "", long_val=()):
    """Split argv into ([(flag, value|None)], [positional]). Short clusters are split per
    letter; letters in `short_val` take a value (attached or the next token); long names in
    `long_val` take the next token when no `=value` is attached."""
    flags, pos = [], []
    i, n = 0, len(rest)
    while i < n:
        a = rest[i]
        i += 1
        if a == "--":
            pos.extend(rest[i:])
            break
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            if eq:
                flags.append((name, val))
            elif _canon(name, long_val) and i < n:
                flags.append((name, rest[i]))
                i += 1
            else:
                flags.append((name, None))
        elif a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                c = a[j]
                if c in short_val:
                    v = a[j + 1:]
                    if not v and i < n:
                        v = rest[i]
                        i += 1
                    flags.append(("-" + c, v or None))
                    break
                flags.append(("-" + c, None))
                j += 1
        else:
            pos.append(a)
    return flags, pos


# ── Paths that must never be read or written ─────────────────────────────────
def _resolve(arg: str) -> str:
    p = os.path.expanduser(arg)
    return p if os.path.isabs(p) else os.path.join(os.path.expanduser("~"), p)


def _secret_path_reason(arg: str):
    """Why `arg` (a path as the command will see it, cwd = home) must not be read, or None."""
    if not arg or arg == "-":
        return None
    p = _resolve(arg)
    homes = {os.path.expanduser("~"), os.path.realpath(os.path.expanduser("~"))}
    for c in {os.path.normpath(p), os.path.realpath(p)}:
        if _SECRET_FILE.match(os.path.basename(c)):
            return f"'{os.path.basename(c)}' looks like a secrets file"
        if ".ssh" in c.split(os.sep):
            return "it is inside a .ssh directory"
        for h in homes:
            for d in _SECRET_HOME_DIRS:
                root = os.path.join(h, d)
                if c == root or c.startswith(root + os.sep):
                    return f"it is inside ~/{d}"
    return None


def _first_secret(paths):
    for a in paths:
        why = _secret_path_reason(a)
        if why:
            return a, why
    return None


# ── Per-binary argument policies: each returns a refusal reason or None ───────
def _pol_version_only(binv, rest):
    if not (len(rest) == 1 and rest[0] in _VERSION_ONLY[binv]):
        return f"{binv} can execute code, so only a single version flag is allowed"


def _pol_printenv(rest):
    if not rest:
        return "bare printenv would dump every variable, secrets included; name the variables you need"
    for a in rest:
        if a.startswith("-"):
            return f"printenv flag '{a}' is not allowed"
        if _SECRET_NAME.search(a):
            return f"'{a}' looks like a secret variable"


# short letters / long names that take a value, per content reader
_READERS = {
    "cat": ("", ()),
    "head": ("nc", ("lines", "bytes")),
    "tail": ("ncs", ("lines", "bytes", "sleep-interval", "pid", "max-unchanged-stats")),
    "grep": ("ABCmefdD", ("after-context", "before-context", "context", "max-count", "regexp",
                          "file", "directories", "devices", "include", "exclude", "exclude-dir",
                          "exclude-from", "label", "binary-files", "group-separator")),
    "cut": ("bcdf", ("bytes", "characters", "delimiter", "fields", "output-delimiter")),
    "sort": ("koTtS", ("key", "output", "field-separator", "temporary-directory", "buffer-size",
                       "batch-size", "parallel", "random-source", "compress-program",
                       "files0-from", "sort")),
    "uniq": ("fsw", ("skip-fields", "skip-chars", "check-chars")),
    "file": ("mfeFP", ("magic-file", "files-from", "exclude", "separator", "parameter")),
    "sha256sum": ("", ()),
    "md5sum": ("", ()),
    "wc": ("", ("files0-from",)),
    "stat": ("c", ("format", "printf")),
}
_READERS["egrep"] = _READERS["zgrep"] = _READERS["grep"]


def _pol_reader(binv, rest):
    short_val, long_val = _READERS[binv]
    flags, pos = _parse(rest, short_val, long_val)
    paths = [v for _n, v in flags if v] + list(pos)
    if binv in ("grep", "egrep", "zgrep"):
        paths, pat_given = [], False
        for name, val in flags:
            c = _canon(name, long_val) if name.startswith("--") else None
            if name in ("-r", "-R") or _lhit(name, ("--recursive", "--dereference-recursive")):
                return "recursive grep could read anything under a directory"
            if name == "-d" or c == "directories":
                if val not in ("read", "skip"):
                    return "grep --directories other than read/skip would recurse"
            if name == "-e" or c == "regexp":
                pat_given = True
            elif name == "-f" or c == "file":
                pat_given = True
                paths.append(val or "")
        paths += pos if pat_given else pos[1:]
    if binv == "sort" and any(n == "-o" or _lhit(n, ("--output", "--compress-program"))
                              for n, _v in flags):
        return "sort -o/--output writes a file (and --compress-program runs a program)"
    if binv == "uniq" and len(pos) > 1:
        return "uniq's second operand is an OUTPUT file"
    if binv == "file" and any(n == "-C" or _lhit(n, ("--compile",)) for n, _v in flags):
        return "file -C/--compile writes a magic file"
    hit = _first_secret(paths)
    if hit:
        return f"'{hit[0]}': {hit[1]}, and deneb never reads secrets"


def _pol_tree(rest):
    flags, _pos = _parse(rest, "LPIoHT", ("output", "charset", "filelimit", "sort", "timefmt",
                                          "fromfile"))
    if any(n == "-o" or _lhit(n, ("--output", "--fromfile")) for n, _v in flags):
        return "tree -o/--fromfile writes or reads a file"


def _pol_du(rest):
    if any(_lhit(a, ("--files0-from",)) for a in rest):
        return "du --files0-from reads a list of names from a file"


_CURL_VAL = {"-m", "--max-time", "--connect-timeout", "-H", "--header"}
_CURL_AUTH = re.compile(r"(?i)(localhost|127\.0\.0\.1|\[::1\])(:\d*)?")


def _curl_url_reason(u: str):
    if any(ord(c) < 33 or c == "\\" for c in u):
        return f"'{u}' has characters curl and deneb could read differently"
    rest = u
    if "://" in u:
        scheme, rest = u.split("://", 1)
        if scheme.lower() not in ("http", "https"):
            return f"scheme '{scheme}' is not allowed (http/https only)"
    if not _CURL_AUTH.fullmatch(re.split(r"[/?#]", rest, maxsplit=1)[0]):
        return f"'{u}' is not localhost, 127.0.0.1 or [::1]"


_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def _loopback_env() -> dict:
    """The environment for curl. Policy keeps curl on localhost, so a proxy can only get in
    the way: one set in the environment or in ~/.curlrc took check's probes to the proxy and
    changed check's results. NO_PROXY=* turns off a proxy from either place."""
    env = {k: v for k, v in os.environ.items() if k not in _PROXY_VARS}
    env["NO_PROXY"] = env["no_proxy"] = "*"
    return env


def _pol_curl(rest):
    urls, i = [], 0
    while i < len(rest):
        a = rest[i]
        if a in _CURL_VAL:
            if i + 1 >= len(rest):
                return f"curl {a} needs a value"
            v = rest[i + 1]
            if a in ("-m", "--max-time", "--connect-timeout") and not re.fullmatch(r"\d+(\.\d+)?", v):
                return f"curl {a} needs a number"
            if a in ("-H", "--header") and v.startswith("@"):
                return f"curl {a} @file would read a file"
            i += 2
            continue
        if a.startswith("-") and a != "-":
            if a in ("--head", "--compressed") or re.fullmatch(r"-[sSfiIv]+", a):
                i += 1
                continue
            return (f"curl flag '{a}' is not allowed (only -s -S -f -i -I --head -v -m "
                    "--connect-timeout -H -w --compressed)")
        urls.append(a)
        i += 1
    if not urls:
        return "curl needs an http(s) URL on localhost, 127.0.0.1 or [::1]"
    for u in urls:
        why = _curl_url_reason(u)
        if why:
            return f"curl: {why}; only plain GETs to localhost are allowed"


def _pol_hostnamectl(rest):
    if rest not in ([], ["status"]):
        return "hostnamectl: only bare or 'status' is allowed"


_LOGINCTL_OK = {"list-sessions", "list-users", "list-seats", "show-session", "show-user",
                "session-status", "user-status", "seat-status"}


def _pol_remote(flags):
    for n, _v in flags:
        if n in ("-H", "-M") or _lhit(n, ("--host", "--machine")):
            return "remote/machine targets (-H/-M) are not allowed"


def _pol_loginctl(rest):
    flags, pos = _parse(rest, "pnoHM", ("property", "lines", "output", "host", "machine",
                                       "signal", "kill-who", "kill-whom"))
    why = _pol_remote(flags)
    if why:
        return f"loginctl: {why}"
    if pos and pos[0] not in _LOGINCTL_OK:
        return f"loginctl '{pos[0]}' is not a read-only verb"


_PIP_OK = {"list", "show", "freeze", "check"}


def _pol_pip(rest):
    if rest in (["--version"], ["-V"]):
        return None
    if not rest or rest[0] not in _PIP_OK:
        return "pip: only list, show, freeze, check and --version are allowed"
    if any(_lhit(a, ("--log", "--python", "--cert", "--client-cert", "--keyring-provider",
                     "--proxy", "--index-url", "--extra-index-url", "--find-links"))
           for a in rest):
        return "pip: that option writes a file, runs code or contacts another host"


_JOURNALCTL_BAD = ("--vacuum-size", "--vacuum-time", "--vacuum-files", "--rotate", "--flush",
                   "--sync", "--relinquish-var", "--smart-relinquish-var", "--setup-keys",
                   "--update-catalog", "--cursor-file")


def _pol_journalctl(rest):
    for a in rest:
        if _lhit(a, _JOURNALCTL_BAD):
            return f"journalctl '{a}' changes or writes journal state"


def _pol_sysctl(rest):
    for a in rest:
        if "=" in a:
            return "sysctl with name=value writes a kernel setting"
        if a.startswith("--"):
            if _lhit(a, ("--write", "--load", "--system")):
                return f"sysctl '{a}' writes kernel settings"
        elif a.startswith("-") and set(a[1:]) & set("wpf"):
            return f"sysctl '{a}' writes or loads kernel settings"


_DATE_OK = re.compile(r"-u|--utc|--universal|-R|--rfc-email|-I(date|hours|minutes|seconds|ns)?"
                      r"|--iso-8601(=\w+)?|--rfc-3339=\w+|\+.*", re.S)


def _pol_date(rest):
    for a in rest:
        if not _DATE_OK.fullmatch(a):
            return f"date '{a}' is not allowed (it could set the clock or read a file)"


def _pol_ss(rest):
    for a in rest:
        if a.startswith("--"):
            if _lhit(a, ("--kill", "--diag", "--filter")):
                return f"ss '{a}' kills sockets or writes/reads a file"
        elif a.startswith("-") and set(a[1:]) & set("KDF"):
            return f"ss '{a}' kills sockets or writes/reads a file"


def _pol_rocm(rest):
    for a in rest:
        if (a == "-r" or a.startswith(("--se", "--re")) or _lhit(a, ("--save", "--load", "--autorespond"))):
            return f"rocm-smi '{a}' changes GPU settings or writes a file"


def _pol_ip(rest):
    for a in rest:
        if (a.startswith("-b") and "-batch".startswith(a)) or \
                (len(a) >= 3 and a.startswith("-fo") and "-force".startswith(a)):
            return f"ip '{a}' runs a batch file or forces changes"
    for k, a in enumerate(rest):
        if a in ("netns", "vrf") and any(t and "exec".startswith(t) for t in rest[k + 1:]):
            return f"ip {a} exec runs another program"


def _pol_systemctl(rest):
    flags, pos = _parse(rest, _SYSTEMD_VAL_SHORT, _SYSTEMD_VAL_LONG)
    why = _pol_remote(flags)
    if why:
        return f"systemctl: {why}"
    verb = pos[0] if pos else ""
    if verb == "show-environment":
        return "systemctl show-environment would print the manager environment (secrets)"
    if (verb and verb not in _SYSTEMCTL_OK) or any(p in _SYSTEMCTL_MUTATING for p in pos[1:]):
        bad = verb if verb not in _SYSTEMCTL_OK else next(p for p in pos[1:] if p in _SYSTEMCTL_MUTATING)
        return ("MUTATING:" + bad)


_ANALYZE_OK = {"time", "blame", "critical-chain", "plot", "dot", "cat-config", "unit-files",
               "unit-paths", "exit-status", "syscall-filter", "filesystems", "architectures",
               "capability", "condition", "calendar", "timestamp", "timespan", "verify",
               "security", "inspect-elf", "compare-versions"}


def _pol_analyze(rest):
    flags, pos = _parse(rest, "HM", ("host", "machine", "root", "image", "unit", "profile",
                                    "iterations", "base-time"))
    why = _pol_remote(flags)
    if why:
        return f"systemd-analyze: {why}"
    if not pos:
        return None
    verb = pos[0]
    if verb in ("log-level", "log-target", "service-watchdogs"):
        return None if len(pos) == 1 else f"systemd-analyze {verb} with an argument SETS it"
    if verb not in _ANALYZE_OK:
        return f"systemd-analyze '{verb}' is not an allowed read-only verb"


_POLICY = {
    "printenv": _pol_printenv, "tree": _pol_tree, "du": _pol_du, "curl": _pol_curl,
    "hostnamectl": _pol_hostnamectl, "loginctl": _pol_loginctl, "pip": _pol_pip,
    "pip3": _pol_pip, "journalctl": _pol_journalctl, "sysctl": _pol_sysctl, "date": _pol_date,
    "ss": _pol_ss, "rocm-smi": _pol_rocm, "ip": _pol_ip, "systemd-analyze": _pol_analyze,
}


# A binary may be named by path only from the system's own bin directories. A bare basename
# check let `/tmp/x/ollama ps` run any file that happened to be called `ollama`.
_BIN_DIRS = {"/usr/bin", "/usr/sbin", "/bin", "/sbin", "/usr/local/bin"}
_ROCM_BIN = re.compile(r"/opt/rocm[^/]*/bin")


def _binary_name(argv0: str):
    """(basename, None) for an allowed spelling of the program, or (None, why) when argv[0]
    names a path outside the system bin directories."""
    if "/" not in argv0:
        return argv0, None
    d = os.path.normpath(os.path.dirname(argv0))
    if d in _BIN_DIRS or _ROCM_BIN.fullmatch(d):
        return os.path.basename(argv0), None
    return None, (f"'{argv0}' is not run by path: a program may be named by path only from "
                  "/usr/bin, /usr/sbin, /bin, /sbin, /usr/local/bin or /opt/rocm*/bin")


def run(cmd: str, timeout: int = 25) -> dict:
    """Run a READ-ONLY shell command (no shell, no sudo). Returns
    {ok, output[, refused]}. Refuses anything not provably read-only + unprivileged."""
    if _CONTROL.search(cmd):
        return _deny("the command contains a control or escape character")
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
    binv, why = _binary_name(argv[0])
    if why:
        return _refuse(why)
    if binv not in _READ_ONLY:
        return _deny(f"'{binv}' is not on deneb's read-only allowlist")
    rest = argv[1:]

    if binv == "systemctl":
        reason = _pol_systemctl(rest)
        if reason and reason.startswith("MUTATING:"):
            verb = reason.split(":", 1)[1]
            return _refuse(f"systemctl '{verb}' is a mutating action — 'run' is read-only. "
                           "To apply it, propose it as a fix action instead: "
                           '{"action":"fix","cmd":"systemctl --user ' + verb + ' …"}')
        if reason:
            return _deny(reason)
    if binv == "ollama" and not (len(rest) == 1 and rest[0] in _OLLAMA_OK):
        return _deny("ollama: only 'ollama ps' and 'ollama list' are allowed")
    reason = None
    if binv in _VERSION_ONLY:
        reason = _pol_version_only(binv, rest)
    elif binv in _READERS:
        reason = _pol_reader(binv, rest)
    elif binv in _POLICY:
        reason = _POLICY[binv](rest)
    if reason:
        return _deny(reason)
    bad = _BAD_ARGS.get(binv)
    if bad:
        for a in rest:
            head = a.split("=", 1)[0]
            if a in bad or head in bad:
                return _deny(f"'{a}' would mutate or execute")

    try:
        p = subprocess.run(  # noqa: S603 — shell=False, allowlisted binary, read-only
            argv, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expanduser("~"), env=_loopback_env() if binv == "curl" else None,
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


# ── The APPLY (write) tier — reversible, non-elevated fixes ONLY ──────────────
# Deny-by-default like the read tier. Only THREE mutating binaries are ever allowed, each
# with tight guards, and only for actions that are reversible and need no root. Anything
# destructive / irreversible / privileged is refused here regardless of what the engine
# asks — the client is the boundary. The loop GATES each of these on the user (y/N) unless
# --auto; this module is the second line, enforcing the hard limits either way.
_WRITE_BINS = {"systemctl", "mkdir", "ln"}
_WRITE_SYSTEMCTL_OK = {"daemon-reload", "restart", "try-restart", "start", "stop",
                       "enable", "disable", "reset-failed"}
# Named explicitly so the refusal message is clear (deny-by-default already blocks them).
_DESTRUCTIVE = {
    "rm", "rmdir", "unlink", "mv", "dd", "mkfs", "shred", "truncate", "chmod", "chown",
    "chattr", "kill", "pkill", "killall", "fuser", "reboot", "poweroff", "halt", "shutdown",
    "init", "telinit", "iptables", "ip6tables", "ufw", "firewall-cmd", "nft", "tc",
    "apt", "apt-get", "dnf", "yum", "pacman", "snap", "zypper", "modprobe", "insmod",
    "rmmod", "mount", "umount", "parted", "fdisk", "sfdisk", "wipefs", "mkswap",
    "crontab", "passwd", "useradd", "usermod", "userdel", "groupadd", "visudo",
    "git", "curl", "wget", "pip", "pip3", "make", "cmake", "tee", "sed",
}
# System roots deneb will never write into (they need root anyway; a clear refusal beats
# a raw permission error). Home, ~/.config/systemd/user, /tmp, user-writable /opt are fine.
_DENY_WRITE_ROOTS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/sys",
                     "/proc", "/dev", "/run", "/var", "/root", "/snap")


def run_write(cmd: str, timeout: int = 90) -> dict:
    """Run a REVERSIBLE, NON-ELEVATED fix command. Deny-by-default: only `systemctl --user`
    (safe verbs), `mkdir`, and `ln -s` are permitted, with guards. Refuses everything
    destructive/irreversible/privileged even if asked."""
    if _CONTROL.search(cmd):
        return _refuse("the command contains a control or escape character — refused")
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return _refuse("could not parse the command safely")
    if not argv:
        return _refuse("empty command")
    if argv[0] in ("sudo", "su", "pkexec", "doas"):
        return _refuse("deneb never escalates privileges — run any root step yourself")
    for a in argv:
        if a in _CHAIN_TOKENS:
            return _refuse("no shell chaining or redirection — one command at a time")
        # A metacharacter EMBEDDED in an arg (e.g. `a;`, `$(...)`) — no valid unit name or
        # path contains these, so it's an attempted chain. Refuse (shell=False makes it inert
        # anyway, but stay honest about intent).
        if any(mc in a for mc in (";", "|", "&", "`", "$", ">", "<", "\n")):
            return _refuse("suspicious shell metacharacter in an argument — refused")
    binv, why = _binary_name(argv[0])
    if why:
        return _refuse(why)
    if binv in _DESTRUCTIVE:
        return _refuse(f"'{binv}' is destructive/irreversible or needs root — deneb will NEVER "
                       "run it. If a fix genuinely needs this, do it yourself.")
    if binv not in _WRITE_BINS:
        return _refuse(f"'{binv}' is not on deneb's apply allowlist. Only reversible, "
                       "non-elevated fixes run: `systemctl --user …`, `mkdir`, `ln -s`.")
    rest = argv[1:]
    if binv == "systemctl":
        if "--user" not in rest:
            return _refuse("only `systemctl --user …` is allowed — system-wide services need "
                           "root; run those yourself.")
        flags, pos = _parse(rest, _SYSTEMD_VAL_SHORT, _SYSTEMD_VAL_LONG)
        if _pol_remote(flags):
            return _refuse("systemctl -H/--host/-M/--machine target another host or container — "
                           "refused; run it yourself if you really need it.")
        verb = next((a for a in rest if not a.startswith("-")), "")
        if verb not in _WRITE_SYSTEMCTL_OK:
            return _refuse(f"systemctl '{verb or '(none)'}' is not an allowed apply action "
                           "(allowed: daemon-reload, start, stop, restart, enable, disable, "
                           "reset-failed).")
        if verb == "enable":
            unit_dir = os.path.realpath(os.path.join(os.path.expanduser("~"), ".config/systemd/user"))
            for u in pos[1:]:
                if "/" in u or u.startswith("."):
                    up = os.path.realpath(_resolve(u))
                    if not up.startswith(unit_dir + os.sep):
                        return _refuse(f"`systemctl enable {u}` would install a unit file from outside "
                                       "~/.config/systemd/user — refused; run it yourself if you trust it.")
    if binv == "ln":
        flags = "".join(a[1:] for a in rest if a.startswith("-") and not a.startswith("--"))
        if "s" not in flags and "--symbolic" not in rest:
            return _refuse("only symbolic links are allowed: `ln -s <target> <link>`.")
        if "f" in flags or any(_lhit(a, ("--force",)) for a in rest):
            return _refuse("refusing `ln -f` — it would overwrite an existing path (not reversible).")
        lflags, lpos = _parse(rest, "tS", ("target-directory", "suffix"))
        tdir = next((v for n, v in lflags if n == "-t" or _lhit(n, ("--target-directory",))), None)
        if tdir:
            dests = [os.path.join(tdir, os.path.basename(t)) for t in lpos]
        elif len(lpos) >= 2:
            dests = [lpos[-1]] + ([os.path.join(lpos[-1], os.path.basename(t)) for t in lpos[:-1]]
                                  if os.path.isdir(_resolve(lpos[-1])) else [])
        else:
            dests = [os.path.basename(t) for t in lpos]
        for d in dests + ([tdir] if tdir else []):
            why = _unsafe_link_dest(d)
            if why:
                return _refuse(f"refusing a link at {d}: {why} — do it yourself if you mean it.")
    if binv == "mkdir":
        mflags, _mpos = _parse(rest, "mZ", ("mode", "context"))
        for n, v in mflags:
            if n == "-m" or _lhit(n, ("--mode",)):
                if not (v and re.fullmatch(r"[0-7]{1,4}", v)) or int(v, 8) & 0o7002:
                    return _refuse(f"mkdir mode '{v}' is refused (only plain octal modes without "
                                   "setuid/setgid/sticky/world-write) — do it yourself if you need it.")
    try:
        p = subprocess.run(  # noqa: S603 — shell=False, allowlisted+guarded, non-elevated
            argv, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expanduser("~"),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"(‘{binv}’ timed out after {timeout}s)"}
    except FileNotFoundError:
        return {"ok": False, "output": f"(not installed on this box: {binv})"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "output": f"(could not run {binv}: {e})"}
    out = (p.stdout or "")
    if p.stderr:
        out += ("\n[stderr]\n" + p.stderr)
    out = out.strip() or f"(done — {binv} exited {p.returncode})"
    return {"ok": p.returncode == 0, "output": _redact(out[:8000]), "exit": p.returncode}


_SHELL_STARTUP = {".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".profile",
                  ".zshrc", ".zshenv", ".zprofile", ".zlogin", ".zlogout", ".xprofile",
                  ".xinitrc", ".inputrc", ".pam_environment"}


def _in_secret_dir(p: str):
    """The ~/<dir> (or .ssh) that the already-resolved path `p` sits inside, or None."""
    if ".ssh" in p.split(os.sep):
        return ".ssh"
    for h in {os.path.expanduser("~"), os.path.realpath(os.path.expanduser("~"))}:
        for d in _SECRET_HOME_DIRS:
            root = os.path.join(h, d)
            if p == root or p.startswith(root + os.sep):
                return "~/" + d
    return None


def _is_home_dotfile(p: str) -> bool:
    homes = {os.path.expanduser("~"), os.path.realpath(os.path.expanduser("~"))}
    return os.path.dirname(p) in homes and os.path.basename(p).startswith(".")


def _unsafe_link_dest(d: str):
    """Why a symlink must not be created at `d`, or None."""
    for p in {os.path.normpath(_resolve(d)), os.path.realpath(_resolve(d))}:
        where = _in_secret_dir(p)
        if where:
            return f"it is inside {where}"
        if os.path.basename(p) == "authorized_keys":
            return "authorized_keys controls who can log in"
        if _is_home_dotfile(p):
            return "it is a dotfile (shell/startup config) in your home directory"
        if "/.config/autostart/" in p + "/":
            return "it is an autostart entry"


def _next_backup(p: str) -> str:
    b, i = p + ".deneb-bak", 0
    while os.path.exists(b):
        i += 1
        b = f"{p}.deneb-bak{i}"
    return b


_UNIT_NAME = re.compile(r"\.(service|timer|socket|path)$")
_EXEC_LINE = re.compile(r"^\s*(Exec\w*)\s*=\s*(.*)$")
_UNIT_BAD_PROGRAMS = re.compile(
    r"^(env|busybox|sh|bash|dash|zsh|ksh|csh|tcsh|fish|ash|python[\d.]*|perl[\d.]*|ruby[\d.]*|node|nodejs|php[\d.]*|lua[\d.]*)$")


def _is_unit_path(p: str) -> bool:
    return bool(_UNIT_NAME.search(p)) or "/systemd/" in p.replace(os.sep, "/")


def _unit_exec_problem(p: str, content: str) -> str:
    """Why this would-be systemd unit must not be written ('' when fine). Any Exec*= whose
    program is a shell, env, an interpreter, or not an absolute path can run arbitrary code."""
    if not _is_unit_path(p):
        return ""
    logical = re.sub(r"\\\r?\n", " ", content or "")
    for line in logical.splitlines():
        m = _EXEC_LINE.match(line)
        if not m:
            continue
        rest = m.group(2).lstrip("@-+!:")
        if not rest:
            continue
        try:
            prog = (shlex.split(rest) or [""])[0]
        except ValueError:
            prog = rest.split()[0]
        if not prog.startswith("/"):
            return f"{m.group(1)}= runs `{ui_safe(prog)}`, which is not an absolute path"
        base = os.path.basename(prog)
        if _UNIT_BAD_PROGRAMS.match(base):
            return f"{m.group(1)}= runs {base}, a shell or interpreter"
    return ""


def ui_safe(s: str) -> str:
    return _CONTROL.sub("?", s)[:80]


def write_file(path: str, content: str, max_bytes: int = 262_144) -> dict:
    """Write `content` to `path`, BACKING UP any existing file first (reversible). Refuses
    secret files and system paths. Runs as the user — no sudo."""
    if content is None:
        return _refuse("no content provided to write")
    if _CONTROL.search(str(path)):
        return _refuse("the path contains a control or escape character — refused")
    if len(content.encode("utf-8", "replace")) > max_bytes:
        return _refuse("content too large to write safely")
    p = os.path.realpath(os.path.expanduser(str(path)))
    if _SECRET_FILE.match(os.path.basename(p)):
        return _refuse("refusing to write a secrets file — deneb never touches "
                       "keys/tokens/credentials.")
    where = _in_secret_dir(p)
    if where:
        return _refuse(f"refusing to write inside {where} — keys and credentials are never edited "
                       "by deneb. Do it yourself if you mean it.")
    if _is_home_dotfile(p) and os.path.basename(p) in _SHELL_STARTUP:
        return _refuse(f"refusing to write {os.path.basename(p)}: shell startup files run on every "
                       "login. Edit it yourself if you mean it.")
    for root in _DENY_WRITE_ROOTS:
        if p == root or p.startswith(root + os.sep):
            return _refuse(f"refusing to write under {root} — system paths need root. deneb only "
                           "edits your own files (home, ~/.config/systemd/user, /tmp).")
    bad = _unit_exec_problem(p, content)
    if bad:
        return _refuse(f"{bad}. write that unit yourself; deneb will not plant one that can run "
                       "a shell or an interpreter.")
    backup = ""
    if os.path.lexists(p):
        if not os.path.isfile(p):
            return _refuse(f"{path} exists but is not a regular file — refusing to overwrite.")
        backup = _next_backup(p)
        try:
            shutil.copy2(p, backup)
        except Exception as e:  # noqa: BLE001
            return _refuse(f"couldn't back up {path} before writing ({e}) — aborting to stay safe.")
    else:
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            return {"ok": False, "output": f"(parent directory doesn't exist: {d} — create it "
                    "first with `mkdir -p`)"}
    try:
        with open(p, "w") as f:
            f.write(content)
    except PermissionError:
        return _refuse(f"permission denied writing {path} — deneb runs as you with no sudo; "
                       "do this one yourself if it needs root.")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "output": f"(could not write {path}: {e})"}
    msg = f"wrote {path} ({len(content.splitlines())} lines)"
    if backup:
        msg += f"; backed up the original to {backup} (restore with: cp {backup} {p})"
    return {"ok": True, "output": msg, "backup": backup}


# Dispatch used by the loop: map an engine action to a local execution.
def execute(action: str, cmd: str = "", path: str = "") -> dict:
    if action == "run":
        return run(cmd)
    if action == "read_file":
        return read_file(path)
    if action == "list_dir":
        return list_dir(path)
    return _refuse(f"unknown tool '{action}'")


def execute_write(action: str, cmd: str = "", path: str = "", content: str = "") -> dict:
    """APPLY-tier dispatch — called by the loop ONLY after the user has approved the gate
    (or --auto). Still hard-bounded by the guards above."""
    if action == "fix":
        return run_write(cmd)
    if action == "write_file":
        return write_file(path, content)
    return _refuse(f"unknown apply action '{action}'")
