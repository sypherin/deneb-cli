"""deneb.watch — `deneb watch`, the MAINTAIN mode.

`deneb check` answers "is this box set up?". Watch answers "is it still healthy, and what
changed since I last looked?". It runs the same deterministic scan plus four upkeep probes
(disk, memory, service restarts, catalog age), compares the result with the previous run,
and reports only what moved: a new problem, a worse one, a recovery. Meant for a systemd
timer (see `deneb watch --schedule`), so a box that quietly fills its disk or starts
crash-looping tells you before the model stops answering.

THE DENEB RULE holds here: this module only READS the box (the allowlisted read-only
tools.run, stdlib reads of /proc and the filesystem) and writes only Deneb's OWN state
files (state, history, lock). Fix commands are PRINTED, never run. tests/test_watch.py and
tests/test_deneb_rule.py hold it to that, including a run of the real probes with every
command recorded.

Exit codes: 0 all ok, 1 a warning or a row it could not read, 2 a failure, 3 deneb watch itself broke.

Layout: the first half is pure (status mapping, thresholds, diff, scrubbing, alert text,
payloads, exit code) so it is tested with plain values; the IO (probes, state files,
notify, the command) is below it.
"""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import socket
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

try:                      # file locking is Unix-only; elsewhere watch just runs unlocked
    import fcntl
except ImportError:       # pragma: no cover
    fcntl = None

from . import __version__, check, hf_catalog, tools
from .colour import Palette

_C = Palette(g="green", r="red", y="amber", d="dim", z="reset", b="bold", teal="teal")

GIB = 2 ** 30
MEMINFO = "/proc/meminfo"
CATALOG_MAX_AGE_DAYS = 30
NOTIFY_ENV = "DENEB_WATCH_NOTIFY"
UNSENT_CAP = 4000
MAX_ALERT = 1900   # Discord rejects a message over 2000 characters; Slack and ntfy take more
_RANK = {"ok": 0, "warn": 1, "fail": 2}
_STATUSES = ("ok", "warn", "fail", "unknown")



# ── pure logic ───────────────────────────────────────────────────────────────
_SCRUBS = [
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.\-]*://)[^/\s@]+@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)https?://hooks\.slack\.com/services/\S+"), "https://hooks.slack.com/services/[REDACTED]"),
    (re.compile(r"(?i)https?://(?:[\w\-]+\.)?discord(?:app)?\.com/api/webhooks/\S+"),
     "https://discord.com/api/webhooks/[REDACTED]"),
    (re.compile(r"(?i)(--(?:api-key|token|key)[ =]\s*)[^\s'\"]+"), r"\1[REDACTED]"),
]


def scrub(text: str) -> str:
    """Everything deneb already redacts (keys, bearer tokens, JWTs), plus what a command line
    or log line can carry that a webhook or login needs kept secret: webhook URLs, URL
    userinfo (user:pass@) and space-separated `--api-key VALUE` / `--token VALUE`."""
    text = str(text)
    for pat, repl in _SCRUBS:
        text = pat.sub(repl, text)
    return tools._redact(text)


def status_of(ok: bool, evidence: str) -> str:
    """A check row's status. check marks a row ✓ with a ⚠ in its evidence when it works
    but has a catch (e.g. the service won't start after a reboot): that is a warning. A row
    that doesn't apply (ok is None: an optional part that isn't set up, or a fact this
    server can't give) is fine, and one that could not be read (check.UNKNOWN) is unknown:
    never a pass, so it can never show as a recovery."""
    if ok == check.UNKNOWN:
        return "unknown"
    if ok is None:
        return "ok"
    if not ok:
        return "fail"
    return "warn" if "⚠" in evidence else "ok"


def finding(id: str, label: str, status: str, evidence: str, fix: str = "") -> dict:
    return {"id": id, "label": label, "status": status, "evidence": scrub(evidence), "fix": fix}


_OPTIONAL_WHERE = {"gateway": ":8002", "tunnel": "cloudflared"}
_RESET_HINT = "start it again; if you removed it on purpose, run deneb watch --reset"


def findings_from_rows(rows, seen=()) -> list:
    """check.scan() rows -> findings. An optional row (gateway, tunnel) that isn't set up is
    fine, unless `seen` (the optional ids watch has seen running before) says it used to be
    there: then its absence is a failure. A present optional row is marked seen=True so the
    caller can remember it."""
    out = []
    for rid, label, ok, ev, fix in rows:
        if rid in check._OPTIONAL and ok is None and rid in seen:
            f = finding(rid, label, "fail",
                        f"was running before, now not found ({_OPTIONAL_WHERE.get(rid, rid)})", _RESET_HINT)
        else:
            f = finding(rid, label, status_of(ok, ev), ev, "" if ok is None else fix)
            if ok is None:
                f["na"] = True       # not applicable: shown as "–", counted as neither ok nor problem
            if rid in check._OPTIONAL and ok is not None:
                f["seen"] = True
        out.append(f)
    return out


def disk_status(free: int, total: int) -> str:
    """warn below 15% or 20 GiB free, fail below 5% or 5 GiB free (bytes in)."""
    pct = 100.0 * free / total if total else 0.0
    if pct < 5 or free < 5 * GIB:
        return "fail"
    if pct < 15 or free < 20 * GIB:
        return "warn"
    return "ok"


def memory_status(avail: int) -> str:
    """warn below 8 GiB available, fail below 3 GiB (bytes in)."""
    if avail < 3 * GIB:
        return "fail"
    return "warn" if avail < 8 * GIB else "ok"


def catalog_status(age_seconds) -> str:
    """The catalog cache is only a refresh of what ships with Deneb: stale is a nudge."""
    if age_seconds is None:
        return "ok"
    return "warn" if age_seconds > CATALOG_MAX_AGE_DAYS * 86400 else "ok"


def restarts_status(delta: int) -> str:
    """+1/+2 restarts since the last run is worth a look, +3 or more is a crash loop."""
    return "fail" if delta >= 3 else ("warn" if delta >= 1 else "ok")


def diff(prev, cur: list) -> list:
    """Compare the previous run (a {id: status} dict, or None on a first run) with the
    current findings. One change per finding, with a kind:
      new-problem  ok/absent -> warn/fail      worse      warn -> fail
      better       fail -> warn                recovered  warn/fail -> ok
      unchanged    everything else
    An "unknown" reading never changes anything: the previous status stands, because a
    probe that could not read is not evidence that the problem went away."""
    prev = prev or {}
    out = []
    for f in cur:
        old = prev.get(f["id"])
        new = f["status"]
        if new == "unknown":
            new = old if old in _RANK else "ok"
        was = old if old in _RANK else "ok"
        if was == "ok" and new in ("warn", "fail"):
            kind = "new-problem"
        elif was == "warn" and new == "fail":
            kind = "worse"
        elif was == "fail" and new == "warn":
            kind = "better"
        elif was in ("warn", "fail") and new == "ok":
            kind = "recovered"
        else:
            kind = "unchanged"
        out.append({"id": f["id"], "label": f["label"], "kind": kind,
                    "from": old if old in _RANK else None, "to": new,
                    "evidence": f["evidence"], "fix": f["fix"]})
    return out


def merge_unknown(prev, cur: list) -> list:
    """The findings to SAVE: an unknown reading keeps the previous status, so a probe that goes
    blind while a fail is stored does not turn green. The exit code also scores the current
    findings, so an unreadable row never exits 0 (see _run)."""
    prev = prev or {}
    merged = []
    for f in cur:
        if f["status"] == "unknown" and prev.get(f["id"]) in _RANK:
            f = dict(f, status=prev[f["id"]])
        merged.append(f)
    return merged


def exit_code(findings: list) -> int:
    """0 all ok, 1 worst is warn or a row could not be read (unknown scores 1, as in `deneb
    check`: not a pass, not a fail), 2 any fail."""
    return max([_RANK.get(f["status"], 1) for f in findings] or [0])


_ALERT_KINDS = ("new-problem", "worse", "recovered")
_KIND_TAG = {"new-problem": "NEW", "worse": "WORSE", "better": "BETTER", "recovered": "RECOVERED"}


def alert_text(changes: list):
    """A short plain-text message, or None when nothing worth waking someone for changed
    (a problem that merely got better is not). Scrubbed before it can leave the box."""
    if not any(c["kind"] in _ALERT_KINDS for c in changes):
        return None
    lines = [f"deneb watch on {socket.gethostname()}:"]
    for c in changes:
        if c["kind"] == "unchanged":
            continue
        lines.append(f"- {_KIND_TAG[c['kind']]} {c['label']} [{c['to']}]: {c['evidence']}")
        if c["to"] in ("warn", "fail") and c["fix"] and f"  fix: {c['fix']}" not in lines:
            lines.append(f"  fix: {c['fix']}")
    return scrub("\n".join(lines))


def compose_alert(unsent, text, limit: int = MAX_ALERT) -> str:
    """The message to send: an earlier alert that never got through, then the new one, in at
    most `limit` characters. The oldest part goes first, so the newest problem always fits;
    a new alert longer than the limit on its own is cut at the end."""
    head = "earlier alert not delivered:\n"
    if text is not None and len(text) >= limit:
        return text[:limit - 1] + "\u2026"
    if not unsent:
        return text or ""
    room = limit - len(head) - (len(text) + 2 if text is not None else 0)
    if room < 40:  # no useful space left for the backlog
        return text
    old = unsent if len(unsent) <= room else "\u2026" + unsent[-(room - 1):]
    return head + old + ("\n\n" + text if text is not None else "")


def build_payload(url: str, text: str):
    """(body, content-type) in the shape the target expects: Slack wants {"text"}, Discord
    {"content"}, anything else (ntfy-style) takes a plain-text body."""
    u = urllib.parse.urlparse(url)
    host = (u.hostname or "").lower()
    if host == "hooks.slack.com":
        return json.dumps({"text": text}).encode(), "application/json"
    if (host == "discord.com" and u.path.startswith("/api/webhooks")) or \
            host == "discordapp.com" or host.endswith(".discordapp.com"):
        return json.dumps({"content": text}).encode(), "application/json"
    return text.encode(), "text/plain; charset=utf-8"


def now_iso() -> str:
    """ISO 8601 with the local UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ── probes (read-only) ───────────────────────────────────────────────────────
def _model_path():
    """The model file's path as the serving process was started with it, or None."""
    ps = check._serving_cmdline()
    m = (re.search(r"\s(?:-m|--model)[ =](/\S+)", " " + ps)
         or re.search(r"\s(/\S+\.gguf)(?!\S)", " " + ps))
    return m.group(1) if m else None


def probe_disk(model_path=None) -> dict:
    """Free space on the filesystem holding the model (home dir if its path is unknown)."""
    where = os.path.dirname(model_path) if model_path and os.path.exists(model_path) else ""
    where = where or os.path.expanduser("~")
    try:
        u = shutil.disk_usage(where)
    except OSError as e:
        return finding("disk", "Disk space", "unknown", f"couldn't read free space ({e})")
    st = disk_status(u.free, u.total)
    pct = 100.0 * u.free / u.total if u.total else 0.0
    return finding("disk", "Disk space", st, f"{u.free / GIB:.1f} GiB free ({pct:.0f}%) on {where}",
                   "" if st == "ok" else
                   f"see what is big:  du -sh {where}/* | sort -h | tail -15")


def probe_memory() -> dict:
    """MemAvailable from /proc/meminfo (Linux only)."""
    try:
        with open(MEMINFO) as fh:
            text = fh.read()
    except OSError:
        return finding("memory", "Memory", "unknown", "can't read /proc/meminfo (not Linux?)")
    m = re.search(r"^MemAvailable:\s+(\d+)\s*kB", text, re.M)
    if not m:
        return finding("memory", "Memory", "unknown", "MemAvailable not reported")
    avail = int(m.group(1)) * 1024
    st = memory_status(avail)
    return finding("memory", "Memory", st, f"{avail / GIB:.1f} GiB available",
                   "" if st == "ok" else "see the biggest processes:  ps aux --sort=-rss | head")


def probe_restarts(prev_count, prev_unit=None):
    """(finding, count, unit) for the systemd unit serving the model (a --user unit, or a
    system unit such as ollama.service). NRestarts only goes up, so it is compared with the
    count saved by the previous run FOR THE SAME UNIT: a different unit (the model was moved
    to a new service) is a first check, not a delta. The returned/stored unit carries its
    scope: a --user unit keeps its plain name, a system unit is "system:<name>", so the two
    are never confused. prev_unit None means "not recorded", taken as the same unit. When
    nothing can be read, the old count and unit are handed back so the next run still has a
    baseline."""
    label = "Service restarts"
    port = check.serving_port()
    pid, _proc = check._port_owner(port)
    if pid == "?":
        # The socket's owner is another user's: the unit can't be read from a pid. If the API
        # says it is Ollama and ollama.service is active, that system unit is the answer.
        name, scope = check._hidden_unit(port)
    else:
        name, scope = check._scoped_unit_of_pid(pid) if pid else check.hinted_unit()
        # nothing listening: the stored unit (what the endpoint row names) is still the answer
    if not name:
        why = (f"no model server is listening on :{port}, so there are no restarts to count"
               if not pid else f"the :{port} server isn't a systemd unit, so restarts can't be counted")
        return finding("restarts", label, "unknown", why), prev_count, prev_unit
    system = scope == "system"
    unit = f"system:{name}" if system else name
    shown = f"{name} (system)" if system else name
    out = tools.run(f"systemctl {'' if system else '--user '}show {name} -p NRestarts --value"
                    ).get("output", "").strip()
    if not out.isdigit():
        return finding("restarts", label, "unknown", f"couldn't read NRestarts for {shown}"), \
            prev_count, prev_unit
    n = int(out)
    fix = f"see why:  journalctl {'' if system else '--user '}-u {name} -e"
    if prev_count is None or (prev_unit is not None and prev_unit != unit):
        return finding("restarts", label, "ok", f"{shown}: {n} restarts so far (first check)"), n, unit
    delta = n - prev_count
    if delta < 0:
        return finding("restarts", label, "ok",
                       f"{shown}: counter reset ({n}, was {prev_count}); the service was "
                       "restarted fresh or the box rebooted"), n, unit
    st = restarts_status(delta)
    if st == "ok":
        return finding("restarts", label, "ok", f"{shown}: no restarts since the last check"), n, unit
    times = f"{delta} time" + ("" if delta == 1 else "s")
    ev = (f"{shown} crash-looping: restarted {times} since the last check" if st == "fail"
          else f"{shown} restarted {times} since the last check")
    return finding("restarts", label, st, ev, fix), n, unit


def probe_catalog(now=None) -> dict:
    """Age of Deneb's model-catalog cache (what `deneb recommend` ranks from)."""
    label = "Model catalog"
    try:
        mtime = os.path.getmtime(hf_catalog.CACHE_PATH)
    except OSError:
        return finding("catalog", label, "ok", "no refreshed cache, using the bundled list")
    age = (time.time() if now is None else now) - mtime
    st = catalog_status(age)
    ev = f"cache is {int(age // 86400)} days old"
    return finding("catalog", label, st, ev, "" if st == "ok" else "deneb catalog --refresh")


def gather(prev_restarts, prev_unit=None, seen=()):
    """Run the check scan and the four probes. Returns (findings, restart_count, unit). A
    probe that raises becomes an "unknown" finding: one broken reader never stops the run.
    `seen` is the optional rows watch has seen running before (see findings_from_rows)."""
    fs = findings_from_rows(check.scan(), seen)

    def guarded(id, label, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — see docstring
            return finding(id, label, "unknown", f"probe error: {type(e).__name__}")
    try:
        path = _model_path()
    except Exception:  # noqa: BLE001
        path = None
    fs.append(guarded("disk", "Disk space", lambda: probe_disk(path)))
    fs.append(guarded("memory", "Memory", probe_memory))
    count, unit = prev_restarts, prev_unit
    try:
        f, count, unit = probe_restarts(prev_restarts, prev_unit)
    except Exception as e:  # noqa: BLE001
        f = finding("restarts", "Service restarts", "unknown", f"probe error: {type(e).__name__}")
    fs.append(f)
    fs.append(guarded("catalog", "Model catalog", probe_catalog))
    return fs, count, unit


# ── state + history (Deneb's own files only) ─────────────────────────────────
def state_dir() -> str:
    d = os.environ.get("DENEB_STATE_DIR")
    if d:
        return d
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return os.path.join(xdg, "deneb")
    return os.path.join(os.path.expanduser("~"), ".local", "state", "deneb")


def _state_path() -> str:
    return os.path.join(state_dir(), "watch-state.json")


def _tmp_path() -> str:
    return _state_path() + ".tmp"


def _history_path() -> str:
    return os.path.join(state_dir(), "history.jsonl")


def _lock_path() -> str:
    return os.path.join(state_dir(), "watch.lock")


def _ensure_dir() -> str:
    """The state dir, private to the user (0700): history holds evidence from the box.
    An existing dir is tightened only when it is deneb's own; a dir chosen with
    DENEB_STATE_DIR (it could be $HOME) keeps its mode, and the files in it are 0600."""
    d = state_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    if os.environ.get("DENEB_STATE_DIR"):
        return d
    try:
        st = os.stat(d)
        if st.st_uid == os.getuid() and (st.st_mode & 0o777) != 0o700:
            os.chmod(d, 0o700)
    except (OSError, AttributeError):
        pass
    return d


def _warn(msg: str) -> None:
    print(f"deneb: warning: {msg}", file=sys.stderr)


def _oserr(e: OSError) -> str:
    return f"{type(e).__name__}: {e.strerror or e}"


def load_state():
    """The last run, validated, or None ("seen" is [] for a file that has none). A missing file is a normal first run. A file that
    cannot be used (not JSON, wrong shape) is reset with one stderr line, so a bad state
    file can never wedge every later run. Entries that are not {status: ok|warn|fail|unknown}
    are dropped; the restart count must be an int, else it is forgotten."""
    try:
        with open(_state_path(), errors="replace") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        _warn("the old watch state was unreadable and was reset (treating this as a first run)")
        return None
    if not isinstance(data, dict) or not isinstance(data.get("findings"), dict):
        _warn("the old watch state was unreadable and was reset (treating this as a first run)")
        return None
    clean, dropped = {}, 0
    for k, v in data["findings"].items():
        if isinstance(v, dict) and v.get("status") in _STATUSES:
            clean[str(k)] = {"status": v["status"], "evidence": str(v.get("evidence", ""))}
        else:
            dropped += 1
    if dropped:
        _warn(f"ignored {dropped} bad entr{'y' if dropped == 1 else 'ies'} in the old watch state")
    r = data.get("restarts")
    unit, unsent = data.get("restart_unit"), data.get("unsent")
    seen = data.get("seen")      # absent in state files written before optional rows existed
    if isinstance(seen, list):
        seen = sorted({x for x in seen if isinstance(x, str)})
    else:
        # An older state file: the optional rows it stored as running (ok or warn) were seen
        # running. Without this, a gateway that dies before the first new-version run is
        # never reported. A stored fail proves nothing (a box that never had one also says
        # fail), so it does not seed.
        seen = sorted(k for k, v in clean.items()
                      if k in check._OPTIONAL and v["status"] in ("ok", "warn")
                      and not v["evidence"].startswith("not set up"))
    port = data.get("port")
    return {"seen": seen,
            "port": port if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535 else None,
            "ts": data.get("ts") if isinstance(data.get("ts"), str) else None,
            "restarts": r if isinstance(r, int) and not isinstance(r, bool) else None,
            "restart_unit": unit if isinstance(unit, str) else None,
            "unsent": unsent if isinstance(unsent, str) and unsent else None,
            "findings": clean}


def save_state(findings: list, restarts, ts: str, unit=None, unsent=None, seen=None, port=None) -> None:
    """Atomic: write a temp file beside the state file, then os.replace, so a crash or a
    full disk never leaves half a state file. (The run lock means one writer, so the temp
    name is fixed; a failed write leaves it to be overwritten by the next run.)"""
    _ensure_dir()
    data = {"ts": ts, "restarts": restarts, "restart_unit": unit,
            "seen": sorted(set(seen or [])), "port": port,
            "unsent": scrub(unsent)[-UNSENT_CAP:] if unsent else None,
            "findings": {f["id"]: {"status": f["status"], "evidence": scrub(f["evidence"])}
                         for f in findings}}
    fd = os.open(_tmp_path(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(_tmp_path(), _state_path())


def _append_lines(events: list) -> None:
    if not events:
        return
    _ensure_dir()
    fd = os.open(_history_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def change_events(changes: list, ts: str) -> list:
    """One history event per CHANGE (nothing for unchanged findings)."""
    return [{"ts": ts, "id": c["id"], "from": c["from"], "to": c["to"],
             "evidence": scrub(c["evidence"])} for c in changes if c["kind"] != "unchanged"]


def append_history(changes: list, ts: str) -> None:
    _append_lines(change_events(changes, ts))


def read_history(n: int) -> list:
    """The last n events. Bad lines (a torn write, a stray byte, valid JSON that is not an
    event) are skipped, never fatal."""
    try:
        with open(_history_path(), errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out[-n:] if n > 0 else []


def _acquire_lock():
    """("ok", fd) when this run holds the lock, ("busy", None) when another watch has it,
    ("none", None) when locking isn't possible here (then it just runs unlocked)."""
    if fcntl is None:
        return "none", None
    try:
        _ensure_dir()
        fd = os.open(_lock_path(), os.O_WRONLY | os.O_CREAT, 0o600)
    except OSError:
        return "none", None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        return ("busy", None) if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK) else ("none", None)
    return "ok", fd


# ── notify (opt-in) ──────────────────────────────────────────────────────────
def _safe_host(url: str) -> str:
    """The host only, for messages. Webhook URLs are secrets: never the userinfo, path or
    query, and never an exception's own text, which can quote them."""
    try:
        return urllib.parse.urlparse(url).hostname or "the notify URL"
    except Exception:  # noqa: BLE001
        return "the notify URL"


def send_notify(url: str, text: str):
    """POST the alert. Returns None on success, else a short error that is safe to print and
    store: the exception CLASS and the host, or the HTTP status. Never str(e), never the URL."""
    try:
        pu = urllib.parse.urlparse(url)
        if pu.scheme not in ("http", "https"):
            return "notify URL must start with http:// or https://"
        from .net import is_private_host
        if pu.scheme == "http" and not is_private_host(pu.hostname or ""):
            return "notify URL must be https:// (plain http only for a host on your own network)"
        body, ctype = build_payload(url, text)
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": ctype, "User-Agent": f"deneb-cli/{__version__}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read(65536)
        return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} from {_safe_host(url)}"
    except Exception as e:  # noqa: BLE001 — a failed send must never crash the run
        return f"{type(e).__name__} reaching {_safe_host(url)}"


def _notify_step(url: str, message: str):
    """send_notify, with a last guard so nothing in the notify path can crash the run."""
    try:
        return send_notify(url, message)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__} while sending the notification"


# ── the command ──────────────────────────────────────────────────────────────
def _mark(status: str, p: dict) -> str:
    sym = {"ok": ("g", "✓"), "warn": ("y", "⚠"), "fail": ("r", "✗"), "unknown": ("d", "?")}[status]
    return f"{p[sym[0]]}{sym[1]}{p['z']}"


def _tag(kind: str, p: dict) -> str:
    t = _KIND_TAG.get(kind)
    if not t:
        return ""
    color = p["g"] if kind in ("recovered", "better") else p["r"]
    return f" {color}{p['b']}{t}{p['z']}"


def _summary(findings: list) -> str:
    n = {s: sum(1 for f in findings if f["status"] == s and not f.get("na")) for s in _STATUSES}
    return f"{n['ok']} ok, {n['warn']} warn, {n['fail']} fail, {n['unknown']} unknown"


def _print_history(n: int) -> int:
    p = _C
    events = read_history(n)
    if not events:
        print(f"{p['d']}no history yet. Run `deneb watch` first.{p['z']}")
        return 0
    for e in events:
        print(f"{e.get('ts', '?')}  {e.get('id', '?')}: {e.get('from') or 'new'} -> "
              f"{e.get('to', '?')}  {p['d']}{scrub(e.get('evidence', ''))}{p['z']}")
    return 0


# `--every 15m`: a positive whole number plus s, m or h, at least a minute. Anything else
# is refused rather than guessed, because the value goes straight into a systemd timer.
_EVERY = re.compile(r"([1-9][0-9]*)([smh])")
_EVERY_SECS = {"s": 1, "m": 60, "h": 3600}


def check_every(every: str):
    """None when `every` is usable, else the reason."""
    m = _EVERY.fullmatch(every)
    if not m:
        return "--every must be a number plus s, m or h (e.g. 90s, 15m, 2h)"
    if int(m.group(1)) * _EVERY_SECS[m.group(2)] < 60:
        return "--every: the minimum is 1m"
    return None


def _exec_path() -> str:
    """Where the unit should find deneb: the absolute path, % escaped (systemd reads %x as
    a specifier) and quoted when it has whitespace. The fallback is pipx's usual spot."""
    exe = shutil.which("deneb")
    if not exe:
        return "%h/.local/bin/deneb"
    exe = exe.replace("%", "%%")
    return f'"{exe}"' if re.search(r"\s", exe) else exe


def schedule_text(every: str) -> str:
    """The systemd --user service + timer for `deneb watch --quiet`, and how to install
    them. TELL-ONLY: this returns text; nothing is written or run. Only headings and notes
    are coloured: command and file lines stay plain so the block pastes and pipes cleanly."""
    p = _C
    b, d, z = p["b"], p["d"], p["z"]
    service = ("[Unit]\nDescription=deneb watch (health check)\n\n"
               "[Service]\nType=oneshot\nEnvironmentFile=-%h/.config/deneb/watch.env\n"
               f"ExecStart={_exec_path()} watch --quiet\nSuccessExitStatus=1 2\n")
    timer = (f"[Unit]\nDescription=run deneb watch every {every}\n\n"
             f"[Timer]\nOnBootSec=5min\nOnUnitActiveSec={every}\n\n"
             "[Install]\nWantedBy=timers.target\n")
    return "\n".join([
        f"\n{b}Run deneb watch every {every}{z}",
        f"{d}deneb printed this and ran nothing. Exit codes: 0 ok, 1 a warning or a row it could not read, 2 failure, 3 deneb watch "
        f"itself broke.{z}",
        f"{d}SuccessExitStatus=1 2 keeps warnings and failures from marking the unit failed: "
        f"problems reach you through the alert (step 3) or the journal. A unit shown as failed in "
        f"`systemctl --user --failed` means the watcher itself broke.{z}",
        f"\n{b}1. Write the two unit files{z}",
        f"{d}   a oneshot service that runs the check, and a timer that fires it{z}",
        "mkdir -p ~/.config/systemd/user",
        f"cat > ~/.config/systemd/user/deneb-watch.service <<'EOF'\n{service}EOF",
        f"cat > ~/.config/systemd/user/deneb-watch.timer <<'EOF'\n{timer}EOF",
        f"\n{b}2. Load and start the timer{z}",
        "systemctl --user daemon-reload",
        "systemctl --user enable --now deneb-watch.timer",
        f"\n{b}3. Optional: get a message when a check breaks, gets worse or recovers{z}",
        f"{d}   keep the URL in a file only you can read, not on the command line (argv shows in "
        f"`ps` and in shell history). It is a secret. These steps do not overwrite an existing "
        f"file; add the line with an editor, or append it and then replace the placeholder:{z}",
        "mkdir -p ~/.config/deneb",
        "touch ~/.config/deneb/watch.env",
        "chmod 600 ~/.config/deneb/watch.env",
        "echo 'DENEB_WATCH_NOTIFY=<your-webhook-url>' >> ~/.config/deneb/watch.env",
        f"\n{b}4. Keep it running while you are logged out{z}",
        "loginctl enable-linger \"$(id -un)\"",
        f"\n{b}Check it{z}",
        "systemctl --user list-timers deneb-watch.timer",
        "journalctl --user -u deneb-watch.service -e\n",
    ])


def unschedule_text() -> str:
    """The undo for `--schedule`: the commands that disable and remove the timer and service
    it told you to create. TELL-ONLY like --schedule: nothing is run or deleted here."""
    from .setup_advisor import _SERVICE_NAME as unit
    b, d, z = _C["b"], _C["d"], _C["z"]
    return "\n".join([
        f"\n{b}Stop and remove the deneb watch timer{z}",
        f"{d}deneb printed this and ran nothing. These undo the steps of `deneb watch --schedule`.{z}",
        f"\n{b}1. Stop the timer and turn it off{z}",
        "systemctl --user disable --now deneb-watch.timer",
        # reset-failed on a unit that never failed exits 1 with "not loaded": say when it applies
        f"{d}   only if `systemctl --user --failed` lists deneb-watch.service (a run that crashed):{z}",
        "systemctl --user reset-failed deneb-watch.service",
        f"\n{b}2. Remove the two unit files{z}",
        "rm ~/.config/systemd/user/deneb-watch.timer ~/.config/systemd/user/deneb-watch.service",
        "systemctl --user daemon-reload",
        f"\n{b}3. Optional: only if you set these up for deneb watch{z}",
        f"{d}   the alert settings file (it holds your webhook URL), and lingering, which kept "
        f"user services running while you were logged out. Turning lingering off would also stop "
        f"the model server unit that deneb setup creates ({unit}) when you log out, and "
        f"any other user service of yours that relies on it:{z}",
        "rm ~/.config/deneb/watch.env",
        "loginctl disable-linger $USER\n",
    ])


_USAGE = """usage: deneb watch [--quiet] [--notify URL]
       deneb watch --history [N]
       deneb watch --schedule [--every 15m]
       deneb watch --unschedule   print the commands that remove what --schedule set up
       deneb watch --reset      forget the saved baseline (use after removing a gateway or tunnel on purpose)"""


def parse_args(argv: list):
    """(options, None) or (None, error). Strict: a typo like `--quite` is an error, not a
    silent full run (which a timer would never notice)."""
    o = {"quiet": False, "history": None, "schedule": False, "every": None, "notify": None,
         "reset": False, "unschedule": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        if a == "--quiet":
            o["quiet"] = True
        elif a == "--schedule":
            o["schedule"] = True
        elif a == "--reset":
            o["reset"] = True
        elif a == "--unschedule":
            o["unschedule"] = True
        elif a == "--history":
            o["history"] = 20
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                v = argv[i + 1]
                if not re.fullmatch(r"[0-9]+", v) or int(v) < 1:
                    return None, "--history N needs a positive whole number"
                o["history"] = int(v)
                i += 1
        elif name in ("--every", "--notify"):
            if not eq:
                if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                    return None, f"{name} needs a value"
                i += 1
                val = argv[i]
            if not val:
                return None, f"{name} needs a value"
            o[name[2:]] = val
        else:
            if not a.startswith("-"):
                return None, ("unexpected argument (not shown, in case it is a webhook URL); "
                              "a notify URL goes after --notify or in DENEB_WATCH_NOTIFY")
            return None, f"unknown option {name!r}"
        i += 1
    if o["every"] is not None and not o["schedule"]:
        return None, "--every only applies with --schedule"
    if o["reset"] and (o["quiet"] or o["history"] is not None or o["schedule"] or o["notify"]):
        return None, "--reset takes no other option"
    if o["unschedule"] and (o["quiet"] or o["history"] is not None or o["schedule"] or o["notify"]
                            or o["reset"] or o["every"] is not None):
        return None, "--unschedule takes no other option"
    return o, None


def _run(quiet: bool, url: str) -> int:
    prev_state = load_state()
    prev = {k: v["status"] for k, v in (prev_state or {}).get("findings", {}).items()}
    unsent = (prev_state or {}).get("unsent")
    prev_seen = (prev_state or {}).get("seen", [])
    # An optional row stored as fail that was never seen running (an old state file, or a
    # box that never had a gateway) is no baseline: it going away is not a recovery.
    prev = {k: v for k, v in prev.items() if not (k in check._OPTIONAL and v == "fail" and k not in prev_seen)}
    # Where the server was last seen: a stopped server is then reported on ITS port and unit.
    check.set_hint((prev_state or {}).get("port"), (prev_state or {}).get("restart_unit") or "")
    findings, count, unit = gather((prev_state or {}).get("restarts"),
                                   (prev_state or {}).get("restart_unit"), prev_seen)
    seen = sorted(set(prev_seen) | {f["id"] for f in findings if f.get("seen")})
    # The port the model was served on. If it changed since the last run, say so: a dead
    # primary model otherwise looks like a healthy one on another port.
    port, old_port = check.resolved_port(), (prev_state or {}).get("port")
    check.clear_hint()
    if port is not None and old_port is not None and port != old_port:
        findings.append(finding("port", "Model server port", "warn",
                                f"model server moved from :{old_port} to :{port} "
                                f"(if :{old_port} was your model, it is down)",
                                f"check what serves :{old_port}, or run deneb watch --reset "
                                "if the move is intended"))
    changes = diff(prev, findings)
    ts = now_iso()
    events = change_events(changes, ts)

    p = _C
    kinds = {c["id"]: c["kind"] for c in changes}
    shown = set()                           # one fault often fails two rows with the same fix
    for f in findings:
        k = kinds.get(f["id"], "unchanged")
        if quiet and k == "unchanged":
            continue
        mark = f"{p['d']}–{p['z']}" if f.get("na") else _mark(f["status"], p)
        print(f"  {mark} {f['label']}{_tag(k, p)}  {p['d']}— {scrub(f['evidence'])}{p['z']}")
        if f["status"] in ("warn", "fail") and f["fix"]:
            print(f"      {p['y']}fix:{p['z']} {'same as above' if f['fix'] in shown else f['fix']}")
            shown.add(f["fix"])
    print(f"deneb watch: {_summary(findings)}")

    # Notify (opt-in). An alert that fails to send is kept in state ("unsent") and goes out
    # with the next run, so a webhook outage cannot silently swallow a problem.
    text = alert_text(changes)
    new_unsent = unsent
    if url and (text is not None or unsent):
        err = _notify_step(url, compose_alert(unsent, text))
        if err:
            _warn(f"notify to {_safe_host(url)} failed: {err}")
            events.append({"ts": ts, "id": "notify", "from": None, "to": "error", "evidence": scrub(err)})
            new_unsent = "\n\n".join(x for x in (unsent, text) if x)[-UNSENT_CAP:]
        else:
            new_unsent = None

    merged = merge_unknown(prev, findings)
    try:
        save_state(merged, count, ts, unit, new_unsent, seen, port if port is not None else old_port)
    except OSError as e:
        _warn(f"couldn't save the watch state ({_oserr(e)}); the next run can't compare with this one")
    try:
        _append_lines(events)
    except OSError as e:
        _warn(f"couldn't write the watch history ({_oserr(e)})")
    # The SAVED state keeps a stored status over a blind probe, but the exit code also scores
    # what was actually seen now: an unreadable row is at least 1, never a quiet 0.
    return max(exit_code(merged), exit_code(findings))


def _reset() -> int:
    """`deneb watch --reset`: forget the baseline (the stored findings and the list of optional
    components seen running) so a component removed on purpose stops being reported. History
    is kept, and gets one event saying so. Same lock, same atomic state write as a normal run;
    still deneb's own files only."""
    if not os.path.exists(_state_path()):
        print("deneb watch: no saved baseline yet, nothing to reset")
        return 0
    old = load_state() or {}
    forgot_seen, forgot_n = old.get("seen", []), len(old.get("findings", {}))
    ts = now_iso()
    try:
        save_state([], old.get("restarts"), ts, old.get("restart_unit"), old.get("unsent"), [], None)
        _append_lines([{"ts": ts, "id": "reset", "from": None, "to": "reset",
                        "evidence": "baseline reset with deneb watch --reset"}])
    except OSError as e:
        print(f"deneb watch: couldn't reset ({_oserr(e)})", file=sys.stderr)
        return 3
    names = f" ({', '.join(forgot_seen)})" if forgot_seen else ""
    print(f"deneb watch: baseline reset. Forgot {forgot_n} stored finding(s) and "
          f"{len(forgot_seen)} remembered optional component(s){names}. "
          "History is kept (see deneb watch --history). The next run starts fresh.")
    return 0


def _main(argv: list) -> int:
    o, err = parse_args(argv)
    if err:
        print(f"deneb watch: {err}\n{_USAGE}", file=sys.stderr)
        return 2
    if o["history"] is not None:
        return _print_history(o["history"])
    if o["unschedule"]:
        print(unschedule_text())
        return 0
    if o["schedule"]:
        every = o["every"] or "15m"
        bad = check_every(every)
        if bad:
            print(f"deneb watch: {bad}", file=sys.stderr)
            return 2
        print(schedule_text(every))
        return 0
    url = o["notify"] or os.environ.get(NOTIFY_ENV, "")
    status, fd = _acquire_lock()
    if status == "busy":
        if o["reset"]:
            print("reset NOT done: another deneb watch is running; try again in a minute")
            return 3
        print("another deneb watch is running, skipping")
        return 0
    try:
        return _reset() if o["reset"] else _run(o["quiet"], url)
    finally:
        if fd is not None:
            os.close(fd)


def main(argv=None) -> int:
    """`deneb watch [--quiet] [--notify URL] | --history [N] | --schedule [--every 15m] | --reset`.
    Exit 0 ok, 1 a warning or a row it could not read, 2 failure, 3 when deneb watch itself broke (so a systemd unit
    that ignores 1 and 2 still shows as failed for a broken watcher)."""
    try:
        return _main(list(argv or []))
    except Exception as e:  # noqa: BLE001 — the whole point: broken watcher => 3
        print(f"deneb watch: internal error: {type(e).__name__}. This is a bug in deneb watch, "
              "not a problem with your box.", file=sys.stderr)
        print(scrub(traceback.format_exc()), file=sys.stderr)
        return 3
