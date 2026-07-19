"""deneb — CLI entry point. Subcommands: auth · logout · check; else troubleshoot."""
from __future__ import annotations

from . import __version__, client, config, loop, ui

CHECK_PROMPT = (
    "Run a complete 'am I done?' check of the Neo local-LLM stack on this box. Using "
    "READ-ONLY checks only, verify each of: llama.cpp is built with the GPU accelerator "
    "(not silently on CPU); the model file is present; the systemd service is enabled and "
    "running (no 203/EXEC); curl :8001/health is actually serving; the auth gateway is up "
    "with API-key auth + rate limiting; the cloudflared tunnel is live. "
    "Assess against what is ACTUALLY installed — the box may run Qwen3.6 rather than the "
    "runbook's default Gemma, so check for whichever model/service is present. "
    "IMPORTANT: detect the gateway and tunnel by PORT and unit-PATTERN, never by a guessed "
    "service name — service names vary per box. For the gateway: check what is serving on "
    "the gateway port (e.g. `ss -ltn` for :8002) AND list units matching the pattern "
    "(`systemctl --user list-units '*gateway*'`). For the tunnel: `systemctl --user "
    "list-units '*cloudflar*' '*tunnel*'` and check for a running cloudflared process. Only "
    "mark them ✗ if truly nothing matches. Check each component ONCE; if it isn't there, "
    "mark it ✗ and move on — do not search many paths. Be efficient. Then give a clear ✓/✗ "
    "checklist with the exact fix for anything failing, and tell me plainly whether this box "
    "has reached the working Neo Altronis state or exactly what's left."
)


def _require_auth() -> bool:
    if not config.get_token():
        ui.error("not signed in — run:  deneb auth --token <your-token>")
        return False
    return True


def _flag(argv: list[str], name: str) -> str | None:
    """Read `--name value` (or `--name=value`) from argv; None if absent."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def cmd_auth(argv: list[str]) -> int:
    token = _flag(argv, "--token")
    if not token:
        ui.error("usage: deneb auth --token <your-token>  [--engine <url>]")
        return 2
    config.save(token=token.strip(), engine=_flag(argv, "--engine"))
    h = client.health()
    ui.info(f"signed in. endpoint: {config.get_engine()}  "
            f"({'reachable' if h.get('ok') else 'not reachable yet — will retry when you run it'})")
    return 0


def cmd_logout(_argv=None) -> int:
    config.clear()
    ui.info("signed out — token removed.")
    return 0


def cmd_profile(_argv=None) -> int:
    """`deneb profile` — read this box and print the structured HardwareProfile. Local,
    deterministic, KEYLESS, read-only: no auth, no engine/LLM round-trip (like `check`)."""
    from . import hardware  # deterministic, local — no engine round-trip needed
    _C = {"g": "\033[32m", "d": "\033[2m", "b": "\033[1m", "z": "\033[0m",
          "teal": "\033[38;5;44m", "amber": "\033[33m"}
    p = hardware.profile_hardware()
    ui.banner()
    print(f"{_C['d']}reading this box (read-only, no engine)…{_C['z']}\n")
    print(f"  {_C['b']}os{_C['z']}    {p.os or '?'} · {p.arch or '?'} · kernel {p.kernel or '?'}")
    print(f"  {_C['b']}cpu{_C['z']}   {p.cpu_model or 'unknown'}  ({p.cpu_cores or '?'} cores)")
    print(f"  {_C['b']}ram{_C['z']}   {str(p.ram_total_mb) + ' MB' if p.ram_total_mb else 'unknown'}")
    print(f"  {_C['b']}disk{_C['z']}  {str(p.disk_free_mb) + ' MB free' if p.disk_free_mb is not None else 'unknown'}")
    print()
    if not p.gpus:
        print(f"  {_C['amber']}no GPU vendor tool found — CPU-only.{_C['z']}")
    for g in p.gpus:
        name = g.name or f"{g.vendor or 'unknown'} GPU"
        print(f"  {_C['teal']}{_C['b']}▸ {g.vendor or '?'} · {name}{_C['z']}  "
              f"{_C['d']}[{g.backend or 'cpu'}]{_C['z']}")
        if g.memory_kind == "unified":
            print(f"      memory: unified {g.unified_mem_mb} MB (shared system-RAM pool; "
                  f"usable ~{p.usable_mem_mb} MB)")
        elif g.memory_kind == "dedicated":
            print(f"      memory: VRAM {g.vram_mb} MB (dedicated)" if g.vram_mb
                  else "      memory: dedicated (VRAM size unread)")
        else:
            print("      memory: unknown")
        extra = g.extra or {}
        tail = " · ".join(x for x in (
            f"gfx {extra['gfx']}" if extra.get("gfx") else "",
            f"sku {extra['sku']}" if extra.get("sku") else "",
            f"driver {g.driver}" if g.driver else "") if x)
        if tail:
            print(f"      {_C['d']}{tail}{_C['z']}")
    print(f"\n  {_C['b']}primary backend:{_C['z']} {p.primary_backend}")
    if p.usable_mem_mb is not None:
        print(f"  {_C['b']}usable model budget:{_C['z']} ~{p.usable_mem_mb} MB "
              f"{_C['d']}(coarse — Phase-2 fit math refines this){_C['z']}")
    if p.notes:
        print(f"\n{_C['d']}notes:{_C['z']}")
        for n in p.notes:
            print(f"{_C['d']}  · {n}{_C['z']}")
    return 0


_RECOMMEND_USE_CASES = ("coding", "vision", "chat", "general")


def _fit_cell(fr) -> str:
    """One-column fit status for the ranked table: a check + headroom in GB when it fits
    comfortably, 'tight' with the MB spare when it fits but barely, else 'no' with how many
    MB short. Purely cosmetic — the real fit math lives in deneb.fit."""
    headroom = int(getattr(fr, "headroom_mb", 0) or 0)
    if not getattr(fr, "fits", False):
        need = int(getattr(fr, "need_mb", 0) or 0)
        short = need - headroom if headroom else need  # headroom is 0 when budget unread
        return f"no (short ~{max(0, -headroom)} MB)" if headroom < 0 else "no"
    if headroom < 2000:
        return f"tight (~{headroom} MB)"
    return f"✓ ~{int(round(headroom / 1024))} GB free"


def cmd_recommend(argv=None) -> int:
    """`deneb recommend [--use coding|vision|chat|general]` — read this box, rank the model
    catalog for the use-case, and print a table + a next-step pointer. Local, deterministic,
    KEYLESS, engine-free: no auth, no LLM, no engine round-trip. It EXECUTES NOTHING (the
    Deneb Rule) — the 'next: deneb setup <pick>' line is a printed pointer, not an invocation."""
    from . import hardware, recommend as rec  # deterministic, local — no engine round-trip
    argv = list(argv or [])
    use_case = (_flag(argv, "--use") or "general").strip().lower()
    if use_case not in _RECOMMEND_USE_CASES:
        ui.error(f"unknown --use '{use_case}'. valid options: "
                 f"{', '.join(_RECOMMEND_USE_CASES)}")
        return 2

    _C = {"g": "\033[32m", "d": "\033[2m", "b": "\033[1m", "z": "\033[0m",
          "teal": "\033[38;5;44m", "amber": "\033[33m"}
    p = hardware.profile_hardware()
    recs = rec.recommend(p, use_case)

    ui.banner()
    budget = f"~{p.usable_mem_mb} MB" if p.usable_mem_mb is not None else "unknown"
    print(f"{_C['d']}ranking local models for {_C['z']}{_C['b']}{use_case}{_C['z']} "
          f"{_C['d']}on this box (budget {budget} · {p.primary_backend}, no engine)…{_C['z']}\n")

    print(f"  {_C['b']}{'#':<2} {'model':<30} {'quant':<7} "
          f"{'fit':<18} {'speed':<9}{_C['z']}")
    for i, r in enumerate(recs, 1):
        name = (getattr(r.model, 'name', '') or '?')[:30]
        qname = getattr(r.quant, 'name', '') or '?'
        tier = getattr(r.fit, 'expected_speed_tier', '') or '?'
        col = _C['g'] if getattr(r.fit, 'fits', False) else _C['amber']
        print(f"  {_C['teal']}{i:<2}{_C['z']} {_C['b']}{name:<30}{_C['z']} "
              f"{qname:<7} {col}{_fit_cell(r.fit):<18}{_C['z']} {tier:<9}")
        print(f"     {_C['d']}why: {r.why}{_C['z']}")

    top = recs[0]
    if not getattr(top.fit, "fits", False):
        print(f"\n  {_C['amber']}nothing in the catalog fits this box for "
              f"{use_case} — the row above is the closest (under-spec) option.{_C['z']}")
    print(f"\n  {_C['b']}next:{_C['z']} deneb setup {getattr(top.model, 'name', '?')}"
          f"   {_C['d']}(setup lands in a later release; this is a pointer, deneb runs "
          f"nothing here){_C['z']}")
    return 0


def _make_confirm(auto: bool):
    """Gate for APPLY actions (fix/write_file). --auto approves automatically; otherwise ask
    the user y/N. A non-interactive stdin (piped) declines — deneb never mutates unprompted."""
    def confirm(_res) -> bool:
        if auto:
            ui.info("--auto: applying")
            return True
        try:
            ans = input("\033[38;5;178m  apply this? [y/N] \033[0m").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return ans in ("y", "yes")
    return confirm


def _one_shot(question: str, image: str | None = None, auto: bool = False) -> int:
    if not _require_auth():
        return 2
    ui.banner()
    try:
        if image:
            res = client.ask(question or "Diagnose the error in this screenshot.", image=image)
        else:
            res = loop.run(question, on_event=ui.event, confirm=_make_confirm(auto))
    except client.DenebError as e:
        ui.error(str(e))
        return 1
    ui.final(res)
    return 0


_EXIT_WORDS = {"exit", "quit", "q", "bye", "close", "stop"}
# ~100k tokens of Q&A memory kept on the box; the engine ALSO windows every request,
# so the model context (256k) can never overflow.
_SESSION_CHAR_BUDGET = 400_000


def _is_exit(q: str) -> bool:
    return q.strip().lower().lstrip("/:\\").strip() in _EXIT_WORDS


def _maybe_compact(session: list[dict]) -> list[dict]:
    """Auto-compaction (Claude-Code style): when the session nears the context budget,
    SUMMARISE the older turns into a compact brief and keep the recent ones verbatim —
    so context is preserved (not forgotten) and the model window never overflows."""
    if sum(len(h.get("content", "")) for h in session) <= _SESSION_CHAR_BUDGET:
        return session
    keep = session[-6:]
    older = session[:-6]
    convo = "\n".join(f"{h.get('role')}: {h.get('content', '')}" for h in older)
    try:
        summary = client.summarize(convo)
    except client.DenebError:
        summary = ""
    if not summary:  # fall back to keeping the anchor + recent turns
        return (session[:1] + keep) if session else keep
    ui.info("(compacted earlier context to stay under the model's window)")
    return [{"role": "user", "content": "[Earlier session — compacted summary]\n" + summary}] + keep


def _interactive(auto: bool = False) -> int:
    ui.banner()
    if not _require_auth():
        return 2
    ui.info('describe what\'s wrong — or type "exit" to quit (Ctrl-C also works). '
            'e.g.  llama-server won\'t start')
    if auto:
        ui.info("--auto is ON: fixes apply without asking (still never destructive/elevated).")
    confirm = _make_confirm(auto)
    session: list[dict] = []  # accumulating Q&A memory (the loop's tool turns are internal)
    while True:
        try:
            q = input("\033[38;5;44mdeneb›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if _is_exit(q):
            ui.info("bye.")
            break
        try:
            res = loop.run(q, history=session, on_event=ui.event, confirm=confirm)
        except client.DenebError as e:
            ui.error(str(e))
            continue
        ui.final(res)
        # Remember the question + the final answer for follow-ups (not the tool noise).
        session.append({"role": "user", "content": q})
        session.append({"role": "assistant", "content": res.get("answer") or ""})
        session = _maybe_compact(session)
    return 0


_HELP = """Deneb (Altronis) — get your AI box to a working private-LLM (Neo) state.

usage:
  deneb                         start interactive troubleshooting
  deneb "<what's wrong>"        one-shot: e.g.  deneb "llama-server won't start"
  deneb check                   scan the box — am I done?
  deneb profile                 read this box — structured hardware profile (os/cpu/ram/gpu)
  deneb recommend [--use ...]   rank local models for this box (--use coding|vision|chat|general)
  deneb --image <path> "<q>"    diagnose a screenshot
  deneb --auto "<what's wrong>" fix without asking each time (still never destructive)
  deneb auth --token <token>    sign in with your Altronis token
  deneb logout                  remove your token
  deneb --version

Deneb can APPLY fixes (edit a config, mkdir, systemctl --user restart) — it shows you each
one and asks before running it (gate by default). It NEVER runs anything destructive,
irreversible, or needing sudo — those it hands you to run yourself.
"""


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        print(f"deneb {__version__}")
        return 0
    if argv and argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return 0
    # Free-form-first: dispatch a leading subcommand, else treat argv as a question.
    if argv and argv[0] == "auth":
        return cmd_auth(argv[1:])
    if argv and argv[0] == "logout":
        return cmd_logout()
    if argv and argv[0] == "check":
        from . import check  # deterministic, local, no engine round-trip needed
        return check.run()
    if argv and argv[0] == "profile":
        return cmd_profile(argv[1:])  # deterministic, local, keyless, no engine round-trip
    if argv and argv[0] == "recommend":
        return cmd_recommend(argv[1:])  # deterministic, local, keyless, engine-free (Deneb Rule)
    image = _flag(argv, "--image")
    auto = "--auto" in argv
    question_words = []
    i = 0
    while i < len(argv):
        if argv[i] == "--image":
            i += 2
            continue
        if argv[i].startswith("--image=") or argv[i] == "--auto":
            i += 1
            continue
        question_words.append(argv[i])
        i += 1
    q = " ".join(question_words).strip()
    if not q and not image:
        return _interactive(auto=auto)
    return _one_shot(q, image=image, auto=auto)
