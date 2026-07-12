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


def _one_shot(question: str, image: str | None = None) -> int:
    if not _require_auth():
        return 2
    ui.banner()
    try:
        if image:
            res = client.ask(question or "Diagnose the error in this screenshot.", image=image)
        else:
            res = loop.run(question, on_event=ui.event)
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


def _interactive() -> int:
    ui.banner()
    if not _require_auth():
        return 2
    ui.info('describe what\'s wrong — or type "exit" to quit (Ctrl-C also works). '
            'e.g.  llama-server won\'t start')
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
            res = loop.run(q, history=session, on_event=ui.event)
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
  deneb --image <path> "<q>"    diagnose a screenshot
  deneb auth --token <token>    sign in with your Altronis token
  deneb logout                  remove your token
  deneb --version
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
    image = _flag(argv, "--image")
    question_words = []
    i = 0
    while i < len(argv):
        if argv[i] == "--image":
            i += 2
            continue
        if argv[i].startswith("--image="):
            i += 1
            continue
        question_words.append(argv[i])
        i += 1
    q = " ".join(question_words).strip()
    if not q and not image:
        return _interactive()
    return _one_shot(q, image=image)
