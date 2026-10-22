"""deneb — CLI entry point. Subcommands: auth · logout · check; else troubleshoot."""
from __future__ import annotations

import re

from . import __version__, brain, client, config, freecmds, loop, net, tokenreq, ui, version
from .colour import Palette

_C = Palette(g="green", d="dim", b="bold", z="reset", teal="teal", amber="amber", gold="gold")

_FREE_HINT = freecmds.hint()


def _prepare(flag=None, model=None, key_env=None, image=None):
    """Poll a pending token request once, pick the brain, print the one-time notices, and point
    the client at it. Returns the Brain, or None after printing why there isn't one."""
    if key_env is not None and not brain.valid_key_env(key_env):
        ui.error(brain.KEY_ENV_REFUSAL)
        return None
    _poll_pending()
    try:
        b = brain.resolve(flag, model, key_env)
    except client.DenebError as e:
        ui.error(str(e))
        return None
    if b is None:
        note = brain.discovery_note()
        if note and "needs an API key" in note:     # a server is there; it wants a key
            ui.error(note[0].upper() + note[1:])
        else:
            ui.error(brain.NO_BRAIN_MESSAGE)
        ui.info(_FREE_HINT)
        return None
    if b.kind == "altronis" and not config.get_token():
        ui.error("the Altronis brain needs a token: deneb auth --request "
                 "(or deneb auth --token <token>). " + _FREE_HINT)
        return None
    client.set_brain(b)
    notice = brain.privacy_notice(b)
    if notice:                  # before the first request, which already carries the key
        ui.info(notice)
    for line in brain.preflight(b, pending=bool(config.load_request())):
        ui.info(line)
    return b


def _poll_pending() -> None:
    """A token request is waiting: ask once whether it has been decided. Never blocks a chat."""
    if not config.load_request():
        return
    state, msg = tokenreq.check_once(timeout=net.POLL_TIMEOUT)
    if state not in ("pending", "error"):      # only `deneb auth --check` reports a dead network
        ui.info(msg)


def _flag(argv: list[str], name: str) -> str | None:
    """Read `--name value` (or `--name=value`) from argv; None if absent."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _ask(label: str, given, optional: bool = False) -> str:
    if given is not None:
        return given.strip()
    try:
        return input(f"{label}{' (optional)' if optional else ''}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _pending_line(req: dict) -> str:
    when = (req.get("created") or "")[:10]
    who = req.get("email") or ""
    bits = [b for b in (f"sent {when}" if when else "", who) if b]
    return " (" + ", ".join(bits) + ")" if bits else ""


def cmd_auth_request(argv: list[str]) -> int:
    pending = config.load_request()
    replace = "--replace" in argv
    if pending and not replace:
        ui.error(f"a token request is already pending{_pending_line(pending)}. check it: "
                 "deneb auth --check. to send a new one anyway: deneb auth --request --replace "
                 "(the first can then no longer be claimed).")
        return 1
    prompting = any(_flag(argv, f) is None for f in ("--name", "--email", "--box"))
    name = _ask("your name", _flag(argv, "--name"))
    email = _ask("your email", _flag(argv, "--email"))
    box = _ask("what box (e.g. Strix Halo 128GB, Ollama)", _flag(argv, "--box"))
    why_flag = _flag(argv, "--why")
    why = _ask("what you want it for", why_flag, optional=True) if (prompting or why_flag is not None) else ""
    if not (1 <= len(name) <= 80):
        ui.error("a name is needed (up to 80 characters)")
        return 2
    if not tokenreq.valid_email(email):
        ui.error("that doesn't look like an email address")
        return 2
    if not (1 <= len(box) <= 120):
        ui.error("say what box this is (up to 120 characters)")
        return 2
    why = why[:500]
    engine = _flag(argv, "--engine") or config.get_engine()
    print(f"sending this to Altronis ({engine}), plus a hash of a secret that stays on this box.")
    print("nothing else:")
    print(f"  name:   {name}")
    print(f"  email:  {email}")
    print(f"  box:    {box}")
    print(f"  reason: {why or '(none)'}")
    print(f"a person approves it, usually the same day (SGT). {tokenreq.REVIEW_LINE}.")
    if pending:
        print("warning: you chose --replace, so the first request can no longer be claimed.")
    try:
        tokenreq.submit(name, email, box, why, engine=engine)
    except tokenreq.RequestError as e:
        ui.error(str(e))
        return 1
    print("request sent. once it's approved, run deneb auth --check (any deneb chat command also checks).")
    return 0


_AUTH_USAGE = ("usage: deneb auth --request [--name N --email E --box B --why W] [--replace]\n"
               "       deneb auth --check\n"
               "       deneb auth --token <your-token>  [--engine <url>]\n"
               "       no token? deneb auth --request, then deneb auth --check")


def cmd_auth(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(_AUTH_USAGE)
        return 0
    if "--request" in argv:
        return cmd_auth_request(argv)
    if "--check" in argv:
        req, want = config.load_request(), (_flag(argv, "--engine") or "").rstrip("/")
        if req and want and req.get("engine") and want != req["engine"].rstrip("/"):
            ui.error(f"that request was sent to {req['engine']}, not {want}. "
                     "run deneb auth --check without --engine.")
            return 1
        state, msg = tokenreq.check_once()
        (ui.info if state in ("approved", "pending") else ui.error)(msg)
        return 0 if state in ("approved", "pending") else 1
    token = _flag(argv, "--token")
    if not token:
        ui.error(_AUTH_USAGE)
        return 2
    config.save(token=token.strip(), engine=_flag(argv, "--engine"))
    h = client.health()
    ui.info(f"signed in. endpoint: {config.get_engine()}  "
            f"({'reachable' if h.get('ok') else 'not reachable yet — will retry when you run it'})")
    return 0


def cmd_brain(argv: list[str]) -> int:
    """`deneb brain` shows the brain; `deneb brain local|altronis|<url> [--model N] [--key-env V]`
    saves it. Only the NAME of the key's environment variable is stored, never the key."""
    model, key_env = _flag(argv, "--model"), _flag(argv, "--key-env")
    if key_env is not None and not brain.valid_key_env(key_env):
        ui.error(brain.KEY_ENV_REFUSAL)
        return 2
    skip = {"--model", "--key-env"}
    pos = [a for i, a in enumerate(argv) if not a.startswith("-") and (i == 0 or argv[i - 1] not in skip)]
    if not pos:
        try:
            b = brain.resolve()
        except client.DenebError as e:
            ui.error(str(e))
            return 1
        if b is None:
            saved = config.get_brain()
            ui.info("no brain right now" + (f" (saved: {saved.get('mode')}, but nothing answers)" if saved else ""))
            ui.info(brain.NO_BRAIN_MESSAGE)
        else:
            ui.info(f"brain: {b.describe()}")
            ui.info(f"why: {b.reason}")
        return 0
    want = pos[0]
    if want in ("local", "altronis"):
        config.save_brain(mode=want, model=model, key_env=key_env)
    elif re.match(r"^https?://[^/\s]+", want):
        config.save_brain(mode="url", url=want.rstrip("/"), model=model, key_env=key_env)
    else:
        ui.error("usage: deneb brain local | altronis | <openai-compatible base URL>  "
                 "[--model NAME] [--key-env VAR]")
        return 2
    ui.info(f"brain saved: {want}" + (f", model {model}" if model else ""))
    if key_env:
        ui.info(f"the key is read from ${key_env} when you run deneb. it is never written to disk.")
    return 0


def cmd_logout(_argv=None) -> int:
    if not config.saved_token() and not config.load_request():
        ui.info("no token saved, nothing to sign out of."
                + (" (DENEB_TOKEN in the environment is not a saved token.)"
                   if config.get_token() else ""))
        return 0
    config.clear()
    config.clear_request()      # a request approved later must not sign this box back in
    ui.info("signed out — token and any pending token request removed.")
    return 0


def _device_lines(p, C) -> list:
    """What Deneb knows about this box's hardware class from the bundled catalog:
    bandwidth (drives tok/s estimates) and the engines that suit it, best first."""
    from . import speed
    dev = speed.device_for_profile(p)
    if not dev:
        return [f"  {C['d']}hardware class: not in Deneb's hardware catalog yet, so speeds "
                f"are rough tiers, not tok/s estimates{C['z']}"]
    bw = speed.usable_bandwidth(p, dev)
    engines = ", ".join(dev.get("engines") or []) or "?"
    out = [f"  {C['b']}hardware class:{C['z']} {dev.get('name')} "
           f"{C['d']}({dev.get('mem_gb')} GB {dev.get('mem_type')}, "
           f"~{dev.get('bw_gbps')} GB/s){C['z']}",
           f"  {C['b']}engines that suit it:{C['z']} {engines} {C['d']}(best first){C['z']}"]
    if bw is None:
        out.append(f"  {C['amber']}running CPU-only, so the GPU's bandwidth does not apply; "
                   f"install the GPU runtime first{C['z']}")
    return out


def cmd_profile(_argv=None) -> int:
    """`deneb profile` — read this box and print the structured HardwareProfile. Local,
    deterministic, KEYLESS, read-only: no auth, no engine/LLM round-trip (like `check`)."""
    from . import hardware  # deterministic, local — no engine round-trip needed
    p = hardware.profile_hardware()
    ui.banner()
    print(f"{_C['d']}reading this box (read-only)…{_C['z']}\n")
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
    for line in _device_lines(p, _C):
        print(line)
    if p.usable_mem_mb is not None:
        print(f"  {_C['b']}usable model budget:{_C['z']} ~{p.usable_mem_mb} MB "
              f"{_C['d']}(deneb recommend checks each model against it){_C['z']}")
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
    if getattr(fr, "tight", False):
        return f"⚠ tight (~{headroom / 1024:.1f} GB)"
    return f"✓ ~{int(round(headroom / 1024))} GB spare"


def _tokenmark_hw_hint(p):
    """Best-guess TokenMark hardware string from the detected profile, or None."""
    parts = [getattr(p, "cpu_model", "") or ""]
    for g in (getattr(p, "gpus", None) or []):
        parts.append(f"{getattr(g, 'vendor', '') or ''} {getattr(g, 'name', '') or ''}")
    blob = " ".join(parts).lower()
    if any(k in blob for k in ("strix", "ryzen ai max", "gfx1151", "8060s", "395")):
        return "Strix Halo"
    # GB10 is the Spark's chip. "grace" alone is not a Spark: a GH200 (Grace + Hopper) and any
    # Neoverse-V2 box named for its CPU must not match DGX Spark rows.
    if any(k in blob for k in ("dgx", "gb10")) or "spark" in blob:
        return "DGX Spark"
    if "5090" in blob:
        return "RTX 5090"
    if "6000" in blob and "pro" in blob:
        return "RTX PRO 6000"
    if "4090" in blob:
        return "RTX 4090"
    return None


def cmd_recommend(argv=None) -> int:
    """`deneb recommend [--use coding|vision|chat|general]` — read this box, rank the model
    catalog for the use-case, and print a table + a next-step pointer. Local, deterministic,
    KEYLESS, engine-free: no auth, no LLM, no engine round-trip. It EXECUTES NOTHING (the
    Deneb Rule) — the 'next: deneb setup <pick>' line is a printed pointer, not an invocation."""
    from . import check, setup_advisor, speed, api  # api gathers the data; these only format it (no engine round-trip)
    argv = list(argv or [])
    use_case = (_flag(argv, "--use") or "general").strip().lower()
    if use_case not in _RECOMMEND_USE_CASES:
        ui.error(f"unknown --use '{use_case}'. valid options: "
                 f"{', '.join(_RECOMMEND_USE_CASES)}")
        return 2

    want_measured = "--measured" in argv or "--tokenmark" in argv
    data = api.recommend(use_case, measured=want_measured)
    p, source, bw, recs, free_gb = (data.profile, data.source, data.bw_gbps, data.recs, data.free_gb)

    ui.banner()
    budget = f"~{round(p.usable_mem_mb / 1024)} GB" if p.usable_mem_mb is not None else "unknown"
    print(f"{_C['d']}ranking local models for {_C['z']}{_C['b']}{use_case}{_C['z']} "
          f"{_C['d']}on this box (memory budget {budget} · {p.primary_backend} · "
          f"{_serving_label(check.serving_now())}; {source})…{_C['z']}")
    for line in _device_lines(p, _C):
        print(line)
    print()

    # est decode needs the device's bandwidth: off the catalog, a column of dashes says nothing
    cols = f"{'#':<2} {'model':<30} {'quant':<11} {'memory':<18} " + (
        f"{'speed':<9} est decode" if bw else "speed")
    print(f"  {_C['b']}{cols}{_C['z']}")
    for i, r in enumerate(recs, 1):
        name = (getattr(r.model, 'name', '') or '?')[:30]
        qname = getattr(r.quant, 'name', '') or '?'
        tier = getattr(r.fit, 'expected_speed_tier', '') or '?'
        col = _C['g'] if getattr(r.fit, 'fits', False) and not getattr(r.fit, 'tight', False) else _C['amber']
        est = f" {speed.fmt_tps(getattr(r.fit, 'est_tps', None))}" if bw else ""
        print(f"  {_C['teal']}{i:<2}{_C['z']} {_C['b']}{name:<30}{_C['z']} "
              f"{qname:<11} {col}{_fit_cell(r.fit):<18}{_C['z']} {f'{tier:<9}{est}'.rstrip()}")
        print(f"     {_C['d']}why: {r.why}{_C['z']}")
        need = setup_advisor.download_gib(r.quant)
        if need and free_gb is not None and need > free_gb:
            print(f"     {_C['amber']}download: ~{need} GiB, more than the {free_gb} GiB free "
                  f"on ~/models{_C['z']}")
        elif need and free_gb is not None:
            print(f"     {_C['d']}download: ~{need} GiB, fits the {free_gb} GiB free on "
                  f"~/models{_C['z']}")
        elif need:
            print(f"     {_C['d']}download: ~{need} GiB{_C['z']}")
    if getattr(recs[0].fit, "fits", False):
        pool = "all catalog models" if use_case == "general" else f"models tagged for {use_case}"
        first = ("Models built for coding come first, then " if use_case == "coding"
                 else "")
        order = "the order" if first else "The order"
        print(f"\n{_C['d']}" + ui.para(
            f"{first}{order} is model size times quant quality, among {pool}. It is not a "
            "skill score: a smaller model can beat a bigger one, so check a pick's published "
            "benchmarks before a big download.", "  ") + _C['z'])
    if free_gb is None:
        print(f"\n  {_C['amber']}could not read the free disk on ~/models; check each download "
              f"fits before you start.{_C['z']}")

    # Opt-in real numbers from TokenMark (community-measured tok/s). Default stays
    # KEYLESS/offline (the Deneb Rule); --measured adds one network call, fail-soft.
    if want_measured:
        hw_hint = data.hw_hint
        any_measured = bool(data.measured)
        lines = []
        for r in recs:
            name = getattr(r.model, "name", "") or ""
            b = data.measured.get(name)
            if b:
                # every field below came off the network: clean it before it reaches the terminal
                from .measured import _s as _clean, _tps as _tpsv, _fmt as _fmtv
                hw = _clean(b.get("hardwareLabel") or b.get("hardware")) or "?"
                tps = _fmtv(_tpsv(b)) if _tpsv(b) else "?"
                lines.append(f"    {name}: {_C['g']}{tps} tok/s{_C['z']} "
                             f"{_C['d']}({_clean(b.get('quant'), 40) or '?'}, {hw}, "
                             f"src {_clean(b.get('author'), 60) or '?'}){_C['z']}")
        if any_measured:
            print(f"\n  {_C['b']}measured on TokenMark{_C['z']} {_C['d']}"
                  f"(real community benchmarks{f', {hw_hint}' if hw_hint else ''}):{_C['z']}")
            print("\n".join(lines))
        else:
            print(f"\n  {_C['d']}(no TokenMark measurements matched — site unreachable or "
                  f"no data for these models yet){_C['z']}")

    if bw:
        print(f"\n  {_C['d']}est decode = single-user tok/s from memory bandwidth, no "
              f"speculative decoding (MTP / draft models add 1.5-2.5x). Real numbers: "
              f"--measured{_C['z']}")
    top = recs[0]
    top_need = setup_advisor.download_gib(top.quant)
    if free_gb is not None and top_need > free_gb:
        room = next(((i, r) for i, r in enumerate(recs, 1)
                     if 0 < setup_advisor.download_gib(r.quant) <= free_gb), None)
        tail = (f"The first pick that fits on disk now is #{room[0]} "
                f"{getattr(room[1].model, 'name', '?')}." if room
                else "None of these picks fits on disk now.")
        print(f"\n{_C['amber']}" + ui.para(
            f"#1 does not fit on disk: it downloads ~{top_need} GiB and ~/models has "
            f"{free_gb} GiB free. Free about {top_need - free_gb} GiB or point the download "
            f"at another drive. {tail}", "  ") + f"{_C['z']}")
    if not getattr(top.fit, "fits", False):
        print(f"\n  {_C['amber']}nothing in the catalog fits this box for "
              f"{use_case} — the row above is the closest (under-spec) option.{_C['z']}")
    print(f"\n  {_C['b']}next:{_C['z']} deneb setup {getattr(top.model, 'name', '?')}"
          f"   {_C['d']}(prints the platform-correct setup steps for this pick; deneb runs "
          f"nothing here){_C['z']}")
    return 0


def _serving_label(serving) -> str:
    """What the recommend and setup headers say serves on this box (check.serving_now's
    answer). They used to say "no engine", meaning no call to the hosted Deneb engine, which
    read as "no inference engine" next to an Ollama that was serving."""
    from . import check
    if not serving:
        return "no model server running"
    port, kind, model = serving
    name = check.SERVER_NAMES.get(kind, "a model server")
    return f"{name} serving {model} on :{port}" if model else f"{name} serving on :{port}"


def _setup_inputs():
    """(distro family, free GB on the models directory), both read-only; None/'unknown' when
    unreadable. The download goes to ~/models, so free space is read on its nearest existing
    parent."""
    import os
    import shutil
    from . import setup_advisor
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            family = setup_advisor.distro_family(f.read())
    except OSError:
        family = "unknown"
    path = os.path.expanduser("~/models")
    while path and not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        free_gb = int(shutil.disk_usage(path).free // (1024 ** 3))
    except OSError:
        free_gb = None
    return family, free_gb


def _ollama_store_free():
    """(free GiB or None, path) where an Ollama on this box keeps models: OLLAMA_MODELS when
    set here, else the Linux service's /usr/share/ollama/.ollama/models when that install
    exists, else ~/.ollama/models. Free space is read on the nearest existing parent, since a
    store Ollama has not written yet need not exist. Read-only."""
    import os
    import shutil
    where = os.environ.get("OLLAMA_MODELS", "").strip()
    if not where:
        where = ("/usr/share/ollama/.ollama/models" if os.path.isdir("/usr/share/ollama")
                 else os.path.expanduser("~/.ollama/models"))
    path = os.path.expanduser(where)
    while path and not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        return int(shutil.disk_usage(path).free // (1024 ** 3)), where
    except OSError:
        return None, where


def cmd_setup(argv=None) -> int:
    """`deneb setup <model>` — read this box, resolve the model in the curated catalog, and
    print the ORDERED, platform-branched tell-only setup steps (runtime -> download -> run):
    each step's exact command, a plain 'what it does', and every inline risk warning. Local,
    deterministic, KEYLESS, engine-free: no auth, no LLM, no engine round-trip. It runs
    NOTHING — the Deneb Rule: Deneb shows every command and the user runs them. An unknown
    model name yields a helpful `deneb recommend` pointer + exit 2 (never a crash); a degraded
    platform still prints best-effort generic steps with a caveat (from setup_advisor)."""
    argv = list(argv or [])
    use_measured = "--measured" in argv
    # the model name is the non-flag positional args after "setup", joined verbatim.
    name = " ".join(a for a in argv if not a.startswith("-")).strip()
    if not name:
        ui.error("usage: deneb setup <model> [--measured]   (e.g. deneb setup Qwen3.6-35B-A3B)\n"
                 "       not sure which model? rank this box first:  deneb recommend --use coding")
        return 2

    from . import api, hf_catalog  # api builds the steps (reads the box, runs nothing); hf_catalog only for the did-you-mean
    data = api.setup(name, measured=use_measured)
    if data is None:
        import difflib
        names = [getattr(m, "name", "") for m in hf_catalog.all_models()]
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.4)
        hint = f" did you mean '{close[0]}'?" if close else ""
        ui.error(f"unknown model '{name}'.{hint}\n"
                 f"       deneb advises on its curated catalog only. see what fits this box "
                 f"+ the exact names:  deneb recommend --use coding")
        return 2
    model, p, serving, steps, sel = (data.model, data.profile, data.serving, data.steps, data.selection)

    mname = getattr(model, "name", name)
    ui.banner()
    print(f"{_C['d']}setup steps for {_C['z']}{_C['b']}{mname}{_C['z']} "
          f"{_C['d']}on this box (backend {p.primary_backend} · {_serving_label(serving)})…{_C['z']}\n")
    print(f"  {_C['amber']}Deneb shows every command and runs NOTHING (the Deneb Rule). "
          f"You run each yourself.{_C['z']}\n")
    if sel:
        for ln in sel.lines:
            print(f"  {_C['d']}{ln}{_C['z']}")
        print()

    for i, s in enumerate(steps, 1):
        print(f"  {_C['teal']}{_C['b']}{i}. {s.title}{_C['z']}")
        if getattr(s, "command", ""):
            print(f"     {_C['g']}$ {s.command}{_C['z']}")
        if getattr(s, "what", ""):
            print(f"     {_C['d']}what: {s.what}{_C['z']}")
        for w in (getattr(s, "warnings", None) or []):
            print(f"     {_C['amber']}⚠ {w}{_C['z']}")
        print()

    route = next((i for i, s in enumerate(steps, 1) if s.kind == "ollama"), None)
    how = (f"either run step {route} (the Ollama route), or run the other steps yourself, top "
           "to bottom (the llama.cpp route)." if route else "run these yourself, top to bottom.")
    print(f"  {_C['b']}next:{_C['z']} {how} "
          f"{_C['d']}Deneb ran none of them (v1 is tell-only); verify each before you run "
          f"it.{_C['z']}")
    return 0


AUTO_WRITES_ASK = "--auto applies fix commands only; file writes always ask"


def _make_confirm(auto: bool):
    """Gate for APPLY actions (fix/write_file). --auto approves fix commands automatically; a
    file write always asks, --auto or not. A non-interactive stdin (piped) declines — deneb never
    mutates unprompted."""
    def confirm(res) -> bool:
        is_write = isinstance(res, dict) and res.get("action") == "write_file"
        if auto and not is_write:
            ui.info("--auto: applying")
            return True
        if auto:
            ui.info(AUTO_WRITES_ASK)
        try:
            ans = input(f"{_C['gold']}  apply this? [y/N] {_C['z']}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return ans in ("y", "yes")
    return confirm


def _one_shot(question: str, image: str | None = None, auto: bool = False, flags=None) -> int:
    ui.banner()
    if not _prepare(*(flags or (None, None, None)), image=image):
        return 2
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


def _interactive(auto: bool = False, flags=None) -> int:
    ui.banner()
    if not _prepare(*(flags or (None, None, None))):
        return 2
    ui.info('describe what\'s wrong — or type "exit" to quit (Ctrl-C also works). '
            'e.g.  llama-server won\'t start')
    if auto:
        ui.info("--auto is ON: fix commands apply without asking (never destructive or elevated); file writes still ask.")
    confirm = _make_confirm(auto)
    session: list[dict] = []  # accumulating Q&A memory (the loop's tool turns are internal)
    while True:
        try:
            q = input(f"{_C['teal']}deneb›{_C['z']} ").strip()
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


def cmd_guide(argv=None) -> int:
    """`deneb guide [platform]` — the PRE-FLIGHT runbook for the box itself.

    `deneb setup <model>` assumes the machine underneath is already sane. On both boxes
    Deneb targets that assumption is wrong in ways that do not announce themselves: a Strix
    Halo on the factory BIOS carveout runs and then dies under sustained load; a DGX Spark
    with mismatched kernel headers installs a driver that never built. Both present as model
    problems. This command is the layer that prevents them.

    Local, deterministic, KEYLESS, engine-free. Runs NOTHING - the Deneb Rule, which matters
    more here than anywhere else in the CLI because these steps edit BIOS and the bootloader.
    """
    from . import platform_guide
    argv = list(argv or [])
    key = " ".join(a for a in argv if not a.startswith("-")).strip()


    detected = ""
    if not key:
        # No platform named: read the box and pick. Detection is advisory, and is stated as
        # such, so a wrong guess is visible rather than silently followed.
        detected = _detect_platform()            # read-only probe (diagnosis, allowed)
        key = detected

    guide = platform_guide.guide_for(key)
    if guide is None:
        names = ", ".join(sorted(platform_guide.GUIDES))
        if detected == "" and not argv:
            ui.error("could not tell what this box is from a hardware probe.\n"
                     f"       name it yourself:  deneb guide <{names}>")
        else:
            ui.error(f"no guide for '{key}'.\n       available:  {names}")
        return 2

    print(f"\n{_C['b']}{guide.label}{_C['z']}")
    if detected:
        print(f"{_C['d']}detected from this box - if that looks wrong, "
              f"name the platform explicitly{_C['z']}")
    print(f"{_C['d']}applies to: {guide.applies_to}{_C['z']}")
    print(f"{_C['teal']}recommended OS: {guide.recommended_os}{_C['z']}")

    # Distro family, so the package advice matches the box rather than assuming Debian.
    fam = ""
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            fam = platform_guide.distro_family(fh.read())
        if fam:
            print(f"{_C['d']}this box looks like a {fam}-family distro{_C['z']}")
    except OSError:
        pass

    for n, step in enumerate(guide.steps, 1):
        print(f"\n{_C['b']}{n}. {step.title}{_C['z']}")
        print(ui.para(step.what))
        if step.command:
            print(f"   {_C['g']}{platform_guide.step_command(step, fam)}{_C['z']}")
        for w in step.warnings:
            print(f"{_C['amber']}{ui.para(w, '   ! ')}{_C['z']}")
        if step.verify:
            print(f"   {_C['d']}verify: {step.verify}{_C['z']}")
            if step.expect:
                print(f"{_C['d']}{ui.para(step.expect, '   expect: ', '           ')}{_C['z']}")

    print(f"\n{_C['d']}source: {guide.source}{_C['z']}")
    print(f"{_C['amber']}" + ui.para("Deneb printed these and ran none of them. Read every "
                                     "command before you run it - these edit firmware and the "
                                     "bootloader.", "") + f"{_C['z']}\n")
    return 0


def _detect_platform() -> str:
    """The platform guide that fits this box, from a read-only hardware probe ("" if none).
    One function for guide and stack, so the two never name the same box differently."""
    from . import hardware, platform_guide, speed
    p = hardware.profile_hardware()
    return platform_guide.detect_platform(p, speed.device_for_profile(p))


def cmd_stack(argv=None) -> int:
    """`deneb stack [service]` — several models behind ONE authenticated endpoint.

    The question these boxes actually pose is not "how do I run a model" but "how do I run
    an LLM, a vision model and OCR at once, and let a cloud app reach them through a single
    endpoint without exposing any of them". `deneb setup` cannot express that: OCR is not
    a chat model, and the gateway and tunnel are not models at all.

    Local, deterministic, KEYLESS, engine-free. Prints and runs nothing (the Deneb Rule).
    """
    from . import services
    argv = list(argv or [])
    want = " ".join(a for a in argv if not a.startswith("-")).strip()


    if want:
        svc = services.resolve_service(want)
        if svc is None:
            ui.error(f"unknown service '{want}'.\n"
                     f"       available:  {', '.join(sorted(services.SERVICES))}\n"
                     f"       or the whole stack:  deneb stack")
            return 2
        chosen = [svc]
        print(f"\n{_C['b']}{svc.label}{_C['z']}")
    else:
        chosen = services.stack_services()
        print(f"\n{_C['b']}Private LLM stack - three endpoints behind one authenticated "
              f"front door{_C['z']}")
        print(f"{_C['d']}" + ui.para("brought up in this order on purpose: backends, then the "
                                     "gateway that fronts them, then exposure. Standing the "
                                     "tunnel up first would publish an unauthenticated model.",
                                     "") + f"{_C['z']}")
        print(f"\n{_C['teal']}  cloud app -> tunnel -> gateway :8002 -> "
              f"LLM :8001, VLM :8080, OCR :8082{_C['z']}")
        print(f"{_C['d']}" + ui.para("(example ports, a recipe to adapt: use any free ports and "
                                     "keep every model server on 127.0.0.1)", "  ") + f"{_C['z']}")

        # Platform caveats. The llama-server commands are portable; what surrounds them is
        # not, and those differences only surface when someone is already at the machine.
        from . import platform_guide
        detected = _detect_platform()
        notes = services.platform_notes(detected)
        if notes:
            label = platform_guide.guide_for(detected)
            print(f"\n{_C['b']}On this box: {label.label if label else detected}{_C['z']}")
            for note in notes:
                print(f"{_C['amber']}{ui.para(note, '  - ', '    ')}{_C['z']}")
        elif detected == "":
            print(f"\n{_C['d']}" + ui.para("could not identify this box; the commands below are "
                                           "the portable part. Run `deneb guide <platform>` for "
                                           "the machine-specific prerequisites.", "")
                  + f"{_C['z']}")

    for svc in chosen:
        where = f" :{svc.port}" if svc.port else ""
        print(f"\n{_C['teal']}{_C['b']}▸ {svc.label}{where}{_C['z']}")
        print(f"{_C['d']}{ui.para(svc.role, '  ')}{_C['z']}")
        if svc.notes:
            print(f"{_C['d']}{ui.para(svc.notes, '  ')}{_C['z']}")
        for n, step in enumerate(svc.steps, 1):
            print(f"\n  {_C['b']}{n}. {step.title}{_C['z']}")
            print(ui.para(step.what, "     "))
            if step.command:
                for line in step.command.split("\n"):
                    print(f"     {_C['g']}{line}{_C['z']}")
            for w in step.warnings:
                print(f"{_C['amber']}{ui.para(w, '     ! ', '       ')}{_C['z']}")
            if step.verify:
                print(f"     {_C['d']}verify: {step.verify}{_C['z']}")
            if step.expect:
                print(f"{_C['d']}{ui.para(step.expect, '     expect: ', '             ')}"
                      f"{_C['z']}")

    print(f"\n{_C['amber']}" + ui.para("Deneb printed these and ran none of them. The gateway "
                                       "step is the security boundary - test that an "
                                       "unauthenticated call gets 401 before you expose "
                                       "anything.", "") + f"{_C['z']}\n")
    return 0


# The README's install line, word for word (tests hold the two together).
INSTALL_LINE = ("curl -fsSL https://raw.githubusercontent.com/sypherin/deneb-cli/main/"
                "install.sh | sh")
# the same installer for a box without curl (a fresh Ubuntu desktop has wget, not curl)
INSTALL_LINE_WGET = ("wget -qO- https://raw.githubusercontent.com/sypherin/deneb-cli/main/"
                     "install.sh | sh")


def cmd_selfupdate(argv=None) -> int:
    """`deneb selfupdate` — how to update Deneb on THIS box, including a remote one.

    Deneb is installed with pipx, and the installer is safe to re-run, so updating is one
    command. It is printed, never run: self-updating is a process replacing its own code,
    which is precisely the kind of thing the Deneb Rule exists to keep out of Deneb's hands,
    and on a client's machine it is their change to make, not ours.

    `--remote user@host` prints the same thing wrapped in ssh, for updating an install you
    are not sitting in front of.
    """
    argv = list(argv or [])
    remote = _flag(argv, "--remote")

    installer = INSTALL_LINE
    local_cmds = [
        ("update Deneb in place", installer,
         "re-runs the installer, which pipx-installs the current version over the old one. "
         "Safe to re-run; it never asks for sudo."),
        ("confirm it landed", "deneb --version",
         "prints the release and the commit it was installed from. The commit changes "
         "with every update, even within one release."),
    ]

    print(f"\n{_C['b']}Update Deneb{_C['z']}")
    print(f"{_C['d']}installed version here: {version.installed()}{_C['z']}")

    if remote:
        print(f"{_C['teal']}target: {remote}{_C['z']}")
        print(f"\n{_C['b']}1. Update Deneb on {remote}{_C['z']}")
        print("   runs the same installer over SSH. The remote user needs a login shell and "
              "pipx on PATH.")
        print(f"   {_C['g']}ssh {remote} '{installer}'{_C['z']}")
        print(f"\n{_C['b']}2. Confirm it landed{_C['z']}")
        print(f"   {_C['g']}ssh {remote} 'deneb --version'{_C['z']}")
        print(f"   {_C['amber']}! if deneb is not found over ssh, pipx's ~/.local/bin is "
              f"missing from the non-interactive PATH. Use the full path:  "
              f"ssh {remote} '~/.local/bin/deneb --version'{_C['z']}")
        print(f"   {_C['amber']}! this runs a remote install on someone else's machine. "
              f"Tell them before you do it.{_C['z']}")
    else:
        for n, (title, cmd, what) in enumerate(local_cmds, 1):
            print(f"\n{_C['b']}{n}. {title}{_C['z']}")
            print(f"   {what}")
            print(f"   {_C['g']}{cmd}{_C['z']}")
            if cmd == installer:
                print(f"   {_C['d']}no curl on this box? the same with wget:{_C['z']}")
                print(f"   {_C['g']}{INSTALL_LINE_WGET}{_C['z']}")
        print(f"\n{_C['d']}updating a box you are not sitting at:  "
              f"deneb selfupdate --remote user@host{_C['z']}")

    print(f"\n{_C['amber']}Deneb printed this and ran nothing. A process that rewrites its "
          f"own code should be your decision, not its.{_C['z']}\n")
    return 0


def cmd_watch(argv=None) -> int:
    """`deneb watch` — keep it healthy: re-run the scan plus disk / memory / restart / catalog
    probes, say what changed since last time, and (opt-in) send an alert. Read-only: it
    prints the fix, never applies it (Deneb Rule). The work lives in deneb/watch.py."""
    from . import watch
    return watch.main(argv)


_HELP = """Deneb: check, set up and watch a local LLM server on your own AI box.

free, no token, nothing sent unless you ask (alerts, --refresh, --measured).
They read the box and print the fix, and run none of it:
@FREE@
  deneb --version

conversational troubleshooting: free with your own model, or a token from Altronis on request:
  deneb                         start interactive troubleshooting
  deneb "<what's wrong>"        one-shot: e.g.  deneb "llama-server won't start"
  deneb --image <path> "<q>"    diagnose a screenshot (needs a vision model)
  deneb --auto "<what's wrong>" apply fix commands without asking (file writes always ask)
  deneb brain                   which model it thinks with, and why
  deneb brain local|altronis|<url> [--model NAME] [--key-env VAR]
                                choose it (--key-env names the variable holding an API key)
  deneb --brain <...> "<q>"     use one brain for this run only
  deneb auth --request          ask Altronis for a token (name, email, box, why)
  deneb auth --check            has my request been approved? (saves the token if so)
  deneb auth --token <token>    save a token you were given
  deneb logout                  remove your saved settings and token

Which model it uses, first match wins: --brain, then deneb brain, then a saved token (the
Altronis engine), then a model server already running on this box (the one deneb check finds).
With your own model (a local server, or any OpenAI-compatible URL) nothing is sent to Altronis;
if that URL is not this machine, Deneb says where your question goes before it sends it.
With the Altronis engine, your question and the log lines Deneb reads are sent to the Altronis engine.
Either way it can apply a fix (edit a config, mkdir, systemctl --user restart), asking before
each one. --auto applies fix commands only; file writes always ask. It never runs anything destructive, irreversible or
needing sudo: those it hands you to run yourself.
"""


_HELP = _HELP.replace("@FREE@", freecmds.help_lines())   # the free block comes from one list

# `deneb <command> --help` prints that command's lines instead of running the command
_SUB_HELP = ("check", "profile", "recommend", "setup", "catalog", "guide", "stack",
             "selfupdate", "self-update", "watch", "brain", "logout")
_SUB_HELP_EXTRA = {
    "recommend": "  deneb recommend --measured    add measured community speeds (reads TokenMark "
                 "over the network)",
    "setup": "  deneb setup <model> --measured   build the steps from TokenMark's best measured "
             "config for this box (reads TokenMark over the network)",
}


def _sub_help(sub: str) -> str:
    """That command's lines from `deneb --help` (with their continuation lines), watch's full
    usage, and where the rest is."""
    sub = "selfupdate" if sub == "self-update" else sub
    lines, keep = [], False
    for ln in _HELP.splitlines():
        if re.match(rf"  deneb {sub}(\s|$)", ln):
            keep = True
            lines.append(ln)
        elif keep and ln.startswith(" " * 20) and ln.strip():
            lines.append(ln)
        else:
            keep = False
    if sub in _SUB_HELP_EXTRA:
        lines.append(_SUB_HELP_EXTRA[sub])
    if sub == "watch":
        from . import watch
        lines += ["", watch._USAGE]
    return "\n".join(lines + ["", "all commands: deneb --help"])


def cmd_catalog(argv=None) -> int:
    """`deneb catalog [--refresh]` — list the models Deneb ranks, with real per-quant sizes.
    --refresh re-reads every watchlist model from Hugging Face (keyless, one network call
    per model) into ~/.cache/deneb/models.json. Writes only Deneb's own cache."""
    from . import hf_catalog
    argv = list(argv or [])
    ui.banner()
    if "--refresh" in argv:
        print(f"{_C['d']}refreshing from Hugging Face…{_C['z']}")
        ents, errs = hf_catalog.refresh()
        for e in errs:
            print(f"  {_C['amber']}! {e}{_C['z']}")
        if not ents:
            ui.error("refresh got no models; keeping the existing catalog.")
            return 1
        hf_catalog.write_json(hf_catalog.CACHE_PATH, {
            "schema": 1, "generated": ents[0]["fetched"], "models": ents, "errors": errs})
        print(f"  {_C['g']}saved {len(ents)} models to {hf_catalog.CACHE_PATH}{_C['z']}\n")
    ents, label = hf_catalog.load_entries()
    if not ents:
        ui.error("no live catalog found (reinstall deneb, or run: deneb catalog --refresh)")
        return 1
    print(f"{_C['d']}{len(ents)} models · {label}{_C['z']}\n")
    for e in ents:
        act = f", {e['active_params_b']}B active" if e.get("active_params_b") else ""
        print(f"  {_C['teal']}{_C['b']}{e['name']}{_C['z']} {_C['d']}{e['params_b']}B "
              f"{e.get('architecture')}{act} · {', '.join(e.get('capabilities') or [])}{_C['z']}")
        qs = "  ".join(f"{q['name']} {round(q['size_mb'] / 1000, 1)}GB" for q in e["quants"])
        print(f"    {qs}")
    return 0


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        print(f"deneb {version.installed()}")
        return 0
    if argv and argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return 0
    if argv and argv[0] in _SUB_HELP and ("--help" in argv[1:] or "-h" in argv[1:]):
        print(_sub_help(argv[0]))
        return 0
    # Free-form-first: dispatch a leading subcommand, else treat argv as a question.
    if argv and argv[0] == "auth":
        return cmd_auth(argv[1:])
    if argv and argv[0] == "logout":
        return cmd_logout()
    if argv and argv[0] == "brain":
        return cmd_brain(argv[1:])
    if argv and argv[0] == "check":
        from . import check  # deterministic, local, no engine round-trip needed
        return check.run()
    if argv and argv[0] == "profile":
        return cmd_profile(argv[1:])  # deterministic, local, keyless, no engine round-trip
    if argv and argv[0] == "recommend":
        return cmd_recommend(argv[1:])  # deterministic, local, keyless, engine-free (Deneb Rule)
    if argv and argv[0] == "setup":
        return cmd_setup(argv[1:])  # deterministic, local, keyless, engine-free (Deneb Rule)
    if argv and argv[0] == "catalog":
        return cmd_catalog(argv[1:])  # local list; --refresh reads Hugging Face (keyless)
    if argv and argv[0] == "guide":
        return cmd_guide(argv[1:])  # deterministic, local, keyless, engine-free (Deneb Rule)
    if argv and argv[0] == "stack":
        return cmd_stack(argv[1:])  # deterministic, local, keyless, engine-free (Deneb Rule)
    if argv and argv[0] in ("selfupdate", "self-update"):
        return cmd_selfupdate(argv[1:])  # tell-only (Deneb Rule)
    if argv and argv[0] == "watch":
        return cmd_watch(argv[1:])  # read-only maintain mode, tells the fix (Deneb Rule)
    image = _flag(argv, "--image")
    flags = (_flag(argv, "--brain"), _flag(argv, "--model"), _flag(argv, "--key-env"))
    auto = "--auto" in argv
    valued = ("--image", "--brain", "--model", "--key-env")
    question_words = []
    i = 0
    while i < len(argv):
        if argv[i] in valued:
            i += 2
            continue
        if argv[i].startswith(tuple(v + "=" for v in valued)) or argv[i] == "--auto":
            i += 1
            continue
        question_words.append(argv[i])
        i += 1
    q = " ".join(question_words).strip()
    if not q and not image:
        return _interactive(auto=auto, flags=flags)
    return _one_shot(q, image=image, auto=auto, flags=flags)
