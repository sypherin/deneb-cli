"""deneb.measured - pick the best MEASURED TokenMark config for `deneb setup --measured`.

TokenMark (tokenmark.app) lists community and Altronis-run benchmark rows: decode tok/s for a
model x quant x engine x hardware. `select()` chooses, from those rows, the one a box can
actually be set up with by Deneb's tell-only steps, and says why when a faster row is out of
reach. Deneb's setup drives llama.cpp only, so a row is usable when ALL of these hold:

  - it is for this hardware and this catalog model, single-user (one stream, one GPU);
  - its engine is llama.cpp, in the build this box gets (CUDA, ROCm, Metal, CPU; or Vulkan on
    an AMD GPU, which setup can build too);
  - its quant is one deneb lists for the model (a GGUF it can tell you to download) and fits
    this box per fit.py;
  - its decoding mode is one the steps can set: none, or MTP (llama-server --spec-type draft-mtp).

The row fields used are the ones tracker-summary.json returns: model, quant, backend, variant,
ctx, mode, concurrency, num_gpus, throughput_kind, decode_tps, hardware, hardwareLabel, url,
doc_url, author, last_updated, first_seen. TokenMark returns nothing about flash-attention or the
KV-cache type, so those are never set from a measurement.

`select()` is pure (rows in, Selection out, no I/O); `choose_for_box()` is the one function that
reads the network, and fails soft: any problem becomes one explanatory line and the catalog
default. Strings that came from the network are cleaned before they are printed or written into
a command the user will paste.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from . import fit, setup_advisor, speed

# The ctx a row was measured at is only worth serving at when it is a real working context: the
# benchmark tables also hold 512- and 2048-token runs that say nothing about how to serve.
MIN_SERVE_CTX = 8192
BIG_CTX = 32768          # above this the KV cache needs memory beyond what fit.py counts
# Deneb's catalog sets no -c (llama-server's own default applies). A measured row may only RAISE
# the context from this floor: a benchmark run at -c 8192 must not shrink what the user gets.
DEFAULT_SERVE_CTX = 32768
# Model families whose main GGUF carries the next-token-prediction (nextn / MTP) layers, so
# `--spec-type draft-mtp` works on the one file (checked with gguf.GGUFReader on
# unsloth/Qwen3.8-27B-UD-Q4_K_XL: qwen35.nextn_predict_layers=1, 4 nextn tensors). Qwen3.5 up.
_MTP_FAMILY = re.compile(r"qwen3\.(?:[5-9]|\d{2})", re.I)
# Speculative decoding emits up to N+1 tokens per read of the weights; allow this much over the
# memory-bandwidth ceiling for an MTP row, and a little rounding room for a plain one.
_BOUND_SLACK_PLAIN = 1.15
_BOUND_SLACK_MTP = 6.0

# TODO(mac): TokenMark pools Mac generations under mac-max / mac-ultra, so an M2 Max and an M4 Max
# share rows and the pick can come from another generation's chip. There are no Mac rows for
# the models deneb sets up yet; when there are, split by the chip in hardwareLabel.

_MTP = re.compile(r"mtp(?:-([0-9]+))?", re.ASCII)
# Mode labels that name how the benchmark was run, not a setting a server takes.
_BENCH_ONLY = re.compile(r"(baseline|depth-\d+|main-\w+|variance-\w+|budget-\w+)")
# llama.cpp variant labels each playbook's build matches (an empty variant matches any).
_VARIANTS = {"cuda": {"cuda"}, "rocm": {"rocm", "hip"}, "metal": {"metal"}, "cpu": {"cpu"},
             "vulkan": {"vulkan"}}


@dataclass
class Choice:
    """What the setup steps take from a measured row."""
    row: dict
    quant: object            # the catalog Quant to download
    playbook: str            # which llama.cpp build the steps use: cuda | rocm | vulkan | metal | cpu
    flags: str = ""          # extra llama-server flags the row implies
    warnings: list = field(default_factory=list)


@dataclass
class Selection:
    choice: Optional[Choice]
    lines: list = field(default_factory=list)       # plain text to print, one line each
    expected: Optional[dict] = None                  # what `deneb check` compares against
    hardware_id: str = ""


def _s(value, cap: int = 120) -> str:
    """Text from the network made safe to print: printable characters only, capped."""
    return "".join(c for c in str(value if value is not None else "") if c.isprintable())[:cap].strip()


def _url(value) -> str:
    u = _s(value, 300)
    return u if u.startswith("https://") and " " not in u else ""


def _tps(row) -> float:
    v = row.get("decode_tps")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return 0.0
    return float(v)


def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if v < 100 else f"{v:.0f}"


def _date(row) -> str:
    return _s(row.get("last_updated") or row.get("first_seen"), 10)


def describe(row: dict) -> str:
    """'llama.cpp vulkan, UD-Q4_K_XL, ctx 131072, mtp' from a row's own fields."""
    eng = " ".join(x for x in (_s(row.get("backend")), _s(row.get("variant"))) if x) or "engine not named"
    parts = [eng, _s(row.get("quant")) or "quant not named"]
    if isinstance(row.get("ctx"), int) and not isinstance(row.get("ctx"), bool):
        parts.append(f"ctx {row['ctx']}")
    if row.get("mode"):
        parts.append(_s(row["mode"], 30))
    return ", ".join(parts)


def hardware_candidates(profile) -> list:
    """TokenMark hardware ids this box may be listed under, most specific first: the catalog
    device's own id, plus the family names TokenMark groups it under (strix-halo, mac-max,
    mac-ultra). Empty when the box is not in the hardware catalog."""
    dev = speed.device_for_profile(profile)
    if not dev:
        return []
    did = str(dev.get("id") or "")
    out = []
    if did == "strix-halo-395":
        out.append("strix-halo")
    m = re.fullmatch(r"m\d+-(max|ultra)", did)
    if m:
        out.append(f"mac-{m.group(1)}")
    out.append(did)
    return out


def _box_name(profile) -> str:
    dev = speed.device_for_profile(profile)
    if dev and dev.get("name"):
        return str(dev["name"])
    for g in getattr(profile, "gpus", None) or []:
        if getattr(g, "name", ""):
            return str(g.name)
    return "this hardware"


def _mode_flags(row, model=None) -> tuple:
    """(flags, problem): the llama-server flags for a row's decoding mode, or why deneb can't.
    MTP flags are emitted only for a model family known to carry the MTP layers in its GGUF;
    `model=None` skips that gate (the mode alone)."""
    mode = _s(row.get("mode"), 40).lower()
    if not mode or _BENCH_ONLY.fullmatch(mode):
        return "", ""
    m = _MTP.fullmatch(mode)
    if m:
        if model is not None and not _MTP_FAMILY.search(getattr(model, "name", "") or ""):
            return "", (f"it used MTP decoding, and deneb cannot confirm that "
                        f"{getattr(model, 'name', 'this model')}'s GGUF carries MTP layers, "
                        "so it will not add --spec-type")
        flags = "--spec-type draft-mtp"
        if m.group(1):
            flags += f" --spec-draft-n-max {m.group(1)}"
        return flags, ""
    return "", f"it ran with the '{mode}' decoding mode, which deneb's llama.cpp steps do not set"


def _over_bound(row, model, quant, profile) -> str:
    """Why a row looks wrong, or "": it reports more tok/s than this hardware's memory
    bandwidth can feed (a plain row above the ceiling, an MTP row far above). TokenMark's own
    API carries no suspect flag, so deneb checks the one physical bound it can."""
    dev = speed.device_for_profile(profile)
    try:
        bw = float(dev["bw_gbps"])
        active = float(getattr(model, "active_params_b", None) or model.params_b)
        ceiling = bw / speed.gb_per_token(active, quant.bpw)
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return ""
    slack = _BOUND_SLACK_MTP if _MTP.fullmatch(_s(row.get("mode"), 40).lower() or "-") else _BOUND_SLACK_PLAIN
    if _tps(row) > ceiling * slack:
        return (f"{_fmt(_tps(row))} tok/s is above what this hardware's memory bandwidth allows "
                f"for {quant.name} (about {_fmt(ceiling)} tok/s without speculation), so deneb "
                "does not trust it")
    return ""


def _unrated(row) -> bool:
    """TokenMark rates the row's source credibility 'none': nothing backs the number."""
    return _s(row.get("credibilityLabel"), 20).lower() == "none"


def _classify(row: dict, model, profile) -> tuple:
    """(Choice, "") when deneb's setup can reproduce this row on this box, else (None, why)."""
    backend = _s(row.get("backend")).lower()
    variant = _s(row.get("variant")).lower()
    box = (getattr(profile, "primary_backend", "") or "").strip().lower()
    if not backend:
        return None, "the row does not name its engine"
    if backend != "llama.cpp":
        return None, f"it ran on {backend}, and deneb's setup only drives llama.cpp"
    playbook = box if box in ("cuda", "rocm", "metal", "cpu") else "cpu"
    if variant:
        if variant in _VARIANTS[playbook]:
            pass
        elif variant == "vulkan" and box == "rocm":
            playbook = "vulkan"
        else:
            return None, (f"it needs a llama.cpp {variant} build, and setup builds the "
                          f"{playbook} one for this box")
    qname = _s(row.get("quant"))
    quant = next((q for q in getattr(model, "quants", None) or []
                  if qname and qname.lower() in {x.lower() for x in (q.name, *q.file_labels())}), None)
    if quant is None:
        return None, (f"its quant ({qname or 'not named'}) is not one deneb lists as a GGUF for "
                      f"{getattr(model, 'name', 'this model')}")
    if int(row.get("num_gpus") or 1) > 1:
        return None, f"it used {row['num_gpus']} GPUs"
    try:
        fits = bool(fit.fits(model, quant, profile).fits)
    except Exception:  # noqa: BLE001 - an unreadable box is "does not fit", never a crash
        fits = False
    if not fits:
        return None, f"{quant.name} does not fit this box's memory"
    flags, problem = _mode_flags(row, model)
    if problem:
        return None, problem
    over = _over_bound(row, model, quant, profile)
    if over:
        return None, over
    warnings = []
    ctx = row.get("ctx")
    ctx = ctx if isinstance(ctx, int) and not isinstance(ctx, bool) else 0
    serve = max(DEFAULT_SERVE_CTX, ctx)
    if serve > ctx and ctx:
        note = (f"the row was timed at a {ctx} token context; it is served at {serve} so the "
                "context is not smaller than deneb's default. Decode may be a little slower "
                "than the measured number at the longer context.")
    elif not ctx:
        note = f"the row names no context; it is served at {serve}, deneb's default."
    else:
        note = ""
    if serve > ctx and getattr(fit.fits(model, quant, profile), "tight", False):
        # memory is the limit: serve at the row's own context, as long as it is a real one
        serve = ctx if ctx >= MIN_SERVE_CTX else 0
        note = (f"memory is tight for {quant.name} here, so the context is held to the row's "
                f"{ctx}." if serve else
                "memory is tight here and the row's context is a benchmark size, so no -c is set; "
                "llama-server's own default applies.")
    if note:
        warnings.append(note)
    if serve:
        flags = f"{flags} -c {serve}".strip()
    if serve > BIG_CTX:
        warnings.append(f"a {serve} token context needs KV-cache memory on top of the weights. "
                        "If llama-server runs out of memory, lower -c.")
    if "--spec-type" in flags:
        warnings.append("MTP decoding: this model's main GGUF carries the MTP layers, so "
                        "--spec-type draft-mtp works on it. If llama-server still stops with "
                        "'context type MTP requested but model doesn't contain MTP layers', "
                        "the file is a build without them: drop the --spec-type flags.")
    return Choice(row=row, quant=quant, playbook=playbook, flags=flags, warnings=warnings), ""


def _candidates(rows, hw_id: str, model) -> tuple:
    """(rows for this hardware and model, fastest first; how many rows could not be read).
    One malformed row is skipped on its own, never allowed to stop the rest."""
    key = setup_advisor._norm_key(getattr(model, "name", ""))
    out, bad = [], 0
    for r in rows or []:
        if not isinstance(r, dict):
            bad += 1
            continue
        try:
            if r.get("hardware") != hw_id:
                continue
            if setup_advisor._norm_key(r.get("model")) != key:
                continue
            if not _tps(r):
                bad += 1
                continue
            if r.get("throughput_kind") != "single-stream" or (r.get("concurrency") or 1) != 1:
                continue
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        out.append(r)
    return sorted(out, key=_tps, reverse=True), bad


def select(rows, tracked, model, profile) -> Selection:
    """Choose the measured row for `model` on `profile`. Pure. `rows` are tracker configs,
    `tracked` the hardware ids they cover. Always returns lines that say what happened."""
    name = getattr(model, "name", "this model")
    box = _box_name(profile)
    ids = hardware_candidates(profile)
    hw_id = next((i for i in ids if i in set(tracked or [])), "")
    if not hw_id:
        return Selection(None, [
            f"--measured: TokenMark has no rows for {box} "
            f"(it tracks {', '.join(sorted(tracked or [])) or 'nothing yet'}). "
            "Using the catalog default."])
    cands, bad = _candidates(rows, hw_id, model)
    skipped = [f"skipped {bad} malformed row{'s' if bad != 1 else ''}"] if bad else []
    rated = [r for r in cands if not _unrated(r)]
    n_unrated = len(cands) - len(rated)
    if not cands or not rated:
        why = (f"TokenMark has only unrated rows (credibility none) for {name} on {box}"
               if cands else f"TokenMark has no single-user rows for {name} on {box}")
        return Selection(None, [f"--measured: {why}. Using the catalog default."]
                         + [f"    ({s_})" for s_ in skipped], hardware_id=hw_id)
    best = rated[0]
    chosen, first_why = None, ""
    for r in rated:
        try:
            c, why = _classify(r, model, profile)
        except Exception:  # noqa: BLE001 - a row we cannot read is skipped, and counted
            bad += 1
            c, why = None, "it could not be read"
        if r is best:
            first_why = why
        if c is not None:
            chosen = c
            break
    skipped = [f"skipped {bad} malformed row{'s' if bad != 1 else ''}"] if bad else []
    if chosen is None:
        return Selection(None, [
            f"--measured: TokenMark's fastest row for {name} on {box} is {_fmt(_tps(best))} "
            f"tok/s ({describe(best)}), but {first_why}, and no "
            "other row for this model can be set up here. Using the catalog default."]
            + [f"    ({s_})" for s_ in skipped], hardware_id=hw_id)
    row = chosen.row
    label = _s(row.get("hardwareLabel")) or box
    src = _url(row.get("url")) or _url(row.get("doc_url"))
    who = _s(row.get("author"), 40)
    lines = [f"--measured: {_fmt(_tps(row))} tok/s single-user decode on TokenMark "
             f"({describe(row)})",
             f"    measured on {label}, {_date(row) or 'date not given'}"
             + (f", by {who}" if who else ""),
             f"    source: {src}" if src else "    source: TokenMark gave no link for this row"]
    if row is not best:
        lines.append(f"    TokenMark's fastest row is {_fmt(_tps(best))} tok/s ({describe(best)}), "
                     f"but {first_why}. Using the best row deneb can set up: "
                     f"{_fmt(_tps(row))} tok/s.")
    lines += [f"    note: {w}" for w in chosen.warnings]
    if n_unrated:
        lines.append(f"    skipped {n_unrated} row{'s' if n_unrated != 1 else ''} TokenMark rates "
                     "credibility none.")
    lines += [f"    {s_}" for s_ in skipped]
    expected = {"tok_s": _tps(row), "model": name, "quant": chosen.quant.name,
                "engine": describe(row), "hardware": hw_id, "hardware_label": label,
                "source": src, "date": _date(row)}
    return Selection(chosen, lines, expected, hw_id)


def record_step(expected: dict):
    """The tell-only step that records the picked speed, for `deneb check` to compare against.
    json.dumps keeps the body on one line, so no network text can end the quoted heredoc."""
    body = json.dumps(expected, ensure_ascii=True, sort_keys=True)
    return setup_advisor.Step(
        title="Record the measured speed for deneb check",
        command=("mkdir -p ~/.config/deneb && cat > ~/.config/deneb/expected.json <<'EOF'\n"
                 f"{body}\nEOF"),
        what=setup_advisor._plain(
            f"writes the TokenMark speed this setup is based on ({_fmt(expected['tok_s'])} tok/s) "
            "to one small file. `deneb check` then times a short answer from the server and "
            "compares. Skip this step and check simply has nothing to compare."),
        warnings=[], kind="record")


def choose_for_box(model, profile, fetch=None) -> Selection:
    """Fetch TokenMark's rows and select. `fetch` returns (rows, tracked) or None (tests pass
    their own; the default is tokenmark.tracker_rows). Never raises: no network or a bad answer
    is one line and the catalog default."""
    try:
        if fetch is None:
            from . import tokenmark
            fetch = tokenmark.tracker_rows
        got = fetch()
        if not got:
            return Selection(None, ["--measured: TokenMark could not be reached or returned no "
                                    "rows. Using the catalog default."])
        return select(got[0], got[1], model, profile)
    except Exception as e:  # noqa: BLE001 - fail soft, but say why
        return Selection(None, [f"--measured: could not read TokenMark ({type(e).__name__}). "
                                "Using the catalog default."])
