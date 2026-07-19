"""deneb.recommend — pure, deterministic model recommendation (REC-01, REC-03).

recommend(profile, use_case) ranks the curated CATALOG for a specific box + use-case,
keeping only fitting model+quant pairs, and returns the top-N as Recommendation records
each with a plain-language "why" string. This is a PURE layer: struct in, struct out — NO
I/O, NO network, NO LLM, no printing (the CLI in deneb.cli does the display). The same
input always yields the same order (deterministic ranking with a stable name tie-break).

Ranking (per the Plan 02-02 spec), applied to the FITTING candidates DESCENDING with
model.name ASCENDING as the final stable tie-break:
  1. specificity      — 1 if use_case is one of the model's own capabilities, else 0
                        (a "general"-only model is an eligible fallback but ranks below a
                        specialist for a specialized use-case).
  2. runs_acceptably  — 1 unless the (fitting) model would crawl (very-slow tier); demotes
                        giants that "fit" but are unusably slow.
  3. capability_proxy — params_b (TOTAL params; for a MoE, total ~ capability while active
                        params drove the speed tier).
  4. speed_rank       — faster tier first.
  5. headroom_mb      — a more comfortable fit first.
  6. quant bpw        — higher-quality quant first.
  7. model.name       — ascending, the deterministic final tie-break.

Per eligible model we keep its BEST fitting quant (the highest-bpw quant whose fits() is
True). Eligibility for a use_case: capabilities include use_case OR "general" (general
models are always eligible as a fallback).

NOTHING-FITS (REC-03): when no model+quant fits, we do NOT return an empty list — we return
a SINGLE honest Recommendation for the smallest option (least need_mb), fits=False, with a
why that says the box is under-spec, names the smallest option + how many MB short it is,
and suggests a smaller model / more RAM / CPU offload. recommend() never raises (T-03-02).
"""
from __future__ import annotations

from dataclasses import dataclass

from .fit import TIER_ORDER, fits
from .models_catalog import CATALOG, Model, Quant, by_capability  # noqa: F401

# The fixed use-case vocabulary; an unknown value normalizes to "general".
USE_CASES = ("coding", "vision", "chat", "general")

# The em-dash / en-dash house-style substitution applied to every why-string (the coarse
# fit-caveats we splice in may contain em-dashes; the recommendation copy must not).
_DASHES = ("—", "–")


@dataclass
class Recommendation:
    """One ranked pick: the model, the chosen quant, its FitResult, and a plain why-string."""
    model: object      # models_catalog.Model
    quant: object      # models_catalog.Quant
    fit: object        # fit.FitResult
    why: str


# ── small pure helpers ────────────────────────────────────────────────────────
def _plain(text: str) -> str:
    """House style: no em/en dashes in generated copy (replace with a spaced hyphen)."""
    for d in _DASHES:
        text = text.replace(d, "-")
    return text


def _gb(mb) -> str:
    """MB -> whole GB for readable why-strings (simple /1024 round). '?' when unknown."""
    try:
        return str(int(round(int(mb) / 1024)))
    except (TypeError, ValueError):
        return "?"


def _norm_use_case(use_case) -> str:
    """Normalize/validate the use_case; an unknown value falls back to 'general'."""
    uc = (str(use_case) if use_case is not None else "").strip().lower()
    return uc if uc in USE_CASES else "general"


def _eligible(use_case: str) -> list:
    """Every catalog model whose capabilities include use_case OR 'general' (general models
    are always eligible as a fallback). Preserves CATALOG order (deterministic)."""
    out = []
    for m in CATALOG:
        caps = getattr(m, "capabilities", None) or []
        if use_case in caps or "general" in caps:
            out.append(m)
    return out


def _best_fitting_quant(model, profile):
    """Among the model's quants that fit, the highest-bpw one (best quality that still fits),
    with its FitResult. Returns (quant, fit) or (None, None) when nothing fits."""
    best = None  # (bpw, quant, fitresult)
    for q in getattr(model, "quants", None) or []:
        fr = fits(model, q, profile)
        if fr.fits:
            bpw = float(getattr(q, "bpw", 0) or 0)
            if best is None or bpw > best[0]:
                best = (bpw, q, fr)
    if best is None:
        return None, None
    return best[1], best[2]


def _speed_rank(tier: str) -> int:
    """Index of `tier` in TIER_ORDER (fast highest); 0 for an unknown tier."""
    try:
        return TIER_ORDER.index(tier)
    except (ValueError, AttributeError):
        return 0


def _rank_key(cand, use_case: str) -> tuple:
    """The DESCENDING composite sort key for a fitting candidate (model.name is applied as a
    separate ascending pre-sort, so equal-key candidates stay name-ordered)."""
    model, quant, fr = cand
    caps = getattr(model, "capabilities", None) or []
    specificity = 1 if use_case in caps else 0
    runs_acceptably = 1 if getattr(fr, "expected_speed_tier", "") != "very-slow" else 0
    capability_proxy = float(getattr(model, "params_b", 0) or 0)
    speed_rank = _speed_rank(getattr(fr, "expected_speed_tier", ""))
    headroom = int(getattr(fr, "headroom_mb", 0) or 0)
    bpw = float(getattr(quant, "bpw", 0) or 0)
    return (specificity, runs_acceptably, capability_proxy, speed_rank, headroom, bpw)


# ── why-strings (pure, deterministic, no LLM, no em-dash) ─────────────────────
def _role_phrase(model, use_case: str, is_top: bool) -> str:
    if is_top:
        return f"most capable {use_case} model"
    caps = getattr(model, "capabilities", None) or []
    if use_case in caps:
        return f"{use_case}-capable"
    return "general-purpose"


def _why(model, quant, fr, use_case: str, profile, is_top: bool) -> str:
    """One plain-English line for a FITTING pick. Names the model + quant + tier + backend
    and the memory budget/headroom, then the first caveat if any. No em-dash."""
    name = getattr(model, "name", "") or "this model"
    qname = getattr(quant, "name", "") or "?"
    usable_gb = _gb(getattr(profile, "usable_mem_mb", None))
    headroom_gb = _gb(getattr(fr, "headroom_mb", 0))
    tier = getattr(fr, "expected_speed_tier", "") or "unknown"
    backend = getattr(profile, "primary_backend", "") or "cpu"
    role = _role_phrase(model, use_case, is_top)
    line = (f"{name} ({qname}): {role} that fits your ~{usable_gb} GB budget with "
            f"~{headroom_gb} GB to spare; expected speed: {tier} on {backend}.")
    caveats = getattr(fr, "caveats", None) or []
    if caveats:
        line += " " + str(caveats[0])
    return _plain(line)


def _smallest_option(models, profile):
    """The (model, quant, FitResult) with the least need_mb across `models` (every quant
    considered), name-ascending as the deterministic tie-break. (None, None, None) when
    there is nothing to consider."""
    best = None  # ((need_mb, name), model, quant, fitresult)
    for m in models:
        for q in getattr(m, "quants", None) or []:
            fr = fits(m, q, profile)
            need = int(getattr(fr, "need_mb", 0) or 0)
            key = (need, getattr(m, "name", "") or "")
            if best is None or key < best[0]:
                best = (key, m, q, fr)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _why_nothing_fits(model, quant, fr, use_case: str, profile) -> str:
    """The honest under-spec explanation (REC-03): the box is under-spec, names the smallest
    option + how many MB short it is, and suggests a smaller model / more RAM / CPU offload."""
    name = getattr(model, "name", "") or "the smallest catalog model"
    qname = getattr(quant, "name", "") or "?"
    need = int(getattr(fr, "need_mb", 0) or 0)
    usable = getattr(profile, "usable_mem_mb", None)
    try:
        usable_int = int(usable) if usable is not None else None
    except (TypeError, ValueError):
        usable_int = None
    if usable_int is None:
        budget = "your box's memory budget could not be read"
        short = ""
    else:
        budget = f"your ~{_gb(usable_int)} GB budget"
        short = f", short by ~{max(0, need - usable_int)} MB"
    line = (f"your box is under-spec for {use_case}: nothing in the catalog fits {budget}. "
            f"The smallest option is {name} ({qname}), needing ~{need} MB{short}. "
            f"Try a smaller model, add more RAM, or use CPU offload (slower).")
    return _plain(line)


def _nothing_fits(eligible, use_case: str, profile) -> Recommendation:
    """REC-03: no model+quant fits — return ONE honest under-spec Recommendation for the
    smallest option (least need_mb) with fits=False. Never raises."""
    try:
        pool = list(eligible) or list(CATALOG)
        m, q, fr = _smallest_option(pool, profile)
        if m is None:
            m, q, fr = _smallest_option(list(CATALOG), profile)
        return Recommendation(model=m, quant=q, fit=fr,
                              why=_why_nothing_fits(m, q, fr, use_case, profile))
    except Exception:  # noqa: BLE001 — the nothing-fits path must itself never raise
        m = CATALOG[0]
        q = m.quants[0]
        fr = fits(m, q, profile)
        return Recommendation(
            model=m, quant=q, fit=fr,
            why=("your box appears under-spec for local LLMs: the smallest option still does "
                 "not fit. Try a smaller model, add more RAM, or use CPU offload (slower)."))


# ── the public entry point ────────────────────────────────────────────────────
def recommend(profile, use_case, top_n: int = 3) -> list:
    """Rank the catalog for `profile` + `use_case`, returning the top-N Recommendations.

    Deterministic + pure: keeps each eligible model's best fitting quant, sorts by the
    <ranking_spec> composite key (name-ascending stable tie-break), slices top_n, and builds
    a plain why-string per pick. Empty result -> the nothing-fits under-spec fallback (never
    an empty list). Degrades honestly and NEVER raises on a degraded/None-budget profile."""
    try:
        uc = _norm_use_case(use_case)
        eligible = _eligible(uc)

        candidates = []
        for m in eligible:
            q, fr = _best_fitting_quant(m, profile)
            if q is not None:
                candidates.append((m, q, fr))

        if not candidates:
            return [_nothing_fits(eligible, uc, profile)]

        # Deterministic order: name ASCENDING first, then the DESCENDING composite key.
        # Python's sort is stable, so equal-composite-key candidates stay name-ordered.
        candidates.sort(key=lambda c: (getattr(c[0], "name", "") or ""))
        candidates.sort(key=lambda c: _rank_key(c, uc), reverse=True)

        try:
            n = max(1, int(top_n))
        except (TypeError, ValueError):
            n = 3
        top = candidates[:n]

        recs = []
        for i, (m, q, fr) in enumerate(top):
            recs.append(Recommendation(
                model=m, quant=q, fit=fr,
                why=_why(m, q, fr, uc, profile, is_top=(i == 0))))
        return recs
    except Exception:  # noqa: BLE001 — recommend must degrade, never crash the caller (T-03-02)
        return [_nothing_fits(list(CATALOG), _norm_use_case(use_case), profile)]
