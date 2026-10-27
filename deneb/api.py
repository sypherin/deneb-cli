"""deneb.api - the shared orchestration behind the keyless verbs.

`deneb recommend` and `deneb setup` (the CLI) and the Deneb MCP answer the same two
questions. This module gathers the box profile and the catalog once and calls the engine
(recommend.py, setup_advisor.py, measured.py) - so every face gives the SAME answer, with
no re-implementation of the selection. It returns DATA; each face formats it (the CLI for
a terminal, the MCP for an agent).

Keyless and engine-free, like the CLI verbs it backs: no auth, no LLM, no hosted-engine
round-trip, and it RUNS NOTHING (the Deneb Rule). Only `measured=True` reads TokenMark
(tokenmark.app), and that call fails soft - a missing site is an empty result, never a raise.

The box-input helpers (_setup_inputs, _ollama_store_free, _tokenmark_hw_hint) stay in cli.py
and are called from there at run time: the test-suite patches them on the `cli` module, so
reading them live keeps that contract. stdlib only, no new dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecommendResult:
    """What `recommend()` found: the ranked catalog picks for a use-case on this box, plus the
    inputs a formatter needs (profile, catalog source, bandwidth, free disk) and, when asked,
    the single best measured TokenMark row per model."""
    profile: object
    use_case: str
    source: str
    bw_gbps: Optional[float]
    free_gb: Optional[int]
    recs: list                                      # recommend.recommend() objects, ranked
    measured: dict = field(default_factory=dict)    # model name -> best TokenMark row (measured only)
    hw_hint: Optional[str] = None                   # TokenMark hardware string used for measured


@dataclass
class SetupResult:
    """What `setup()` built: the resolved model, the box profile, what serves now, the ordered
    tell-only steps, and (when asked) the measured selection whose lines explain the pick."""
    model: object
    profile: object
    serving: object
    steps: list                                     # setup_advisor Step objects, in order
    selection: object = None                        # measured.Selection or None (measured only)


def suggest_model(name: str) -> Optional[str]:
    """The closest catalog model name to `name`, for a 'did you mean ...?' hint, or None.
    A cosmetic suggestion only - it never selects or sets up anything."""
    import difflib
    from . import hf_catalog, setup_advisor
    names = [getattr(m, "name", "") for m in hf_catalog.all_models()]
    close = difflib.get_close_matches(name, names, n=1, cutoff=0.4)
    return close[0] if close else None


def recommend(use_case: str, measured: bool = False, profile=None) -> RecommendResult:
    """Rank the model catalog for `use_case` on this box. `measured=True` adds the best real
    TokenMark tok/s per model (one GET, fail-soft). `profile` lets a caller pass an already
    read profile; the default reads the box once. The same calls cmd_recommend made."""
    from . import cli, hardware, hf_catalog, speed, recommend as rec
    p = profile if profile is not None else hardware.profile_hardware()
    models, source = hf_catalog.ranking_catalog()
    bw = speed.usable_bandwidth(p, speed.device_for_profile(p))
    recs = rec.recommend(p, use_case, catalog=models, bw_gbps=bw)
    _family, free_gb = cli._setup_inputs()
    meas: dict = {}
    hw_hint = None
    if measured:
        from . import measured as measured_mod, tokenmark
        # TokenMark's own hardware ids for this box (exact: a10 never pulls a100 rows); the
        # loose name hint is only the fallback for a box the hardware catalog does not know.
        ids = measured_mod.hardware_candidates(p) or None
        dev = speed.device_for_profile(p)
        hw_hint = cli._tokenmark_hw_hint(p) or (str(dev["name"]) if dev and dev.get("name") else None)
        for r in recs:
            name = getattr(r.model, "name", "") or ""
            # measured is a soft add-on: a reachable-but-malformed response, or any error,
            # drops that model's measurement - it never breaks the recommendation. (Mirrors
            # the top-level guard in measured.choose_for_box on the setup path.)
            try:
                rows = tokenmark.measured_for_model(name, hardware=hw_hint, hardware_ids=ids)
            except Exception:
                rows = None
            if rows:
                meas[name] = rows[0]
    return RecommendResult(p, use_case, source, bw, free_gb, recs, meas, hw_hint)


def setup(model_name: str, measured: bool = False, profile=None) -> Optional[SetupResult]:
    """Resolve `model_name` in the curated catalog and build the ordered tell-only setup steps
    for this box. Returns None when the name is unknown (the caller shows the did-you-mean).
    `measured=True` picks the measured TokenMark row Deneb can reproduce here and records it
    for `deneb check`. The same calls cmd_setup made."""
    from . import cli, hardware, hf_catalog, setup_advisor, check
    model = setup_advisor.resolve_model(model_name, hf_catalog.all_models())
    if model is None:
        return None
    p = profile if profile is not None else hardware.profile_hardware()
    family, free_gb = cli._setup_inputs()
    serving = check.serving_now()
    store = cli._ollama_store_free() if serving and serving[1] == "ollama" else None
    sel = None
    if measured:
        from . import measured as measured_mod
        sel = measured_mod.choose_for_box(model, p)
    steps = setup_advisor.one_way_to_start(
        setup_advisor.setup_steps(model, p, family=family, free_gb=free_gb, serving=serving,
                                  ollama_free=store, measured=sel.choice if sel else None),
        systemd=cli._has_systemd())
    if sel and sel.choice:
        steps.append(measured_mod.record_step(sel.expected))
    return SetupResult(model, p, serving, steps, sel)
