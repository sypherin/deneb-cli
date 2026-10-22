"""deneb.api is the one orchestration both the CLI and the Deneb MCP call. These tests pin that
it DELEGATES to the engine (recommend.py / setup_advisor.py / measured.py) instead of growing a
second ranking of its own. If a future edit re-implements selection inside api, the parity
assertions below go red - that is the point of the file.

Keyless by default: the recommend/setup paths touch no network, so these run offline. The two
measured paths are monkeypatched, never hitting tokenmark.app.
"""
from deneb import api, hardware, hf_catalog, recommend, speed


def _engine_rec_names(use_case):
    """The ranked names straight from the engine, built the same way api builds them."""
    p = hardware.profile_hardware()
    models, _ = hf_catalog.ranking_catalog()
    bw = speed.usable_bandwidth(p, speed.device_for_profile(p))
    return [getattr(r.model, "name", "") for r in recommend.recommend(p, use_case, catalog=models, bw_gbps=bw)]


def test_recommend_matches_engine_order():
    # the crux: api must return the engine's ranking verbatim, for every use-case
    for uc in ("coding", "vision", "chat", "general"):
        got = [getattr(r.model, "name", "") for r in api.recommend(uc).recs]
        assert got, f"{uc}: empty"
        assert got == _engine_rec_names(uc), uc


def test_recommend_result_shape_is_keyless():
    data = api.recommend("coding")
    assert data.use_case == "coding"
    assert data.profile is not None
    assert isinstance(data.recs, list) and data.recs
    assert data.measured == {}       # no network on the default path
    assert data.hw_hint is None


def test_recommend_measured_empty_is_failsoft(monkeypatch):
    from deneb import tokenmark
    monkeypatch.setattr(tokenmark, "measured_for_model", lambda *a, **k: [])
    data = api.recommend("coding", measured=True)
    assert data.measured == {}                       # no rows -> empty, never a raise
    assert data.hw_hint is None or isinstance(data.hw_hint, str)  # box hint, str or None


def test_recommend_measured_raise_is_failsoft(monkeypatch):
    # the contract: measured=True must NEVER propagate a TokenMark failure. A reachable-but-bad
    # response (or any error) drops the measurement, it does not break the recommendation.
    from deneb import tokenmark

    def boom(*a, **k):
        raise RuntimeError("tokenmark returned garbage")

    monkeypatch.setattr(tokenmark, "measured_for_model", boom)
    data = api.recommend("coding", measured=True)     # must not raise
    assert data.measured == {}
    assert data.recs                                  # the picks still came back


def test_setup_unknown_model_returns_none():
    assert api.setup("no-such-model-xyz-123") is None


def test_setup_known_model_builds_tell_only_steps():
    name = _engine_rec_names("coding")[0]
    data = api.setup(name)
    assert data is not None
    assert getattr(data.model, "name", "") == name
    assert isinstance(data.steps, list) and data.steps
    assert all(getattr(s, "title", "") for s in data.steps)  # every step is at least titled
    assert data.selection is None     # keyless default


def test_setup_measured_is_failsoft(monkeypatch):
    from deneb import measured
    name = _engine_rec_names("coding")[0]
    monkeypatch.setattr(measured, "choose_for_box",
                        lambda *a, **k: measured.Selection(choice=None, lines=[], expected=None))
    data = api.setup(name, measured=True)
    assert data is not None
    assert data.selection is not None
    assert isinstance(data.steps, list) and data.steps


def test_suggest_model_finds_a_near_miss():
    name = _engine_rec_names("coding")[0]
    near = name[:-2] if len(name) > 5 else name
    assert api.suggest_model(near) is not None


def test_suggest_model_handles_garbage_without_raising():
    res = api.suggest_model("################")
    assert res is None or isinstance(res, str)
