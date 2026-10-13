"""14 Oct stranger QC round 3: the seven rows still failing after the "fix essentials" round."""
from __future__ import annotations

import io
import os
import re
import shutil
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deneb import check, cli, recommend as rec, setup_advisor, watch  # noqa: E402
from deneb.models_catalog import Model, Quant  # noqa: E402
from fakes import FakeOpenAI  # noqa: E402

_OLLAMA = (11434, "ollama", "qwen2.5:0.5b")


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _main(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, _plain(buf.getvalue())


def _flat(s: str) -> str:
    return " ".join(s.split())


# F5: "Every quant of it is split" was false for gpt-oss-120b: Hugging Face serves a one-file
# F16 that deneb's catalog does not list. deneb can only speak for the quants it lists.
def test_the_all_split_note_speaks_only_for_the_quants_deneb_lists():
    import test_setup_advisor as tsa
    q = Quant("UD-Q4_K_XL", 20000, 4.8, parts=3)
    m = Model("Demo-30B", 30, "moe", 3, ["coding", "general"], [q], "unsloth/Demo-30B-GGUF")
    steps = setup_advisor.setup_steps(m, tsa.strix_box(107871), quant=q, family="dnf",
                                      free_gb=500, serving=_OLLAMA)
    note = next(s for s in steps if "Ollama" in s.title)
    assert "Every quant deneb lists for it is split" in note.what, note.what
    assert "Every quant of it is split" not in note.what, note.what


# NEW (round 3): on a box without curl (a fresh Ubuntu desktop) check said "not responding on
# :8001" while a server answered there: every endpoint probe was a curl command.
def _no_curl(monkeypatch):
    real = check._probe

    def probe(cmd):
        if cmd.startswith("curl "):
            return "(not installed on this box: curl)"
        return real(cmd)
    monkeypatch.setattr(check, "_probe", probe)


def test_a_box_without_curl_still_finds_the_server_answering(monkeypatch):
    _no_curl(monkeypatch)
    with FakeOpenAI(["x"], models=("qwen3.8-27b",)) as f:
        ok, model, _mv, _health = check._endpoint_state(f.port)
    assert ok and model == "qwen3.8-27b", (ok, model)


def test_without_curl_a_closed_port_reads_as_nothing_answering(monkeypatch):
    _no_curl(monkeypatch)
    with FakeOpenAI(["x"]) as f:
        port = f.port
    ok, _model, mv, _health = check._endpoint_state(port)     # the server is gone now
    assert not ok and not mv.startswith("(not installed"), mv


def test_without_curl_the_probe_ignores_a_proxy(monkeypatch):
    _no_curl(monkeypatch)
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    with FakeOpenAI(["x"], models=("m1",)) as f:
        ok, model, _mv, _health = check._endpoint_state(f.port)
    assert ok and model == "m1"


# NEW (round 3): the Ollama route said "downloads ~N GiB into Ollama's own model store" and
# never checked that store had room; the disk note only read ~/models.
def _ollama_setup(ollama_free):
    import test_setup_advisor as tsa
    q = Quant("UD-Q4_K_XL", 20000, 4.8, parts=1)
    m = Model("Demo-30B", 30, "moe", 3, ["coding", "general"], [q], "unsloth/Demo-30B-GGUF")
    steps = setup_advisor.setup_steps(m, tsa.strix_box(107871), quant=q, family="dnf",
                                      free_gb=500, serving=_OLLAMA, ollama_free=ollama_free)
    return next(s for s in steps if s.kind == "ollama")


def test_the_ollama_route_warns_when_its_store_has_no_room():
    step = _ollama_setup((5, "/usr/share/ollama"))
    first = _flat(step.warnings[0])
    assert "only 5 GiB is free on /usr/share/ollama" in first, step.warnings


def test_the_ollama_warning_does_not_claim_where_ollama_keeps_models():
    first = _flat(_ollama_setup((5, "/usr/share/ollama/.ollama/models")).warnings[0])
    assert "unless its service sets OLLAMA_MODELS" not in first, first
    assert "per-user ollama serve" in first and "check that disk instead" in first, first


def test_the_ollama_route_is_quiet_when_its_store_has_room():
    step = _ollama_setup((300, "/usr/share/ollama"))
    assert not any("is free on" in w for w in step.warnings), step.warnings


def test_the_ollama_store_follows_ollama_models(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "not" / "yet"))
    free, where = cli._ollama_store_free()
    assert where == str(tmp_path / "not" / "yet")
    assert free == shutil.disk_usage(tmp_path).free // (1024 ** 3)


def test_the_ollama_store_is_the_service_path_when_that_install_exists(monkeypatch):
    real_isdir = os.path.isdir
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: p == "/usr/share/ollama" or real_isdir(p))
    assert cli._ollama_store_free()[1] == "/usr/share/ollama/.ollama/models"
    monkeypatch.setattr(os.path, "isdir", lambda p: False if p == "/usr/share/ollama" else real_isdir(p))
    assert cli._ollama_store_free()[1] == os.path.expanduser("~/.ollama/models")


def test_setup_passes_the_ollama_store_space_through(monkeypatch):
    import test_setup_advisor as tsa
    from deneb import hardware
    monkeypatch.setattr(hardware, "profile_hardware", lambda: tsa.strix_box(107871))
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("dnf", 500))
    monkeypatch.setattr(cli, "_ollama_store_free", lambda: (1, "/srv/ollama-models"))
    monkeypatch.setattr(check, "serving_now", lambda: _OLLAMA)
    _rc, out = _main(["setup", "Qwen3-Coder-30B-A3B-Instruct"])
    assert "only 1 GiB is free on /srv/ollama-models" in _flat(out), out


# F24: `recommend --use coding` gave the same order as general, so the Coder models deneb
# lists never led it. The live catalog's Coder models now come first for coding.
def _live():
    import json
    from deneb import hf_catalog
    with open(hf_catalog.SNAPSHOT_PATH, encoding="utf-8") as f:
        ents = json.load(f)["models"]
    return [m for m in (hf_catalog.entry_to_model(e) for e in ents) if m]


def test_coding_puts_models_built_for_coding_first():
    import test_setup_advisor as tsa
    recs = rec.recommend(tsa.strix_box(107871), "coding", catalog=_live())
    assert recs[0].model.built_for == "coding", [r.model.name for r in recs]
    assert "built for coding" in recs[0].why, recs[0].why


def test_the_live_catalog_marks_only_its_coder_models():
    marked = sorted(m.name for m in _live() if m.built_for)
    assert marked and all("Coder" in n for n in marked), marked


def test_general_and_vision_order_are_unchanged_by_the_coding_tier():
    import test_setup_advisor as tsa
    for use in ("general", "vision"):
        top = rec.recommend(tsa.strix_box(107871), use, catalog=_live())[0]
        assert not top.model.built_for, (use, top.model.name)


def test_the_static_catalogs_older_coders_are_not_lifted():
    import test_setup_advisor as tsa
    top = rec.recommend(tsa.strix_box(107871), "coding")[0]
    assert top.model.name == "Qwen3.6-35B-A3B", top.model.name


def test_the_coding_tier_only_lifts_a_pick_that_fits_as_comfortably():
    coder = Model("Coder-7B", 7, "dense", 7, ["coding", "general"], [], "x/Coder-7B",
                  built_for="coding")
    big = Model("Big-70B", 70, "dense", 70, ["chat", "coding", "general"], [], "x/Big-70B")
    q = Quant("Q4_K_M", 4000, 4.8)

    class Fit:
        tight, expected_speed_tier, headroom_mb = False, "fast", 1000

    class Tight(Fit):
        tight = True
    assert rec._rank_key((coder, q, Fit()), "coding") > rec._rank_key((big, q, Fit()), "coding")
    assert rec._rank_key((big, q, Fit()), "general") > rec._rank_key((coder, q, Fit()), "general")
    assert rec._rank_key((big, q, Fit()), "coding") > rec._rank_key((coder, q, Tight()), "coding")


def test_the_coding_tier_does_not_lift_a_very_slow_coder():
    coder = Model("Coder-80B", 80, "dense", 80, ["coding", "general"], [], "x/Coder-80B",
                  built_for="coding")
    big = Model("Big-30B", 30, "moe", 3, ["chat", "coding", "general"], [], "x/Big-30B")
    q = Quant("Q4_K_M", 4000, 4.8)

    class Fast:
        tight, expected_speed_tier, headroom_mb = False, "fast", 1000

    class VerySlow(Fast):
        expected_speed_tier = "very-slow"
    assert rec._rank_key((big, q, Fast()), "coding") > rec._rank_key((coder, q, VerySlow()), "coding")


def test_the_order_note_says_coder_models_come_first(monkeypatch):
    import test_setup_advisor as tsa
    from deneb import hardware, hf_catalog
    monkeypatch.setattr(hardware, "profile_hardware", lambda: tsa.strix_box(107871))
    monkeypatch.setattr(cli, "_setup_inputs", lambda: ("dnf", 500))
    monkeypatch.setattr(hf_catalog, "ranking_catalog", lambda: (_live(), "live catalog (test)"))
    _rc, out = _main(["recommend", "--use", "coding"])
    assert "Models built for coding come first" in _flat(out), out
    _rc, out = _main(["recommend"])
    assert "built for coding" not in out, out


def test_readme_and_help_do_not_promise_recommend_lists_the_setup_example():
    readme = open(os.path.join(os.path.dirname(__file__), "..", "README.md"), encoding="utf-8").read()
    assert "a model recommend listed" not in readme


# NEW (round 3): selfupdate printed only the curl line; a box without curl needs the wget one
def test_selfupdate_gives_the_wget_line_too():
    _rc, out = _main(["selfupdate"])
    flat = _flat(out)
    assert cli.INSTALL_LINE in flat
    assert ("wget -qO- https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh"
            in flat), out


# F19: run as printed, `reset-failed` on a unit that never failed exits 1 with "not loaded"
def test_unschedule_says_reset_failed_is_only_for_a_failed_unit():
    lines = _plain(watch.unschedule_text()).splitlines()
    reset = lines.index("systemctl --user reset-failed deneb-watch.service")
    before = " ".join(lines[reset - 1].split())
    assert "only if" in before and "systemctl --user --failed" in before, lines[reset - 1]


# NEW (round 3): `deneb <command> --help` ran the command (setup said "usage", recommend ran)
_SUBS = ("check", "recommend", "setup", "watch", "profile", "catalog", "guide", "stack",
         "selfupdate", "brain", "logout")


def test_every_command_answers_its_own_help_without_running(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the command ran")
    for name in ("cmd_profile", "cmd_recommend", "cmd_setup", "cmd_catalog", "cmd_guide",
                 "cmd_stack", "cmd_selfupdate", "cmd_watch", "cmd_brain", "cmd_logout"):
        monkeypatch.setattr(cli, name, boom)
    monkeypatch.setattr(check, "run", boom)
    for sub in _SUBS:
        for flag in ("--help", "-h"):
            rc, out = _main([sub, flag])
            assert rc == 0, (sub, flag, out)
            assert f"deneb {sub}" in out and "deneb --help" in out, (sub, out)


def test_subcommand_help_shows_that_commands_options():
    _rc, out = _main(["watch", "--help"])
    assert "--schedule" in out and "--unschedule" in out, out
    _rc, out = _main(["recommend", "-h"])
    assert "--use" in out, out


def test_auth_keeps_its_own_usage():
    _rc, out = _main(["auth", "--help"])
    assert "deneb auth --request" in out
