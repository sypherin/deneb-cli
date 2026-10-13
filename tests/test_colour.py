"""Colour only for a human at a terminal. Piped output, a timer's journal and NO_COLOR
(https://no-color.org) get plain text, from every command, through one switch."""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import check, cli, ui  # noqa: E402

ESC = "\033"
PKG = Path(__file__).resolve().parent.parent / "deneb"


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _check_output(stream) -> str:
    real = check._probe, check._read
    check._reset_port_cache()
    check._probe = lambda cmd: ""          # an empty box: every row still prints
    check._read = lambda path: ""
    try:
        with redirect_stdout(stream):
            check.run()
    finally:
        check._probe, check._read = real
        check._reset_port_cache()
    return stream.getvalue()


def _no_color(value):
    old = os.environ.pop("NO_COLOR", None)
    if value is not None:
        os.environ["NO_COLOR"] = value
    return old


def _restore_no_color(old):
    os.environ.pop("NO_COLOR", None)
    if old is not None:
        os.environ["NO_COLOR"] = old


def test_check_piped_prints_no_escape_codes():
    old = _no_color(None)
    try:
        out = _check_output(io.StringIO())
    finally:
        _restore_no_color(old)
    assert out.count("\n") >= 6, out           # the rows really printed
    assert ESC not in out, repr(out[:200])


def test_check_on_a_terminal_is_coloured_and_no_color_turns_it_off():
    old = _no_color(None)
    try:
        assert ESC in _check_output(_Tty())
        os.environ["NO_COLOR"] = "1"
        out = _check_output(_Tty())
        assert out.count("\n") >= 6 and ESC not in out
        os.environ["NO_COLOR"] = ""            # no-color.org: an empty value does not count
        assert ESC in _check_output(_Tty())
    finally:
        _restore_no_color(old)


def test_banner_and_errors_piped_are_plain():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ui.banner()
        ui.error("x")
        ui.info("y")
    assert "Deneb" in buf.getvalue() and ESC not in buf.getvalue()


def test_guide_piped_is_plain():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["guide", "strix-halo"])
    assert rc == 0 and buf.getvalue().count("\n") > 10
    assert ESC not in buf.getvalue()


def test_escape_codes_are_written_in_one_module_only():
    hits = []
    for f in sorted(PKG.glob("*.py")):
        if f.name == "colour.py":
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if "\\033[" in line or "\\x1b[" in line or "\\u001b[" in line:
                hits.append(f"{f.name}:{n}")
    assert not hits, hits


def test_importing_ui_does_not_scan_the_code_point_range(monkeypatch):
    """The Cf table used to be built at import by calling unicodedata.category on all 1.1M
    code points. Count the calls during a reload (no timing): it must be a handful, not 1.1M."""
    import importlib
    import unicodedata
    from deneb import ui
    real, calls = unicodedata.category, []
    monkeypatch.setattr(unicodedata, "category", lambda ch: calls.append(ch) or real(ch))
    try:
        importlib.reload(ui)
    finally:
        monkeypatch.undo()
        importlib.reload(ui)
    assert len(calls) < 1000, len(calls)


def test_visible_escapes_exactly_the_same_characters_as_before():
    import unicodedata
    from deneb import ui
    for i in range(0x110000):
        ch = chr(i)
        if 0xD800 <= i <= 0xDFFF:
            continue
        old = (i < 0x20 or 0x7F <= i <= 0x9F or i in (0x2028, 0x2029, 0x180E)
               or unicodedata.category(ch) == "Cf")
        assert (ui.visible(ch) != ch) == old, hex(i)
