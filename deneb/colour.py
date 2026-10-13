"""deneb.colour — the one switch for terminal colour. ANSI only for a human at a terminal:
piped output, a timer's journal and NO_COLOR (https://no-color.org) get plain text.
"""
from __future__ import annotations

import os
import sys

_CODES = {
    "green": "\033[32m", "red": "\033[31m", "amber": "\033[33m", "gold": "\033[38;5;178m",
    "teal": "\033[38;5;44m", "grey": "\033[90m", "dim": "\033[2m", "bold": "\033[1m",
    "reset": "\033[0m",
}


def enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class Palette:
    """A module's short names for the codes above. Decided at each lookup, so output sent to a
    pipe (or a test's buffer) after import still comes out plain."""

    def __init__(self, **names: str):
        self._names = names

    def __getitem__(self, key: str) -> str:
        return _CODES[self._names[key]] if enabled() else ""
