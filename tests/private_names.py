"""Checks that must spell out private names read them from a file kept outside the repo.

DENEB_PRIVATE_DENYLIST names a text file, one regex per line (blank lines and lines starting
with # are ignored). Unset, every check here does nothing, so the public repo never has to
contain the names themselves."""
from __future__ import annotations

import os
import re

import pytest

ENV = "DENEB_PRIVATE_DENYLIST"


def patterns() -> list:
    path = os.environ.get(ENV)
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    return [re.compile(ln, re.I) for ln in lines if ln and not ln.startswith("#")]


def require() -> list:
    """The patterns, or skip the calling test when the env var is unset."""
    pats = patterns()
    if not pats:
        pytest.skip(f"{ENV} is not set")
    return pats


def assert_clean(text: str, label: str = "") -> None:
    """No private name in `text`. Does nothing when the deny-list is not configured."""
    for rx in patterns():
        hit = rx.search(text)
        assert not hit, f"private name matched in {label or 'text'} (pattern #{patterns().index(rx) + 1})"
