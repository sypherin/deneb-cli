"""The version string `deneb --version` prints: the release, plus the git commit pipx
installed from, so an upgrade can be confirmed even between two builds of one release.

pip records where a package came from in the standard direct_url.json (PEP 610). A git
install carries vcs_info.commit_id; a PyPI, local-dir or editable install does not, and
then the plain release is the honest answer.
"""
from __future__ import annotations

import json
import re

from . import __version__

_SHA = re.compile(r"[0-9a-f]{7,40}")


def _direct_url():
    try:
        from importlib.metadata import distribution
        return distribution("deneb-cli").read_text("direct_url.json")
    except Exception:  # not installed as a package (running from a checkout): no commit known
        return None


def describe(direct_url_text) -> str:
    try:
        sha = json.loads(direct_url_text or "{}").get("vcs_info", {}).get("commit_id", "")
    except (ValueError, AttributeError):
        sha = ""
    if isinstance(sha, str) and _SHA.fullmatch(sha):
        return f"{__version__} (commit {sha[:7]})"
    return __version__


def installed() -> str:
    return describe(_direct_url())
