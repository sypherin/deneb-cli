"""14 Oct QC F1 + F13: an upgrade must be visible, and there is one install line.

F1: the version stayed 0.3.0 across 49 commits, so `deneb --version` could not show an
upgrade had landed. Now it also names the git commit pipx installed from (pip's standard
direct_url.json), so two builds never look the same even if a bump is forgotten.
F13: selfupdate printed a third install URL; it now prints the README's line, word for word.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deneb import __version__, cli, version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GIT = {"url": "https://github.com/sypherin/deneb-cli.git",
       "vcs_info": {"vcs": "git", "commit_id": "49c5675aa0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5"}}


def _out(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


def test_pyproject_and_package_agree_and_moved_past_0_3_0():
    m = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    assert m and m.group(1) == __version__
    assert __version__ != "0.3.0"


def test_a_git_install_names_its_commit():
    assert version.describe(json.dumps(GIT)) == f"{__version__} (commit 49c5675)"


def test_no_commit_known_means_the_plain_version():
    assert version.describe(None) == __version__
    assert version.describe("") == __version__
    assert version.describe("{not json") == __version__
    assert version.describe(json.dumps({"url": "file:///src", "dir_info": {}})) == __version__
    assert version.describe(json.dumps({"vcs_info": {"commit_id": "zz; rm -rf"}})) == __version__


def test_dash_dash_version_uses_it(monkeypatch):
    monkeypatch.setattr(version, "_direct_url", lambda: json.dumps(GIT))
    rc, out = _out(["--version"])
    assert rc == 0 and out.strip() == f"deneb {__version__} (commit 49c5675)"


def _readme_install_line():
    for ln in (ROOT / "README.md").read_text().splitlines():
        if ln.startswith("curl -fsSL ") and ln.endswith("| sh"):
            return ln
    raise AssertionError("README has no curl install line")


def test_selfupdate_prints_the_readme_install_line_and_no_other_url():
    rc, out = _out(["selfupdate"])
    assert rc == 0
    assert _readme_install_line() in out, out
    assert "deneb-engine.altronis.sg" not in out
    urls = set(re.findall(r"https?://\S+", out))
    assert urls == {"https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh"}, urls


def test_selfupdate_remote_wraps_the_same_line():
    rc, out = _out(["selfupdate", "--remote", "sam@aibox"])
    assert rc == 0 and f"ssh sam@aibox '{_readme_install_line()}'" in out, out


def test_installer_header_matches_the_readme():
    assert "#   " + _readme_install_line() in (ROOT / "install.sh").read_text()
