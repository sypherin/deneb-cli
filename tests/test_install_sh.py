"""install.sh under a scratch HOME and a fake PATH of stubs only: no real pipx, python3, git,
curl, sudo or network is ever reached. The script is run with `sh` by absolute path."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "git+https://github.com/sypherin/deneb-cli.git"


def _stub(bindir: Path, name: str, body: str) -> None:
    f = bindir / name
    f.write_text("#!/bin/sh\n" + body)
    f.chmod(f.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path, stubs: dict):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    for name, body in stubs.items():
        _stub(bindir, name, f'echo "{name} $*" >> "{log}"\n' + body)
    home = tmp_path / "home"
    home.mkdir()
    r = subprocess.run(["/bin/sh", str(ROOT / "install.sh")], capture_output=True, text=True,
                       timeout=30, env={"HOME": str(home), "PATH": str(bindir)})
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


# python3 with no pipx and a pip that refuses (PEP 668): both `-m pipx --version` and
# `-m pip install` fail
_BLOCKED_PYTHON = "exit 1\n"


def test_pep668_box_prints_the_three_install_commands_and_stops(tmp_path):
    r, calls = _run(tmp_path, {"python3": _BLOCKED_PYTHON, "git": "exit 0\n", "sudo": "exit 0\n"})
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    for cmd in ("sudo dnf install pipx", "sudo apt install pipx", "brew install pipx"):
        assert cmd in out, out
    assert "then re-run this installer" in out
    assert "installing deneb" not in out
    assert not any(c.startswith("sudo") for c in calls), calls
    assert any("pip install --user" in c for c in calls), "the no-sudo pip attempt is kept"
    assert "python3 -m pip install --user pipx" not in out, "the old hint fails the same way"


def test_a_working_pipx_reaches_the_install_and_nothing_else_runs(tmp_path):
    r, calls = _run(tmp_path, {"python3": "exit 0\n", "git": "exit 0\n", "pipx": "exit 0\n",
                               "sudo": "exit 0\n"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"pipx install --force {REPO_URL}" in calls, calls
    assert not any(c.startswith("sudo") for c in calls), calls
    assert "installing deneb" in r.stdout


def test_a_missing_pipx_that_pip_can_install_still_goes_on(tmp_path):
    # python3 -m pipx --version fails once (not set up), `-m pip install` works, then pipx works
    body = ('case "$*" in\n'
            '  "-m pipx --version") [ -f "$HOME/.pipx-ok" ] || exit 1; exit 0;;\n'
            '  "-m pip install --user -q pipx") : > "$HOME/.pipx-ok"; exit 0;;\n'
            'esac\nexit 0\n')
    r, calls = _run(tmp_path, {"python3": body, "git": "exit 0\n", "sudo": "exit 0\n"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert any(c.startswith("python3 -m pipx install --force " + REPO_URL) for c in calls), calls
    assert not any(c.startswith("sudo") for c in calls)


def test_the_script_never_runs_sudo_and_has_no_em_dash():
    text = (ROOT / "install.sh").read_text()
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        assert not s.startswith("sudo ") and "&& sudo" not in s and "| sudo" not in s, ln
    assert "—" not in text


def test_readme_describes_the_installer_pipx_behaviour():
    text = " ".join((ROOT / "README.md").read_text().split())      # undo the line wrapping
    assert "installs pipx for you" not in text
    assert "tries to set it up without sudo" in text and "prints the command to install pipx" in text


# --- 14 Oct QC: F2 (deneb not found after install on Ubuntu desktop), F15 (no install line) ---

_PIPX_ENV = ('case "$*" in\n'
             '  "environment --value PIPX_BIN_DIR") echo "$HOME/.local/bin";;\n'
             'esac\nexit 0\n')


def test_deneb_not_on_path_prints_the_full_path_and_the_export_line(tmp_path):
    r, _ = _run(tmp_path, {"python3": "exit 0\n", "git": "exit 0\n", "pipx": _PIPX_ENV})
    out = r.stdout + r.stderr
    home = tmp_path / "home"
    assert r.returncode == 0, out
    assert f"{home}/.local/bin/deneb check" in out, out
    assert f'export PATH="{home}/.local/bin:$PATH"' in out, out
    assert "log out and back in" in out, out
    assert "open a new terminal" not in out, "a new tab does not pick up the PATH on Ubuntu"


def test_deneb_already_on_path_gets_no_path_advice(tmp_path):
    r, _ = _run(tmp_path, {"python3": "exit 0\n", "git": "exit 0\n", "pipx": _PIPX_ENV,
                           "deneb": "exit 0\n"})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "export PATH=" not in out and "log out and back in" not in out, out


def test_an_old_pipx_without_environment_falls_back_to_local_bin(tmp_path):
    r, _ = _run(tmp_path, {"python3": "exit 0\n", "git": "exit 0\n",
                           "pipx": 'case "$1" in environment) exit 2;; esac\nexit 0\n'})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f'export PATH="{tmp_path / "home"}/.local/bin:$PATH"' in out, out


def test_missing_git_prints_the_install_line_for_each_system(tmp_path):
    r, calls = _run(tmp_path, {"python3": "exit 0\n", "sudo": "exit 0\n"})
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    for cmd in ("sudo dnf install git", "sudo apt install git", "brew install git"):
        assert cmd in out, out
    assert "then re-run this installer" in out
    assert not any(c.startswith("sudo") for c in calls), calls


def test_missing_python_prints_the_install_line_for_each_system(tmp_path):
    r, _ = _run(tmp_path, {"git": "exit 0\n"})
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    for cmd in ("sudo dnf install python3", "sudo apt install python3", "brew install python"):
        assert cmd in out, out


def test_installer_output_fits_80_columns(tmp_path):
    import re
    r, _ = _run(tmp_path, {"python3": "exit 0\n", "git": "exit 0\n", "pipx": _PIPX_ENV})
    plain = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    home = str(tmp_path / "home")
    for ln in plain.splitlines():            # a stranger's HOME is /home/<name>, not pytest's tmp
        ln = ln.replace(home, "/home/sam")
        assert len(ln) <= 80, ln
