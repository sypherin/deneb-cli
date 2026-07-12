"""Red-team the APPLY (write) tier — deneb.tools.run_write / write_file / execute_write.

The write tier can MUTATE a customer's box, so it is the highest-blast-radius code in the
client. This test asserts the hard limits hold regardless of what the engine asks:
  - destructive / irreversible / privileged commands are REFUSED (rm, mv, dd, chmod, kill,
    sudo, system-wide systemctl, firewall, package installs, git/curl/sed, redirects…)
  - only reversible, non-elevated fixes run (systemctl --user safe verbs, mkdir, ln -s)
  - write_file BACKS UP an existing file before overwriting, refuses secrets + system paths

Run: python3 tests/test_write_tier.py   (also collectable by pytest)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deneb import tools  # noqa: E402


def _refused(res):
    return res.get("refused") is True or str(res.get("output", "")).startswith("[refused]")


# ── commands that must ALWAYS be refused ─────────────────────────────────────
REFUSE = [
    "rm -rf /home/cf/models",          # destroy data
    "rm file.txt",
    "rmdir /opt/llama",
    "mv /opt/llama /tmp/x",            # move = can destroy
    "dd if=/dev/zero of=/dev/sda",
    "mkfs.ext4 /dev/sdb",
    "shred -u secret",
    "truncate -s 0 model.gguf",
    "chmod 777 /etc/passwd",
    "chown root:root /opt",
    "kill -9 876334",                 # could kill the model server
    "pkill -f llama-server",
    "killall python3",
    "sudo systemctl restart llama-server",   # elevation
    "su -c 'rm x'",
    "pkexec rm x",
    "systemctl restart llama-server",         # system scope (needs root)
    "systemctl enable llama-server.service",  # no --user
    "systemctl --user mask llama-server",     # mask is not a safe verb
    "systemctl --user kill llama-server",
    "systemctl --user poweroff",
    "ufw allow 8001",                 # firewall
    "iptables -F",
    "firewall-cmd --add-port=8001/tcp",
    "apt-get install -y llama",       # package install (needs root)
    "dnf install llama",
    "pip install vllm",               # heavy / arbitrary code
    "git clone https://x/y",          # deferred from v1 write tier
    "curl -O https://x/model.gguf",   # download not in v1 write tier
    "wget https://x -O m.gguf",
    "sed -i s/a/b/ unit.service",     # in-place edit outside write_file's backup
    "tee /etc/x",
    "make install",
    "ln -f /a /b",                    # force-overwrite symlink (not reversible)
    "ln /a /b",                       # hard link (only -s allowed)
    "mkdir x && rm y",                # chaining
    "systemctl --user restart a; rm b",
    "echo hi > /etc/x",               # redirect
]

# ── commands that SHOULD be allowed (reversible, non-elevated) ───────────────
ALLOW = [
    "systemctl --user daemon-reload",
    "systemctl --user restart llama-server.service",
    "systemctl --user start neo-gateway.service",
    "systemctl --user stop llama-server.service",
    "systemctl --user enable llama-server.service",
    "systemctl --user disable llama-server.service",
    "systemctl --user reset-failed llama-server.service",
    "mkdir -p /home/x/models",
    "ln -s /home/cf/build/llama.cpp/build/bin/llama-server /home/x/.local/bin/llama-server",
]


def test_destructive_all_refused():
    bad = [c for c in REFUSE if not _refused(tools.run_write(c))]
    assert not bad, f"these destructive/elevated commands were NOT refused: {bad}"


def test_reversible_pass_guard():
    # They pass the ALLOWLIST guard (they may then fail to actually run — e.g. no such
    # unit — but must not be [refused] by the boundary). We check the guard verdict only.
    blocked = [c for c in ALLOW if _refused(tools.run_write(c))]
    assert not blocked, f"these safe fixes were wrongly refused by the guard: {blocked}"


# ── write_file: backup + guards ──────────────────────────────────────────────
def test_write_file_backs_up_existing():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "llama-server.service")
    with open(p, "w") as f:
        f.write("ExecStart=/opt/llama/llama-server\n")   # the wrong path
    res = tools.write_file(p, "ExecStart=/home/cf/build/bin/llama-server\n")
    assert res.get("ok") is True
    assert res.get("backup") and os.path.exists(res["backup"]), "no backup was made"
    # the backup preserves the ORIGINAL, the file has the NEW content
    assert "opt/llama" in open(res["backup"]).read()
    assert "home/cf/build" in open(p).read()


def test_write_file_refuses_secret_and_system():
    assert _refused(tools.write_file("/home/x/.env", "SECRET=1"))          # secret file
    assert _refused(tools.write_file("/home/x/keys.json", "{}"))           # secret file
    assert _refused(tools.write_file("/etc/systemd/system/x.service", "")) # system path
    assert _refused(tools.write_file("/usr/bin/x", ""))


def test_write_file_new_file_ok_no_backup():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "new.conf")
    res = tools.write_file(p, "key=value\n")
    assert res.get("ok") is True and res.get("backup") == "" and os.path.exists(p)


def test_execute_write_dispatch_and_unknown():
    assert _refused(tools.execute_write("run", cmd="rm x"))     # 'run' is read-tier, not apply
    assert _refused(tools.execute_write("nope"))
    # fix dispatches to run_write (this one is destructive → refused)
    assert _refused(tools.execute_write("fix", cmd="rm -rf /"))


def test_read_tier_still_cannot_write():
    # the read dispatch must never accept a write action
    assert _refused(tools.execute("fix", cmd="systemctl --user restart x"))
    assert _refused(tools.execute("write_file", path="/tmp/x"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
