"""Verify Deneb never surfaces secrets: redaction + secret-file guard."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deneb import tools  # noqa: E402


def main() -> int:
    fails = []

    # 1. Redaction of secret-looking tokens in any output.
    samples = [
        "Bearer sk-altronis-49ef76b9402920aad107ebe9d7d8c7982372ee1a71e89dc6",
        'api_key = "abcd1234efgh5678ijkl"',
        "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "hash 49ef76b9402920aad107ebe9d7d8c7982372ee1a71e89dc6",
        "password=hunter2supersecret",
    ]
    for s in samples:
        r = tools._redact(s)
        # the long secret body must be gone
        if "49ef76b9402920aad107ebe9d7d8c7982372ee1a" in r or "ghp_ABCDEFGHIJKLMNOPQRST" in r \
           or "abcd1234efgh5678ijkl" in r or "hunter2supersecret" in r:
            fails.append(("NOT redacted", s, r))

    # 2. read_file on a secrets file returns existence, NOT the content.
    with tempfile.TemporaryDirectory() as d:
        kp = os.path.join(d, "keys.json")
        with open(kp, "w") as f:
            f.write('{"default":"sk-altronis-49ef76b9402920aad107ebe9d7d8c7982372ee1a71e89dc6"}')
        out = tools.read_file(kp)["output"]
        if "sk-altronis-49ef7" in out or "49ef76b9402920aad107" in out:
            fails.append(("secrets file content leaked", kp, out))
        if "does not read secret values" not in out:
            fails.append(("secrets file not guarded", kp, out))

        # 3. a normal file with a secret inside still gets redacted on read.
        cfg = os.path.join(d, "neo.conf")
        with open(cfg, "w") as f:
            f.write("upstream=8001\nauth_token: ABCDEFGHIJKLMNOPQRSTUV123456\n")
        out2 = tools.read_file(cfg)["output"]
        if "ABCDEFGHIJKLMNOPQRSTUV123456" in out2:
            fails.append(("secret in normal file not redacted", cfg, out2))

    if fails:
        print("SECRET-HYGIENE FAILURES:")
        for f in fails:
            print("  ", f)
        return 1
    print("SECRET HYGIENE PASS — tokens redacted, secret files guarded, config secrets redacted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
