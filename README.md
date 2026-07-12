# Deneb

Deneb is Altronis's terminal agent for standing up a **private local-LLM ("Neo") stack**
on your own AI box — NVIDIA **DGX Spark**, AMD **Strix Halo**, and similar. It installs,
configures, and **troubleshoots** the stack: it reads your machine's logs and config
directly, walks you to the fix, and tells you when you've reached a working, secure
endpoint — the **Neo Altronis reference state**.

It does exactly one job. It won't write your code or answer off-topic questions — it
gets your box to a working private LLM, and nothing else.

---

## Install

On the box you're setting up (needs internet):

```sh
curl -fsSL https://deneb.altronis.sg/install | sh
```

Sign in once with your Altronis token:

```sh
deneb auth --token <paste-your-token>
```

Done. `deneb --version` to confirm.

> No GitHub account, no Python fiddling, no config files to edit. One command installs,
> one command signs you in.

---

## Use it

Run `deneb` and describe what's wrong, or ask a one-shot question:

```sh
deneb                                  # interactive troubleshooting
deneb "llama-server won't start"
deneb "curl :8001/health returns nothing"
```

Deneb gathers the real evidence off your box itself (journalctl, the systemd unit, the
model path, `nvidia-smi` / `rocminfo`) and walks you to the fix — grounded in the
Altronis Neo runbook, with the exact commands shown verbatim.

### "Am I done?"

```sh
deneb check
```

Scans your box against the Neo reference and tells you exactly where you stand:

```
✓ llama.cpp built with CUDA (GPU, not CPU)
✓ model present at the expected path
✓ neo.service enabled + running
✗ :8001/health — not serving        → fix: <exact step>
✗ cloudflared tunnel — down          → fix: <exact step>
```

Green across the board = you're done.

### Paste a screenshot

Stuck on an on-screen error? Paste the screenshot, or:

```sh
deneb --image error.png "what's this?"
```

Deneb reads it (OCR) and diagnoses.

---

## What it can and can't do

- **Read-only.** It reads your logs, files, and config and runs read-only diagnostics.
  It never deletes, writes, restarts services, or touches firewall/ports — when a fix
  needs a mutating command, it hands you the exact command to run yourself.
- **Talks only to your own Neo Altronis cloud** (via your token). Your box's details are
  used to diagnose and go nowhere else. No third-party AI.
- **Stays in scope:** local-LLM setup on AI boxes. Anything else, it declines in one line.

---

## Uninstall

```sh
deneb logout          # forget your token
pipx uninstall deneb
```

---

*Built by Altronis. The intelligence — the Neo runbook + safety guardrails — lives in
your private Neo cloud; this client is open source, so you (or your client) can read
every line of what it does on the box.*
