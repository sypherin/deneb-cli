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

On the box you're setting up (needs internet). **One line:**

```sh
curl -fsSL https://deneb-engine.altronis.sg/install | sh
```

(Equivalent: `curl -fsSL https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh` —
same script, served from this repo.)

Or, if you prefer to install it yourself with [pipx](https://pipx.pypa.io):

```sh
pipx install git+https://github.com/sypherin/deneb-cli.git
```

Then sign in once with your Altronis token:

```sh
deneb auth --token <paste-your-token>
```

Done. `deneb --version` to confirm.

> No config files to edit. One command installs, one command signs you in. To upgrade
> later, re-run the installer (or `pipx upgrade deneb`).

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

A **deterministic** scan (no LLM, ~0.3s): it checks each Neo component by the port it
serves on and the process that owns it — never by a guessed service name — and tells you
exactly where you stand:

```
✓ LLM endpoint serving (:8001)          — serving Qwen3.6-35B-A3B
✓ Model service enabled + running       — llama-server.service, enabled at boot
✓ Built with GPU accelerator (not CPU)  — GPU offload on, cuda   (vulkan/rocm on AMD)
✓ Model file present                    — Qwen3.6-35B-A3B.gguf
✗ Auth gateway up (:8002)               — nothing listening on :8002
      fix: stand up the bearer-key gateway on :8002 in front of :8001
✗ Cloudflare tunnel live                — no tunnel (cloudflared not running)
      fix: install + run cloudflared outbound-only
```

Green across the board = you've reached the working Neo Altronis state. For anything
failing, ask `deneb "why is <that> failing"` for a deeper look.

### Paste a screenshot

Stuck on an on-screen error? Paste the screenshot, or:

```sh
deneb --image error.png "what's this?"
```

Deneb reads it (OCR) and diagnoses.

---

## Fixing (not just telling you)

Once Deneb has diagnosed the cause, it can **apply the fix itself** — edit a unit's
`ExecStart`, `mkdir` a missing directory, `systemctl --user daemon-reload` + `restart`,
symlink a binary into place — then re-verify it worked.

- **Gate by default.** It shows you each change and asks before running it. Pass `--auto`
  to let it apply fixes without asking (still bounded by the limits below).
- **Always reversible + unprivileged.** Before editing any file it makes a `.deneb-bak`
  backup. It only ever runs reversible, non-elevated actions.
- **Never destructive, irreversible, or privileged.** No `rm`/`mv`/`dd`/`chmod`, no `sudo`,
  no system-wide `systemctl`, no firewall/port change, no package install. If the real fix
  needs one of those, Deneb hands you the exact command to run yourself.

## What it can and can't do

- **The client is the boundary.** The read-only diagnostics and the apply allowlist are both
  enforced here, deny-by-default, `shell=False`, no `sudo` — auditable in [`deneb/tools.py`](deneb/tools.py)
  and covered by the tests in [`tests/`](tests). Even a tampered server can't make it run
  something destructive.
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
