# Deneb

Check, set up and watch a local LLM server on your own AI box: AMD Strix Halo (Ryzen AI
Max+ 395), NVIDIA DGX Spark (GB10), GeForce and workstation GPUs, Radeon, Intel Arc and
Apple M-series. It works with llama.cpp (`llama-server`), vLLM, Ollama and LM Studio.

The main commands are free, need no account or token, and send nothing unless you ask for it
(an alert, a catalog refresh, measured speeds). They read the box and print the exact fix.
They never run it: you do.

```sh
deneb check                    # is the model served, on the GPU, ready?
deneb recommend --use coding   # which models fit this box, with estimated tokens/s
deneb setup Qwen3.6-35B-A3B    # the setup steps for any model deneb catalog lists
deneb watch --schedule         # keep it healthy: what changed since the last check, with alerts
```

MIT licensed, Python 3.9+, standard library only.

---

## Install

On the box itself:

```sh
curl -fsSL https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh
```

No `curl`? Ubuntu desktop ships without it, but has `wget`:

```sh
wget -qO- https://raw.githubusercontent.com/sypherin/deneb-cli/main/install.sh | sh
```

The script installs `deneb` with [pipx](https://pipx.pypa.io) and never runs sudo. It
needs `python3` and `git`, and prints the install command for whichever is missing. If pipx is missing, the installer tries to set it up without sudo.
On systems that block that, such as recent Ubuntu and Debian, it stops and prints the command
to install pipx. Or install it yourself:

```sh
pipx install git+https://github.com/sypherin/deneb-cli.git
```

`deneb --version` to confirm. To upgrade, re-run either command (with pipx:
`pipx install --force git+https://github.com/sypherin/deneb-cli.git`).

---

## `deneb check`: is the box ready?

A read-only scan with no model call. It finds the model server by the port it listens on and
the process that owns it, never by a guessed service name, then reports what is right and
what is not, with the fix for each failure:

```
  ✓ LLM endpoint serving (:11434)  — serving qwen2.5:0.5b
  ✓ Model service enabled + running  — ollama.service (system) serving :11434, enabled at boot
  ✓ Built with GPU accelerator (not CPU)  — qwen2.5:0.5b 100% GPU
  – Model file present  — Ollama manages its own model files; not checked
  – Auth gateway up (:8002, key + rate-limit)  — not set up (optional: only needed to reach the model from other machines)
  – Cloudflare tunnel live  — not set up (optional: only needed to reach the model from other machines)

✓ all checks pass: the model is served and ready.
```

- **Which port.** It tries the usual ports in this order: 8001, 8080 (llama-server), 8000
  (vLLM), 11434 (Ollama) and 1234 (LM Studio). A port counts when something listens on it,
  `/v1/models` answers with a `data` list, and the listener is a model server: a known one
  (llama-server, vLLM, Ollama, LM Studio), one whose owner check can't see, or one that
  lists an entry with `"object": "model"`. Any other app that happens to answer JSON is
  skipped. Set `DENEB_PORT` to point it somewhere else. Without `ss` (macOS, a minimal
  container) it probes those ports with `curl` alone and reads the kind from the HTTP API.
- **Which service.** It reads the systemd unit that owns the server process, user or system
  (`ollama.service`, for example). A server started by hand in a terminal is not a unit, and
  check says so. When the server runs as another user (the default Ollama install does), the
  process is hidden. Check then asks the server's own API what it is, never guesses a process
  by name, and for Ollama looks up `ollama.service`. If it can't tell, the row is a warning
  that says the owner is out of sight, and the GPU and model file rows are skipped for that
  server.
- **GPU or CPU.** For llama.cpp it reads the GPU offload flag (`-ngl`) from the running
  process, and which accelerator (CUDA, ROCm or Vulkan) the server has loaded or its build
  links, so a CPU-only build is flagged. For Ollama it reads `ollama ps`, which can only tell once a model is
  loaded: with none loaded the row is `?` until you send it one request. For vLLM it looks
  for `--device cpu`. A server it doesn't recognise is not judged.
- **Model file.** Checked for llama.cpp. vLLM, Ollama and LM Studio manage their own model
  files, so that row is skipped for them.
- **Gateway and tunnel.** An auth gateway in front of the model and a Cloudflare tunnel are
  checked only if you run them. If you don't, the rows show `–` and do not count as failures.

Exit codes: `0` every row passed, `1` only warnings (a ⚠ row) or a row it could not read (shown as `?`), `2` at least one ✗ row.

---

## `deneb watch`: keep it healthy

`deneb check` tells you the box is set up. `deneb watch` tells you it is still healthy, and
what changed since the last look. It runs the same scan plus four upkeep probes: free disk on
the model's filesystem, available memory, restarts of the model's service (a jump of 3 or more
since the last run is flagged as a crash loop), and the age of Deneb's model catalog. It
remembers the last run, marks what is NEW, WORSE or RECOVERED, and prints the fix for
anything wrong.

```sh
deneb watch                            # run once
deneb watch --quiet                    # only the changes and a one-line summary (for timers)
deneb watch --history [N]              # the last N change events (default 20)
deneb watch --schedule [--every 15m]   # print a systemd --user timer that runs it for you
deneb watch --unschedule               # print the commands that remove that timer again
deneb watch --reset                    # forget the last run and start a new baseline
```

If a gateway or tunnel was seen on an earlier run and is now gone, watch reports it as a
failure. If you removed it on purpose, `deneb watch --reset`.

Watch also remembers which port the model was on. If the model moves, say from :8001 to
:8080, it warns once, marked NEW: "model server moved from :8001 to :8080 (if :8001 was
your model, it is down)". A dead model can otherwise look healthy when another server
answers on a different port. `deneb watch --reset` forgets the stored port. If another
watch holds the lock, `--reset` prints "reset NOT done" and exits 3, so you know to retry.

Exit codes: `0` all ok, `1` a warning or a row it could not read (`?`), `2` a failure, `3` watch itself broke. The printed
service unit has `SuccessExitStatus=1 2`, so a problem on the box does not mark the unit
failed: it reaches you through the alert or `journalctl --user -u deneb-watch.service -e`.
A unit shown in `systemctl --user --failed` means the watcher itself broke. The timer runs
every `15m` unless you pass `--every`; the minimum is `1m`. Colour is used only on a
terminal, and not with `NO_COLOR` set.

**Alerts** are opt in. Put a Slack incoming webhook, a Discord webhook, or any other URL (an
ntfy topic, for example, which gets a plain-text POST) in `DENEB_WATCH_NOTIFY`. The
`--schedule` steps show how to keep it in `~/.config/deneb/watch.env` with mode 600. That is
better than `deneb watch --notify <url>`, because command-line arguments show up in `ps` and
in your shell history. An alert goes out when a check breaks, gets worse or recovers. Nothing
is sent when nothing changed, or when a problem only eased (fail to warn), since that needs no
action. The alert text is scrubbed of keys, webhook URLs and `user:pass@` first. Deneb never
prints a webhook URL, and a failed send reports only the error type and the host. A failed alert is kept and goes out with the
next run, marked "earlier alert not delivered".

State lives in `~/.local/state/deneb` (`DENEB_STATE_DIR` moves it): `watch-state.json`,
`history.jsonl` and a lock file, so two overlapping runs cannot collide. `--schedule` prints
the unit files and the commands to install them, and writes nothing.

---

## Pick a model and set it up

```sh
deneb profile                            # hardware profile and which hardware class it is
deneb recommend --use coding             # rank current models for this box, with est. tokens/s
deneb recommend --use coding --measured  # add measured community speeds from TokenMark
deneb catalog [--refresh]                # the models Deneb ranks, with real per-quant sizes
deneb setup Qwen3.6-35B-A3B              # the setup steps for a model on this box
deneb guide strix-halo                   # pre-flight runbook for the box (or dgx-spark)
deneb stack                              # several models behind one authed endpoint
deneb selfupdate [--remote user@host]    # how to update Deneb, here or on another box
```

Deneb knows 35 hardware classes: GeForce RTX 30, 40 and 50, RTX PRO 6000, RTX 6000 Ada,
RTX A6000, L40S, A100 80GB, H100 SXM, H200, B200, MI300X, DGX Spark, Jetson AGX Thor, Strix
Halo, Radeon RX 7900 XTX and 9070 XT, Intel Arc B580 and Pro B60, and Apple M1 Max, M1 Ultra,
M2 Max, M2 Ultra, M3 Max, M3 Ultra, M4 Pro and M4 Max. It estimates
single-user decode speed from memory bandwidth and the model's active parameters; the test
suite checks the estimate against measured TokenMark speeds.

The model list ships as a hand-picked watchlist. `deneb catalog --refresh` reads each model's
real per-quant file sizes and parameter counts from Hugging Face, with no key.

`deneb setup <model>` reads your box, resolves the model and prints the ordered steps for
your platform (CUDA, ROCm, Metal or CPU): build or get the runtime, download the GGUF, start
`llama-server`. Each step has its exact command, one line on what it does, and a warning
where it needs `sudo`, downloads gigabytes or starts a service. It runs none of them.
When Ollama is already serving, the first step is a route through it instead: an
`ollama pull` of a quant that is one file, or why Ollama can't take the model (it can't
pull a GGUF split into several files yet).

---

## What it writes, and what it sends

- **Writes:** only its own files in your home directory: watch's state and history in
  `~/.local/state/deneb`, the model list `catalog --refresh` saves in `~/.cache/deneb`, and
  in `~/.config/deneb` (mode 600) your saved token, your `deneb brain` choice (the name of an
  API-key variable, never the key) and, while one is pending, `request.json` for a token request.
- **Sends:** nothing, unless you ask. `recommend --measured` reads TokenMark,
  `catalog --refresh` reads Hugging Face, and watch posts an alert to the URL you set.
  `deneb auth --request` sends your name, email, box and reason to Altronis. The conversational
  mode sends your question and the log lines it reads to the model you chose: nothing to
  Altronis if that is your own model, to the Altronis engine if that is the brain.
- **Runs:** read-only probes (`ss`, `systemctl status`, `journalctl`, `nvidia-smi`,
  `rocminfo`, `ollama ps` and the like), deny-by-default, `shell=False`, no sudo. `curl` is a
  plain GET to localhost only (no `-w`, `-o`, `-d`, `-L`); `cat`, `head`, `grep` and similar refuse
  secret files and recursive `grep`; `printenv` needs explicit non-secret names; `python3`, `gcc`,
  `cmake` and `nvcc` take one version flag only. The list is
  in [`deneb/tools.py`](deneb/tools.py) and [`tests/test_deneb_rule.py`](tests/test_deneb_rule.py)
  fails if a free command reaches anything that could change the box.

---

## Conversational troubleshooting (free with your own model)

Deneb also has a conversational mode: describe the problem and it reads the logs, the
systemd unit and the model path, then walks you to the fix.

```sh
deneb "llama-server won't start"
deneb --image error.png "what's this?"   # diagnose a screenshot (needs a vision model)
deneb                                     # interactive
```

**Bring your own model.** If a model server is already running on your box (llama.cpp,
vLLM, Ollama, LM Studio: the one `deneb check` finds), Deneb thinks with it. Nothing is sent
to Altronis, and no token is needed. To choose a model yourself:

```sh
deneb brain                                         # which model it uses, and why
deneb brain local                                   # the server on this box
deneb brain http://host:8000/v1 --model NAME --key-env MY_API_KEY
deneb --brain altronis "why is it slow"             # this run only
```

Which model it uses, first match wins: `--brain`, then `deneb brain`, then a saved Altronis
token, then a server running on this box. `--key-env` saves only the name of the variable;
the key is read from your environment when you run Deneb and is never written to disk. If the
URL is not `localhost`, Deneb tells you where your question goes before it sends it. A model of
3B parameters or fewer gets a warning that answers will be weak. Deneb answers with the
`deneb check` scan of your box and the public setup docs bundled with it (from
[strix-halo-setup](https://github.com/sypherin/strix-halo-setup), MIT) in front of the model.
There is no escalation to a person on your own model. `--image` works only if the model
accepts images.

**No model on the box?** Ask Altronis for a token:

```sh
deneb auth --request      # name, email, box, why. a person at Altronis reviews each request
deneb auth --check        # approved? it saves the token for you
```

Only one request can be pending: asking again says so, and `--replace` sends a new one (the
first can then no longer be claimed). A pending request is also checked once whenever you start
a chat. With the Altronis engine
your question and the log lines Deneb reads are sent to that engine, and there is a daily limit
per token. `deneb auth --token <token>` saves one you were given.

It can apply a fix (edit a unit's `ExecStart`, `mkdir` a missing directory,
`systemctl --user daemon-reload` and `restart`, symlink a binary), backing up any file it
edits to `.deneb-bak` first. By default it shows each change and asks first. `--auto` applies fix commands
without asking, but file writes always ask.
It also refuses to write a systemd unit whose `Exec` line runs a shell, `env`, an interpreter or a
relative path. Either way it never runs `rm`, `mv`, `dd`, `chmod`, `sudo`, a
system-wide `systemctl`, a firewall change or a package install: when the fix needs one of
those, it prints the command for you to run. These limits are enforced in this client, whichever
model is thinking, so a tampered or mistaken model cannot lift them.

`deneb logout` removes your saved settings, your token and any pending token request.

---

## Uninstall

If you set up the timer, take it down first. Step 1 prints the commands that undo it
(`deneb watch --unschedule` prints only, it changes nothing); read them and run them yourself:

```sh
deneb watch --unschedule
```

Step 2, only once the timer is gone, remove the program:

```sh
pipx uninstall deneb-cli
```

Then remove `~/.local/state/deneb`, `~/.cache/deneb` and `~/.config/deneb` if you want them
gone too.

---

Built by [Altronis](https://altronis.sg/deneb/cli), Singapore. Bugs and requests:
[issues](https://github.com/sypherin/deneb-cli/issues).
