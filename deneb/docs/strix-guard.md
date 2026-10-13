<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/strix-guard.md, copied 2026-10-13 at commit 2cba0ba. -->
# strix-guard — remote kill switch + overload guard

A single-box AI rig can wedge itself. A multi-GB model write or a training job
fills page cache and GTT, every writer blocks in `D` state, and the desktop
stops responding. On many kernels this produces **no OOM kill and no panic**, so
nothing recovers you and nothing even logs it. If your only way in is SSH to
that same box, you are stuck driving to it to hold the power button.

`strix-guard` is a ~600-line stdlib-only Python daemon that gives you two
things:

1. **A kill switch** — a private-network-only web page with `REBOOT` / `HARD REBOOT` /
   `KILL BIGGEST` buttons, usable from a phone.
2. **An overload guard** — it watches memory and writeback pressure and kills
   the largest *unprotected* process **before** the box thrashes into a freeze.

## Why the layers matter

Recovery paths, weakest assumption last:

| Layer | Works when | Mechanism |
|---|---|---|
| Graceful reboot | systemd still responds | `systemctl reboot` |
| **Hard reboot** | kernel alive, userspace wedged | SysRq `s`,`u`,`b` |
| Auto-recovery | CPU stuck in kernel | `softlockup_panic` + `panic=10` |
| Auto-recovery | PID 1 stops running | hardware watchdog (`RuntimeWatchdogSec`) |
| **Nothing on the box** | kernel gone, power stuck | needs a smart plug — see below |

**Check your SysRq value first.** If `sysctl -n kernel.sysrq` is not `1` (or at
least has bit 128 set), `echo b > /proc/sysrq-trigger` is a **silent no-op** and
you have no emergency reboot path at all. A value of `16` (sync only) is a
common default and is not enough.

```bash
sysctl -n kernel.sysrq          # 1 = all functions enabled
```

**Check whether your kernel can even detect a hung task:**

```bash
ls /proc/sys/kernel/ | grep hung
```

If that prints nothing, `CONFIG_DETECT_HUNG_TASK` is off in your kernel. A
D-state pile-up will then be **invisible** — no `task blocked for more than N
seconds` messages, no panic, nothing in the journal. That is exactly why some
freezes leave a completely silent log. The overload guard exists to cover this
gap by acting *before* the stall.

## The metric most guards get wrong

Watching `MemAvailable` alone will not catch a writeback stall. Dirty page cache
still counts as "available", so the box can be fully stalled flushing a
multi-GB file while `MemAvailable` looks perfectly healthy. `strix-guard` also
watches `Dirty` + `Writeback` from `/proc/meminfo` and alerts separately
(`DIRTY_WARN_MIB`).

If you write large model files on a big-RAM box, also check your dirty limits:

```bash
sysctl vm.dirty_ratio vm.dirty_background_ratio
```

The `tuned` profile `throughput-performance` sets `dirty_ratio=40`. On a 128 GB
box that permits **~50 GB of dirty pages** before the kernel hard-throttles
writers — at which point every process that touches the filesystem blocks. For a
desktop that also writes multi-GB GGUFs, byte-based limits are far safer:

**Do not just `sysctl -w` this.** If `tuned` is running it re-applies its profile
on start and on every profile switch, and silently puts the ratio back. Set it
through a child profile instead:

```bash
# tuned >= 2.24 reads user profiles from /etc/tuned/PROFILES/<name>/,
# not /etc/tuned/<name>/. The old path is ignored with a confusing
# "Requested profile doesn't exist".
sudo mkdir -p /etc/tuned/profiles/strix-ai
sudo tee /etc/tuned/profiles/strix-ai/tuned.conf >/dev/null <<'EOF'
[main]
summary=throughput-performance with byte-based dirty limits
include=throughput-performance

[vm]
dirty_background_bytes=4294967296   # 4 GiB
dirty_bytes=8589934592              # 8 GiB
EOF
sudo tuned-adm profile strix-ai
```

Use `dirty_bytes`, not `dirty_ratio`. tuned's vm plugin routes an **absolute**
value to `vm.dirty_bytes` and zeroes the ratio; a value ending in `%` does the
reverse. `dirty_ratio` is explicitly deprecated in the plugin because it does not
support profile inheritance.

**If `tuned-ppd` is active, that is not enough.** It maps power-profiles-daemon
names onto tuned profiles, so the GNOME power panel — or just a `tuned-ppd`
restart — reapplies `performance` and reverts you. Repoint it:

```bash
# /etc/tuned/ppd.conf, [profiles] section
-performance=throughput-performance
+performance=strix-ai
```

Verify it actually stuck, including across a restart of both services:

```bash
sudo systemctl restart tuned tuned-ppd
sysctl vm.dirty_bytes vm.dirty_ratio   # want 8589934592 and 0
```

## Install

```bash
sudo bash bin/strix-guard-install.sh
sudo nano /etc/strix-guard/guard.conf     # set GUARD_USER, MANAGED_UNITS, PROTECT_EXTRA
sudo systemctl restart strix-guard
sudo cat /etc/strix-guard/token           # bookmark http://<tailscale-ip>:7799/?t=<token>
```

## Configuration

Everything host-specific lives in `/etc/strix-guard/guard.conf`; see
[`configs/strix-guard.conf.example`](../configs/strix-guard.conf.example). The
values worth setting:

- `PROTECT_EXTRA` — cmdline substrings of the agents/bots that must survive an
  overload. The OS itself (systemd, dbus, journald, sshd, tailscaled, ...) is
  protected unconditionally.
- `MANAGED_UNITS` — explicit allowlist of units the web UI may restart. Nothing
  outside this list can be touched through the web UI.
- `GUARD_USER` — whose `systemctl --user` units those are.
- `ACT_MIB` / `WARN_MIB` — memory thresholds.

## Security model

- Binds **only** to the machine's Tailscale IPv4. It refuses to start without
  one, so it is never exposed to the LAN or the internet.
- Every destructive action requires `POST` + a bearer token
  (`/etc/strix-guard/token`, mode 0600, compared with `hmac.compare_digest`).
- `/healthz` is the only unauthenticated route, for uptime checks.
- Restart targets are an allowlist, not free-form.
- The UI requires a double-tap to confirm before firing anything.

It runs as root because it needs SysRq, `SIGKILL` on arbitrary pids, and
`systemctl reboot`. The unit therefore hardens everything it does *not* need
(`NoNewPrivileges`, `ProtectHome=read-only`, `RestrictAddressFamilies`, ...)
rather than dropping privileges.

## What this does not solve

If the kernel is gone entirely — no network, no SysRq — **no software on the box
can help you**. The only fix is out-of-band power: a smart plug plus a BIOS
setting of *Restore on AC Power Loss = Power On*, so you can power-cycle from
your phone. Budget roughly USD 15–25 for the plug. `strix-guard` covers every
layer above that one.
