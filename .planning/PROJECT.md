# Deneb — v1 (advisory-first, hardware-general, free tier)

## What This Is

Deneb is a terminal tool that helps anyone stand up a private local LLM on their own box.
v1 is the FREE, hardware-general, keyless, deterministic advisory tier: it reads your actual
machine, tells you what you can run, and walks you through the setup — telling you each
command, never executing it. (Execution + the Altronis-stack-specific hardening is the paid
client tier, deferred to later phases.)

## Core Value

"Will THIS model run on MY box, and how do I set it up?" — answered by reading the real
machine, deterministically, on any common hardware, before any LLM exists. This is also the
answer to "why not just ask ChatGPT" (grounded in your box + deterministic, not a general model).

## Requirements

### Active (v1)

- [ ] Hardware-general profiling: detect GPU(s), VRAM/unified memory, driver/kernel, OS/arch,
      CPU, RAM, disk — across NVIDIA/CUDA, AMD/ROCm, Apple/Metal, and CPU-only. Graceful when a
      vendor tool is absent.
- [ ] Model↔hardware compatibility matrix + deterministic fit math (does model X at quant Y fit
      + run on this profile; expected tier of tok/s; known caveats).
- [ ] `deneb recommend [--use coding|vision|chat|general]`: hardware profile → ranked model+quant
      recommendations with a plain "why".
- [ ] Tell-only guided setup: for a chosen recommendation, output the setup steps as ADVICE
      (commands shown + explained + risk-flagged), NEVER executed (the Deneb Rule).
- [ ] Keyless / free; deterministic-first (no LLM in the core path).
- [ ] Pure logic unit-tested (fit math, recommendation ranking, profile parsing).

### Out of Scope (v1 — later phases)

- Command EXECUTION (downloads, systemd jobs, exposure, updates) — the paid executor; v1 is
  tell-only per the Deneb Rule.
- LLM-assisted novel troubleshooting — v1 core is deterministic; LLM fallback is a later add.
- The Altronis reference-stack-specific hardening + client key tier.
- Online-verify + self-updating knowledge (Phase 6 of the master plan) — v1 ships a curated
  static matrix; online-verify comes later.

## Context

- Existing repo `~/deneb-cli`, package `deneb/`: `check.py` (deterministic port/PID/unit probes —
  the pattern to extend), `cli.py` (subcommands), `client.py` (engine), `tools.py` (allowlisted
  run/read), `config.py`, `loop.py` (LLM troubleshoot — NOT used by v1's deterministic core), `ui.py`.
- Master design + resolved decisions: `~/deneb-cli/LEVELUP-PLAN.md`.
- This box: python3.14, numpy + pytest present; rocm-smi + lspci present (AMD Strix Halo);
  nvidia-smi + system_profiler absent — so detection must handle missing vendor tools.

## Constraints

- **THE DENEB RULE (absolute)**: always TELL the command first; v1 NEVER executes (tell-only).
  Always surface risks/warnings on any command shown.
- **Deterministic-first**: v1 core needs NO LLM. Works air-gapped, before any local LLM exists.
- **Hardware-general**: must not assume DGX/Strix — degrade gracefully on any box.
- **Tech**: extend the `deneb/` package + `tools` allowlist; pure logic in dependency-light
  modules; pytest on pure logic. No new heavy deps (numpy already present; prefer stdlib).
- **Quality**: unit tests on fit math + ranking + parsing; verify `deneb recommend` runs end-to-end
  on this real box; Zach's machine = final gate.

## Key Decisions

- v1 = free hardware-general advisory tier (Zach confirmed 2026-07-19).
- Deterministic-first engine, LLM as fallback only (resolves the bootstrap paradox).
- Curated static compat matrix in v1; online-verify/self-learning deferred.
- Tell-only (no execution) in v1 — execution is the paid tier, later.
