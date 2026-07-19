---
phase: 03-tell-only-setup-cli-qa
plan: 02
subsystem: cli
tags: [setup, cli, keyless, tell-only, deneb-rule, no-execution, rocm, pytest, v1-complete]

# Dependency graph
requires:
  - phase: 01-hardware-profiling
    provides: profile_hardware() live read-only HardwareProfile (primary_backend, gpus[].extra.gfx)
  - phase: 03-01
    provides: setup_advisor.setup_steps() + resolve_model() + CATALOG (pure, tell-only core)
provides:
  - "deneb setup <model> — keyless, deterministic, engine-free CLI that prints the ordered platform-branched setup steps (runtime -> download -> run) with inline warnings, executing nothing"
  - "cmd_setup dispatch + _HELP + README updated to the four v1 keyless commands (check/profile/recommend/setup)"
  - "tests/test_deneb_rule.py — the FORMAL Deneb-Rule no-execution assertion (source purity + behavioral fail-loud spy over setup/recommend/profile) (QA-03)"
affects: [v1-complete, paid-executor-tier, funnel-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tell-only CLI command: mirrors cmd_recommend (keyless, local, engine-free — imports only hardware + setup_advisor); reads box, resolves model, prints Step structs, returns 0"
    - "Deneb-Rule proof = source purity (grep the module + inspect.getsource the fns) + a behavioral fail-loud spy that monkeypatches the write/exec surface and drives every tell-only path"
    - "Positive-control test (tripwire actually fires) so a green no-execution result is meaningful, not a silent no-op"

key-files:
  created:
    - tests/test_deneb_rule.py
  modified:
    - deneb/cli.py
    - README.md

key-decisions:
  - "Model name = the joined non-flag positional args after 'setup' (so 'deneb setup Qwen3.6-35B-A3B' and quoted/spaced names both work); empty -> usage + exit 2"
  - "Unknown model -> difflib closest-name hint + a 'deneb recommend --use coding' pointer + exit 2 (PLAT-03, never crashes)"
  - "cmd_setup accesses the catalog via setup_advisor.CATALOG (no extra import) to keep the fn's import surface = hardware + setup_advisor only"
  - "QA-03 uses manual patch + contextlib.redirect_stdout (no pytest fixtures) so the file runs both via `python3 tests/test_deneb_rule.py` and under pytest, matching repo style"
  - "os.system (not platform.system) is the patched exec surface — platform.system is a pure name query profile_hardware needs"

patterns-established:
  - "The tell-only-path forbidden-token set: cmd bodies contain none of tools./subprocess/os.system/os.popen/client./loop.run/run_write/execute"
  - "House no-em-dash rule extends to CLI chrome copy, not just generated advice"

requirements-completed: [SET-01, PLAT-01, PLAT-02, PLAT-03, QA-02, QA-03, QA-04]

# Metrics
duration: ~25min
completed: 2026-07-19
---

# Phase 3 Plan 02: Tell-Only Setup CLI + Deneb-Rule QA Gate Summary

**`deneb setup <model>` wires the pure advisory into a keyless, deterministic, engine-free CLI that reads this box, resolves the model, and prints the ordered platform-branched setup steps with inline warnings — executing NOTHING — and `tests/test_deneb_rule.py` formally PROVES no v1 tell-only path (setup/recommend/profile) runs anything. This completes Deneb v1.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-07-19
- **Tasks:** 2
- **Files:** 3 (1 created, 2 modified)
- **Tests:** 99 -> 106 green (+7)

## Accomplishments
- **`deneb setup <model>` (SET-01 display, PLAT-01/02/03):** keyless, deterministic, engine-free `cmd_setup` in `deneb/cli.py` — imports only `hardware` + `setup_advisor` (no client/loop/tools executor). Reads the live box via `profile_hardware()` (read-only, allowed), resolves the model, calls `setup_steps()`, and prints a header + each numbered step (title, exact `$ command`, `what:` line, and each `⚠` warning on its own line) + a "Deneb ran none of them" footer. Returns 0. Runs nothing.
- **Honest degradation (PLAT-03):** empty name -> usage + `deneb recommend` pointer + exit 2; unknown model -> a `difflib` closest-name hint + the `deneb recommend --use coding` pointer + exit 2 (never crashes); a degraded platform still prints the generic CPU-safe steps + caveat (inherited from `setup_steps`).
- **QA-03 — the FORMAL Deneb-Rule assertion (`tests/test_deneb_rule.py`, 7 tests):** two-pronged proof across ALL three tell-only paths:
  - **Source purity:** `setup_advisor.py` contains none of subprocess/os.system/os.popen/Popen/exec(/eval(/tools.run/execute/write/run_write and imports no `tools`; `cmd_setup`/`cmd_recommend`/`cmd_profile` bodies reference no executor token.
  - **Behavioral spy (strong proof):** fail-loud-patches `tools.run_write/execute/execute_write/write_file` + `os.system`, then drives `setup_steps` + `cmd_setup` + `cmd_recommend` + `cmd_profile` — NONE fire, while `cmd_setup` still prints the `llama-server` run command + the `127.0.0.1:8001` service warning as TEXT. A positive-control test proves the tripwire really fires when a write primitive is called (so the green result is meaningful). Read-only `tools.run` is left intact so `profile_hardware` still reads the box.
- **Help + README:** `deneb --help` and README now list the four v1 keyless commands (check / profile / recommend / setup, tell-only); the `deneb recommend` next-pointer text is updated (setup exists now); paid CTA/teaser kept out of scope.
- **QA-02 live e2e on this Strix box:** `deneb setup Qwen3.6-35B-A3B` (recommend's top pick) prints ordered ROCm-branched steps and executes nothing; `deneb recommend --use coding` still works.

## Task Commits

1. **Task 1: `deneb setup <model>` CLI + help/README + live e2e (SET-01, PLAT-01/02/03, QA-02)** — `0c9c886` (feat)
2. **Task 2: QA-03 Deneb-Rule no-execution assertion (source purity + behavioral spy)** — `5839435` (test)
3. **Style: drop em-dash from cmd_setup header (house no-em-dash rule)** — `ab67775` (style)

## Files Created/Modified
- `deneb/cli.py` — added `cmd_setup(argv)` (keyless, print-only), `main()` dispatch for `setup`, `_HELP` line, and updated the `cmd_recommend` next-pointer text.
- `tests/test_deneb_rule.py` — the QA-03 gate: source-purity checks + the behavioral fail-loud spy + a positive control, runnable standalone and under pytest.
- `README.md` — the "what should I run + how do I set it up" section listing the four keyless v1 commands; setup framed as tell-only.

## Deviations from Plan

None — plan executed exactly as written. One in-scope quality touch: removed an em-dash from the new `cmd_setup` header copy to honor the repo's no-em-dash house style (the same rule `setup_advisor._plain` and `test_no_emdash_in_generated_copy` enforce on generated advice). Committed separately as `ab67775` (style). No behavior change.

## v1 Requirements — Final Status

All v1 requirements are now complete:

| Category | Requirements | Status |
|----------|--------------|--------|
| Hardware Profiling | HW-01..04 | done (P1) |
| Compatibility + Fit | FIT-01, FIT-02, FIT-02b | done (P2) |
| Recommendation | REC-01, REC-02, REC-03 | done (P2) |
| Tell-Only Setup | SET-01, SET-02, SET-03 | done (P3) |
| Platform / CLI | PLAT-01, PLAT-02, PLAT-03 | done (P3-02) |
| Quality | QA-01, QA-02, QA-03, QA-04 | done (P3-02) |

## Threat Surface Notes
- **T-03-21 (a tell-only path executes a setup command — the Deneb-Rule violation):** mitigated. `cmd_setup` is print-only, keyless, engine-free; QA-03 asserts source purity of `setup_advisor` + the three CLI fns AND runs a behavioral fail-loud spy proving `setup_steps`/`cmd_setup`/`cmd_recommend`/`cmd_profile` fire no write/exec primitive. Positive control confirms the spy is real.
- **T-03-22 (garbage `setup <model>` arg):** mitigated. `resolve_model` -> None -> helpful hint + `deneb recommend` pointer + exit 2; verified live (`deneb setup not-a-real-model` -> exit 2).
- **T-03-23 (0.0.0.0 exposure):** mitigated. The run step binds `127.0.0.1` only (from 03-01); `cmd_setup` prints it verbatim and adds no exposure step.
- **T-03-24 (degraded/None-budget profile):** mitigated. `setup_steps` is guarded (03-01); `cmd_setup` handles the profile without raising.
- No new threat surface introduced beyond the plan's register.

## Known Stubs
None — `cmd_setup` prints real, fully-formed curated command strings from the catalog + playbook for every model on every backend. The commands are ADVICE by design (the Deneb Rule), not stubs; the user runs them.

## Live Demo (QA-02 / QA-04 proxy — this Strix Halo box)

`deneb setup Qwen3.6-35B-A3B` (recommend's top coding pick), verified live, executing nothing:

```
◇ Deneb  Altronis · private-LLM setup for your AI box
setup steps for Qwen3.6-35B-A3B on this box (backend rocm, no engine)…

  Deneb shows every command and runs NOTHING (the Deneb Rule). You run each yourself.

  1. Install build prerequisites (ROCm)
     $ sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev rocm-hip-sdk
     what: installs the compiler, CMake and the ROCm/HIP SDK build prerequisites (Debian/Ubuntu; adapt for your distro).
     ⚠ needs sudo - installs system packages / drivers with root privilege.

  2. Build llama.cpp (ROCm/HIP)
     $ git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && HIPCXX=$(hipconfig -l)/clang cmake -B build -DGGML_HIP=ON && cmake --build build --config Release -j
     what: clones and compiles llama.cpp with ROCm/HIP acceleration for your AMD GPU
     ⚠ compiles llama.cpp from source - can take several minutes and needs the build prerequisites above.

  3. Download the Q8_0 GGUF
     $ hf download Qwen/Qwen3-30B-A3B-GGUF --include "*Q8_0*.gguf" --local-dir ~/models/Qwen3-30B-A3B-GGUF
     what: downloads the Q8_0 GGUF weights
     ⚠ downloads ~37 GB to ~/models/Qwen3-30B-A3B-GGUF - check you have the disk space.

  4. Run the model (llama-server)
     $ llama-server -m ~/models/Qwen3-30B-A3B-GGUF/*Q8_0*.gguf -ngl 999 --host 127.0.0.1 --port 8001
     what: starts the model server locally
     ⚠ starts a local server on 127.0.0.1:8001 - it stays running until you stop it (Ctrl-C).
     ⚠ gfx1151 (Strix Halo): needs a recent kernel with the ROCm fixes - older kernels have known gfx1151 instability that can hang the box.

  next: run these yourself, top to bottom. Deneb ran none of them (v1 is tell-only); verify each before you run it.
```

`deneb setup not-a-real-model` -> `✗ unknown model 'not-a-real-model'. … deneb recommend --use coding` -> `exit=2`.
`deneb recommend --use coding` -> top row `1  Qwen3.6-35B-A3B  Q8_0  ✓ ~65 GB free  fast` (still works).

**Final gate:** `python3 -m pytest -q` -> **106 passed**.

---
*Phase: 03-tell-only-setup-cli-qa — v1 COMPLETE*
*Completed: 2026-07-19*

## Self-Check: PASSED
- FOUND: tests/test_deneb_rule.py
- FOUND: deneb/cli.py (modified)
- FOUND: README.md (modified)
- FOUND: .planning/phases/03-tell-only-setup-cli-qa/03-02-SUMMARY.md
- FOUND commits: 0c9c886 (Task 1), 5839435 (Task 2), ab67775 (style)
