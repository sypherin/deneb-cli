---
phase: 03-tell-only-setup-cli-qa
plan: 01
subsystem: advisory
tags: [setup, playbook, llama.cpp, gguf, rocm, cuda, metal, tell-only, deneb-rule, pytest]

# Dependency graph
requires:
  - phase: 01-hardware-profiling
    provides: HardwareProfile / GPUInfo shape (primary_backend, usable_mem_mb, gpus[].extra.gfx)
  - phase: 02-catalog-fit-recommend
    provides: CATALOG + Model/Quant structs; pure fits() predicate for highest-bpw-that-fits
provides:
  - "deneb/setup_advisor.py — pure Step model + curated platform-branched PLAYBOOK (cuda/rocm/metal/cpu)"
  - "setup_steps(model, profile, quant=None) — ordered runtime -> download -> run advice, SET-03 warned inline"
  - "resolve_model(name) — case/separator-insensitive catalog lookup (None on miss, never raises)"
  - "tests/test_setup_advisor.py — 16 pure step-gen tests (QA-01 for the setup path)"
affects: [03-02-setup-cli, deneb-setup-command, paid-executor-tier]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Curated DATA playbook (dataclasses) branched by backend; command templates, not per-model inlining"
    - "Pure struct-in/Step-out advisory; the Deneb Rule enforced at the source (imports no executor)"
    - "Guarded body -> single honest NOTE on any failure; never raises (PLAT-03)"

key-files:
  created:
    - deneb/setup_advisor.py
    - tests/test_setup_advisor.py
  modified: []

key-decisions:
  - "PLAYBOOK holds runtime step templates + run flags + the gfx caveat as DATA; setup_steps fills placeholders and stamps SET-03 warnings"
  - "gfx1151/Strix kernel caveat applied to the run step ONLY when a gfx1151/gfx1150 GPU is actually present (not blanket-rocm)"
  - "mmproj projector emitted as a kind='note' (not a 2nd 'download') so the pipeline stays runtime -> one download -> one run"
  - "Run command binds 127.0.0.1:8001 (never 0.0.0.0) — exposure is out of scope for v1 (T-03-12)"
  - "Metal branch = prebuilt via `brew install llama.cpp` (no compile); cuda/rocm/cpu = build-from-source"

patterns-established:
  - "SET-03 warning text lives in named module constants; helpers stamp them from RuntimeStep flags"
  - "House-style _plain() dash-scrub reused from recommend.py; zero em-dashes in generated copy"

requirements-completed: [SET-01, SET-02, SET-03, QA-01]

# Metrics
duration: ~15min
completed: 2026-07-19
---

# Phase 3 Plan 01: Tell-Only Setup Advisory Core Summary

**Pure, platform-branched `setup_steps()` that turns a catalog model + hardware profile into ordered runtime -> download -> run advice — every sudo/GB/service step warned inline, executing nothing (the Deneb Rule at the source).**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-19T22:05:00Z (approx)
- **Completed:** 2026-07-19T22:11:00Z (approx)
- **Tasks:** 2
- **Files modified:** 2 (both created)

## Accomplishments
- `deneb/setup_advisor.py`: a PURE module (no tools / no process-spawn / no os-exec / no LLM) holding the `Step` advice unit, a curated `PLAYBOOK` branched by accelerator (cuda/rocm/metal/cpu), and `resolve_model()`.
- `setup_steps(model, profile, quant=None)`: ordered runtime -> download -> run pipeline, platform-branched, with the advised quant = highest-bpw quant that `fits()` this box (lowest + under-spec warning when nothing fits) — consistent with `recommend`.
- SET-03 enforced inline: sudo install steps carry a sudo warning, the download step names the size in `~N GB`, the run step flags the `127.0.0.1:8001` service, and rocm boxes with a gfx1151/gfx1150 GPU also carry the Strix kernel caveat.
- Honest degradation (PLAT-03): unknown/empty backend -> generic CPU-safe steps + a NOTE; malformed input -> a single honest NOTE; never raises, never empty.
- Vision models surface the mmproj projector (extra note + `--mmproj` on the run command).
- 16 new pure unit tests (QA-01 for the setup path); full suite 83 -> 99 green.

## Task Commits

Each task was committed atomically (Task 2 followed TDD: RED then GREEN):

1. **Task 1: Step + curated platform-branched PLAYBOOK + resolve_model (SET-01, SET-02)** - `e21d889` (feat)
2. **Task 2 (RED): failing setup_steps() step-gen + SET-03 warning tests** - `12f650d` (test)
3. **Task 2 (GREEN): pure setup_steps() step-gen + inline SET-03 warnings** - `198123f` (feat)

No REFACTOR commit needed — helpers were small and named on first pass.

## Files Created/Modified
- `deneb/setup_advisor.py` - Step dataclass, RuntimeStep/Playbook DATA model, curated PLAYBOOK (cuda/rocm/metal/cpu), resolve_model(), and the pure guarded setup_steps() with SET-03 warning stamping.
- `tests/test_setup_advisor.py` - 16 pure tests over constructed HardwareProfiles: ordered pipeline, per-backend branching, sudo/GB/service warnings, gfx1151 caveat (positive + negative), best-quant, explicit-quant override, under-spec, vision mmproj, degraded platform, never-raises-on-junk, no-em-dash, resolve_model matching.

## Decisions Made
- **PLAYBOOK is DATA, code fills it:** runtime step templates + run flags + the gfx caveat live in the playbook; `setup_steps` fills `{repo_basename}`/`{quant}`/flags and stamps warnings. Keeps SET-02 maintainable.
- **gfx1151 caveat is conditional, not blanket-rocm:** applied to the run step only when a gfx1151/gfx1150 GPU is present, matching fit.py's rule and avoiding a false caveat on a discrete-AMD rocm box.
- **mmproj as a note, not a second download:** preserves the "exactly one download, exactly one run" pipeline invariant the tests pin.
- **127.0.0.1 bind only:** the run command never emits 0.0.0.0 (T-03-12); network exposure is deferred to the paid executor tier.

## Deviations from Plan

None - plan executed exactly as written. (One cosmetic adjustment during Task 1 verify: reworded the module docstring so the word "subprocess" no longer appears in prose, because the plan's purity grep `-E "subprocess|..."` matched the docstring text — this is a false-positive fix, the module never contained an executor call. Not a behavior change.)

## Issues Encountered
- The plan's purity-grep verify (`grep -nE "subprocess|os\.system|..."`) initially flagged the docstring line that described the module as importing "no subprocess". Reworded the prose to "no process-spawning"; the grep now returns clean. No code impact.

## Threat Surface Notes
- T-03-11 (executor could run a command): mitigated — module imports no tools/process-spawn/os-exec; the purity grep is clean; the formal behavioral no-exec assertion lands in Plan 03-02 (QA-03).
- T-03-12 (0.0.0.0 exposure): mitigated — run command binds 127.0.0.1 only; asserted by `test_run_step_flags_service_and_binds_localhost`.
- T-03-13/14 (garbage input / silent danger): mitigated — resolve_model returns None; setup_steps is guarded; every sudo/download/service step carries its inline warning (warning-coverage tests are the gate).
- No new threat surface introduced beyond the plan's register.

## Known Stubs
None - `setup_steps` returns fully-formed, real, curated command strings for every catalog model on every backend (verified across 420 generated steps: catalog x 6 constructed boxes, zero em-dashes, no crashes). The command strings are ADVICE by design (the Deneb Rule), not stubs.

## Next Phase Readiness
- Ready for Plan 03-02: wire `setup_steps()` + `resolve_model()` into `deneb setup <model>` (display/table), turn a None resolve into a helpful error + `deneb recommend` pointer, and add the formal no-execution QA gate (QA-03).
- The pure core is fully unit-tested and deterministic; the CLI plan can construct profiles from `profile_hardware()` and print steps without touching this module's logic.

---
*Phase: 03-tell-only-setup-cli-qa*
*Completed: 2026-07-19*

## Self-Check: PASSED
- FOUND: deneb/setup_advisor.py
- FOUND: tests/test_setup_advisor.py
- FOUND: .planning/phases/03-tell-only-setup-cli-qa/03-01-SUMMARY.md
- FOUND commits: e21d889 (Task 1), 12f650d (Task 2 RED), 198123f (Task 2 GREEN)
