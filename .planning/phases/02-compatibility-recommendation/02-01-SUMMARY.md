---
phase: 02-compatibility-recommendation
plan: 01
subsystem: compatibility
tags: [catalog, gguf, fit-math, moe, speed-tier, caveats, dataclasses, tdd, pure-logic]

# Dependency graph
requires:
  - phase: 01-hardware-profiling
    provides: HardwareProfile (usable_mem_mb already OS/unified-headroomed), GPUInfo (memory_kind, extra.gfx), primary_backend
provides:
  - "deneb/models_catalog.py: Model/Quant schema + 17-model curated CATALOG + by_capability() (FIT-01)"
  - "deneb/fit.py: pure fits(model, quant, profile) -> FitResult with fit-math, speed-tier heuristic, and FIT-02b caveat rules"
  - "FitResult contract {fits, headroom_mb, expected_speed_tier, caveats, need_mb} that Plan 02 recommend() ranks over"
affects: [02-02-recommendation, recommend, setup, cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure str/struct-in, struct-out layer (no I/O, no tools, no LLM) — every fact unit-tested against constructed profiles (QA-01)"
    - "Curated static data module with explicit honesty docstring (approximations, not HF-verified) instead of pretending to authoritative facts"
    - "Interface-first ordering: catalog structs (Task 1) fixed before the logic that consumes them (Task 2)"

key-files:
  created:
    - deneb/models_catalog.py
    - deneb/fit.py
    - tests/test_catalog.py
    - tests/test_fit.py
  modified: []

key-decisions:
  - "size_mb + repo-ids marked in the module docstring as curated best-effort approximations (NOT HF-verified); online-verify deferred to a later phase — honest degradation over false authority"
  - "Speed tier keys off EFFECTIVE params (active for MoE, total for dense) so 35B-A3B is 'fast'; cuda/rocm share a tier at this granularity, ROCm risk surfaced via caveats not the tier"
  - "RUNTIME_RESERVE_MB (max 512 / 10%) is added ON TOP of usable_mem_mb (which already has OS headroom from Phase 1) — no double-counting of margin"
  - "_quants() helper builds bpw-ordered Quant lists from a {name:size_mb} map — keeps each Model literal flat while avoiding repeating the bpw constants (maintainability gate: no twice-written logic)"

patterns-established:
  - "Deterministic caveat ordering (gfx1151 -> unified -> tight -> cpu-only -> unreadable budget) so tests can pin them"
  - "Whole fit() body guarded — None/empty/malformed profile degrades to a valid not-fits FitResult, never raises (T-02-01)"

requirements-completed: [FIT-01, FIT-02, FIT-02b]

# Metrics
duration: 9min
completed: 2026-07-19
---

# Phase 2 Plan 01: Compatibility Core Summary

**Curated 17-model GGUF catalog (params/arch/quants/tags/repo) plus a pure `fits(model, quant, profile)` that computes headroom, an effective-params speed tier (MoE-aware), and the FIT-02b caveat rules — 23 new tests, all four canonical fit cases pinned.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-07-19T13:09Z
- **Completed:** 2026-07-19T13:18Z
- **Tasks:** 2 (Task 2 = TDD RED+GREEN)
- **Files modified:** 4 created

## Accomplishments
- `deneb/models_catalog.py` — Model/Quant dataclasses + a 17-model CATALOG spanning Qwen coder/3.x-MoE, Llama 3.x, Gemma, Mistral/Mixtral, Phi, DeepSeek-Coder, and 2 vision models; each with params, dense/MoE + active params, quants with conservative approx `size_mb`, capability tags, and a GGUF repo. `by_capability()` filter (coding=6, vision=2).
- Honesty baked in: module docstring marks sizes/repo-ids as curated best-effort approximations (NOT HF-verified, online-verify deferred); vision models carry an mmproj-projector note.
- `deneb/fit.py` — pure `fits()` returning `FitResult{fits, headroom_mb, expected_speed_tier, caveats, need_mb}`; need = size + `RUNTIME_RESERVE_MB` (max 512 / 10%) compared vs `usable_mem_mb` (no double-counting of OS headroom); effective-params speed tier with cpu-only 2-step downgrade; deterministic caveat rules.
- All four canonical cases pinned: 7B-Q4 fits 16GB; 70B-Q4 does NOT fit 16GB; 70B-Q4 fits a 128GB unified box; 35B-A3B MoE is "fast". Every caveat rule fires on its trigger; None/empty-gpu profiles degrade without raising.

## Task Commits

1. **Task 1: Curated model catalog data module (FIT-01)** - `63145db` (feat)
2. **Task 2 RED: failing fits() tests** - `0973acb` (test)
3. **Task 2 GREEN: fits() fit-math + speed tier + caveats (FIT-02, FIT-02b)** - `1c9a5c1` (feat)

**Plan metadata:** committed separately (docs).

_TDD gate compliance: RED (`test(...)` 0973acb) precedes GREEN (`feat(...)` 1c9a5c1). No REFACTOR commit needed — catalog uses a DRY `_quants()` helper and fit logic was clean on first pass._

## Files Created/Modified
- `deneb/models_catalog.py` (145 lines) - Model/Quant schema, CATALOG (17 models), by_capability()
- `deneb/fit.py` (211 lines) - FitResult + fits() + RUNTIME_RESERVE_MB / TIER_ORDER / _effective_params / _speed_tier / _caveats
- `tests/test_catalog.py` (125 lines) - 8 catalog integrity tests
- `tests/test_fit.py` (190 lines) - 15 fits() canonical + caveat + graceful-degradation tests

## Decisions Made
- Marked catalog sizes/repos as curated approximations in the docstring rather than presenting them as measured facts — the Deneb Rule (surface the caveat). Online verification deferred to a later phase.
- Speed tier uses effective params (active for MoE) and treats cuda/rocm/metal as one accelerated band; ROCm-specific risk goes to caveats, not the tier label.
- `RUNTIME_RESERVE_MB` layered on top of the Phase-1 `usable_mem_mb` (which already subtracts OS/unified headroom) — deliberately no second margin.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. RED confirmed genuinely (ModuleNotFoundError before implementation); GREEN reached first try; full suite green.

## Known Stubs
None. Both modules are complete pure logic. (Catalog `size_mb`/`gguf_repo` values are intentionally curated approximations, documented as such and slated for online-verify in a later phase — this is an explicit, disclosed choice, not a stub blocking the plan goal.)

## Threat Flags
None. No new network endpoints, auth paths, or trust-boundary surface — both modules are pure, no-I/O. T-02-01 (degraded-profile DoS) and T-02-02 (false-fit OOM) are mitigated as planned: conservative sizes + RUNTIME_RESERVE_MB + tight caveat + guarded body + monotonicity test.

## Next Phase Readiness
- Plan 02 (recommend()) has a fixed contract: import `Model`/`Quant`/`CATALOG`/`by_capability` from `deneb.models_catalog` and `fits`/`FitResult` from `deneb.fit`. `FitResult.need_mb` and `headroom_mb` are available for ranking and why-strings.
- No blockers. Full test suite: 72 passed (49 Phase-1 + 8 catalog + 15 fit).

## Self-Check: PASSED

All 4 created source/test files exist on disk; all 3 task commits (63145db, 0973acb, 1c9a5c1) present in git history. Full suite `python3 -m pytest tests/ -q` = 72 passed.

---
*Phase: 02-compatibility-recommendation*
*Completed: 2026-07-19*
