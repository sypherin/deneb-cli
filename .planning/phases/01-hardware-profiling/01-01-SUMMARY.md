---
phase: 01-hardware-profiling
plan: 01
subsystem: hardware-profiling
tags: [dataclasses, parsers, rocm-smi, nvidia-smi, system_profiler, unified-memory, pytest, tdd]

# Dependency graph
requires: []
provides:
  - "HardwareProfile / GPUInfo / MemoryClassification data model (deneb/hardware.py)"
  - "Seven pure str->struct parsers: parse_nvidia_smi, parse_rocm_smi (json+text), parse_system_profiler, parse_sysctl_memsize, parse_cpuinfo, parse_meminfo, parse_lspci_gpus"
  - "classify_memory: unified-vs-dedicated + usable-budget logic (HW-03 trap handled)"
  - "tests/fixtures/hardware/ corpus of 10 captured real + canonical tool outputs"
affects: [01-02-hardware-live-probe, compatibility-fit, recommendation]

# Tech tracking
tech-stack:
  added: []  # stdlib only (dataclasses, json, re) — no new deps per PLAT-02
  patterns:
    - "INVOCATION-vs-PARSING split: pure parsers here, live _probe_* layer in Plan 02 (mirrors check.py)"
    - "Defensive parser contract: every parser returns []/0/{} on empty/refused/junk, never raises"
    - "Parser-routes-carveout, classifier-decides-kind: rocm parser tags carve-out vs vram; classify_memory makes the final unified/dedicated call"

key-files:
  created:
    - deneb/hardware.py
    - tests/test_hardware.py
    - tests/fixtures/hardware/ (10 fixtures)
  modified: []

key-decisions:
  - "parse_rocm_smi routes the reported VRAM to vram_carveout_mb on an APU (gfx in unified set OR STRX SKU) and to vram_mb on a discrete card; classify_memory owns the final unified/dedicated verdict"
  - "Unified usable budget = system RAM minus max(8192 MB floor, 15%) headroom, citing the <80GB-free Strix freeze scar; labelled coarse/Phase-2-refined"
  - "Fixtures written with exact captured bytes (tabs preserved) since the parser tests assert against them"

patterns-established:
  - "Pure I/O-free hardware parsers unit-tested against captured fixtures (HW-04/QA-01)"
  - "TDD RED (failing tests) then GREEN (implementation) as separate atomic commits for tdd=true tasks"

requirements-completed: [HW-01, HW-03, HW-04]

# Metrics
duration: 7min
completed: 2026-07-19
---

# Phase 1 Plan 01: Hardware Profiling Pure Core Summary

**Pure, defensive HardwareProfile data model + seven vendor/CPU/RAM parsers + unified-memory classification that budgets the Strix Halo's real ~124 GB shared pool (not the 1 GB rocm-smi VRAM carve-out), all unit-tested against 10 captured fixtures with no live tool.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-19T12:37:17Z
- **Completed:** 2026-07-19T12:44:26Z
- **Tasks:** 2
- **Files created:** 12 (deneb/hardware.py, tests/test_hardware.py, 10 fixtures)

## Accomplishments
- HardwareProfile / GPUInfo / MemoryClassification stdlib dataclasses with sane defaults + to_dict() for the Plan 02 printer (no new deps, PLAT-02).
- Seven pure parsers: NVIDIA csv (single + multi GPU), rocm-smi in BOTH --json and text-table forms, Apple system_profiler, sysctl hw.memsize, /proc/cpuinfo, /proc/meminfo, and an lspci PCI-id fallback.
- classify_memory locks the HW-03 trap: Strix gfx1151 (1 GiB carve-out on 124 GiB RAM) -> unified budget from system RAM; discrete gfx1100 stays dedicated with its 24564 MB VRAM as the budget.
- Every parser degrades to []/0/{} on empty / whitespace / refused / garbage input and never raises (T-01-01). 21 new tests, full suite 43 passed.

## Task Commits

1. **Task 1: fixtures + HardwareProfile/GPUInfo data model** - `d263354` (feat)
2. **Task 2 (RED): failing pure-parser + classify_memory tests** - `fc1aa0c` (test)
3. **Task 2 (GREEN): implement parsers + memory classification** - `4a0279a` (feat)

_TDD task 2 committed RED then GREEN as separate atomic commits; no REFACTOR commit needed (implementation was clean)._

## Files Created/Modified
- `deneb/hardware.py` - data model + 7 pure parsers + classify_memory (I/O-free; imports only dataclasses/json/re/typing)
- `tests/test_hardware.py` - 21 tests loading fixtures via pathlib; test_check.py style (stdlib asserts, pytest-collectable, __main__ runner)
- `tests/fixtures/hardware/rocm-smi_strix_gfx1151.json` - the HW-03 trap: 1 GiB carve-out on a 128 GiB unified box
- `tests/fixtures/hardware/rocm-smi_strix_gfx1151.txt` - text-table variant (tabs preserved)
- `tests/fixtures/hardware/rocm-smi_discrete_gfx1100.json` - discrete AMD (proves not-all-AMD-is-unified)
- `tests/fixtures/hardware/nvidia-smi_single.csv`, `nvidia-smi_multi.csv` - NVIDIA csv,noheader
- `tests/fixtures/hardware/system_profiler_m2max.txt`, `sysctl_hw_memsize.txt` - Apple/Metal + unified RAM
- `tests/fixtures/hardware/proc_cpuinfo_amd.txt`, `proc_meminfo.txt`, `lspci_strix.txt` - CPU/RAM/PCI

## Decisions Made
- rocm parser tags carve-out vs real VRAM; classify_memory makes the final unified/dedicated call (single source of the unified rules).
- Unified headroom = max(8192 MB, 15% of RAM); floor justified in-code by the <80GB-free Strix freeze scar; explicitly labelled coarse and Phase-2-refined.
- Fixtures written with exact captured bytes (tabs intact) because the parser tests assert against them.

## Deviations from Plan

None - plan executed exactly as written. No bugs, missing-critical, or blocking issues encountered; the threat model's two mitigations (T-01-01 defensive parsers, T-01-02 unified-memory lock) were implemented as specified and are covered by tests.

## Issues Encountered
None.

## Known Stubs
None. classify_memory's headroom math is intentionally coarse (documented in-code and in the plan as refined by the Phase-2 fit math) — this is a planned refinement, not a data stub; the parser + classification paths are fully wired and return real derived values.

## User Setup Required
None - pure logic, no external service configuration.

## Next Phase Readiness
- The parser + struct contracts are frozen for Plan 02's live-probe layer to consume: Plan 02 adds thin `_probe_*()` functions (via the `tools` allowlist) that feed these pure parsers and assemble a full HardwareProfile, then calls classify_memory per GPU.
- No blockers. `python3 -m pytest tests/ -q` is fully green (43 passed).

## Self-Check: PASSED

All 12 created files verified present on disk; all 3 task commits (d263354, fc1aa0c, 4a0279a) verified in git log. `deneb/hardware.py` = 369 lines (min 150), contains `def parse_rocm_smi`. Full suite: 43 passed.

---
*Phase: 01-hardware-profiling*
*Completed: 2026-07-19*
