---
phase: 01-hardware-profiling
plan: 02
subsystem: hardware-profiling
tags: [live-probe, orchestration, rocm-smi, nvidia-smi, system_profiler, unified-memory, graceful-degradation, cli, pytest]

# Dependency graph
requires:
  - phase: 01-01
    provides: "HardwareProfile/GPUInfo/MemoryClassification data model + 7 pure parsers + classify_memory"
provides:
  - "profile_hardware() live orchestration (deneb/hardware.py) — reads the real box via the tools boundary, feeds captured output to the pure parsers, assembles a full HardwareProfile"
  - "Per-vendor independent probes (_probe_nvidia/_amd/_apple/_lspci_fallback) that degrade to [] when a tool is absent (HW-02)"
  - "system_profiler added to the tools.py read-only allowlist (Apple GPU path)"
  - "`deneb profile` CLI subcommand — keyless, deterministic, read-only, engine-free hardware read-out"
  - "6 live-orchestration tests: NVIDIA-only/AMD-only/Apple-only/CPU-only simulated boxes + a live Strix Halo e2e assertion"
affects: [compatibility-fit, recommendation, tell-only-setup]

# Tech tracking
tech-stack:
  added: []  # stdlib only (os, platform, shutil) — no new deps per PLAT-02
  patterns:
    - "Thin per-vendor probe feeding a pure parser (mirrors check.py's _probe shape); each vendor path independent"
    - "_safe_probe wraps every probe in try/except -> [] + note, so one failing tool never aborts the profile (T-02-01)"
    - "Live layer imported at the bottom of hardware.py (os/platform/shutil/tools); pure parser section stays I/O-free"
    - "Tests swap hardware.tools/hardware.platform for fakes (not the real module) via a context manager, so the live e2e test still reads the real box"

key-files:
  created: []
  modified:
    - deneb/tools.py
    - deneb/hardware.py
    - deneb/cli.py
    - tests/test_hardware.py

key-decisions:
  - "Live layer imports (os/platform/shutil/tools) sit BENEATH the pure parsers with a banner; the pure section stays literally import-free of the tool runner (docstring scoped to match)"
  - "lspci fallback only names a GPU whose vendor SMI tool is absent (amd/nvidia) and never overrides a real SMI reading; deduped against seen vendors, tagged smi_absent"
  - "Apple unified pool comes from a single profile-level sysctl hw.memsize read that classify_memory applies — no redundant sysctl inside _probe_apple"
  - "usable_mem_mb = primary GPU's classify_memory budget, else (CPU-only) system RAM minus max(8192 MB, 15%) headroom (the <80GB-free Strix freeze scar)"
  - "Live e2e test self-skips off the Strix box (Linux + rocm-smi present guard) so the suite stays portable/green elsewhere while actively asserting on this box"

patterns-established:
  - "Invocation-vs-parsing split completed: pure parsers (01-01) + thin live probes (01-02) assembling a HardwareProfile"
  - "Graceful multi-vendor degradation proven by 4 monkeypatched single-tool boxes + a real-box live gate"

requirements-completed: [HW-01, HW-02, HW-03]

# Metrics
duration: 8min
completed: 2026-07-19
---

# Phase 1 Plan 02: Live Hardware Profiling Orchestration Summary

**profile_hardware() now reads the real machine through the read-only tools boundary and assembles a full HardwareProfile — each vendor path independent so a missing nvidia-smi/rocm-smi/system_profiler degrades to not-present and never crashes (HW-02) — with a keyless `deneb profile` read-out that on this Strix Halo box correctly reports one AMD/rocm gfx1151 GPU with unified ~124GB (not the 1GB carve-out) while NVIDIA and Apple stay absent and unmentioned.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-19T12:48:14Z
- **Completed:** 2026-07-19T12:55:45Z
- **Tasks:** 2
- **Files modified:** 4 (deneb/tools.py, deneb/hardware.py, deneb/cli.py, tests/test_hardware.py)

## Accomplishments
- profile_hardware() live orchestration: derives os/arch/kernel from platform, cpu from /proc/cpuinfo, RAM from /proc/meminfo (sysctl on Darwin), disk from shutil.disk_usage, then runs all vendor probes independently, concatenates their GPUs, classifies memory per GPU, and sets primary_backend/usable_mem_mb.
- Four independent per-vendor probes (nvidia/amd/apple/lspci-fallback) that each feed the matching pure parser and return [] on the "(not installed on this box: X)" / "[refused]" sentinels — a missing tool marks that vendor not-present, never blocks another vendor (HW-02).
- system_profiler added to the tools.py `_READ_ONLY` allowlist only (read-only macOS info tool, no write/version-only guards needed; T-02-02 held).
- `deneb profile` subcommand: local, deterministic, keyless (no _require_auth), engine-free (no client/loop) — prints os/cpu/ram/disk + per-GPU vendor/name/backend/memory + notes.
- 6 new tests (21 -> 27 in test_hardware.py; full suite 43 -> 49): NVIDIA-only, AMD-only (unified from RAM not carve-out), Apple-only (Darwin-patched), CPU-only (gpus==[], backend cpu), an all-four "never raises" sweep, and a live Strix Halo e2e assertion.

## Task Commits

Each task was committed atomically:

1. **Task 1: live profile_hardware() orchestration + system_profiler allowlist** - `d2106c0` (feat)
2. **Task 2: `deneb profile` command + graceful multi-vendor degradation tests** - `059a5b8` (feat)

## Files Created/Modified
- `deneb/tools.py` - added `system_profiler` to `_READ_ONLY` (Apple GPU probe); no other tier touched
- `deneb/hardware.py` - appended the live probe layer beneath the pure parsers: `_tool_output`/`_read_output`, `_parse_rocm_driver`, `_probe_nvidia`/`_probe_amd`/`_probe_apple`/`_probe_lspci_fallback`, `_safe_probe`, and `profile_hardware()`; docstring scoped so the I/O-free claim covers the pure section only
- `deneb/cli.py` - `cmd_profile()` printer + `profile` dispatch + `_HELP` line
- `tests/test_hardware.py` - live-orchestration section: fake tools/platform swap via context manager, 4 simulated single-tool boxes, a never-raises sweep, and a self-skipping live Strix assertion

## Live e2e Result (this box)
`python3 -m deneb profile` prints:
- os Linux · x86_64 · kernel 7.2.0-...fc43.x86_64
- cpu AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (32 cores)
- ram 126908 MB · disk 214383 MB free
- GPU: amd · AMD Radeon 8060S Graphics [rocm] — unified **126908 MB** (shared pool; usable ~107872 MB), gfx1151 · sku STRXLGEN · driver 7.2.0-...
- primary backend rocm · usable model budget ~107872 MB
- NVIDIA and Apple: absent and unmentioned; the 1024 MB VRAM carve-out is recorded (vram_carveout_mb) but never used as the budget.

## Decisions Made
- Apple unified pool read once at the profile level (sysctl hw.memsize) and applied by classify_memory, rather than duplicating the sysctl call inside _probe_apple — same result, no redundant probe.
- lspci fallback restricted to amd/nvidia and deduped against vendors already found by SMI, so it only ever adds a "GPU present, SMI tool absent — VRAM unread" record, never a duplicate or an override.
- CPU-only / SMI-unread usable budget falls back to system RAM minus the documented headroom (the coarse Phase-2-refined floor), so the profile always returns a budget.
- Live e2e test self-skips off this box (guards on Linux + `which rocm-smi`) so the suite is portable while still hard-asserting on the Strix box under both pytest and the standalone `__main__` runner.

## Deviations from Plan

None requiring a rule. Plan executed as written. Two clarifying implementation choices worth noting (neither changes behavior vs the plan's intent):
- The plan lists `sysctl hw.memsize` inside `_probe_apple`; I read it once at the profile level instead and let classify_memory apply the unified pool (the plan's own assembly step already passes ram_total_mb into classify_memory). The Apple GPU still ends up unified with unified_mem_mb from sysctl — verified by test_apple_only_box_one_metal_gpu_unified.
- Lightly scoped the hardware.py module docstring's "performs NO I/O" sentence to the pure parser section, since the live layer (the only I/O) is now present below it — honesty fix, no code impact.

## Issues Encountered
None. nvidia-smi and system_profiler being absent on this box produced clean "(not installed on this box: X)" sentinels that the parsers already treat as absence; no probe raised.

## Known Stubs
None. Every field the printer shows is derived from a real probe or stdlib call. The unified headroom (max 8192 MB / 15%) is intentionally coarse and labelled Phase-2-refined in both code and output — a planned refinement, not a data stub; the paths are fully wired.

## Threat Flags
None. No new network endpoints, auth paths, or schema changes. The one allowlist expansion (system_profiler) is a read-only macOS info tool covered by the plan's T-02-02 disposition; it lives only in `_READ_ONLY`, never in `_WRITE_BINS`/`_DESTRUCTIVE`/`_BAD_ARGS`.

## User Setup Required
None - deterministic local read-only logic, no external service configuration, no keys.

## Next Phase Readiness
- Phase 1 complete: `profile_hardware()` returns a full HardwareProfile on any common box (NVIDIA/AMD/Apple/CPU), degrading gracefully, with the unified-memory trap handled live. Phase 2 (compatibility + `deneb recommend`) consumes this profile's usable_mem_mb / primary_backend / per-GPU memory_kind for the fit math.
- No blockers. `python3 -m pytest tests/ -q` = 49 passed; `python3 -m deneb profile` verified live on this box.

## Self-Check: PASSED

All 4 modified files present on disk; both task commits (d2106c0, 059a5b8) verified in git log. `deneb/hardware.py` = 563 lines, contains `def profile_hardware`. `deneb/tools.py` contains `"system_profiler"` in `_READ_ONLY` only. `deneb/cli.py` contains `cmd_profile` + `profile` dispatch. Full suite: 49 passed. Live `deneb profile` prints the AMD unified profile with no NVIDIA/Apple mention and no 1GB budget.

---
*Phase: 01-hardware-profiling*
*Completed: 2026-07-19*
