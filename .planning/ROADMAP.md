# Roadmap: Deneb v1 (advisory-first, hardware-general)

## Overview

Three phases. P1 builds the deterministic hardware-profiling foundation (read any box). P2 adds
the curated compatibility matrix + fit math + the `deneb recommend` command (the core value —
"what should I run"). P3 adds the tell-only `deneb setup` advisory + wires the keyless CLI + the
full QA gate. No execution, no LLM, deterministic throughout.

## Phases

- [x] **Phase 1: Hardware Profiling** — read any common box (NVIDIA/AMD/Apple/CPU), structured profile, graceful degradation, tested parsers (completed 2026-07-19)
- [x] **Phase 2: Compatibility + Recommendation** — curated model catalog, fit math, `deneb recommend --use` with ranked why (completed 2026-07-19)
- [ ] **Phase 3: Tell-Only Setup + CLI + QA** — `deneb setup <model>` advisory (tell-only, risk-flagged), keyless wiring, full QA gate

## Phase Details

### Phase 1: Hardware Profiling
**Goal**: `deneb` can read the real machine on any common hardware and produce a structured HardwareProfile, degrading gracefully when a vendor tool is absent.
**Depends on**: Nothing
**Requirements**: HW-01, HW-02, HW-03, HW-04
**Success Criteria**:
  1. `profile_hardware()` returns os/arch/cpu/ram/disk + per-GPU vendor/name/mem/driver/backend on this real box
  2. NVIDIA/AMD/Apple/CPU paths independent; a missing vendor tool marks "not present", never crashes
  3. Unified-memory boxes reported correctly (not as VRAM)
  4. Pure parsers unit-tested against captured sample tool outputs
**Plans**: 2 plans
  - [x] 01-01-PLAN.md — pure data model + per-vendor parsers + unified/usable memory classification, tested against captured fixtures
  - [x] 01-02-PLAN.md — live profile_hardware() orchestration + tools allowlist + graceful multi-vendor degradation + `deneb profile` (e2e on this box)

### Phase 2: Compatibility + Recommendation
**Goal**: Given a profile + use-case, Deneb deterministically recommends the best model+quant with a plain "why".
**Depends on**: Phase 1
**Requirements**: FIT-01, FIT-02, FIT-02b, REC-01, REC-02, REC-03
**Success Criteria**:
  1. Curated model catalog (data) with quants/sizes/capability tags/GGUF sources
  2. `fits()` pure + unit-tested (7B-Q4 fits 16GB; 70B-Q4 does not; unified-mem headroom honored)
  3. `deneb recommend --use coding` on this box returns a sane ranked recommendation with why-strings
  4. "nothing fits" handled honestly
**Plans**: 2 plans
  - [x] 02-01-PLAN.md — curated model catalog (FIT-01) + pure `fits()` fit-math, speed-tier heuristic, and known-caveat rules (FIT-02, FIT-02b), TDD
  - [x] 02-02-PLAN.md — pure deterministic `recommend()` ranking + why-strings + nothing-fits (REC-01, REC-03) + `deneb recommend --use` CLI live on this box (REC-02)

### Phase 3: Tell-Only Setup + CLI + QA
**Goal**: `deneb setup <model>` tells the user the platform-correct setup steps (never executes, risk-flagged), the CLI is keyless, and the QA gate passes on Zach's box.
**Depends on**: Phase 2
**Requirements**: SET-01, SET-02, SET-03, PLAT-01, PLAT-02, PLAT-03, QA-01, QA-02, QA-03, QA-04
**Success Criteria**:
  1. `deneb setup <pick>` prints ordered, platform-branched steps as advice with inline warnings, executing nothing
  2. A test asserts no v1 path executes (the Deneb Rule holds)
  3. All commands keyless + deterministic (no engine/LLM)
  4. Full pytest green + `deneb recommend`/`setup` verified live on this box + Zach's phone/box final gate
**Plans**: 2 plans
  - [ ] 03-01-PLAN.md — pure tell-only setup advisory core: Step model + curated platform-branched PLAYBOOK + `setup_steps()` step-gen with inline sudo/download/service warnings (SET-01, SET-02, SET-03, QA-01), TDD
  - [ ] 03-02-PLAN.md — `deneb setup <model>` CLI wiring (keyless, print-only) + help/README + the QA-03 Deneb-Rule no-execution assertion + full-suite green + live e2e on this box (PLAT-01/02/03, QA-02, QA-03, QA-04)

## Progress

| Phase | Plans Complete | Status |
|-------|----------------|--------|
| 1. Hardware Profiling | 2/2 | Complete |
| 2. Compatibility + Recommendation | 2/2 | Complete |
| 3. Tell-Only Setup + CLI + QA | 0/2 | Planned |

## Coverage

All v1 requirements mapped, each to exactly one phase:

| Category | Requirements | Phase |
|----------|--------------|-------|
| Hardware Profiling | HW-01..04 | 1 |
| Compatibility + Fit | FIT-01, FIT-02, FIT-02b | 2 |
| Recommendation | REC-01, REC-02, REC-03 | 2 |
| Tell-Only Setup | SET-01, SET-02, SET-03 | 3 |
| Platform/CLI | PLAT-01, PLAT-02, PLAT-03 | 3 |
| Quality | QA-01, QA-02, QA-03, QA-04 | 3 |
