# Roadmap: Deneb v1 (advisory-first, hardware-general)

## Overview

Three phases. P1 builds the deterministic hardware-profiling foundation (read any box). P2 adds
the curated compatibility matrix + fit math + the `deneb recommend` command (the core value —
"what should I run"). P3 adds the tell-only `deneb setup` advisory + wires the keyless CLI + the
full QA gate. No execution, no LLM, deterministic throughout.

## Phases

- [ ] **Phase 1: Hardware Profiling** — read any common box (NVIDIA/AMD/Apple/CPU), structured profile, graceful degradation, tested parsers
- [ ] **Phase 2: Compatibility + Recommendation** — curated model catalog, fit math, `deneb recommend --use` with ranked why
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
  - [ ] 01-02-PLAN.md — live profile_hardware() orchestration + tools allowlist + graceful multi-vendor degradation + `deneb profile` (e2e on this box)

### Phase 2: Compatibility + Recommendation
**Goal**: Given a profile + use-case, Deneb deterministically recommends the best model+quant with a plain "why".
**Depends on**: Phase 1
**Requirements**: FIT-01, FIT-02, FIT-02b, REC-01, REC-02, REC-03
**Success Criteria**:
  1. Curated model catalog (data) with quants/sizes/capability tags/GGUF sources
  2. `fits()` pure + unit-tested (7B-Q4 fits 16GB; 70B-Q4 does not; unified-mem headroom honored)
  3. `deneb recommend --use coding` on this box returns a sane ranked recommendation with why-strings
  4. "nothing fits" handled honestly
**Plans**: TBD

### Phase 3: Tell-Only Setup + CLI + QA
**Goal**: `deneb setup <model>` tells the user the platform-correct setup steps (never executes, risk-flagged), the CLI is keyless, and the QA gate passes on Zach's box.
**Depends on**: Phase 2
**Requirements**: SET-01, SET-02, SET-03, PLAT-01, PLAT-02, PLAT-03, QA-01, QA-02, QA-03, QA-04
**Success Criteria**:
  1. `deneb setup <pick>` prints ordered, platform-branched steps as advice with inline warnings, executing nothing
  2. A test asserts no v1 path executes (the Deneb Rule holds)
  3. All commands keyless + deterministic (no engine/LLM)
  4. Full pytest green + `deneb recommend`/`setup` verified live on this box + Zach's phone/box final gate

## Progress

| Phase | Plans Complete | Status |
|-------|----------------|--------|
| 1. Hardware Profiling | 0/2 | Planned |
| 2. Compatibility + Recommendation | 0/TBD | Not started |
| 3. Tell-Only Setup + CLI + QA | 0/TBD | Not started |

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
