---
phase: 02-compatibility-recommendation
plan: 02
subsystem: recommendation
tags: [recommend, ranking, deterministic, cli, moe, why-strings, nothing-fits, tdd, pure-logic, keyless]

# Dependency graph
requires:
  - phase: 02-compatibility-recommendation (plan 01)
    provides: "CATALOG + by_capability() (models_catalog) and pure fits() -> FitResult{fits, headroom_mb, expected_speed_tier, caveats, need_mb} + TIER_ORDER (fit.py)"
  - phase: 01-hardware-profiling
    provides: "profile_hardware() -> HardwareProfile (usable_mem_mb already OS/unified-headroomed, primary_backend, gpus)"
provides:
  - "deneb/recommend.py: pure deterministic recommend(profile, use_case, top_n=3) -> list[Recommendation]{model, quant, fit, why} + nothing-fits under-spec fallback (REC-01, REC-03)"
  - "deneb recommend [--use coding|vision|chat|general] CLI: keyless, engine-free, prints a ranked table + 'next: deneb setup <top>' pointer (REC-02)"
affects: [setup, cli, 03-setup-advisory, qa]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic ranking via name-ascending pre-sort then a stable reverse-sort on a composite key tuple (specificity, runs_acceptably, params, speed_rank, headroom, bpw) — no dict/set ordering dependence"
    - "Pure struct-in/struct-out recommendation layer (no I/O, no tools, no LLM, no printing) — the CLI owns all display; verified by grep"
    - "Honest nothing-fits: never an empty list — a single smallest-need under-spec pick with a plain 'short by N MB' why (REC-03)"
    - "CLI subcommand mirrors cmd_profile's keyless/local/engine-free shape; the 'next: deneb setup <pick>' pointer is a printed STRING, never an invocation (Deneb Rule)"

key-files:
  created:
    - deneb/recommend.py
    - tests/test_recommend.py
  modified:
    - deneb/cli.py

key-decisions:
  - "Ranking key #1 is specificity (specialist > general-only) as a tuple-leading term, so ALL use-case specialists outrank ALL general-only models regardless of size — matches 'specialized > general'"
  - "runs_acceptably (tier != very-slow) is key #2, demoting giants that 'fit' but crawl (e.g. dense 70B) below usable models — proven live on --use chat where Mixtral/Qwen/Gemma outrank the very-slow 70B"
  - "capability_proxy = TOTAL params_b (MoE total, not active) so the 35B-A3B MoE's capability is credited while its active-3B speed already scored 'fast' in fit.py — the combination lands it #1 for coding on the big box"
  - "why-strings strip em/en dashes (_plain) because the spliced fit-caveats contain em-dashes — house style: no em-dashes in generated copy"
  - "recommend() and the nothing-fits fallback are both fully guarded (double try/except + ultimate hardcoded fallback) — a None/degraded profile yields the honest nothing-fits path, never a raise (T-03-02)"

patterns-established:
  - "Composite-key deterministic ranking with an explicit name tie-break so REC-03 tests can pin exact order"
  - "cmd_recommend imports ONLY hardware + recommend (no client/loop/tools.run) — the recommend path executes nothing"

requirements-completed: [REC-01, REC-02, REC-03]

# Metrics
duration: 7min
completed: 2026-07-19
---

# Phase 2 Plan 02: Compatibility Recommendation Summary

**Pure deterministic `recommend(profile, use_case)` that fit-filters the catalog and ranks it (specialist > general, most-capable-that-still-runs, comfortable fit, faster) with plain no-em-dash "why" strings and an honest nothing-fits fallback, plus a keyless engine-free `deneb recommend [--use ...]` CLI — verified live on this Strix box where `--use coding` correctly tops with the fast Qwen3.6-35B-A3B MoE. 11 new tests, full suite 83 green.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-07-19T13:24Z
- **Completed:** 2026-07-19T13:31Z
- **Tasks:** 2 (Task 1 = TDD RED+GREEN)
- **Files modified:** 2 created, 1 modified

## Accomplishments
- `deneb/recommend.py` — pure `recommend(profile, use_case, top_n=3)` returning `Recommendation{model, quant, fit, why}`. Per eligible model (capability OR "general") it keeps the highest-bpw quant that `fits()`, then ranks the fitting candidates by a composite key — specificity, runs_acceptably, total params, speed_rank, headroom, bpw — with a stable `model.name` ascending tie-break for full determinism. Slices top-N, builds a plain-English why per pick.
- Honest **nothing-fits (REC-03)**: never an empty list — returns ONE Recommendation for the smallest-need option with `fits=False` and a why that says the box is under-spec, names the smallest option, and reports how many MB short + suggests smaller model / more RAM / CPU offload.
- `deneb recommend [--use coding|vision|chat|general]` CLI — keyless, local, deterministic, **engine-free** (imports only `hardware` + `recommend`, no client/loop/tools.run). Builds the live profile, prints a header (use-case + budget + backend), a ranked table (#, model, quant, fit/headroom, speed, why) and a `next: deneb setup <top-pick>` **printed pointer** (never an invocation — Deneb Rule). Validates `--use`; unknown -> prints valid options + exit 2.
- **Live e2e on this Strix Halo box** (~107872 MB unified, rocm/gfx1151): `--use coding` tops with **Qwen3.6-35B-A3B (Q8_0, fast, ~65 GB free)** — the real primary coding model here — over the slow dense 33B/32B coders. Actual output pasted below.

## Live e2e output (this box)

```
$ python3 -m deneb recommend --use coding
◇ Deneb  Altronis · private-LLM setup for your AI box
ranking local models for coding on this box (budget ~107872 MB · rocm, no engine)…

  #  model                          quant   fit                speed
  1  Qwen3.6-35B-A3B                Q8_0    ✓ ~65 GB free      fast
     why: Qwen3.6-35B-A3B (Q8_0): most capable coding model that fits your ~105 GB budget with ~65 GB to spare; expected speed: fast on rocm. gfx1151 (Strix Halo): needs a recent kernel with the ROCm fixes - older kernels have known gfx1151 instability.
  2  DeepSeek-Coder-33B-Instruct    Q8_0    ✓ ~67 GB free      slow
     why: DeepSeek-Coder-33B-Instruct (Q8_0): coding-capable that fits your ~105 GB budget with ~67 GB to spare; expected speed: slow on rocm. ...
  3  Qwen2.5-Coder-32B-Instruct     Q8_0    ✓ ~68 GB free      slow
     why: Qwen2.5-Coder-32B-Instruct (Q8_0): coding-capable that fits your ~105 GB budget with ~68 GB to spare; expected speed: slow on rocm. ...

  next: deneb setup Qwen3.6-35B-A3B   (setup lands in a later release; this is a pointer, deneb runs nothing here)
```

`--use vision` tops with Llava-v1.6-Mistral-7B then Qwen2.5-VL-7B (both vision-tagged, fast), then Mixtral (general). `--use chat` tops with Mixtral-8x7B (47B, medium) then Qwen3.6-35B-A3B (fast) then Gemma-2-27B — the dense 70B is correctly demoted (it "fits" but is very-slow). `--use bogus` prints the valid options and exits 2.

## Task Commits

Each task committed atomically:

1. **Task 1 RED: failing recommend() tests** - `9daa35c` (test)
2. **Task 1 GREEN: pure recommend() ranking + why + nothing-fits (REC-01, REC-03)** - `b4f6bb5` (feat)
3. **Task 2: `deneb recommend [--use ...]` CLI + live e2e (REC-02)** - `683b970` (feat)

**Plan metadata:** committed separately (docs).

_TDD gate compliance: RED (`test(...)` 9daa35c) precedes GREEN (`feat(...)` b4f6bb5); RED confirmed genuinely (ModuleNotFoundError before implementation). No REFACTOR commit needed — helpers were small and clean on first pass._

## Files Created/Modified
- `deneb/recommend.py` (253 lines) - `Recommendation` dataclass + `recommend()` + pure helpers (`_eligible`, `_best_fitting_quant`, `_rank_key`, `_why`, `_smallest_option`, `_nothing_fits`, `_plain`/`_gb`)
- `tests/test_recommend.py` (180 lines) - 11 tests: determinism, contract, big/small-box coding, specialist>general, best-quant, vision filter, nothing-fits, plain-why, None-budget degrade, unknown-use-case
- `deneb/cli.py` (+64 lines) - `cmd_recommend()` + `_fit_cell()` helper + `recommend` dispatch in `main()` + `_HELP` line

## Decisions Made
- Specificity leads the ranking tuple, so a use-case specialist always outranks a general-only model of any size (matches the plan's "specialized > general").
- Total params (not MoE active) is the capability proxy in ranking, while fit.py's speed tier already used active params — this is exactly why the fast 35B-A3B MoE lands #1 for coding over the slow dense 32B/33B.
- why-strings pass through `_plain()` to strip em/en dashes, because the fit-caveats we splice in contain em-dashes (house style: no em-dashes in generated copy).

## Deviations from Plan

None - plan executed exactly as written. (Ranking, CLI shape, validation, and the live top-pick expectation all matched the plan; no auto-fixes were required.)

## Issues Encountered
None. RED failed genuinely (module absent) before implementation; GREEN reached first try (11/11); the live top-pick matched the plan's stated expectation (Qwen3.6-35B-A3B). Full suite green throughout.

## User Setup Required
None - no external service configuration required. The command is keyless and read-only (uses the same read-only hardware probes as `deneb profile`/`deneb check`).

## Known Stubs
None. `recommend.py` is complete pure logic; the CLI is fully wired. The `next: deneb setup <pick>` line is an intentional forward pointer to the Phase-3 setup command — a disclosed printed string, not a stub blocking this plan's goal.

## Threat Flags
None new. T-03-01 (unvalidated `--use`) mitigated: CLI validates against {coding,vision,chat,general} -> options + exit 2, and `recommend()` normalizes unknown -> "general". T-03-02 (degraded profile DoS) mitigated: None-budget yields the honest nothing-fits path, guarded, never raises (explicit test). T-03-03 (Deneb Rule / EoP) mitigated: `cmd_recommend` is print-only, keyless, engine-free — grep of its body finds no `tools.run|client.|loop.run|subprocess` (the whole-file grep matches only the pre-existing auth/one-shot/interactive paths, none in the recommend path). No new packages (T-03-SC).

## Next Phase Readiness
- Phase 2 is COMPLETE (both plans done). `recommend()` gives Phase 3 a ranked, fit-validated top pick per use-case; the `deneb setup <model>` pointer names the exact target the setup-advisory phase should consume.
- No blockers. Full test suite: **83 passed** (72 prior + 11 new). Purity + Deneb-Rule greps clean; `--use bogus` exits 2.

## Self-Check: PASSED

All created/modified files exist on disk (`deneb/recommend.py`, `tests/test_recommend.py`, `deneb/cli.py`); all 3 task commits (`9daa35c`, `b4f6bb5`, `683b970`) present in git history. Full suite `python3 -m pytest tests/ -q` = 83 passed. Live `deneb recommend --use coding` returns the expected sane top pick (Qwen3.6-35B-A3B).

---
*Phase: 02-compatibility-recommendation*
*Completed: 2026-07-19*
