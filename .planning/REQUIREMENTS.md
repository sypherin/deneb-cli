# Requirements: Deneb v1 (advisory-first, hardware-general)

**Defined:** 2026-07-19
**Core Value:** "Will this model run on my box, and how do I set it up?" — answered deterministically from the real machine, on any common hardware.

## v1 Requirements

### Hardware Profiling (HW)

- [x] **HW-01**: `profile_hardware()` returns a structured HardwareProfile: os/arch, cpu, ram_total,
      disk_free, and per-GPU {vendor, name, vram_mb or unified_mem_mb, driver, backend(cuda/rocm/metal/cpu)}.
- [x] **HW-02**: Detection spans NVIDIA (nvidia-smi), AMD (rocm-smi), Apple (system_profiler/sysctl),
      and CPU-only. Each vendor path is independent; absence of a vendor tool degrades gracefully
      (never crashes, marks that vendor "not present").
- [x] **HW-03**: Unified-memory platforms (Apple Silicon, Strix Halo) reported as unified_mem, not VRAM;
      the fit math accounts for the shared pool (the <80GB-free scar informs a usable-headroom margin).
- [x] **HW-04**: Pure parsers for each tool's output are unit-tested against captured sample outputs
      (no live tool needed in tests).

### Compatibility + Fit (FIT)

- [ ] **FIT-01**: A curated model catalog (data file): models × available quants × approx file-size/
      VRAM-need × capability tags (coding/vision/chat/general) × GGUF source.
- [ ] **FIT-02**: `fits(model, quant, profile)` pure function: does it fit the usable memory (with a
      safety margin), returns {fits: bool, headroom_mb, expected_speed_tier, caveats[]}. Unit-tested
      with a canonical case set (e.g. a 7B-Q4 fits 16GB; a 70B-Q4 does not).
- [ ] **FIT-02b**: Known-caveat rules apply (e.g. gfx1151 kernel note on Strix, unified-mem headroom).

### Recommendation (REC)

- [ ] **REC-01**: `recommend(profile, use_case)` ranks the catalog to the best-fitting model+quant(s)
      for the box + use-case; deterministic ordering; returns top-N with a plain-language "why".
- [ ] **REC-02**: `deneb recommend [--use coding|vision|chat|general]` CLI: prints the ranked table
      (model, quant, fit, expected speed, why) + a one-line next-step pointer to `deneb setup <pick>`.
- [ ] **REC-03**: Ranking + why-string are pure and unit-tested; a "nothing fits" path is handled
      honestly (tell the user their box is under-spec + the smallest option).

### Tell-Only Setup Advisory (SET)

- [ ] **SET-01**: `deneb setup <model>` outputs the ordered setup steps for that model on this
      platform as ADVICE — each step: the exact command, what it does, and a risk/warning flag where
      appropriate (sudo, big download, service change). NEVER executes (the Deneb Rule).
- [ ] **SET-02**: Steps are platform-branched (CUDA vs ROCm vs Metal vs CPU) from the curated playbook;
      pure step-generation is unit-tested.
- [ ] **SET-03**: Every generated step that would need sudo / download N GB / change a service carries
      its warning inline (per the Deneb Rule risk-surfacing principle).

### Platform / CLI (PLAT)

- [ ] **PLAT-01**: All v1 commands are keyless (free tier) and deterministic — no engine round-trip,
      no LLM, works offline. `deneb check` stays as-is; add `recommend` + `setup` (tell-only).
- [ ] **PLAT-02**: Reuse the existing `tools` allowlist for read-only probes; no new heavy deps.
- [ ] **PLAT-03**: Graceful, honest degradation everywhere (missing tool, unknown GPU, nothing fits) —
      never crash, never guess a hardware fact it can't read.

### Quality (QA)

- [ ] **QA-01**: pytest on all pure logic: profile parsers (HW-04), fit math (FIT-02), recommend
      ranking (REC-03), setup step generation (SET-02).
- [ ] **QA-02**: E2E smoke: `deneb recommend` runs on THIS real box (Strix Halo, rocm-smi) and returns
      a sane recommendation; `deneb setup <pick>` prints tell-only steps with warnings.
- [ ] **QA-03**: Verify the Deneb Rule holds — no v1 command path executes anything; a test asserts
      `setup` only prints, never runs.
- [ ] **QA-04**: Zach's machine = final acceptance gate.

## v2+ (deferred — master plan)

- Execution tier (downloads/jobs/exposure/updates), LLM fallback, online-verify + self-learning,
  Altronis-stack hardening, client key/paid tier, teardown, config export.

## Out of Scope

- Any command execution in v1 (tell-only).
- LLM in the core path.
- Windows (v1 targets Linux + macOS common local-LLM boxes; Windows later).
