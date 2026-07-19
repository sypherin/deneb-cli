# Deneb Level-Up Plan — Read-only Advisor → Safe Executor

**Written:** 2026-07-19 · **Status:** proposed, awaiting Zach's go (then GSD execution)

## Vision

Today Deneb DIAGNOSES and WALKS you to the fix (read-only). Level it up so it can
actually **do the setup**: download models, create the systemd jobs, expose the endpoint
(Cloudflare or Vodien), and keep the stack updated — safely, gated, logged, reversible —
using the Altronis DGX/Strix runbook as its executable playbook.

## Tiering (Zach, 2026-07-19) — this is the money split

| Tier | Access | What it does |
|------|--------|--------------|
| **FREE** | keyless `deneb check` + free-key basics | Diagnosis (deterministic probes) + basic guidance/troubleshooting. The lead magnet + hook. |
| **PAID (client key)** | authed to engine premium tier | **Actual EXECUTION** (downloads, job setup, exposure, updates) + **advanced details** (full runbook intelligence, hardening to reference-state, their specific stack, support). The moat. |

Execution IS the paid tier. Diagnosis + basics stay free. Enforced server-side in the
engine per key tier (never in the local CLI — can't be cracked by editing python).

## Current state (verified 2026-07-19)

- `deneb/check.py` — deterministic read-only probes (endpoint, service, gpu, model file,
  gateway, tunnel) → ✓/✗ + fix hint. Keyless. **Keep as the free hook.**
- `deneb/loop.py` + `client.py` — LLM troubleshoot via engine; returns commands for the
  human to run. **Advisory only — does not execute.**
- Engine brain (neo-assistant `agent.py`) — proposes exact commands + `[ESCALATE]` for
  human-needed steps. The propose/escalate half of an executor exists; **the local execute
  half does not.**
- Runbook to ingest: `~/neo-gateway/docs/DGX_SPARK_SETUP_2026-07-10.md` (incl the new
  Option A Cloudflare + Option B Vodien exposure sections).

## Design principles (non-negotiable — high blast radius: sudo, GBs, prod boxes)

0. **THE DENEB RULE (Zach, 2026-07-19 — absolute, overrides everything below).**
   Deneb ALWAYS **TELLS** the user the exact command and what it does — first, every time.
   It **only EXECUTES a command when the user explicitly ASKS** for that specific command.
   Advise-first is the default AND the rule: there is **no silent run, no blanket auto-apply,
   no batch "yes to all"** that executes without the user asking per command. Even in the paid
   execution tier, the flow is: Deneb tells → user asks "run it" → Deneb runs THAT command →
   reports → tells the next. (No `--auto`/`--yes-to-all` that bypasses this. A per-command
   confirm is the floor.)

1. **ALWAYS surface risks + warnings** where appropriate, BEFORE the user decides. Every
   proposed command flags what it will do that matters: needs `sudo`, opens a network port,
   restarts a running service, deletes/overwrites, downloads N GB, changes a config of record,
   touches data. No command is shown without its warning context. This is what makes Deneb the
   *governed* tool — on-brand with the judgment/governance positioning.

2. **Gated execution loop.** Engine proposes a STEP {intent, exact command, why, expected
   result, risks, rollback}; CLI shows it + its warnings, waits for the user to ask, runs
   locally on ask, captures output, verifies, reports. Loop: tell → ask → run → verify → tell next.

3. **Dry-run / preview** first-class (`deneb setup --plan` shows every step + risks, runs nothing).
4. **Full audit log** — every command + output + who approved + timestamp, queryable
   (reuse the neo audit pattern; this is also a governance/sales artifact).
5. **Idempotent** — re-running skips completed steps (state file + the existing `check`
   probes as the "is this step already done?" oracle).
6. **Reversible** — each step carries a rollback; failure → offer undo, never leave a
   half-state. Never destructive without explicit confirm.
7. **Platform-branched** — DGX (ARM64 / CUDA) vs Strix Halo (x86 / ROCm): the runbook's
   audit fixes (`--no-mmap` vs `--mmap`, ubatch, `CUDA_VISIBLE_DEVICES` vs Vulkan/ROCm env,
   ARM64 cloudflared) branch per detected platform. `check` already detects the box.
8. **Fail loud** — no success status that masks a failed sub-step (build discipline).

## Phases (GSD-suited; ~multi-week)

### Phase 0 — Ingest the runbook as an executable playbook  *(the knowledge/skill Zach asked to add)*
Convert `DGX_SPARK_SETUP_2026-07-10.md` into a structured, versioned playbook: each section
→ ordered steps `{id, intent, preconditions, platform, commands, verify_probe, rollback,
idempotency_key, tier}`. Sections: prereqs, llama.cpp build (CUDA/ROCm), model downloads
(Gemma/Qwen-VL/Surya v1+v2), gateway auth, Surya, Qwen-VL, **exposure A (Cloudflare tunnel)
+ B (Vodien: Caddy+DNS+ports+ufw — the just-added idiot-proof section)**, point-the-app.
Lives engine-side (gated); `check` probes become the per-step verify oracles.

### Phase 1 — Safe execution core  *(the foundation — get safety right before capabilities)*
The gated executor loop + audit log + dry-run + idempotency state + rollback. Extend the
engine's propose/escalate into propose→consent→execute→verify. Everything else rides on this.

### Phase 2 — Model downloads
`deneb setup model <name>`: `hf download` with progress, **resume**, disk-space precheck
(⚠ the <80GB Strix freeze scar), size/integrity verify, correct target dir, detached for big
GGUFs (DS4-download pattern). Idempotent (skip if present + valid).

### Phase 3 — Job setup (systemd services + timers)
Create + enable the stack units from the playbook (llama-server/Gemma, Qwen-VL, Surya v1/v2,
gateway), platform-branched, each verified up via its `check` probe. Idempotent; rollback =
disable/remove on failure. `[Unit]`/`[Install]` sections correct (the old kit gap).

### Phase 4 — Exposure automation
Option A (cloudflared: arm64 binary, login, create, route dns, config.yml, service) OR
Option B (Caddy install+config, guided DNS + router port-forward which can't be automated so
verified-not-executed, ufw lockdown). Uses the exposure content ingested in P0.

### Phase 5 — Updates + maintenance
`deneb update`: safely update llama.cpp (rebuild), models, cloudflared/caddy, packages —
"verify don't reinstall" discipline, pre-update backup, rollback on failure. Ongoing health.

## Cross-cutting
- **Tiering enforcement**: free = diagnose+basics; paid = execute+advanced (engine-gated).
- **Funnel**: keyless check → teaser → free-key (email = lead capture) → client key
  (execution+advanced). Soft "want it done for you? → altronis.sg" CTA on every check.
- **Tests**: extend `eval/` + `node --test`/pytest on the pure playbook/idempotency/rollback
  logic (QA-01 style). Every executor step needs a dry-run test + a rollback test.
- **Docs**: README gains the tier table + the paid CTA (currently missing).

### Phase 6 — Living knowledge: model ↔ hardware awareness + online verify + self-update  *(Zach, 2026-07-19)*
Deneb must not rely on a static baked runbook — the LLM landscape changes weekly. Add:
- **Model↔hardware compatibility knowledge**: a maintained matrix — model × quant × GGUF source
  × hardware profile (DGX ARM64/CUDA, Strix Halo x86/ROCm, VRAM / unified-mem ceiling, driver/
  kernel caveats e.g. the gfx1151 kernel bugs) → does it fit, does it run, expected tok/s, known
  breakage. This answers the #1 user question: "will THIS model run on MY box?"
- **Online-verify-first**: before recommending/acting on anything version-sensitive (which model,
  which quant, does X run on Y), Deneb checks live sources (HF model card/repo, release notes)
  to confirm against its own knowledge. Never assert a model/hardware fact from stale memory —
  verify live, **tag confidence (known/verified vs guessed)**, cite the source.
- **Self-updating memory (GATED)**: when Deneb learns something new (a model/quant works or fails
  on a hardware profile, a new release, a driver fix), it proposes writing it back to the
  knowledge base so all users benefit — a learning flywheel where every box is a data point.
  ⚠ Gated: a single noisy result must NOT auto-mutate canonical knowledge; promote to canonical
  only on confidence threshold / corroboration (same governance discipline as everything else).
  Aggregate LEARNINGS, never user data.

## Additional gaps surfaced (2026-07-19 "what else did we miss")

1. **⚠ AIR-GAP TENSION (the sharp one).** Your best private-LLM buyers — regulated, sensitive,
   IP-heavy — are often the ones whose boxes CAN'T reach the internet (that's WHY they run
   private LLM). "Always check online" breaks for exactly them. Deneb needs an **offline / air-
   gapped mode**: works from baked knowledge only, clearly says "couldn't verify live — this is
   as of <date>", and can be pre-seeded with a knowledge bundle. Online-verify degrades
   gracefully, never blocks. This isn't an edge case — it's your core market.
2. **"What should I even run?" recommendation** (highest-value FREE feature). Before setup:
   given detected hardware + use-case (coding / vision / OCR / chat), recommend the model +
   quant. This is JUDGMENT (your brand) and it needs the Phase-6 compat knowledge. Great hook,
   funnels to paid setup.
3. **Deep hardware profiling** — foundation for 1/2 and Phase 6: GPU model, VRAM/unified mem,
   driver+kernel versions, disk, OS/arch, thermal. Drives every recommendation. Extend `check`.
4. **Unify troubleshoot + setup loops** — when a setup step fails, the existing troubleshoot
   brain diagnoses WHY and proposes the fix. One loop, not two.
5. **Whole-stack orchestration** — dependency-ordered across services (models → gateway →
   exposure), not one model at a time. Coherent up/down of the full stack.
6. **Security-posture verification** — verify the result is SECURE not just working: key set,
   no unauthed ports, gateway in front, endpoint returns 401 without key. (Partly in `check`.)
7. **Clean teardown / uninstall** — whole-stack undo, not just per-step rollback.
8. **Config export / portability** — dump the working config for backup, reproducibility, or
   handing to Altronis support (feeds the paid tier).
9. **Playbook versioning** — engine playbook updates vs installed CLI version; prompt-to-update.

## Positioning: why Deneb ≠ "just ask ChatGPT / Claude" (Zach, 2026-07-19)

The #1 objection to Deneb is "why not just ask a general LLM how to set up my local LLM?".
The answer must be built-in and loud, because it's also the moat. Defensible reasons a
general LLM STRUCTURALLY cannot match:

1. **Grounded in YOUR actual box, not guessing.** Deneb READS your real logs, services, GPU,
   driver/kernel versions, config, disk. ChatGPT gives generic advice for an imagined setup and
   confidently hallucinates service names + paths. Grounded-vs-guessing is the killer difference.
2. **Current, not frozen at a training cutoff.** General LLMs don't know this month's models, the
   ROCm/CUDA quirk that shipped last week, the driver that just broke gfx1151. Deneb verifies
   online + self-updates (Phase 6). A general model can't — its knowledge is static.
3. **Specialist depth, not generalist breadth.** Deneb carries the exact private-LLM stack + the
   hardware compatibility matrix + the specific failure modes → won't send you down a generic
   wrong path.
4. **It ACTS (safely, consent-first), not just talks.** Verified playbook, step-by-step, checks
   each step worked — vs copy-pasting ChatGPT's `sudo` suggestion and praying.
5. **It LEARNS + REMEMBERS.** Every real setup on real hardware improves its compat knowledge; it
   accumulates ground truth no general LLM has, because general LLMs don't learn from your box.
6. **Safe + governed by design** (the Deneb Rule + risk warnings) — vs blindly trusting a general
   model's command.

> **⚠ Honest caveat (this claim is only true if we BUILD it).** The differentiation comes from the
> GROUNDING (reads your box) + the CURATED runbook + the deterministic `check` probes + the
> self-learning memory — NOT from a smarter base model. If Deneb becomes "a general LLM with a
> system prompt," the "better than ChatGPT" claim is hollow and buyers will see through it. So the
> product MUST lean on grounding + curated knowledge + deterministic checks + memory — the exact
> things a general LLM structurally lacks. Build those, or don't make the claim.

## Open decisions — need Zach before building (2026-07-19)

1. **WHO is Deneb for? — RESOLVED (Zach 2026-07-19): FREE = everybody, PAID = his clients only.**
   ⚠ Implication that MUST shape the build: "free for everybody" only holds if the FREE tier works
   on COMMON hardware (Mac M-series, consumer NVIDIA 3090/4090, generic Linux/AMD) — NOT just
   DGX/Strix. So: FREE tier = HARDWARE-GENERAL ("is my box ready + what should I run + generic
   guidance", reads any box) — this is where the model↔hardware knowledge + online-verify (Phase 6)
   earns its keep. PAID tier = ALTRONIS-STACK + HARDWARE SPECIFIC (set up + harden THEIR private
   stack to reference state, execution, support) — narrow is fine, it's the high-touch client work.
   Note: this converges with decision #5 — the lean v1 IS the free hardware-general advisory tier.
2. **Cloud brain vs local — RESOLVED via the BOOTSTRAP PARADOX (Zach 2026-07-19).** Zach's catch:
   an AI-setup tool can't depend on a working local LLM to do the setup, because the local LLM is
   the very thing that doesn't exist yet. RESOLUTION — **deterministic-first engine, LLM as
   fallback only:**
   - Deneb's CORE needs NO LLM: `check` probes (already deterministic), the curated playbook
     (steps = command + verify + rollback = DATA), hardware detection + model-fit math, the
     model↔hardware compat matrix (a lookup table). That's ~80-90% of the value, runs locally +
     air-gapped + BEFORE any LLM exists, on any box.
   - LLM only for NOVEL troubleshooting (error not in the playbook), sourced in priority:
     (1) default = phone home to the Altronis cloud engine (box has net, no local LLM yet — fine);
     (2) once Deneb has set up the local LLM, hand off to THAT; (3) air-gapped pre-setup = closest-
     playbook-match + escalate to Altronis support, or a bundled offline knowledge pack (+ optional
     tiny bundled troubleshooting model — Deneb's own small one, NOT the big private LLM).
   - Bonus: this IS the "why not ChatGPT" answer — Deneb's power is the grounded probes + curated
     data (deterministic), not an LLM. ChatGPT is pure LLM, nothing grounded.
   - "Whole package for clients" = the deterministic core + offline knowledge bundle (+ small
     troubleshooting model if they want offline AI-assist). They do NOT need the big private LLM
     running for Deneb to set it up — that's the whole point.
   ⚠ Design shift: current `loop.py` is LLM-first (troubleshoot via engine). Flip to
   deterministic-first, LLM-as-fallback. This dissolves the paradox and works everywhere.
3. **Opinionated vs flexible.** Only the Altronis reference stack (Gemma+Qwen-VL+Surya+gateway),
   or any model the user picks? Opinionated = simpler + on-brand; flexible = broader + harder.
4. **Monetization.** How the paid tier charges: one-off setup fee / subscription / bundled into the
   consulting retainer / per-box license. Shapes the whole key + tier system.
5. **Minimum v1 (recommended to decide FIRST).** Full 6 phases + gaps = multi-month; risks never
   shipping (anti-over-build rule). Proposed lean v1: "what should I run on my box" recommendation +
   diagnose + TELL-only guided setup, keyless/free — ship, validate the funnel/reach-outs, THEN
   build execution (the big risky part). Validate demand before pouring months in.
6. **Liability / support posture.** Deneb runs `sudo` on a user's prod box. Even consent-first +
   warnings, if it breaks something — disclaimer + support stance, esp for paid clients?
7. **Model licensing** — Gemma/Qwen/Surya licenses when Deneb auto-downloads + sets them up
   commercially for clients: compliance check, not assumed.

## Recommended execution
Run through GSD as its own milestone: `/gsd:new-milestone` in ~/deneb-cli → phases above.
Phase 1 (safe execution core) is the highest-blast-radius part — plan + verify it hardest.
Phase 6 (living knowledge) + the air-gap mode are the parts most likely to be under-scoped —
flag them for their own discuss round before planning.
