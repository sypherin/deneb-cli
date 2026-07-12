# Deneb CLI — build plan (v1)

A one-install, zero-Claude, **narrow** terminal agent that gets an AI box (NVIDIA DGX
Spark, AMD Strix Halo, …) into the working **Neo Altronis** reference state — install,
configure, troubleshoot the local-LLM stack, and verify it's actually serving. Nothing
else. Brain = Neo Altronis only. For Zach now; login as sypherin@gmail.com via a
paste-once token.

Status legend:  ✅ done · 🔨 building · ⬜ todo · ⏸ deferred (stated, not silently dropped)

---

## 0. Reuse map (check-first — REUSE what exists, don't rebuild)

Audited Zach's codebase; almost everything intelligent already exists and is REUSED:

| Need | Already exists → REUSE | New? |
|------|------------------------|------|
| Persona / guardrails / escalation / compliance backstop | `~/neo-assistant/src/service.py` (`NEO_SYS`, `_REGULATED_RE`, `_notify_escalation`) | reuse |
| KB retrieval (TF-IDF, allowlist) | `~/neo-assistant/src/knowledge.py` | reuse |
| Screenshot OCR (Surya :8090) | `~/neo-assistant/src/vlm.py` | reuse |
| Web search (SearXNG :8888) | `~/neo-assistant/src/websearch.py` | reuse |
| LLM call (Neo :8001) | `~/neo-assistant/src/llm.py` | reuse |
| Runbook / KB content | `~/neo-gateway/docs/*` + the 10-Jul CF guide | reuse |
| Cloud endpoint + Bearer auth | `neo.altronis.sg` gateway + `deneb-engine.altronis.sg` | reuse |
| Client TUI / HTTP / args | `rich` · stdlib `urllib` · `argparse` (libraries, not hand-rolled) | reuse |
| Server-side agent step | *nothing existed* (old Deneb is single-shot chat) | NEW — but only glue over the reused modules above |
| Read-only tool executor + allowlist (client) | *nothing existed* (this is the new security boundary) | NEW |

Checked and REJECTED as foundations (with reason): wrapping Claude Code / ccr (branding
can't be stripped — you wanted zero Claude), forking Goose / gemini-cli (inherit a whole
framework + scrub *their* branding — heavier than a thin client, no gain). So: **reuse the
Neo brain + standard libs; the only genuinely new code is the thin client + a small
server-side step.**

## 1. Architecture

```
  ┌─────────────────────────── the box (DGX / Strix) ───────────────────────────┐
  │  deneb  (OPEN, MIT, on the box — auditable, READ-ONLY)                        │
  │   • holds the conversation + executes read-only tools LOCALLY                 │
  │   • tools: run(allowlisted read-only cmd) · read_file · list_dir · ocr(paste) │
  │   • enforces the read-only allowlist itself (defence in depth)                │
  │   • Deneb-branded TUI. NO Claude / third-party branding anywhere.             │
  └───────────────┬──────────────────────────────────────────────────────────────┘
                  │  HTTPS + Authorization: Bearer <paste-once token>
                  ▼
  ┌────────── your Neo cloud (PRIVATE, gated) — deneb-engine.altronis.sg ─────────┐
  │  /agent-step  (stateless): retrieve KB → apply persona+guardrails → call Neo  │
  │               LLM → return NEXT read-only tool  OR  final diagnosis           │
  │  KB / persona / guardrails / web-search NEVER leave the server.               │
  │  brain = Neo Altronis (:8001) ONLY. no GLM, no fallback.                      │
  │  web-search = SearXNG (:8888) + page-fetch, server-side, triggered by agent.  │
  └───────────────────────────────────────────────────────────────────────────────┘
```

Loop: client POSTs `{history}` → engine returns `{type:"action", …}` or
`{type:"final", answer, sources, escalated}` → client executes the read-only tool,
appends the observation, repeats → renders the final diagnosis + verbatim runbook
sources. **The moat (KB/guardrails/model) stays server-side; the client is a dumb,
auditable, read-only executor.**

**Web search / reading pages (answering "how will it search?"):** the *engine* does it,
not the client. When the runbook is stale/insufficient the agent emits a search; the
engine hits **SearXNG (your private meta-search, :8888)** and can fetch+extract a
specific page (e.g. an NVIDIA/AMD docs URL), grounds the answer, and returns it. The
client box never searches directly — so it works even on a locked-down client network
(only needs to reach your Neo endpoint), and the search config/keys stay yours.

---

## 2. Definition of Done (the contract — every box ticked AND verified by running)

**Features**
- F1 one-line install runs clean on a fresh box
- F2 `deneb auth --token` → paste once, stored chmod-600, validated against the engine
- F3 interactive troubleshooting: reads logs/files/dirs agentically, diagnoses grounded in the runbook, shows verbatim runbook sources
- F4 `deneb check` — the "am I done?" scan against the Neo reference (llama.cpp+accel · model path · service · :8001/health · gateway · tunnel) → done, or exact gaps + fixes
- F5 paste a screenshot → OCR (Surya) → diagnose
- F6 real endpoint verification (pings *serving* endpoints, not just "service started")
- F7 scope-lock: declines anything off Neo-setup in one line
- F8 knowledge: 10-Jul guide + comprehensive DGX + Strix tooling + SDK/repo/HF sources + web-search
- F9 ZERO Claude / third-party branding (splash, help, output, package name)
- F10 Neo-only brain; "Neo unreachable → escalate", no silent fallback

**Quality bar (anti-bug)**
- Q1 the read-only allowlist is SECURITY-critical → unit-tested + RED-TEAMED: attempts to make it run `rm` / expose a port / restart / write MUST be refused. Tested, not assumed.
- Q2 scenario suite of real failures run END-TO-END, each must diagnose correctly:
  203/EXEC · CPU-only build (no `-ngl` accel) · missing `[Install]` · wrong model path ·
  `:8001/health` down · gateway missing · tunnel down · a regulated-data prompt (must escalate).
- Q3 fails LOUD, never fake success.
- Q4 anything deferred is stated here, never silently skipped.
- Q5 verified end-to-end on THIS box (Strix Halo). DGX-specific path verified structurally + Zach's final 2-min run at CF (can't truly test a DGX I don't have — stated plainly).

---

## 3. Phases

### Phase 0 — Engine: agentic step + CLI auth  ✅ DONE
- `src/agent.py` — stateless `agent_step(history)` (reuses NEO_SYS persona + KB + guardrails + escalation), returns next read-only tool or final. ✅ compiles + JSON-parse smoke-tested.
- `serve.py` — `/agent-step` route + `Authorization: Bearer` auth (per-user `DENEB_CLI_TOKENS`, separate from the site origin secret). ✅ compiles.
- **Acceptance:** POST `/agent-step {history}` returns a valid action/final. *(verify: run engine with NEO_ALLOW_OPEN=1, curl it — Phase 6.)*

### Phase 1 — Knowledge refresh  🔨
- 1a Rebuild KB off the **10-Jul** guide (convert `neo-onprem-guide.html` → md; keep ONPREM Strix coverage). ⚠ current KB is 9-May = stale.
- 1b Curated **DGX Spark** reference (DGX OS 7 · CUDA 13.x · 580-open driver · GB10 Blackwell compute 12.1 · ARM64 · nvidia-smi/NGC/TensorRT/NIM).
- 1c Curated **Strix Halo** reference (ROCm/Vulkan · kyuz0 toolboxes · Lemonade).
- 1d Curated **ecosystem sources** (canonical NVIDIA/AMD SDK pages · llama.cpp / vLLM / ollama · Hugging Face + GGUF orgs).
- 1e SECRET SCAN every new KB doc before allowlisting (no tailnet/UIDs/keys/CF-confidential — the KB is customer-safe).
- **Acceptance:** `knowledge.py` retrieval returns the 10-Jul steps for a DGX query; secret scan clean.

### Phase 2 — Client core  ⬜
- 2a package skeleton `deneb/` (pyproject, entry point `deneb`, no third-party branding).
- 2b `deneb auth --token` (store chmod-600 `~/.config/deneb/token`), `deneb logout`, config for engine URL.
- 2c the agent loop (POST /agent-step, hold history, render).
- 2d **read-only tool executor + ALLOWLIST** (the security core): `run` only matches an allowlist of read-only commands (journalctl, systemctl status, ls, cat, find, grep, nvidia-smi, rocminfo, curl localhost health, …); everything else refused with a logged reason. `read_file`/`list_dir` path-guarded.
- 2e Deneb TUI (spinner, tool-call trace "▸ checking …", final answer + sources panel).
- **Acceptance:** `deneb "why won't neo start"` runs a real multi-step loop reading real logs and produces a grounded diagnosis (Phase 6 verifies).

### Phase 3 — `deneb check` (the "am I done?" scan)  ⬜
- Structured completeness scan driven by the Neo reference: each component → a read-only probe → ✓/✗ + the exact fix for any ✗. Ends with "you're done" or the gap list.
- **Acceptance:** on this Strix box it correctly reports the real state of the running Neo stack.

### Phase 4 — Image paste → OCR  ⬜
- `deneb` accepts a pasted screenshot / image path → sends to the engine → Surya OCR (reuse `vlm.py`) → transcription folded into the loop.
- **Acceptance:** a journalctl screenshot → correct OCR + diagnosis (reuse the web-Deneb Surya path already proven).

### Phase 5 — Packaging + installer  ⬜
- `pipx`-installable; one-line `curl -fsSL https://deneb.altronis.sg/install | sh` (served from your domain). ARM64 + x86 clean.
- **Acceptance:** fresh-venv install → `deneb --version` → `deneb auth` → `deneb check` works.

### Phase 6 — Test · red-team · verify-on-box  ⬜
- Q1 allowlist unit tests + red-team; Q2 scenario suite end-to-end; Q5 real run on this box. Fix until green. `qc`-style gate must pass before "done".
- **Acceptance:** all DoD boxes ticked + the suite green + a clean run recorded.

---

## 4. What I need from Zach
- A token to wire in (I'll set up issuance; you paste it once).
- Confirm the CLI points at `deneb-engine.altronis.sg` (or a dedicated `deneb` route).
- The final 2-min run on a real DGX at CF (the one thing I can't test here).

## 5. Deferred out of v1 (explicit, not silent)
- ⏸ Per-user token issuance UI (v1 = a token you set in `DENEB_CLI_TOKENS`).
- ⏸ Offline/bootstrap mode (you parked it — assume internet).
- ⏸ CF self-serve / multi-user entitlement (v1 = just you).
- ⏸ Write/apply-fix actions (v1 = read-only + it tells you the fix to run).
