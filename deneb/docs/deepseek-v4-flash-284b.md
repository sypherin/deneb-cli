<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/deepseek-v4-flash-284b.md, copied 2026-10-13 at commit 2cba0ba. -->
# DeepSeek V4 Flash (284B) on Strix Halo — Vulkan, no CUDA

Running **DeepSeek-V4-Flash-0731, 284B parameters**, fully GPU-offloaded on a 128GB
Strix Halo mini PC (Ryzen AI Max 395 + Radeon 8060S iGPU) via llama.cpp's **Vulkan**
backend. No CUDA, no cloud, no API key. Measured 2026-08-01.

The short version: **it runs, at read-aloud speed, at full 131k context** — and it is
still not the model you want for spec-compliant code generation. Both halves matter.

## TL;DR — the config that works

```bash
llama-server -m DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf \
  -ngl 99 -c 131072 -ub 1024 -fa on \
  -np 1 --cache-prompt --jinja --no-warmup \
  --reasoning-budget 4096
# env: GGML_VK_PREFER_HOST_MEMORY=ON
```

Sampling per Unsloth's guidance for this model: `--temp 1.0 --top-p 1.0 --min-p 0.0`.

## Measured numbers

llama.cpp mainline (build fb30ba9), Vulkan/RADV on gfx1151, unsloth `UD-IQ2_XXS`
(84.62 GiB on disk, 284.33B params), full offload:

| test | t/s |
|---|---|
| pp512 | 130.17 |
| tg32 | 12.24 |
| pp512 @ 4k depth | 94.52 |
| tg32 @ 4k depth | 11.57 |

Generation decays only **~5%** from empty context to 4k depth. Sustained 11.8 t/s was
reproduced across four separate 500–717 second generations, so the figure is stable
rather than a lucky sample.

**Memory:** 97G used with the model resident and 131k context allocated, 26G free on a
128GB box. The 84.6 GiB of weights is irreducible at this quant — everything else is
rounding.

## Context: 131,072 tokens, first try

Full context loads with the config above. This surprised us — an earlier naive attempt
at 16k crashed, which made 16k look like a ceiling. It was not; see the allocation trap
below.

## Traps that cost a full day

**1. `llama-cli` in interactive mode is indistinguishable from a hung model.**
Run unattended with no TTY, it prints `>`, reads instant EOF, treats that as a finished
turn, prints `>` again, forever. We collected **2.6 GB of `>` characters** while
concluding "the model crawls at 1% GPU". It was never asked anything. Use `llama-bench`,
or `llama-cli -st` with stdin closed. If you see `>` repeated in a log, that is this bug.

**2. Vulkan refuses single allocations above ~5GB.**
A naive `-c 16384` asks for one 17.7 GB compute buffer and the server dies at startup
with `failed to allocate ... exceeds device buffer size limit`. The fix is not less
context — it is a smaller batch. `-ub 1024` (or lower) shrinks that buffer enough that
**131k context allocates fine**.

**3. It is a thinking model; unbudgeted reasoning eats the answer.**
With reasoning uncapped and a 6000-token output budget, it spent ~5,200 tokens planning
and emitted 366 characters of actual code. `--reasoning-budget` is mandatory. Measured
appetite: ~1,900 tokens for straightforward generation, ~3,200+ for debugging existing
code. 4096 is a reasonable default; the model self-limits below the cap when satisfied.

**4. `llama-server` defaults to 4 parallel slots — 4× the KV cache for nothing.**
Use `-np 1` for single-user work.

**5. Quantized KV cache halves throughput on Vulkan.**
`-ctk q8_0 -ctv q8_0` dropped generation from 11.8 t/s to **4.8 t/s** — dequantization
overhead on every attention op. It is a memory lever, not a speed lever. Only enable it
if you are actually memory-bound.

**6. Size your client timeout to your token budget.**
Raising `max_tokens` to 32,000 at ~5 t/s needs ~110 minutes. A 60-minute HTTP timeout
kills a perfectly healthy generation and looks like a model failure.

## Which quant fits 128GB

Unsloth ships the full ladder for this model:

| quant | size |
|---|---|
| UD-IQ1_S | 83 GB |
| UD-IQ1_M | 87 GB |
| UD-IQ2_XXS | 91 GB (84.6 GiB on disk — what these numbers use) |
| UD-IQ2_M | 91 GB |
| UD-Q2_K_XL | 97 GB |
| UD-IQ3_XXS | 103 GB |
| UD-Q4_K_XL | 155 GB |
| UD-Q8_K_XL | 162 GB |

On 128GB unified there is real headroom up to about **Q2_K_XL**. IQ3_XXS at 103GB sits
right at the edge once KV cache and compute buffers are added. Q4_K_XL and above are out
of reach on this class of machine.

Note the naming mixes two families: `IQ*` (IQ1_S, IQ2_XXS, IQ3_XXS) and `*_K_XL`
(Q2_K_XL, Q4_K_XL). There is no "IQ2_XL".

## Quality: the honest half

Scored against a strict local spec-compliance benchmark — a detailed Pac-Man
specification with an automated Playwright gate that checks a testability contract
before it will run any gameplay assertions.

**DeepSeek-V4 at IQ2_XXS failed the gate**, and the failure is instructive rather than
random. The spec required specific top-level state variables and function declarations
so the harness can reach live game state. The model produced exactly the right names —
then left them as empty stubs:

```js
let pacman, ghosts, maze, score, lives, powerMode, gameState, dotCount, totalDots;
// Top-level function declarations
function movePacman() {}
function moveGhost() {}
function checkGhostCollisions() {}
```

…and wrote the real implementations 120 lines further down. In JavaScript the later
declarations win, so the game largely still works — but the variables stay `undefined`
until `startGame()` runs, the contract probe throws, and the gate stops before testing
anything.

That is **spec-compliance as ritual**: pattern-matching the shape of a requirement
while missing its purpose.

### The quant experiment: does better preservation fix it?

Worth testing, because the failure above is a *comprehension* failure and comprehension
lives in attention. A second quant of the same model — same 2-bit routed experts, but
attention projections, shared experts and output kept at **Q8** — was run against the
identical spec and judge.

It behaved measurably differently:

| build | testability contract | checks passed |
|---|---|---|
| uniform 2-bit (IQ2_XXS) | **failed** — declared the names, stubbed the bodies | 0 of 14 reached |
| Q8 attention / shared / output | **passed** | 4 of 14 |

So better preservation of those layers *did* buy comprehension: the second build
understood what the contract was for, where the first only matched its shape. It did
not buy the ability to build a working game — the remaining ten failures cluster
entirely in the movement layer (nothing eats a dot, three of four ghosts never leave
the house), which the 2-bit routed experts evidently cannot hold together.

A useful corollary: **the thinking budget can eat the whole job.** At
`--reasoning-budget 4096` this model spent 48 minutes redesigning the maze in its
reasoning ("actually, let me make it simpler" — repeatedly) and truncated mid-file.
Cutting it to 2048 produced a complete 24KB file that reached every gate check. On a
thinking model, an over-generous reasoning cap is not a safety margin, it is a failure
mode.

### Scoreboard

Same specification, verified-identical prompt. All rows except the last are the
automated 14-check judge; the 35B row is a human-play verdict (see correction below):

| model | result | time |
|---|---|---|
| DeepSeek-V4-284B, uniform 2-bit | 0 of 14 checks reached | 44 min |
| DeepSeek-V4-284B, Q8-attention quant | 4 of 14 | 55 min |
| Qwen3.6-27B, reasoning-distilled | 6 of 14 | 10 min |
| Qwen3.6-35B-A3B, reasoning-distilled | playable first try (human-judged)* | 77 sec |

The 35B is roughly **43× faster** and produced the only build a human called playable.
That is the practical verdict.

\* **Correction (same day, after re-running the judge on the archived artifacts):** the
35B row was a human-play verdict, not the automated judge's. Its first-try build
predates the judge's testability contract and scores 0-of-14-reached on it (the game
plays, but exposes none of the required test hooks); the fix-rounds final scores
9 of 14. The 77-second figure is real, but the row is not same-judge comparable with
the rows above. For the record, the best *cold* score this judge has ever given is
6 of 14 at 68 t/s, by a Fable-5 distill of the same 35B base tested later the same day.

It does write large coherent programs: a complete self-contained Pac-Man (canvas, four
ghosts, collision, score, game loop, balanced braces, closing `</html>`) in ~12 minutes
at 8,434 tokens — playable structure, with real logic bugs in turn buffering and ghost
pathing.

## Why a smaller model beats it here

The useful mental model:

> **usable capability ≈ active parameters × bits per weight × task fit**

The comparison is not 35B versus 284B. It is *35B at ~4.5 bits, instruction-tuned* versus
*284B at 2.06 bits, reasoning-tuned*. Quantization damage is not linear — below roughly
3 bits, the first casualty is multi-step instruction-following, which is exactly what a
strict spec benchmark measures. And a mixture-of-experts model with ~3B active parameters
per token will comfortably out-run a dense model with far fewer total parameters.

The local-model scene fixates on the first term. On this hardware, the second and third
decide whether the thing is actually useful.

## Speculative decoding (DSpark): it works, but buys no speedup here

DeepSeek ships a **DSpark** block-parallel drafter for V4-Flash — it drafts a whole block
of tokens in one pass, the full model verifies them, and the accepted prefix is kept. On
an NVIDIA DGX Spark it is reported at roughly 1.9x generation speed. The question for this
box: does it carry to the Vulkan/RDNA path?

**It runs and accepts tokens — and the wall-clock does not move.** Measured 2026-08-08.

### Build

Mainline llama.cpp reads V4-Flash natively (`src/models/deepseek4.cpp`) **and** carries
the DSpark speculative path. There are two distinct spec types in
`common/speculative.cpp` — `draft-dflash` and `draft-dspark` — and DSpark needs
`draft-dspark`. Passing `draft-dflash` silently leaves `speculative: false` in `/slots`;
that wrong flag is the single biggest trap.

```bash
# mainline llama.cpp, commit 69bf643 (2026-08-08), Vulkan:
cmake -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

GGML_VK_PREFER_HOST_MEMORY=ON llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf \
  -md dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 \
  -ngl 99 -fa 1 -c 32768 --load-mode mmap --jinja
```

The draft must come from the **same quant family** as the target. Pairing the Unsloth
`UD-IQ2_XXS` target with an antirez-requantised draft gave **0.15%** acceptance — the two
distributions did not match. Unsloth's own `dspark-…-Q8_0.gguf` (10.1 GiB) against the
Unsloth target is the pairing that works. Confirm with `/slots`: `speculative: true`.

### Measured (gfx1151, Vulkan, Unsloth UD-IQ2_XXS target + Unsloth dspark-Q8_0 draft)

| config | draft acceptance | tg t/s |
|---|---|---|
| plain, no spec | — | ~9.5 |
| `draft-dspark --spec-draft-n-max 3` | 61.8% (mean 2.85 tok/draft) | 9.5 |
| `draft-dspark --spec-draft-n-max 5` | 46.3% (mean 3.31 tok/draft) | 9.6 |

Acceptance is genuinely high, and throughput does not change. The draft's own forward pass
plus the verify step cost about what the accepted tokens save, at every block setting.

### Why — and why this is *not* "Vulkan can't do speculative decoding"

The same box runs **Qwen3-35B-A3B with `--spec-type draft-mtp` at ~78 t/s**, where
speculative decoding pays off cleanly. The difference is the **draft's weight**: an MTP
head is essentially one extra layer, near-free to run; DSpark's draft is a full ~10 GiB
model, so its per-step cost is comparable to the target's on a bandwidth-limited iGPU. DGX
Spark's ~1.9x comes from CUDA's optimised speculative kernels and tensor cores making that
heavy draft and the batched verify nearly free; the Vulkan/RDNA path has no equivalent, so
the overhead cancels the acceptance gain. The acceptance and the flat throughput are
measured; the CUDA-vs-Vulkan explanation is the best-supported hypothesis, not a profiled
fact.

So DSpark here is a correctness success and a speed no-op: V4-Flash stays ~8–10 t/s with
or without it. For real speedup on Strix Halo, the lever is a small-MoE model carrying a
cheap MTP draft, not a heavyweight standalone drafter.

## Verdict

Worth running if you want a frontier-scale model that never leaves your machine and you
can work at ~12 t/s: hard analysis, overnight batch work, second opinions on things you
would not send to a cloud API. Not worth running as a daily coding driver, where a
well-preserved smaller model wins decisively. Speculative decoding (DSpark) does not change
this — it works but yields no speedup on the Vulkan path, as above.

Either way, a 284B model at full 131k context on a mini PC iGPU is a genuinely different
2026 than most people assume.
