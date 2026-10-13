<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/vllm-gfx1151.md, copied 2026-10-13 at commit 2cba0ba. -->
# vLLM on Strix Halo (gfx1151)

**It works.** Serving and generating, verified end to end on a Ryzen AI MAX+ 395
(Radeon 8060S, gfx1151) using the
[lemonade-sdk/vllm-rocm](https://github.com/lemonade-sdk/vllm-rocm) qualified
bundle. That project lists gfx1151 as a supported target, auto-builds for it
daily, and gates releases on a hardware qualification run — so when it does not
start, the fault is almost always in how it was launched, not in the hardware.

Everything below was measured on 2026-08-03 with
`Qwen3.6-27B-AWQ-INT4` (19.05 GiB of weights) at `--max-model-len 16384`.

## Measured throughput

Single-stream numbers are the wrong way to judge vLLM. Its entire reason to
exist is continuous batching, so measure aggregate throughput under concurrency:

| concurrency | aggregate tok/s | per-user tok/s | p100 latency |
|---|---|---|---|
| 1  | 5.66  | 5.66 | 22.6 s |
| 8  | 38.6  | 4.84 | 26.5 s |
| 16 | 71.3  | 4.46 | 28.7 s |
| 32 | 111.2 | 3.48 | 36.8 s |

32/32 requests succeeded. vLLM's own logger peaked at 128.0 tok/s with 32
running. That is **19.6x aggregate throughput for 32x the users**, while each
individual user drops only from 5.7 to 3.5 tok/s.

**These are floor numbers.** The run had `--enforce-eager`, which vLLM confirms
in the log as *"Enforce eager set, disabling torch.compile and CUDAGraphs"* —
so both graph capture and compilation were off. See "Why enforce-eager" below.

### Sanity-check them against the roofline

Do not trust a throughput number you have not bounded. Strix Halo has ~256 GB/s
of memory bandwidth; decoding reads the whole weight set once per step:

```
20.5 GB of weights / 256 GB/s  = 12.5 decode steps/s
  concurrency 1  ceiling: 12.5 tok/s   measured   5.66  (45% of roofline)
  concurrency 32 ceiling: 400  tok/s   measured 111.2   (28% of roofline)
```

Both sit in a consistent efficiency band well under the physical limit, which is
what eager mode should look like. A number *above* the roofline would mean the
measurement was wrong, not that the hardware was fast.

### What this is and is not good for

- **Not a daily driver.** llama.cpp serving a 35B-A3B MoE with MTP does 66-86
  tok/s single-stream on the same box. For one developer coding, vLLM here is
  roughly 10x worse and always will be — a 27B model activates ~9x the
  parameters per token that a 3B-active MoE does.
- **Good for many concurrent users**, which is the case llama.cpp with
  `--parallel 1` does not serve at all: ~30 users at 111 tok/s aggregate on one
  desktop-class box.
- **Watch the per-user figure when promising anything.** At 32 concurrent, a
  500-token answer takes ~2.4 minutes. That is fine for async/batch document
  work and poor for live interactive chat.

## Launching it

See `bin/vllm-serve-strix.sh` in this repo. The four things that matter:

### 1. Launch through the bundle's own `bin/vllm-server`

**Never call `bin/python3 -m vllm.entrypoints.openai.api_server` directly.**
`bin/vllm-server` is not a convenience wrapper, it is the supported entrypoint,
and it exports four things nothing else does:

| Export | Why it matters |
|---|---|
| `LD_LIBRARY_PATH` → bundled `_rocm_sdk_*/lib`, `torch/lib` | the ROCm runtime the bundled torch links against |
| `PYTHONPATH` → `_rocm_sdk_core/share/amd_smi` | the real amdsmi; without it vLLM cannot detect ROCm |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | the ROCm flash-attention path |
| `CC` → bundled `llvm/bin/clang` | **Triton needs a working compiler to JIT its kernels** |

Bypassing the shim manufactures four convincing but fake "packaging bugs":

1. no amdsmi → `Failed to infer device type`
2. `ModuleNotFoundError: flash_attn_2_cuda` (the bundled flash_attn is the CUDA build)
3. slow, wrong-library-path model loads
4. `EngineDeadError` the instant Triton tries to compile `kernel_paged_attention_2d`

Number 4 is the expensive one. It reads exactly like "gfx1151 has no ROCm paged
attention kernel" and sends you into upstream issue threads. It is not a missing
kernel. It is a missing `CC`.

**Nothing here needs compiling or patching.** The bundle is prebuilt and is used
exactly as shipped — no source changes, no rebuilt wheels, no custom container.
Both hand-fixes that seemed necessary while bypassing the shim were confirmed
unnecessary once under it:

```sh
# with the shim's env applied, the shipped packages import fine:
#   flash_attn   IMPORT OK   2.8.3
#   amdsmi       IMPORT OK   26.5.0
```

So do not write an amdsmi shim, and do not move `flash_attn` aside. Both are
symptoms of the wrong launcher, not of a broken bundle.

The whole sweep was then re-measured with `flash_attn` restored to shipped state
and launched via `bin/vllm-serve-strix.sh`, to confirm the published numbers did
not depend on that rename. They do not — every point landed within noise:

| concurrency | flash_attn moved aside | flash_attn present | delta |
|---|---|---|---|
| 1  | 5.66  | 5.63   | -0.6% |
| 8  | 38.6  | 39.74  | +3.0% |
| 16 | 71.3  | 70.95  | -0.5% |
| 32 | 111.2 | 110.83 | -0.3% |

The engine log confirms the restored package is genuinely in use
(`Using Flash Attention (Triton backend) for ViT model on RDNA`), so this is a
real comparison rather than two runs of the same configuration.

### 2. `--gpu-memory-utilization` is a fraction of TOTAL memory

Not of *free* memory. On a 128 GB box already running llama-server on :8001, a
judgment gate on :8005 and ComfyUI on :7860, asking for `0.55` (= 70 GiB) claims
memory that does not exist. The symptom is not a clean OOM — it is thrashing
that looks like slow compilation:

| gpu-memory-utilization | other services resident | "Model loading took" |
|---|---|---|
| 0.55 (70 GiB) | yes | 311.9 s |
| 0.35 (45 GiB) | yes | 106.0 s |

Same 19.05 GiB of weights both times. Check free GTT before choosing:

```sh
awk '{print int($1/1048576)" MiB GTT in use"}' /sys/class/drm/card1/device/mem_info_gtt_used
```

### 3. Hybrid models need `--max-num-seqs` lowered

Qwen3.6 uses GDN linear attention, so it is a hybrid — **every decode sequence
needs its own Mamba state block**. Those blocks come out of the same budget you
just capped, and vLLM's default `max_num_seqs` is 1024:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (421).
Each decode sequence requires one Mamba cache block, so CUDA graph capture
cannot proceed. Please lower max_num_seqs to at most 421 or increase
gpu_memory_utilization.
```

This fires **after** a completely successful load — weights, compile, warmup and
KV cache sizing all pass first, so it looks like a late crash rather than a
config error. `--max-num-seqs 256` clears it. Dense models will not hit this.

### 4. Why `--enforce-eager`

The phase immediately after that assert is HIP graph capture, and
[vllm-project#32180](https://github.com/vllm-project/vllm/issues/32180) reports
graph capture hitting driver-level timeouts on gfx1151. On a box where the same
GPU is serving other things, a driver hang takes them all down with it.

So: get a serving endpoint first with `--enforce-eager`, and treat graph capture
as a separate, deliberate experiment rather than something you discover at 1am.
The cost is real throughput — the numbers above are a floor — but the failure
mode you are avoiding is a hard hang, not a slow start.

## What a healthy startup looks like

With `--enforce-eager` on a warm Triton cache, ~7.5 minutes to first response:

| phase | time |
|---|---|
| weights read from disk | 17.6 s |
| model loading total (19.05 GiB) | 106.0 s |
| profiling / warmup / Triton JIT | ~5.5 min |
| KV cache sized (22.03 GiB, 314,026 tokens) | — |
| `Application startup complete` → serving | 441 s from launch |

Triton caches compiled kernels in `~/.triton/cache` keyed by (kernel, shapes,
arch), so restarts reuse them. A vLLM, Triton or ROCm version change invalidates
the cache and you pay the compile again.

## Gotchas that cost real time

- **GGUF will not load.** `ValueError: GGUF model with architecture qwen35 is
  not supported yet` — vLLM's GGUF loader does not know Qwen3.5/3.6. Use an HF
  format model; AWQ-INT4 is the best-supported 4-bit quant on ROCm.
- **`~/.local/lib/python3.14` may hold a CUDA torch** that shadows the bundled
  ROCm one (`torch.cuda.is_available() == False`). Set `PYTHONNOUSERSITE=1`.
- **`tar` may not preserve execute bits** on the extracted bundle. `chmod +x bin/*`.
- **`pkill -f vllm` kills your own shell.** The pattern matches the command line
  of the very shell running it. Bracket it: `pkill -9 -f '[v]llm.entrypoints'`.
- **The engine subprocess is named `VLLM::EngineCore` — uppercase.** A lowercase
  pattern misses it, so a corpse survives every "fresh start" and holds tens of
  GiB against the next run. Kill both, and confirm GTT actually drops:

```sh
pkill -9 -f '[v]llm.entrypoints'; pkill -9 -f '[V]LLM::EngineCore'
awk '{print int($1/1048576)" MiB GTT"}' /sys/class/drm/card1/device/mem_info_gtt_used
```

## Verifying you are really running under the shim

Do not trust the script — read the live process:

```sh
P=$(pgrep -f '[v]llm.entrypoints' | head -1)
tr '\0' '\n' < /proc/$P/environ | grep -E '^(CC|FLASH_ATTENTION_TRITON_AMD_ENABLE)='
```

`CC` unset means you are not under the shim, whatever the script says. This
check exists because a heredoc that silently failed to write once left the old
config running while the new one was being credited for the result.

## Diagnosing "is it hung?"

Measure a rate, not a state. `ps` showing the process alive proves nothing, and
sampling the **parent** `api_server` is misleading — it idles near 0% while the
`VLLM::EngineCore` child does the work.

```sh
E=$(pgrep -f '[V]LLM::EngineCore' | head -1)
A=$(awk '{print $14+$15}' /proc/$E/stat); sleep 20
B=$(awk '{print $14+$15}' /proc/$E/stat)
echo "+$((B-A)) jiffies / 20s   # 100 = one full core"
```

Thousands of jiffies means it is compiling. Zero means it is actually stuck.
