<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/muse-glimmer-30b-strix.md, copied 2026-10-13 at commit 2cba0ba. -->
# Muse-Glimmer-30B on Strix Halo: the fast config, and two optimisations that lost

Muse-Glimmer-30B is a 30B **vision** LLM (chat + image input in one model). On a Strix Halo
(Ryzen AI MAX+ 395, gfx1151) the fastest way to run it is a **custom ROCm-FP4 build**, not the
mainline Vulkan path most Strix guides recommend. This page records the config, the builds, and
the two "obvious" optimisations we A/B-tested that both turned out **slower**, so you don't
repeat them.

## TL;DR

- **~26.5 tok/s** generation on a dense 30B is close to the memory-bandwidth ceiling (~256 GB/s
  ÷ ~15 GB read/token). Spec-decode (dflash) lifts it above the naive single-token ceiling.
- The **FP4 quant on ROCm beats the Q4_K quant on Vulkan** here, because the model is
  bandwidth-bound and FP4 (15.2 GB) reads ~10% fewer bytes/token than Q4_K (16.76 GB). Vulkan's
  faster kernels don't make up the difference.
- **Measure in the condition you actually serve in.** An A/B without the reasoning setting we run
  with looked like a +9% win; with it, the same change was a loss. See the FP8-drafter row.

## The winning config (FP4 on ROCm)

Model + drafter + vision projector are the ROCm-FP4 builds (`meta-models/Muse-Glimmer-30B`,
quantised to a Strix FP4 recipe; GGUFs at `vmlinux/Muse-Glimmer-30B-ROCmFPX-GGUF`).

```bash
llama-server \
  -m   ~/models/muse-glimmer/Muse-Glimmer-30B-ROCmFP4.gguf \
  -md  ~/models/muse-glimmer/Muse-Glimmer-30B-DFlash-ROCmFP4.gguf \
  --mmproj ~/models/muse-glimmer/mmproj-Muse-Glimmer-30B-BF16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 6 \
  --ctx-size 65536 -ngl 99 -ngld 99 -fa 1 --no-mmap --jinja \
  -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
  --host 0.0.0.0 --port 8022
```

Notes: the **FP4-ROCmFP4 drafter** (1.39 GB) is the one to use, see the A/B. `-fa 1` + q8_0 KV
(main and draft) + `-ngl/-ngld 99` all-on-GPU. Vision needs the `--mmproj` projector; it stays
BF16.

## The builds (for reproducibility)

FP4 is a ROCm-only format, so the model only loads on a fork that carries both the `muse-glimmer`
arch and the FP4 kernels: **[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)**,
tested at commit `00d5452` (ggml `0.11.1`).

```bash
# ROCm build, this is what runs the FP4 model at 26.5 t/s
cmake -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-server -j

# Vulkan build of the SAME fork, an escape hatch, but SLOWER (see A/B).
# Loads the normal Q4_K quant (FP4 won't run on Vulkan). No manual port needed;
# the fork already has the arch; you're just swapping the compute backend.
cmake -B build-vulkan -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build-vulkan --target llama-server -j
```

Mainline llama.cpp (any Vulkan release binary) rejects the model outright:
`error loading model: unknown model architecture: 'muse-glimmer'`. Only the fork above knows it.

## A/B results (gfx1151, 128 GB, reasoning_strength=low, warm, 3-run avg)

Same 200-word prompt, `max_tokens≈230`, `cache_prompt=false`, `reasoning_strength:low` (the
setting we actually serve with). Numbers are the server's own `predicted_per_second`.

| config | quant | backend | drafter | **tok/s** | draft accept |
|---|---|---|---|---:|---:|
| **winner** | FP4 (15.2 GB) | ROCm (fork) | **FP4 dflash** (1.39 GB) | **26.5** | ~29% |
|  | FP4 (15.2 GB) | ROCm (fork) | FP8 dflash (2.65 GB) | 23.6 | ~28% |
|  | Q4_K (16.76 GB) | **Vulkan** (fork) | Q4_K dflash | 24.4 | ~32% |

### Why the two "upgrades" lost

- **FP8 drafter** (higher-precision, built "for acceptance"): under a *raw* read (no reasoning
  setting) it looked like +9% (28 vs 26 t/s, 36% vs 29% accept). But we don't serve raw, with
  `reasoning_strength:low` the acceptance edge collapses to ~28% and the bigger drafter's slower
  draft step makes it a net **loss** (23.6 vs 26.5). Lesson: benchmark in-condition.
- **Q4_K on Vulkan**: Vulkan kernels are genuinely faster on Strix, but generation here is
  **memory-bandwidth-bound**, throughput is set by bytes read per token. Q4_K is ~10% bigger
  than FP4, and that outweighs the kernel speedup: **24.4 < 26.5**. The smaller quant wins.

## When Vulkan would win

If a future Muse-Glimmer ships **without** an FP4 quant, or you move off the ROCmFPX fork onto
mainline, the Vulkan build above is the only path, and then Q4_K-on-Vulkan (24.4 t/s) is simply
the config. It loads and runs the dflash spec-decode cleanly; it's just second-best while FP4
exists.

**If you want raw speed on Strix, don't reach for Muse-Glimmer at all**, it's the vision model.
A 30B **MoE** (e.g. Qwen3.6-35B-A3B, ~3B active/token) does ~78 t/s on the Vulkan build because it
reads far less per token. Muse-Glimmer's ~26 t/s is the price of dense chat + vision in one model.
