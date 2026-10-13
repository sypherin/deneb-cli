<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/vulkan-pinned-build.md, copied 2026-10-13 at commit 2cba0ba. -->
# Pin the dated Vulkan tag: a 2.3x token-generation regression rides the floating one

**Credit where it belongs:** the container images, the RADV performance work and the
benchmark methodology below are all [kyuz0](https://github.com/kyuz0)'s, from
[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).
This page adds nothing to that work. It records one operational lesson we learned the
expensive way, so other Strix Halo owners don't repeat it.

## The lesson

Use the **date-stamped** tag. The floating `:vulkan-radv-performance` tag moves, and at
least one upstream llama.cpp build that landed on it cost **2.3x on token generation**.

| tag | llama.cpp build | tg (see setup below) |
|---|---|---|
| `vulkan-radv-performance_20260804T174052` | **10283** (`b7b85da9c`) | **19.15 t/s** |
| `vulkan-radv-performance_20260808T115141` | 10346 (`86e3f34fc`) | 7.79 t/s |
| `vulkan-radv-performance` (floating) | whatever is newest (10346 when we checked) | varies |

Build numbers verified by running `llama-server --version` inside each image, not read
off a changelog.

```bash
# the good one
podman pull docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv-performance_20260804T174052
podman run --rm --entrypoint /bin/bash \
  docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv-performance_20260804T174052 \
  -lc 'llama-server --version'
# => version: 10283 (b7b85da9c)
```

The regression is progressive across the 10283 → 10346 range rather than a single bad
commit, so bisecting to "the" offending change did not produce a clean answer.

## What was measured

- **Hardware:** AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), 128 GB unified, Fedora 43
- **Model:** DeepSeek-V4-Flash `UD-IQ3_XXS` (unsloth quant)
- **Flags:** kyuz0's own benchmark flags, `-p 0 -n 128 -ub 2048 -b 2048 -fa 1 -ngl 99 -r 5`
- **Kernel cmdline:** `amd_iommu=off` (worth 5-11% here, reproduced on two builds)

18.28 t/s on 2026-08-08 and 19.15 t/s on 2026-08-09 on the same pinned image; the delta
is the `iommu=pt` → `amd_iommu=off` change, not a different container.

Numbers are from this one machine and this one quant. Treat them as a direction to check,
not a spec.

## Ruled out

Things we tested that did **not** explain the gap, recorded so nobody re-spent the time:
MMID flags (helped prefill ~17%, tg 0%), Mesa/device-line selection, coopmat (already on),
clock forcing, ROCm instead of Vulkan (-12%), AMDVLK, submit batching, and power/board
limits. On the last of those, prefill was actually ~11% *ahead*, so the machine was not
throttling.

## Keep your own copy

DockerHub tags are not archival. A dated tag survived an upstream force-push once, which
is luck rather than a guarantee, and a local `podman image prune` will remove your only
copy just as effectively as an upstream deletion.

```bash
podman save -o kyuz0-vulkan-radv-performance_20260804T174052_build10283.tar \
  docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv-performance_20260804T174052
# restore:  podman load -i <that file>
```

Write it to storage that is not the same physical disk as your container store, and write
to `<name>.partial` then rename on success so a truncated file can never look complete.

## Please support the upstream project

If this saved you time, the value came from
[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).
Star that repo, file issues there, not here.
