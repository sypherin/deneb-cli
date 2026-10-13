<!-- Source: github.com/sypherin/strix-halo-setup (MIT, (c) 2026 Zachary Aw), file docs/unsloth-rocm-gfx1151.md, copied 2026-10-13 at commit 2cba0ba. -->
# Fine-tuning on Strix Halo: Unsloth on ROCm (gfx1151)

Verified on a Ryzen AI Max / Strix Halo box, 128 GB unified memory, Fedora 43.
Everything below was run on the machine, not read from docs.

## Install

Unsloth shipped official AMD/ROCm support on 2026-07-20 (v0.1.50-beta); gfx1151 is
explicitly supported. Install into an isolated prefix rather than the system Python:

```bash
curl -fsSL https://unsloth.ai/install.sh -o install.sh
# read it first: it is AGPL, self-contained, and its apt/sudo path is Debian-only
UNSLOTH_PYTHON=3.12 UNSLOTH_SKIP_AUTOSTART=1 NO_COLOR=1 sh install.sh
```

- `UNSLOTH_PYTHON=3.12` matters. Fedora 43 ships Python 3.14, which is too new for the
  ROCm torch wheels; the installer builds its own uv venv at `~/.unsloth/studio/`.
- `UNSLOTH_SKIP_AUTOSTART=1` keeps it from launching a server at install time and
  competing with anything already holding the GPU.
- Footprint is about 9.4 GB.

For scripted training, call the venv interpreter directly instead of activating:

```bash
~/.unsloth/studio/unsloth_studio/bin/python train.py
```

## Verified stack

| Component | Version |
|---|---|
| torch | 2.11.0+rocm7.13.0 |
| HIP | 7.13.99004 |
| system ROCm | 7.2.1 |
| transformers | 5.14.1 |
| peft | 0.18.1 |
| unsloth | 2026.7.4 |
| device 0 | Radeon 8060S Graphics (gfx1151) |

`torch.cuda.is_available()` is `True` and `import unsloth` succeeds on this stack.

## Three caveats

1. **Default Studio port is 8888**, which collides with plenty of self-hosted tooling.
   Launch with `unsloth studio -p 8899` if 8888 is taken.
2. **No flash-attn wheel for gfx1151.** It is optional; training runs on the prebuilt
   ROCm kernels without it. Don't waste an afternoon compiling it.
3. **`Python.h` missing** makes a small `hip_utils.c` JIT helper fail to compile at
   import. Non-fatal - import and training both work, a compiled fast path is skipped.
   Install the dev headers for the venv's Python 3.12 if you care about the last few
   percent.

## LoRA on a multimodal base (Gemma 4 E4B)

This is the part that costs a day if you discover it the hard way. Gemma 4's vision and
audio towers wrap their projections in `Gemma4ClippableLinear`, which peft cannot inject
into. Unsloth's fast path also could not load this base for training. Two consequences:

**Scope `target_modules` to the text backbone by regex:**

```python
LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
)
```

**Skip TRL's multimodal collator.** A plain `transformers.Trainer` with
`DataCollatorForSeq2Seq` over text-only inputs trains fine:

```python
model = getattr(transformers, "AutoModelForImageTextToText").from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
model.config.use_cache = False
model.enable_input_require_grads()      # required before gradient checkpointing on a PEFT wrap
```

Reference numbers: 156 examples, batch 1, grad-accum 8, 3 epochs, lr 2e-4, bf16,
gradient checkpointing - 60 optimizer steps at ~6.4 s/step, about 6.5 minutes wall.

## Merge and convert

`merge_and_unload()` **does** write back through the `Gemma4ClippableLinear` wrappers.
That was the open question; it works, and returns a `Gemma4ForConditionalGeneration`
you can `save_pretrained()`.

llama.cpp converts the result without special flags:

```bash
python convert_hf_to_gguf.py <merged-dir> --outfile model-Q8_0.gguf.partial --outtype q8_0
```

Its converter registers `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`,
`Gemma4AssistantForCausalLM` and `Gemma4UnifiedForConditionalGeneration`, so claims that
llama.cpp lacks `gemma4` support are out of date. Check the converter's registry rather
than the changelog.

## Unified memory discipline

On Strix Halo the GPU allocates from the same 128 GB as everything else, so a training
or conversion job competes directly with any resident model server. Wrap heavy jobs in a
guard that enforces four rules:

1. **Stop the GPU consumers first** - image generation, vision models, anything holding
   GTT - and restore them on exit via a trap, so a failed job doesn't leave them down.
2. **Cap the job.** Give it a hard memory ceiling:
   `-p MemoryMax=60G -p MemorySwapMax=2G`.
3. **Detach it properly** - see the next section. This is the one that bit us.
4. **Sentinel on the right metrics.** Poll `MemAvailable` and SIGKILL the job below a
   floor (we use 12 GiB) - but `MemAvailable` alone is not enough. Dirty page cache
   still counts as *available*, so a box can be completely stalled flushing a multi-GB
   GGUF while `MemAvailable` looks healthy. Watch `Dirty` + `Writeback` from
   `/proc/meminfo` as well.
5. **Write big artifacts to `<out>.partial` and `mv` on success.** A multi-GB GGUF write
   that dies partway otherwise leaves a truncated file sitting at the real filename,
   looking perfectly valid to anything that opens it.

One more on point 5: check free space on the filesystem you are actually writing to.
Model directories are often symlinks to a second drive, so `df $HOME` can report plenty
of room while the target volume is full.

## Long jobs die the moment you background them

Symptom: the training run dies after ~25 seconds, every time, but only when launched
as a background task. Memory is nowhere near the cap. No OOM kill, no `systemd-oomd`,
nothing in the journal but systemd's neutral `Consumed 27.148s CPU time, 7.7G memory
peak`. Run the identical command in the foreground and it completes.

The cause is `--scope`. A scope is **not** a detached job - systemd only registers the
cgroup, while the process stays a child of the shell that called `systemd-run`. Kill or
reap that shell and the whole tree goes with it. Any agent harness, CI runner, or
terminal multiplexer that cleans up background tasks will take your training run with
it.

Use a transient **unit** instead, which the user manager forks and owns:

```bash
# dies with the calling shell
systemd-run --user --scope -p MemoryMax=60G ./long-job.sh

# survives it
systemd-run --user --unit=my-train -p MemoryMax=60G ./long-job.sh
```

Two details worth copying:

- **Skip `--collect`.** It garbage-collects the unit on exit, which throws away the very
  thing you need: `systemctl --user status my-train` showing the real exit code or
  signal. Without it you cannot tell a crash from a kill - systemd's `Consumed ...` line
  is identical for both. Run `systemctl --user reset-failed my-train` before relaunching.
- **Don't set `Type=oneshot`** without `TimeoutStartSec=infinity`. For oneshot the start
  timeout covers the entire run, so the default 90 s will kill a long job outright. The
  default `Type=simple` has no such trap.

Nesting works, so an existing guard script that uses `--scope` internally does not need
rewriting - just launch the guard itself as a transient unit and the inner scope inherits
the protection.
