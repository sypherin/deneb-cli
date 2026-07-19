"""deneb.models_catalog — the curated local-LLM model matrix (FIT-01).

PURE DATA. This module holds the CATALOG: popular local-LLM GGUFs (Qwen coder /
Qwen3.x MoE, Llama 3.x, Gemma, Mistral/Mixtral, Phi, DeepSeek-Coder, and a couple
of vision models), each described as a Model with its params, architecture
(dense vs MoE + active params), the quants it ships (each with an approximate
memory-need in MB), capability tags, and its Hugging Face GGUF repo. The pure
`fits()` math in deneb.fit consumes these structs; recommend() (Plan 02) ranks
over them.

⚠ HONESTY — these numbers are CURATED BEST-EFFORT APPROXIMATIONS, NOT measured
   facts and NOT Hugging Face-verified:
     - Every `size_mb` is a CONSERVATIVE (rounded-up) approximation of the GGUF
       weight size, chosen so the fit math errs toward "does not fit" rather than
       OOMing the user's box. They are NOT benchmarks.
     - Every `gguf_repo` id is best-effort curated and MUST be verified before any
       download (the Deneb Rule: surface the caveat, don't pretend).
     - Online verification of repo ids + real file sizes is DEFERRED to a later
       phase (LEVELUP Phase 6). Until then, treat this matrix as a static v1 guess,
       not authoritative data.
     - Vision models additionally need an `mmproj` projector file alongside the
       main GGUF — recorded in each vision model's `notes` so the later setup
       advisory can surface it.

`bpw` (bits/weight per quant) is used only for deterministic quant ordering and the
integrity test's size-vs-bpw monotonicity check — it is not itself a fit input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── the fixed capability vocabulary (tags are drawn only from this set) ───────
CAPABILITIES = ("coding", "vision", "chat", "general")

# ── per-quant bits/weight reference (ordering + monotonicity only) ────────────
# Q4_K_M < Q5_K_M < Q6_K < Q8_0 — the fit math relies on size_mb rising with bpw.
_BPW = {"Q4_K_M": 4.85, "Q5_K_M": 5.6, "Q6_K": 6.6, "Q8_0": 8.5}


@dataclass
class Quant:
    """One quantization of a model. size_mb is a conservative approx GGUF weight
    size in MB (see module docstring); bpw is the reference bits/weight."""
    name: str
    size_mb: int
    bpw: float


@dataclass
class Model:
    """One catalog model.

    architecture    ∈ {"dense", "moe"}.
    active_params_b — MoE active-expert params in billions; None for dense models.
                      (This is what the speed tier keys off for MoE.)
    capabilities    — subset of CAPABILITIES.
    quants          — available quants, ordered ascending by bpw.
    gguf_repo       — best-effort HF GGUF repo id (verify before download).
    notes           — free-text caveat (e.g. the vision-model mmproj requirement).
    """
    name: str
    params_b: float
    architecture: str
    active_params_b: Optional[float]
    capabilities: list
    quants: list
    gguf_repo: str
    notes: str = ""


def _quants(sizes: dict) -> list:
    """Build a bpw-ordered list[Quant] from a {quant_name: size_mb} mapping, so each
    Model literal stays flat (sizes inline) without repeating the bpw constants."""
    return [Quant(name=n, size_mb=sizes[n], bpw=_BPW[n])
            for n in sorted(sizes, key=lambda k: _BPW[k])]


# ── THE CURATED CATALOG (FIT-01) — one Model literal per model, quants inline ──
# Sizes/repos are conservative curated approximations, NOT HF-verified (see docstring).
CATALOG = [
    Model("Qwen2.5-Coder-7B-Instruct", 7, "dense", None, ["coding", "general"],
          _quants({"Q4_K_M": 4700, "Q5_K_M": 5400, "Q8_0": 8100}),
          "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"),
    Model("Qwen2.5-Coder-14B-Instruct", 14, "dense", None, ["coding", "general"],
          _quants({"Q4_K_M": 9000, "Q5_K_M": 10500, "Q8_0": 15700}),
          "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"),
    Model("Qwen2.5-Coder-32B-Instruct", 32, "dense", None, ["coding", "general"],
          _quants({"Q4_K_M": 20000, "Q5_K_M": 23300, "Q8_0": 34800}),
          "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"),
    Model("Qwen3.6-35B-A3B", 35, "moe", 3, ["coding", "chat", "general"],
          _quants({"Q4_K_M": 21500, "Q5_K_M": 25000, "Q8_0": 37500}),
          "Qwen/Qwen3-30B-A3B-GGUF",
          notes="MoE: 35B total, ~3B active experts — fast tok/s for its size."),
    Model("Llama-3.2-3B-Instruct", 3, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 2100, "Q5_K_M": 2400, "Q8_0": 3500}),
          "bartowski/Llama-3.2-3B-Instruct-GGUF"),
    Model("Llama-3.1-8B-Instruct", 8, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 4900, "Q5_K_M": 5700, "Q8_0": 8500}),
          "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
    Model("Llama-3.3-70B-Instruct", 70, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 42500, "Q5_K_M": 49900, "Q8_0": 74900}),
          "bartowski/Llama-3.3-70B-Instruct-GGUF"),
    Model("Gemma-2-9B-it", 9, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 5800, "Q5_K_M": 6600, "Q8_0": 9800}),
          "bartowski/gemma-2-9b-it-GGUF"),
    Model("Gemma-2-27B-it", 27, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 16600, "Q5_K_M": 19400, "Q8_0": 28900}),
          "bartowski/gemma-2-27b-it-GGUF"),
    Model("Mistral-7B-Instruct-v0.3", 7, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 4400, "Q5_K_M": 5100, "Q8_0": 7700}),
          "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
    Model("Mixtral-8x7B-Instruct-v0.1", 47, "moe", 13, ["chat", "general"],
          _quants({"Q4_K_M": 28400, "Q5_K_M": 33200, "Q8_0": 49600}),
          "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
          notes="MoE: 47B total, ~13B active (2 of 8 experts) per token."),
    Model("Phi-3.5-mini-instruct", 3.8, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 2400, "Q5_K_M": 2800, "Q8_0": 4100}),
          "bartowski/Phi-3.5-mini-instruct-GGUF"),
    Model("Phi-4", 14, "dense", None, ["chat", "general"],
          _quants({"Q4_K_M": 9100, "Q5_K_M": 10600, "Q8_0": 15600}),
          "bartowski/phi-4-GGUF"),
    Model("DeepSeek-Coder-6.7B-Instruct", 6.7, "dense", None, ["coding", "general"],
          _quants({"Q4_K_M": 4100, "Q5_K_M": 4800, "Q8_0": 7200}),
          "TheBloke/deepseek-coder-6.7B-instruct-GGUF"),
    Model("DeepSeek-Coder-33B-Instruct", 33, "dense", None, ["coding", "general"],
          _quants({"Q4_K_M": 19900, "Q5_K_M": 23500, "Q8_0": 35400}),
          "TheBloke/deepseek-coder-33B-instruct-GGUF"),
    Model("Qwen2.5-VL-7B-Instruct", 7, "dense", None, ["vision", "chat"],
          _quants({"Q4_K_M": 5000, "Q5_K_M": 5800, "Q8_0": 8300}),
          "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
          notes="Vision model — also needs its mmproj projector GGUF alongside the "
                "main weights."),
    Model("Llava-v1.6-Mistral-7B", 7, "dense", None, ["vision", "chat"],
          _quants({"Q4_K_M": 4400, "Q5_K_M": 5100, "Q8_0": 7700}),
          "cjpais/llava-1.6-mistral-7b-gguf",
          notes="Vision model — also needs its mmproj projector GGUF alongside the "
                "main weights."),
]


def by_capability(tag: str) -> list:
    """Return the catalog models whose capabilities include `tag` (e.g. 'coding',
    'vision'). Empty list for an unknown tag — never raises."""
    return [m for m in CATALOG if tag in m.capabilities]
