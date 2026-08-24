"""deneb.services — the multi-service STACK: several models behind one authed endpoint.

`deneb setup <model>` answers "how do I run one model from the catalog". That is not the
question anyone actually has on these boxes. The real question is the one a private-LLM
deployment forces: an LLM, a vision model and an OCR service all running at once, on
different ports, reachable by a cloud app through a single authenticated endpoint, without
any of them being exposed directly.

Two things here are not expressible in the model catalog at all, which is why this module
exists rather than more entries in CATALOG:

  * Surya is OCR. It is not a llama.cpp chat model, and one of its two generations is not a
    GGUF at all - it runs as a container. A model catalog has no way to say that.
  * The gateway and the tunnel are not models. They are the part that makes the stack usable
    from outside, and the part that decides whether it is safe.

Commands here were read off a working deployment rather than composed from documentation.
Where something is NOT verified, the step says so in its own text; see the Surya 2 GGUF
conversion in particular.

PURE, like the rest of the advisory core: structs in, Step structs out. Imports no executor,
spawns nothing, prints nothing. THE DENEB RULE holds - Deneb tells, you run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_DASHES = ("—", "–")


def _plain(text: str) -> str:
    for d in _DASHES:
        text = text.replace(d, "-")
    return text


@dataclass
class ServiceStep:
    title: str
    what: str
    command: str = ""
    warnings: list = field(default_factory=list)
    verify: str = ""
    expect: str = ""


@dataclass
class Service:
    key: str
    label: str
    role: str
    port: int
    steps: list                       # list[ServiceStep]
    notes: str = ""


_LLAMA = "llama-server"
_SERVICE_WARNING = "starts a long-running server - it keeps running until you stop it."
_BIG_DOWNLOAD = "downloads several GB."

# ── the inference services ────────────────────────────────────────────────────
# Flags are the ones a working deployment actually runs, not defaults. The
# non-obvious ones are called out, because they are the difference between a
# server that starts and a server that is usable.

_LLM = Service(
    key="llm",
    label="Gemma 4 26B (text LLM)",
    role="chat / completion, the OpenAI-compatible endpoint the app talks to",
    port=8001,
    notes="the model most of the traffic hits; give it the speculative-decode draft model",
    steps=[
        ServiceStep(
            title="Fetch the model and its MTP draft",
            what=_plain(
                "Two files, not one. The second is the multi-token-prediction draft model "
                "used for speculative decoding - without it the server still runs, just "
                "markedly slower, so it is easy to skip and then wonder why throughput is "
                "poor."),
            command=("hf download unsloth/gemma-4-26B-A4B-it-GGUF "
                     "--include 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf' "
                     "--include 'MTP/mtp-gemma-4-26B-A4B-it-F16.gguf' "
                     "--local-dir ~/models/gemma-4-q4"),
            warnings=[_BIG_DOWNLOAD],
        ),
        ServiceStep(
            title="Serve it",
            what=_plain(
                "-ngl 99 puts every layer on the GPU. --spec-type draft-mtp with the draft "
                "model is the speculative-decode path. --host 0.0.0.0 is deliberate here "
                "ONLY because the gateway in front is what enforces auth; on a box with no "
                "gateway this must be 127.0.0.1."),
            command=(f"{_LLAMA} -m ~/models/gemma-4-q4/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf "
                     "-md ~/models/gemma-4-q4/MTP/mtp-gemma-4-26B-A4B-it-F16.gguf "
                     "--spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 "
                     "--ctx-size 131072 --port 8001 --host 0.0.0.0 -ngl 99 --mmap --jinja "
                     "--ubatch-size 1024 -fa 1 -ctk q8_0 -ctv q8_0 --cache-prompt"),
            warnings=[_SERVICE_WARNING,
                      "--host 0.0.0.0 exposes this port on every interface. Only do that "
                      "behind the gateway."],
            verify="curl -s localhost:8001/health",
            expect='{"status":"ok"}',
        ),
    ],
)

_VLM = Service(
    key="vlm",
    label="Qwen3-VL 32B (vision LLM)",
    role="images in, text out - screenshots, diagrams, photographed documents",
    port=8080,
    notes=("needs an mmproj projector file alongside the weights, or images are ignored. "
           "Alternative: if the main LLM is Qwen3.8-27B, it has native vision and doubles as "
           "the VLM, so this separate vision service is not needed."),
    steps=[
        ServiceStep(
            title="Fetch the weights AND the projector",
            what=_plain(
                "A vision model is two files: the weights and the mmproj multimodal "
                "projector. Start it without the projector and it loads happily, answers "
                "text, and silently cannot see images - which reads as a bad model rather "
                "than a missing file."),
            command=("hf download unsloth/Qwen3-VL-32B-Instruct-GGUF "
                     "--include 'Qwen3-VL-32B-Instruct-Q4_K_M.gguf' "
                     "--include 'mmproj-F16.gguf' --local-dir ~/models/qwen-vlm"),
            warnings=[_BIG_DOWNLOAD],
        ),
        ServiceStep(
            title="Serve it",
            what=_plain(
                "--no-mmap because the weights are pinned rather than paged, which matters "
                "on a unified-memory box. Context is deliberately small: vision tokens are "
                "expensive and a large window here costs memory the LLM needs."),
            command=(f"{_LLAMA} -m ~/models/qwen-vlm/Qwen3-VL-32B-Instruct-Q4_K_M.gguf "
                     "--mmproj ~/models/qwen-vlm/mmproj-F16.gguf "
                     "-ngl 99 --no-mmap --flash-attn on -c 8192 --ubatch-size 512 "
                     "--host 127.0.0.1 --port 8080 --temp 0.1"),
            warnings=[_SERVICE_WARNING],
            verify="curl -s localhost:8080/health",
            expect='{"status":"ok"}',
        ),
    ],
)

_SURYA = Service(
    key="surya",
    label="Surya 1 (OCR, container)",
    role="document OCR - layout, reading order, text extraction",
    port=8090,
    notes="not a GGUF and not llama.cpp: a Python library that you must wrap in an HTTP server",
    steps=[
        ServiceStep(
            title="Understand that Surya 1 ships no server",
            what=_plain(
                "This is the part that surprises people. Surya 1 is a Python LIBRARY - it "
                "gives you predictor classes, not an endpoint. There is no upstream image "
                "to pull and no `surya serve`. Something has to wrap it in HTTP, and that "
                "wrapper is yours to write and to containerise. Budget for it: it is a real "
                "component, not a config line, and it is the reason 'just run Surya' takes a "
                "day rather than an hour."),
            warnings=["there is no official ready-made server image - do not plan around "
                      "finding one."],
        ),
        ServiceStep(
            title="Write the wrapper with the models loaded ONCE at startup",
            what=_plain(
                "The pipeline is three predictors in sequence: detection finds text boxes, "
                "recognition reads them, layout classifies the regions. Load all three at "
                "process start (FastAPI's lifespan hook, or equivalent) and hold them. "
                "Loading them per request is the classic mistake here - it works in testing "
                "with one page and then adds the full model-load time to every single call "
                "in production."),
            command=("# contract the wrapper must satisfy:\n"
                     "#   GET  /healthz              -> 200 once the models are resident\n"
                     "#   POST /layout               -> multipart form field: image\n"
                     "#        returns tokens: [{type, bbox (normalised 0-1), text, conf}]\n"
                     "# load DetectionPredictor + RecognitionPredictor + LayoutPredictor\n"
                     "# at startup, never per request"),
            warnings=["a health endpoint that returns 200 before the models finish loading "
                      "will have every orchestrator declare the service ready while the "
                      "first real request times out."],
        ),
        ServiceStep(
            title="Run it, bound to loopback",
            what=_plain(
                "Containerised so the Python and GPU dependencies stay pinned. It needs the "
                "GPU device passed through, and it stays on loopback because the gateway is "
                "what anything outside talks to."),
            command="podman start surya || podman run -d --name surya --device /dev/dri -p 127.0.0.1:8090:8090 <your-image>",
            warnings=[_SERVICE_WARNING,
                      "the image is one you built around your own wrapper - there is no "
                      "canonical name for it."],
            verify="curl -s localhost:8090/healthz",
            expect="200 once the models are resident, not before",
        ),
        ServiceStep(
            title="Pre-warm with real page shapes",
            what=_plain(
                "The first request compiles GPU kernels for whatever image geometry it sees, "
                "and that compile can take over two minutes. Warming with a tiny test image "
                "compiles kernels for a shape no real scan uses, so the first genuine upload "
                "still pays the cost. Warm with an actual portrait and landscape page."),
            command="curl -sS --max-time 600 -F 'file=@page-portrait.png' localhost:8090/ocr >/dev/null",
            warnings=["the first call can take several minutes while kernels compile."],
        ),
    ],
)

_SURYA2 = Service(
    key="surya2",
    label="Surya 2 (OCR, GGUF under llama.cpp)",
    role="OCR successor - runs on llama-server like any other vision model",
    port=8093,
    notes=("a vision model in GGUF form, so it needs weights + mmproj and temp 0. "
           "KNOWN UPSTREAM BUG (datalab-to/surya #542): serving Surya 2 GGUF under the "
           "llama.cpp backend can fail immediately with a grammar-parse error on the bbox "
           "schema. If you hit it, use the vllm backend (NVIDIA GPU) for Surya 2, or stay on "
           "Surya 1, until it is fixed upstream."),
    steps=[
        ServiceStep(
            title="Patch the checkpoint so the converter recognises it",
            what=_plain(
                "This is the step that is not in anyone's documentation, and the one that "
                "makes the rest work. Surya 2 is a ~630M vision model built on a Qwen3.5 "
                "backbone with a Qwen3-VL-style vision merger, but its published config "
                "does not name an architecture llama.cpp's converter accepts, so a "
                "straight conversion fails. The checkpoint has to be patched to declare the "
                "architecture the converter knows before anything else will run. A working "
                "build's own metadata records exactly this - it reports "
                "architecture 'qwen35', a 'qwen3vl_merger' projector, and a model name of "
                "'_Patched_Ckpt' - which is the fingerprint of a patched config, not a "
                "vanilla one."),
            command=("hf download datalab-to/surya-ocr-2 --local-dir ~/src/surya-2-hf\n"
                     "# then, in ~/src/surya-2-hf/config.json, make the architecture one the\n"
                     "# converter supports (the working build reports qwen35 + a\n"
                     "# qwen3vl_merger projector). Compare against a Qwen3-VL config and\n"
                     "# align the fields the converter reads."),
            warnings=[_BIG_DOWNLOAD,
                      "the exact edit depends on the converter version you have - check "
                      "which architectures it accepts before guessing at the value."],
        ),
        ServiceStep(
            title="Convert to GGUF, weights and projector",
            what=_plain(
                "Two outputs from the patched checkpoint: the model and the mmproj vision "
                "projector. Both are needed; the server loads without the projector and "
                "then silently cannot see images."),
            command=("python3 ~/llama.cpp/convert_hf_to_gguf.py ~/src/surya-2-hf "
                     "--outfile ~/models/surya-2/surya-2.gguf.partial --outtype f16 && "
                     "mv ~/models/surya-2/surya-2.gguf.partial ~/models/surya-2/surya-2.gguf\n"
                     "python3 ~/llama.cpp/convert_hf_to_gguf.py ~/src/surya-2-hf --mmproj "
                     "--outfile ~/models/surya-2/surya-2-mmproj.gguf.partial && "
                     "mv ~/models/surya-2/surya-2-mmproj.gguf.partial "
                     "~/models/surya-2/surya-2-mmproj.gguf"),
            warnings=["conversion is memory-hungry - on a box that is also serving models, "
                      "run it when nothing else large is resident.",
                      "writing to .partial and moving on success means a failed run cannot "
                      "leave a truncated model that loads and then misbehaves."],
            verify="ls -la ~/models/surya-2/",
            expect=_plain(
                "roughly 1.2 GB of weights and 200 MB of projector. Keep them at f16: a "
                "working deployment runs this model unquantised, and at 630M parameters "
                "there is little to save and accuracy to lose. OCR errors are silent - a "
                "wrong character looks exactly like a right one."),
        ),
        ServiceStep(
            title="Serve it",
            what=_plain(
                "temp 0 because OCR is transcription, not generation: any sampling "
                "temperature invents characters. --parallel 2 lets a page batch overlap."),
            command=(f"{_LLAMA} -m ~/models/surya-2/surya-2.gguf "
                     "--mmproj ~/models/surya-2/surya-2-mmproj.gguf "
                     "-ngl 99 --no-mmap --flash-attn on -c 32768 --parallel 2 --jinja "
                     "--host 127.0.0.1 --port 8093 --temp 0"),
            warnings=[_SERVICE_WARNING],
            verify="curl -s localhost:8093/health",
            expect='{"status":"ok"}',
        ),
        ServiceStep(
            title="Before cutting over from Surya 1",
            what=_plain(
                "Run a fixed set of real pages through both generations and compare the "
                "output. A newer OCR model is not automatically better on your documents, "
                "and the failure is silent - wrong characters in extracted text look exactly "
                "like right ones until someone checks."),
            warnings=["do not retire Surya 1 until the comparison passes on your own pages."],
        ),
    ],
)

# ── the part that makes it one endpoint ───────────────────────────────────────
_GATEWAY = Service(
    key="gateway",
    label="Auth gateway",
    role="the single endpoint a cloud app talks to - the only thing exposed",
    port=8002,
    notes="the security boundary of the whole stack",
    steps=[
        ServiceStep(
            title="Put a reverse proxy in front, with auth",
            what=_plain(
                "Every model server above binds loopback and none of them has any "
                "authentication whatsoever - llama-server will answer anyone who can reach "
                "it. The gateway is what turns that into something a cloud app may talk to: "
                "Bearer-token auth on /v1/*, a rate limit, and a body-size cap. A call "
                "without a valid key must return 401. Any of a small reference proxy, Caddy "
                "with an auth module, or Traefik with forward-auth satisfies this - the "
                "contract is what matters, not the tool."),
            command=("# route by path to the right backend, all on localhost:\n"
                     "#   /v1/chat/completions -> 127.0.0.1:8001   (LLM)\n"
                     "#   /v1/vision           -> 127.0.0.1:8080   (VLM)\n"
                     "#   /v1/ocr              -> 127.0.0.1:8093   (Surya 2)\n"
                     "# enforce: Authorization: Bearer sk-... on every /v1/* route"),
            warnings=["this is the security boundary. If it is misconfigured, the whole "
                      "stack is open - test the 401 path before you expose anything."],
            verify="curl -s -o /dev/null -w '%{http_code}' localhost:8002/v1/models",
            expect="401 without a key, 200 with one. A 200 without a key means it is open.",
        ),
        ServiceStep(
            title="Prove the models are NOT reachable directly",
            what=_plain(
                "The gateway is worthless if the backends are also listening on a public "
                "interface. Check from another machine, not from the box itself - localhost "
                "always answers and will tell you everything is fine."),
            command="# from a DIFFERENT machine:\ncurl -m 5 http://<box-ip>:8001/health",
            expect="a timeout or refusal. Anything else means the LLM is exposed unauthenticated.",
            warnings=["run this from another machine. Testing from the box proves nothing."],
        ),
    ],
)

_TUNNEL = Service(
    key="tunnel",
    label="Outbound tunnel",
    role="reach the endpoint from a cloud app without opening a port",
    port=0,
    notes="point it at the GATEWAY, never at a model server",
    steps=[
        ServiceStep(
            title="Tunnel to the gateway, not to the model",
            what=_plain(
                "An outbound tunnel means no inbound firewall rule and no public IP. Point "
                "it at the gateway port. Pointing it at the LLM directly is the single "
                "worst mistake available here: it publishes an unauthenticated model to the "
                "internet, and it looks like it is working."),
            command="cloudflared tunnel --url http://127.0.0.1:8002",
            warnings=["target the gateway (8002), never a model port.",
                      "raise the timeouts: inference responses are slow and a default "
                      "proxy timeout will cut long generations off mid-stream."],
        ),
    ],
)

SERVICES = {s.key: s for s in (_LLM, _VLM, _SURYA, _SURYA2, _GATEWAY, _TUNNEL)}

# The order matters: backends first, then the thing that fronts them, then exposure.
# Standing the tunnel up before the gateway publishes an unauthenticated model.
STACK_ORDER = ("llm", "vlm", "surya2", "gateway", "tunnel")


def resolve_service(name) -> "Service | None":
    """Match a service by key or by a loose alias. None when unknown."""
    key = str(name or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not key:
        return None
    aliases = {
        "llm": "llm", "gemma": "llm", "gemma4": "llm", "text": "llm", "chat": "llm",
        "vlm": "vlm", "qwenvlm": "vlm", "vision": "vlm", "qwen3vl": "vlm",
        "surya": "surya", "surya1": "surya", "ocr": "surya",
        "surya2": "surya2", "suryaocr2": "surya2",
        "gateway": "gateway", "proxy": "gateway", "auth": "gateway", "caddy": "gateway",
        "traefik": "gateway", "endpoint": "gateway",
        "tunnel": "tunnel", "cloudflared": "tunnel", "cloudflare": "tunnel",
    }
    return SERVICES.get(aliases.get(key, key))


def stack_services() -> list:
    """The full stack, in the order it must be brought up."""
    return [SERVICES[k] for k in STACK_ORDER if k in SERVICES]


# ── platform caveats ──────────────────────────────────────────────────────────
# The llama-server commands above are portable: a GGUF is a GGUF and the flags are the
# same whichever accelerator built the binary. What is NOT portable is everything around
# them, and the differences are the kind that only surface once someone is standing at the
# machine.

_PLATFORM_NOTES = {
    "dgx-spark": [
        _plain(
            "Build llama.cpp with CUDA first, or every service here silently runs on the "
            "CPU. `deneb guide dgx-spark` covers it - and note that box ships with CUDA "
            "already installed, so do not install a toolkit."),
        _plain(
            "This machine is aarch64. GGUF models are architecture-independent and carry "
            "over unchanged, but CONTAINER images do not: an x86-only image will either "
            "refuse to run or fall into emulation and be unusably slow. That makes Surya 2 "
            "(GGUF under llama-server) the sound OCR choice here, and Surya 1 (container) "
            "the one to check an arm64 image exists for before committing to it."),
        _plain(
            "Memory is unified - the 128 GB is shared with the OS, not private VRAM. Size "
            "the three services against the pool TOGETHER: they are resident at the same "
            "time, and it is their combined footprint that has to fit, not each in turn."),
    ],
    "strix-halo": [
        _plain(
            "Build llama.cpp against ROCm or Vulkan first - `deneb guide strix-halo` covers "
            "the kernel, BIOS and boot-parameter work that has to happen before any of it "
            "is stable under sustained load."),
        _plain(
            "Memory is unified. Size the three services against the pool together, not one "
            "at a time - they are all resident at once."),
    ],
}


def platform_notes(platform_key: str) -> list:
    """Caveats that apply to running the whole stack on a given platform. [] when unknown."""
    return list(_PLATFORM_NOTES.get(str(platform_key or "").strip().lower(), []))
