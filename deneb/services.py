"""deneb.services — the multi-service STACK: several models behind one authed endpoint.

`deneb setup <model>` answers "how do I run one model from the catalog". That is not the
question anyone actually has on these boxes. The real question is the one a private-LLM
deployment forces: an LLM, a vision model and an OCR service all running at once, on
different ports, reachable by a cloud app through a single authenticated endpoint, without
any of them being exposed directly.

Two things here are not expressible in the model catalog at all, which is why this module
exists rather than more entries in CATALOG:

  * OCR is not a chat model. It is a vision model run at temperature 0 whose output is
    transcription, and a model catalog has no way to say that.
  * The gateway and the tunnel are not models. They are the part that makes the stack usable
    from outside, and the part that decides whether it is safe.

Ports (8001, 8080, 8082, 8002) are examples, not requirements; use any free ports. The
gateway's 8002 is the port `deneb check` and `deneb watch` probe for a gateway.

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
# Flags are chosen, not defaults. The non-obvious ones are called out, because
# they are the difference between a server that starts and a server that is usable.

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
                "model is the speculative-decode path. --host 127.0.0.1 keeps the port on "
                "loopback: the gateway runs on the same box and reaches it there, and "
                "nothing else should."),
            command=(f"{_LLAMA} -m ~/models/gemma-4-q4/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf "
                     "-md ~/models/gemma-4-q4/MTP/mtp-gemma-4-26B-A4B-it-F16.gguf "
                     "--spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 "
                     "--ctx-size 131072 --port 8001 --host 127.0.0.1 -ngl 99 --mmap --jinja "
                     "--ubatch-size 1024 -fa 1 -ctk q8_0 -ctv q8_0 --cache-prompt"),
            warnings=[_SERVICE_WARNING,
                      "never bind a model server to every network interface: llama-server has no "
                      "authentication, so that is an open model."],
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

_OCR = Service(
    key="ocr",
    label="Surya 2 (OCR, GGUF under llama.cpp)",
    role="document OCR, layout and tables - served by llama-server like any vision model",
    port=8082,
    notes=("a vision model in GGUF form, so it needs weights + mmproj and temp 0. "
           "Known upstream bug (datalab-to/surya #542): the surya client on the llama.cpp "
           "backend can fail at once with a grammar-parse error on its bbox schema. If you "
           "hit it, use the vllm backend (NVIDIA GPU) until it is fixed upstream."),
    steps=[
        ServiceStep(
            title="Fetch the published GGUF and its projector",
            what=_plain(
                "Datalab publishes Surya 2 as a GGUF: a 650M model and its mmproj vision "
                "projector. Both are needed; the server loads without the projector and "
                "then silently cannot see images. They ship at 16-bit, so leave them that "
                "way: at this size there is little memory to save, and quantising costs "
                "accuracy you will not notice, because a wrong character looks exactly "
                "like a right one."),
            command=("hf download datalab-to/surya-ocr-2-gguf "
                     "--include 'surya-2.gguf' --include 'surya-2-mmproj.gguf' "
                     "--local-dir ~/models/surya-2"),
            warnings=["downloads about 1.5 GB.",
                      "check the weights licence before business use: a modified AI Pubs "
                      "OpenRAIL-M (the code is Apache 2.0), free for research, personal use "
                      "and startups under $5M funding or revenue. The model card has the "
                      "terms."],
            verify="ls -la ~/models/surya-2/",
            expect="surya-2.gguf about 1.3 GB and surya-2-mmproj.gguf about 205 MB",
        ),
        ServiceStep(
            title="Serve it",
            what=_plain(
                "temp 0 because OCR is transcription, not generation: any sampling "
                "temperature invents characters. --parallel 2 lets a page batch overlap. "
                "The surya-ocr Python package can then use this server instead of "
                "starting its own: set SURYA_INFERENCE_URL=http://127.0.0.1:8082/v1."),
            command=(f"{_LLAMA} -m ~/models/surya-2/surya-2.gguf "
                     "--mmproj ~/models/surya-2/surya-2-mmproj.gguf "
                     "-ngl 99 --no-mmap --flash-attn on -c 32768 --parallel 2 --jinja "
                     "--host 127.0.0.1 --port 8082 --temp 0"),
            warnings=[_SERVICE_WARNING],
            verify="curl -s localhost:8082/health",
            expect='{"status":"ok"}',
        ),
        ServiceStep(
            title="Check it on your own pages before anything depends on it",
            what=_plain(
                "Run a fixed set of your own pages through it and read the output against "
                "the originals. Benchmark scores are not your documents, and OCR fails "
                "silently: wrong characters in extracted text look exactly like right ones "
                "until someone checks."),
            warnings=["keep that page set: rerun it after any model or server upgrade."],
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
                     "#   /v1/ocr              -> 127.0.0.1:8082   (OCR)\n"
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

SERVICES = {s.key: s for s in (_LLM, _VLM, _OCR, _GATEWAY, _TUNNEL)}

# The order matters: backends first, then the thing that fronts them, then exposure.
# Standing the tunnel up before the gateway publishes an unauthenticated model.
STACK_ORDER = ("llm", "vlm", "ocr", "gateway", "tunnel")


def resolve_service(name) -> "Service | None":
    """Match a service by key or by a loose alias. None when unknown."""
    key = str(name or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not key:
        return None
    aliases = {
        "llm": "llm", "gemma": "llm", "gemma4": "llm", "text": "llm", "chat": "llm",
        "vlm": "vlm", "qwenvlm": "vlm", "vision": "vlm", "qwen3vl": "vlm",
        "ocr": "ocr", "surya": "ocr", "surya2": "ocr", "suryaocr2": "ocr",
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
            "refuse to run or fall into emulation and be unusably slow. Every model here "
            "is a GGUF under llama-server, so this bites only on what you add around them: "
            "check an arm64 image exists before committing to any container."),
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
