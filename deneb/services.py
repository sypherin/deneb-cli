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
                     "--include 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf' --local-dir ~/models/gemma-4-q4"),
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
                     "-md ~/models/gemma-4-q4/mtp-gemma-4-26B-A4B-it.gguf "
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
    notes="needs an mmproj projector file alongside the weights, or images are ignored",
    steps=[
        ServiceStep(
            title="Fetch the weights AND the projector",
            what=_plain(
                "A vision model is two files: the weights and the mmproj multimodal "
                "projector. Start it without the projector and it loads happily, answers "
                "text, and silently cannot see images - which reads as a bad model rather "
                "than a missing file."),
            command=("hf download unsloth/Qwen3-VL-32B-Instruct-GGUF "
                     "--include '*Q4_K_M.gguf' --include '*mmproj*F16.gguf' "
                     "--local-dir ~/models/qwen-vlm"),
            warnings=[_BIG_DOWNLOAD],
        ),
        ServiceStep(
            title="Serve it",
            what=_plain(
                "--no-mmap because the weights are pinned rather than paged, which matters "
                "on a unified-memory box. Context is deliberately small: vision tokens are "
                "expensive and a large window here costs memory the LLM needs."),
            command=(f"{_LLAMA} -m ~/models/qwen-vlm/Qwen3VL-32B-Instruct-Q4_K_M.gguf "
                     "--mmproj ~/models/qwen-vlm/mmproj-Qwen3VL-32B-Instruct-F16.gguf "
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
    notes="not a GGUF and not llama.cpp: a Python service in a container",
    steps=[
        ServiceStep(
            title="Run the OCR service",
            what=_plain(
                "Surya 1 is a Python stack, so it runs as a container rather than under "
                "llama-server. It is the generation that has been in production use; Surya 2 "
                "below is the GGUF successor."),
            command="podman start surya-only || podman run -d --name surya-only --device /dev/dri -p 127.0.0.1:8090:8090 <surya-image>",
            warnings=[_SERVICE_WARNING,
                      "the image name is site-specific - use the one your deployment built."],
            verify="curl -s localhost:8090/healthz",
            expect="ok",
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
    notes="a vision model in GGUF form, so it needs weights + mmproj and temp 0",
    steps=[
        ServiceStep(
            title="Convert the model to GGUF",
            what=_plain(
                "Surya 2 is published as HuggingFace weights, so it is converted with "
                "llama.cpp's own converter, then quantised. This produces the two files the "
                "server needs: the weights and the mmproj projector. Run it on the machine "
                "that will serve it, or on any machine - the output is portable. "
                "NOT VERIFIED END TO END by Deneb's authors on Surya 2 specifically: this is "
                "the standard llama.cpp vision-model conversion path, and the projector flag "
                "name has changed between llama.cpp releases. Check "
                "`convert_hf_to_gguf.py --help` on the version you actually have before "
                "assuming these flags."),
            command=("hf download datalab-to/surya-ocr-2 --local-dir ~/src/surya-2-hf && "
                     "python3 ~/llama.cpp/convert_hf_to_gguf.py ~/src/surya-2-hf "
                     "--outfile ~/models/surya-2/surya-2-f16.gguf --outtype f16 && "
                     "python3 ~/llama.cpp/convert_hf_to_gguf.py ~/src/surya-2-hf "
                     "--mmproj --outfile ~/models/surya-2/surya-2-mmproj.gguf"),
            warnings=[_BIG_DOWNLOAD,
                      "conversion is memory-hungry - on a shared box run it when nothing "
                      "else large is resident.",
                      "flags vary by llama.cpp version; verify against --help first."],
        ),
        ServiceStep(
            title="Quantise it",
            what=_plain(
                "The f16 output is larger than it needs to be for OCR. Q4_K_M keeps quality "
                "for this task at a fraction of the size."),
            command=("~/llama.cpp/build/bin/llama-quantize "
                     "~/models/surya-2/surya-2-f16.gguf ~/models/surya-2/surya-2.gguf Q4_K_M"),
            warnings=["write to a .partial name and move it into place only on success, so "
                      "a failed run cannot leave a truncated model that loads and misbehaves."],
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
