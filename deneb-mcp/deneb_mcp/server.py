"""Deneb MCP server - the keyless Deneb verbs as Model Context Protocol tools.

An agent gets the same advice `deneb recommend` / `deneb setup` give at the terminal, because
both faces call one engine: `deneb.api`. There is NO second ranking here and NO side door -
the tools gather the box once through `api`, shape the result (deneb_mcp.shape) and return it.

Like the CLI verbs it mirrors, every tool is KEYLESS and engine-free: no auth, no LLM, no
hosted-engine round-trip. It executes NONE of the setup commands (the Deneb Rule) - it only
reads the box with local, read-only, allowlisted probes (ss / lspci / nvidia-smi / rocm-smi and
a localhost health check). The only external network is `measured=True`, which reads
tokenmark.app and fails soft. This server never touches the hosted Deneb engine or its
credentials; the engine-backed verbs (ask/diagnose) are not exposed here.

Run:  deneb-mcp            (stdio transport, for an MCP client)
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from deneb import api, check, hardware
from deneb_mcp import shape

mcp = FastMCP("deneb")

_USE_CASES = ("coding", "vision", "chat", "general")


@mcp.tool()
def deneb_recommend(use_case: str = "general", measured: bool = False) -> dict:
    """Rank Deneb's curated model catalog for THIS box and a use-case.

    Reads the local hardware, ranks the catalog by what fits and runs well here, and returns
    the picks with memory fit, expected speed tier, estimated decode tok/s and download size.
    Deterministic and keyless; it reads the box with read-only probes and executes none of the
    setup commands. The ranking is Deneb's size x quant-quality order, not a skill score - check
    a pick's published benchmarks before a big download.

    Args:
        use_case: one of coding, vision, chat, general (default general).
        measured: also attach the best real community tok/s per model from tokenmark.app.
            One network call, fail-soft (no data -> the picks come back without measurements).

    Returns a dict: box (backend, memory, bandwidth, free disk), the ranked picks, and a
    `next` pointer to deneb_setup for the top pick.
    """
    uc = (use_case or "general").strip().lower()
    if uc not in _USE_CASES:
        return {"error": f"unknown use_case '{use_case}'", "valid": list(_USE_CASES)}
    return shape.recommend_payload(api.recommend(uc, measured=bool(measured)))


@mcp.tool()
def deneb_setup(model: str, measured: bool = False) -> dict:
    """The platform-correct, ordered setup steps to run a catalog model on THIS box.

    Resolves the model in Deneb's curated catalog, reads the box, and returns the ordered
    tell-only steps (runtime -> download -> run): each step's exact command, what it does, and
    its inline warnings. Deneb shows the commands and RUNS NOTHING - you run each yourself.

    Args:
        model: a model name from the catalog (see deneb_recommend for exact names).
        measured: pick the measured tokenmark.app config Deneb can reproduce here and include
            its explanation lines. One network call, fail-soft.

    Returns a dict with the resolved model, the box, and the ordered `steps`. An unknown model
    returns {"error": ..., "did_you_mean": <closest name or null>} instead.
    """
    name = (model or "").strip()
    if not name:
        return {"error": "model name is required", "hint": "call deneb_recommend to see exact names"}
    data = api.setup(name, measured=bool(measured))
    if data is None:
        return {"error": f"unknown model '{name}'", "did_you_mean": api.suggest_model(name),
                "hint": "Deneb advises on its curated catalog only; call deneb_recommend for the exact names"}
    return shape.setup_payload(data, name)


@mcp.tool()
def deneb_box() -> dict:
    """What box is this, and what is already serving models on it.

    Reads the local hardware profile (backend, usable memory) and probes for a model server
    already answering, so an agent can orient before asking for a recommendation. Read-only: it
    runs only local, allowlisted detection probes and a localhost health check - no setup
    commands, no external network.
    """
    return shape.box_payload(hardware.profile_hardware(), check.serving_now())


def main() -> None:
    """Console entry point: serve the tools over stdio for an MCP client."""
    mcp.run()


if __name__ == "__main__":
    main()
