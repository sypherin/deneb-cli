# deneb-mcp

A Model Context Protocol (MCP) server that gives an AI agent the same keyless model-setup
advice the `deneb` CLI gives at the terminal, on your own AI box or a rented cloud GPU
(H100, A100, A10).

Both faces call one engine, `deneb.api`, so the tools below return the same answer as
`deneb recommend` and `deneb setup`. There is no second ranking and no side door. Like the CLI
verbs they mirror, the tools are keyless and engine-free: no auth, no LLM, no hosted-engine
round-trip. They execute none of the setup commands (the Deneb Rule); they read the box only
with local, read-only, allowlisted probes (`ss` / `lspci` / `nvidia-smi` / `rocm-smi` and a
localhost health check). The only external network is `measured=True`, which reads tokenmark.app
and fails soft.

This server does not touch the hosted Deneb engine or any credentials. The engine-backed verbs
(ask / diagnose) are deliberately not exposed here.

## Tools

- `deneb_recommend(use_case="general", measured=False)`: rank Deneb's model catalog for this box.
  `use_case` is one of `coding`, `vision`, `chat`, `general`. `measured=True` attaches the best
  real community tok/s per model from tokenmark.app. Returns the box, the ranked picks (memory
  fit, speed tier, estimated decode tok/s, download size) and a pointer to `deneb_setup`.
- `deneb_setup(model, measured=False)`: the ordered, platform-correct setup steps to run a
  catalog model here: each step's command, what it does, and its warnings. Deneb shows the
  commands and runs none of them. An unknown name returns a `did_you_mean` suggestion.
  `measured=True` builds the steps from TokenMark's fastest measured run for this box. On one
  x86_64 H100 (SXM or PCIe), A100 40GB or A10 with Qwen3.8-27B that is vLLM 0.31 on port 8001
  (kept up by `vllm-server.service` where the box has systemd); elsewhere it is llama.cpp. The `engine` field says which.
- `deneb_box()`: the local hardware profile and what model server is already serving, so an
  agent can orient before asking for a recommendation. Read-only local probes, no external network.

The ranking is Deneb's model-size times quant-quality order for the use-case, not a skill score.
Check a pick's published benchmarks before a large download.

## Install

`deneb-mcp` depends on `deneb-cli` (it imports `deneb` in-process) and on `mcp`. For local
development from a checkout of this repo:

```
pip install -e .          # from the deneb-cli repo root, installs deneb-cli
pip install -e deneb-mcp  # installs this server
```

## Run

```
deneb-mcp
```

The server speaks MCP over stdio. Point an MCP client at that command. Example client config:

```json
{
  "mcpServers": {
    "deneb": { "command": "deneb-mcp" }
  }
}
```

## Why in-process

The MCP imports `deneb` directly rather than shelling out to the CLI or calling a web API. The
selection logic lives in the engine that owns the question, and every face (CLI, MCP) calls that
engine. Keeping the MCP in the same process keeps it in sync by construction: there is no copy of
the ranking to drift.
