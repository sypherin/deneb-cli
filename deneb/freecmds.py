"""The free commands (no token, nothing sent unless asked), defined once.

The no-token message, the engine error and `deneb --help` are generated from this list, and
a test checks the README names every entry. The first four are also the next-step order
everywhere: check, recommend, setup, watch --schedule.
"""
from __future__ import annotations

# (name, arguments shown after it, one-line description)
FREE = (
    ("deneb check", "", "scan the box: is the model served, on the GPU, ready?"),
    ("deneb recommend", "[--use ...]", "rank local models for this box (--use coding|vision|chat|general)"),
    ("deneb setup", "<model>", "print the platform-branched setup steps for a model"),
    ("deneb watch", "[--quiet|--history|--schedule|--unschedule|--reset]",
     "keep it healthy: what changed since the last check"),
    ("deneb profile", "", "read this box: structured hardware profile (os/cpu/ram/gpu)"),
    ("deneb catalog", "[--refresh]",
     "the models Deneb ranks, real per-quant sizes (--refresh reads Hugging Face)"),
    ("deneb guide", "[platform]", "pre-flight runbook for the box (strix-halo | dgx-spark)"),
    ("deneb stack", "[service]", "several models behind one authed endpoint\n"
     "                                (llm | vlm | ocr | gateway | tunnel)"),
    ("deneb selfupdate", "[--remote u@h]", "how to update Deneb, here or on another box"),
)

NAMES = tuple(n for n, _a, _d in FREE)


def hint() -> str:
    """The one-line "these need no token" pointer."""
    return ("the free commands need no token: "
            + ", ".join(n + (" <model>" if n == "deneb setup" else "") for n in NAMES))


def help_lines() -> str:
    """The free-command block of `deneb --help`, one aligned line per entry."""
    out = []
    for name, args, desc in FREE:
        left = f"  {name} {args}".rstrip()
        out.append(f"{left:<32}{desc}" if len(left) < 32 else f"{left}  {desc}")
    return "\n".join(out)
