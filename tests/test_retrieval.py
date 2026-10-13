"""Keyword retriever over the bundled public docs + the platform guide."""
from __future__ import annotations

import os
import re

from deneb import retrieval
import private_names

DOCS = os.path.join(os.path.dirname(__file__), "..", "deneb", "docs")
EXPECT = {"strix-halo-setup-README.md", "strix-guard.md", "unsloth-rocm-gfx1151.md",
          "vulkan-pinned-build.md", "vllm-gfx1151.md", "claude-code-local-qwen3.6-mtp.md",
          "comfyui-qwen-image.md", "deepseek-v4-flash-284b.md", "muse-glimmer-30b-strix.md"}


def test_exactly_the_nine_public_docs_are_bundled_with_a_source_header():
    assert set(os.listdir(DOCS)) - {"__pycache__"} == EXPECT
    for name in EXPECT:
        head = open(os.path.join(DOCS, name), encoding="utf-8").readline()
        assert "sypherin/strix-halo-setup" in head and "MIT" in head and "2026-10-13" in head, name


def test_bundled_docs_carry_no_private_names():
    for name in EXPECT:
        text = open(os.path.join(DOCS, name), encoding="utf-8").read()
        assert "runbook" not in text.lower(), name
        private_names.assert_clean(text, name)       # the private strings live outside the repo


def test_search_finds_the_right_doc_for_a_topic():
    hits = retrieval.search("vulkan build pinned llama.cpp", k=3)
    assert hits and any("vulkan-pinned-build" in h["label"] for h in hits)


def test_search_includes_the_platform_guide():
    hits = retrieval.search("strix halo BIOS carveout UMA frame buffer", k=6)
    assert any(h["label"].startswith("platform guide") for h in hits)


def test_search_labels_cite_a_source_and_text_is_bounded():
    hits = retrieval.search("systemd service restart", k=4)
    assert 1 <= len(hits) <= 4
    for h in hits:
        assert h["label"] and len(h["text"]) <= retrieval.MAX_CHUNK + 200


def test_nonsense_query_returns_nothing_not_noise():
    assert retrieval.search("zzzqqq xxyyzz", k=4) == []
    assert retrieval.search("", k=4) == []


def test_context_block_is_capped():
    block = retrieval.context_block("llama-server gpu offload", k=6, max_chars=3000)
    assert len(block) <= 3000
