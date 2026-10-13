"""deneb.retrieval — a small keyword retriever for the local brain.

Searches the public docs bundled in deneb/docs (copied from github.com/sypherin/strix-halo-setup,
MIT) plus Deneb's own platform guide. Standard library only: heading-sized chunks, BM25-style
scoring. No network, no embeddings, nothing written.
"""
from __future__ import annotations

import math
import os
import re

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
MAX_CHUNK = 1400
_STOP = frozenset("""a an and are as at be but by can do does for from has have how i if in is it its
me my no not of on or our so that the their then there these this to was we what when where which why
will with you your""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")

_index = None   # (chunks, df, avg_len), built once per process


def _tokens(text: str) -> list:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if len(t) > 1 and t not in _STOP:
            out.append(t)
    return out


def _split(label: str, text: str):
    """Chunks of one doc: cut at markdown headings, long sections cut at blank lines."""
    heading, buf = "", []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        piece = ""
        for para in re.split(r"\n\s*\n", body):
            if piece and len(piece) + len(para) > MAX_CHUNK:
                yield (f"{label} > {heading}" if heading else label, piece.strip())
                piece = ""
            while len(para) > MAX_CHUNK:
                yield (f"{label} > {heading}" if heading else label, para[:MAX_CHUNK])
                para = para[MAX_CHUNK:]
            piece += para + "\n\n"
        if piece.strip():
            yield (f"{label} > {heading}" if heading else label, piece.strip())

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            yield from flush()
            buf = []
            heading = m.group(2).strip()
        buf.append(line)
    yield from flush()


def _platform_chunks():
    from . import platform_guide
    for key in sorted(platform_guide.GUIDES):
        g = platform_guide.GUIDES[key]
        head = f"platform guide: {g.label}"
        yield head, f"{g.label}. Applies to: {g.applies_to}. Recommended OS: {g.recommended_os}."
        for n, s in enumerate(g.steps, 1):
            parts = [s.title, s.what, s.command or "", " ".join(s.warnings or []),
                     f"verify: {s.verify}" if s.verify else ""]
            yield f"{head} > {n}. {s.title}", "\n".join(p for p in parts if p).strip()


def _build():
    global _index
    chunks = []
    if os.path.isdir(DOCS_DIR):
        for name in sorted(os.listdir(DOCS_DIR)):
            if name.endswith(".md"):
                with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
                    for label, body in _split(name, f.read()):
                        chunks.append((label, body))
    chunks.extend(_platform_chunks())
    rows, df = [], {}
    for label, body in chunks:
        toks = _tokens(label + " " + body)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df[t] = df.get(t, 0) + 1
        rows.append((label, body, tf, max(len(toks), 1)))
    avg = (sum(r[3] for r in rows) / len(rows)) if rows else 1.0
    _index = (rows, df, avg)
    return _index


def search(query: str, k: int = 4) -> list:
    """Top chunks for a question: [{label, text, score}], best first. Empty when nothing matches."""
    q = list(dict.fromkeys(_tokens(query)))
    if not q:
        return []
    rows, df, avg = _index or _build()
    n = len(rows)
    scored = []
    for label, body, tf, ln in rows:
        s = 0.0
        for t in q:
            f = tf.get(t)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * ln / avg))
        if s > 0:
            scored.append((s, label, body))
    scored.sort(key=lambda x: -x[0])
    return [{"label": lbl, "text": body, "score": round(sc, 3)} for sc, lbl, body in scored[:k]]


def context_block(query: str, k: int = 4, max_chars: int = 6000) -> str:
    """The retrieved chunks as one text block, each headed by its label, capped at max_chars."""
    out, used = [], 0
    for h in search(query, k):
        piece = f"[{h['label']}]\n{h['text']}"
        if used + len(piece) > max_chars:
            piece = piece[:max(0, max_chars - used)]
        if not piece.strip():
            break
        out.append(piece)
        used += len(piece) + 2
        if used >= max_chars:
            break
    return "\n\n".join(out)[:max_chars]
