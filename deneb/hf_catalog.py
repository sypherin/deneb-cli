"""deneb.hf_catalog — the self-updating model catalog.

The hand-written CATALOG in models_catalog.py goes stale (it still called Qwen3.6
"current"). This module turns a short curated WATCHLIST (deneb/data/model_watchlist.json:
name, GGUF repo, dense/MoE, active params, capabilities) into full catalog entries using
LIVE Hugging Face data: total params from the GGUF header and the REAL on-disk size of
every quant (sharded files summed). Nothing about size is guessed any more.

Layers:
  * pure:   parse_gguf_tree(), build_entry(), entry_to_model()  (unit-tested, no I/O)
  * I/O:    fetch_entry(), refresh()                            (Hugging Face API, keyless)
  * cache:  load_models() reads ~/.cache/deneb/models.json if present, else the bundled
            deneb/data/models_snapshot.json shipped with the package.

The Deneb engine's daily landscape job imports this same module, so the CLI and the
engine can never disagree about what a quant weighs.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import urllib.request
from typing import Optional

from .models_catalog import Model, Quant

_HERE = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(_HERE, "data", "model_watchlist.json")
SNAPSHOT_PATH = os.path.join(_HERE, "data", "models_snapshot.json")
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deneb", "models.json")
HF = "https://huggingface.co"

# Quants worth recommending. 1-bit builds are left out on purpose: they load, but the
# quality loss is too large to put in front of a client as a default.
KEEP_QUANTS = ("UD-Q2_K_XL", "UD-Q3_K_XL", "IQ4_XS", "UD-IQ4_XS", "Q4_K_M", "UD-Q4_K_XL",
               "MXFP4", "Q5_K_M", "UD-Q5_K_XL", "Q6_K", "UD-Q6_K_XL", "Q8_0", "UD-Q8_K_XL")

_QUANT_RE = re.compile(
    r"-((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32))(?:-\d{5}-of-\d{5})?\.gguf$",
    re.IGNORECASE)


# ── pure ──────────────────────────────────────────────────────────────────────
def _file_label(path: str) -> Optional[str]:
    """The quant label exactly as one GGUF file name spells it, or None for files that are not
    model weights (mmproj projectors, MTP draft heads, imatrix data)."""
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    if not low.endswith(".gguf") or "mmproj" in base or low.startswith("mtp/") \
            or base.startswith("mtp-") or "imatrix" in low:
        return None
    m = _QUANT_RE.search(path.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def quant_of(path: str) -> Optional[str]:
    """Quant label of one GGUF file path, or None for files that are not model weights.
    The catalog's name for it: upper case, MXFP4_MOE called MXFP4."""
    q = _file_label(path)
    return q.upper().replace("MXFP4_MOE", "MXFP4") if q else None


def gguf_layout(files: list) -> dict:
    """HF tree listing -> {quant: {"parts": file count, "tag": label as the files spell it}}.
    Setup needs both: Ollama can't pull a quant split into several files, and a download
    filter or `find` must match the file's own label, not the catalog's name for it.
    Deterministic, no I/O."""
    out: dict = {}
    for f in files or []:
        if not isinstance(f, dict) or f.get("type") != "file":
            continue
        path = f.get("path") or ""
        q = quant_of(path)
        if q:
            row = out.setdefault(q, {"parts": 0, "tag": _file_label(path)})
            row["parts"] += 1
    return out


def parse_gguf_tree(files: list) -> tuple:
    """HF tree listing -> ({quant: total_bytes}, has_mmproj). Shards of one quant are
    summed. Deterministic, no I/O."""
    sizes: dict = {}
    mmproj = False
    for f in files or []:
        if not isinstance(f, dict) or f.get("type") != "file":
            continue
        path = f.get("path") or ""
        if path.lower().endswith(".gguf") and "mmproj" in path.lower().rsplit("/", 1)[-1]:
            mmproj = True
            continue
        q = quant_of(path)
        if q:
            sizes[q] = sizes.get(q, 0) + int(f.get("size") or 0)
    return sizes, mmproj


def build_entry(watch: dict, total_params: Optional[int], sizes: dict, mmproj: bool,
                fetched: str, layout: Optional[dict] = None) -> Optional[dict]:
    """One catalog entry (plain dict, JSON-safe) from a watchlist row plus HF facts.
    None when there is nothing usable (no params or no recommendable quant). `layout` is
    gguf_layout's; a quant it does not cover gets no parts/tag (read back as not recorded)."""
    if not total_params or total_params <= 0:
        return None
    params_b = round(total_params / 1e9, 1)
    quants = []
    for q, b in sizes.items():
        if q not in KEEP_QUANTS or b <= 0:
            continue
        size_mb = int(round(b / 1e6))
        quants.append({"name": q, "size_mb": size_mb,
                       "bpw": round(b * 8 / total_params, 2), **(layout or {}).get(q, {})})
    if not quants:
        return None
    quants.sort(key=lambda x: x["bpw"])
    caps = list(watch.get("capabilities") or ["general"])
    if mmproj and "vision" not in caps:
        caps.append("vision")
    return {
        "name": watch["name"], "params_b": params_b,
        "architecture": watch.get("architecture") or "dense",
        "active_params_b": watch.get("active_params_b"),
        "capabilities": caps, "gguf_repo": watch["gguf_repo"], "mmproj": bool(mmproj),
        "quants": quants, "notes": watch.get("notes") or "", "source": watch.get("source") or "",
        "fetched": fetched,
    }


def _safe_tag(tag) -> str:
    """A recorded file label, or "" (not recorded) when it isn't one: setup pastes it into
    shell commands, and the cache file it comes from is plain JSON anyone can edit."""
    return tag if isinstance(tag, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", tag) else ""


_REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,79}")
_QNAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,40}")


def _checked(value, pattern) -> str:
    """The value if it is a string that fully matches the allowlist, else raise. setup pastes
    repo, name and quant names into shell commands and the cache they come from is plain JSON."""
    if not isinstance(value, str) or not pattern.fullmatch(value) or ".." in value:
        raise ValueError("unsafe catalog field")
    return value


def entry_to_model(e: dict) -> Optional[Model]:
    """Catalog dict -> the Model dataclass fit()/recommend() already consume."""
    try:
        quants = [Quant(name=_checked(q["name"], _QNAME_RE), size_mb=int(q["size_mb"]), bpw=float(q["bpw"]),
                        parts=int(q.get("parts") or 0), tag=_safe_tag(q.get("tag")))
                  for q in e["quants"]]
        notes = e.get("notes") or ""
        if e.get("mmproj"):
            notes = (notes + " Vision needs the mmproj file from the same repo.").strip()
        caps = list(e.get("capabilities") or [])
        # a model tagged only coding (and general) is a Coder model: recommend puts it first
        # for coding
        built_for = "coding" if "coding" in caps and set(caps) <= {"coding", "general"} else ""
        return Model(_checked(e["name"], _NAME_RE), float(e["params_b"]), e.get("architecture") or "dense",
                     e.get("active_params_b"), caps,
                     quants, _checked(e["gguf_repo"], _REPO_RE), notes=notes, built_for=built_for)
    except (KeyError, TypeError, ValueError):
        return None


# ── I/O ───────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "deneb-cli (+https://altronis.sg/deneb)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        from .net import read_capped
        return json.loads(read_capped(r).decode("utf-8", "replace"))


def load_watchlist(path: str = WATCHLIST_PATH) -> list:
    with open(path) as f:
        return list(json.load(f).get("models") or [])


def fetch_entry(watch: dict, fetched: str) -> Optional[dict]:
    repo = watch["gguf_repo"]
    meta = _get(f"{HF}/api/models/{repo}?expand[]=gguf")
    tree = _get(f"{HF}/api/models/{repo}/tree/main?recursive=1")
    sizes, mmproj = parse_gguf_tree(tree)
    return build_entry(watch, (meta.get("gguf") or {}).get("total"), sizes, mmproj, fetched,
                       layout=gguf_layout(tree))


def refresh(watchlist: Optional[list] = None) -> tuple:
    """Fetch every watchlist model. Returns (entries, errors). Fails soft per model:
    one gated or renamed repo is reported, never fatal."""
    now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).isoformat(timespec="minutes")
    entries, errors = [], []
    for w in (watchlist if watchlist is not None else load_watchlist()):
        try:
            e = fetch_entry(w, now)
            if e:
                entries.append(e)
            else:
                errors.append(f"{w.get('gguf_repo')}: no usable quant/params")
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{w.get('gguf_repo')}: {ex}")
    return entries, errors


def write_json(path: str, payload: dict) -> None:
    """Atomic write: .partial then rename, so a crash never leaves half a catalog."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_entries() -> tuple:
    """(entries, source_label). User cache wins over the bundled snapshot."""
    # Said as what it is: a list fetched on a date, not a live feed (QC, 20 Oct)
    for path, label in ((CACHE_PATH, "model list refreshed {}"),
                        (SNAPSHOT_PATH, "model list of {}, shipped with deneb")):
        try:
            with open(path) as f:
                d = json.load(f)
            ents = d.get("models") or []
            if ents:
                return ents, label.format(str(d.get("generated", ""))[:10])
        except (OSError, ValueError):
            continue
    return [], "none"


def load_models() -> list:
    return [m for m in (entry_to_model(e) for e in load_entries()[0]) if m]


def ranking_catalog() -> tuple:
    """(models, source_label) for `deneb recommend`: the live catalog when there is one,
    else the curated static CATALOG. Never a mix: an old 70B from the static list would
    outrank current models on raw size."""
    from .models_catalog import CATALOG
    ents, label = load_entries()
    models = [m for m in (entry_to_model(e) for e in ents) if m]
    return (models, label) if models else (list(CATALOG), "built-in model list")


def all_models() -> list:
    """Every model Deneb can name: live entries first, then static ones not superseded,
    so `deneb setup <old name>` still resolves."""
    from .models_catalog import CATALOG
    live = load_models()
    seen = {m.name.lower() for m in live}
    return live + [m for m in CATALOG if m.name.lower() not in seen]
