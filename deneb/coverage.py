"""deneb.coverage - does deneb cover every GPU TokenMark measures? A maintainer check.

TokenMark adds hardware as people run benchmarks on it (it added H100 / A100 / A10 cloud runs in
October 2026 before deneb could set any of them up). This reads TokenMark's rows and says, per
hardware id:
  - unknown_hardware: deneb's hardware catalog has no entry for it, so `deneb profile` cannot
    name the box and `--measured` cannot find its rows;
  - no_setup: a dedicated GPU deneb knows, but no measured row on it is one deneb's setup can
    reproduce (the reasons are each row's own `--measured` reason);
  - reproducible: the row `deneb setup <model> --measured` would pick on that card, per model;
  - catalog_only: unified-memory boxes (Strix Halo, DGX Spark, Macs, Jetson). Their usable
    memory depends on the box's own settings, so no stand-in box is built for them here; the
    check is that the catalog knows them.

Each dedicated GPU is checked with a stand-in box built from its catalog entry: its name, the
memory nvidia-smi would show (catalog size less ~6% for ECC and the driver: an A10 sold as 24 GB
shows 23028 MiB) and the CPU architecture the catalog names. Then the same pick code that
`deneb setup --measured` runs.

  python3 -m deneb.coverage [--json]     exit 0 covered, 1 a gap, 3 TokenMark unreachable
"""
from __future__ import annotations

import json
import re
import sys

from . import hf_catalog, measured, setup_advisor, speed, tokenmark
from .hardware import GPUInfo, HardwareProfile

# TokenMark's family ids for boxes it groups (measured.hardware_candidates maps them back)
_FAMILIES = {"strix-halo": "strix-halo-395", "mac-max": "m4-max", "mac-ultra": "m3-ultra"}
_BACKENDS = {"nvidia": "cuda", "amd": "rocm", "intel": "vulkan", "apple": "metal"}
# nvidia-smi MiB over the catalog's GiB: HBM cards show nearly all of it (H100 81559 / 81920),
# GDDR cards lose a slice to ECC and reserve (A10 23028 / 24576)
_SHOWN_FRACTION = {"hbm": 0.995, "other": 0.94}


def _unified(dev: dict) -> bool:
    return dev.get("class") in ("unified-memory-box", "edge") or "unified" in str(dev.get("mem_type") or "").lower()


def stand_in(dev: dict, name: str = "") -> HardwareProfile:
    """A box with only this GPU in it, as deneb's hardware probe would read it."""
    vendor = str(dev.get("vendor") or "").lower()
    backend = _BACKENDS.get(vendor, "cpu")
    arch = "aarch64" if re.search(r"aarch64|arm64", str(dev.get("arch") or ""), re.I) else "x86_64"
    shown = _SHOWN_FRACTION["hbm" if "hbm" in str(dev.get("mem_type") or "").lower() else "other"]
    vram = int(float(dev.get("mem_gb") or 0) * 1024 * shown)
    return HardwareProfile(primary_backend=backend, usable_mem_mb=vram, arch=arch,
                           gpus=[GPUInfo(vendor=vendor, backend=backend, name=name or dev.get("name") or "",
                                         vram_mb=vram, memory_kind="dedicated")])


def _resolving_box(dev: dict, devices: list):
    """A stand-in box the catalog's own match rules resolve back to `dev`, or None."""
    for name in [dev.get("name") or ""] + list(dev.get("match") or []):
        box = stand_in(dev, name)
        hit = speed.match_device(speed.profile_names(box), devices)
        if hit and hit.get("id") == dev.get("id"):
            return box
    return None


def report(rows, devices=None) -> dict:
    """{unknown_hardware, no_setup, reproducible, catalog_only, unknown_models} for TokenMark
    rows. Pure apart from reading the bundled catalogs."""
    devices = speed.load_devices() if devices is None else devices
    by_id = {d.get("id"): d for d in devices}
    models = hf_catalog.all_models()
    hw_rows: dict = {}
    for r in rows or []:
        if isinstance(r, dict) and isinstance(r.get("hardware"), str) and r["hardware"]:
            hw_rows.setdefault(r["hardware"], []).append(r)
    out = {"unknown_hardware": [], "no_setup": [], "reproducible": {}, "catalog_only": [],
           "unknown_models": []}
    for hw in sorted(hw_rows):
        dev = by_id.get(hw)
        if dev is None and hw in _FAMILIES and _FAMILIES[hw] in by_id:
            out["catalog_only"].append(hw)
            continue
        if dev is None:
            out["unknown_hardware"].append(hw)
            continue
        if _unified(dev):
            out["catalog_only"].append(hw)
            continue
        box = _resolving_box(dev, devices)
        if box is None:
            out["no_setup"].append({"hardware": hw, "why": [
                f"the catalog's match rules do not find {dev.get('name') or hw} from its own name"]})
            continue
        picks, whys = [], []
        for mname in sorted({str(r.get("model") or "") for r in hw_rows[hw]}):
            model = setup_advisor.resolve_model(mname, models)
            if model is None:
                if mname and mname not in out["unknown_models"]:
                    out["unknown_models"].append(mname)
                continue
            cands, _bad = measured._candidates(rows, hw, model)
            rated = [r for r in cands if not measured._unrated(r)]
            if not rated:
                whys.append(f"{mname}: no rated single-user rows")
                continue
            peers = measured._peers(rows, hw, model)
            for r in rated:
                try:
                    c, why = measured._classify(r, model, box, peers)
                except Exception as e:  # noqa: BLE001 - a row the pick code chokes on is a gap, said
                    c, why = None, f"the pick code failed on it ({type(e).__name__})"
                if c is not None:
                    picks.append({"model": model.name, "tok_s": measured._tps(r), "engine": c.engine,
                                  "quant": c.quant.name, "row": measured.describe(r)})
                    break
                whys.append(f"{mname} {measured._fmt(measured._tps(r))} tok/s ({measured.describe(r)}): {why}")
        if picks:
            out["reproducible"][hw] = picks
        else:
            out["no_setup"].append({"hardware": hw, "why": whys or ["no rows deneb can read"]})
    out["unknown_models"].sort()
    return out


def _text(rep: dict) -> str:
    lines = []
    for hw in rep["unknown_hardware"]:
        lines.append(f"GAP  {hw}: TokenMark measures it, deneb's hardware catalog has no entry "
                     "(add it to deneb/data/hardware_catalog.json)")
    for g in rep["no_setup"]:
        lines.append(f"GAP  {g['hardware']}: no measured row deneb's setup can reproduce")
        lines += [f"       {w}" for w in g["why"]]
    for hw, picks in sorted(rep["reproducible"].items()):
        for p in picks:
            lines.append(f"ok   {hw}: {p['model']} {measured._fmt(p['tok_s'])} tok/s via {p['engine']} "
                         f"({p['row']})")
    if rep["catalog_only"]:
        lines.append(f"ok   in the catalog (unified memory, setup not simulated): "
                     f"{', '.join(rep['catalog_only'])}")
    if rep["unknown_models"]:
        lines.append(f"note TokenMark rows for models deneb does not list: {', '.join(rep['unknown_models'])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    got = tokenmark.tracker_rows()
    if got is None:
        print("TokenMark's tracker could not be read (offline, or no rows)", file=sys.stderr)
        return 3
    rep = report(got[0])
    print(json.dumps(rep, indent=1) if "--json" in argv else _text(rep))
    return 1 if rep["unknown_hardware"] or rep["no_setup"] else 0


if __name__ == "__main__":
    sys.exit(main())
