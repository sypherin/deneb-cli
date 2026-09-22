"""deneb.speed: bandwidth decode estimate + hardware catalog matching."""
import json
import os

from deneb import speed
from deneb.hardware import GPUInfo, HardwareProfile

DEVICES = speed.load_devices()

# Real single-stream measurements from TokenMark (tokenmark.app, snapshot 2026-09-21),
# no speculative decoding. (hw id, active params B, bpw, moe, measured tok/s).
# The estimate must bracket each within a 25% margin either side; if a new engine
# generation beats this, refit EFFICIENCY rather than widening the margin blindly.
MEASURED = [
    ("strix-halo-395", 27.0, 8.5, False, 7.8),    # Qwen3.6-27B Q8_0, llama.cpp
    ("strix-halo-395", 27.0, 8.5, False, 6.6),    # Qwen3.6-27B UD-Q8_K_XL
    ("strix-halo-395", 3.0, 4.9, True, 53.0),     # Qwen3.6-35B-A3B UD-Q4_K_XL
    ("strix-halo-395", 3.0, 8.5, True, 46.5),     # Qwen3.6-35B-A3B UD-Q8_K_XL
    ("strix-halo-395", 3.3, 4.9, True, 73.7),     # Qwen3-Coder-30B-A3B UD-Q4_K_XL
    ("strix-halo-395", 5.1, 4.25, True, 50.6),    # gpt-oss-120b MXFP4
    ("strix-halo-395", 6.0, 4.9, True, 27.2),     # Qwen3.8-Flash-Next IQ4_XS
    ("dgx-spark", 3.0, 4.9, True, 62.7),          # Qwen3.6-35B-A3B Q4_K_XL, llama.cpp
]


def _dev(i):
    return next(d for d in DEVICES if d["id"] == i)


def test_catalog_loads_and_is_complete():
    assert len(DEVICES) >= 35
    assert len({d['id'] for d in DEVICES}) == len(DEVICES)
    for d in DEVICES:
        for k in ("id", "name", "vendor", "mem_gb", "bw_gbps", "engines", "source", "match"):
            assert d.get(k), (d.get("id"), k)
        assert d["bw_gbps"] > 50


def test_estimate_brackets_real_measurements():
    for hw, act, bpw, moe, meas in MEASURED:
        lo, hi = speed.est_decode_tps(act, bpw, _dev(hw)["bw_gbps"], moe=moe)
        assert lo * 0.75 <= meas <= hi * 1.25, (hw, act, bpw, moe, meas, lo, hi)


def test_estimate_math_and_bad_input():
    # 10B dense at 8 bpw = 10 GB/token; 1000 GB/s -> 10/550 s + 1 ms .. 10/750 s + 0.5 ms
    assert speed.est_decode_tps(10, 8, 1000) == (52, 72)
    assert speed.est_decode_tps(None, 8, 1000) is None
    assert speed.est_decode_tps(10, 8, 0) is None
    assert speed.est_decode_tps("x", 8, 1000) is None


def test_tier_for_tps():
    assert speed.tier_for_tps(80) == "fast"
    assert speed.tier_for_tps(20) == "medium"
    assert speed.tier_for_tps(8) == "slow"
    assert speed.tier_for_tps(3) == "very-slow"
    assert speed.tier_for_tps(None) == ""


def test_match_real_tool_names():
    m = lambda *n: (speed.match_device(list(n), DEVICES) or {}).get("id")
    assert m("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert m("NVIDIA GeForce RTX 4090 Laptop GPU") is None   # laptop has less bandwidth
    assert m("NVIDIA GB10") == "dgx-spark"
    assert m("AMD Radeon 8060S Graphics", "gfx1151") == "strix-halo-395"
    assert m("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S") == "strix-halo-395"
    assert m("AMD Radeon RX 7900 XTX") == "rx-7900-xtx"
    assert m("Apple M4 Max") == "m4-max"
    assert m("Apple M2 Max") == "m2-max"
    assert m("Apple M3 Pro") is None                          # not in catalog: honest None
    assert m("NVIDIA GeForce RTX 4080 SUPER") == "rtx-4080-super"
    assert m("NVIDIA GeForce RTX 4080") == "rtx-4080"
    assert m("NVIDIA GeForce RTX 3060") == "rtx-3060-12gb"
    assert m("NVIDIA GeForce RTX 3060 Ti") is None            # different card, not listed
    assert m("NVIDIA RTX 6000 Ada Generation") == "rtx-6000-ada"
    assert m("NVIDIA RTX A6000") == "rtx-a6000"
    assert m("NVIDIA A100-SXM4-80GB") == "a100-80gb"
    assert m("") is None
    assert speed.match_device([], DEVICES) is None


def test_device_for_profile_uses_gpu_then_cpu():
    p = HardwareProfile(cpu_model="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
                        gpus=[GPUInfo(vendor="amd", name="AMD Radeon 8060S Graphics",
                                      extra={"gfx": "gfx1151"})])
    assert speed.device_for_profile(p, DEVICES)["id"] == "strix-halo-395"
    assert speed.device_for_profile(HardwareProfile(), DEVICES) is None


def test_catalog_is_packaged():
    here = os.path.dirname(speed.__file__)
    assert os.path.exists(os.path.join(here, "data", "hardware_catalog.json"))
    json.load(open(os.path.join(here, "data", "hardware_catalog.json")))


def test_usable_bandwidth_rules():
    class P:  # minimal profile stand-in
        def __init__(self, backend):
            self.primary_backend = backend
    gpu = {"bw_gbps": 1008, "class": "consumer-gpu"}
    box = {"bw_gbps": 256, "class": "unified-memory-box"}
    assert speed.usable_bandwidth(P("cuda"), gpu) == 1008
    assert speed.usable_bandwidth(P("cpu"), gpu) is None      # VRAM unused on CPU run
    assert speed.usable_bandwidth(P("cpu"), box) == 256       # same pool either way
    assert speed.usable_bandwidth(P("rocm"), None) is None


def test_small_moe_on_fast_gpu_is_overhead_bound():
    # Qwen3-Coder-30B-A3B Q4_K_M on an RTX 4090. Community llama.cpp runs report roughly
    # 130-190 tok/s; a pure bandwidth bound says ~350. The overhead term keeps us honest.
    lo, hi = speed.est_decode_tps(3.3, 4.86, _dev("rtx-4090")["bw_gbps"], moe=True)
    assert 100 <= lo and hi <= 220, (lo, hi)
