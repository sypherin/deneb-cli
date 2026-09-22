"""recommend() over the bundled live (Hugging Face) snapshot + hardware bandwidth."""
from deneb import hf_catalog, speed
from deneb.hardware import GPUInfo, HardwareProfile
from deneb.recommend import quality_factor, quant_bits, recommend

MODELS, SOURCE = hf_catalog.ranking_catalog()


def _box(name, backend, usable, kind="dedicated"):
    g = GPUInfo(vendor="x", name=name, backend=backend, memory_kind=kind)
    return HardwareProfile(gpus=[g], primary_backend=backend, usable_mem_mb=usable)


BOXES = [
    _box("NVIDIA GeForce RTX 4090", "cuda", 23500),
    _box("NVIDIA GB10", "cuda", 104000, "unified"),
    _box("AMD Radeon 8060S Graphics", "rocm", 107871, "unified"),
    _box("Apple M4 Max", "metal", 50000, "unified"),
    _box("NVIDIA GeForce RTX 3060", "cuda", 11500),
]


def test_ranking_uses_live_snapshot():
    assert SOURCE.startswith("live catalog")
    assert any(m.name == "Qwen3.8-Flash-Next" for m in MODELS)


def test_every_known_box_gets_fitting_picks_with_estimates():
    for p in BOXES:
        bw = speed.usable_bandwidth(p, speed.device_for_profile(p))
        assert bw, p.gpus[0].name
        for uc in ("coding", "vision", "chat", "general"):
            recs = recommend(p, uc, catalog=MODELS, bw_gbps=bw)
            assert recs and all(r.fit.fits for r in recs), (p.gpus[0].name, uc)
            for r in recs:
                lo, hi = r.fit.est_tps
                assert 0 < lo <= hi
                assert "tok/s" in r.why and "—" not in r.why
                assert quant_bits(r.quant.name) >= 2   # no 1-bit ever reaches a client


def test_live_ranking_is_deterministic():
    p = BOXES[0]
    a = [(r.model.name, r.quant.name) for r in recommend(p, "coding", catalog=MODELS, bw_gbps=1008)]
    b = [(r.model.name, r.quant.name) for r in recommend(p, "coding", catalog=MODELS, bw_gbps=1008)]
    assert a == b


def test_cpu_run_on_discrete_gpu_box_has_no_estimate():
    p = _box("NVIDIA GeForce RTX 4090", "cpu", 30000)
    assert speed.usable_bandwidth(p, speed.device_for_profile(p)) is None
    for r in recommend(p, "chat", catalog=MODELS, bw_gbps=None):
        assert r.fit.est_tps is None


def test_quant_bits_and_quality_factor():
    assert quant_bits("UD-Q2_K_XL") == 2 and quality_factor("UD-Q2_K_XL") == 0.6
    assert quant_bits("UD-Q3_K_XL") == 3 and quality_factor("Q3_K_M") == 0.85
    assert quant_bits("UD-IQ4_XS") == 4 and quality_factor("IQ4_XS") == 1.0
    assert quant_bits("MXFP4") == 4 and quant_bits("Q8_0") == 8
    assert quant_bits("BF16") == 16
