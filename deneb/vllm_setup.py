"""deneb.vllm_setup - the vLLM route for `deneb setup --measured` on an Nvidia GPU box.

TokenMark's own cloud GPU runs (Altronis, 14 Oct 2026: H100 PCIe, H100 SXM, A100 40GB, A10 on
Lambda) served Qwen3.8-27B with vLLM 0.31.0, and on every one of those cards the vLLM row is the
fastest single-user number TokenMark has. When `--measured` picks such a row, the steps here
reproduce it: the same vLLM build, the same checkpoint, the same server settings, read off the
row and the multi-user rows of the same run.

What the runs used (bin/vllm-job.sh in the TokenMark ownbench results):
  - the vLLM 0.31.0 CUDA 12.9 wheel from the GitHub release (the PyPI wheel is CUDA 13, which needs
    a 580+ driver; the cloud images ran driver 570.148.08), installed with uv
    (`--torch-backend=cu129`), on x86_64;
  - VLLM_USE_DEEP_GEMM=0: DeepGEMM JIT-compiles with the system nvcc and needs 12.9+, the images
    shipped 12.8;
  - --max-model-len (the row's ctx), --gpu-memory-utilization 0.92 on 80 GB cards and 0.95 below,
    --max-num-seqs (the most users the run served at once), --no-enable-prefix-caching,
    --language-model-only, and for MTP rows --speculative-config with the row's draft count.

Pure: rows and a profile in, Choice / Step lists out. Nothing here downloads, installs or starts
anything. measured.py calls `classify()`; setup_advisor.setup_steps calls `steps()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import measured, setup_advisor, speed
from .setup_advisor import Step, _plain
from .tokenmark import _count

VLLM_VERSION = "0.31.0"
TORCH_BACKEND = "cu129"
WHEEL = (f"https://github.com/vllm-project/vllm/releases/download/v{VLLM_VERSION}/"
         f"vllm-{VLLM_VERSION}+{TORCH_BACKEND}-cp38-abi3-manylinux_2_28_x86_64.whl")
MEASURED_DRIVER = "570.148.08"      # the Nvidia driver on every TokenMark vLLM box
VENV = "~/vllm-venv"
UNIT = setup_advisor._VLLM_UNIT    # "vllm-server.service"
PORT = 8001
# GPU memory kept free beyond the weights for the KV cache and CUDA graphs. The A10 run (16.84 GiB
# of INT4 weights, 22.5 GiB card at 0.95) served 32k context with 4.5 GiB; less than this and vLLM
# has no room for one full-length sequence.
_RESERVE_GIB = 4.0
MIN_CTX, MAX_CTX = 32768, 262144
FALLBACK_SEQS = 16                  # a run with no multi-user rows: the A10's measured limit
MIN_SEQS, MAX_SEQS = 2, 1024


@dataclass(frozen=True)
class Weights:
    """A vLLM checkpoint deneb can tell you to download, with what TokenMark's runs measured."""
    name: str               # the quant label TokenMark rows carry
    repo: str
    download_gib: float     # the repo's files, summed from the Hugging Face API
    loaded_gib: float       # "Model loading took ... GiB" in the vLLM log, plain decoding
    mtp_gib: float          # the same with the MTP head loaded
    bpw: float              # bits per weight read per token (loaded size over the parameters)

    @property
    def dir(self) -> str:
        return setup_advisor._repo_basename(self.repo)


# Keyed by (catalog model key, TokenMark quant). Sizes checked 27 Oct 2026; loads from the 14 Oct logs.
WEIGHTS = {
    ("qwen3827b", "INT4"): Weights("INT4", "RedHatAI/Qwen3.8-27B-INT4", 18.14, 16.84, 17.63, 5.30),
    ("qwen3827b", "FP8"): Weights("FP8", "Qwen/Qwen3.8-27B-FP8", 28.77, 28.02, 28.49, 8.82),
}


def weights_for(model, quant_name: str):
    key = setup_advisor._norm_key(getattr(model, "name", ""))
    return next((w for (k, q), w in WEIGHTS.items() if k == key and q.lower() == quant_name.lower()), None)


def _gpu(profile):
    """The GPU vLLM serves on: the first Nvidia one (CUDA device 0 on a box with one kind)."""
    return next((g for g in getattr(profile, "gpus", None) or []
                 if (getattr(g, "vendor", "") or "").lower() == "nvidia"), None)


def _util(vram_mb: int) -> float:
    return 0.92 if vram_mb >= 70000 else 0.95


def _run_key(row) -> tuple:
    """Rows of one benchmark run: the same summary file, quant and mode (urls differ by #fragment)."""
    return (measured._url(row.get("url")).split("#")[0], measured._s(row.get("quant")).lower(),
            measured._s(row.get("mode"), 40).lower())


def run_peers(row, peers) -> list:
    """The multi-user (aggregate) rows of the same run as `row`."""
    key = _run_key(row)
    if not key[0]:
        return []
    out = []
    for p in peers or []:
        try:
            if (isinstance(p, dict) and p.get("throughput_kind") == "aggregate"
                    and _count(p.get("concurrency")) > 1 and measured._tps(p) and _run_key(p) == key
                    and not measured._unrated(p)):
                out.append(p)
        except Exception:  # noqa: BLE001 - one unreadable row never stops the rest
            continue
    return out


def peak(row, peers):
    """(combined tok/s, users) for the run's best multi-user row, or None."""
    agg = run_peers(row, peers)
    if not agg:
        return None
    best = max(agg, key=measured._tps)
    return measured._tps(best), _count(best.get("concurrency"))


def _spec_flags(row, model) -> tuple:
    """(--speculative-config flag or "", problem)."""
    mode = measured._s(row.get("mode"), 40).lower()
    if not mode or measured._BENCH_ONLY.fullmatch(mode):
        return "", ""
    m = measured._MTP.fullmatch(mode)
    if not m:
        return "", f"it ran on vllm with the '{mode}' decoding mode, which deneb's vLLM steps do not set"
    if not m.group(1):
        return "", "it ran on vllm with MTP but does not say how many draft tokens, so deneb cannot repeat it"
    problem = measured.draft_problem(m.group(1))
    if problem:
        return "", problem
    if not measured._MTP_FAMILY.search(getattr(model, "name", "") or ""):
        return "", (f"it ran on vllm with MTP, and deneb cannot confirm "
                    f"{getattr(model, 'name', 'this model')} carries MTP layers")
    return f"""--speculative-config '{{"method":"mtp","num_speculative_tokens":{int(m.group(1))}}}'""", ""


def classify(row: dict, model, profile, peers=None) -> tuple:
    """(measured.Choice, "") when deneb's vLLM steps can reproduce this row on this box, else
    (None, why). `peers` are the other rows for this hardware and model: the multi-user rows of
    the same run set --max-num-seqs."""
    qname = measured._s(row.get("quant"))
    w = weights_for(model, qname)
    if w is None:
        have = ", ".join(q for (k, q) in WEIGHTS if k == setup_advisor._norm_key(getattr(model, "name", "")))
        return None, (f"it ran on vllm with {qname or 'unnamed'} weights, and deneb has no vLLM recipe "
                      f"for those on {getattr(model, 'name', 'this model')}"
                      + (f" (it has {have})" if have else ""))
    gpu = _gpu(profile)
    backend = (getattr(profile, "primary_backend", "") or "").strip().lower()
    if gpu is None or backend != "cuda":
        return None, (f"it ran on vllm, and deneb's vLLM steps need an Nvidia GPU with CUDA "
                      f"(this box is {backend or 'not detected'})")
    arch = (getattr(profile, "arch", "") or "").strip().lower()
    if arch not in ("x86_64", "amd64"):
        return None, (f"it ran on vllm, and the vLLM {VLLM_VERSION} CUDA 12.9 wheel deneb installs is "
                      f"built for x86_64 (this box is {arch or 'not detected'})")
    if _count(row.get("num_gpus")) > 1:
        return None, f"it ran on vllm across {_count(row['num_gpus'])} GPUs, and deneb's vLLM steps serve on one"
    flags, problem = _spec_flags(row, model)
    if problem:
        return None, problem
    vram_mb = _count(getattr(gpu, "vram_mb", 0))
    if not vram_mb:
        return None, (f"it ran on vllm, and deneb could not read this GPU's memory, so it cannot "
                      f"tell whether {w.name} fits")
    util = _util(vram_mb)
    need = (w.mtp_gib if flags else w.loaded_gib) + _RESERVE_GIB
    room = vram_mb / 1024 * util
    if need > room:
        return None, (f"it ran on vllm, and {w.name} does not fit this GPU: the weights take "
                      f"{measured._fmt(need - _RESERVE_GIB)} GiB plus {measured._fmt(_RESERVE_GIB)} GiB "
                      f"for the KV cache, and vLLM may use {measured._fmt(room)} GiB here")
    over = measured._over_bound(row, model, w, profile)
    if over:
        return None, over
    warnings = []
    ctx = _count(row.get("ctx"))
    serve_ctx = min(max(ctx, MIN_CTX), MAX_CTX)
    if ctx and serve_ctx != ctx:
        warnings.append(f"the row was timed at a {ctx} token context; vLLM is set to "
                        f"--max-model-len {serve_ctx}, so decode may differ a little from the "
                        "measured number."
                        + (" If vLLM will not start, lower --max-num-seqs." if serve_ctx > ctx else ""))
    elif not ctx:
        warnings.append(f"the row names no context; vLLM is set to --max-model-len {serve_ctx}.")
    agg = run_peers(row, peers)
    if agg:
        seqs = max(_count(p.get("concurrency")) for p in agg)
        seqs = min(max(seqs, MIN_SEQS), MAX_SEQS)
    else:
        seqs = FALLBACK_SEQS
        warnings.append(f"TokenMark has no multi-user rows for this run, so --max-num-seqs {seqs} (a "
                        "safe start on any card here). vLLM refuses to start when the setting is "
                        "more than this GPU's cache holds; raise it step by step.")
    return measured.Choice(row=row, quant=w, playbook="cuda", flags=flags, warnings=warnings,
                           engine="vllm", ctx=serve_ctx, max_seqs=seqs, util=util), ""


def many_users_lines(choice, peers, model, profile) -> list:
    """What TokenMark measured with many users at once on this run, and the run of another set-up
    that served more in total when there is one (said, never switched to: the pick is single-user)."""
    lines = []
    mine = peak(choice.row, peers)
    if mine:
        lines.append(f"    many users at once: {measured._fmt(mine[0])} tok/s combined at {mine[1]} users "
                     f"(--max-num-seqs {choice.max_seqs})")
    best = None
    seen = {_run_key(choice.row)}
    for r in peers or []:
        try:
            if (not isinstance(r, dict) or r.get("throughput_kind") != "single-stream"
                    or measured._s(r.get("backend")).lower() != "vllm" or _run_key(r) in seen
                    or measured._unrated(r)):
                continue
            seen.add(_run_key(r))
            other = peak(r, peers)
            if not other or (mine and other[0] <= mine[0]):
                continue
            alt = classify(r, model, profile, peers)[0]
            if alt is not None and (best is None or other[0] > best[1][0]):
                best = (r, other, alt)
        except Exception:  # noqa: BLE001
            continue
    if best:
        r, (tps, users), alt = best
        lines.append(f"    for many users, {measured.describe(r)} did more: {measured._fmt(tps)} tok/s "
                     f"combined at {users} users ({measured._fmt(measured._tps(r))} tok/s single-user). "
                     f"deneb sets up the fastest single user; if many people share the server, "
                     f"{_changes(choice, alt)} in the service step.")
    return lines


def _changes(choice, alt) -> str:
    """The edits that turn the picked set-up into `alt`, in words a person applies by hand."""
    out = []
    if alt.quant.repo != choice.quant.repo:
        out.append(f"download {alt.quant.repo} and serve ~/models/{alt.quant.dir}")
    if alt.flags != choice.flags:
        out.append(f"use {alt.flags}" if alt.flags else "drop --speculative-config")
    if alt.ctx != choice.ctx:
        out.append(f"set --max-model-len {alt.ctx}")
    if alt.max_seqs != choice.max_seqs:
        out.append(f"set --max-num-seqs {alt.max_seqs}")
    return ", ".join(out) or "serve that run's settings"


def _serve_args(model, choice, home: str) -> list:
    """vllm serve's arguments. `home` is "~" for a shell, "%h" for a systemd unit."""
    args = [f"{home}/models/{choice.quant.dir}", "--served-model-name", getattr(model, "name", "model"),
            "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(choice.ctx),
            "--gpu-memory-utilization", f"{choice.util:.2f}", "--max-num-seqs", str(choice.max_seqs),
            "--no-enable-prefix-caching", "--language-model-only"]
    if choice.flags:
        args.append(choice.flags)
    return args


# where Astral's installer puts uv; every step that runs uv puts it on PATH again, so a step pasted
# into a new terminal still finds it
_UV_PATH = 'export PATH="$HOME/.local/bin:$PATH"; '

# uv when missing (Astral's installer works on images without pipx), on PATH, and new enough for
# --torch-backend; one line so it pastes as one
_UV_INSTALL = ("command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
               + _UV_PATH +
               "uv pip install --help | grep -q -- --torch-backend || echo \"this uv is too old for "
               "--torch-backend: run uv self update (or pipx upgrade uv)\" >&2")

_STOP_LLAMA = (f"if [ -f ~/.config/systemd/user/{setup_advisor._SERVICE_NAME} ]; then "
               f"systemctl --user disable --now {setup_advisor._SERVICE_NAME}; fi")


def steps(model, profile, choice, family: str = "unknown", free_gb=None, serving=None,
          models_dir: str = "~/models") -> list:
    """The ordered tell-only steps that reproduce a vLLM pick. Pure; never runs anything."""
    w = choice.quant
    name = getattr(model, "name", "model")
    need = math.ceil(w.download_gib)
    out = []
    if free_gb is not None and need > free_gb:
        out.append(Step(
            title="Not enough free disk space", command="", kind="note",
            what=_plain(f"the {w.name} weights need about {need} GiB ({w.download_gib} GiB) but only "
                        f"{free_gb} GiB is free on {models_dir}, and the vLLM install takes several "
                        "GiB more in your home directory. Free some space or point the download at "
                        "another drive first. On a cloud container the big disk is often mounted "
                        "elsewhere (RunPod: /workspace): before ~/models exists, link it there with "
                        "mkdir -p /workspace/models && ln -s /workspace/models ~/models.")))
    if serving and serving[1] == "ollama":
        out.append(Step(
            title="Ollama is serving on this box", command="", kind="note",
            what=_plain(f"Ollama is serving on 127.0.0.1:{serving[0]}. This route runs vLLM on "
                        f"127.0.0.1:{PORT} instead. vLLM claims {int(choice.util * 100)}% of the GPU's "
                        "memory when it starts, so a model Ollama holds on the same GPU makes it fail: "
                        "stop Ollama first (`sudo systemctl stop ollama` on Linux).")))
    elif serving and serving[0] == PORT:
        out.append(Step(
            title=f"Something already serves on port {PORT}", command="", kind="note",
            what=_plain(f"{serving[1] or 'a server'} is already on 127.0.0.1:{PORT}, the port vLLM "
                        f"uses here. The service step below stops deneb's llama.cpp unit "
                        f"({setup_advisor._SERVICE_NAME}) when there is one; any other server on "
                        f"{PORT}, stop it yourself first.")))
    out.append(Step(
        title="Check the Nvidia driver", kind="runtime",
        command="nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv",
        what=_plain(f"shows the GPU, its driver and its memory. TokenMark's runs used driver "
                    f"{MEASURED_DRIVER} with this CUDA 12.9 build of vLLM; cloud GPU images ship "
                    "the driver already. No output or an error: install Nvidia's driver first."),
        warnings=[]))
    out.append(Step(
        title="Install uv", kind="runtime", command=_UV_INSTALL,
        what=_plain("installs uv, the Python package tool the next two steps use, when the box has "
                    "none, puts ~/.local/bin on PATH, and checks the uv you have can "
                    "pick PyTorch's CUDA build (--torch-backend); an old one says to update it."),
        warnings=[_plain("when uv is missing this runs Astral's install script from astral.sh "
                         "(uv's own installer), which puts uv in ~/.local/bin.")]))
    out.append(Step(
        title=f"Install vLLM {VLLM_VERSION} (CUDA 12.9 build)", kind="runtime",
        command=(_UV_PATH + f"uv venv --python 3.12 {VENV} && uv pip install --python {VENV}/bin/python "
                 f"'{WHEEL}' --torch-backend={TORCH_BACKEND}"),
        what=_plain(f"makes a Python environment in {VENV} and installs the vLLM {VLLM_VERSION} "
                    "build TokenMark measured with, plus PyTorch for CUDA 12.9. uv downloads "
                    "Python 3.12 first when the box does not have it. Nothing outside "
                    f"{VENV} changes, apart from uv's own cache."),
        warnings=[_plain(f"downloads several GiB of Python packages (vLLM, PyTorch, Nvidia's CUDA "
                         f"libraries) into {VENV}.")]))
    out.append(Step(
        title=f"Download the {w.name} weights", kind="download",
        command=(_UV_PATH + f"uvx --from 'huggingface_hub[hf_xet]' hf download {w.repo} "
                 f"--local-dir {models_dir}/{w.dir}"),
        what=_plain(f"downloads {w.repo}, the {w.name} checkpoint TokenMark's run served"),
        warnings=[_plain(f"downloads ~{need} GiB to {models_dir}/{w.dir} - check you have the disk space."),
                  _plain("if it answers 401 or 403, Hugging Face wants you signed in: run "
                         "export HF_TOKEN=<your token> (huggingface.co/settings/tokens) and run "
                         "the download again.")]))
    env = "VLLM_USE_DEEP_GEMM=0"
    run_cmd = (f'PATH="$HOME/vllm-venv/bin:$PATH" {env} {VENV}/bin/vllm serve '
               + " ".join(_serve_args(model, choice, "~")))
    out.append(Step(
        title="Run the model (vllm serve)", command=run_cmd, kind="run",
        what=_plain(f"starts vLLM with the settings TokenMark measured. The first start takes a few "
                    f"minutes (it compiles and captures CUDA graphs); it is ready when "
                    f"curl -s 127.0.0.1:{PORT}/health answers."),
        warnings=[_plain(setup_advisor._SERVICE_WARNING),
                  _plain(f"{env} is how TokenMark ran it: DeepGEMM compiles with the system CUDA "
                         "compiler and needs 12.9 or newer."),
                  _plain("--language-model-only serves text only, as measured: the model's vision "
                         "part is not loaded. Remove the flag to send images; that needs more GPU "
                         "memory and was not timed.")]))
    # StartLimit: a start that fails every time (out of GPU memory, say) gives up after 4 tries
    # in 30 minutes instead of reloading the weights every 10 seconds for ever
    unit = ("[Unit]\nDescription=vLLM model server (127.0.0.1 only)\n"
            "StartLimitIntervalSec=1800\nStartLimitBurst=4\n\n[Service]\n"
            f"Environment={env}\n"
            "Environment=PATH=%h/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin\n"
            f"ExecStart=%h/vllm-venv/bin/vllm serve {' '.join(_serve_args(model, choice, '%h'))}\n"
            "Restart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n")
    path = f"~/.config/systemd/user/{UNIT}"
    guard = (f"{{ [ -f {models_dir}/{w.dir}/config.json ] || {{ echo \"no model found in "
             f"{models_dir}/{w.dir}: run the download step first\" >&2; false; }}; }}")
    # quoted 'EOF': the unit is written as is (%h is systemd's home directory). One command line,
    # so a failed guard skips the rest; `restart` so a re-run picks up new settings.
    out.append(Step(
        title="Keep it running (systemd user service)", kind="service",
        command=(f"{guard} && {_STOP_LLAMA} && mkdir -p ~/.config/systemd/user && "
                 f"cat > {path} <<'EOF' && systemctl --user daemon-reload && "
                 f"systemctl --user enable {UNIT} && systemctl --user restart {UNIT}\n{unit}EOF"),
        what=_plain(f"writes a user service for the server above and starts it, so it restarts after "
                    "a crash and `deneb check` / `deneb watch` can find it. It first stops deneb's "
                    f"llama.cpp unit ({setup_advisor._SERVICE_NAME}) if you have one: both use port "
                    f"{PORT}. Follow the start with `journalctl --user -u {UNIT} -f`."),
        warnings=[_plain(f"starts a local server on 127.0.0.1:{PORT} - it stays running until you stop "
                         f"it: `systemctl --user stop {UNIT}`.")]))
    out.append(setup_advisor._LINGER_STEP)
    return out


def device_class(profile) -> str:
    """The catalog class of this box ("datacenter-gpu", "unified-memory-box", ...), or ""."""
    dev = speed.device_for_profile(profile)
    return str((dev or {}).get("class") or "")
