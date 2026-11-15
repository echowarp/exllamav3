"""Code defaults of the exl3cuda feature switches.

An unset EXL3_* switch takes the value of the reference build (Qwen3.8-27B EXL3 4.0 bpw on an RTX 5090, int4 KV,
MTP k = 7; README "Environment variables", configs/int4.env) when the first visible GPU is sm_120, and the old
"off" value otherwise: the csrc/ kernels are sm_120-only builds and the Triton paths are tuned for that card.
A variable that is set (also to an empty string) always wins. Features keyed on a KV-cache dtype (VA, I8FA: int8;
VA4, KVW4, I4FA: int4) stay idle with any other dtype; missing kernel libraries log a warning and leave vLLM's
path in place.

sm120() asks vLLM's platform (NVML on CUDA), so it does not initialise CUDA in the process that loads the plugin.
device_ok(dev) is the per-device check used right before a csrc/ kernel is first launched.
"""
import functools
import os


@functools.lru_cache(maxsize=None)
def sm120() -> bool:
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_cuda():
            return False
        cap = current_platform.get_device_capability()
        return cap is not None and (cap.major, cap.minor) == (12, 0)
    except Exception:  # noqa: no vLLM / no NVML / no GPU: reference defaults off
        return False


def env(name: str, ref: str, other: str) -> str:
    """os.environ[name] if set, else ref on sm_120 and other elsewhere."""
    v = os.environ.get(name)
    if v is not None:
        return v
    return ref if sm120() else other


_DEV: dict = {}


def device_ok(dev=None) -> bool:
    """True when dev (a torch device, default: current) is an sm_120 GPU."""
    try:
        import torch
        d = torch.device(dev) if dev is not None else torch.device("cuda", torch.cuda.current_device())
        idx = d.index if d.index is not None else torch.cuda.current_device()
        ok = _DEV.get(idx)
        if ok is None:
            ok = _DEV[idx] = d.type == "cuda" and tuple(torch.cuda.get_device_capability(idx)) == (12, 0)
        return ok
    except Exception:  # noqa
        return False
