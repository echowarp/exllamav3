"""FAST-LAUNCH: cached Triton launches for small per-step kernels (LLMT_FASTLAUNCH=1 enables it; default off).

Triton's JITFunction.run re-binds and re-specializes every argument on every launch (~1.3 ms per step of host time in
single-stream decode on a slow host CPU, measured with py-spy). launch(jitfn, grid, *args, **kw) runs the normal path
once per specialization key and caches the CompiledKernel; later calls go straight to CompiledKernel[grid](*all
params in order) = the exact call JITFunction.run makes (bound_args incl. constexprs), so the result is bit-identical.
Key = everything Triton specializes on here: tensor dtype + 16-byte pointer alignment, int ==1 / %16==0 / 32-vs-64-bit,
constexpr values, launch options. Anything unexpected -> the normal path. With LLMT_FASTLAUNCH unset or 0 every call
is jitfn[grid](*args, **kw), unchanged.

Measured on an RTX 5090 (Qwen3.8-27B EXL3, int4 KV, MTP): single-stream steps/s +0-1.9 %, multi-stream cells within
noise. Used by draftdyn (DYN-ROW extras), tail_fuse (draft_scatter) and the plugin's eagle step update."""
import os
import torch

_ON = os.environ.get("LLMT_FASTLAUNCH", "0") == "1"
_C: dict = {}
_OPT = ("num_warps", "num_stages", "num_ctas", "maxnreg")


def _spec(a):
    if isinstance(a, torch.Tensor):
        return ("T", a.dtype, a.data_ptr() % 16 == 0)
    if isinstance(a, bool):
        return ("b", a)
    if isinstance(a, int):
        return ("i", a == 1, a % 16 == 0, -2**31 <= a < 2**31)
    if isinstance(a, float):
        return ("f",)
    if a is None:
        return ("n",)
    return ("o", a)


def launch(jitfn, grid, *args, **kw):
    if not _ON or not hasattr(jitfn, "arg_names") or not hasattr(jitfn, "params"):
        return jitfn[grid](*args, **kw)
    try:
        names = jitfn.arg_names
        vals = list(args) + [kw[n] for n in names[len(args):]]
        if len(vals) != len(names):
            raise KeyError
        opts = tuple(sorted((k, v) for k, v in kw.items() if k not in names))
        key = (id(jitfn), opts) + tuple(("c", v) if jitfn.params[i].is_constexpr else _spec(v) for i, v in enumerate(vals))
    except Exception:  # noqa
        return jitfn[grid](*args, **kw)
    k = _C.get(key)
    if k is None:
        k = jitfn[grid](*args, **kw)
        _C[key] = k
        return k
    g = tuple(grid)
    k[g + (1,) * (3 - len(g))](*vals)   # JITFunction.run pads the grid to 3 dims; CompiledKernel[grid] doesn't
    return k
