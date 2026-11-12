"""bit-exact faster int8-KV FA2 prefill for vllm_plugin._i8_prefill_fa.

EXL3_I8FA=1 replaces the two torch pieces around the FA2 calls with CUDA kernels from i8kv.so (EXL3_I8FA_DIR/<name>/<name>.so, default
the package's lib/):
  * gather-dequant: deq2 reads the paged int8_per_token_head cache (block table, [256 int8 | f32 scale] rows) and writes
    the fp16 K and V of one key range in one launch, fp16(float(x) * scale) RNE = the torch expression bit for bit;
  * LSE merge: w = exp(lse - amax) and w.sum(0) stay in torch (same ops), the [Q, H, 256] part (float(o_p) * w_p summed in
    part order, / ws, to fp16) is one kernel with explicit _rn arithmetic.
The FA2 calls themselves (plugin _fa: image FA2 or the FA16 variant) and the 32k-token chunking are unchanged, so the
output is bitwise equal to the old path. Paged route: the in-kernel-dequant FA2 (i8fa_a.so for stock FA2, i8fa_fp.so for
FA16 f; bitwise as well) runs all key parts in one launch; it is 1.6-2x slower per CTA but wins when few query rows meet
many 32k parts (low occupancy of the serial FA calls). EXL3_I8FA_PAGED_MAXQ: "auto" (default) = Q <= 128 and >= 3 parts
(measured crossover: 120k ctx x 64 rows 6.4 -> 3.1 ms A, 4.6 -> 2.2 ms F), N = Q <= N always, 0 = off.
Any unsupported case or error falls back to the old path.
"""
import os
import importlib
import sys
import logging

import torch

logger = logging.getLogger("vllm")
_ON = os.environ.get("EXL3_I8FA", "0") == "1"
_DIR = os.environ.get("EXL3_I8FA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
_PAGED_ENV = os.environ.get("EXL3_I8FA_PAGED_MAXQ", "auto").strip().lower() or "auto"
_PAGED_AUTO = _PAGED_ENV == "auto"
_PAGED_MAXQ = 128 if _PAGED_AUTO else int(_PAGED_ENV)
_PAGED_MINPARTS = 3 if _PAGED_AUTO else 1
_KV = None
_PAGED = None
_WARNED: list = []


def _load(name):
    sys.path.insert(0, f"{_DIR}/{name}")
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def setup(fa16_variant: str):
    """Called once by the plugin; returns True when the fast path is on."""
    global _KV, _PAGED, _ON
    if not _ON:
        return False
    try:
        _KV = _load("i8kv")
    except Exception as e:  # noqa
        logger.warning("exl3: I8FA unavailable (%s); old int8-KV prefill path", e)
        _ON = False
        return False
    pv = {"": "i8fa_a", "f": "i8fa_fp"}.get(fa16_variant)
    if _PAGED_MAXQ > 0 and pv is not None:
        try:
            _PAGED = _load(pv)
        except Exception as e:  # noqa
            logger.warning("exl3: I8FA paged kernel %s unavailable (%s)", pv, e)
    logger.info("exl3: I8FA on (fused gather-dequant + merge%s)",
                f"; paged FA2 for <= {_PAGED_MAXQ} rows and >= {_PAGED_MINPARTS} parts ({pv})" if _PAGED is not None else "")
    return True


def _ok(q, kc, vc, out):
    return (q.dtype == torch.float16 and out.dtype == torch.float16 and q.dim() == 3 and q.shape[-1] == 256
            and q.stride(-1) == 1 and out.stride(-1) == 1 and kc.dim() == 4 and kc.shape[-1] == 260
            and kc.dtype == torch.int8 and kc.stride(-1) == 1 and vc.stride() == kc.stride()
            and kc.data_ptr() % 4 == 0 and vc.data_ptr() % 4 == 0 and all(s % 4 == 0 for s in kc.stride()[:3])
            and out.data_ptr() % 16 == 0 and out.stride(0) % 8 == 0 and out.stride(1) % 8 == 0)


def prefill(q, key_cache, value_cache, pages, seq_len, scale, out, fa, kb):
    """Same contract as vllm_plugin._i8_prefill_fa; returns out, or None for 'use the old path'."""
    if _KV is None:
        return None
    try:
        if not _ok(q, key_cache, value_cache, out):
            return None
        Q = q.shape[0]
        prefix = seq_len - Q
        nparts = (prefix + kb - 1) // kb + 1
        if nparts > 8:
            return None
        if pages.dtype != torch.int32 or not pages.is_contiguous():
            pages = pages.to(torch.int32).contiguous()
        if _PAGED is not None and Q <= _PAGED_MAXQ and nparts >= _PAGED_MINPARTS and key_cache.data_ptr() % 8 == 0 \
                and all(s % 8 == 0 for s in key_cache.stride()[:3]):
            parts = []
            for s0 in range(0, prefix, kb):
                parts += [s0, min(s0 + kb, prefix) - s0, 0]
            parts += [prefix, Q, 1]
            lse = torch.empty(nparts, q.shape[1], Q, dtype=torch.float32, device=q.device)
            if nparts == 1:
                _PAGED.fwd(q, key_cache, value_cache, pages, parts, float(scale), out, lse, -1)
                return out
            o = torch.empty(nparts, Q, q.shape[1], 256, dtype=torch.float16, device=q.device)
            _PAGED.fwd(q, key_cache, value_cache, pages, parts, float(scale), o, lse, -1)
            w = torch.exp(lse - lse.amax(0))
            _KV.merge(list(o.unbind(0)), w, w.sum(0), out)
            return out
        res = []
        for s0 in range(0, prefix, kb):
            e0 = min(s0 + kb, prefix)
            k16, v16 = _KV.deq2(key_cache, value_cache, pages, s0, e0)
            res.append(fa(q, k16, v16, False, scale))
        k16, v16 = _KV.deq2(key_cache, value_cache, pages, prefix, seq_len)
        res.append(fa(q, k16, v16, True, scale))
        if len(res) == 1:
            out.copy_(res[0][0])
            return out
        lses = torch.stack([pp[1] for pp in res])          # [n, H, Q], as the old path
        w = torch.exp(lses - lses.amax(0))
        _KV.merge([pp[0] for pp in res], w, w.sum(0), out)
        return out
    except Exception as e:  # noqa
        if not _WARNED:
            _WARNED.append(1)
            logger.warning("exl3: I8FA error (%s); old path for this call", e)
        return None
