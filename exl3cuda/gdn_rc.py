"""GDN recompute on CUDA: Triton port of the Intel Arc (XPU) SYCL kernel exl3_gdn_recompute.

Speculative decoding for the GDN layers with ONE committed SSM state per request: each step, per (request, v-head,
row tile), the previous step's accepted tokens ("pending", stored normalized in a double-buffered side pool) are applied
to the committed state, which is stored once; this step's 1+k tokens then produce outputs without storing intermediate
states, and their vectors become the next step's pending tokens. vLLM can therefore run with
MambaSpec.num_speculative_blocks = 0 (no 1+k state pages per request).

Semantics are the SYCL kernel's (same argument list):
  q, k [T, H, K] fp16 (raw; L2-normalized here, q scaled), v [T, HV, V] fp16, g, beta [T, HV] fp32, o [T, HV, V] fp16
  h: SSM state pool, strided view [blocks, HV, V, K] (any float dtype; K contiguous)
  state_idx [N]: committed state block per request
  pk [2, S, T1, H, K], pv [2, S, T1, HV, V], pg / pb [2, S, T1, HV] (fp32): pending pool, read parity par[n],
      write parity 1 - par[n], pool slot slot[n]
  n_pend [N]: pending tokens to apply (previous step's accepted count); qsl [N+1]: this step's token ranges
  bnd_idx [N]: store the state after pending token bnd_idx into block bnd_block[n] (decode-time boundary state for
      prefix caching; -1 = none)
Update per token (fp32): S *= exp(g); d = S k^; S += beta (v - d) k^T; output o = S q^ (q^ = q/|q| * K^-1/2,
k^ = k/|k|, eps 1e-6), GQA: v-head hv reads k-head hv // (HV // H).
"""
import torch
import triton
import triton.language as tl


# the recompute loop is latency-bound (one program walks the pending
# and new tokens in sequence), so more, smaller programs win: BV=16 / 1 warp is 34-47% faster than BV=32 / 4 warps at
# 1-16 sequences (rc_tune2.py: 11.5 -> 7.6 us at 1 seq, 62.3 -> 38.3 at 16), same per-element math (BV only splits V).
import os as _os
_RC_BV = int(_os.environ.get("EXL3_RC_BV", "32"))
_RC_WARPS = int(_os.environ.get("EXL3_RC_WARPS", "4"))


@triton.jit
def _gdn_rc_kernel(q, k, v, g, beta, o, h, state_idx, pk, pv, pg, pb, n_pend, slot, par, qsl, bnd_idx, bnd_block,
                   S, T1, scale, hs0, hs1, hs2,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                   HAS_BND: tl.constexpr):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    sidx = tl.load(state_idx + i_n).to(tl.int64)
    p_h = h + sidx * hs0 + i_hv * hs1 + o_v[:, None].to(tl.int64) * hs2 + o_k[None, :]
    b_h = tl.load(p_h).to(tl.float32)
    npend = tl.load(n_pend + i_n)
    sl = tl.load(slot + i_n).to(tl.int64)
    pr = tl.load(par + i_n).to(tl.int64)
    rb = (pr * S + sl) * T1                    # read base (token index into the pool)
    wb = ((1 - pr) * S + sl) * T1              # write base
    if HAS_BND:
        bi = tl.load(bnd_idx + i_n)
        bblk = tl.load(bnd_block + i_n).to(tl.int64)
    # 1. apply pending (accepted) tokens; store the committed state once
    for t in range(0, npend):
        pt = rb + t
        b_k = tl.load(pk + (pt * H + i_h) * K + o_k)
        b_vp = tl.load(pv + (pt * HV + i_hv) * V + o_v)
        b_g = tl.load(pg + pt * HV + i_hv)
        b_b = tl.load(pb + pt * HV + i_hv)
        b_h *= tl.exp(b_g)
        b_u = (b_vp - tl.sum(b_h * b_k[None, :], 1)) * b_b
        b_h += b_u[:, None] * b_k[None, :]
        if HAS_BND:
            if t == bi:
                tl.store(h + bblk * hs0 + i_hv * hs1 + o_v[:, None].to(tl.int64) * hs2 + o_k[None, :],
                         b_h.to(h.dtype.element_ty))
    tl.store(p_h, b_h.to(h.dtype.element_ty))
    # 2. this step's tokens: outputs only; record them as the next step's pending tokens
    t0 = tl.load(qsl + i_n)
    tn = tl.load(qsl + i_n + 1) - t0
    for t in range(0, tn):
        tok = (t0 + t).to(tl.int64)
        b_q = tl.load(q + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_k = tl.load(k + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_v = tl.load(v + (tok * HV + i_hv) * V + o_v).to(tl.float32)
        b_q = b_q * (scale / tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (1.0 / tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_g = tl.load(g + tok * HV + i_hv)
        b_b = tl.load(beta + tok * HV + i_hv)
        wt = wb + t
        if i_v == 0:
            if i_hv % (HV // H) == 0:
                tl.store(pk + (wt * H + i_h) * K + o_k, b_k)
            tl.store(pg + wt * HV + i_hv, b_g)
            tl.store(pb + wt * HV + i_hv, b_b)
        tl.store(pv + (wt * HV + i_hv) * V + o_v, b_v)
        b_h *= tl.exp(b_g)
        b_u = (b_v - tl.sum(b_h * b_k[None, :], 1)) * b_b
        b_h += b_u[:, None] * b_k[None, :]
        tl.store(o + (tok * HV + i_hv) * V + o_v, tl.sum(b_h * b_q[None, :], 1).to(o.dtype.element_ty))


def gdn_recompute(q, k, v, g, beta, o, h, state_idx, pk, pv, pg, pb, n_pend, slot, par, qsl, T1, scale,
                  bnd_idx=None, bnd_block=None, BV=None):
    """Same argument list as the SYCL kernel exl3_gdn_recompute. N = state_idx.numel() requests."""
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    N = state_idx.numel()
    S = pk.shape[1]
    BV = BV or _RC_BV
    assert h.stride(-1) == 1 and K == 128 and V % BV == 0
    has_bnd = bnd_idx is not None and bnd_block is not None
    grid = (V // BV, HV, N)
    _gdn_rc_kernel[grid](q, k, v, g, beta, o, h, state_idx, pk, pv, pg, pb, n_pend, slot, par, qsl,
                         bnd_idx if has_bnd else n_pend, bnd_block if has_bnd else n_pend,
                         S, T1, scale, h.stride(0), h.stride(1), h.stride(2),
                         H=H, HV=HV, K=K, V=V, BV=BV, HAS_BND=has_bnd, num_warps=_RC_WARPS)
    return o


# the same update with the per-layer prologue folded in. _rc_forward used to run ~12
# tiny kernels per GDN layer before/after the recompute kernel (q/k/v .contiguous() copies out of the conv output, the
# gating g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b) as separate elementwise ops, an output copy):
# ~600 single-block launches per step on the 5090 (48 GDN layers), each costing more as a kernel boundary than as work.
# Here q/k/v are read from the conv output with a token stride, g/beta are computed per (token, v-head) from the raw
# a/b rows (fp32, torch's softplus with threshold 20), and o may be any [T, HV, V] view with a token stride.
@triton.jit
def _gdn_rc_fused_kernel(x, xs, a, a_s, b, b_s, A_log, dt_bias, o, os_, h, state_idx, pk, pv, pg, pb, n_pend, slot,
                         par, qsl, bnd_idx, bnd_block, S, T1, scale, hs0, hs1, hs2,
                         H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                         HAS_BND: tl.constexpr):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    KD: tl.constexpr = H * K
    sidx = tl.load(state_idx + i_n).to(tl.int64)
    p_h = h + sidx * hs0 + i_hv * hs1 + o_v[:, None].to(tl.int64) * hs2 + o_k[None, :]
    b_h = tl.load(p_h).to(tl.float32)
    npend = tl.load(n_pend + i_n)
    sl = tl.load(slot + i_n).to(tl.int64)
    pr = tl.load(par + i_n).to(tl.int64)
    rb = (pr * S + sl) * T1
    wb = ((1 - pr) * S + sl) * T1
    if HAS_BND:
        bi = tl.load(bnd_idx + i_n)
        bblk = tl.load(bnd_block + i_n).to(tl.int64)
    for t in range(0, npend):
        pt = rb + t
        b_k = tl.load(pk + (pt * H + i_h) * K + o_k)
        b_vp = tl.load(pv + (pt * HV + i_hv) * V + o_v)
        b_g = tl.load(pg + pt * HV + i_hv)
        b_b = tl.load(pb + pt * HV + i_hv)
        b_h *= tl.exp(b_g)
        b_u = (b_vp - tl.sum(b_h * b_k[None, :], 1)) * b_b
        b_h += b_u[:, None] * b_k[None, :]
        if HAS_BND:
            if t == bi:
                tl.store(h + bblk * hs0 + i_hv * hs1 + o_v[:, None].to(tl.int64) * hs2 + o_k[None, :],
                         b_h.to(h.dtype.element_ty))
    tl.store(p_h, b_h.to(h.dtype.element_ty))
    neg_a = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    dtb = tl.load(dt_bias + i_hv).to(tl.float32)
    t0 = tl.load(qsl + i_n)
    tn = tl.load(qsl + i_n + 1) - t0
    for t in range(0, tn):
        tok = (t0 + t).to(tl.int64)
        b_q = tl.load(x + tok * xs + i_h * K + o_k).to(tl.float32)
        b_k = tl.load(x + tok * xs + KD + i_h * K + o_k).to(tl.float32)
        b_v = tl.load(x + tok * xs + 2 * KD + i_hv * V + o_v).to(tl.float32)
        b_q = b_q * (scale / tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (1.0 / tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        za = tl.load(a + tok * a_s + i_hv).to(tl.float32) + dtb
        sp = tl.where(za > 20.0, za, tl.log(1.0 + tl.exp(za)))
        b_g = neg_a * sp
        b_b = tl.sigmoid(tl.load(b + tok * b_s + i_hv).to(tl.float32))
        wt = wb + t
        if i_v == 0:
            if i_hv % (HV // H) == 0:
                tl.store(pk + (wt * H + i_h) * K + o_k, b_k)
            tl.store(pg + wt * HV + i_hv, b_g)
            tl.store(pb + wt * HV + i_hv, b_b)
        tl.store(pv + (wt * HV + i_hv) * V + o_v, b_v)
        b_h *= tl.exp(b_g)
        b_u = (b_v - tl.sum(b_h * b_k[None, :], 1)) * b_b
        b_h += b_u[:, None] * b_k[None, :]
        tl.store(o + tok * os_ + i_hv * V + o_v, tl.sum(b_h * b_q[None, :], 1).to(o.dtype.element_ty))


def gdn_recompute_fused(x, H, HV, K, V, a, b, A_log, dt_bias, o, h, state_idx, pk, pv, pg, pb, n_pend, slot, par, qsl,
                        T1, scale, bnd_idx=None, bnd_block=None, BV=None):
    """x [T, 2*H*K + HV*V] conv output rows (q | k | v, unit stride inside a row); a, b [T, HV] raw gate inputs;
    o [T, HV, V] (token stride free, inner dims contiguous)."""
    N = state_idx.numel()
    S = pk.shape[1]
    BV = BV or _RC_BV
    assert h.stride(-1) == 1 and K == 128 and V % BV == 0 and x.stride(-1) == 1
    assert a.stride(-1) == 1 and b.stride(-1) == 1 and o.stride(-1) == 1 and o.stride(-2) == V
    has_bnd = bnd_idx is not None and bnd_block is not None
    grid = (V // BV, HV, N)
    _gdn_rc_fused_kernel[grid](x, x.stride(0), a, a.stride(0), b, b.stride(0), A_log, dt_bias, o, o.stride(0), h,
                               state_idx, pk, pv, pg, pb, n_pend, slot, par, qsl,
                               bnd_idx if has_bnd else n_pend, bnd_block if has_bnd else n_pend,
                               S, T1, scale, h.stride(0), h.stride(1), h.stride(2),
                               H=H, HV=HV, K=K, V=V, BV=BV, HAS_BND=has_bnd, num_warps=_RC_WARPS)
    return o
