"""Triton kernels for int8 (W8A8) prefill GEMMs on the RTX 5090.

Same numerics as the Intel Arc (XPU) plugin's EXL3_INT8_PREFILL: activations in the Hadamard basis quantized per token (symmetric,
amax/127), reconstructed rotated-basis weights quantized per output column with static scales, int32 accumulate.

  quant_rows(x)               fp16 [m, k]            -> int8 [m, k], fp32 scale [m]
  quant_wT(w, sw, out)        fp16 [k, n] row-major  -> int8 [n, k] (= column-major [k, n], what cuBLASLt IMMA wants)
  dequant(acc, sx, sw, y)     int32 [m, n] * sx[m] * sw[n] -> fp16 y (may be a column slice of a wider tensor)
  gemm_i8_scaled(xq, wqT, sx, sw, y)  int8 GEMM with the scale epilogue fused (no int32 round trip)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, k, stride_x, stride_q, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    xr = x_ptr + row * stride_x
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, k, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(xr + offs, mask=offs < k, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
    s = tl.maximum(tl.max(amax, axis=0), 1e-8) / 127.0
    tl.store(s_ptr + row, s)
    inv = 1.0 / s
    qr = q_ptr + row * stride_q
    for k0 in range(0, k, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(xr + offs, mask=offs < k, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.rint(v * inv)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(qr + offs, q.to(tl.int8), mask=offs < k)


def quant_rows(x: torch.Tensor):
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(m,)](x, q, s, k, x.stride(0), q.stride(0), BLOCK=1024, num_warps=4)
    return q, s


@triton.jit
def _had_rot(x_ptr, suh_ptr, h, om, k0, oc, rmask, stride_x):
    xb = tl.load(x_ptr + om[:, None] * stride_x + k0 + oc[None, :], mask=rmask, other=0.0)
    su = tl.load(suh_ptr + k0 + oc)
    # (x * suh) @ H128/sqrt(128), rounded to fp16 like exllamav3's had_r_128 output
    return tl.dot(xb * su[None, :], h, out_dtype=tl.float32).to(tl.float16).to(tl.float32)


@triton.jit
def _had_amax_kernel(x_ptr, suh_ptr, h_ptr, amax_ptr, m, k, stride_x, BM: tl.constexpr, KCH: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    oc = tl.arange(0, 128)
    rmask = om[:, None] < m
    h = tl.load(h_ptr + oc[:, None] * 128 + oc[None, :])
    amax = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(pk * KCH, min(pk * KCH + KCH, k), 128):
        amax = tl.maximum(amax, tl.max(tl.abs(_had_rot(x_ptr, suh_ptr, h, om, k0, oc, rmask, stride_x)), axis=1))
    tl.atomic_max(amax_ptr + om, amax, mask=om < m)          # amax >= 0: float order = int order, atomics are exact


@triton.jit
def _had_q_kernel(x_ptr, suh_ptr, h_ptr, amax_ptr, q_ptr, s_ptr, m, k, stride_x, stride_q, BM: tl.constexpr,
                  KCH: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    oc = tl.arange(0, 128)
    rmask = om[:, None] < m
    h = tl.load(h_ptr + oc[:, None] * 128 + oc[None, :])
    s = tl.maximum(tl.load(amax_ptr + om, mask=om < m, other=1.0), 1e-8) / 127.0
    if pk == 0:
        tl.store(s_ptr + om, s, mask=om < m)
    inv = 1.0 / s
    for k0 in range(pk * KCH, min(pk * KCH + KCH, k), 128):
        r = _had_rot(x_ptr, suh_ptr, h, om, k0, oc, rmask, stride_x)
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(r * inv[:, None]), -127.0), 127.0)
        tl.store(q_ptr + om[:, None] * stride_q + k0 + oc[None, :], q.to(tl.int8), mask=rmask)


def had_quant(x: torch.Tensor, suh: torch.Tensor, BM: int = 32, KCH: int = 1024):
    """= quant_rows(had_r_128(x, suh)) without the fp16 intermediate: pass 1 rotates each 128-column block
    and reduces the row amax (split over k, atomic max), pass 2 rotates again and stores int8. x is read twice (mostly
    from L2), the fp16 rotated activations are never written. x fp16 [m, k] (k % 128 == 0), suh [k]."""
    m, k = x.shape
    assert k % 128 == 0 and x.stride(1) == 1
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    amax = torch.zeros((m,), dtype=torch.float32, device=x.device)
    suh = suh.half().contiguous()
    h = h128(x.device)
    grid = (triton.cdiv(m, BM), triton.cdiv(k, KCH))
    _had_amax_kernel[grid](x, suh, h, amax, m, k, x.stride(0), BM=BM, KCH=KCH, num_warps=4)
    _had_q_kernel[grid](x, suh, h, amax, q, s, m, k, x.stride(0), q.stride(0), BM=BM, KCH=KCH, num_warps=4)
    return q, s


@triton.jit
def _quant_wT_kernel(w_ptr, sw_ptr, o_ptr, k, n, stride_wk, stride_on, BK: tl.constexpr, BN: tl.constexpr):
    pk = tl.program_id(0)
    pn = tl.program_id(1)
    ok = pk * BK + tl.arange(0, BK)
    on = pn * BN + tl.arange(0, BN)
    mask = (ok[:, None] < k) & (on[None, :] < n)
    w = tl.load(w_ptr + ok[:, None] * stride_wk + on[None, :], mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.load(sw_ptr + on, mask=on < n, other=1.0)
    q = tl.extra.cuda.libdevice.rint(w * inv[None, :])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    # transposed store: out[n, k]
    tl.store(o_ptr + on[None, :] * stride_on + ok[:, None], q, mask=mask)


def quant_wT(w: torch.Tensor, sw: torch.Tensor, out: torch.Tensor):
    k, n = w.shape
    BK, BN = 64, 64
    grid = (triton.cdiv(k, BK), triton.cdiv(n, BN))
    _quant_wT_kernel[grid](w, sw, out, k, n, w.stride(0), out.stride(0), BK=BK, BN=BN, num_warps=4)
    return out


@triton.jit
def _recon_q8T_kernel(tr_ptr, sw_ptr, o_ptr, k, n_slice, n0, tiles_n, stride_on, K: tl.constexpr, BK: tl.constexpr,
                      BN: tl.constexpr, F64: tl.constexpr):
    """EXL3 trellis (mul1 codebook) -> int8 [n, k] in one pass (the XPU plugin's reconstruct_q8 idea): decode each
    weight's 16-bit state from its tile's bit stream, the codebook value (bit-exact with exllamav3's decode_3inst<2>,
    via a 2^16-entry LUT), quantize with the static column scale and store transposed. No fp16 weight round trip."""
    pk = tl.program_id(0)
    pn = tl.program_id(1)
    ok = pk * BK + tl.arange(0, BK)                    # rows of W (input features)
    onl = pn * BN + tl.arange(0, BN)                   # columns within the slice
    on = n0 + onl                                      # absolute output columns
    mask = (ok[:, None] < k) & (onl[None, :] < n_slice)
    WORDS: tl.constexpr = 8 * K
    r = ok // 16
    c = on // 16
    rr = ok % 16
    cc = on % 16
    # stream index of tile position (rr, cc): exllamav3's tensor-core fragment order
    j = ((rr & 1) + 8 * ((rr >> 1) & 3) + 2 * (rr >> 3))[:, None] + (4 * (cc >> 3) + 32 * (cc & 7))[None, :]
    b = ((j + 1) * K - 16 + 32 * WORDS) % (32 * WORDS)  # first bit of the 16-bit window (circular per tile)
    wi = b // 32
    off = b % 32
    base = (r[:, None].to(tl.int64) * tiles_n + c[None, :]) * WORDS
    w0 = tl.load(tr_ptr + base + wi, mask=mask, other=0).to(tl.uint32, bitcast=True)
    w1 = tl.load(tr_ptr + base + (wi + 1) % WORDS, mask=mask, other=0).to(tl.uint32, bitcast=True)
    offu = off.to(tl.uint32)
    hi = tl.where(off <= 16, w0 >> (16 - offu), (w0 << (offu - 16)) | (w1 >> (48 - offu)))
    st = hi & 0xFFFF
    # codebook: x = st * 0x83DCD12D (mod 2^32); fp16 bits 0x6400 + byte sum of x; value = h * 0.006767 - 10.3828 (fp16)
    x = st * 0x83DCD12D
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    h = (bsum + 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
    if F64:     # exact product + one rounding, like the fp16 fma
        v = (h.to(tl.float64) * 0.00676727294921875 - 10.3828125).to(tl.float16).to(tl.float32)
    else:       # h * a is exact in fp32 (11 x 11 bits); fp32 add then fp16 rounding (checked bit-exact on all weights)
        v = (h.to(tl.float32) * 0.00676727294921875 - 10.3828125).to(tl.float16).to(tl.float32)
    inv = 1.0 / tl.load(sw_ptr + onl, mask=onl < n_slice, other=1.0)
    q = tl.extra.cuda.libdevice.rint(v * inv[None, :])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(o_ptr + onl[None, :] * stride_on + ok[:, None], q, mask=mask)


def recon_q8T(trellis: torch.Tensor, K: int, sw: torch.Tensor, out: torch.Tensor, n0: int = 0, BK: int = 64, BN: int = 64,
              F64: bool = False, num_warps: int = 4):
    """out[n_slice, k] int8 = quant(reconstruct(trellis)[:, n0:n0+n_slice] / sw) (mul1 codebook only).
    trellis int16 [k/16, n/16, 16*K]; sw fp32 [n_slice] (the slice's column scales); out may be a [n_slice, k] buffer."""
    kt, nt = trellis.shape[0], trellis.shape[1]
    k = kt * 16
    n_slice = out.shape[0]
    assert n0 % 16 == 0 and out.shape[1] == k and out.stride(1) == 1
    words = trellis.view(torch.int32)
    grid = (triton.cdiv(k, BK), triton.cdiv(n_slice, BN))
    _recon_q8T_kernel[grid](words, sw, out, k, n_slice, n0, nt, out.stride(0), K=K, BK=BK, BN=BN, F64=F64,
                            num_warps=num_warps)
    return out


@triton.jit
def _dequant_kernel(a_ptr, sx_ptr, sw_ptr, y_ptr, m, n, stride_a, stride_y, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    mask = (om[:, None] < m) & (on[None, :] < n)
    a = tl.load(a_ptr + om[:, None] * stride_a + on[None, :], mask=mask, other=0).to(tl.float32)
    sx = tl.load(sx_ptr + om, mask=om < m, other=0.0)
    sw = tl.load(sw_ptr + on, mask=on < n, other=0.0)
    tl.store(y_ptr + om[:, None] * stride_y + on[None, :], (a * sx[:, None] * sw[None, :]).to(tl.float16), mask=mask)


def dequant(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, y: torch.Tensor):
    m, n = acc.shape
    BM, BN = 64, 128
    _dequant_kernel[(triton.cdiv(m, BM), triton.cdiv(n, BN))](acc, sx, sw, y, m, n, acc.stride(0), y.stride(0),
                                                             BM=BM, BN=BN, num_warps=4)
    return y


_GEMM_CONFIGS = [
    triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP": 8}, num_stages=4, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP": 8}, num_stages=5, num_warps=4),
]


@triton.autotune(configs=_GEMM_CONFIGS, key=["m_bucket", "n", "k"], cache_results=True)
@triton.jit
def _gemm_i8_kernel(a_ptr, b_ptr, sx_ptr, sw_ptr, y_ptr, m, n, k, m_bucket, stride_am, stride_bn, stride_ym,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_pm = tl.cdiv(m, BM)
    num_pn = tl.cdiv(n, BN)
    group_size = GROUP * num_pn
    gid = pid // group_size
    first_pm = gid * GROUP
    gm = min(num_pm - first_pm, GROUP)
    pm = first_pm + ((pid % group_size) % gm)
    pn = (pid % group_size) // gm
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * stride_am + ok[None, :]
    b_ptrs = b_ptr + on[None, :] * stride_bn + ok[:, None]          # B stored [n, k]: load a [BK, BN] tile
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, k, BK):
        a = tl.load(a_ptrs, mask=(om[:, None] < m) & (ok[None, :] + k0 < k), other=0)
        b = tl.load(b_ptrs, mask=(on[None, :] < n) & (ok[:, None] + k0 < k), other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK
    sx = tl.load(sx_ptr + om, mask=om < m, other=0.0)
    sw = tl.load(sw_ptr + on, mask=on < n, other=0.0)
    out = (acc.to(tl.float32) * sx[:, None] * sw[None, :]).to(tl.float16)
    tl.store(y_ptr + om[:, None] * stride_ym + on[None, :], out, mask=(om[:, None] < m) & (on[None, :] < n))


def gemm_i8_scaled(xq: torch.Tensor, wqT: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, y: torch.Tensor):
    """y[m, n] (fp16, may be a column slice) = (xq[m, k] @ wqT[n, k].T) * sx[m, None] * sw[None, n]."""
    m, k = xq.shape
    n = wqT.shape[0]
    m_bucket = 1 << max(8, (m - 1).bit_length())      # autotune per power-of-two row bucket
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _gemm_i8_kernel[grid](xq, wqT, sx, sw, y, m, n, k, m_bucket, xq.stride(0), wqT.stride(0), y.stride(0))
    return y


# ------------------------------------------------------------------------------------------------
# int8 GEMM with the scale epilogue AND exllamav3's output Hadamard fused: had_r_128(y, y, None, svh)
# is y_blk <- (y_blk @ H128 / sqrt(128)) * svh per 128-column block, so with BN = 128 tiles aligned to 128 columns
# the epilogue does one more 128x128 fp16 tensor-core product and the svh multiply; y may be a column slice
# (fused-module members write their own range: no cat).
_HAD_CONFIGS = [
    triton.Config({"BM": 128, "BK": 128, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BK": 128, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 256, "BK": 128, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 128, "BK": 64, "GROUP": 8}, num_stages=5, num_warps=4),
    triton.Config({"BM": 64, "BK": 128, "GROUP": 8}, num_stages=4, num_warps=4),
]


@triton.autotune(configs=_HAD_CONFIGS, key=["m_bucket", "n", "k"], cache_results=True)
@triton.jit
def _gemm_i8_had_kernel(a_ptr, b_ptr, sx_ptr, sw_ptr, svh_ptr, h_ptr, y_ptr, m, n, k, m_bucket, stride_am, stride_bn,
                        stride_ym, BM: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    BN: tl.constexpr = 128
    pid = tl.program_id(0)
    num_pm = tl.cdiv(m, BM)
    num_pn = tl.cdiv(n, BN)
    group_size = GROUP * num_pn
    gid = pid // group_size
    first_pm = gid * GROUP
    gm = min(num_pm - first_pm, GROUP)
    pm = first_pm + ((pid % group_size) % gm)
    pn = (pid % group_size) // gm
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * stride_am + ok[None, :]
    b_ptrs = b_ptr + on[None, :] * stride_bn + ok[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, k, BK):
        a = tl.load(a_ptrs, mask=(om[:, None] < m) & (ok[None, :] + k0 < k), other=0)
        b = tl.load(b_ptrs, mask=(on[None, :] < n) & (ok[:, None] + k0 < k), other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK
    sx = tl.load(sx_ptr + om, mask=om < m, other=0.0)
    sw = tl.load(sw_ptr + on, mask=on < n, other=0.0)
    v = (acc.to(tl.float32) * sx[:, None] * sw[None, :]).to(tl.float16)   # = the fp16 y the separate had kernel read
    hr = tl.arange(0, BN)
    h = tl.load(h_ptr + hr[:, None] * BN + hr[None, :])                    # H128 / sqrt(128), fp16
    yv = tl.dot(v, h, out_dtype=tl.float32)
    svh = tl.load(svh_ptr + on, mask=on < n, other=0.0).to(tl.float32)
    tl.store(y_ptr + om[:, None] * stride_ym + on[None, :], (yv * svh[None, :]).to(tl.float16),
             mask=(om[:, None] < m) & (on[None, :] < n))


_H128 = {}


def h128(device):
    h = _H128.get(device)
    if h is None:
        H = torch.ones(1, 1)
        while H.shape[0] < 128:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        h = _H128[device] = (H / 128 ** 0.5).to(device=device, dtype=torch.float16).contiguous()
    return h


def gemm_i8_had(xq, wqT, sx, sw, svh, y):
    """y[m, n] (fp16, may be a column slice; n % 128 == 0) = had_out((xq @ wqT.T) * sx * sw, svh)."""
    m, k = xq.shape
    n = wqT.shape[0]
    assert n % 128 == 0
    m_bucket = 1 << max(8, (m - 1).bit_length())
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, 128),)
    _gemm_i8_had_kernel[grid](xq, wqT, sx, sw, svh, h128(xq.device), y, m, n, k, m_bucket, xq.stride(0),
                              wqT.stride(0), y.stride(0))
    return y


# ------------------------------------------------------------------------------------------------
# Fused draft sampling (the XPU plugin's exl3_draft_sample idea, CUDA-graph safe): upstream does temperature div,
# softmax (2 passes), probs.clone(), probs / q and argmax as separate kernels over [rows, 248320] fp32. Here one program
# per row: pass 1 max/sum of logits/T, pass 2 writes probs and keeps argmax(probs / q). The exponential noise q still
# comes from torch's q.exponential_() (graph-safe Philox offsets), so the distribution is exactly upstream's.
@triton.jit
def _ds_stats_kernel(l_ptr, t_ptr, m_ptr, s_ptr, V, stride_l, CH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    nc = tl.num_programs(1)
    inv_t = 1.0 / tl.load(t_ptr + row)
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK,), tl.float32)
    for v0 in range(c * CH, c * CH + CH, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        x = tl.load(l_ptr + row * stride_l + offs, mask=offs < V, other=float("-inf")).to(tl.float32) * inv_t
        m_new = tl.maximum(m, x)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        s = s * alpha + tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        m = m_new
    mx = tl.max(m, axis=0)
    tot = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - mx)), axis=0)
    tl.store(m_ptr + row * nc + c, mx)
    tl.store(s_ptr + row * nc + c, tot)


@triton.jit
def _ds_write_kernel(l_ptr, t_ptr, q_ptr, p_ptr, m_ptr, s_ptr, bv_ptr, bi_ptr, V, stride_l, stride_q, stride_p,
                     NC: tl.constexpr, CH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1)
    inv_t = 1.0 / tl.load(t_ptr + row)
    cm = tl.load(m_ptr + row * NC + tl.arange(0, NC))
    cs = tl.load(s_ptr + row * NC + tl.arange(0, NC))
    mx = tl.max(cm, axis=0)
    tot = tl.sum(tl.where(cm == float("-inf"), 0.0, cs * tl.exp(cm - mx)), axis=0)
    inv = 1.0 / tot
    best = tl.full((BLOCK,), -1.0, tl.float32)
    bidx = tl.zeros((BLOCK,), tl.int64)
    for v0 in range(c * CH, c * CH + CH, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        msk = offs < V
        x = tl.load(l_ptr + row * stride_l + offs, mask=msk, other=float("-inf")).to(tl.float32) * inv_t
        pr = tl.exp(x - mx) * inv
        tl.store(p_ptr + row * stride_p + offs, pr, mask=msk)
        qv = tl.load(q_ptr + row * stride_q + offs, mask=msk, other=1.0)
        r = tl.where(msk, pr / qv, -1.0)
        upd = r > best
        best = tl.where(upd, r, best)
        bidx = tl.where(upd, offs.to(tl.int64), bidx)
    bmax = tl.max(best, axis=0)
    tl.store(bv_ptr + row * NC + c, bmax)
    tl.store(bi_ptr + row * NC + c, tl.min(tl.where(best == bmax, bidx, V), axis=0))


_DS_NC = 64      # programs per row (two-stage reduction: one SM per row was latency-bound, 111 vs 88 us upstream)


def draft_sample(logits: torch.Tensor, temperature: torch.Tensor, q: torch.Tensor):
    """-> (next_token_ids int64 [rows], probs fp32 [rows, V]) with upstream's semantics (softmax(logits / T),
    argmax(probs / q)); q = exponential noise of shape [rows, V] (fp32)."""
    rows, V = logits.shape
    BLOCK = 1024
    CH = triton.cdiv(triton.cdiv(V, _DS_NC), BLOCK) * BLOCK
    dev = logits.device
    probs = torch.empty((rows, V), dtype=torch.float32, device=dev)
    cm = torch.empty((rows, _DS_NC), dtype=torch.float32, device=dev)
    cs = torch.empty_like(cm)
    bv = torch.empty_like(cm)
    bi = torch.empty((rows, _DS_NC), dtype=torch.int64, device=dev)
    grid = (rows, _DS_NC)
    _ds_stats_kernel[grid](logits, temperature, cm, cs, V, logits.stride(0), CH=CH, BLOCK=BLOCK, num_warps=4)
    _ds_write_kernel[grid](logits, temperature, q, probs, cm, cs, bv, bi, V, logits.stride(0), q.stride(0),
                           probs.stride(0), NC=_DS_NC, CH=CH, BLOCK=BLOCK, num_warps=4)
    tok = bi.gather(1, bv.argmax(1, keepdim=True)).view(-1)
    return tok, probs
