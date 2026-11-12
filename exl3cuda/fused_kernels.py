"""Fused EXL3-trellis -> int8 decode for the RTX 5090 int8 prefill linears.

Without it (exl3cuda/int8_kernels.py, plugin _int8_linear) every int8 prefill linear runs, per output slice of SLICE_N=16384
columns:   recon_q8T(trellis, K, sw[n0:n1], wqT, n0)          trellis (K bits/weight) -> int8 wqT [n_s, k] in DRAM
           gemm_i8_had(xq, wqT, sx, sw[n0:n1], svh[n0:n1], y[:, n0:n1])
This module removes (or shrinks) the int8 round trip. Same numerics, bit for bit: the int8 weights equal recon_q8T's
(same index math, codebook, column scales, rint, clamp +-127) and the epilogue is a verbatim copy of
_gemm_i8_had_kernel's, so y is torch.equal to the reference (int32 accumulation is exact and order-free).

Paths (all take the same inputs as recon_q8T + gemm_i8_had combined):
  F1  recon_gemm_had(xq, sx, trellis, K, sw_s, svh_s, y_s, n0, cfg)
      Fused: each CTA decodes its [BK, 128] B tile from the trellis inside the K loop (no int8 weights anywhere) and
      runs the int8 MMA + the scale/Hadamard epilogue. The decode is repeated once per M tile (m / BM times), so it
      pays off for small m / large BM. Optional split-K (int32 partials + a separate epilogue kernel, still exact) for
      small grids (m <= 1024, n = 5120 has only 40 N tiles).
  F2  l2_split_linear(xq, sx, trellis, K, sw, svh, y, budget)
      Two kernels like today, but (a) recon_fast: ALU-optimised decode (index math hoisted out of the K loop,
      dp4a byte sum, funnel shift, no fp16/int conversions) and (b) slices small enough (n_s * k <= budget, default
      32 MB, L2 is 96 MB) that the int8 slice stays in L2 between the decode and the GEMM: no DRAM write/read of int8.
  recon_fast(trellis, K, sw_s, out, n0)   drop-in replacement for int8_kernels.recon_q8T (bit-identical output).
  fused_linear(...)                       hybrid dispatcher (F1 for m <= EXL3_INT8_FUSED_MAX_M, else F2).

Decode of one weight (verified exhaustively on CPU for all 1021 codebook sums):
  st   = 16-bit window of the tile's circular bit stream     (shf.l.wrap funnel shift; reference index math)
  x    = st * 0x83DCD12D (mod 2^32);  F = dp4a(x, 0x01010101, 0x4B000000) = bits of float(2^23 + bytesum(x))
  v    = fma(F, A, D) = A*bytesum + (1024A - B)    exact in fp32 (A = 0.00676727294921875, B = 10.3828125)
  v16  = v rounded to fp16 precision by integer RNE on the fp32 bits (exact for this value set, no cvt)
  q    = low byte of bits(clamp(v16 * (1/sw), +-127) + 1.5*2^23)        (= clamp(rint(.)), no cvt)
ASM=False is a pure-Triton spelling of the same function (64-bit funnel shift, explicit byte sum, int->float cvt):
slower, but independent of the inline PTX, so it doubles as a cross-check.
"""
import os
import torch
import triton
import triton.language as tl

if __package__:                        # copied into the plugin package (exl3cuda/)
    from . import int8_kernels as K8
else:                                  # bench: /x (the live exl3cuda/ source) on sys.path, same module as the reference
    import int8_kernels as K8

# (jit functions may only read constexpr globals)
_A = tl.constexpr(0.00676727294921875)  # codebook slope (fp16-exact: 887 * 2^-17)
_D = tl.constexpr(-56771.453125)        # (1024*A - B) - A*2^23, exact in fp32
_C = tl.constexpr(-3.453125)            # 1024*A - B

# t = (w0:w1 << (off & 31)) >> 32 ; st = t >> 16 ; x = st * mul1 ; F = dp4a(x, 1x4, 0x4B000000)
# $1 = w0 (bits 63:32), $2 = w1 (bits 31:0), $3 = shift (only the low 5 bits are used: .wrap)
_ASM_DECODE = tl.constexpr(
    "{\n"
    ".reg .b32 t, c, d;\n"
    "shf.l.wrap.b32 t, $2, $1, $3;\n"
    "shr.b32 t, t, 16;\n"
    "mov.b32 c, 0x83DCD12D;\n"
    "mul.lo.u32 t, t, c;\n"
    "mov.b32 c, 0x01010101;\n"
    "mov.b32 d, 0x4B000000;\n"
    "dp4a.u32.u32 $0, t, c, d;\n"
    "}"
)


@triton.jit
def _jr(rr):
    # tensor-core fragment order, row part (verbatim from _recon_q8T_kernel)
    return (rr & 1) + 8 * ((rr >> 1) & 3) + 2 * (rr >> 3)


@triton.jit
def _jc(cc):
    return 4 * (cc >> 3) + 32 * (cc & 7)


@triton.jit
def _stream_offsets(j, tile, K: tl.constexpr):
    """j: stream index of each element within its 16x16 tile, tile: tile index relative to the chunk's first tile row
    (r_local * tiles_n + c). Returns (offA2, offB): offA2 = (word offset of w0) << 5 | bit offset, offB = word offset
    of w1 (circular within the tile). Same formulas as _recon_q8T_kernel."""
    WORDS: tl.constexpr = 8 * K
    b = ((j + 1) * K - 16 + 32 * WORDS) % (32 * WORDS)
    wi = b // 32
    off = b % 32
    base = tile * WORDS
    offA2 = ((base + wi) << 5) | off
    offB = base + (wi + 1) % WORDS
    return offA2, offB


@triton.jit
def _decode(tr_k, offA2, offB, inv, ASM: tl.constexpr):
    """int8 weights of one chunk. tr_k: scalar int32 pointer at the chunk's first tile row; inv: 1/sw broadcastable."""
    w0 = tl.load(tr_k + (offA2 >> 5)).to(tl.uint32, bitcast=True)
    w1 = tl.load(tr_k + offB).to(tl.uint32, bitcast=True)
    if ASM:
        f = tl.inline_asm_elementwise(_ASM_DECODE, "=r,r,r,r", [w0, w1, offA2.to(tl.uint32, bitcast=True)],
                                      dtype=tl.uint32, is_pure=True, pack=1)
        F = f.to(tl.float32, bitcast=True)
        v = tl.fma(F, tl.zeros_like(F) + _A, tl.zeros_like(F) + _D)        # exact (see module doc)
    else:
        off = (offA2 & 31).to(tl.uint32, bitcast=True)
        x64 = (w0.to(tl.uint64) << 32) | w1.to(tl.uint64)
        st = ((x64 >> (48 - off).to(tl.uint64)) & 0xFFFF).to(tl.uint32)
        x = st * 0x83DCD12D
        bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
        v = bsum.to(tl.float32) * _A + _C                                  # bsum*A exact; one rounding of an exact value
    # round to fp16 precision (RNE on the 13 dropped mantissa bits; exact for |v| < 4 with 2^-17 resolution)
    u = v.to(tl.uint32, bitcast=True)
    u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
    p = u.to(tl.float32, bitcast=True) * inv
    p = tl.minimum(tl.maximum(p, -127.0), 127.0)                           # also blocks mul+add contraction
    r = (p + 12582912.0).to(tl.uint32, bitcast=True)                       # 1.5 * 2^23: integer RNE in the low bits
    return r.to(tl.int8)                                                   # low byte = clamp(rint(v * inv))


# ------------------------------------------------------------------------------------------------------------------
# recon_fast: trellis -> int8 [n_s, k] (bit-identical to recon_q8T), tile [BN, BK] (k contiguous: vector stores),
# index tensors computed once per program, K loop over a k range (grid = n_s/BN x KSPLIT).
@triton.jit
def _recon_fast_kernel(tr_ptr, sw_ptr, o_ptr, k, n0, tiles_n, k_chunk, stride_on, K: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, ASM: tl.constexpr):
    WORDS: tl.constexpr = 8 * K
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    onl = pn * BN + tl.arange(0, BN)
    on = n0 + onl
    ok = tl.arange(0, BK)
    j = _jr(ok % 16)[None, :] + _jc(on % 16)[:, None]
    tile = ((ok // 16) * tiles_n)[None, :] + (on // 16)[:, None]
    offA2, offB = _stream_offsets(j, tile, K)
    inv = (1.0 / tl.load(sw_ptr + onl))[:, None]
    k_lo = pk * k_chunk
    k_hi = tl.minimum(k_lo + k_chunk, k)
    tr_k = tr_ptr + (k_lo // 16).to(tl.int64) * tiles_n * WORDS
    step = (BK // 16) * tiles_n * WORDS
    o_ptrs = o_ptr + onl[:, None].to(tl.int64) * stride_on + k_lo + ok[None, :]
    for _ in range(k_lo, k_hi, BK):
        q = _decode(tr_k, offA2, offB, inv, ASM)
        tl.store(o_ptrs, q)
        o_ptrs += BK
        tr_k += step


def recon_fast(trellis: torch.Tensor, K: int, sw: torch.Tensor, out: torch.Tensor, n0: int = 0, BN: int = 64,
               BK: int = 128, asm: bool = True, num_warps: int = 4, min_programs: int = 680):
    """Drop-in for int8_kernels.recon_q8T: out[n_s, k] int8 (sw = the slice's column scales, n0 = slice start)."""
    kt, nt = trellis.shape[0], trellis.shape[1]
    k = kt * 16
    n_s = out.shape[0]
    assert n0 % 16 == 0 and out.shape[1] == k and out.stride(1) == 1 and trellis.is_contiguous()
    assert k % BK == 0 and n_s % BN == 0, "recon_fast: k % BK and n_s % BN must be 0"
    nb = n_s // BN
    ksplit = max(1, min(k // BK, triton.cdiv(min_programs, nb)))
    k_chunk = triton.cdiv(k // BK, ksplit) * BK
    ksplit = triton.cdiv(k, k_chunk)
    _recon_fast_kernel[(nb, ksplit)](trellis.view(torch.int32), sw, out, k, n0, nt, k_chunk, out.stride(0), K=K, BN=BN,
                                     BK=BK, ASM=asm, num_warps=num_warps)
    return out


# ------------------------------------------------------------------------------------------------------------------
# F1: fused decode + int8 GEMM + scale/Hadamard epilogue. CTA order: all M tiles of one N tile are adjacent, so the
# trellis bytes of an N tile are read from DRAM once and the repeats hit L2.
@triton.jit
def _fused_kernel(a_ptr, tr_ptr, sx_ptr, sw_ptr, svh_ptr, h_ptr, y_ptr, ws_ptr, m, n, k_len, n0, tiles_n, stride_am,
                  stride_ym, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                  ASM: tl.constexpr):
    BN: tl.constexpr = 128
    WORDS: tl.constexpr = 8 * K
    pid = tl.program_id(0)
    ps = tl.program_id(1)
    num_pm = tl.cdiv(m, BM)
    pm = pid % num_pm
    pn = pid // num_pm
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)                  # columns within the slice (as in _gemm_i8_had_kernel)
    ona = n0 + on                                    # absolute columns (trellis)
    ok = tl.arange(0, BK)
    j = _jr(ok % 16)[:, None] + _jc(ona % 16)[None, :]
    tile = ((ok // 16) * tiles_n)[:, None] + (ona // 16)[None, :]
    offA2, offB = _stream_offsets(j, tile, K)
    inv = (1.0 / tl.load(sw_ptr + on))[None, :]
    k_lo = ps * k_len
    tr_k = tr_ptr + (k_lo // 16).to(tl.int64) * tiles_n * WORDS
    step = (BK // 16) * tiles_n * WORDS
    a_ptrs = a_ptr + om[:, None].to(tl.int64) * stride_am + k_lo + ok[None, :]
    amask = om[:, None] < m
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, k_len, BK):
        a = tl.load(a_ptrs, mask=amask, other=0)
        b = _decode(tr_k, offA2, offB, inv, ASM)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BK
        tr_k += step
    if SPLIT == 1:
        # ---- epilogue: verbatim from _gemm_i8_had_kernel ----
        sx = tl.load(sx_ptr + om, mask=om < m, other=0.0)
        sw = tl.load(sw_ptr + on, mask=on < n, other=0.0)
        v = (acc.to(tl.float32) * sx[:, None] * sw[None, :]).to(tl.float16)
        hr = tl.arange(0, BN)
        h = tl.load(h_ptr + hr[:, None] * BN + hr[None, :])
        yv = tl.dot(v, h, out_dtype=tl.float32)
        svh = tl.load(svh_ptr + on, mask=on < n, other=0.0).to(tl.float32)
        tl.store(y_ptr + om[:, None] * stride_ym + on[None, :], (yv * svh[None, :]).to(tl.float16),
                 mask=(om[:, None] < m) & (on[None, :] < n))
    else:
        wp = ws_ptr + (ps * m + om[:, None]).to(tl.int64) * n + on[None, :]
        tl.store(wp, acc, mask=om[:, None] < m)


@triton.jit
def _splitk_epi_kernel(ws_ptr, sx_ptr, sw_ptr, svh_ptr, h_ptr, y_ptr, m, n, stride_ym, SPLIT: tl.constexpr,
                       BM: tl.constexpr):
    BN: tl.constexpr = 128
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(ws_ptr + (s * m + om[:, None]).to(tl.int64) * n + on[None, :], mask=om[:, None] < m, other=0)
    # ---- epilogue: verbatim from _gemm_i8_had_kernel ----
    sx = tl.load(sx_ptr + om, mask=om < m, other=0.0)
    sw = tl.load(sw_ptr + on, mask=on < n, other=0.0)
    v = (acc.to(tl.float32) * sx[:, None] * sw[None, :]).to(tl.float16)
    hr = tl.arange(0, BN)
    h = tl.load(h_ptr + hr[:, None] * BN + hr[None, :])
    yv = tl.dot(v, h, out_dtype=tl.float32)
    svh = tl.load(svh_ptr + on, mask=on < n, other=0.0).to(tl.float32)
    tl.store(y_ptr + om[:, None] * stride_ym + on[None, :], (yv * svh[None, :]).to(tl.float16),
             mask=(om[:, None] < m) & (on[None, :] < n))


# F1 configs measured by bench_fused.py (name -> kwargs). BN is fixed at 128 (Hadamard block).
FUSED_CONFIGS = {
    "bm64_bk64_w4_s3": dict(BM=64, BK=64, num_warps=4, num_stages=3),
    "bm64_bk128_w4_s3": dict(BM=64, BK=128, num_warps=4, num_stages=3),
    "bm128_bk64_w4_s3": dict(BM=128, BK=64, num_warps=4, num_stages=3),
    "bm128_bk64_w8_s3": dict(BM=128, BK=64, num_warps=8, num_stages=3),
    "bm128_bk128_w8_s3": dict(BM=128, BK=128, num_warps=8, num_stages=3),
    "bm256_bk64_w8_s3": dict(BM=256, BK=64, num_warps=8, num_stages=3),
    "bm256_bk64_w16_s2": dict(BM=256, BK=64, num_warps=16, num_stages=2),
    "bm256_bk128_w16_s2": dict(BM=256, BK=128, num_warps=16, num_stages=2),
}
SPLITK_CONFIGS = {                     # used for m <= 1024 only
    "bm64_bk64_w4_s3_sk2": dict(BM=64, BK=64, num_warps=4, num_stages=3, SPLIT=2),
    "bm64_bk64_w4_s3_sk4": dict(BM=64, BK=64, num_warps=4, num_stages=3, SPLIT=4),
    "bm128_bk64_w8_s3_sk2": dict(BM=128, BK=64, num_warps=8, num_stages=3, SPLIT=2),
    "bm128_bk64_w8_s3_sk4": dict(BM=128, BK=64, num_warps=8, num_stages=3, SPLIT=4),
}
_WS: dict = {}


def _ws(dev, numel):
    b = _WS.get(dev)
    if b is None or b.numel() < numel:
        b = _WS[dev] = torch.empty(numel, dtype=torch.int32, device=dev)
    return b[:numel]


def default_cfg(m: int, n_s: int) -> dict:
    """Placeholder pick until bench_fused.py has run (update from its per-m winners)."""
    if m <= 256:
        return dict(FUSED_CONFIGS["bm256_bk64_w16_s2"])
    return dict(FUSED_CONFIGS["bm256_bk64_w8_s3"])


def recon_gemm_had(xq, sx, trellis, K: int, sw, svh, y, n0: int = 0, cfg: dict = None, asm: bool = True):
    """= recon_q8T(trellis, K, sw, wqT, n0) ; gemm_i8_had(xq, wqT, sx, sw, svh, y) without materialising wqT.
    xq int8 [m, k], sx fp32 [m], trellis int16 [k/16, n/16, 16K] (whole linear), sw fp32 [n_s] / svh fp16 [n_s] =
    the slice's scales, y fp16 [m, n_s] (may be a column slice of a wider tensor), n0 = absolute first column."""
    m, k = xq.shape
    n_s = sw.shape[0]
    kt, nt = trellis.shape[0], trellis.shape[1]
    assert kt * 16 == k and n_s % 128 == 0 and n0 % 16 == 0 and n0 + n_s <= nt * 16
    assert xq.stride(1) == 1 and y.stride(1) == 1 and y.shape == (m, n_s) and trellis.is_contiguous()
    cfg = dict(cfg or default_cfg(m, n_s))
    split = cfg.pop("SPLIT", 1)
    BM, BK = cfg.pop("BM"), cfg.pop("BK")
    if k % (BK * split):
        split = 1
    assert k % BK == 0
    k_len = k // split
    h = K8.h128(xq.device)
    ws = _ws(xq.device, split * m * n_s) if split > 1 else y
    grid = (triton.cdiv(m, BM) * (n_s // 128), split)
    _fused_kernel[grid](xq, trellis.view(torch.int32), sx, sw, svh, h, y, ws, m, n_s, k_len, n0, nt, xq.stride(0),
                        y.stride(0), K=K, BM=BM, BK=BK, SPLIT=split, ASM=asm, **cfg)
    if split > 1:
        EBM = 64
        _splitk_epi_kernel[(triton.cdiv(m, EBM), n_s // 128)](ws, sx, sw, svh, h, y, m, n_s, y.stride(0), SPLIT=split,
                                                              BM=EBM, num_warps=4)
    return y


# ------------------------------------------------------------------------------------------------------------------
# F2: L2-resident split (recon_fast + the existing gemm_i8_had per small slice).
_WBUF: dict = {}


def _wbuf(dev, numel):
    b = _WBUF.get(dev)
    if b is None or b.numel() < numel:
        b = _WBUF[dev] = torch.empty(numel, dtype=torch.int8, device=dev)
    return b[:numel]


def slice_cols(k: int, n: int, budget: int) -> int:
    """Columns per slice so that the int8 slice (n_s * k bytes) fits the L2 budget (multiple of 128)."""
    return max(128, min(n, budget // k // 128 * 128))


def l2_split_linear(xq, sx, trellis, K: int, sw, svh, y, budget: int = 32 << 20, asm: bool = True, recon=None):
    """Whole linear: y[m, n] = int8 linear, slices of slice_cols(k, n, budget) columns. recon defaults to recon_fast;
    pass int8_kernels.recon_q8T to measure the slicing alone."""
    m, k = xq.shape
    n = sw.shape[0]
    ns = slice_cols(k, n, budget)
    buf = _wbuf(xq.device, ns * k)
    for n0 in range(0, n, ns):
        n1 = min(n0 + ns, n)
        wqT = buf[: (n1 - n0) * k].view(n1 - n0, k)
        if recon is None:
            recon_fast(trellis, K, sw[n0:n1], wqT, n0, asm=asm)
        else:
            recon(trellis, K, sw[n0:n1], wqT, n0)
        K8.gemm_i8_had(xq, wqT, sx, sw[n0:n1], svh[n0:n1], y[:, n0:n1])
    return y


# ------------------------------------------------------------------------------------------------------------------
# Hybrid dispatcher for the plugin. Thresholds/configs to be set from a benchmark.
FUSED_MAX_M = int(os.environ.get("EXL3_INT8_FUSED_MAX_M", "512"))
L2_BUDGET = int(os.environ.get("EXL3_INT8_L2_BUDGET_MB", "32")) << 20
FUSED_ASM = os.environ.get("EXL3_INT8_FUSED_ASM", "1") == "1"


def fused_linear(xq, sx, trellis, K: int, sw, svh, y):
    """Whole linear (sw/svh/y cover all n columns of this linear): F1 for m <= FUSED_MAX_M, else F2."""
    m = xq.shape[0]
    if m <= FUSED_MAX_M:
        return recon_gemm_had(xq, sx, trellis, K, sw, svh, y, 0, asm=FUSED_ASM)
    return l2_split_linear(xq, sx, trellis, K, sw, svh, y, L2_BUDGET, asm=FUSED_ASM)
