"""EXL3 GEMM for 17-64 rows: multi-stream MTP decode (7 rows per stream) on the RTX 5090.

exllamav3's fused trellis GEMM processes rows in 16-row blocks and decodes the weights again per block: one gate_proj
takes 37 us at 16 rows, 59 us at 17-32, 88 us at 64 (DRAM floor ~25 us). Here each weight tile is decoded once per
program (bit-exact decode of the mul1 codebook, as in int8_kernels.recon_q8T) and multiplied against all rows with
fp16 tensor-core dots, fp32 accumulate. Split-K for parallelism; the output Hadamard is linear, so each split applies
H128/sqrt(128) * svh to its partial sums and atomically adds fp32 into y32, then one cast to fp16.

  y = had_out(had_in(x, suh) @ W_inner, svh)    same math as LinearEXL3.forward
"""
import torch
import triton
import triton.language as tl

from .int8_kernels import h128


@triton.jit
def _decode_tile(tr_ptr, ok, on, tiles_n, K: tl.constexpr):
    """fp16 W_inner[ok, on] decoded from the trellis (exllamav3 mul1 codebook, bit-exact)."""
    WORDS: tl.constexpr = 8 * K
    r = ok // 16
    c = on // 16
    rr = ok % 16
    cc = on % 16
    j = ((rr & 1) + 8 * ((rr >> 1) & 3) + 2 * (rr >> 3))[:, None] + (4 * (cc >> 3) + 32 * (cc & 7))[None, :]
    b = ((j + 1) * K - 16 + 32 * WORDS) % (32 * WORDS)
    wi = b // 32
    off = (b % 32).to(tl.uint32)
    base = (r[:, None].to(tl.int64) * tiles_n + c[None, :]) * WORDS
    w0 = tl.load(tr_ptr + base + wi).to(tl.uint32, bitcast=True)
    w1 = tl.load(tr_ptr + base + (wi + 1) % WORDS).to(tl.uint32, bitcast=True)
    hi = tl.where(off <= 16, w0 >> (16 - off), (w0 << (off - 16)) | (w1 >> (48 - off)))
    x = (hi & 0xFFFF) * 0x83DCD12D
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    h = (bsum + 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
    return (h.to(tl.float32) * 0.00676727294921875 - 10.3828125).to(tl.float16)


@triton.jit
def _exl3_mma_kernel(x_ptr, tr_ptr, svh_ptr, h_ptr, y_ptr, m, k, n, tiles_n, stride_x, stride_y, k_per_split,
                     K: tl.constexpr, MP: tl.constexpr, BK: tl.constexpr):
    BN: tl.constexpr = 128
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    om = tl.arange(0, MP)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((MP, BN), dtype=tl.float32)
    k_lo = ps * k_per_split
    for k0 in range(k_lo, k_lo + k_per_split, BK):
        ok = k0 + tl.arange(0, BK)
        xb = tl.load(x_ptr + om[:, None] * stride_x + ok[None, :], mask=om[:, None] < m, other=0.0)
        w = _decode_tile(tr_ptr, ok, on, tiles_n, K)
        acc = tl.dot(xb, w, acc, out_dtype=tl.float32)
    hr = tl.arange(0, BN)
    h = tl.load(h_ptr + hr[:, None] * BN + hr[None, :])
    yv = tl.dot(acc.to(tl.float16), h, out_dtype=tl.float32)     # fp16 rounding before H, like the separate had pass
    svh = tl.load(svh_ptr + on).to(tl.float32)
    tl.atomic_add(y_ptr + om[:, None] * stride_y + on[None, :], yv * svh[None, :], mask=om[:, None] < m, sem="relaxed")


def exl3_mma(xh: torch.Tensor, trellis: torch.Tensor, K: int, svh: torch.Tensor, split: int = 0, BK: int = 64,
             num_warps: int = 4, num_stages: int = 2):
    """y fp16 [m, n] = had_out(xh @ W_inner, svh); xh = had_in(x, suh) fp16 [m, k], m <= 64, n % 128 == 0."""
    m, k = xh.shape
    kt, nt = trellis.shape[0], trellis.shape[1]
    n = nt * 16
    assert k == kt * 16 and n % 128 == 0 and m <= 64 and xh.stride(1) == 1
    MP = 16 if m <= 16 else (32 if m <= 32 else 64)
    if split <= 0:                                   # enough programs for ~2 waves over 170 SMs
        split = 1
        while (n // 128) * split < 340 and k // (split * 2) >= 512 and (k // (split * 2)) % BK == 0:
            split *= 2
    assert k % split == 0 and (k // split) % BK == 0
    y32 = torch.zeros((m, n), dtype=torch.float32, device=xh.device)
    _exl3_mma_kernel[(n // 128, split)](xh, trellis.view(torch.int32), svh, h128(xh.device), y32, m, k, n, nt,
                                         xh.stride(0), y32.stride(0), k // split, K=K, MP=MP, BK=BK,
                                         num_warps=num_warps, num_stages=num_stages)
    return y32.half()
