"""DRAFT-DYN + context bias for the MTP draft head of Qwen3.x-27B.

Per request and per draft head (near = exl3_draft, passes 0..FAR_FROM-1; far = exl3_draft_far):
  * EXL3_DRAFT_DYN=E (far head; EXL3_DRAFT_DYN_NEAR=E' for the near head, default E): up to E extra draft-head rows,
    the request's most frequent prompt tokens that are NOT in that static head. Their int4 rows (same per-row g64
    quantization as DRAFT-Q4) are gathered from a host copy of the whole lm_head when the batch rows change.
  * EXL3_DRAFT_BIAS=b: +b on the drafter logit of every prompt token (static rows and extra rows alike).
Exactness with probabilistic drafting (production): the drafter samples from q = softmax over (static rows U extra rows)
of (logits + bias) / T, and the probs handed to rejection sampling are exactly that q scattered to vocab ids
(extra ids are disjoint from the static ids and from each other, pads get -inf -> probability 0). Any proposal q keeps
the target's output distribution exact; only acceptance moves.

This module holds the device-independent pieces (row selection, host-side batch assembly, the sampled-token / probs
fix-up), so they can be tested on the CPU.

RTX 5090 port: everything above the "CUDA port" section marker is shared verbatim with the Intel Arc (XPU) plugin's
version of this file. The CUDA plugin has no DS-SUB compact
sampler: the MTP compute_logits returns a full-vocab row (static head scattered into -inf) and vLLM's own draft sampler
(fused Triton draft_sample / DRAFT-TOPKP / upstream) samples it and returns its probs. So the CUDA path does not need
ext_index / finish_sample: it scatters the extra rows' logits straight to their real vocab ids (pads to virtual columns
past the lm_head width, sliced away) and the sampler's q is, by construction, the distribution it sampled from. The
CUDA-only helpers are in the "CUDA port" section at the end (SlotMap, scatter_index, gather_host_rows,
quantize_cols_into, dequant_rows_fp16, full_logits).
"""
from __future__ import annotations

import numpy as np
import torch

NEG_INF = float("-inf")


def static_pos_map(idx: torch.Tensor, n_ids: int) -> np.ndarray:
    """id -> position in the static head (or -1), for ids 0..n_ids-1."""
    pos = np.full(n_ids, -1, dtype=np.int64)
    ix = idx.detach().cpu().numpy().astype(np.int64)
    pos[ix] = np.arange(ix.size, dtype=np.int64)
    return pos


def rows_for_request(prompt_ids, pos: np.ndarray, E: int, vocab: int):
    """-> (extra ids: up to E prompt tokens outside the static head, most frequent first, ties by lower id;
           static positions of the prompt's distinct tokens that are in the static head)."""
    p = np.asarray(prompt_ids if prompt_ids is not None else [], dtype=np.int64)
    p = p[(p >= 0) & (p < min(vocab, pos.size))]
    if p.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    u, c = np.unique(p, return_counts=True)            # u sorted ascending
    sp = pos[u]
    spos = sp[sp >= 0]
    if E <= 0:
        return np.zeros(0, dtype=np.int64), spos
    out_u, out_c = u[sp < 0], c[sp < 0]
    out_u = out_u[np.lexsort((out_u, -out_c))][:E]   # count desc, id asc (deterministic)
    return out_u.astype(np.int64), spos


def assemble(entries, E: int, Ns: int, beta: float, host=None, gs: int = 64, K: int = 0):
    """Batch arrays (numpy) for B = len(entries) rows; entries[i] = (extra_ids, static_pos).
    X [B, E] int64 (pads 0), XB [B, E] fp16 (beta, pads -inf), SB [B, Ns] fp16 (beta at the static positions; None if
    beta == 0), and with host = (packed [V, K/8] int32, scales [V, G] fp16, zeros [V, G] int8): Wq [B*E, K/8] int32,
    Ws [G, B*E] fp16, Wz [G, B*E] int8 (pad rows: zero weights, masked by XB)."""
    B = len(entries)
    X = np.zeros((B, max(E, 0)), dtype=np.int64)
    XB = np.full((B, max(E, 0)), NEG_INF, dtype=np.float16)
    SB = np.zeros((B, Ns), dtype=np.float16) if beta else None
    for i, (ex, spos) in enumerate(entries):
        n = min(len(ex), E)
        if n:
            X[i, :n] = ex[:n]
            XB[i, :n] = beta
        if SB is not None and len(spos):
            SB[i, spos] = beta
    out = dict(X=X, XB=XB, SB=SB)
    if host is not None and E > 0:
        hq, hs, hz = host
        G = hs.shape[1]
        ids = X.reshape(-1)
        valid = (XB.reshape(-1) != NEG_INF)
        Wq = hq[ids].copy()
        Wq[~valid] = 0
        Ws = hs[ids].T.copy()
        Ws[:, ~valid] = 0
        Wz = hz[ids].T.copy()
        Wz[:, ~valid] = 0
        out.update(Wq=Wq, Ws=Ws, Wz=Wz)
    return out


def ext_index(idx_long: torch.Tensor, vocab: int, E: int) -> torch.Tensor:
    """Static ids followed by E virtual ids vocab..vocab+E-1 (one per extra column of a row)."""
    return torch.cat([idx_long, torch.arange(vocab, vocab + E, dtype=torch.int64, device=idx_long.device)])


def finish_sample(tok: torch.Tensor, probs_ext: torch.Tensor, X: torch.Tensor, vocab: int):
    """Map virtual ids (>= vocab) back to the row's extra ids and fold the extra columns of probs_ext [B, vocab+E] into
    the real ids (scatter_add: extra ids are distinct from static ids; pads carry probability 0).
    Returns (tok [B] int64, probs [B, vocab] view)."""
    virt = tok >= vocab
    j = (tok - vocab).clamp(min=0)
    tok = torch.where(virt, X.gather(1, j[:, None]).squeeze(1), tok)
    probs = probs_ext[:, :vocab]
    probs.scatter_add_(1, X, probs_ext[:, vocab:])
    return tok, probs


def finish_greedy(L: torch.Tensor, idx_long: torch.Tensor, X: torch.Tensor):
    Ns = idx_long.numel()
    am = L.argmax(dim=-1)
    return torch.where(am < Ns, idx_long.index_select(0, am.clamp(max=Ns - 1)),
                       X.gather(1, (am - Ns).clamp(min=0)[:, None]).squeeze(1))


@torch.no_grad()
def quantize_rows_int4(w: torch.Tensor, gs: int):
    """Same per-row asymmetric int4 (group gs along K) as vllm_plugin._build_draft_q4. w [n, K] -> packed [n, K/8]
    int32, scales [n, G] fp16, zeros [n, G] int8 (row-major per row; transpose for the aten op's [G, N] layout)."""
    n, K = w.shape
    G = K // gs
    w = w.float().reshape(n, G, gs)
    mn, mx = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-8)
    z = (-mn / s).round().clamp(0, 15)
    q = (w / s + z).round().clamp(0, 15).reshape(n, K).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().view(torch.int32)
    return packed, s.squeeze(-1).half(), z.squeeze(-1).to(torch.int8)


def dequant_rows_int4(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, gs: int):
    """Inverse of quantize_rows_int4 (CPU reference for tests): -> fp32 [n, K]."""
    b = packed.contiguous().view(torch.uint8)
    lo, hi = (b & 15).float(), (b >> 4).float()
    q = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    n, K = q.shape
    q = q.reshape(n, K // gs, gs)
    return ((q - zeros.float()[..., None]) * scales.float()[..., None]).reshape(n, K)


# ---------------------------------------------------------------------------------------------------------------
# CUDA port. Device-agnostic torch / numpy, CPU-testable.

class SlotMap:
    """Persistent per-request device slots (0..S-1) for the extra rows' int4 copies. A request keeps its slot while it
    stays in the batch, so a batch change uploads only the newly arrived requests' rows over PCIe (the 5090 sits on a
    Gen2 x4 link: ~0.7 MB per request at E=256). Requests that left the batch free their slot at the next update."""

    def __init__(self, S: int):
        self.S = S
        self.slot: dict = {}

    def update(self, rids):
        """-> (slots [B] for the batch rows in order, [(rid, slot)] that need their rows uploaded), or None if B > S."""
        rids = list(rids)
        if len(rids) > self.S or len(set(rids)) != len(rids):
            return None
        keep = set(rids)
        for r in [r for r in self.slot if r not in keep]:
            del self.slot[r]
        free = sorted(set(range(self.S)) - set(self.slot.values()), reverse=True)
        new = []
        for r in rids:
            if r not in self.slot:
                self.slot[r] = free.pop()
                new.append((r, self.slot[r]))
        return [self.slot[r] for r in rids], new


def scatter_index(X: np.ndarray, XB: np.ndarray, base: int) -> np.ndarray:
    """Per-row scatter targets for the CUDA full-vocab row: the extra id where the column is valid, base + j (a virtual
    column at or past the lm_head width, sliced away before sampling) for pad column j. Pads therefore never touch a
    real id (the XPU version's X pads are 0, harmless there because it maps sampled columns, but id 0 may be a static id)."""
    Xs = X.astype(np.int64, copy=True)
    pad = np.isneginf(XB.astype(np.float32))
    j = np.broadcast_to(np.arange(X.shape[1], dtype=np.int64)[None, :], X.shape)
    Xs[pad] = base + j[pad]
    return Xs


def gather_host_rows(host, ids: np.ndarray, valid: np.ndarray):
    """host = (packed [V, K/8] int32, scales [V, G] fp16, zeros [V, G] int8); ids/valid [n, E]. -> row-major
    (q [n, E, K/8], s [n, E, G], z [n, E, G]) with zero pad rows (masked to -inf / virtual columns anyway)."""
    hq, hs, hz = host
    ids = np.where(valid, ids, 0)
    q, s, z = hq[ids], hs[ids], hz[ids]
    q[~valid] = 0
    s[~valid] = 0
    z[~valid] = 0
    return q, s, z


@torch.no_grad()
def quantize_cols_into(hq: np.ndarray, hs: np.ndarray, hz: np.ndarray, wT: torch.Tensor, a: int, gs: int):
    """Host-table build in K chunks: wT [V', c] = columns a..a+c of rows r0..r0+V' (the caller slices hq/hs/hz to those
    rows); a and c multiples of gs (and of 8). Groups run along K, so per-chunk quantization equals quantize_rows_int4 of
    the full rows (tested)."""
    c = wT.shape[1]
    assert a % gs == 0 and c % gs == 0
    q, s, z = quantize_rows_int4(wT, gs)
    hq[:, a // 8:(a + c) // 8] = q.cpu().numpy()
    hs[:, a // gs:(a + c) // gs] = s.cpu().numpy()
    hz[:, a // gs:(a + c) // gs] = z.cpu().numpy()


@torch.no_grad()
def dequant_rows_fp16(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, gs: int,
                      out: torch.Tensor | None = None, dtype=torch.float16):
    """Batched dequant (any leading dims): packed [..., K/8] int32, scales/zeros [..., G] -> [..., K] (into out)."""
    lead = packed.shape[:-1]
    K = packed.shape[-1] * 8
    G = K // gs
    b = packed.contiguous().view(torch.uint8).reshape(*lead, K // 2)
    q = torch.stack([(b & 15), (b >> 4)], -1).reshape(*lead, G, gs).to(torch.float32)
    w = ((q - zeros.to(torch.float32)[..., None]) * scales.to(torch.float32)[..., None]).reshape(*lead, K)
    if out is None:
        return w.to(dtype)
    out.copy_(w)
    return out


# Fused draft row. Without it, every DRAFT-DYN draft pass skips TAIL-FUSE's one-kernel row write and runs ~8 small
# ops (~0.23 ms of host time each on a Xeon E5 host). Fast path: TAIL-FUSE's draft_scatter for the static row
# (bit-identical) plus one Triton kernel for the extra rows (W.h accumulated in fp32, rounded to fp16, + XB, stored at
# Xs). Against cuBLAS bmm only the accumulation order differs, and only in draft logits; the target verifies.
# LLMT_DYNROW=0 = the unfused path; LLMT_DYNROW_CHECK=N compares both for N calls (max |diff|, -inf mask, argmax).
import os as _os
_DYNROW = _os.environ.get("LLMT_DYNROW", "1") == "1"
_DYNROW_CHECK = [int(_os.environ.get("LLMT_DYNROW_CHECK", "0"))]
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dyn_extra_kernel(W_ptr, h_ptr, XB_ptr, Xs_ptr, out_ptr, E, K, sWb, sWe, shb, sxb, sXb, sob,
                          BE: tl.constexpr, BK: tl.constexpr):
        b = tl.program_id(0)
        e = tl.program_id(1) * BE + tl.arange(0, BE)
        em = e < E
        acc = tl.zeros((BE,), tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            km = k < K
            w = tl.load(W_ptr + b.to(tl.int64) * sWb + e[:, None].to(tl.int64) * sWe + k[None, :],
                        mask=em[:, None] & km[None, :], other=0.0)
            hv = tl.load(h_ptr + b.to(tl.int64) * shb + k, mask=km, other=0.0)
            acc += tl.sum(w.to(tl.float32) * hv.to(tl.float32)[None, :], axis=1)
        xb = tl.load(XB_ptr + b.to(tl.int64) * sxb + e, mask=em, other=0.0)
        v = acc.to(tl.float16) + xb
        col = tl.load(Xs_ptr + b.to(tl.int64) * sXb + e, mask=em, other=0)
        tl.store(out_ptr + b.to(tl.int64) * sob + col, v.to(out_ptr.dtype.element_ty), mask=em)
except Exception:  # noqa
    _DYNROW = False


def _full_logits_fast(sub, inv, n_out, width, vocab, h, W, Xs, XB, SB):
    from exl3cuda.tail_fuse import draft_scatter
    if SB is not None:
        sub = sub + SB
    B = sub.shape[0]
    out = sub.new_empty((B, width))
    s2 = sub if sub.is_contiguous() else sub.contiguous()
    draft_scatter(s2, inv, out, 128)                     # columns < n_out: static logits or -inf (block granular)
    if W is not None and W.shape[1] > 0:
        hh = h if h.dtype == W.dtype else h.to(W.dtype)
        E, K = W.shape[1], W.shape[2]
        BE = 32
        _dyn_extra_kernel[(B, triton.cdiv(E, BE))](W, hh, XB, Xs, out, E, K, W.stride(0), W.stride(1), hh.stride(0),
                                                    XB.stride(0), Xs.stride(0), out.stride(0), BE=BE, BK=256,
                                                    num_warps=4)
    return out[:, :vocab]


def full_logits(sub: torch.Tensor, idx_long: torch.Tensor, width: int, vocab: int, h: torch.Tensor | None = None,
                W: torch.Tensor | None = None, Xs: torch.Tensor | None = None, XB: torch.Tensor | None = None,
                SB: torch.Tensor | None = None, inv: torch.Tensor | None = None,
                n_out: int | None = None) -> torch.Tensor:
    """The draft row; the fused path when TAIL-FUSE's block map (inv, n_out) is given and the strides allow it."""
    if (_DYNROW and inv is not None and n_out is not None and width >= n_out and inv.numel() * 128 == n_out
            and (W is None or (W.stride(2) == 1 and h is not None and h.stride(-1) == 1 and Xs.stride(1) == 1
                               and XB.stride(1) == 1))):
        fast = _full_logits_fast(sub, inv, n_out, width, vocab, h, W, Xs, XB, SB)
        if _DYNROW_CHECK[0] <= 0:
            return fast
        _DYNROW_CHECK[0] -= 1
        ref = _full_logits_ref(sub, idx_long, width, vocab, h, W, Xs, XB, SB)
        fin = torch.isfinite(ref)
        same_inf = bool(torch.equal(fin, torch.isfinite(fast)))
        d = float((fast.float() - ref.float()).abs()[fin].max()) if bool(fin.any()) else 0.0
        am = bool(torch.equal(fast.argmax(-1), ref.argmax(-1)))
        if _DYNROW_CHECK[0] % 50 == 0 or not (same_inf and am):
            import logging
            logging.getLogger("vllm").info("DYN-ROW check: max|diff| %.3g, -inf mask %s, argmax %s (%d checks left)",
                                           d, "same" if same_inf else "DIFFERENT", "same" if am else "DIFFERENT",
                                           _DYNROW_CHECK[0])
        return ref
    return _full_logits_ref(sub, idx_long, width, vocab, h, W, Xs, XB, SB)


def _full_logits_ref(sub: torch.Tensor, idx_long: torch.Tensor, width: int, vocab: int, h: torch.Tensor | None = None,
                     W: torch.Tensor | None = None, Xs: torch.Tensor | None = None, XB: torch.Tensor | None = None,
                     SB: torch.Tensor | None = None) -> torch.Tensor:
    """The CUDA draft row [B, vocab]: static logits (+ SB bias at the prompt tokens' static positions) at idx, the
    rows' extra logits (W [B, E, K] . h [B, K] + XB: bias, or -inf on pads) scattered to Xs (real extra ids, pads at
    virtual columns >= the lm_head width), -inf elsewhere. width = row width incl. the virtual columns (>= max(Xs) + 1).
    Extra ids are disjoint from the static ids and unique per row, so scatter_ has no duplicate targets."""
    if SB is not None:
        sub = sub + SB
    logits = sub.new_full((sub.shape[0], width), float("-inf"))
    logits.index_copy_(1, idx_long, sub)
    if W is not None and W.shape[1] > 0:
        hh = h if h.dtype == W.dtype else h.to(W.dtype)
        ex = torch.bmm(W, hh.unsqueeze(-1)).squeeze(-1)
        logits.scatter_(1, Xs, ex.to(sub.dtype) + XB)
    return logits[:, :vocab]


# ---------------------------------------------------------------------------------------------------------------
# DYN2: the same rows, tables and per-pass logits as the first CUDA port above, with the
# per-batch-change host cost removed. The first port ran rows_for_request (np.unique over 22k-token prompts), the host-table
# gather, ~10 pin_memory() calls and ~50 dequant launches synchronously in propose_draft_token_ids on every batch change
# (-13.6% throughput at 8 streams on the 5090). DYN2 splits it:
#   * admission (after GPUModelRunner._update_states, or the first time a row is seen): the request gets a persistent
#     device slot and a job for one background thread: incremental prompt-token counts (TokCounter), its rows, the int4
#     gather straight into a double-buffered pinned staging set, then on a side stream (after an event of the main
#     stream): H2D into a device staging set, in-place fp16 dequant into Wslot[slot] (bitwise = dequant_rows_fp16),
#     its slot rows of the X/XB/SB tables, an event;
#   * every propose (main thread, host-only Python unless the row -> slot map changed): poll the events of waiting rows
#     (cudaEventQuery), map each batch row to its slot (active) or to the NULL slot (zeros / -inf / no bias = the static
#     head, exact); if the map changed: one H2D of the map from a pinned ring and 3-4 index_select(out=) gathers from the
#     slot tables into the batch-ordered tables the per-pass bmm reads (only from the first changed row on).
# No host sync, no pin_memory, no prompt scan in the step path. The per-pass math (full_logits) is unchanged.

class TokCounter:
    """Prompt-token counts kept incrementally: extend(toks) consumes toks[n:] (toks = the full prompt so far, list or
    numpy); ids outside [0, lim) are ignored, like rows_for_request. A shorter toks restarts the count."""
    __slots__ = ("u", "c", "n", "lim")

    def __init__(self, lim: int):
        self.lim = int(lim)
        self.reset()

    def reset(self):
        self.u = np.zeros(0, dtype=np.int64)
        self.c = np.zeros(0, dtype=np.int64)
        self.n = 0

    def extend(self, toks) -> bool:
        L = len(toks)
        if L < self.n:
            self.reset()
        if L == self.n:
            return False
        p = np.asarray(toks[self.n:], dtype=np.int64)
        self.n = L
        p = p[(p >= 0) & (p < self.lim)]
        if p.size == 0:
            return True
        u2, c2 = np.unique(p, return_counts=True)
        if self.u.size == 0:
            self.u, self.c = u2, c2.astype(np.int64)
            return True
        u = np.union1d(self.u, u2)
        c = np.zeros(u.size, dtype=np.int64)
        c[np.searchsorted(u, self.u)] += self.c
        c[np.searchsorted(u, u2)] += c2
        self.u, self.c = u, c
        return True

    def rows(self, pos: np.ndarray, E: int):
        """== rows_for_request(all tokens so far, pos, E, vocab) with lim = min(vocab, pos.size)."""
        u, c = self.u, self.c
        if u.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        sp = pos[u]
        spos = sp[sp >= 0]
        if E <= 0:
            return np.zeros(0, dtype=np.int64), spos
        m = sp < 0
        out_u, out_c = u[m], c[m]
        out_u = out_u[np.lexsort((out_u, -out_c))][:E]
        return out_u.astype(np.int64), spos


@torch.no_grad()
def dequant_into(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, gs: int, out: torch.Tensor):
    """In-place fp16 dequant of [E, K/8] int32 / [E, G] fp16 / [E, G] int8 into out [E, K] fp16 (6 launches, 2 small
    uint8 temps). Bitwise equal to dequant_rows_fp16: q - z is an exact small integer in fp16 and (q - z) * s is one
    correctly rounded product either way (tested)."""
    E, K = out.shape
    G = K // gs
    b = packed.view(torch.uint8).reshape(E, K // 2)
    o2 = out.view(E, K // 2, 2)
    o2[..., 0].copy_(b & 15)
    o2[..., 1].copy_(b >> 4)
    o3 = out.view(E, G, gs)
    o3.sub_(zeros.view(E, G, 1).to(out.dtype))
    o3.mul_(scales.view(E, G, 1))
    return out


class _Req:
    __slots__ = ("rid", "cnt", "slot", "gen", "state", "ev", "src")

    def __init__(self, rid, lim):
        self.rid = rid
        self.cnt = TokCounter(lim)
        self.slot = None          # device slot, or None (rows map to NULL = static head)
        self.gen = -1
        self.state = "idle"       # idle | queued | up (job done, event recorded) | active | fail
        self.ev = None
        self.src = None           # prompt tokens not counted yet (the worker counts them; cnt is worker-only)


class DynManager:
    """Per-request slots + background uploads + batch-ordered gathers for one or more DYN groups (head dicts).
    groups[i]: dict(E, Ns, K, gs, G, base, beta, pos, host=(hq, hs, hz) or None) plus the device tensors allocated by
    alloc(). Thread model: main thread = admit()/step()/rows(); one worker thread runs _job() (or inline, bg=False)."""

    RING = 64

    def __init__(self, groups, S: int, dev, vocab_lim: int, bg: bool = True, log=None):
        self.groups, self.S, self.NULL, self.dev, self.lim = groups, int(S), int(S), dev, int(vocab_lim)
        self.bg, self.log = bg, log
        self.req: dict = {}
        self.owner = [None] * self.S
        self.sgen = [0] * self.S
        self.free = list(range(self.S))
        self.gmap: list = []                  # (slot, gen) per batch row as gathered on the device
        self.rids: list | None = None
        self.B, self.any, self.waiting, self.dirty = 0, False, 0, False
        self.side = None
        self.q = None
        self.thread = None
        self.stage_i = 0
        self.stage_ev = [None, None]
        self.ring_i = 0
        self.disabled = False
        self.st = dict(dyn=0, skip=0, n=0, chg=0, gat=0, grow=0, upl=0, act=0, fail=0, stale=0,
                       hook_ns=0, hook_max=0, chg_ns=0, chg_max=0, calls=0, job_ns=0, jobs=0)

    # ---- allocation (at load, before vLLM sizes the KV cache) ------------------------------------------------
    def alloc(self):
        """Per group one packed uint8 table per slot row: [X int64 E | W fp16 E*K | XB fp16 E | SB fp16 Ns] (each part
        only if used), rows padded to 16 bytes. Slot tables (S+1 rows, row S = NULL) and batch tables (S rows) are
        views into them, so a regather is one index_select per group (host ~10 us per launch on this CPU)."""
        S1, dev = self.S + 1, self.dev
        mb = pmb = 0.0
        for g in self.groups:
            E, Ns, K, G = g["E"], g["Ns"], g["K"], g["G"]
            parts, off = [], 0
            if E > 0:
                parts += [("X", off, 8 * E, torch.int64, None)]
                off += 8 * E
                parts += [("W", off, 2 * E * K, torch.float16, (E, K))]
                off += 2 * E * K
                parts += [("XB", off, 2 * E, torch.float16, None)]
                off += 2 * E
            if g["beta"]:
                parts += [("SB", off, 2 * Ns, torch.float16, None)]
                off += 2 * Ns
            T = (off + 15) // 16 * 16
            g["PackS"] = torch.zeros(S1, T, dtype=torch.uint8, device=dev)
            g["PackB"] = torch.zeros(self.S, T, dtype=torch.uint8, device=dev)
            for name, o, n, dt, shp in parts:
                for src, sfx in (("PackS", "S"), ("PackB", "")):
                    v = g[src][:, o:o + n].view(dt)
                    if shp is not None:
                        v = v.unflatten(1, shp)
                    g[{"X": "Xs", "W": "W", "XB": "XB", "SB": "SB"}[name] + sfx] = v
            if E > 0:
                g["Wslot"], g["Wb"] = g.pop("WS"), g.pop("W")
                g["XsS"].copy_(torch.arange(g["base"], g["base"] + E, dtype=torch.int64, device=dev).repeat(S1, 1))
                g["XBS"].fill_(float("-inf"))
                g["Xs"].copy_(g["XsS"][:self.S])
                g["XB"].copy_(g["XBS"][:self.S])
                g["Dq"] = torch.zeros(E, K // 8, dtype=torch.int32, device=dev)
                g["Ds"] = torch.zeros(E, G, dtype=torch.float16, device=dev)
                g["Dz"] = torch.zeros(E, G, dtype=torch.int8, device=dev)
            mb += (g["PackS"].numel() + g["PackB"].numel()) / 1e6
            if E > 0:
                mb += sum(g[k].numel() * g[k].element_size() for k in ("Dq", "Ds", "Dz")) / 1e6
            g["stage"] = []
            for _ in range(2):
                st = {}
                if E > 0:
                    st.update(q=_pin(torch.zeros(E, K // 8, dtype=torch.int32)),
                              s=_pin(torch.zeros(E, G, dtype=torch.float16)),
                              z=_pin(torch.zeros(E, G, dtype=torch.int8)),
                              X=_pin(torch.zeros(E, dtype=torch.int64)),
                              XB=_pin(torch.zeros(E, dtype=torch.float16)))
                if g["beta"]:
                    st["SB"] = _pin(torch.zeros(Ns, dtype=torch.float16))
                pmb += sum(v.numel() * v.element_size() for v in st.values()) / 1e6
                g["stage"].append(st)
        self.slot_idx = torch.zeros(self.S, dtype=torch.int64, device=dev)
        self.ring = _pin(torch.full((self.RING, self.S), self.NULL, dtype=torch.int64))
        self.ring_np = self.ring.numpy()
        self.packs = [(g["PackS"], g["PackB"]) for g in self.groups]
        # precomputed views (a slice / select costs ~2 us of host time each on this CPU)
        self.ring_rows = [self.ring[k] for k in range(self.RING)]
        self.si_views = [self.slot_idx[:b] for b in range(self.S + 1)]
        self.pb_views = [[PB[:b] for b in range(self.S + 1)] for _, PB in self.packs]
        self.row_views = [([PB[i] for i in range(self.S)], [PS[i] for i in range(S1)]) for PS, PB in self.packs]
        return mb, pmb

    # ---- admission -------------------------------------------------------------------------------------------
    def _assign(self, r, batch):
        if r.slot is not None:
            return True
        if self.free:
            s = min(self.free)
            self.free.remove(s)
        else:                                  # evict an owner that is not in the current batch
            s = next((i for i, o in enumerate(self.owner) if o is not None and o.rid not in batch), None)
            if s is None:
                return False
            o = self.owner[s]
            o.slot, o.state, o.ev = None, "idle", None
        self.sgen[s] += 1
        self.owner[s] = r
        r.slot, r.gen = s, self.sgen[s]
        return True

    def _submit(self, r):
        """Queue r's (re)upload into its slot, with its not-yet-counted prompt tokens (r.src)."""
        ev_main = torch.cuda.Event()
        ev_main.record()                       # current stream: all reads of the slot's old contents are before this
        r.state, r.ev = "queued", None
        self.dirty = True                      # the next step() re-maps even if the batch rows did not change
        job = (r, r.slot, r.gen, r.src, ev_main)
        r.src = None
        if not self.bg:
            self._job(*job)
            return
        if self.thread is None:
            import queue
            import threading
            self.q = queue.SimpleQueue()
            self.thread = threading.Thread(target=self._worker, name="exl3-draftdyn2", daemon=True)
            self.thread.start()
        self.q.put(job)

    def admit(self, rid, src, batch=()):
        """New request (or a streaming-session prompt update): counts from src, slot, upload job."""
        if self.disabled or rid is None:
            return
        r = self.req.get(rid)
        if r is None:
            r = self.req[rid] = _Req(rid, self.lim)
        elif r.slot is not None:               # re-prep into the same slot: new generation, inactive until done
            self.sgen[r.slot] += 1
            r.gen = self.sgen[r.slot]
        if src is not None:
            r.src = src                        # the full prompt so far; the worker counts the part it has not seen
        if self._assign(r, batch):
            self._submit(r)
        else:                                  # rare (no slot): uploaded at the next assignment
            r.state = "idle"

    def drop(self, rid):
        r = self.req.pop(rid, None)
        if r is None or r.slot is None:
            return
        s = r.slot
        self.owner[s] = None
        self.sgen[s] += 1
        self.free.append(s)
        r.slot, r.state = None, "dead"

    # ---- background job --------------------------------------------------------------------------------------
    DELAY_S = 0.002   # worker start delay per burst: the job then overlaps the target forward's GPU wait (GIL free)
    #                   instead of the host work right after admission; activation is >= one step later anyway

    def _worker(self):
        import time
        while True:
            job = self.q.get()
            if self.DELAY_S > 0:
                time.sleep(self.DELAY_S)
            try:
                self._job(*job)
            except Exception as e:  # noqa
                job[0].state = "fail"
                self.st["fail"] += 1
                if self.log is not None:
                    self.log.warning("exl3: DRAFT-DYN2 upload job failed (%r); that request drafts static", e)

    @torch.no_grad()
    def _job(self, r, s, gen, src, ev_main):
        import time
        t0 = time.perf_counter_ns()
        if src is not None:
            r.cnt.extend(src)                  # even for a stale job: the request may get another slot later
        if self.sgen[s] != gen:
            self.st["stale"] += 1
            return
        k = self.stage_i
        self.stage_i ^= 1
        if self.stage_ev[k] is not None:
            self.stage_ev[k].synchronize()     # worker thread only: the pinned set's previous H2D is done
        rows = []
        for g in self.groups:
            ex, spos = r.cnt.rows(g["pos"], g["E"])
            st = g["stage"][k]
            E = g["E"]
            if E > 0:
                n = min(len(ex), E)
                X, XB = st["X"].numpy(), st["XB"].numpy()
                X[:n] = ex[:n]
                X[n:] = g["base"] + np.arange(n, E)
                XB[:n] = g["beta"]
                XB[n:] = NEG_INF
                hq, hs, hz = g["host"]
                ids = np.zeros(E, dtype=np.int64)
                ids[:n] = ex[:n]
                q, sc, z = st["q"].numpy(), st["s"].numpy(), st["z"].numpy()
                np.take(hq, ids, axis=0, out=q)
                np.take(hs, ids, axis=0, out=sc)
                np.take(hz, ids, axis=0, out=z)
                if n < E:
                    q[n:] = 0
                    sc[n:] = 0
                    z[n:] = 0
            if g["beta"]:
                SB = st["SB"].numpy()
                SB[:] = 0
                if len(spos):
                    SB[spos] = g["beta"]
            rows.append(st)
        if self.side is None:
            self.side = torch.cuda.Stream(device=self.dev)
        side = self.side
        with torch.cuda.device(self.dev), torch.cuda.stream(side):
            side.wait_event(ev_main)
            for g, st in zip(self.groups, rows):
                if g["E"] > 0:
                    g["Dq"].copy_(st["q"], non_blocking=True)
                    g["Ds"].copy_(st["s"], non_blocking=True)
                    g["Dz"].copy_(st["z"], non_blocking=True)
                    g["XsS"][s].copy_(st["X"], non_blocking=True)
                    g["XBS"][s].copy_(st["XB"], non_blocking=True)
                    dequant_into(g["Dq"], g["Ds"], g["Dz"], g["gs"], g["Wslot"][s])
                if g["beta"]:
                    g["SBS"][s].copy_(st["SB"], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(side)
        self.stage_ev[k] = ev
        self.st["upl"] += 1
        if self.sgen[s] == gen:
            r.ev = ev
            r.state = "up"
        else:
            self.st["stale"] += 1
        self.st["job_ns"] += time.perf_counter_ns() - t0
        self.st["jobs"] += 1

    def idle(self):
        """Tests: wait until the worker queue is empty and the last job finished."""
        import time
        while self.q is not None and (not self.q.empty() or any(r.state == "queued" for r in self.req.values())):
            time.sleep(0.001)

    # ---- per propose (main thread) ---------------------------------------------------------------------------
    def step(self, rids, get_src=None):
        """rids = the drafter's batch rows (input_batch order). get_src(rid) -> prompt tokens for a row never
        admitted (fallback when the _update_states hook did not see it), or None."""
        if self.disabled:
            return
        if rids == self.rids and not self.waiting and not self.dirty:
            return
        self.dirty = False
        if rids != self.rids:
            self.rids = list(rids)
            self.st["chg"] += 1
        req, NULL, sgen = self.req, self.NULL, self.sgen
        m, waiting, batch = [], 0, None
        for rid in rids:
            r = req.get(rid)
            if r is None or r.slot is None:    # never admitted (no _update_states hook) or evicted earlier
                if rid is not None and (r is None or r.state != "fail"):
                    if batch is None:
                        batch = set(rids)
                    if r is None:
                        self.admit(rid, get_src(rid) if get_src is not None else None, batch)
                    elif self._assign(r, batch):
                        self._submit(r)
                    r = req.get(rid)
                if r is None or r.slot is None:
                    if rid is not None and (r is None or r.state != "fail"):
                        waiting += 1
                    m.append((NULL, 0))
                    continue
            if r.state == "up" and r.gen == sgen[r.slot] and r.ev is not None and r.ev.query():
                r.state = "active"
                self.st["act"] += 1
            if r.state == "active" and r.gen == sgen[r.slot]:
                m.append((r.slot, r.gen))
            else:
                if r.state != "fail":
                    waiting += 1
                m.append((NULL, 0))
        self.waiting = waiting
        B = len(m)
        if B > self.S:
            self.B, self.any = 0, False
            return
        gm = self.gmap
        ng = len(gm)
        ch = [i for i in range(B) if i >= ng or m[i] != gm[i]]
        if ch:
            self._gather(m, ch, B)
            self.gmap = m + gm[B:] if ng > B else m
        self.B = B
        self.any = any(sl != NULL for sl, _ in m)

    ROWCOPY = 2                                            # <= this many changed rows: D2D row copies, no H2D

    def _gather(self, m, ch, B):
        """Bring the batch tables' rows ch (of 0..B-1) to the slots in m: one D2D row copy per changed row and group
        (~7 us host each), or for more rows one pinned H2D of the map + one index_select per group (~20 us)."""
        if len(ch) <= self.ROWCOPY:
            for rowsB, rowsS in self.row_views:
                for i in ch:
                    rowsB[i].copy_(rowsS[m[i][0]])
        else:
            k = self.ring_i
            self.ring_i = (k + 1) % self.RING
            self.ring_np[k, :B] = [sl for sl, _ in m]
            self.slot_idx.copy_(self.ring_rows[k], non_blocking=True)
            si = self.si_views[B]
            for (PS, _), pbv in zip(self.packs, self.pb_views):
                torch.index_select(PS, 0, si, out=pbv[B])
        self.st["gat"] += 1
        self.st["grow"] += len(ch)

    def reset(self):
        """After a failure: everything static from now on."""
        self.disabled = True
        self.B, self.any = 0, False


def _pin(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.pin_memory()
    except Exception:  # noqa (CPU-only tests)
        return t
