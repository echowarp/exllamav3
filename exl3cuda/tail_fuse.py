"""TAIL-FUSE: trims the decode "small-kernel tail" (1-stream decode step: ~1,000
1-3 us kernels, ~8% of GPU time). Every part is BIT-EXACT vs the stock path (same fp ops in the same order; only
launch shapes, addresses or the number of launches change) and CUDA-graph capturable (no host syncs, no H2D copies,
all tables built at weight-load time).

EXL3_TAIL_FUSE=1 turns on all parts; or a comma list of: cat, scratch, conv, ks, dlog
  cat     _Sliced (fused gate+up / q+k+v / GDN qkv+z decode mgemm) writes the concatenated (m, n) row layout directly:
          no torch.cat and no per-call pointer add (`coff[m] + ptr`). Needs the TAIL-CAT ext (exl3_mgemm kwargs
          c_row_stride / c_base, in this repo's exllamav3_ext); with an older ext this part falls back to `scratch`.
          -142 cat + -71 add launches per 1-stream step.
  scratch FUSE-SCRATCH buffer sized for EXL3_FUSE_MAX_M rows of the widest fused module (the 1M-element default only
          held gate_up for m <= 30, so gate_up paid a `coff[m] + ptr` add per call at FUSE_MAX_M 64). -71 adds.
  conv    GDN-RC causal_conv1d_update with BLOCK_N sized to fill the GPU (stock: 256 -> 40 programs at 1 stream).
          Same Triton kernel and per-element math; only the channel tiling changes.
  ks      K smoothing fused into the int8 per-token-head KV-cache write: (key - mean) in the cache kernel instead of an
          aten sub + the stock kernel. Same fp32 sub + fp16 rounding as aten's Half sub.
  dlog    pruned draft head logits: the -inf fill + index_copy_ into the full-vocab row done by one Triton kernel.

The parts are independent; each one logs one "exl3: TAIL-FUSE ..." line when installed.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from exl3cuda.hwdefaults import env as _hwenv
from vllm.triton_utils import tl, triton

logger = init_logger("vllm.exl3cuda")

_ALL = ("cat", "scratch", "conv", "ks", "dlog")


def _parts():
    v = _hwenv("EXL3_TAIL_FUSE", "1", "0").strip()
    if v in ("", "0"):
        return set()
    if v == "1":
        return set(_ALL)
    return {p.strip() for p in v.split(",") if p.strip()}


# ------------------------------------------------------------------------------------------------------------------
# cat / scratch: fused decode mgemm output layout


def ext_has_cat() -> bool:
    try:
        from exllamav3.ext import exllamav3_ext as ext
        return "c_base" in (ext.exl3_mgemm.__doc__ or "")
    except Exception:  # noqa
        return False


def sliced_call_direct(self, x2):
    """_Sliced.__call__ for the TAIL-CAT ext: every slice writes its columns of the (m, n) output row in place
    (C row stride n, slice offsets = the m = 1 table coff[1], base = out.data_ptr()). Same kernel launch, same values
    as the block layout + torch.cat; only output addresses differ."""
    m = x2.shape[0]
    out = torch.empty((m, self.n), dtype=torch.half, device=x2.device)
    had = torch.empty((self.nsrc, m, self.k), dtype=torch.half, device=x2.device)
    self.ext.exl3_mgemm(x2.view(1, m, self.k), self.ptrs_tr, self.carrier.expand(self.S, m, self.w), self.ptrs_suh,
                        had, self.ptrs_svh, None, None, self.K, -1, self.mcg, self.mul1, -1, -1, 0, 1, self.size_n,
                        self.coff[1], self.stride, self.src, self.nsrc, self.n, out.data_ptr())
    return out


def _install_cat(P, parts):
    use_cat = "cat" in parts and ext_has_cat()
    if "cat" in parts and not use_cat:
        logger.warning("exl3: TAIL-FUSE cat: the loaded exllamav3_ext has no c_base/c_row_stride (old ext); "
                       "using the scratch part instead")
        parts = parts | {"scratch"}
    if use_cat:
        P._Sliced.__call__ = sliced_call_direct
        logger.info("exl3: TAIL-FUSE cat on: fused decode mgemm writes the concatenated row (no torch.cat, no "
                    "pointer add)")
    elif "scratch" in parts and P.FUSE_SCRATCH:
        need = 34816 * max(P.FUSE_MAX_M, 1)          # widest fused decode module: gate_up (2 x 17408)
        if need > P._FUSE_SCRATCH_ELEMS:
            P._FUSE_SCRATCH_ELEMS = need
            logger.info("exl3: TAIL-FUSE scratch on: FUSE-SCRATCH buffer %d elements (%.1f MB) for m <= %d",
                        need, need * 2 / 2**20, P.FUSE_MAX_M)


# ------------------------------------------------------------------------------------------------------------------
# conv: causal_conv1d_update with a wider grid

def _conv_block_n(batch, dim):
    bn = int(os.environ.get("EXL3_TAIL_CONV_BN", "0"))
    if bn:
        return bn
    # sweep: BLOCK_N 64 wins 0.1-0.3 us at 1-3 sequences; from 4 sequences the stock 256 is as fast
    return 64 if batch <= 3 else 256


def causal_conv1d_update_fast(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                              num_accepted_tokens=None, query_start_loc=None, max_query_len=-1,
                              null_block_id=None, block_idx_last_scheduled_token=None, initial_state_idx=None,
                              validate_data=False):
    """vLLM 0.26 causal_conv1d_update (mamba/ops/causal_conv1d.py) verbatim, except BLOCK_N / num_warps. Each program
    still handles whole channels for all tokens of one sequence: per-element math and order are unchanged."""
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc
    if null_block_id is None:
        null_block_id = cc.NULL_BLOCK_ID
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    out = x
    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    bn = _conv_block_n(batch, dim)
    nw = int(os.environ.get("EXL3_TAIL_CONV_WARPS", "0")) or (2 if bn == 64 else 4)
    cc._causal_conv1d_update_kernel[(batch, triton.cdiv(dim, bn))](
        x, weight, bias, conv_state, conv_state_indices, num_accepted_tokens, query_start_loc,
        block_idx_last_scheduled_token, initial_state_idx, out,
        batch, dim, seqlen, state_len, num_cache_lines,
        stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
        stride_istate_seq, stride_istate_dim, stride_istate_token, stride_state_indices,
        stride_o_seq, stride_o_dim, stride_o_token,
        null_block_id,
        HAS_BIAS=bias is not None, KERNEL_WIDTH=width, SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None, IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None, NP2_STATELEN=np2_statelen,
        HAS_NULL_BLOCK=null_block_id is not None, BLOCK_N=bn, num_warps=nw)
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


def _install_conv():
    # _rc_forward (GDN-RC, the decode/verify path) imports causal_conv1d_update from this module at call time, so the
    # module attribute is the hook; vLLM's own GDN code imported the name at import time and keeps the stock function
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc
    if getattr(cc, "_exl3_tail", False):
        return
    cc._exl3_tail_orig_update = cc.causal_conv1d_update
    cc.causal_conv1d_update = causal_conv1d_update_fast
    cc._exl3_tail = True
    logger.info("exl3: TAIL-FUSE conv on: GDN-RC causal_conv1d_update BLOCK_N 256 -> %s",
                os.environ.get("EXL3_TAIL_CONV_BN") or "64 at <= 3 sequences")


# ------------------------------------------------------------------------------------------------------------------
# ks: K smoothing inside the per-token-head int8 KV-cache write

@triton.jit
def _reshape_cache_pth_ks(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, k_scale_cache_ptr, v_scale_cache_ptr, slot_mapping_ptr,
    kmean_ptr,
    stride_key_tok: tl.int64, stride_key_head: tl.int64, stride_val_tok: tl.int64, stride_val_head: tl.int64,
    stride_kc_blk: tl.int64, stride_kc_slot: tl.int64, stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64, stride_vc_slot: tl.int64, stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64, stride_ks_slot: tl.int64, stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64, stride_vs_slot: tl.int64, stride_vs_head: tl.int64,
    stride_km_head: tl.int64,
    block_size: tl.constexpr, head_size: tl.constexpr, head_size_v: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
    QUANT_MAX: tl.constexpr = 127.0, QUANT_MIN: tl.constexpr = -128.0, IS_INT_QUANT: tl.constexpr = False,
):
    # vLLM 0.26 _reshape_cache_per_token_head (v1/attention/ops/triton_reshape_and_cache_flash.py) + the K mean:
    # k = fp16(fp32(key) - fp32(mean)) is what aten's Half `key - m` produces; the rest is the stock kernel verbatim
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // block_size
    slot_in_blk = slot % block_size
    dim_offs = tl.arange(0, HEAD_SIZE_PADDED)

    k_mask = dim_offs < head_size
    k_raw = tl.load(key_ptr + tok * stride_key_tok + head * stride_key_head + dim_offs, mask=k_mask, other=0.0)
    k_m = tl.load(kmean_ptr + head * stride_km_head + dim_offs, mask=k_mask, other=0.0)
    k_h = (k_raw.to(tl.float32) - k_m.to(tl.float32)).to(k_raw.dtype).to(tl.float32)

    k_scale = tl.maximum(tl.max(tl.abs(k_h)) / QUANT_MAX, 1e-6)
    tl.store(k_scale_cache_ptr + blk * stride_ks_blk + slot_in_blk * stride_ks_slot + head * stride_ks_head, k_scale)
    k_q = k_h * (1.0 / k_scale)
    if IS_INT_QUANT:
        k_q = tl.where(k_q >= 0, k_q + 0.5, k_q - 0.5)
    k_q = tl.clamp(k_q, QUANT_MIN, QUANT_MAX)
    tl.store(key_cache_ptr + blk * stride_kc_blk + slot_in_blk * stride_kc_slot + head * stride_kc_head + dim_offs,
             k_q, mask=k_mask)

    v_mask = dim_offs < head_size_v
    v_h = tl.load(value_ptr + tok * stride_val_tok + head * stride_val_head + dim_offs, mask=v_mask,
                  other=0.0).to(tl.float32)
    v_scale = tl.maximum(tl.max(tl.abs(v_h)) / QUANT_MAX, 1e-6)
    tl.store(v_scale_cache_ptr + blk * stride_vs_blk + slot_in_blk * stride_vs_slot + head * stride_vs_head, v_scale)
    v_q = v_h * (1.0 / v_scale)
    if IS_INT_QUANT:
        v_q = tl.where(v_q >= 0, v_q + 0.5, v_q - 0.5)
    v_q = tl.clamp(v_q, QUANT_MIN, QUANT_MAX)
    tl.store(value_cache_ptr + blk * stride_vc_blk + slot_in_blk * stride_vc_slot + head * stride_vc_head + dim_offs,
             v_q, mask=v_mask)


def reshape_cache_pth_ks(key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slot_mapping, kmean):
    """triton_reshape_and_cache_flash_per_token_head_quant(key - kmean, value, ...) for int8 caches, one launch."""
    from vllm.v1.attention.ops import triton_reshape_and_cache_flash as rc
    quant_max, quant_min = rc._PER_TOKEN_HEAD_QUANT_PARAMS[key_cache.dtype]
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))
    num_warps = min(16, max(1, head_size_padded // 32))
    assert key.stride(-1) == 1 and value.stride(-1) == 1 and kmean.stride(-1) == 1
    _reshape_cache_pth_ks[(num_tokens, num_kv_heads)](
        key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slot_mapping, kmean,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
        v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2),
        kmean.stride(0),
        block_size=key_cache.shape[1], head_size=head_size, head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded, QUANT_MAX=quant_max, QUANT_MIN=quant_min,
        IS_INT_QUANT=key_cache.dtype == torch.int8, num_warps=num_warps)


def _install_ks():
    ks_path = os.environ.get("EXL3_KSMOOTH")
    if not ks_path or os.environ.get("EXL3_KSMOOTH_CALIB"):
        logger.info("exl3: TAIL-FUSE ks: K smoothing off or calibrating; not installed")
        return
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_interface import KVQuantMode
    if getattr(TritonAttentionImpl, "_exl3_tail_ks", False):
        return
    try:
        kmeans = dict(torch.load(ks_path, map_location="cpu").items())
    except Exception as e:  # noqa
        logger.warning("exl3: TAIL-FUSE ks: %s not loaded (%s); not installed", ks_path, e)
        return
    prev = TritonAttentionImpl.do_kv_cache_update      # the plugin's KSMOOTH wrapper (fallback for everything else)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        name = getattr(layer, "layer_name", None)
        if (name in kmeans and getattr(self, "_kv_quant_mode", None) == KVQuantMode.INT8_PER_TOKEN_HEAD
                and self._is_per_token_head_quant
                and self.attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
                and key.dim() == 3 and key.stride(-1) == 1 and value.stride(-1) == 1):
            m = getattr(self, "_exl3_km", None)
            if m is None or m.device != key.device or m.dtype != key.dtype:
                m = self._exl3_km = kmeans[name].to(device=key.device, dtype=key.dtype).contiguous()
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            if key_cache.dtype == torch.int8:
                reshape_cache_pth_ks(key, value, key_cache, value_cache, self._k_scale_cache, self._v_scale_cache,
                                     slot_mapping, m)
                return
        return prev(self, layer, key, value, kv_cache, slot_mapping)

    TritonAttentionImpl.do_kv_cache_update = do_kv_cache_update
    TritonAttentionImpl._exl3_tail_ks = True
    logger.info("exl3: TAIL-FUSE ks on: K smoothing fused into the int8 KV-cache write (%d layers)", len(kmeans))


# ------------------------------------------------------------------------------------------------------------------
# dlog: pruned draft head -> full-vocab logits row in one kernel

@triton.jit
def _draft_scatter_kernel(sub_ptr, blk_ptr, out_ptr, n_rows, n_out, stride_sub, stride_out,
                          RB: tl.constexpr, BS: tl.constexpr):
    # program (row tile, output block b): out[rows, b*BS:(b+1)*BS] = sub[rows, j*BS:(j+1)*BS] if block b is draft
    # block j, else -inf. blk_ptr[b] = j or -1 (inverse block map, built at weight-load time). Pure copy / fill.
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    b = tl.program_id(1)
    cols = tl.arange(0, BS)
    j = tl.load(blk_ptr + b)
    rmask = rows < n_rows
    omask = rmask[:, None] & ((b * BS + cols) < n_out)[None, :]
    if j >= 0:
        v = tl.load(sub_ptr + rows[:, None].to(tl.int64) * stride_sub + j * BS + cols[None, :], mask=rmask[:, None])
    else:
        v = tl.full((RB, BS), float("-inf"), tl.float32).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * stride_out + b * BS + cols[None, :], v, mask=omask)


# LLMT_FASTLAUNCH_CHECK=N: compare the next N fast launches of draft_scatter with a plain launch (bit-identical check)
_FL_CHECK = [int(os.environ.get("LLMT_FASTLAUNCH_CHECK", "0"))]


def draft_scatter(sub, inv, logits, BS=128):
    rows = logits.shape[0]
    RB = 1 if rows == 1 else (2 if rows == 2 else (4 if rows <= 4 else 8))
    from .fastlaunch import launch as _fl
    g = (triton.cdiv(rows, RB), inv.numel())
    _fl(_draft_scatter_kernel, g, sub, inv, logits, rows, logits.shape[1], sub.stride(0), logits.stride(0), RB=RB, BS=BS,
        num_warps=1 if RB <= 2 else 4)
    if _FL_CHECK[0] > 0:
        _FL_CHECK[0] -= 1
        ref = torch.empty_like(logits)
        _draft_scatter_kernel[g](sub, inv, ref, rows, ref.shape[1], sub.stride(0), ref.stride(0), RB=RB, BS=BS,
                                 num_warps=1 if RB <= 2 else 4)
        n = inv.numel() * BS
        same = bool(torch.equal(ref[:, :n].view(torch.int16), logits[:, :n].view(torch.int16)))
        if not same or _FL_CHECK[0] % 100 == 0:
            logger.info("FAST-LAUNCH check draft_scatter: %s (%d left)", "bit-identical" if same else "DIFFERENT", _FL_CHECK[0])


def _install_dlog(P):
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:  # noqa
        return
    cls = qwen3_5_mtp.Qwen3_5MTP
    if not getattr(cls, "_exl3_patched", False) or getattr(cls, "_exl3_tail_dlog", False):
        logger.info("exl3: TAIL-FUSE dlog: draft-vocab head not patched; not installed")
        return
    BS = 128
    orig_build = P._build_draft_head

    def _build_draft_head(layer, path, cbt, attr="exl3_draft"):
        # weight-load time (host syncs allowed): inverse block map out block -> draft block (-1 = not drafted)
        orig_build(layer, path, cbt, attr)
        d = getattr(layer, attr, None)
        if d is None:
            return
        idx = d["idx"].cpu()
        n_out = int(layer.svh.shape[0])
        ok = idx.numel() % BS == 0 and n_out % BS == 0
        if ok:
            blocks = idx.view(-1, BS)
            first = blocks[:, 0]
            ok = bool(torch.equal(blocks, first[:, None] + torch.arange(BS)[None, :])) and bool((first % BS == 0).all())
        if ok:
            t = torch.full((n_out // BS,), -1, dtype=torch.int32)
            t[first // BS] = torch.arange(first.numel(), dtype=torch.int32)
            ok = int((t >= 0).sum()) == first.numel()          # no duplicate blocks
        d["tail_inv"] = t.to(d["idx"].device) if ok else None
        d["tail_n_out"] = n_out
        logger.info("exl3: TAIL-FUSE dlog: %s inverse block map %s", attr, "built" if ok else "not usable (fallback)")

    P._build_draft_head = _build_draft_head

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return cls._exl3_tail_orig(self, hidden_states, spec_step_idx)
        far = getattr(lm, "exl3_draft_far", None)
        attr = "exl3_draft"
        if far is not None:
            k = getattr(self, "_exl3_pass", 0)
            self._exl3_pass = k + 1
            if k >= P._DRAFT_FAR_FROM:
                d = far
                attr = "exl3_draft_far"
        sub = torch.ops.exl3cuda.linear(hidden_states, d["id"], d["n"])
        if getattr(P, "_DYN_ON", False):                  # DRAFT-DYN2 rows first (as the plugin's compute_logits)
            r = P._dyn_draft_row(d, attr, sub, hidden_states, lm.exl3_n, self.config.vocab_size)
            if r is not None:
                return r
        inv = d.get("tail_inv")
        n_out = lm.exl3_n
        if inv is None or n_out != d.get("tail_n_out") or not sub.is_contiguous():
            logits = hidden_states.new_full((hidden_states.shape[0], n_out), float("-inf"))
            logits.index_copy_(1, d["idx"], sub)
            return logits[:, : self.config.vocab_size]
        logits = hidden_states.new_empty((hidden_states.shape[0], n_out))
        if sub.dtype != logits.dtype:
            sub = sub.to(logits.dtype)
        draft_scatter(sub, inv, logits, BS)
        return logits[:, : self.config.vocab_size]

    cls._exl3_tail_orig = cls.compute_logits
    cls.compute_logits = compute_logits
    cls._exl3_tail_dlog = True
    logger.info("exl3: TAIL-FUSE dlog on: draft logits row written by one kernel (no fill + index_copy_)")


# ------------------------------------------------------------------------------------------------------------------

def install(P):
    """P: the exl3cuda.vllm_plugin module (called at the end of register(), before any weights are loaded)."""
    parts = _parts()
    if not parts:
        return
    unknown = parts - set(_ALL)
    if unknown:
        logger.warning("exl3: TAIL-FUSE: unknown parts %s ignored", sorted(unknown))
    for name, fn in (("cat", lambda: _install_cat(P, parts)), ("conv", _install_conv), ("ks", _install_ks),
                     ("dlog", lambda: _install_dlog(P))):
        if name == "cat" and not ({"cat", "scratch"} & parts):
            continue
        if name != "cat" and name not in parts:
            continue
        try:
            fn()
        except Exception as e:  # noqa: never block engine start; the stock path stays in place
            logger.warning("exl3: TAIL-FUSE %s not installed (%s: %s)", name, type(e).__name__, e)
    logger.info("exl3: TAIL-FUSE on (EXL3_TAIL_FUSE=%s: %s)", os.environ.get("EXL3_TAIL_FUSE"),
                ",".join(p for p in _ALL if p in parts))
