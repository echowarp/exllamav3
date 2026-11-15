"""GDN-FAST: fast path of vLLM's GDNAttentionMetadataBuilder.build for the common decode step
(every scheduled request is a spec decode, padded requests at the back, FULL CUDA graph replay). Python stack
profile of the 1-stream decode step: the builder + mamba_get_block_table_tensor issue
~100 eager torch ops per step (the largest share of the ~1.5 ms/step of host-bound input prep while the GPU idles).
Here: two .tolist() on CPU tensors for the classification and ONE Triton launch that writes the persistent graph
buffers (state indices from the align-mode block-table gather, sequence mask, query start locations, accepted
counts). Integer outputs, identical to the original by construction; EXL3_GDN_FAST=2 checks every field against the
original builder on each step (logs the first mismatches, keeps the original's result). Any other step shape -> the
original builder. EXL3_GDN_FAST=0 = off; unset = 1 on sm_120 (hwdefaults.py), off elsewhere. Installed by GDN-PORT."""
import logging
import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from exl3cuda.hwdefaults import env as _hwenv
logger = init_logger("vllm.exl3cuda.gdn_fast")
MODE = int(_hwenv("EXL3_GDN_FAST", "1", "0") or 0)
_ST = {"fast": 0, "orig": 0, "checked": 0, "mismatch": 0}


class _Disabled(Exception):
    pass


@triton.jit
def _gdn_fast_kernel(BT, bt_s0, SL, QSL, NAT, SSI, MASK, SQSL, NATO, n, bs, null_id, mbs,
                     C: tl.constexpr, CP: tl.constexpr, ALIGN: tl.constexpr, BCAST: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, CP)                                      # CP = next power of 2 >= C (Triton ranges)
    jm = j < C
    live = i < n
    inb = i < bs
    if ALIGN:
        sl = tl.load(SL + i, mask=live, other=1)
        start = tl.maximum((sl - 1) // mbs, 0)
    else:
        start = tl.zeros([BLOCK], tl.int32)
    if BCAST:   # align mode with 1 + num_speculative_blocks == 1: vLLM's copy_ broadcasts the one column to all C
        col = start[:, None] + j[None, :] * 0
    else:
        col = start[:, None] + j[None, :]
    v = tl.load(BT + i[:, None].to(tl.int64) * bt_s0 + col, mask=live[:, None] & jm[None, :], other=0)
    v = tl.where(live[:, None], v, null_id)
    tl.store(SSI + i[:, None] * C + j[None, :], v.to(tl.int32), mask=inb[:, None] & jm[None, :])
    tl.store(MASK + i, live.to(tl.uint8), mask=inb)
    q = tl.load(QSL + tl.minimum(i, n))                       # i > n: qsl[n] (= spec_query_start_loc[-1])
    tl.store(SQSL + i, q.to(tl.int32), mask=i < bs + 1)
    a = tl.load(NAT + i, mask=live, other=1)
    tl.store(NATO + i, tl.where(live, a, 1).to(tl.int32), mask=inb)


def try_fast(self, m, num_accepted_tokens, num_decode_draft_tokens_cpu, meta_cls, null_id):
    """Returns GDNAttentionMetadata or None (shape not covered)."""
    if (not self.use_spec_decode or num_decode_draft_tokens_cpu is None or not self.use_full_cuda_graph
            or num_accepted_tokens is None):
        return None
    d = num_decode_draft_tokens_cpu.tolist()
    n = 0
    while n < len(d) and d[n] >= 0:
        n += 1
    if n == 0 or any(x >= 0 for x in d[n:]) or sum(d[:n]) == 0:
        return None
    qsl = m.query_start_loc_cpu.tolist()
    if len(qsl) != len(d) + 1 or any(qsl[k + 1] != qsl[k] for k in range(n, len(d))):
        return None                                           # a non-spec request with tokens (prefill/decode)
    tokens = qsl[-1]
    bs = m.num_reqs
    if n > self.decode_cudagraph_max_bs or tokens > self.decode_cudagraph_max_bs or bs > self.decode_cudagraph_max_bs:
        return None
    C = self.num_spec + 1
    spec = self.kv_cache_spec
    align = self.vllm_config.cache_config.mamba_cache_mode not in ("all", "none")
    bt = m.block_table_tensor
    width = (1 + spec.num_speculative_blocks) if align else bt.shape[1]
    if width != 1 and width < C:
        return None                                           # partial width: leave it to the original
    bcast = align and width == 1
    if bt.stride(1) != 1 or self.spec_state_indices_tensor.shape[1] != C or not self.spec_state_indices_tensor.is_contiguous():
        return None
    if not getattr(self, "_llmt_fast_arange", False):         # spec_token_indx buffer = arange (written once)
        self.spec_token_indx.copy_(torch.arange(self.spec_token_indx.numel(), dtype=torch.int32,
                                                device=self.spec_token_indx.device))
        self._llmt_fast_arange = True
    spec_token_size = min(n * C, tokens)
    blk = triton.next_power_of_2(max(bs + 1, 2))
    _gdn_fast_kernel[(1,)](bt, bt.stride(0), m.seq_lens, m.query_start_loc, num_accepted_tokens,
                           self.spec_state_indices_tensor, self.spec_sequence_masks.view(torch.uint8),
                           self.spec_query_start_loc, self.num_accepted_tokens, n, bs, null_id,
                           spec.block_size if align else 1, C=C, CP=triton.next_power_of_2(C), ALIGN=align, BCAST=bcast, BLOCK=blk, num_warps=1)
    return meta_cls(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=n, num_spec_decode_tokens=tokens, num_actual_tokens=m.num_actual_tokens,
        has_initial_state=None, chunk_indices=None, chunk_offsets=None, prefill_query_start_loc=None,
        prefill_state_indices=None, prefill_has_initial_state=None,
        spec_query_start_loc=self.spec_query_start_loc[: bs + 1], non_spec_query_start_loc=None,
        spec_state_indices_tensor=self.spec_state_indices_tensor[:bs], non_spec_state_indices_tensor=None,
        spec_sequence_masks=self.spec_sequence_masks[:bs], spec_token_indx=self.spec_token_indx[:spec_token_size],
        non_spec_token_indx=self.non_spec_token_indx[:0], num_accepted_tokens=self.num_accepted_tokens[:bs],
        nums_dict=None, batch_ptr=None, token_chunk_offset_ptr=None)


_FIELDS = ("num_prefills", "num_prefill_tokens", "num_decodes", "num_decode_tokens", "num_spec_decodes",
           "num_spec_decode_tokens", "num_actual_tokens", "has_initial_state", "chunk_indices", "chunk_offsets",
           "prefill_query_start_loc", "prefill_state_indices", "prefill_has_initial_state", "spec_query_start_loc",
           "non_spec_query_start_loc", "spec_state_indices_tensor", "non_spec_state_indices_tensor",
           "spec_sequence_masks", "spec_token_indx", "non_spec_token_indx", "num_accepted_tokens", "nums_dict",
           "batch_ptr", "token_chunk_offset_ptr")


def _snap(md):
    return {f: (getattr(md, f).clone() if isinstance(getattr(md, f), torch.Tensor) else getattr(md, f)) for f in _FIELDS}


def _diff(a, b):
    bad = []
    for f in _FIELDS:
        x, y = a[f], b[f]
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape
                    and x.dtype == y.dtype and torch.equal(x, y)):
                bad.append(f)
        elif x != y:
            bad.append(f)
    return bad


def install(cls_b, orig_build):
    """Wraps GDNAttentionMetadataBuilder.build (called by gdn_port.install with the builder it wraps)."""
    if MODE == 0:
        return orig_build
    from vllm.v1.attention.backends import gdn_attn as ga
    meta_cls, null_id = ga.GDNAttentionMetadata, ga.NULL_BLOCK_ID

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        md = None
        try:
            if _ST.get("err", 0):
                raise _Disabled()
            md = try_fast(self, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu, meta_cls, null_id)
        except _Disabled:
            md = None
        except Exception as e:  # noqa: never break a step; first failure disables the fast path for good
            if _ST.get("err", 0) < 3:
                logger.warning("exl3: GDN-FAST failed (%s: %s); original builder", type(e).__name__, e)
            _ST["err"] = _ST.get("err", 0) + 1
            md = None
        if md is not None and MODE == 2:
            a = _snap(md)
            ref = orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                             num_decode_draft_tokens_cpu, fast_build)
            bad = _diff(a, _snap(ref))
            _ST["checked"] += 1
            if bad:
                _ST["mismatch"] += 1
                if _ST["mismatch"] <= 5:
                    ra = _snap(ref)
                    logger.warning("exl3: GDN-FAST MISMATCH vs original in %s; %s", bad, "; ".join(
                        f"{f}: fast {a[f].tolist()[:3] if isinstance(a[f], torch.Tensor) else a[f]} "
                        f"orig {ra[f].tolist()[:3] if isinstance(ra[f], torch.Tensor) else ra[f]}" for f in bad[:2]))
            if _ST["checked"] in (1, 100, 1000, 10000, 100000):
                logger.info("exl3: GDN-FAST check: %d steps compared, %d mismatches", _ST["checked"], _ST["mismatch"])
            return ref
        if md is None:
            _ST["orig"] += 1
            return orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                              num_decode_draft_tokens_cpu, fast_build)
        _ST["fast"] += 1
        if _ST["fast"] in (1, 1000, 100000):
            logger.info("exl3: GDN-FAST %d fast builds, %d original", _ST["fast"], _ST["orig"])
        return md
    logger.info("exl3: GDN-FAST on (mode %d)", MODE)
    return build
