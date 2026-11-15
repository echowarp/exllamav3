"""GDN-PORT: GDN prefill through the Magnitude WY port (csrc/gdnchunk/gdn_chunk.cu)
instead of vLLM's causal_conv1d_fn + fused_post_conv_prep + FLA chunk_gated_delta_rule. EXL3_GDN_PORT=1 (default on
sm_120, see hwdefaults.py; 0 = FLA).

Why (kernel benchmark, RTX 5090):
- more accurate vs fp64: out rel-RMS 4.3e-4 vs FLA 6.9e-4; state 2.9e-4 vs 6.2e-4;
- split-invariant: resume from a checkpoint == one-shot, bit for bit (FLA is not); needed for prefix-state reuse;
- 2.2x faster at 2k rows, 1.14x at 8k, and ~30x for tiny prefills (0.024 vs 0.79 ms per layer: FLA has a fixed
  ~0.8 ms/layer cost, ~38 ms per step over 48 layers, for any prefill row in a decode step).

What it does: one launch set per GDN layer for all prefill sequences of the step. The kernel reads and writes the
engine's pages in place:
- state = kv_cache[1] [N, HV, V, K] f32;
- window = rows 0..2 of the SD conv page kv_cache[0] [N, 3+k, CH]. vLLM's prefill conv writes those rows too (checked
  bit-equal).
Per sequence: absolute start = num_computed_tokens (pieces are aligned to absolute 16-row positions);
src = state index if it has an initial state, else -1 (zero state and window); one exact checkpoint at the end into
the same index (in place is safe: the window is published by a separate launch).
Pad indices (< 0) are clamped to block 0 (vLLM's null block) instead of indexing out of bounds.
The tail buffer is a 1-row dummy (exact checkpoints never write or read it; tail_count stays 0).
Falls back to vLLM for:
- metadata the hook did not prepare;
- a non-spec batch with peeled decodes;
- a DS conv layout or non-f16 activations.
"""
import importlib.machinery
import importlib.util
import logging
import os

import torch

try:                                                    # vLLM's logger: a plain logging.getLogger has no handler
    from vllm.logger import init_logger                 # in the engine (messages were lost without it)
    logger = init_logger("vllm.exl3cuda")
except Exception:  # noqa
    logger = logging.getLogger(__name__)
_SO = os.environ.get("EXL3_GDN_PORT_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "gdn_chunk_mag.so"))
_ROWS = int(os.environ.get("EXL3_GDN_PORT_ROWS", "64"))      # scan tile; 64 fastest at 2k and 8k (bench)
_EPS = 1e-6
_ST = {"ext": None, "tail": None, "tail_count": {}, "calls": 0, "fallback": 0}


def _ext():
    if _ST["ext"] is None:
        loader = importlib.machinery.ExtensionFileLoader("gdn_chunk_mag", _SO)
        spec = importlib.util.spec_from_file_location("gdn_chunk_mag", _SO, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _ST["ext"] = mod
        logger.info("exl3: GDN-PORT extension loaded from %s (rows %d)", _SO, _ROWS)
    return _ST["ext"]


def _prepare(md, m, spec_mask_gpu):
    """Index tensors for the port, once per GDN group and step (all on the GPU, no host sync)."""
    if md.num_prefills <= 0 or md.prefill_query_start_loc is None:
        return None
    if md.spec_sequence_masks is None and md.num_decodes > 0:
        return None                                    # peeled decodes: vLLM path (rare with MTP on)
    cu = md.prefill_query_start_loc.to(torch.int32)
    ctx = m.compute_num_computed_tokens()
    if spec_mask_gpu is not None:
        ctx = ctx[~spec_mask_gpu]
    B = int(cu.numel() - 1)                            # same sequence list vLLM hands to its chunk kernel
    ctx = ctx[:B].to(torch.int32)
    rows = cu[1:] - cu[:-1]
    idx = md.prefill_state_indices.to(torch.int32)
    has = md.prefill_has_initial_state
    src = torch.where(has, idx, torch.full_like(idx, -1))
    src = torch.where(src < 0, torch.full_like(src, -1), src)
    dst = idx.clamp(min=0)
    return dict(cu=cu.contiguous(), pos0=ctx.contiguous(), src=src.contiguous(),
                ck_cu=torch.arange(B + 1, dtype=torch.int32, device=cu.device),
                ck_pos=(ctx + rows).contiguous(), ck_dst=dst.contiguous(),
                ck_kind=torch.zeros(B, dtype=torch.int32, device=cu.device), B=B)


def _weights(layer):
    w = getattr(layer, "_llmt_port_w", None)
    if w is None:
        cw = layer.conv1d.weight
        conv = cw.view(cw.size(0), cw.size(2)).float().contiguous()
        rate = (-torch.exp(layer.A_log.float())).contiguous()
        tb = layer.dt_bias.float().contiguous()
        w = layer._llmt_port_w = (conv, rate, tb)
    return w


def _run(layer, md, mixed_qkv, b, a, core_attn_out):
    """True when the port handled the layer; False = caller runs vLLM's path."""
    pm = getattr(md, "_llmt_port", None)
    if pm is None or md.spec_sequence_masks is not None:
        return False
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import is_conv_state_dim_first
    if is_conv_state_dim_first() or mixed_qkv.dtype != torch.float16 or layer.conv1d.bias is not None:
        return False
    nT = int(md.num_actual_tokens)
    kvc = layer.kv_cache
    state, conv_pages = kvc[1], kvc[0]
    window = conv_pages[:, :3, :]
    S = state.size(0)
    tc = _ST["tail_count"].get((S, state.device))
    if tc is None:
        tc = _ST["tail_count"][(S, state.device)] = torch.zeros(S, dtype=torch.int32, device=state.device)
    ext = _ext()
    if _ST["tail"] is None:
        _ST["tail"] = torch.zeros(1, ext.TAIL if hasattr(ext, "TAIL") else 15, 10336, dtype=torch.float16,
                                  device=state.device)
    conv, rate, tb = _weights(layer)
    q, a_, b_ = mixed_qkv[:nT], a[:nT], b[:nT]
    if a_.stride(-1) != 1:
        a_ = a_.contiguous()
    if b_.stride(-1) != 1:
        b_ = b_.contiguous()
    out = core_attn_out[:nT]
    direct = out.is_contiguous()
    o = out if direct else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    scratch = torch.empty(ext.scratch_bytes(nT, pm["B"], pm["B"]), dtype=torch.uint8, device=q.device)
    ext.forward(q, a_, b_, conv, rate, tb, pm["cu"], pm["pos0"], pm["src"], pm["ck_cu"], pm["ck_pos"], pm["ck_dst"],
                pm["ck_kind"], state, window, _ST["tail"], tc, o, scratch, _EPS, True, _ROWS)
    if not direct:
        out.copy_(o)
    _ST["calls"] += 1
    if _ST["calls"] in (1, 1000, 10000, 100000, 1000000):
        logger.info("exl3: GDN-PORT %d layer calls through the port (this one: %d rows, %d sequences); vLLM fallbacks "
                    "%d", _ST["calls"], nT, pm["B"], _ST["fallback"])
    return True


def install():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qg
    from vllm.v1.attention.backends import gdn_attn as ga
    from .hwdefaults import sm120
    if not sm120():
        raise RuntimeError("GDN-PORT needs an sm_120 GPU (csrc/gdnchunk is an sm_120 build)")
    _ext()                                             # fail at start, not mid-request
    cls_b = ga.GDNAttentionMetadataBuilder
    if getattr(cls_b, "_llmt_port", False):
        return
    orig_build = cls_b.build
    try:                                               # GDN-FAST: decode-step fast path under the port
        from exl3cuda import gdn_fast
        orig_build = gdn_fast.install(cls_b, orig_build)
    except Exception as e:  # noqa
        logger.warning("exl3: GDN-FAST not installed (%s)", e)

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        md = orig_build(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        try:
            pm = _prepare(md, common_attn_metadata, md.spec_sequence_masks)
            if pm is not None:
                object.__setattr__(md, "_llmt_port", pm)
        except Exception as e:  # noqa: never break a step; the layer falls back to vLLM
            _ST["fallback"] += 1
            if _ST["fallback"] <= 3:
                logger.warning("exl3: GDN-PORT prepare failed (%s: %s); vLLM path", type(e).__name__, e)
        return md
    cls_b.build = build
    cls_b._llmt_port = True

    cls = qg.QwenGatedDeltaNetAttention
    orig_core = cls._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        md_all = get_forward_context().attn_metadata
        md = md_all.get(self.prefix) if isinstance(md_all, dict) else None
        if md is not None and getattr(md, "_llmt_port", None) is not None:
            if _run(self, md, mixed_qkv, b, a, core_attn_out):
                return
            _ST["fallback"] += 1
        return orig_core(self, mixed_qkv, b, a, core_attn_out)
    cls._forward_core = _forward_core
    logger.info("exl3: GDN-PORT enabled: GDN prefill via the Magnitude WY port (%s)", _SO)
