"""
vLLM out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on NVIDIA CUDA (RTX 5090).

CUDA port of the Intel Arc (XPU) exl3 plugin: the checkpoint loading, fused-module handling, mixed-bitrate modules and
the graded MTP draft-vocabulary head are shared with it; every linear runs on exllamav3's CUDA EXL3 kernels
(exllamav3.modules.quant.exl3.LinearEXL3: small batches run the fused trellis GEMM, large batches reconstruct + HGEMM).
One LinearEXL3 per member of a fused module (each member has its own input sign vector suh); outputs are concatenated.

The call is wrapped in the torch.library custom op exl3cuda::linear so torch.compile and CUDA graphs see one opaque op.
Everything beyond the plain EXL3 linear is opt-in through EXL3_* environment variables (see README.md).

Registered via the `vllm.general_plugins` entry point, so every vLLM process imports it before model construction.
"""
from __future__ import annotations
import itertools
import json
import os
import re
import time
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from .hwdefaults import env as _hwenv, sm120 as _sm120, device_ok as _device_ok
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger("vllm.exl3cuda")

# vLLM fused module -> checkpoint constituents
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
CB_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


def _norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj'
       'mtp.layers.0.mlp.gate_proj' / 'model.mtp.layers.0...' -> 'mtp.layers.0.mlp.gate_proj'
       '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):

    def __init__(self, bits: float, head_bits: int, codebook: str, storage: dict | None = None,
                 mtp_bits: int | None = None):
        super().__init__()
        self.bits = bits
        self.mtp_bits = int(mtp_bits or bits)
        self.head_bits = head_bits
        self.codebook = codebook
        self.storage = storage or {}

    def __repr__(self):
        return f"Exl3Config(bits={self.bits}, head_bits={self.head_bits}, codebook={self.codebook})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(bits=config.get("bits", 4), head_bits=config.get("head_bits", 6),
                   codebook=config.get("codebook", "mul1"), mtp_bits=config.get("mtp_bits"))

    def _model_file(self, model_name, fname, revision):
        path = os.path.join(model_name, fname)
        if os.path.isfile(path):
            return path
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_name, fname, revision=revision)
        except Exception:
            return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        # Which modules are EXL3 is decided by the checkpoint itself: a module is EXL3 iff the weight
        # index has '<module>.trellis'. (quantization_config.json's tensor_storage omits the MTP head.)
        idx_path = self._model_file(model_name, "model.safetensors.index.json", revision)
        qc_path = self._model_file(model_name, "quantization_config.json", revision)
        stored = {}
        if qc_path:
            with open(qc_path) as f:
                for key, v in json.load(f).get("tensor_storage", {}).items():
                    if v.get("quant_format") == "exl3":
                        stored[_norm_key(key)] = int(v.get("bits_per_weight", self.bits))
        if idx_path:
            with open(idx_path) as f:
                wmap = json.load(f)["weight_map"]
            for name in wmap:
                if name.endswith(".trellis"):
                    key = _norm_key(name[: -len(".trellis")])
                    default = self.mtp_bits if key.startswith("mtp.") else int(self.bits)
                    self.storage[key] = stored.get(key, default)
        else:
            self.storage.update(stored)
        logger.info("exl3: %d EXL3 modules (%d in MTP head)", len(self.storage),
                    sum(1 for k in self.storage if k.startswith("mtp.")))

    def _bits_for(self, prefix: str) -> int | None:
        parts = prefix.split(".")
        if "visual" in parts or "vision_tower" in parts:
            return None
        key = _norm_key(prefix)
        if not self.storage:
            if key.endswith("in_proj_ba"):
                return None
            return self.head_bits if key == "lm_head" else int(self.bits)
        base, _, leaf = key.rpartition(".")
        members = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in members]
        bits = [self.storage.get(n) for n in names]
        if all(b is None for b in bits):
            return None
        if any(b != bits[0] for b in bits):
            # mixed-precision checkpoints (e.g. turboderp SC_*_H5): one EXL3 call per equal-K run of members
            assert all(b is not None for b in bits), f"exl3: partly quantized fused module {prefix}: {bits}"
            return tuple(bits)
        return bits[0]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bits = self._bits_for(prefix)
            if bits is None:
                return UnquantizedLinearMethod()
            if isinstance(bits, tuple):
                return Exl3MixedLinearMethod(self, bits)
            m = Exl3LinearMethod(self, bits)
            m.prefix = prefix                      # EXL3_DRAFT_WEIGHTS: find mtp.* modules at load time
            return m
        if isinstance(layer, ParallelLMHead):
            bits = self._bits_for(prefix)
            if bits is None:
                return None
            return Exl3LinearMethod(self, bits)
        return None


class Exl3LinearMethod(LinearMethodBase):

    def __init__(self, config: Exl3Config, bits: int):
        self.config = config
        self.K = int(bits)
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        if k != input_size or n != (output_size if not isinstance(layer, ParallelLMHead) else n):
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        assert k % 128 == 0 and all(s % 128 == 0 for s in output_partition_sizes), \
            f"exl3: dims must be multiples of 128 (k={k}, n={output_partition_sizes})"
        K = self.K
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_groups = set()
        layer.exl3_K = K
        layer.exl3_cb = self.cb

        def reg(name, tensor, loader):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), self._load_trellis(layer))
        reg("suh", torch.empty((len(output_partition_sizes), k), dtype=torch.float16), self._load_suh(layer))
        reg("svh", torch.empty((n,), dtype=torch.float16), self._load_svh(layer))
        if self.config.codebook != "3inst":
            reg(self.config.codebook, torch.empty((), dtype=torch.int32), lambda p, w, *a, **kw: p.data.copy_(w))

    @staticmethod
    def _shards(layer, loaded, sid=None) -> list[int]:
        if sid is None:
            sid = getattr(loaded, "shard_id", None)
        if sid is None:
            return list(range(len(layer.exl3_sizes)))
        if isinstance(sid, str):
            return [QKV_IDS[sid]]
        if isinstance(sid, int):
            return [sid]
        return list(sid)

    def _range(self, layer, loaded, sid=None):
        s = self._shards(layer, loaded, sid)
        assert s == list(range(s[0], s[-1] + 1)), s
        return s, layer.exl3_offsets[s[0]], layer.exl3_offsets[s[-1] + 1]

    def _load_trellis(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            assert w.dtype == torch.int16 and w.shape[-1] == param.shape[-1], \
                f"exl3: trellis bits mismatch {tuple(w.shape)} vs {tuple(param.shape)}"
            assert w.shape[1] * 16 == b - a and w.shape[0] == param.shape[0], (w.shape, a, b)
            param.data[:, a // 16: b // 16].copy_(w)
            layer.exl3_groups.add(tuple(s))
        return f

    def _load_suh(self, layer):
        def f(param, w, sid=None):
            s, _, _ = self._range(layer, w, sid)
            for i in s:
                param.data[i].copy_(w)
        return f

    def _load_svh(self, layer):
        def f(param, w, sid=None):
            _, a, b = self._range(layer, w, sid)
            param.data[a:b].copy_(w)
        return f

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_groups) or [tuple(range(len(layer.exl3_sizes)))]
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        bounds = [layer.exl3_offsets[g[0]] for g in groups] + [layer.exl3_offsets[-1]]
        cbt = _codebook_tensors(layer)
        k = layer.trellis.shape[0] * 16
        segs = []
        for gi, g in enumerate(groups):
            a, b = bounds[gi], bounds[gi + 1]
            tr = layer.trellis.data if len(groups) == 1 else layer.trellis.data[:, a // 16: b // 16].contiguous()
            segs.append(dict(tr=tr, suh=layer.suh.data[g[0]].contiguous(), svh=layer.svh.data[a:b].contiguous()))
        replaced = _draft_weights_swap(getattr(self, "prefix", None), groups, segs, layer.trellis.device)
        _cap_key = _norm_key(getattr(self, "prefix", "") or "")
        heads = isinstance(layer, ParallelLMHead) and os.environ.get("EXL3_DRAFT_VOCAB") and _spec_is_mtp()
        late = None
        if heads and layer.trellis.is_meta:   # weight keeper meta build: the heads need data, built in whx_finish
            late = [_whx_reserve() for _ in range(_draft_head_regs())]
        elif heads:
            _draft_heads(layer, cbt)
        layer.exl3_id, layer.exl3_n = _register(segs, k, cbt), bounds[-1]
        if late is not None:
            _WHX["late"].append((layer, late))
        elif _DYN_ON and isinstance(layer, ParallelLMHead) and getattr(layer, "exl3_draft", None) is not None:
            _build_draft_dyn(layer)   # DRAFT-DYN + context bias, DYN2 host path (needs layer.exl3_id)
        if _CAP_DIR and _cap_key.startswith("mtp."):
            _CAPTURE[layer.exl3_id] = _cap_key
        if len(groups) > 1 or replaced:   # the per-member (or draft-package) copies replace the fused trellis
            layer.trellis = Parameter(torch.empty(0, dtype=torch.int16, device=layer.trellis.device), requires_grad=False)

    def apply(self, layer, x, bias=None):
        y = torch.ops.exl3cuda.linear(x, layer.exl3_id, layer.exl3_n)
        if bias is not None:
            y = y + bias
        return y

    # lm_head (ParallelLMHead) path
    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


class Exl3MixedLinearMethod(LinearMethodBase):
    """Fused module whose checkpoint members have different bit widths. Each loaded
    checkpoint tensor is kept as its own group (own trellis K, suh, svh); consecutive groups with equal K are
    concatenated into one EXL3 call, and apply() concatenates the call outputs along the output dim."""

    def __init__(self, config: Exl3Config, bits: tuple):
        self.config = config
        self.bits = bits
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        if k != input_size:
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_mx = {}                 # group (tuple of partition ids) -> {"tr", "suh", "svh"}
        layer.exl3_cb = self.cb
        layer.exl3_k_in = k

        def reg(name, loader):
            p = Parameter(torch.empty(1, dtype=torch.int16 if name == "trellis" else torch.float16),
                          requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        def store(key):
            def f(param, w, sid=None):
                s = tuple(Exl3LinearMethod._shards(layer, w, sid))
                layer.exl3_mx.setdefault(s, {})[key] = w.to(param.device)
            return f
        reg("trellis", store("tr"))
        reg("suh", store("suh"))
        reg("svh", store("svh"))
        if self.config.codebook != "3inst":
            p = Parameter(torch.empty((), dtype=torch.int32), requires_grad=False)
            p.weight_loader = lambda p, w, *a, **kw: p.data.copy_(w)
            layer.register_parameter(self.config.codebook, p)

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_mx)
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        cbt = _codebook_tensors(layer)
        segs = [dict(tr=layer.exl3_mx[g]["tr"].contiguous(), suh=layer.exl3_mx[g]["suh"].contiguous(),
                     svh=layer.exl3_mx[g]["svh"].contiguous()) for g in groups]
        layer.exl3_id = _register(segs, layer.exl3_k_in, cbt)
        layer.exl3_n = sum(s["svh"].shape[0] for s in segs)
        layer.exl3_mx = None
        for name in ("trellis", "suh", "svh"):
            delattr(layer, name)
        logger.debug("exl3: mixed fused module, K per member %s", [s["tr"].shape[-1] // 16 for s in segs])

    def apply(self, layer, x, bias=None):
        y = torch.ops.exl3cuda.linear(x, layer.exl3_id, layer.exl3_n)
        if bias is not None:
            y = y + bias
        return y

    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


# ------------------------------------------------------------------------------------------------
# CUDA execution: exllamav3 LinearEXL3 per segment, behind the opaque custom op exl3cuda::linear.

_SEGS: list[list] = []          # layer id -> [LinearEXL3, ...] (one per fused-module member)


def _codebook_tensors(layer):
    """exllamav3 selects the codebook by which tensor is present (mcg / mul1); 3inst = neither."""
    return {"mcg": getattr(layer, "mcg", None), "mul1": getattr(layer, "mul1", None)}


def _lins(segs, k, cbt) -> list:
    from exllamav3.modules.quant.exl3 import LinearEXL3
    return [LinearEXL3(None, k, s["svh"].shape[0], suh=s["suh"], svh=s["svh"], trellis=s["tr"],
                       mcg=cbt["mcg"], mul1=cbt["mul1"], out_dtype=torch.half) for s in segs]


def _fuse(i: int):
    lins = _SEGS[i]
    if FUSE_DECODE and len(lins) > 1:
        try:
            _FUSED[i] = _Sliced(lins, FUSE_WIDTH)
        except ValueError as e:
            logger.debug("exl3: fused decode not used for this module: %s", e)


def _register(segs, k, cbt) -> int:
    meta = any(s["tr"].is_meta for s in segs)
    v = _WhxMeta(segs, k, cbt) if meta else _lins(segs, k, cbt)
    if _WHX["fill"]:                      # whx_finish: a draft head takes the id reserved at its meta build
        i = _WHX["fill"].pop(0)
        _SEGS[i] = v
    else:
        _SEGS.append(v)
        i = len(_SEGS) - 1
    if not meta:
        _fuse(i)
    return i


# ------------------------------------------------------------------------------------------------
# Weight keeper hooks (AllBroker weights/vllm/whx_loader.py, EXL3_WHX). The keeper's attach builds the model on the meta
# device and then puts views of the held VRAM in place of its tensors. Two things here need help for that:
#   - the EXL3 linears' tensors live in _SEGS (LinearEXL3 objects), not in the model's parameters: whx_registry() hands
#     them to the keeper by module name, so an export moves them and an attach matches them by path;
#   - LinearEXL3 (its C++ BC object) and _Sliced (raw data pointers) must be built from real tensors: a meta build
#     registers _WhxMeta placeholders, and whx_finish() builds them once the held tensors are in place. It also redoes
#     _Sliced after an export (the export moved the tensors, so the pointer tables were stale) and runs the draft-head /
#     DRAFT-DYN steps a meta build deferred, at the _SEGS ids reserved for them (the ids are compiled into the graphs).
# The draft heads are not held: every attach rebuilds them from the held lm_head (or the draft package file).
_WHX: dict = {"fill": [], "late": []}


class _WhxMeta:
    """_SEGS placeholder of a meta build: one tensor dict per member (tr, suh, svh, mcg / mul1 if present)."""

    def __init__(self, segs, k, cbt):
        self.k = k
        self.members = [dict(tr=s["tr"], suh=s["suh"], svh=s["svh"],
                             **{n: t for n, t in cbt.items() if t is not None}) for s in segs]


def _whx_reserve() -> int:
    _SEGS.append(None)
    return len(_SEGS) - 1


def _draft_head_regs() -> int:
    """How many _register calls _draft_heads makes (whx_finish checks it)."""
    dw = _draft_weights()
    if dw is not None and "draft_head.trellis" in dw:
        return 1                                          # near and far share the draft package's head
    far = os.environ.get("EXL3_DRAFT_VOCAB_FAR")
    return 1 + int(bool(far) and os.path.realpath(far) != os.path.realpath(os.environ["EXL3_DRAFT_VOCAB"]))


def whx_registry(model) -> dict:
    """{module name: [{tr, suh, svh[, mcg][, mul1][, bias]} per member]} for every EXL3 linear of the model."""
    out = {}
    for name, m in model.named_modules():
        i = getattr(m, "exl3_id", None)
        if type(i) is not int or not 0 <= i < len(_SEGS) or _SEGS[i] is None:
            continue
        v = _SEGS[i]
        if isinstance(v, _WhxMeta):
            out[name] = v.members
        else:
            out[name] = [{n: t for n, t in (("tr", l.trellis), ("suh", l.suh), ("svh", l.svh), ("mcg", l.mcg_tensor),
                                             ("mul1", l.mul1_tensor), ("bias", l.bias)) if t is not None} for l in v]
    return out


def whx_finish(model):
    mods = list(model.modules())
    ids = sorted({m.exl3_id for m in mods if type(getattr(m, "exl3_id", None)) is int})
    built = 0
    for i in ids:
        v = _SEGS[i]
        if isinstance(v, _WhxMeta):
            if any(t.is_meta for d in v.members for t in d.values()):
                raise RuntimeError(f"exl3: WHX: linear {i} still has meta tensors")
            segs = [dict(tr=d["tr"], suh=d["suh"], svh=d["svh"]) for d in v.members]
            cbt = {"mcg": v.members[0].get("mcg"), "mul1": v.members[0].get("mul1")}
            _SEGS[i] = _lins(segs, v.k, cbt)
            built += 1
        _FUSED.pop(i, None)
        _fuse(i)
    mine = {id(m) for m in mods}
    late = [x for x in _WHX["late"] if id(x[0]) in mine]
    for item in late:
        _WHX["late"].remove(item)
        layer, res = item
        n0 = len(_SEGS)
        _WHX["fill"][:] = res
        try:
            _draft_heads(layer, _codebook_tensors(layer))   # the held views, not the meta build's tensors
        finally:
            left = list(_WHX["fill"])
            _WHX["fill"].clear()
        if left or len(_SEGS) != n0:
            raise RuntimeError(f"exl3: WHX: draft heads used {len(res) - len(left) + len(_SEGS) - n0} linear ids, "
                               f"{len(res)} reserved")
        if _DYN_ON and getattr(layer, "exl3_draft", None) is not None:
            _build_draft_dyn(layer)
    logger.info("exl3: WHX: %d linears built from held memory, %d fused tables redone, %d lm_head(s) finished",
                built, sum(1 for i in ids if i in _FUSED), len(late))


# FUSED-DECODE: at decode/verify row counts, run a fused module's members (gate+up, q+k+v,
# GDN qkv+z) as ONE exllamav3 sliced mgemm launch instead of one exl3 GEMM per member plus a torch.cat. The members
# keep their own trellis, suh (input transform) and svh; the kernel schedules equal-width column slices of all members
# as one grid, so the per-launch ramp/tail (~8-10 us at these sizes on the 5090) is paid once. Same math as the
# per-member path (fp16 output rounding may differ in the last bit). EXL3_FUSE_DECODE=1, rows <= EXL3_FUSE_MAX_M,
# slice width EXL3_FUSE_WIDTH (0 = gcd of the member widths).
FUSE_DECODE = _hwenv("EXL3_FUSE_DECODE", "1", "0") == "1"
FUSE_MAX_M = int(_hwenv("EXL3_FUSE_MAX_M", "64", "16"))
FUSE_WIDTH = int(os.environ.get("EXL3_FUSE_WIDTH", "0"))
_FUSED: dict = {}
# FUSE-SCRATCH: for m > 1 the member blocks are scratch until the torch.cat, so all fused modules
# share one persistent scratch buffer and precomputed output-pointer tables; this drops the per-call `coff[m] + ptr`
# device add (133 single-block launches per 1-stream step). Calls run in stream order, the cat copies out before the
# next module writes. EXL3_FUSE_SCRATCH=1 enables.
FUSE_SCRATCH = _hwenv("EXL3_FUSE_SCRATCH", "1", "0") == "1"
_FUSE_SCRATCH_ELEMS = 65536 * 16
_SCRATCH: dict = {}


class _Sliced:
    def __init__(self, lins, width=0):
        import math
        l0 = lins[0]
        if not all(isinstance(l.K, int) and l.K == l0.K and getattr(l, "frac", None) is None and l.mcg == l0.mcg
                   and l.mul1 == l0.mul1 and l.in_features == l0.in_features and l.trellis.is_contiguous()
                   and l.trellis.dtype == torch.int16 and l.suh is not None and l.svh is not None for l in lins):
            raise ValueError("members differ (K / codebook / layout)")
        self.ns = [l.out_features for l in lins]
        self.n, self.k, self.K, self.nsrc = sum(self.ns), l0.in_features, l0.K, len(lins)
        self.mcg, self.mul1 = l0.mcg, l0.mul1
        w = width or math.gcd(*self.ns)
        if w % 128 or any(n % w for n in self.ns):
            raise ValueError(f"slice width {w} unsuitable for {self.ns}")
        self.w = w
        dev = l0.trellis.device
        tp, sp, stride, src, a, b = [], [], [], [], [], []
        off = 0
        for i, l in enumerate(lins):
            for n0 in range(0, l.out_features, w):
                tp.append(l.trellis.data_ptr() + (n0 // 16) * l.trellis.shape[-1] * l.trellis.element_size())
                sp.append(l.svh.data_ptr() + n0 * l.svh.element_size())
                stride.append(l.out_features)
                src.append(i)
                a.append(off * 2)          # member block offset (bytes per row of m): block i starts at m * off
                b.append(n0 * 2)
            off += l.out_features
        self.offs = [sum(self.ns[:i]) for i in range(self.nsrc + 1)]
        self.ptrs_tr = torch.tensor(tp, dtype=torch.long, device=dev)
        self.ptrs_svh = torch.tensor(sp, dtype=torch.long, device=dev)
        self.ptrs_suh = torch.tensor([l.suh.data_ptr() for l in lins], dtype=torch.long, device=dev)
        self.size_n = torch.full((len(tp),), w, dtype=torch.int32, device=dev)
        self.stride = torch.tensor(stride, dtype=torch.int32, device=dev)
        self.src = torch.tensor(src, dtype=torch.int32, device=dev)
        self.S = len(tp)
        # output byte offset per (rows m, slice): members are laid out as consecutive (m, n_i) blocks; precomputed for
        # every m so no host->device copy happens inside graph capture
        A, B = torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)
        self.coff = torch.stack([A * m + B for m in range(FUSE_MAX_M + 1)]).to(dev)
        self.cps = None
        if FUSE_SCRATCH and self.n * FUSE_MAX_M <= _FUSE_SCRATCH_ELEMS:
            sb = _SCRATCH.get(dev)
            if sb is None:
                sb = _SCRATCH[dev] = torch.empty((_FUSE_SCRATCH_ELEMS,), dtype=torch.half, device=dev)
            self.scratch = sb
            self.cps = self.coff + sb.data_ptr()       # [FUSE_MAX_M + 1, S], computed once (not under graph capture)
        self.carrier = torch.empty((1, 1, w), dtype=torch.half, device=dev)
        from exllamav3.ext import exllamav3_ext
        self.ext = exllamav3_ext

    def __call__(self, x2):
        m = x2.shape[0]
        if m > 1 and self.cps is not None:
            buf = self.scratch[: m * self.n]
            cp = self.cps[m]
        else:
            buf = torch.empty((m * self.n,), dtype=torch.half, device=x2.device)
            cp = self.coff[m] + buf.data_ptr()
        had = torch.empty((self.nsrc, m, self.k), dtype=torch.half, device=x2.device)
        self.ext.exl3_mgemm(x2.view(1, m, self.k), self.ptrs_tr, self.carrier.expand(self.S, m, self.w), self.ptrs_suh,
                            had, self.ptrs_svh, None, None, self.K, -1, self.mcg, self.mul1, -1, -1, 0, 1, self.size_n,
                            cp, self.stride, self.src, self.nsrc)
        if m == 1:                         # one row: the member blocks already are the concatenated row
            return buf.view(1, self.n)
        return torch.cat([buf[m * a: m * b].view(m, b - a) for a, b in zip(self.offs, self.offs[1:])], dim=-1)


# INT8-PREFILL (CUDA port of the XPU plugin's EXL3_INT8_PREFILL, same numerics as its Python prototype): for prefill
# rows, GEMM in int8 on the tensor cores instead of fp16. Activations are rotated into the Hadamard basis (no outliers)
# and quantized per token; the reconstructed rotated-basis weights are quantized per output column with scales computed
# once (weights are static). Then int32 accumulate (torch._int_mm = cuBLASLt IMMA), rescale, output Hadamard.
INT8_PREFILL = _hwenv("EXL3_INT8_PREFILL", "1", "0") == "1"
INT8_MIN_ROWS = int(os.environ.get("EXL3_INT8_MIN_ROWS", "256"))
SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))
_WBUF: dict = {}


def _wbuf(dev, numel, dtype):
    key = (dev, dtype)
    b = _WBUF.get(key)
    if b is None or b.numel() < numel:
        b = _WBUF[key] = torch.empty(numel, dtype=dtype, device=dev)
    return b[:numel]


def _col_scales(lin):
    """Per-output-column int8 scale of the rotated-basis weight (computed once per segment)."""
    sw = getattr(lin, "_llmt_sw", None)
    if sw is None:
        k, n = lin.in_features, lin.out_features
        sw = torch.empty(n, dtype=torch.float32, device=lin.trellis.device)
        for n0 in range(0, n, SLICE_N):
            n1 = min(n0 + SLICE_N, n)
            w = _wbuf(lin.trellis.device, k * (n1 - n0), torch.half).view(k, n1 - n0)
            lin._reconstruct_inner(w, n0)
            sw[n0:n1] = w.abs().amax(0).float().clamp_min(1e-8) / 127.0
        lin._llmt_sw = sw
    return sw


def _int8_linear(lin, x, y=None):
    """Fused int8 prefill linear (Triton kernels in int8_kernels.py): had_in -> per-token int8 -> per slice:
    reconstruct fp16 -> int8 column-major (static column scales) -> int8 GEMM with scale epilogue + output Hadamard
    fused (EXL3_INT8_HAD_EPI=1, default) or followed by exllamav3's had_r_128. y: optional output (may be a column
    slice of the fused module's output, so members write in place)."""
    from exllamav3.ext import exllamav3_ext as ext
    from . import int8_kernels as K8
    rows, k = x.shape
    n = lin.out_features
    if _HAD_QUANT and rows >= _HAD_QUANT_MIN and k % 128 == 0:
        xq, sx = K8.had_quant(x, lin.suh)            # input Hadamard + per-token quant fused (no fp16 intermediate)
    else:
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, lin.suh, None, 1.0)
        xq, sx = K8.quant_rows(xh)
    sw = _col_scales(lin)
    fused = _HAD_EPI and n % 128 == 0
    if y is None or not fused:
        y_ = torch.empty((rows, n), dtype=torch.half, device=x.device)
    else:
        y_ = y
    if _L2SPLIT and fused and _recon_q8_ok(lin) and k % 128 == 0:
        # L2SPLIT: recon_fast (bit-identical, faster
        # trellis decode) into int8 slices sized to stay in L2 (64 MB; full width for k > 8192, the down projection),
        # then the same gemm_i8_had. Bit-identical output; per-forward linears -5.0/-6.8/-7.6 ms at 512/1024/2048 rows.
        from . import fused_kernels as FK
        FK.l2_split_linear(xq, sx, lin.trellis, lin.K, sw, lin.svh, y_, _L2_BUDGET if k <= 8192 else SLICE_N * k,
                           asm=False)
        return y_
    for n0 in range(0, n, SLICE_N):
        n1 = min(n0 + SLICE_N, n)
        wqT = _wbuf(x.device, k * (n1 - n0), torch.int8).view(n1 - n0, k)
        if _recon_q8_ok(lin):                         # trellis -> int8 [n, k] in one pass (bit-exact with the two below)
            K8.recon_q8T(lin.trellis, lin.K, sw[n0:n1], wqT, n0)
        else:
            w = _wbuf(x.device, k * (n1 - n0), torch.half).view(k, n1 - n0)
            lin._reconstruct_inner(w, n0)
            K8.quant_wT(w, sw[n0:n1], wqT)
        if fused:
            K8.gemm_i8_had(xq, wqT, sx, sw[n0:n1], lin.svh[n0:n1], y_[:, n0:n1])
        else:
            K8.gemm_i8_scaled(xq, wqT, sx, sw[n0:n1], y_[:, n0:n1])
    if not fused:
        ext.had_r_128(y_, y_, None, lin.svh, 1.0)
        if y is not None:
            y.copy_(y_)
            return y
    return y_


_HAD_EPI = os.environ.get("EXL3_INT8_HAD_EPI", "1") == "1"
_L2SPLIT = _hwenv("EXL3_INT8_L2SPLIT", "1", "0") == "1"
_L2_BUDGET = int(os.environ.get("EXL3_INT8_L2_BUDGET_MB", "64")) << 20
# EXL3_INT8_HAD_QUANT=1 (default): fused input Hadamard + activation quant (int8_kernels.had_quant) for chunks of
# >= EXL3_INT8_HAD_QUANT_MIN rows (below that, 3 launches cost more than the saved memory traffic).
# EXL3_INT8_RECON_Q8=1 (default): decode the trellis straight to transposed int8 (int8_kernels.recon_q8T) instead of
# reconstruct fp16 + quant_wT. Integer K and the mul1 codebook (0x83DCD12D) only; anything else takes the old path.
_RECON_Q8 = os.environ.get("EXL3_INT8_RECON_Q8", "1") == "1"


def _recon_q8_ok(lin) -> bool:
    ok = getattr(lin, "_llmt_rq8", None)
    if ok is None:
        mul = getattr(lin, "mul1_tensor", None)
        ok = (isinstance(lin.K, int) and getattr(lin, "frac", None) is None and not lin.mcg and mul is not None
              and int(mul.item()) & 0xFFFFFFFF == 0x83DCD12D and lin.trellis.dtype == torch.int16
              and lin.trellis.is_contiguous())
        lin._llmt_rq8 = ok
    return _RECON_Q8 and ok


_HAD_QUANT = os.environ.get("EXL3_INT8_HAD_QUANT", "1") == "1"
_HAD_QUANT_MIN = int(os.environ.get("EXL3_INT8_HAD_QUANT_MIN", "2048"))


# MTP-H capture: EXL3_CAPTURE_MTP_H=<dir> accumulates X^T X (fp32) of the real inputs of every
# mtp.* linear and the draft head (one Hessian per fused module: members share the input), for calibrated draft
# re-quantization of the draft linears. Run the capture engine with --enforce-eager (Python must see every
# call); saved every EXL3_CAPTURE_EVERY calls of a module and at 3 x that. Not for production.
_CAP_DIR = os.environ.get("EXL3_CAPTURE_MTP_H")
_CAP_EVERY = int(os.environ.get("EXL3_CAPTURE_EVERY", "2000"))
_CAPTURE: dict = {}
_CAP_H: dict = {}


def _capture_h(layer_id, x2):
    key = _CAPTURE[layer_id]
    st = _CAP_H.get(key)
    if st is None:
        st = _CAP_H[key] = [torch.zeros((x2.shape[1], x2.shape[1]), dtype=torch.float32, device=x2.device), 0, 0]
    xf = x2.float()
    st[0].addmm_(xf.t(), xf)
    st[1] += x2.shape[0]
    st[2] += 1
    if st[2] % _CAP_EVERY == 0:
        os.makedirs(_CAP_DIR, exist_ok=True)
        torch.save({"H": st[0].cpu(), "count": st[1], "calls": st[2]}, os.path.join(_CAP_DIR, key + ".pt"))
        logger.info("MTP-H: saved %s (%d rows, %d calls)", key, st[1], st[2])


@torch.library.custom_op("exl3cuda::linear", mutates_args=())
def _linear(x: torch.Tensor, layer_id: int, n: int) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if _CAP_DIR and layer_id in _CAPTURE and not torch.cuda.is_current_stream_capturing():
        _capture_h(layer_id, x2)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if x2.dtype != torch.half:
        x2 = x2.half()
    if INT8_PREFILL and x2.shape[0] >= INT8_MIN_ROWS:
        segs = _SEGS[layer_id]
        if len(segs) == 1:
            return _int8_linear(segs[0], x2).to(x.dtype).reshape(*shape[:-1], n)
        y = torch.empty((x2.shape[0], n), dtype=torch.half, device=x2.device)
        off = 0
        for lin in segs:                                  # members write their own column range (no cat)
            _int8_linear(lin, x2, y[:, off: off + lin.out_features])
            off += lin.out_features
        return y.to(x.dtype).reshape(*shape[:-1], n)
    elif x2.shape[0] <= FUSE_MAX_M and layer_id in _FUSED:
        return _FUSED[layer_id](x2).to(x.dtype).reshape(*shape[:-1], n)
    else:
        ys = [lin.forward(x2, {}) for lin in _SEGS[layer_id]]
    y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)
    return y.to(x.dtype).reshape(*shape[:-1], n)


@_linear.register_fake
def _(x, layer_id, n):
    return x.new_empty((*x.shape[:-1], n))


# ------------------------------------------------------------------------------------------------
# Pruned-vocabulary draft head for MTP speculative decoding (same logic as the XPU plugin).
# The drafter proposes tokens from a subset of 128-token blocks (EXL3 lm_head columns can only be
# sliced by whole 128-column Hadamard blocks); the target still verifies with the full lm_head, so
# generated text is unchanged. Enabled by EXL3_DRAFT_VOCAB=<draft_vocab.json>.

_DRAFT_FAR_FROM = int(os.environ.get("EXL3_DRAFT_FAR_FROM", "2"))


def _spec_is_mtp():
    try:
        from vllm.config import get_current_vllm_config
        sc = get_current_vllm_config().speculative_config
    except Exception:
        return True
    return sc is not None and "mtp" in str(getattr(sc, "method", "")).lower()


def _draft_heads(layer, cbt):
    _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB"], cbt)
    # GRADED-DRAFT: a smaller draft vocab for the deeper draft positions
    # (EXL3_DRAFT_VOCAB_FAR, from position EXL3_DRAFT_FAR_FROM, 0-based): those passes read fewer head blocks.
    far = os.environ.get("EXL3_DRAFT_VOCAB_FAR")
    if far and os.path.realpath(far) == os.path.realpath(os.environ["EXL3_DRAFT_VOCAB"]):
        layer.exl3_draft_far = layer.exl3_draft      # same vocab: share one copy (~250 MB)
        logger.info("exl3: far draft vocab = near vocab: one head copy")
    elif far:
        _build_draft_head(layer, far, cbt, attr="exl3_draft_far")


def _build_draft_head(layer, path, cbt, attr="exl3_draft"):
    dw = _draft_weights()
    if dw is not None and "draft_head.trellis" in dw:
        # EXL3_DRAFT_WEIGHTS: the head's rows were re-quantized on their own (lower bitrate); near and far share it
        dev = layer.trellis.device
        cached = getattr(layer, "_exl3_dw_head", None)
        if cached is None:
            bs = int(dw["draft_head.block_size"])
            blocks = dw["draft_head.blocks"].long()
            idx = (blocks[:, None] * bs + torch.arange(bs)[None, :]).flatten()
            idx = idx[idx < layer.svh.shape[0]].to(dev)
            tr = dw["draft_head.trellis"].to(dev)
            lid = _register([dict(tr=tr, suh=dw["draft_head.suh"].to(dev), svh=dw["draft_head.svh"].to(dev))],
                            layer.trellis.shape[0] * 16, cbt)
            cached = layer._exl3_dw_head = dict(id=lid, n=idx.numel(), idx=idx)
            logger.info("exl3: MTP draft head from EXL3_DRAFT_WEIGHTS: %d rows at %d bpw", idx.numel(), tr.shape[-1] // 16)
        setattr(layer, attr, cached)
        return
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    # block_size: tokens per vocab block. Must be a multiple of 128: EXL3 applies the output Hadamard over each 128-column
    # group, so a subset of a group's columns is not a valid head (tested: 16-token blocks -> 1.3 tok/step)
    bs = int(spec.get("block_size", 128))
    assert bs % 128 == 0, "draft vocab: block_size must be a multiple of 128 (EXL3 output Hadamard groups)"
    n_total_blocks = layer.svh.shape[0] // bs
    blocks = blocks[blocks < n_total_blocks]
    if (len(blocks) * bs) % 128:
        blocks = blocks[: len(blocks) - (len(blocks) * bs % 128) // bs]
    dev = layer.trellis.device
    tpb = bs // 16
    tiles = (blocks[:, None] * tpb + torch.arange(tpb)[None, :]).flatten().to(dev)
    trellis = layer.trellis.data.index_select(1, tiles).contiguous()
    svh = layer.svh.data[: n_total_blocks * bs].view(-1, bs).index_select(0, blocks.to(dev)).flatten().contiguous()
    idx = (blocks[:, None] * bs + torch.arange(bs)[None, :]).flatten().to(dev)
    lid = _register([dict(tr=trellis, suh=layer.suh.data[0].contiguous(), svh=svh)], layer.trellis.shape[0] * 16, cbt)
    if _CAP_DIR and attr == "exl3_draft":
        _CAPTURE[lid] = "draft_head"
    setattr(layer, attr, dict(id=lid, n=svh.shape[0], idx=idx))
    logger.info("exl3: MTP draft head (%s) uses %d of %d vocab blocks (%.1f%% of lm_head)",
                attr, len(blocks), n_total_blocks, 100.0 * len(blocks) / n_total_blocks)


_DW = None


def _draft_weights():
    """EXL3_DRAFT_WEIGHTS=<safetensors>: re-quantized MTP linears + draft head (CPU dict; user-supplied)."""
    global _DW
    path = os.environ.get("EXL3_DRAFT_WEIGHTS")
    if not path:
        return None
    if _DW is None:
        from safetensors.torch import load_file
        _DW = load_file(path)
        logger.info("exl3: EXL3_DRAFT_WEIGHTS %s (%d tensors)", path, len(_DW))
    return _DW


def _draft_weights_swap(prefix, groups, segs, dev) -> bool:
    """Replace the segments of an mtp.* linear with the draft package's tensors (one member per group)."""
    dw = _draft_weights()
    if dw is None or prefix is None:
        return False
    key = _norm_key(prefix)
    if not key.startswith("mtp."):
        return False
    base, leaf = key.rsplit(".", 1)
    members = FUSED.get(leaf, [leaf])
    names = [f"{base}.{m}" for m in members]
    if len(segs) != len(groups) or any(len(g) != 1 for g in groups) and len(groups) > 1:
        return False
    if len(groups) == 1 and len(members) > 1:
        return False                                   # fused in the checkpoint: no per-member draft tensors
    sel = [names[g[0]] for g in groups] if len(members) > 1 else names
    if not all(f"{n}.trellis" in dw for n in sel):
        logger.warning("exl3: EXL3_DRAFT_WEIGHTS has no tensors for %s; keeping the checkpoint's", key)
        return False
    for i, n in enumerate(sel):
        segs[i] = dict(tr=dw[f"{n}.trellis"].to(dev), suh=dw[f"{n}.suh"].to(dev), svh=dw[f"{n}.svh"].to(dev))
    logger.info("exl3: %s from EXL3_DRAFT_WEIGHTS (%d bpw)", key, segs[0]["tr"].shape[-1] // 16)
    return True


# ------------------------------------------------------------------------------------------------
# DRAFT-DYN + context bias: port of the XPU plugin's DRAFT-DYN (+4-7% tokens/step on real calls there). Per request and per draft head, up to E extra draft-head rows
# (the request's most frequent prompt tokens that are not in the static head; near passes E' = EXL3_DRAFT_DYN_NEAR, far
# passes E = EXL3_DRAFT_DYN) and a +b drafter-logit bias on every prompt token (EXL3_DRAFT_BIAS). Off unless
# EXL3_DRAFT_DYN > 0 or EXL3_DRAFT_BIAS != 0. Exact: compute_logits returns a full-vocab row and vLLM's draft sampler
# samples it and returns the probs it sampled from (see draftdyn.py); pads go to virtual columns past the lm_head width.
# CUDA specifics vs the XPU plugin: rows are int4 g64 on the host (whole lm_head, ~0.7 GB RAM), uploaded only for newly arrived
# requests into persistent per-request device slots (pinned, non_blocking, on a side stream that first waits for the
# main stream), dequantized on the device into a batch-ordered fp16 buffer [S, E, K]; the per-pass extra logits are one
# bmm (row b's own E rows, no B x B*E waste). No host sync: the sampler polls the update's event (cudaEventQuery) and
# drafts from the static head alone until it is done, or, for an update without uploads (reorder / a request left),
# makes the main stream wait for it on the GPU. compute_logits runs eagerly (outside the drafter's PIECEWISE graphs);
# every buffer is persistent and allocated at load (before vLLM sizes the KV cache).
# DYN2: the per-batch-change host work moved off the step path (8 streams on the 5090 were -13.6% with it).
# Requests get a persistent slot at admission (after _update_states); one worker thread counts the prompt tokens
# (incrementally), gathers the int4 rows into pinned staging and, on a side stream after a main-stream event, uploads +
# dequantizes them into Wslot[slot] and the slot's X/XB/SB rows. propose only polls the waiting rows' events, maps rows
# to slots (not-yet-uploaded rows -> the NULL slot = static head, exact) and, when the map changed, issues one pinned
# H2D of the map + index_select gathers into the batch-ordered tables. No host sync, no prompt scan, no pin_memory per
# step. Per-pass math unchanged (full_logits). See draftdyn.py "DYN2".

def _env_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return int(default)


_DYN_E = _env_int("EXL3_DRAFT_DYN", 256 if _sm120() else 0)           # reference build: 256 far / 128 near
_DYN_E_NEAR = _env_int("EXL3_DRAFT_DYN_NEAR", 128 if ("EXL3_DRAFT_DYN" not in os.environ and _sm120()) else _DYN_E)
try:
    _DYN_BETA = float(os.environ.get("EXL3_DRAFT_BIAS", "0") or 0)
except ValueError:
    _DYN_BETA = 0.0
_DYN_GS = _env_int("EXL3_DRAFT_DYN_GS", 64)              # int4 group size of the host rows
_DYN_ON = max(_DYN_E, _DYN_E_NEAR) > 0 or _DYN_BETA != 0
_DYN_BG = os.environ.get("EXL3_DRAFT_DYN_BG", "1") != "0"   # DYN2: upload jobs on a worker thread (0 = inline)
_DYN: dict = {"groups": [], "mgr": None, "hooked": False}
_DYN_HOST: dict = {}


@torch.no_grad()
def _dyn_host_table(layer, gs):
    """The whole lm_head as per-row int4 g{gs} rows on the host: (packed [V, K/8] int32, scales [V, G] fp16, zeros
    [V, G] int8), ~0.7 GB for the 27B. Built from the layer's own LinearEXL3 (identity rows through lm.forward, i.e.
    the fp16 weights the target uses; not the plugin op, whose INT8_PREFILL branch would int8 the weights), 512 K
    columns at a time (y [512, V] fp16, ~254 MB transient), quantized on the device in 32k-row chunks. Once per process."""
    from .draftdyn import quantize_cols_into
    lin = _SEGS[layer.exl3_id][0]
    V, K = lin.out_features, lin.in_features
    key = (V, K, gs, float(layer.svh.data[:4096].float().sum()))
    if key in _DYN_HOST:
        return _DYN_HOST[key]
    import numpy as np
    G = K // gs
    hq = np.empty((V, K // 8), dtype=np.int32)
    hs = np.empty((V, G), dtype=np.float16)
    hz = np.empty((V, G), dtype=np.int8)
    dev = lin.trellis.device
    CH = 512 if K % 512 == 0 else gs
    for a in range(0, K, CH):
        eye = torch.zeros(CH, K, dtype=torch.float16, device=dev)
        eye[torch.arange(CH, device=dev), torch.arange(a, a + CH, device=dev)] = 1
        y = lin.forward(eye, {})                          # [CH, V] = lm_head[:, a:a + CH]^T
        for r0 in range(0, V, 32768):
            r1 = min(V, r0 + 32768)
            quantize_cols_into(hq[r0:r1], hs[r0:r1], hz[r0:r1], y[:, r0:r1].t().contiguous(), a, gs)
        del y, eye
    torch.cuda.empty_cache()
    table = (hq, hs, hz)
    _DYN_HOST[key] = table
    logger.info("exl3: DRAFT-DYN host int4 lm_head: %d x %d, group %d (%.0f MB host)", V, K, gs,
                sum(x.nbytes for x in table) / 1e6)
    return table


def _build_draft_dyn(layer):
    """Per-head-dict DYN groups (near and far share one group when they share one head dict, as an
    EXL3_DRAFT_WEIGHTS head does: E = max of the two, near passes use the first E' columns, which are exactly the E'
    most frequent: rows are sorted by count then id) and the DYN2 manager with all buffers allocated at load (before
    vLLM sizes the KV cache)."""
    from .draftdyn import DynManager, static_pos_map
    try:
        from vllm.config import get_current_vllm_config
        S = int(get_current_vllm_config().scheduler_config.max_num_seqs)
    except Exception:  # noqa
        S = 16
    dev = layer.suh.device
    K = layer.suh.shape[-1]
    gs = _DYN_GS
    G = K // gs
    base = int(layer.exl3_n)                              # virtual pad columns start at the lm_head width
    host = _dyn_host_table(layer, gs) if max(_DYN_E, _DYN_E_NEAR) > 0 else None
    groups: dict = {}
    for attr, E in (("exl3_draft", _DYN_E_NEAR), ("exl3_draft_far", _DYN_E)):
        d = getattr(layer, attr, None)
        if d is None:
            continue
        g = groups.get(id(d))
        if g is None:
            g = groups[id(d)] = dict(d=d, Es={}, beta=_DYN_BETA, gs=gs, G=G, K=K, base=base)
        g["Es"][attr] = max(0, E)
    lim = base
    for g in groups.values():
        d = g["d"]
        E = max(g["Es"].values())
        ix = d["idx"].cpu()
        pos = static_pos_map(ix, max(base, int(ix.max()) + 1))
        lim = min(lim, pos.size)
        g.update(E=E, Ns=d["idx"].numel(), width=base + E, host=host if E > 0 else None, pos=pos)
    mgr = DynManager(list(groups.values()), S, dev, lim, bg=_DYN_BG, log=logger)
    mgr.DELAY_S = _env_int("EXL3_DRAFT_DYN_DELAY_US", 2000) / 1e6
    mb, pmb = mgr.alloc()
    for g in groups.values():
        g["mgr"] = mgr
        g["d"]["dyn"] = g
    _DYN["mgr"] = mgr
    _DYN["groups"] = list(groups.values())
    _patch_dyn_hook()
    logger.info("exl3: DRAFT-DYN2 on: near E=%d, far E=%d, bias %.2f, %d slots, %d head group(s), %s uploads "
                "(%.1f MB device, %.1f MB pinned)", _DYN_E_NEAR, _DYN_E, _DYN_BETA, S, len(groups),
                "background" if _DYN_BG else "inline", mb, pmb)


def _dyn_mgr(runner):
    """The manager of the lm_head the drafter actually uses (the target's, shared by vLLM)."""
    m = getattr(getattr(runner, "drafter", None), "model", None)
    if m is not None and not hasattr(m, "lm_head") and hasattr(m, "unwrap"):
        m = m.unwrap()
    lm = getattr(m, "lm_head", None)
    for attr in ("exl3_draft", "exl3_draft_far"):
        d = getattr(lm, attr, None) if lm is not None else None
        g = d.get("dyn") if isinstance(d, dict) else None
        if g is not None:
            return g["mgr"]
    return _DYN.get("mgr")


def _dyn_prompt(runner, rid):
    """A request's prompt tokens without a Python-list conversion: the input batch's numpy row (vLLM copied the prompt
    there in add_request) when the request is in the batch, else the cached state's list (converted by the worker)."""
    ib = runner.input_batch
    i = ib.req_id_to_index.get(rid)
    if i is not None:
        n = int(ib.num_prompt_tokens[i])
        return ib.token_ids_cpu[i, :n].copy()            # ~5 us memcpy at 22k tokens
    st = runner.requests.get(rid)
    p = getattr(st, "prompt_token_ids", None) if st is not None else None
    return list(p) if p is not None else None


def _dyn_admit(runner, so):
    """After _update_states: finished requests free their slots, new (and streaming-updated) requests get a slot and
    a background job. Host cost: nothing unless the step has new or finished requests."""
    mgr = _dyn_mgr(runner)
    if mgr is None or mgr.disabled:
        return
    fin = getattr(so, "finished_req_ids", None)
    new = getattr(so, "scheduled_new_reqs", None)
    if not fin and not new:
        return
    t0 = time.perf_counter_ns()
    for rid in fin or ():
        mgr.drop(rid)
    if new:
        batch = runner.input_batch.req_id_to_index         # membership only (evictions skip batch rows)
        for nr in new:
            rid = getattr(nr, "req_id", None)
            if rid is not None:
                mgr.admit(rid, _dyn_prompt(runner, rid), batch)
    if len(mgr.req) > 4 * mgr.S:                          # safety sweep (requests that left without a finish event)
        for rid in [r for r in mgr.req if r not in runner.requests]:
            mgr.drop(rid)
    st = mgr.st
    dt = time.perf_counter_ns() - t0
    st["adm_ns"] = st.get("adm_ns", 0) + dt
    st["adm"] = st.get("adm", 0) + 1
    st["adm_max"] = max(st.get("adm_max", 0), dt)


def _dyn_step(runner):
    """Start of every propose_draft_token_ids: per-row activation + the row -> slot gather when the map changed."""
    mgr = _dyn_mgr(runner)
    if mgr is None or mgr.disabled or torch.cuda.is_current_stream_capturing():
        return
    t0 = time.perf_counter_ns()
    ib = runner.input_batch
    rids = ib.req_ids
    n = len(ib.req_id_to_index)
    if len(rids) != n:
        rids = rids[:n]
    chg0 = mgr.st["chg"]
    mgr.step(rids, lambda rid: _dyn_prompt(runner, rid))
    dt = time.perf_counter_ns() - t0
    st = mgr.st
    st["calls"] += 1
    st["hook_ns"] += dt
    st["hook_max"] = max(st["hook_max"], dt)
    if st["chg"] != chg0:
        st["chg_ns"] += dt
        st["chg_max"] = max(st["chg_max"], dt)


def _dyn_count(mgr, kind):
    st = mgr.st
    st[kind] += 1
    st["n"] += 1
    if st["n"] % 5000 == 0:
        c = max(1, st["chg"])
        logger.info("exl3: DRAFT-DYN2 last 5000 draft passes: %d with DYN, %d static; %d batch changes (host %.1f us "
                    "mean, %.1f max), %d regathers (%d rows), hook %.1f us mean over %d calls; %d uploads (%d activated, "
                    "%d stale, %d failed, worker %.2f ms mean), admission %.1f us mean / %.1f max; %d rows waiting",
                    st["dyn"], st["skip"], st["chg"], st["chg_ns"] / c / 1e3, st["chg_max"] / 1e3, st["gat"],
                    st["grow"], st["hook_ns"] / max(1, st["calls"]) / 1e3, st["calls"], st["upl"], st["act"],
                    st["stale"], st["fail"], st["job_ns"] / max(1, st["jobs"]) / 1e6,
                    st.get("adm_ns", 0) / max(1, st.get("adm", 0)) / 1e3, st.get("adm_max", 0) / 1e3, mgr.waiting)
        for k in list(st):
            st[k] = 0


def _dyn_draft_row(d, attr, sub, h, width, vocab):
    """compute_logits hook: the full-vocab draft row with each row's extra rows + bias (rows whose upload is not done
    map to the NULL slot = the static head), or None (static head for the whole pass)."""
    g = d.get("dyn")
    if g is None:
        return None
    mgr = g["mgr"]
    B = h.shape[0]
    if not mgr.any or mgr.B != B or torch.cuda.is_current_stream_capturing():
        _dyn_count(mgr, "skip")
        return None
    Ep = g["Es"].get(attr, g["Es"].get("exl3_draft", 0))
    SB = g["SB"][:B] if g["beta"] else None
    if Ep <= 0 and SB is None:
        return None
    _dyn_count(mgr, "dyn")
    from .draftdyn import full_logits
    inv, n_out = d.get("tail_inv"), d.get("tail_n_out")     # TAIL-FUSE's inverse block map (None = unfused path)
    if Ep <= 0:
        return full_logits(sub, d["idx"], width, vocab, SB=SB, inv=inv, n_out=n_out)
    return full_logits(sub, d["idx"], g["width"], vocab, h=h, W=g["Wb"][:B, :Ep], Xs=g["Xs"][:B, :Ep],
                       XB=g["XB"][:B, :Ep], SB=SB, inv=inv, n_out=n_out)


def _dyn_fail(where, e):
    logger.warning("exl3: DRAFT-DYN2 %s failed (%r); static heads only from now on", where, e)
    mgr = _DYN.get("mgr")
    if mgr is not None:
        mgr.reset()


def _patch_dyn_hook():
    if _DYN["hooked"]:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as R
    orig = R.propose_draft_token_ids

    def propose_draft_token_ids(self, *a, **k):
        try:
            _dyn_step(self)
        except Exception as e:  # noqa
            _dyn_fail("batch update", e)
            R.propose_draft_token_ids = orig
        return orig(self, *a, **k)

    R.propose_draft_token_ids = propose_draft_token_ids
    orig_us = getattr(R, "_update_states", None)
    if orig_us is not None:
        def _update_states(self, scheduler_output, *a, **k):
            out = orig_us(self, scheduler_output, *a, **k)
            try:
                _dyn_admit(self, scheduler_output)
            except Exception as e:  # noqa
                _dyn_fail("admission", e)
                R._update_states = orig_us
            return out

        R._update_states = _update_states
    _DYN["hooked"] = True
    logger.info("exl3: DRAFT-DYN2 batch hook installed (DYN %d, near %d, bias %s, admission hook %s)", _DYN_E,
                _DYN_E_NEAR, _DYN_BETA, "on" if orig_us is not None else "off (first-propose fallback)")


def _patch_mtp_draft_logits():
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:
        return
    cls = qwen3_5_mtp.Qwen3_5MTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return orig(self, hidden_states, spec_step_idx)
        far = getattr(lm, "exl3_draft_far", None)
        attr = "exl3_draft"
        if far is not None:
            k = getattr(self, "_exl3_pass", 0)
            self._exl3_pass = k + 1
            if k >= _DRAFT_FAR_FROM:
                d = far
                attr = "exl3_draft_far"
        sub = torch.ops.exl3cuda.linear(hidden_states, d["id"], d["n"])
        if _DYN_ON:                                       # DYN2: None = this pass drafts from the static head
            r = _dyn_draft_row(d, attr, sub, hidden_states, lm.exl3_n, self.config.vocab_size)
            if r is not None:
                return r
        logits = hidden_states.new_full((hidden_states.shape[0], lm.exl3_n), float("-inf"))
        logits.index_copy_(1, d["idx"], sub)
        return logits[:, : self.config.vocab_size]

    cls.compute_logits = compute_logits
    cls._exl3_patched = True
    # DRAFT-WINDOW: EXL3_DRAFT_WINDOW=W gives the MTP layer's attention a sliding window of the
    # last W tokens at decode (the impl's window; KV layout unchanged). Drafts only: the target verifies every token, so
    # outputs stay exact; the MTP layer also gets the target's hidden state, which carries the long context. Saves the
    # draft passes' full-context KV reads (~60 us per pass at 20k tokens on the 5090).
    _dwin = int(os.environ.get("EXL3_DRAFT_WINDOW", "0"))
    if _dwin > 0:
        _orig_lw = cls.load_weights

        def load_weights(self, *args, **kwargs):
            r = _orig_lw(self, *args, **kwargs)
            n = 0
            for mod in self.modules():
                impl = getattr(mod, "impl", None)
                if impl is not None and hasattr(impl, "sliding_window"):
                    impl.sliding_window = (_dwin - 1, 0)
                    n += 1
            logger.info("exl3: DRAFT-WINDOW: %d MTP attention layer(s) limited to the last %d tokens", n, _dwin)
            return r

        cls.load_weights = load_weights
    if os.environ.get("EXL3_DRAFT_VOCAB_FAR"):
        try:
            from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as _P
            _orig_propose = _P.propose

            def propose(self, *args, **kwargs):
                try:
                    self.model._exl3_pass = 0
                except Exception:
                    pass
                return _orig_propose(self, *args, **kwargs)

            _P.propose = propose
            logger.info("exl3: graded draft vocab: far head from draft position %d", _DRAFT_FAR_FROM)
        except Exception as e:  # noqa
            logger.warning("exl3: graded draft vocab not installed (%s)", e)


# ------------------------------------------------------------------------------------------------
# KSMOOTH (same code as the XPU plugin): K smoothing for int8_per_token_head KV (pure torch, no kernels).

def _patch_ksmooth():
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        from vllm.v1.kv_cache_interface import KVQuantMode
    except Exception as e:  # noqa
        logger.warning("exl3: K smoothing not installed (%s)", e)
        return
    if getattr(TritonAttentionImpl, "_exl3_ks_patched", False):
        return
    # K smoothing (SageAttention): store K - m (per layer / kv head / channel constant). Q.(K - m) = Q.K - Q.m and Q.m is
    # the same for every key of a query, so softmax is unchanged; m removes the common channel offsets / outliers that
    # eat the per-token int8 range (attention sinks). EXL3_KSMOOTH=<file> applies, EXL3_KSMOOTH_CALIB=<file> measures
    # (eager engine only: accumulates key sums of real tokens and saves {layer: mean[H, D]}).
    ks_path, cal_path = os.environ.get("EXL3_KSMOOTH"), os.environ.get("EXL3_KSMOOTH_CALIB")
    if ks_path or cal_path:
        orig_upd = TritonAttentionImpl.do_kv_cache_update
        kmeans = {}
        if ks_path:
            try:
                kmeans = {k: v for k, v in torch.load(ks_path, map_location="cpu").items()}
                logger.info("exl3: K smoothing on (%d layers from %s)", len(kmeans), ks_path)
            except Exception as e:  # noqa
                logger.warning("exl3: K smoothing file %s not loaded (%s)", ks_path, e)
        cal = {"n": 0}

        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            name = getattr(layer, "layer_name", None)
            qm = getattr(self, "_kv_quant_mode", None)
            if qm == KVQuantMode.INT4_PER_TOKEN_HEAD and name is not None and name in kmeans \
                    and os.environ.get("EXL3_KSMOOTH_INT4", "1") == "1":
                # KS4: K smoothing for int4 too (the XPU int4 KV write subtracts the same table before the
                # rotation; 5090 int4 teacher-forced KL without it: 0.0049-0.0051 vs 0.0038 on the XPU with it)
                m = getattr(self, "_exl3_km", None)
                if m is None or m.device != key.device:
                    m = self._exl3_km = kmeans[name].to(device=key.device, dtype=key.dtype)
                    if not getattr(TritonAttentionImpl, "_exl3_ks4_logged", False):
                        TritonAttentionImpl._exl3_ks4_logged = True
                        logger.info("exl3: KS4: K smoothing applied to int4 KV")
                key = key - m
            elif (qm == KVQuantMode.INT8_PER_TOKEN_HEAD and name is not None):
                if cal_path:
                    valid = slot_mapping[: key.shape[0]] >= 0
                    ks = key[: valid.shape[0]][valid].float()
                    if ks.shape[0] > 8 and os.environ.get("EXL3_KSMOOTH_ROBUST", "1") == "1":
                        # robust: drop token-heads whose key norm exceeds 3x the median (attention sinks / massive
                        # activations would otherwise dominate the mean and offset every normal key)
                        nrm = ks.norm(dim=-1)                                  # [T, H]
                        keep = nrm <= 3.0 * nrm.median(dim=0, keepdim=True).values
                        ks = ks * keep.unsqueeze(-1)
                        cnt = keep.sum(0).double().cpu()                      # [H]
                        acc = cal.setdefault(name, [torch.zeros(key.shape[1:], dtype=torch.float64), torch.zeros(key.shape[1], dtype=torch.float64)])
                        acc[0] += ks.sum(0).double().cpu(); acc[1] += cnt
                        cal["n"] += 1
                        if cal["n"] % 64 == 0:
                            torch.save({k: (v[0] / v[1].clamp_min(1).unsqueeze(-1)).float() for k, v in cal.items() if k != "n"}, cal_path)
                        ks = ks[:0]
                    if ks.numel():
                        acc = cal.setdefault(name, [torch.zeros(key.shape[1:], dtype=torch.float64), 0])
                        acc[0] += ks.sum(0).double().cpu(); acc[1] += ks.shape[0]
                        cal["n"] += 1
                        if cal["n"] % 64 == 0:
                            torch.save({k: (v[0] / max(1, v[1])).float() for k, v in cal.items() if k != "n"}, cal_path)
                if name in kmeans:
                    m = getattr(self, "_exl3_km", None)
                    if m is None or m.device != key.device:
                        m = self._exl3_km = kmeans[name].to(device=key.device, dtype=key.dtype)
                    key = key - m
            return orig_upd(self, layer, key, value, kv_cache, slot_mapping)

        TritonAttentionImpl.do_kv_cache_update = do_kv_cache_update

        TritonAttentionImpl._exl3_ks_patched = True


# ------------------------------------------------------------------------------------------------
# INT8KV-FA (CUDA port of the XPU plugin's block-dequantized FA2 prefill): with int8_per_token_head KV,
# vLLM's Triton unified attention runs prefill chunks at ~35-40 TFLOPS on the 5090 (30% of prefill time in the
# profile). Requests with more than 8 query rows instead read their whole context from the cache (int8 rows
# [256 | f32 scale], same bytes the Triton kernel reads), dequantize it to fp16 in blocks of EXL3_INT8KV_KB tokens and
# run vllm_flash_attn's FA2 varlen (head 256, ~200 TFLOPS), merging blocks by log-sum-exp. Rows with <= 8 query
# tokens (decode / spec verify) in the same batch go to the Triton kernel as a sub-batch; pure decode batches (and
# the FULL CUDA-graph decode path) are untouched. EXL3_INT8KV_FA=0 disables.
_FA_KB = int(os.environ.get("EXL3_INT8KV_KB", "32768"))


def _i8_gather_dequant(cache, pages, bs, start, end):
    """cache: per-token-head view [blocks, BS, H, 260] int8 (data 0..255, f32 scale at 256..259) -> fp16 [n, H, 256]."""
    p0, p1 = start // bs, (end + bs - 1) // bs
    blk = cache.index_select(0, pages[p0:p1]).flatten(0, 1)
    off = start - p0 * bs
    x = blk[off: off + (end - start)]
    sc = x[..., 256:260].contiguous().view(torch.float32)
    return (x[..., :256].to(torch.float32) * sc).to(torch.float16)


_FA_CU: dict = {}   # (n, device) -> int32 [0, n] on the device (FA-CU cache)


def _fa_cu(n: int, device):
    """cu_seqlens [0, n] on the device, cached. torch.tensor([...], device=...) copies from pageable host memory, which
    is a synchronous copy (the CPU waits until the stream has drained): per attention layer per prefill step it was
    5.3% of the EngineCore's samples at 4 streams (py-spy)."""
    t = _FA_CU.get((n, device))
    if t is None:
        if len(_FA_CU) > 8192:
            _FA_CU.clear()
        t = _FA_CU[(n, device)] = torch.tensor([0, n], dtype=torch.int32, device=device)
    return t


# FA16 (kernels: csrc/fa16): EXL3_FA16=b|f routes the int8-KV prefill FA2 calls to a pruned
# FA2 build (hdim 256 fp16, 64x64 tile) with fp16-accumulate MMAs: b = P.V fp16 per 64-key block into the fp32 acc_o,
# f = b + Q.K fp16 per 64 dims flushed to fp32. Numerics change (not bitwise); unset = stock FA2.
# Module fa_<v> is imported from EXL3_FA16_DIR/fa_<v>/ (default <package>/lib, where csrc/fa16/build.py puts it), then sys.path.
_FA16_V = _hwenv("EXL3_FA16", "f", "").strip().lower()
_FA16 = None
if _FA16_V in ("b", "f"):
    try:
        import importlib, sys as _sys
        _fa16_dir = os.environ.get("EXL3_FA16_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
        _sys.path.insert(0, os.path.join(_fa16_dir, f"fa_{_FA16_V}"))
        _FA16 = importlib.import_module(f"fa_{_FA16_V}")
        logger.info("exl3: FA16 on (variant %s)", _FA16_V)
    except Exception as e:  # noqa
        logger.warning("exl3: FA16 %s unavailable (%s); stock FA2", _FA16_V, e)
        _FA16 = None


def _fa(q, k, v, causal, scale):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    Q, L = q.shape[0], k.shape[0]
    cq = _fa_cu(Q, q.device)
    ck = _fa_cu(L, q.device)
    if _FA16 is not None and q.dtype == torch.float16 and q.shape[-1] == 256 and _device_ok(q.device):
        return tuple(_FA16.fwd(q, k, v, cq, ck, Q, L, scale, causal))
    return flash_attn_varlen_func(q, k, v, Q, cq, L, cu_seqlens_k=ck, causal=causal, softmax_scale=scale,
                                  return_softmax_lse=True)


# I8FA (kernels: csrc/i8fa): EXL3_I8FA=1 = fused CUDA gather-dequant + LSE merge around the same FA2
# calls (bitwise equal); module exl3cuda/i8fa_prefill.py, kernels in EXL3_I8FA_DIR. Unset / failure = the code below.
try:
    from exl3cuda import i8fa_prefill as _i8fa_mod
    _I8FA = _i8fa_mod if _i8fa_mod.setup(_FA16_V if _FA16 is not None else "") else None
except Exception as _e:  # noqa
    _I8FA = None
    if _hwenv("EXL3_I8FA", "1", "0") == "1":
        logger.warning("exl3: I8FA module not loaded (%s)", _e)


def _i8_prefill_fa(q, key_cache, value_cache, pages, seq_len, scale, out):
    if _I8FA is not None and _device_ok(q.device) and _I8FA.prefill(q, key_cache, value_cache, pages, seq_len, scale, out, _fa, _FA_KB) is not None:
        return out
    Q, bs = q.shape[0], key_cache.shape[1]
    prefix = seq_len - Q
    parts = []
    for s0 in range(0, prefix, _FA_KB):
        e0 = min(s0 + _FA_KB, prefix)
        parts.append(_fa(q, _i8_gather_dequant(key_cache, pages, bs, s0, e0),
                         _i8_gather_dequant(value_cache, pages, bs, s0, e0), False, scale))
    parts.append(_fa(q, _i8_gather_dequant(key_cache, pages, bs, prefix, seq_len),
                     _i8_gather_dequant(value_cache, pages, bs, prefix, seq_len), True, scale))
    if len(parts) == 1:
        out.copy_(parts[0][0])
        return out
    lses = torch.stack([pp[1] for pp in parts])            # [n, H, Q]
    m = lses.amax(0)
    w = torch.exp(lses - m)                                 # [n, H, Q]
    num = sum(pp[0].float() * wi.transpose(0, 1).unsqueeze(-1) for pp, wi in zip(parts, w))
    out.copy_((num / w.sum(0).transpose(0, 1).unsqueeze(-1)).to(out.dtype))
    return out


# I4FA (int4 KV, same strategy as the XPU plugin: own prefill path; stock unified_attention_int4 prefill is
# slow and its fp32 single_rht temporaries OOM'd at 8 x 4000): int4_per_token_head prefill = gather-dequant of the
# paged int4 cache into fp16 K / V in vLLM's rotated space (n - zp) * s, q rotated by fp16 matmul with
# diag(signs) H / 256 (entries +-1/256: exact products, fp32 accumulation), the same FA2 / FA16 calls and LSE merge as
# the int8 path, output un-rotated with H diag(signs) / 256. EXL3_I4FA=1 (default) on.
_I4FA = os.environ.get("EXL3_I4FA", "1") == "1"
_I4ROT: dict = {}


def _i4_rot(device):
    r = _I4ROT.get(device)
    if r is None:
        from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
        sg = _get_rht_signs(256, 0, device).float()
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < 256:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        fwd = (sg[:, None] * H / 256).half()          # q_r = q @ fwd  (= H(D q) / 256 as a row vector product)
        inv = (H * sg[None, :] / 256).half()          # o = o_r @ inv  (= D H o_r / 256)
        r = _I4ROT[device] = (fwd.contiguous(), inv.contiguous(), sg.contiguous())
    return r


def _i4_gather_dequant(cache, pages, bs, start, end):
    """cache: int4 per-token-head view [blocks, BS, H, 132] uint8 (128 B nibbles, f32 scale with zp in the low 4 bits)
    -> fp16 [n, H, 256] in the rotated space."""
    p0 = start // bs
    p1 = (end + bs - 1) // bs
    blk = cache.index_select(0, pages[p0:p1]).flatten(0, 1)
    off = start - p0 * bs
    x = blk[off: off + (end - start)]
    w = x[..., 128:132].contiguous().view(torch.int32)               # [n, H, 1]
    zp = (w & 15).to(torch.float32)
    sc = (w & -16).view(torch.float32)
    d = x[..., :128]
    n4 = torch.stack([d & 15, d >> 4], -1).flatten(-2).to(torch.float32)
    return ((n4 - zp) * sc).to(torch.float16)


def _i4_prefill_fa(q, key_cache, value_cache, pages, seq_len, scale, out):
    fwd, inv, _ = _i4_rot(q.device)
    Q, H, bs = q.shape[0], q.shape[1], key_cache.shape[1]
    qr = (q.reshape(Q * H, 256) @ fwd).view(Q, H, 256)
    prefix = seq_len - Q
    parts = []
    for s0 in range(0, prefix, _FA_KB):
        e0 = min(s0 + _FA_KB, prefix)
        parts.append(_fa(qr, _i4_gather_dequant(key_cache, pages, bs, s0, e0),
                         _i4_gather_dequant(value_cache, pages, bs, s0, e0), False, scale))
    parts.append(_fa(qr, _i4_gather_dequant(key_cache, pages, bs, prefix, seq_len),
                     _i4_gather_dequant(value_cache, pages, bs, prefix, seq_len), True, scale))
    if len(parts) == 1:
        o_r = parts[0][0]
    else:
        lses = torch.stack([pp[1] for pp in parts])            # [n, H, Q]
        m = lses.amax(0)
        w = torch.exp(lses - m)
        num = sum(pp[0].float() * wi.transpose(0, 1).unsqueeze(-1) for pp, wi in zip(parts, w))
        o_r = (num / w.sum(0).transpose(0, 1).unsqueeze(-1)).to(torch.float16)
    out.copy_((o_r.reshape(Q * H, 256) @ inv).view(Q, H, 256))
    return out

_MIX3D = os.environ.get("EXL3_MIX3D", "1") == "1"
_MIX3D_LOGGED: list = []


def _mix3d(md):
    """MIX-3D: the decode / verify rows of a mixed step use the split-softmax 3D kernel like pure decode
    steps do (rf profile: without these buffers the 2D grid is ~36 programs, 3.5 ms per layer at 22k contexts).
    unified_attention falls back to 2D by itself when rows > EXL3_ATTN_3D_MAXQ, the buffers are too small or there are
    more sequences than seq_threshold_3D. EXL3_MIX3D=0 = previous behaviour."""
    if not _MIX3D:
        return {}
    d = {k: getattr(md, k, None) for k in ("seq_threshold_3D", "num_par_softmax_segments", "softmax_segm_output",
                                           "softmax_segm_max", "softmax_segm_expsum")}
    if not _MIX3D_LOGGED:
        _MIX3D_LOGGED.append(1)
        logger.info("exl3: MIX-3D on: mixed-step decode rows get the 3D buffers (%s)",
                    "present" if all(v is not None for v in d.values()) else "MISSING: 2D fallback")
    return d


def _patch_fastlaunch_eagle():
    """FAST-LAUNCH (fastlaunch.py, LLMT_FASTLAUNCH=1; default off) for vLLM's per-draft-step
    eagle_step_update_slot_mapping_and_metadata (same body, launch via fastlaunch.launch: bit-identical, ~7 launches
    per decode step)."""
    if os.environ.get("LLMT_FASTLAUNCH", "0") != "1":
        return
    try:
        import vllm.v1.spec_decode.utils as U
        import vllm.v1.spec_decode.llm_base_proposer as LBP
        from .fastlaunch import launch as _fl
        kern, PAD = U.eagle_step_slot_mapping_metadata_kernel, U.PADDING_SLOT_ID
    except Exception as e:  # noqa
        logger.warning("exl3: FAST-LAUNCH eagle step not installed (%s)", e)
        return

    def eagle_step_update_slot_mapping_and_metadata(positions_1d, block_table_tensor, seq_lens, block_size, max_model_len,
                                                    out_clamped_positions, out_slot_mapping, input_batch_size=None):
        batch_size = positions_1d.shape[0]
        if input_batch_size is None:
            input_batch_size = batch_size
        _fl(kern, (input_batch_size,), positions_1d, block_table_tensor, block_table_tensor.stride(0), seq_lens,
            out_clamped_positions, out_slot_mapping, block_size=block_size, max_model_len=max_model_len,
            n_blocks_per_req=block_table_tensor.shape[1], PAD_ID=PAD, batch_size=batch_size)

    U.eagle_step_update_slot_mapping_and_metadata = eagle_step_update_slot_mapping_and_metadata
    if hasattr(LBP, "eagle_step_update_slot_mapping_and_metadata"):
        LBP.eagle_step_update_slot_mapping_and_metadata = eagle_step_update_slot_mapping_and_metadata
    logger.info("exl3: FAST-LAUNCH on (eagle step update, draft_scatter, DYN-ROW extras: cached CompiledKernel launches)")


def _patch_int8kv_fa():
    if os.environ.get("EXL3_INT8KV_FA", "1") != "1":
        return
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        from vllm.v1.kv_cache_interface import KVQuantMode
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    except Exception as e:  # noqa
        logger.warning("exl3: int8 KV FA prefill not installed (%s)", e)
        return
    if getattr(TritonAttentionImpl, "_exl3_fa_patched", False):
        return
    orig = TritonAttentionImpl.forward
    # rows per request up to this count are decode / verify rows (FA prefill only above). 8 = k=7 verify; a 23-token
    # prompt-lookup draft verifies 1 + 23 rows: EXL3_VERIFY_Q=24, else its CUDA-graph capture would take the prefill path.
    VQ = int(os.environ.get("EXL3_VERIFY_Q", "8"))

    # VA (long-context decode): verify / decode attention (<= 8 query rows per sequence) on the custom CUDA kernel
    # csrc/va/va_kernel.cu (FlashDecoding split over the context, 42 CTA-segments per KV head and sequence = one wave on
    # 170 SMs; kernel benchmark 1.9x Triton at 64-131k, error = Triton's).
    # Pure torch.ops launch with fixed shapes: CUDA-graph safe (persistent scratch, padded sequences have seq_len 0).
    # EXL3_VA=1 on; EXL3_VA_SO = library; anything unexpected -> the original Triton path (logged once).
    _VA = _hwenv("EXL3_VA", "1", "0") == "1"
    _VA_WAVE = int(os.environ.get("EXL3_VA_WAVE", "42"))
    _VA_ST = {}

    def _va_ready(dev):
        if "ok" not in _VA_ST:
            try:
                if not _device_ok(dev):
                    raise RuntimeError(f"{dev} is not an sm_120 GPU (the VA / VA4 kernels are sm_120 builds)")
                torch.ops.load_library(os.environ.get("EXL3_VA_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "va5090.so")))
                n = _VA_WAVE * 4 * 48
                _VA_ST["po"] = torch.empty(n * 256, dtype=torch.float32, device=dev)
                _VA_ST["pm"] = torch.empty(n, dtype=torch.float32, device=dev)
                _VA_ST["pl"] = torch.empty(n, dtype=torch.float32, device=dev)
                _VA_ST["ok"] = True
                logger.info("exl3: VA on: custom verify attention (%d CTA-segments per KV head, scratch %.1f MiB)",
                            _VA_WAVE, n * 258 * 4 / 2**20)
            except Exception as e:  # noqa: BLE001
                _VA_ST["ok"] = False
                logger.warning("exl3: VA not available (%r): Triton verify attention", e)
        return _VA_ST["ok"]

    def _va_try(self, query, kv_cache, md, output):
        if (md.max_query_len > 8 or self.num_heads != 24 or self.num_kv_heads != 4 or self.head_size != 256
                or not getattr(self, "_is_per_token_head_quant", False)
                or self._kv_quant_mode != KVQuantMode.INT8_PER_TOKEN_HEAD
                or self.alibi_slopes is not None or self.sinks is not None
                or self.sliding_window not in (None, (-1, -1)) or (self.logits_soft_cap or 0) != 0
                or not getattr(md, "causal", True) or getattr(md, "mm_prefix_range_tensor", None) is not None
                or query.dtype != torch.float16):
            return False
        nseq = md.seq_lens.shape[0]
        if nseq > _VA_WAVE or not _va_ready(query.device):
            return False
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        kk = key_cache.data_ptr()
        lay = _VA_ST.setdefault("lay", {})
        ok = lay.get(kk)
        if ok is None:
            ks, vs = self._k_scale_cache, self._v_scale_cache
            ok = (key_cache.dtype == torch.int8 and key_cache.stride(3) == 1 and key_cache.stride(1) % 8 == 0
                  and value_cache.data_ptr() == kk + 260 and ks is not None and vs is not None
                  and ks.data_ptr() == kk + 256 and vs.data_ptr() == kk + 516
                  and key_cache.stride(0) % 8 == 0 and key_cache.stride(2) % 8 == 0)
            lay[kk] = ok
            if not ok:
                logger.warning("exl3: VA: unexpected KV layout (strides %s): Triton verify attention",
                               tuple(key_cache.stride()))
        if not ok:
            return False
        T = md.num_actual_tokens
        q = query[:T].view(-1, self.num_heads, 256)
        out = output[:T].view(-1, self.num_heads, 256)
        if not (q.stride(2) == 1 and out.stride(2) == 1 and q.stride(0) % 4 == 0 and q.stride(1) % 4 == 0):
            if not _VA_ST.get("warned_stride"):
                _VA_ST["warned_stride"] = True
                logger.warning("exl3: VA: query/output strides %s / %s not supported: Triton", tuple(q.stride()),
                               tuple(out.stride()))
            return False
        _VA_ST["n"] = _VA_ST.get("n", 0) + 1
        if _VA_ST["n"] in (1, 1000):
            logger.info("exl3: VA dispatch #%d (q strides %s, capturing %s)", _VA_ST["n"], tuple(q.stride()),
                        torch.cuda.is_current_stream_capturing())
        torch.ops.va5090.forward(q, key_cache, key_cache.stride(0), key_cache.stride(2), key_cache.stride(1),
                                 md.block_table, md.seq_lens,
                                 md.query_start_loc, out, _VA_ST["po"], _VA_ST["pm"], _VA_ST["pl"],
                                 max(1, _VA_WAVE // nseq), key_cache.shape[1], self.scale, md.max_query_len)
        return True

    # VA4 (goal: int8-KV short-context speed with much better long context; same strategies as the XPU plugin):
    # int4_per_token_head verify / decode attention (<= 8 query rows) on csrc/va4/va4c_kernel.cu: q pre-rotation +
    # int4 record [K 128 | ks | V 128 | vs] + un-rotating merge. Same split / scratch as VA. EXL3_VA4=1 (default with
    # EXL3_VA=1), EXL3_VA4_SO = library, EXL3_VA4_OP = its op namespace (va4c for the kernel in this package).
    _VA4 = _VA and os.environ.get("EXL3_VA4", "1") == "1"

    def _va4_try(self, query, kv_cache, md, output):
        if (md.max_query_len > 8 or self.num_heads != 24 or self.num_kv_heads != 4 or self.head_size != 256
                or self._kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD
                or self.alibi_slopes is not None or self.sinks is not None
                or self.sliding_window not in (None, (-1, -1)) or (self.logits_soft_cap or 0) != 0
                or not getattr(md, "causal", True) or getattr(md, "mm_prefix_range_tensor", None) is not None
                or query.dtype != torch.float16):
            return False
        nseq = md.seq_lens.shape[0]
        if nseq > _VA_WAVE or not _va_ready(query.device):
            return False
        if "va4" not in _VA_ST:
            try:
                torch.ops.load_library(os.environ.get("EXL3_VA4_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "va4c.so")))
                from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
                _VA_ST["signs"] = _get_rht_signs(256, 0, query.device).float().contiguous()
                _VA_ST["qr"] = torch.empty(_VA_WAVE * 8, 24, 256, dtype=torch.float16, device=query.device)
                _VA_ST["va4"] = True
                logger.info("exl3: VA4 on: int4 verify attention (pre-rotated q, un-rotating merge)")
            except Exception as e:  # noqa: BLE001
                _VA_ST["va4"] = False
                logger.warning("exl3: VA4 not available (%r): stock int4 Triton", e)
        if not _VA_ST["va4"]:
            return False
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        kk = key_cache.data_ptr()
        lay = _VA_ST.setdefault("lay4", {})
        ok = lay.get(kk)
        if ok is None:
            ks, vs = self._k_scale_cache, self._v_scale_cache
            ok = (key_cache.dtype == torch.uint8 and key_cache.stride(3) == 1 and key_cache.stride(1) % 8 == 0
                  and value_cache.data_ptr() == kk + 132 and ks is not None and vs is not None
                  and ks.data_ptr() == kk + 128 and vs.data_ptr() == kk + 260
                  and key_cache.stride(0) % 8 == 0 and key_cache.stride(2) % 8 == 0)
            lay[kk] = ok
            if not ok:
                logger.warning("exl3: VA4: unexpected KV layout (strides %s): stock int4 Triton", tuple(key_cache.stride()))
        if not ok:
            return False
        T = md.num_actual_tokens
        if T > _VA_ST["qr"].shape[0]:
            return False
        q = query[:T].view(-1, self.num_heads, 256)
        out = output[:T].view(-1, self.num_heads, 256)
        if not (q.stride(2) == 1 and out.stride(2) == 1):
            return False
        _VA_ST["n4"] = _VA_ST.get("n4", 0) + 1
        if _VA_ST["n4"] in (1, 1000):
            logger.info("exl3: VA4 dispatch #%d (q strides %s, capturing %s)", _VA_ST["n4"], tuple(q.stride()),
                        torch.cuda.is_current_stream_capturing())
        getattr(torch.ops, os.environ.get("EXL3_VA4_OP", "va4c")).forward(q, key_cache, key_cache.stride(0), key_cache.stride(2), key_cache.stride(1),
                              md.block_table, md.seq_lens, md.query_start_loc, out, _VA_ST["po"], _VA_ST["pm"],
                              _VA_ST["pl"], _VA_ST["signs"], _VA_ST["qr"], max(1, _VA_WAVE // nseq), key_cache.shape[1],
                              self.scale, md.max_query_len)
        return True

    def _i4_forward_mixed_old(self, query, kv_cache, md, output):
        """int4 prefill / mixed step: long rows through _i4_prefill_fa, the decode / verify rows next to the chunk
        through va4 on a subset (stock int4 Triton if va4 is unavailable)."""
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        T = md.num_actual_tokens
        nr = md.query_start_loc.shape[0] - 1
        H = self.num_heads
        q = query[:T].view(-1, H, 256)
        out = output[:T].view(-1, H, 256)
        qcpu = getattr(md, "query_start_loc_cpu", None)
        qsl = qcpu[: nr + 1].tolist() if qcpu is not None and qcpu.shape[0] >= nr + 1 else md.query_start_loc[: nr + 1].cpu().tolist()
        scpu = getattr(md, "seq_lens_cpu", None)
        sl = scpu[:nr].tolist() if scpu is not None and scpu.shape[0] >= nr else md.seq_lens[:nr].cpu().tolist()
        # VA4-PSEUDO: rows of a prefill with <= EXL3_VA4_PREFILL_Q (64) rows go through va4 as 8-row
        # pseudo-sequences (rows c..c+7 of a chunk ending at seq_len L see L - (b - e) tokens: exact causal mask),
        # together with the decode / verify rows, in launches of <= _VA_WAVE pseudo-sequences; longer prefills I4FA.
        pq = int(os.environ.get("EXL3_VA4_PREFILL_Q", "64"))
        use_va4 = _VA4 and _va_ready(q.device) and bool(_VA_ST.get("va4"))
        pseudo = []
        for i in range(nr):
            a, b = qsl[i], qsl[i + 1]
            n = b - a
            if n <= 0:
                continue
            if n <= VQ and use_va4:
                pseudo.append((i, a, b, sl[i]))
            elif n <= pq and use_va4:
                for c in range(a, b, 8):
                    e = min(c + 8, b)
                    pseudo.append((i, c, e, sl[i] - (b - e)))
            elif n > VQ:
                _i4_prefill_fa(q[a:b], key_cache, value_cache, md.block_table[i].to(torch.int32), sl[i],
                               float(self.scale), out[a:b])
            else:
                pseudo.append((i, a, b, sl[i]))          # va4 unavailable: stock path below
        if not pseudo:
            return output
        for g0 in range(0, len(pseudo), _VA_WAVE):
            grp = pseudo[g0:g0 + _VA_WAVE]
            rows = torch.cat([torch.arange(c, e) for _, c, e, _ in grp]).to(q.device)
            lens = [e - c for _, c, e, _ in grp]
            cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=q.device)
            idx = torch.tensor([i for i, _, _, _ in grp], device=q.device)
            sls = torch.tensor([L for _, _, _, L in grp], dtype=torch.int32, device=q.device)
            qs = q.index_select(0, rows).contiguous()
            os_ = torch.empty_like(qs)
            bt = md.block_table.index_select(0, idx)
            if use_va4:
                getattr(torch.ops, os.environ.get("EXL3_VA4_OP", "va4c")).forward(
                    qs, key_cache, key_cache.stride(0), key_cache.stride(2), key_cache.stride(1), bt, sls, cu, os_,
                    _VA_ST["po"], _VA_ST["pm"], _VA_ST["pl"], _VA_ST["signs"], _VA_ST["qr"],
                    max(1, _VA_WAVE // len(grp)), key_cache.shape[1], self.scale, max(lens))
            else:
                unified_attention(q=qs, k=key_cache, v=value_cache, out=os_, cu_seqlens_q=cu, max_seqlen_q=max(lens),
                                  seqused_k=sls, max_seqlen_k=max(L for _, _, _, L in grp),
                                  softmax_scale=self.scale, causal=True, alibi_slopes=None, window_size=(-1, -1),
                                  block_table=bt, softcap=self.logits_soft_cap,
                                  q_descale=None, k_descale=None, v_descale=None,
                                  kv_quant_mode=self._kv_quant_mode,
                                  k_scale_cache=self._k_scale_cache, v_scale_cache=self._v_scale_cache, **_mix3d(md))
            out.index_copy_(0, rows, os_)
        return output

    # I4-PLAN: the mixed-step row plan (FA prefill rows, va4 pseudo-sequences, their device index tensors) is the same
    # for every attention layer of a step, so it is built once per step on the first layer and reused by the others.
    # No host sync: the row layout comes from query_start_loc_cpu (exact on the host), decode / chunk seq lens are
    # gathered on the GPU (md.seq_lens[idx] - offset), and only long prefill rows (no pending drafts, so the CPU upper
    # bound is exact) use a host int. Index tensors go host->device from pinned memory, non_blocking. vLLM 0.26's
    # TritonAttentionMetadata has no query_start_loc_cpu / seq_lens_cpu, so the old path fell back to .cpu() on every
    # layer: 2 DtoH syncs + 4 pageable HtoD copies per layer (~96 per mixed step), which serialized host and GPU.
    # EXL3_I4PLAN=0 = old path; EXL3_I4PLAN_CHECK=N checks the host seq lens against the device for the first N plans
    # (one sync each).
    _I4PLAN = os.environ.get("EXL3_I4PLAN", "1") == "1"
    _I4PLAN_CHECK = [int(os.environ.get("EXL3_I4PLAN_CHECK", "0"))]
    _I4PQ = int(os.environ.get("EXL3_VA4_PREFILL_Q", "64"))

    def _pin(vals, dtype, dev):
        return torch.tensor(vals, dtype=dtype, pin_memory=True).to(dev, non_blocking=True)

    def _i4_plan(md, dev, use_va4):
        nr = md.query_start_loc.shape[0] - 1
        qcpu = getattr(md, "_exl3_qsl_cpu", None)
        qsl = qcpu[: nr + 1].tolist() if qcpu is not None and qcpu.shape[0] >= nr + 1 else md.query_start_loc[: nr + 1].cpu().tolist()
        ub = getattr(md, "_exl3_sl_ub", None)
        ubl = ub[:nr].tolist() if ub is not None and ub.shape[0] >= nr else md.seq_lens[:nr].cpu().tolist()
        if _I4PLAN_CHECK[0] > 0:
            _I4PLAN_CHECK[0] -= 1
            real = md.seq_lens[:nr].cpu().tolist()
            bad = [(i, ubl[i], real[i]) for i in range(nr) if qsl[i + 1] - qsl[i] > VQ and ubl[i] != real[i]]
            if bad:
                logger.warning("exl3: I4-PLAN host seq len mismatch on prefill rows %s: syncing for exact values", bad[:4])
                ubl = real
        fa, pseudo = [], []
        for i in range(nr):
            a, b = qsl[i], qsl[i + 1]
            n = b - a
            if n <= 0:
                continue
            if n <= VQ and use_va4:
                pseudo.append((i, a, b, 0))
            elif n <= _I4PQ and use_va4:
                for c in range(a, b, 8):
                    e = min(c + 8, b)
                    pseudo.append((i, c, e, b - e))
            elif n > VQ:
                fa.append((a, b, md.block_table[i].to(torch.int32), ubl[i]))
            else:
                pseudo.append((i, a, b, 0))
        groups = []
        for g0 in range(0, len(pseudo), _VA_WAVE):
            grp = pseudo[g0:g0 + _VA_WAVE]
            rows = [r for _, c, e, _ in grp for r in range(c, e)]
            lens = [e - c for _, c, e, _ in grp]
            idx = _pin([i for i, _, _, _ in grp], torch.long, dev)
            offs = _pin([o for _, _, _, o in grp], md.seq_lens.dtype, dev)
            groups.append({"rows": _pin(rows, torch.long, dev), "n": len(grp), "maxlen": max(lens),
                           "cu": _pin([0] + list(itertools.accumulate(lens)), torch.int32, dev),
                           "sls": (md.seq_lens.index_select(0, idx) - offs).to(torch.int32),
                           "bt": md.block_table.index_select(0, idx),
                           "maxL": max(ubl[i] - o for i, _, _, o in grp)})
        return {"fa": fa, "groups": groups}

    def _i4_forward_mixed(self, query, kv_cache, md, output):
        """int4 prefill / mixed step: long rows through _i4_prefill_fa, the decode / verify rows next to the chunk
        through va4 on a subset (stock int4 Triton if va4 is unavailable). Plan built once per step (I4-PLAN)."""
        if not _I4PLAN:
            return _i4_forward_mixed_old(self, query, kv_cache, md, output)
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        T = md.num_actual_tokens
        H = self.num_heads
        q = query[:T].view(-1, H, 256)
        out = output[:T].view(-1, H, 256)
        use_va4 = _VA4 and _va_ready(q.device) and bool(_VA_ST.get("va4"))
        plan = getattr(md, "_exl3_i4plan", None)
        if plan is None or plan.get("va4") != use_va4:
            plan = _i4_plan(md, q.device, use_va4); plan["va4"] = use_va4
            try:
                md._exl3_i4plan = plan
            except Exception:  # noqa
                pass
        for a, b, pages, L in plan["fa"]:
            _i4_prefill_fa(q[a:b], key_cache, value_cache, pages, L, float(self.scale), out[a:b])
        for g in plan["groups"]:
            qs = q.index_select(0, g["rows"]).contiguous()
            os_ = torch.empty_like(qs)
            if use_va4:
                getattr(torch.ops, os.environ.get("EXL3_VA4_OP", "va4c")).forward(
                    qs, key_cache, key_cache.stride(0), key_cache.stride(2), key_cache.stride(1), g["bt"], g["sls"],
                    g["cu"], os_, _VA_ST["po"], _VA_ST["pm"], _VA_ST["pl"], _VA_ST["signs"], _VA_ST["qr"],
                    max(1, _VA_WAVE // g["n"]), key_cache.shape[1], self.scale, g["maxlen"])
            else:
                unified_attention(q=qs, k=key_cache, v=value_cache, out=os_, cu_seqlens_q=g["cu"], max_seqlen_q=g["maxlen"],
                                  seqused_k=g["sls"], max_seqlen_k=g["maxL"],
                                  softmax_scale=self.scale, causal=True, alibi_slopes=None, window_size=(-1, -1),
                                  block_table=g["bt"], softcap=self.logits_soft_cap,
                                  q_descale=None, k_descale=None, v_descale=None,
                                  kv_quant_mode=self._kv_quant_mode,
                                  k_scale_cache=self._k_scale_cache, v_scale_cache=self._v_scale_cache, **_mix3d(md))
            out.index_copy_(0, g["rows"], os_)
        return output

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        md = attn_metadata
        if (_VA4 and md is not None and md.max_query_len <= VQ and output_scale is None and output_block_scale is None
                and getattr(self, "_kv_quant_mode", None) == KVQuantMode.INT4_PER_TOKEN_HEAD
                and _va4_try(self, query, kv_cache, md, output)):
            return output
        if (_I4FA and md is not None and md.max_query_len > VQ
                and getattr(self, "_kv_quant_mode", None) == KVQuantMode.INT4_PER_TOKEN_HEAD and self.head_size == 256
                and self.num_heads == 24 and self.num_kv_heads == 4
                and output_scale is None and output_block_scale is None and self.alibi_slopes is None
                and self.sinks is None and getattr(md, "mm_prefix_range_tensor", None) is None
                and self.sliding_window in (None, (-1, -1)) and getattr(md, "causal", True)):
            return _i4_forward_mixed(self, query, kv_cache, md, output)
        if (_VA and md is not None and md.max_query_len <= VQ and output_scale is None and output_block_scale is None
                and _va_try(self, query, kv_cache, md, output)):
            return output
        if (md is None or md.max_query_len <= VQ or not getattr(self, "_is_per_token_head_quant", False)
                or self._kv_quant_mode != KVQuantMode.INT8_PER_TOKEN_HEAD or self.head_size != 256
                or output_scale is not None or output_block_scale is not None or self.alibi_slopes is not None
                or self.sinks is not None or getattr(md, "mm_prefix_range_tensor", None) is not None
                or self.sliding_window not in (None, (-1, -1)) or not getattr(md, "causal", True)):
            return orig(self, layer, query, key, value, kv_cache, md, output, output_scale, output_block_scale)
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        T = md.num_actual_tokens
        nr = md.query_start_loc.shape[0] - 1
        H = self.num_heads
        q = query[:T].view(-1, H, 256)
        out = output[:T].view(-1, H, 256)
        # the metadata carries CPU copies (query_start_loc_cpu / seq_lens_cpu) on this vLLM; use them instead of a
        # device->host sync per mixed step (py-spy: 13-18% of the main thread's wait time; under async scheduling the
        # sync also stops the CPU from preparing the next step). Falls back to the sync when a copy is missing.
        qcpu = getattr(md, "query_start_loc_cpu", None)
        qsl = qcpu[: nr + 1].tolist() if qcpu is not None and qcpu.shape[0] >= nr + 1 else md.query_start_loc[: nr + 1].cpu().tolist()
        scpu = getattr(md, "seq_lens_cpu", None)
        sl = scpu[:nr].tolist() if scpu is not None and scpu.shape[0] >= nr else md.seq_lens[:nr].cpu().tolist()
        short = [i for i in range(nr) if 0 < qsl[i + 1] - qsl[i] <= VQ]
        for i in range(nr):
            a, b = qsl[i], qsl[i + 1]
            if b - a > VQ:
                _i8_prefill_fa(q[a:b], key_cache, value_cache, md.block_table[i].to(torch.int32), sl[i],
                               float(self.scale), out[a:b])
        if short:                                               # decode / verify rows next to the chunk
            idx = torch.tensor(short, device=q.device)
            rows = torch.cat([torch.arange(qsl[i], qsl[i + 1]) for i in short]).to(q.device)
            lens = [qsl[i + 1] - qsl[i] for i in short]
            cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=q.device)
            qs = q.index_select(0, rows).contiguous()
            os_ = torch.empty_like(qs)
            unified_attention(q=qs, k=key_cache, v=value_cache, out=os_, cu_seqlens_q=cu, max_seqlen_q=max(lens),
                              seqused_k=md.seq_lens.index_select(0, idx), max_seqlen_k=max(sl[i] for i in short),
                              softmax_scale=self.scale, causal=True, alibi_slopes=None, window_size=(-1, -1),
                              block_table=md.block_table.index_select(0, idx), softcap=self.logits_soft_cap,
                              q_descale=None, k_descale=None, v_descale=None,
                              kv_quant_mode=self._kv_quant_mode,
                              k_scale_cache=self._k_scale_cache, v_scale_cache=self._v_scale_cache, **_mix3d(md))
            out.index_copy_(0, rows, os_)
        return output

    TritonAttentionImpl.forward = forward
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
        _ob = TritonAttentionMetadataBuilder.build

        def _build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = _ob(self, common_prefix_len, common_attn_metadata, fast_build)
            try:
                md._exl3_qsl_cpu = common_attn_metadata.query_start_loc_cpu
                md._exl3_sl_ub = (common_attn_metadata._seq_lens_cpu if common_attn_metadata._seq_lens_cpu is not None
                                  else common_attn_metadata.seq_lens_cpu_upper_bound)
                md._exl3_i4plan = None
            except Exception:  # noqa
                pass
            return md
        TritonAttentionMetadataBuilder.build = _build
        logger.info("exl3: I4-PLAN %s (mixed-step attention plan once per step, no host sync)", "on" if _I4PLAN else "off")
    except Exception as e:  # noqa
        logger.warning("exl3: I4-PLAN builder hook not installed (%s)", e)
    TritonAttentionImpl._exl3_fa_patched = True
    logger.info("exl3: int8 KV prefill via block-dequantized FA2 (blocks of %d tokens)", _FA_KB)


# ------------------------------------------------------------------------------------------------
# GDN-RECOMPUTE, CUDA port of the XPU plugin's _patch_gdn_recompute / _rc_dispatch: MTP spec decoding
# for the 48 GDN layers with ONE committed SSM state per request
# (Triton kernel exl3cuda.gdn_rc.gdn_recompute), so MambaSpec.num_speculative_blocks = 0 and a running request no
# longer reserves 1 + k state pages per GDN group. The runner-side bookkeeping (pool slot / parity / pending count /
# decode-time boundary states) is in the patched vLLM gpu_model_runner.py (_llmt_gdn_rc), which follows GDN_RC["on"].
# EXL3_GDN_RECOMPUTE=1 / 0; unset = on with an sm_120 GPU. Off (warning) when either half cannot install.
GDN_RC = {"on": _hwenv("EXL3_GDN_RECOMPUTE", "1", "0") == "1", "S": int(os.environ.get("EXL3_GDN_RC_SLOTS", "16"))}


def _rc_alloc_all(static_forward_context, device):
    """Pending pools for every GDN layer, allocated at KV-cache init (before CUDA-graph capture: a pool zeroed inside
    a captured graph would be re-zeroed on every replay)."""
    T1 = GDN_RC.get("T1", 7); S = GDN_RC["S"]; n = 0
    for name in GDN_RC.get("layer_group", {}):
        layer = static_forward_context.get(name)
        if layer is None or hasattr(layer, "_exl3_pool"):
            continue
        H = layer.num_k_heads // layer.tp_size; HV = layer.num_v_heads // layer.tp_size
        K = layer.head_k_dim; V = layer.head_v_dim
        layer._exl3_pool = (torch.zeros(2, S, T1, H, K, device=device), torch.zeros(2, S, T1, HV, V, device=device),
                            torch.zeros(2, S, T1, HV, device=device), torch.zeros(2, S, T1, HV, device=device))
        n += 1
    logger.info("exl3: GDN recompute: pending pools for %d layers (T1=%d, %d slots)", n, T1, S)


GDN_RC["alloc"] = _rc_alloc_all


def _patch_gdn_recompute():
    if not GDN_RC["on"]:
        return
    try:                                              # needs the companion vLLM runner (GDN-RC bookkeeping)
        import importlib.util
        _o = importlib.util.find_spec("vllm.v1.worker.gpu_model_runner").origin
        with open(_o) as _f:
            _patched = "_llmt_gdn_rc" in _f.read()
    except Exception:  # noqa
        _patched = False
    if not _patched:
        logger.warning("exl3: GDN recompute needs the companion vLLM gpu_model_runner (_llmt_gdn_rc); not installed")
        GDN_RC["on"] = False
        return
    import dataclasses
    try:
        from vllm.model_executor.layers.mamba import abstract as mabs
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.model_executor.layers.mamba import mamba_utils as mmu
        from vllm.v1.worker import mamba_utils as wmu
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qg
    except Exception as e:  # noqa
        logger.warning("exl3: GDN recompute not installed (%s)", e)
        GDN_RC["on"] = False                          # the runner reads this flag (vLLM gpu_model_runner._gdn_rc_on)
        return
    if getattr(mabs.MambaBase, "_exl3_rc", False):
        return
    # 1. no speculative state blocks (as on the XPU)
    orig_spec = mabs.MambaBase.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig_spec(self, vllm_config)
        if (GDN_RC["on"] and os.environ.get("EXL3_GDN_RECOMPUTE") is None
                and getattr(getattr(vllm_config, "model_config", None), "quantization", None) != "exl3"):
            GDN_RC["on"] = False      # the default applies to EXL3 models only; the vLLM runner follows this flag
            logger.info("exl3: GDN recompute off (not an EXL3 model; EXL3_GDN_RECOMPUTE=1 forces it)")
        if GDN_RC["on"] and isinstance(spec, MambaSpec) and spec.num_speculative_blocks:
            GDN_RC["T1"] = spec.num_speculative_blocks + 1
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec
    mabs.MambaBase.get_kv_cache_spec = get_kv_cache_spec
    mabs.MambaBase._exl3_rc = True

    # 2. block-crossing SSM copy takes the committed state (no accept bias); same function object (identity checks)
    #    The new code runs with mamba_utils' globals: the switch and the original body are put there.
    import types as _types
    _t = mmu.get_temporal_copy_spec
    mmu._exl3_rc_state = GDN_RC
    mmu._exl3_temporal_orig = _types.FunctionType(_t.__code__, _t.__globals__, "_orig", _t.__defaults__, _t.__closure__)

    def _temporal(state, block_ids, cur_block_idx, num_accepted_tokens):
        if not _exl3_rc_state["on"]:  # noqa: F821 (mamba_utils global)
            return _exl3_temporal_orig(state, block_ids, cur_block_idx, num_accepted_tokens)  # noqa: F821
        src_state = state[block_ids[cur_block_idx]]
        return MambaCopySpec(start_addr=src_state.data_ptr(), num_elements=src_state.numel())
    _t.__code__ = _temporal.__code__

    # 3. fused postprocess: no SSM copies (the kernel stores decode-time boundary states itself)
    orig_init = wmu.MambaSpecDecodeGPUContext.initialize_from_forward_context

    def initialize_from_forward_context(self, *a, **kw):
        was = self.is_initialized
        orig_init(self, *a, **kw)
        if not was and GDN_RC["on"]:
            self.state_inner_sizes.masked_fill_(self.state_conv_widths == 0, 0)
            logger.info("exl3: GDN recompute: postprocess SSM copies disabled")
    wmu.MambaSpecDecodeGPUContext.initialize_from_forward_context = initialize_from_forward_context

    # 4. the CUDA layer calls self._forward_core inside the opaque custom op (runtime branches are fine there)
    cls = qg.QwenGatedDeltaNetAttention
    orig_core = cls._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        fc = get_forward_context()
        md_all = fc.attn_metadata
        md = md_all.get(self.prefix) if isinstance(md_all, dict) else None
        if (not GDN_RC["on"] or md is None or md.spec_sequence_masks is None or int(md.num_spec_decodes or 0) == 0
                or "buf" not in GDN_RC or not hasattr(self, "_exl3_pool")):
            return orig_core(self, mixed_qkv, b, a, core_attn_out)
        return _rc_forward(self, orig_core, md, md_all, mixed_qkv, b, a, core_attn_out)
    cls._forward_core = _forward_core
    logger.info("exl3: GDN recompute enabled (num_speculative_blocks = 0, %d pending slots)", GDN_RC["S"])


# RC-FUSED-IN: gdn_rc.gdn_recompute_fused does the q/k/v split, gating and output write inside
# the recompute kernel (~12 fewer tiny launches per GDN layer); EXL3_RC_FUSED_IN=1.
_RC_FUSED_IN = _hwenv("EXL3_RC_FUSED_IN", "1", "0") == "1"
# RC-SLICE (on the XPU the gather/scatter cost 5.9% of a mixed 30k-token step):
# in mixed steps the non-spec (prefill) rows were gathered with index_select per GDN layer and the output scattered back
# with index_copy. GDN-RC already requires the spec rows to lead the batch (runner sets GDN_RC["lead"] per step), so
# the non-spec tokens are the contiguous range [num_spec_decode_tokens, num_actual_tokens): use views, and let the
# prefill path write straight into core_attn_out. Same rows, same order -> bit-identical. EXL3_RC_SLICE=1.
_RC_SLICE = _hwenv("EXL3_RC_SLICE", "1", "0") == "1"


def _rc_forward(self, orig_core, md, md_all, mixed_qkv, b, a, core_attn_out):
    import dataclasses
    import torch.nn.functional as F
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import is_conv_state_dim_first
    from .gdn_rc import gdn_recompute
    H = self.num_k_heads // self.tp_size; HV = self.num_v_heads // self.tp_size
    K, V = self.head_k_dim, self.head_v_dim
    KD = H * K
    ns = int(md.num_spec_decodes)
    nact = int(md.num_actual_tokens)
    kvc = self.kv_cache
    conv_state = kvc[0] if is_conv_state_dim_first() else kvc[0].transpose(-1, -2)
    ssm_state = kvc[1]
    conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    pure = int(md.num_prefills) == 0 and int(md.num_decodes) == 0
    mixed_qkv, b, a = mixed_qkv[:nact], b[:nact], a[:nact]
    ts = int(md.num_spec_decode_tokens or 0)
    sliced = _RC_SLICE and not pure and GDN_RC.get("lead", False) and ts > 0
    if not pure:
        # prefill / non-spec rows: the original path on the non-spec subset with a spec-free metadata copy
        nsi = md.non_spec_token_indx
        nT = nact - ts if sliced else int(nsi.numel())
        md2 = dataclasses.replace(md, spec_sequence_masks=None, num_spec_decodes=0, num_spec_decode_tokens=0,
                                  spec_query_start_loc=None, spec_state_indices_tensor=None, spec_token_indx=None,
                                  non_spec_token_indx=None, num_accepted_tokens=None, num_actual_tokens=nT)
        if getattr(md, "_llmt_port", None) is not None:   # GDN-PORT index tensors (gdn_port.py) survive the copy
            object.__setattr__(md2, "_llmt_port", md._llmt_port)
        if sliced:   # RC-SLICE: views, output written in place
            out2 = core_attn_out[ts:nact]
            out2.zero_()
            md_all[self.prefix] = md2
            try:
                orig_core(self, mixed_qkv[ts:nact], b[ts:nact], a[ts:nact], out2)
            finally:
                md_all[self.prefix] = md
            sq, sb, sa = mixed_qkv[:ts], b[:ts], a[:ts]
        else:
            out2 = torch.zeros((nT, HV, V), dtype=core_attn_out.dtype, device=core_attn_out.device)
            md_all[self.prefix] = md2
            try:
                orig_core(self, mixed_qkv.index_select(0, nsi), b.index_select(0, nsi), a.index_select(0, nsi), out2)
            finally:
                md_all[self.prefix] = md
            core_attn_out[:nact].index_copy_(0, nsi, out2)
            si = md.spec_token_indx
            sq, sb, sa = mixed_qkv.index_select(0, si), b.index_select(0, si), a.index_select(0, si)
    else:
        sq, sb, sa = mixed_qkv, b, a
    T = sq.size(0)
    T1 = GDN_RC.get("T1", 7)
    if _RC_FUSED_IN:
        # the per-step index tensors are the same for every GDN layer of a group: build them once per metadata object
        key = (ns, GDN_RC.get("step", 0))   # step counter from the runner's per-step GDN-RC hook (metadata objects
        cached = getattr(md, "_llmt_rc_idx", None)   # or their persistent buffers may be reused across steps)
        if cached is None or cached[0] != key:
            cached = (key, md.spec_state_indices_tensor[:ns, 0].to(torch.int32).contiguous(),
                      md.spec_query_start_loc[:ns + 1].to(torch.int32).contiguous())
            try:
                object.__setattr__(md, "_llmt_rc_idx", cached)
            except Exception:
                pass
        sidx, qsl = cached[1], cached[2]
    else:
        sidx = md.spec_state_indices_tensor[:ns, 0].to(torch.int32).contiguous()
        qsl = md.spec_query_start_loc[:ns + 1].to(torch.int32).contiguous()
    x = causal_conv1d_update(sq, conv_state, conv_w, self.conv1d.bias, self.activation, conv_state_indices=sidx,
                             num_accepted_tokens=md.num_accepted_tokens, query_start_loc=md.spec_query_start_loc,
                             max_query_len=T1, validate_data=False)
    if _RC_FUSED_IN and x.stride(-1) == 1:
        from .gdn_rc import gdn_recompute_fused
        st = GDN_RC["buf"]
        gi = GDN_RC["layer_group"].get(self.prefix, 0)
        sa_ = sa if sa.stride(-1) == 1 else sa.contiguous()
        sb_ = sb if sb.stride(-1) == 1 else sb.contiguous()
        direct = pure and core_attn_out.dim() == 3 and core_attn_out.stride(-1) == 1 and core_attn_out.stride(-2) == V
        o = core_attn_out[:T] if direct else torch.empty((T, HV, V), dtype=core_attn_out.dtype, device=core_attn_out.device)
        gdn_recompute_fused(x, H, HV, K, V, sa_, sb_, self.A_log, self.dt_bias, o, ssm_state, sidx, *self._exl3_pool,
                            st["npend"][:ns], st["slot"][:ns], st["par"][:ns], qsl, T1, K ** -0.5,
                            st["bidx"][:ns], st["bblk"][gi, :ns])
        if not direct:
            if pure:
                core_attn_out[:T].copy_(o)
            else:
                core_attn_out[:nact].index_copy_(0, md.spec_token_indx, o)
        return
    q = x[:, :KD].reshape(T, H, K).contiguous()
    k = x[:, KD:2 * KD].reshape(T, H, K).contiguous()
    v = x[:, 2 * KD:].reshape(T, HV, V).contiguous()
    g = (-torch.exp(self.A_log.float()) * F.softplus(sa.float() + self.dt_bias.float())).contiguous()
    beta = torch.sigmoid(sb.float()).contiguous()
    o = torch.empty((T, HV, V), dtype=core_attn_out.dtype, device=core_attn_out.device)
    st = GDN_RC["buf"]
    gi = GDN_RC["layer_group"].get(self.prefix, 0)
    gdn_recompute(q, k, v, g, beta, o, ssm_state, sidx, *self._exl3_pool, st["npend"][:ns], st["slot"][:ns],
                  st["par"][:ns], qsl, T1, K ** -0.5, st["bidx"][:ns], st["bblk"][gi, :ns])
    if pure:
        core_attn_out[:T].copy_(o)
    else:
        core_attn_out[:nact].index_copy_(0, md.spec_token_indx, o)


# ------------------------------------------------------------------------------------------------
# KV-BLOCK-ALIGN: round the hybrid-negotiated attention block size up to a multiple of
# EXL3_KV_BLOCK_ALIGN (default 64), exactly like the XPU plugin's patch_xpu_block_size (EXL3_KV_BLOCK_EXACT=1). Both
# GPUs negotiate 1616 tokens for Qwen3.8 + int8 KV; the XPU rounds to 1664 for its GDN kernel, so CUDA must too or
# block boundaries (where GDN states are stored) fall at different token positions and cross-GPU cache hits fail.
def _patch_block_align():
    align = int(os.environ.get("EXL3_KV_BLOCK_ALIGN", "64"))
    if align <= 1:
        return
    try:
        from vllm.platforms import current_platform
        from vllm.platforms.interface import Platform
    except Exception as e:  # noqa
        logger.warning("exl3: block align not installed (%s)", e)
        return
    cls = type(current_platform)
    if getattr(cls, "_exl3_block_patched", False):
        return

    def update_block_size_for_backend(klass, vllm_config) -> None:
        Platform.update_block_size_for_backend.__func__(klass, vllm_config)
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        cc = vllm_config.cache_config
        layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        if not any(l.get_attn_backend().get_name() == "GDN_ATTN" for l in layers.values()):
            return
        bs = max(cc.block_size, align)
        new = (bs + align - 1) // align * align
        if new == cc.block_size:
            logger.info("exl3: keeping attention block size %d (multiple of %d)", new, align)
            return
        if cc.mamba_cache_mode == "align":
            cc.mamba_block_size = new
        if cc.mamba_page_size_padded is not None:
            cc.mamba_page_size_padded = new * (cc.mamba_page_size_padded // cc.block_size)
        cc.block_size = new
        logger.info("exl3: attention block size %d (multiple of %d, as on the Arc)", new, align)

    cls.update_block_size_for_backend = classmethod(update_block_size_for_backend)
    cls._exl3_block_patched = True


def _patch_dyn_sd_full():
    """EXL3_DYN_SD_FULL=1: keep FULL_AND_PIECEWISE cudagraphs with dynamic speculative depth
    (speculative_config.num_speculative_tokens_per_batch_size). Upstream drops to PIECEWISE "for reliability"; on the
    v1 runner a FULL graph is only dispatched for uniform decode batches with query length 1 + num_speculative_tokens
    and the exact request count (BatchDescriptor num_reqs), so steps with a smaller k take PIECEWISE graphs and steps
    with the full k (low concurrency) keep FULL graphs. Measured without this: dynamic k = -44% at 1 stream."""
    if _hwenv("EXL3_DYN_SD_FULL", "2", "0") not in ("1", "2"):
        return
    from vllm.config.vllm import VllmConfig

    def _keep(self):
        sc = self.speculative_config
        if sc is not None and sc.uses_dynamic_speculative_decoding():
            logger.info("EXL3_DYN_SD_FULL: dynamic speculative depth %s with cudagraph_mode %s (not forced to "
                        "PIECEWISE)", sc.num_speculative_tokens_per_batch_size, self.compilation_config.cudagraph_mode.name)
    VllmConfig._maybe_override_dynamic_sd_cudagraph_mode = _keep
    if _hwenv("EXL3_DYN_SD_FULL", "2", "0") == "2":
        _patch_multi_q_full_graphs()


def _patch_multi_q_full_graphs():
    """EXL3_DYN_SD_FULL=2: FULL decode cudagraphs for EVERY spec depth in the dynamic schedule, not only
    the maximum one. Upstream (v1 runner) captures FULL uniform-decode graphs for query length 1 + num_speculative_tokens
    only, so a step with a smaller k takes PIECEWISE graphs (measured: 2 streams at k=5 163 vs 309 tok/s). Here the
    dispatcher also gets FULL keys (b*q tokens, b requests, uniform) for each other q = 1 + k of the schedule and every
    b <= max_num_seqs, the capture of those keys runs with the runner's/dispatcher's query length switched to q, and at
    runtime a step whose requests all have q tokens dispatches with that q. Keys of different q never collide: FULL
    descriptors carry (num_tokens, num_reqs) and num_tokens = q * num_reqs."""
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.forward_context import BatchDescriptor
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def _qlens(vllm_config):
        sc = vllm_config.speculative_config
        sched = getattr(sc, "num_speculative_tokens_per_batch_size", None) if sc else None
        return sorted({1 + int(e[2]) for e in sched}) if sched else []

    _orig_init = CudagraphDispatcher.initialize_cudagraph_keys

    def _init_keys(self, cudagraph_mode, uniform_decode_query_len=1):
        _orig_init(self, cudagraph_mode, uniform_decode_query_len)
        self._llmt_q = None
        if cudagraph_mode == CUDAGraphMode.NONE or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL \
                or not cudagraph_mode.separate_routine():
            return
        max_size = self.compilation_config.max_cudagraph_capture_size or 0
        added = []
        for q in _qlens(self.vllm_config):
            if q == uniform_decode_query_len:
                continue
            for b in range(1, self.vllm_config.scheduler_config.max_num_seqs + 1):
                if b * q <= max_size:
                    self.add_cudagraph_key(CUDAGraphMode.FULL, BatchDescriptor(num_tokens=b * q, num_reqs=b, uniform=True))
                    added.append((q, b))
        logger.info("EXL3_DYN_SD_FULL=2: +%d FULL decode cudagraph keys for query lengths %s",
                    len(added), sorted({q for q, _ in added}))

    _orig_pad = CudagraphDispatcher._create_padded_batch_descriptor

    def _pad(self, num_tokens, uniform_decode, has_lora, num_active_loras=0):
        q = getattr(self, "_llmt_q", None)
        if uniform_decode and q and num_tokens % q == 0:
            return BatchDescriptor(num_tokens=num_tokens, num_reqs=num_tokens // q, uniform=True, has_lora=has_lora,
                                   num_active_loras=num_active_loras)
        return _orig_pad(self, num_tokens, uniform_decode, has_lora, num_active_loras)

    CudagraphDispatcher.initialize_cudagraph_keys = _init_keys
    CudagraphDispatcher._create_padded_batch_descriptor = _pad

    def _with_q(runner, q, fn):
        d = runner.cudagraph_dispatcher
        saved = (runner.uniform_decode_query_len, getattr(d, "_llmt_q", None), d.uniform_decode_query_len)
        runner.uniform_decode_query_len, d._llmt_q, d.uniform_decode_query_len = q, q, q
        try:
            return fn()
        finally:
            runner.uniform_decode_query_len, d._llmt_q, d.uniform_decode_query_len = saved

    _orig_cap = GPUModelRunner._warmup_and_capture

    def _cap(self, desc, cudagraph_runtime_mode, *a, **k):
        if desc.uniform and desc.num_reqs and desc.num_tokens // desc.num_reqs != self.uniform_decode_query_len:
            q = desc.num_tokens // desc.num_reqs
            return _with_q(self, q, lambda: _orig_cap(self, desc, cudagraph_runtime_mode, *a, **k))
        return _orig_cap(self, desc, cudagraph_runtime_mode, *a, **k)

    GPUModelRunner._warmup_and_capture = _cap

    _orig_det = GPUModelRunner._determine_batch_execution_and_padding

    def _det(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, *a, **k):
        q = int(max_num_scheduled_tokens)
        if (q != self.uniform_decode_query_len and k.get("force_uniform_decode") is None
                and q in getattr(self, "_llmt_qset", ()) and num_tokens == q * num_reqs):
            return _with_q(self, q, lambda: _orig_det(self, num_tokens, num_reqs, num_scheduled_tokens_np,
                                                      max_num_scheduled_tokens, *a, **k))
        return _orig_det(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, *a, **k)

    GPUModelRunner._determine_batch_execution_and_padding = _det

    _orig_rinit = GPUModelRunner.__init__

    def _rinit(self, vllm_config, *a, **k):
        _orig_rinit(self, vllm_config, *a, **k)
        self._llmt_qset = set(_qlens(vllm_config))

    GPUModelRunner.__init__ = _rinit


def _patch_uniform_spec_only():
    """UNIFORM-SPEC (EXL3_UNIFORM_SPEC=1 default; 0 = upstream): upstream calls a step "uniform
    decode" from token counts alone (every request has 1 + k scheduled tokens), so a PREFILL chunk of exactly 1 + k
    tokens (a 13-token prompt at k = 12; with dynamic depth also 4 and 8 tokens, e.g. the last few tokens after a
    --prefix-match-unit stop or a block-aligned chunk split) was dispatched as a spec decode: FULL decode cudagraph whose
    GDN layers run the spec-decode kernels (accepted-count conv window, recompute pool) on metadata built for a prefill
    -> NaN logits at the first decode (NAN-PREEMPT recomputes, same split -> again). Same class as GDN-1TOK. Here a step
    counts as uniform decode only if every scheduled request is a spec decode (drafts scheduled, 1 + drafts tokens)."""
    if os.environ.get("EXL3_UNIFORM_SPEC", "1") != "1":
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _orig_exec = GPUModelRunner.execute_model

    _prof = {"at": int(os.environ.get("EXL3_PROFILE_AT", "0") or 0), "n": int(os.environ.get("EXL3_PROFILE_N", "8") or 8), "i": 0, "p": None}

    def execute_model(self, scheduler_output, *a, **k):
        # kernel profile: EXL3_PROFILE_AT=<step index> starts torch.profiler for EXL3_PROFILE_N steps once and logs
        # the top kernels by CUDA time (what a decode-only / mixed step spends its ~30 ms / ~1.2 s on).
        pr = _prof
        if pr["at"]:
            pr["i"] += 1
            if pr["i"] == pr["at"]:
                import torch.profiler as _tp
                pr["p"] = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA], record_shapes=False)
                pr["p"].__enter__()
            elif pr["p"] is not None and pr["i"] == pr["at"] + pr["n"]:
                p_ = pr["p"]; pr["p"] = None; pr["at"] = 0
                p_.__exit__(None, None, None)
                try:
                    tab = p_.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70)
                    logger.info("exl3 PROFILE %d steps:\n%s", pr["n"], tab)
                except Exception as e:
                    logger.warning("exl3 PROFILE table failed: %r", e)
        try:
            spec = scheduler_output.scheduled_spec_decode_tokens or {}
            nst = scheduler_output.num_scheduled_tokens
            self._llmt_all_spec = bool(nst) and len(spec) == len(nst) and all(
                nst.get(r, 0) == len(d) + 1 for r, d in spec.items())
        except Exception:
            self._llmt_all_spec = None
        return _orig_exec(self, scheduler_output, *a, **k)
    GPUModelRunner.execute_model = execute_model

    _orig_det = GPUModelRunner._determine_batch_execution_and_padding

    def _det(self, *a, **k):
        if k.get("force_uniform_decode") is None and getattr(self, "_llmt_all_spec", None) is False:
            k["force_uniform_decode"] = False
            self._llmt_us_n = getattr(self, "_llmt_us_n", 0) + 1
            if self._llmt_us_n <= 3 or self._llmt_us_n % 1000 == 0:
                logger.info("exl3: UNIFORM-SPEC: step with prefill/non-spec rows kept off the uniform-decode path (%d)",
                            self._llmt_us_n)
        return _orig_det(self, *a, **k)
    GPUModelRunner._determine_batch_execution_and_padding = _det
    logger.info("exl3: UNIFORM-SPEC on (uniform-decode dispatch only for all-spec-decode steps)")


def _patch_graph_dump():
    """EXL3_GRAPH_DUMP=1 (diagnosis): enable PyTorch's CUDA graph debug mode for every graph vLLM captures
    and write each one as a .dot file (node types: kernel, memcpy, memset, host, event, mem alloc/free) to
    /root/.cache/graph-dump/. Used to find why the FULL decode graph's cudaGraphLaunch costs ~2.3 ms of CPU."""
    if os.environ.get("EXL3_GRAPH_DUMP", "0") != "1":
        return
    import itertools
    import torch
    import vllm.compilation.cuda_graph as cg
    out = "/root/.cache/graph-dump"
    os.makedirs(out, exist_ok=True)
    counter = itertools.count()
    base = torch.cuda.CUDAGraph

    class _DumpGraph(base):
        def __init__(self, *a, **k):
            k.setdefault("keep_graph", True)               # debug_dump needs the graph kept after instantiation
            super().__init__(*a, **k)
            self.enable_debug_mode()

        def capture_end(self):
            super().capture_end()
            try:
                self.debug_dump(os.path.join(out, f"g{next(counter):03d}-pid{os.getpid()}.dot"))
            except Exception as e:  # noqa
                logger.warning("EXL3_GRAPH_DUMP: dump failed: %s", e)

    cg.torch.cuda.CUDAGraph = _DumpGraph
    logger.info("EXL3_GRAPH_DUMP: dumping captured CUDA graphs to %s", out)


def _patch_kvw4():
    """KVW4 (port of the XPU int4 KV write): fused int4_per_token_head KV write (csrc/kvw4/kvw4_kernel.cu: signs, fp32
    Hadamard, fp16 round, vLLM's quantization math, nibble pack; one launch for K and V) instead of vLLM's
    reshape_and_cache_int4 (2 x single_rht through bf16 hadacore + casts + the pack kernel; eager 449 us vs 8 us at 8
    tokens). 97 % byte-equal to stock (stock rounds the input to bf16 first), error equal or lower. EXL3_KVW4=0 = off."""
    if _hwenv("EXL3_KVW4", "1", "0") != "1":
        return
    try:
        import vllm.v1.attention.ops.int4_per_token_head as i4m
        torch.ops.load_library(os.environ.get("EXL3_KVW4_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "kvw4.so")))
    except Exception as e:  # noqa
        logger.warning("exl3: KVW4 not available (%s); stock int4 KV write", e)
        return
    orig = i4m.reshape_and_cache_int4
    st = {}

    def reshape_and_cache_int4(key, value, key_cache, value_cache, slot_mapping, *, k_scale_cache, v_scale_cache):
        if (key.dtype == torch.float16 and value.dtype == torch.float16 and key.dim() == 3 and key.shape[-1] == 256
                and _device_ok(key.device)
                and key.stride(-1) == 1 and value.stride(-1) == 1 and key_cache.dtype == torch.uint8
                and value_cache.data_ptr() == key_cache.data_ptr() + 132 and slot_mapping.dtype == torch.int64
                and k_scale_cache.data_ptr() == key_cache.data_ptr() + 128):
            sg = st.get(key.device)
            if sg is None:
                sg = st[key.device] = i4m._get_rht_signs(256, 0, key.device).float().contiguous()
                logger.info("exl3: KVW4 on: fused int4 KV write")
            torch.ops.kvw4.write(key, value, key_cache, key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                                 slot_mapping, key_cache.shape[1], sg)
            return
        return orig(key, value, key_cache, value_cache, slot_mapping, k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache)
    i4m.reshape_and_cache_int4 = reshape_and_cache_int4


def _import_whx_loader():
    """The weight keeper's engine-side loader lives in AllBroker (memory policy is not part of this package).
    EXL3_WHX_MODULE names it; default: llmt_kv.whx_loader, then a whx_loader module next to this package, then a
    top-level whx_loader."""
    import importlib
    names = [os.environ["EXL3_WHX_MODULE"]] if os.environ.get("EXL3_WHX_MODULE") else [
        "llmt_kv.whx_loader", __package__ + ".whx_loader", "whx_loader"]
    err = None
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError as e:
            err = e
    raise err


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    _patch_graph_dump()
    _patch_dyn_sd_full()
    _patch_uniform_spec_only()
    try:                                      # weight loading without mmap (EXL3_NOMMAP_LOAD, default on)
        from llmt_kv import nommap_loader     # lives in the llmt_kv package (kvbroker client)
        nommap_loader.patch()
    except ImportError:                       # llmt_kv not installed: vLLM's own loader
        logger.info("nommap_loader not installed (llmt_kv missing); using vLLM's safetensors loader")
    except Exception as e:  # noqa: never block engine start; vLLM's own loader stays in place
        logger.warning("nommap_loader not active (%s: %s); using vLLM's safetensors loader", type(e).__name__, e)
    if os.environ.get("EXL3_WHX"):            # optional weight-keeper loader (AllBroker; not part of this package)
        try:
            whx_loader = _import_whx_loader()
            whx_loader.install()
        except Exception as e:  # noqa: never block engine start; weights load from disk
            logger.warning("exl3: WHX not active (%s: %s)", type(e).__name__, e)
    _patch_block_align()
    _patch_ksmooth()
    if _hwenv("EXL3_GDN_PORT", "1", "0") == "1":   # before GDN-RC: its wrapper must call the port-aware core
        try:
            from . import gdn_port
            gdn_port.install()
        except Exception as e:  # noqa: never block engine start; vLLM's GDN prefill stays in place
            logger.warning("exl3: GDN-PORT not active (%s: %s)", type(e).__name__, e)
    _patch_gdn_recompute()
    _patch_int8kv_fa()
    _patch_fastlaunch_eagle()
    _patch_kvw4()
    if os.environ.get("EXL3_DRAFT_VOCAB"):
        _patch_mtp_draft_logits()
    if _hwenv("EXL3_TAIL_FUSE", "1", "0") not in ("", "0"):   # TAIL-FUSE (tail_fuse.py)
        try:
            import sys
            from . import tail_fuse
            tail_fuse.install(sys.modules[__name__])
        except Exception as e:  # noqa: never block engine start; the stock paths stay in place
            logger.warning("exl3: TAIL-FUSE not active (%s: %s)", type(e).__name__, e)
    return None
