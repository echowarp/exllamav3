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
        replaced = False
        layer.exl3_id, layer.exl3_n = _register(segs, k, cbt), bounds[-1]
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


def _register(segs, k, cbt) -> int:
    from exllamav3.modules.quant.exl3 import LinearEXL3
    lins = [LinearEXL3(None, k, s["svh"].shape[0], suh=s["suh"], svh=s["svh"], trellis=s["tr"],
                       mcg=cbt["mcg"], mul1=cbt["mul1"], out_dtype=torch.half) for s in segs]
    _SEGS.append(lins)
    if FUSE_DECODE and len(lins) > 1:
        try:
            _FUSED[len(_SEGS) - 1] = _Sliced(lins, FUSE_WIDTH)
        except ValueError as e:
            logger.debug("exl3: fused decode not used for this module: %s", e)
    return len(_SEGS) - 1


# FUSED-DECODE: at decode/verify row counts, run a fused module's members (gate+up, q+k+v,
# GDN qkv+z) as ONE exllamav3 sliced mgemm launch instead of one exl3 GEMM per member plus a torch.cat. The members
# keep their own trellis, suh (input transform) and svh; the kernel schedules equal-width column slices of all members
# as one grid, so the per-launch ramp/tail (~8-10 us at these sizes on the 5090) is paid once. Same math as the
# per-member path (fp16 output rounding may differ in the last bit). EXL3_FUSE_DECODE=1, rows <= EXL3_FUSE_MAX_M,
# slice width EXL3_FUSE_WIDTH (0 = gcd of the member widths).
FUSE_DECODE = os.environ.get("EXL3_FUSE_DECODE", "0") == "1"
FUSE_MAX_M = int(os.environ.get("EXL3_FUSE_MAX_M", "16"))
FUSE_WIDTH = int(os.environ.get("EXL3_FUSE_WIDTH", "0"))
_FUSED: dict = {}
# FUSE-SCRATCH: for m > 1 the member blocks are scratch until the torch.cat, so all fused modules
# share one persistent scratch buffer and precomputed output-pointer tables; this drops the per-call `coff[m] + ptr`
# device add (133 single-block launches per 1-stream step). Calls run in stream order, the cat copies out before the
# next module writes. EXL3_FUSE_SCRATCH=1 enables.
FUSE_SCRATCH = os.environ.get("EXL3_FUSE_SCRATCH", "0") == "1"
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
INT8_PREFILL = os.environ.get("EXL3_INT8_PREFILL", "0") == "1"
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
_L2SPLIT = os.environ.get("EXL3_INT8_L2SPLIT", "0") == "1"
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


@torch.library.custom_op("exl3cuda::linear", mutates_args=())
def _linear(x: torch.Tensor, layer_id: int, n: int) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
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
            if (qm == KVQuantMode.INT8_PER_TOKEN_HEAD and name is not None):
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
_FA16_V = os.environ.get("EXL3_FA16", "").strip().lower()
_FA16 = None
if _FA16_V in ("b", "f"):
    try:
        import importlib, sys as _sys
        _sys.path.insert(0, f"/fa16ext/fa_{_FA16_V}")
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
    if _FA16 is not None and q.dtype == torch.float16 and q.shape[-1] == 256:
        return tuple(_FA16.fwd(q, k, v, cq, ck, Q, L, scale, causal))
    return flash_attn_varlen_func(q, k, v, Q, cq, L, cu_seqlens_k=ck, causal=causal, softmax_scale=scale,
                                  return_softmax_lse=True)


def _i8_prefill_fa(q, key_cache, value_cache, pages, seq_len, scale, out):
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

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        md = attn_metadata
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
                              k_scale_cache=self._k_scale_cache, v_scale_cache=self._v_scale_cache)
            out.index_copy_(0, rows, os_)
        return output

    TritonAttentionImpl.forward = forward
    TritonAttentionImpl._exl3_fa_patched = True
    logger.info("exl3: int8 KV prefill via block-dequantized FA2 (blocks of %d tokens)", _FA_KB)


# ------------------------------------------------------------------------------------------------
# GDN-RECOMPUTE, CUDA port of the XPU plugin's _patch_gdn_recompute / _rc_dispatch: MTP spec decoding
# for the 48 GDN layers with ONE committed SSM state per request
# (Triton kernel exl3cuda.gdn_rc.gdn_recompute), so MambaSpec.num_speculative_blocks = 0 and a running request no
# longer reserves 1 + k state pages per GDN group. The runner-side bookkeeping (pool slot / parity / pending count /
# decode-time boundary states) is in the patched vLLM gpu_model_runner.py (_llmt_gdn_rc). EXL3_GDN_RECOMPUTE=1.
GDN_RC = {"on": os.environ.get("EXL3_GDN_RECOMPUTE", "0") == "1", "S": int(os.environ.get("EXL3_GDN_RC_SLOTS", "16"))}


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
    import dataclasses
    try:
        from vllm.model_executor.layers.mamba import abstract as mabs
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.model_executor.layers.mamba import mamba_utils as mmu
        from vllm.v1.worker import mamba_utils as wmu
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qg
    except Exception as e:  # noqa
        logger.warning("exl3: GDN recompute not installed (%s)", e)
        return
    if getattr(mabs.MambaBase, "_exl3_rc", False):
        return
    # 1. no speculative state blocks (as on the XPU)
    orig_spec = mabs.MambaBase.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig_spec(self, vllm_config)
        if isinstance(spec, MambaSpec) and spec.num_speculative_blocks:
            GDN_RC["T1"] = spec.num_speculative_blocks + 1
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec
    mabs.MambaBase.get_kv_cache_spec = get_kv_cache_spec
    mabs.MambaBase._exl3_rc = True

    # 2. block-crossing SSM copy takes the committed state (no accept bias); same function object (identity checks)
    def _temporal(state, block_ids, cur_block_idx, num_accepted_tokens):
        src_state = state[block_ids[cur_block_idx]]
        return MambaCopySpec(start_addr=src_state.data_ptr(), num_elements=src_state.numel())
    mmu.get_temporal_copy_spec.__code__ = _temporal.__code__

    # 3. fused postprocess: no SSM copies (the kernel stores decode-time boundary states itself)
    orig_init = wmu.MambaSpecDecodeGPUContext.initialize_from_forward_context

    def initialize_from_forward_context(self, *a, **kw):
        was = self.is_initialized
        orig_init(self, *a, **kw)
        if not was:
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
        if (md is None or md.spec_sequence_masks is None or int(md.num_spec_decodes or 0) == 0
                or "buf" not in GDN_RC or not hasattr(self, "_exl3_pool")):
            return orig_core(self, mixed_qkv, b, a, core_attn_out)
        return _rc_forward(self, orig_core, md, md_all, mixed_qkv, b, a, core_attn_out)
    cls._forward_core = _forward_core
    logger.info("exl3: GDN recompute enabled (num_speculative_blocks = 0, %d pending slots)", GDN_RC["S"])


# RC-FUSED-IN: gdn_rc.gdn_recompute_fused does the q/k/v split, gating and output write inside
# the recompute kernel (~12 fewer tiny launches per GDN layer); EXL3_RC_FUSED_IN=1.
_RC_FUSED_IN = os.environ.get("EXL3_RC_FUSED_IN", "0") == "1"
# RC-SLICE (on the XPU the gather/scatter cost 5.9% of a mixed 30k-token step):
# in mixed steps the non-spec (prefill) rows were gathered with index_select per GDN layer and the output scattered back
# with index_copy. GDN-RC already requires the spec rows to lead the batch (runner sets GDN_RC["lead"] per step), so
# the non-spec tokens are the contiguous range [num_spec_decode_tokens, num_actual_tokens): use views, and let the
# prefill path write straight into core_attn_out. Same rows, same order -> bit-identical. EXL3_RC_SLICE=1.
_RC_SLICE = os.environ.get("EXL3_RC_SLICE", "0") == "1"


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


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    _patch_graph_dump()
    try:                                      # weight loading without mmap (EXL3_NOMMAP_LOAD, default on)
        from llmt_kv import nommap_loader     # lives in the llmt_kv package (kvbroker client)
        nommap_loader.patch()
    except ImportError:                       # llmt_kv not installed: vLLM's own loader
        logger.info("nommap_loader not installed (llmt_kv missing); using vLLM's safetensors loader")
    except Exception as e:  # noqa: never block engine start; vLLM's own loader stays in place
        logger.warning("nommap_loader not active (%s: %s); using vLLM's safetensors loader", type(e).__name__, e)
    if os.environ.get("EXL3_WHX"):            # optional weight-keeper loader (module whx_loader, not part of this package)
        try:
            from . import whx_loader
            whx_loader.install()
        except Exception as e:  # noqa: never block engine start; weights load from disk
            logger.warning("exl3: WHX not active (%s: %s)", type(e).__name__, e)
    _patch_block_align()
    _patch_ksmooth()
    _patch_gdn_recompute()
    _patch_int8kv_fa()
    return None
