# exl3cuda: EXL3 weights in vLLM on NVIDIA CUDA (RTX 5090)

A vLLM out-of-tree plugin that serves exllamav3 EXL3 checkpoints (trellis-quantized weights) with exllamav3's CUDA
kernels, plus the kernels and engine patches used to serve Qwen3.8-27B (EXL3 4.0 bpw, MTP speculative decoding, int4 KV
cache, long contexts, several concurrent agents) on one RTX 5090. It is a sibling package of `exllamav3` in this
repository and uses its extension (`exllamav3_ext`), including this fork's kernel changes (sm_120 fp16 MMA
accumulation, gemv3 and the mgemm `c_row_stride` / `c_base` outputs).

Tested with vLLM 0.26.0 (CUDA 13), PyTorch 2.11, sm_120 only. Everything beyond the plain EXL3 linear is opt-in through
`EXL3_*` environment variables; any feature whose kernel or vLLM internals are missing logs a warning and leaves the
stock path in place. Some features need companion patches to vLLM itself (noted per variable).

    pip install ./exl3cuda                     # after building the kernels you want (see below)
    vllm serve <EXL3 checkpoint dir> ...       # quant_method "exl3" in its config.json selects the plugin

## Layout

| Path (under `exl3cuda/`) | What |
|---|---|
| `__init__.py`, `pyproject.toml` | package; `pip install ./exl3cuda` (registers the vLLM plugin entry point) |
| `int8_kernels.py`, `fused_kernels.py` | Triton int8 (W8A8) prefill GEMM pieces; fused trellis -> int8 decode (L2 split) |
| `exl3_mma.py` | Triton EXL3 GEMM for 17-64 rows (decode each weight tile once per program) |
| `vllm_plugin.py` | the vLLM plugin: `exl3` quantization config, linear methods, custom op `exl3cuda::linear` and all engine patches |

Built kernels go to `exl3cuda/lib/` (ignored by git, installed as package data).

## Environment variables

Production = the Qwen3.8-27B RTX 5090 service (int4_per_token_head KV, MTP k = 7); "default (x)" = not set, code default x.

| Variable | Production | What it does |
|---|---|---|
| `EXL3_INT8_FUSED_MAX_M` | default (512) | fused_kernels.fused_linear: F1 (fused decode + GEMM) up to this many rows, F2 above |
| `EXL3_INT8_FUSED_ASM` | default (1) | fused_kernels: inline-PTX decode (0 = pure-Triton spelling, same bits) |
| `EXL3_FUSE_DECODE` | 1 | FUSED-DECODE: a fused module's members (gate+up, q+k+v, GDN qkv+z) as one sliced exllamav3 mgemm at decode/verify row counts |
| `EXL3_FUSE_MAX_M` | 64 | FUSED-DECODE row limit (default 16) |
| `EXL3_FUSE_WIDTH` | default (0) | FUSED-DECODE slice width (0 = gcd of the member widths) |
| `EXL3_FUSE_SCRATCH` | 1 | FUSE-SCRATCH: one shared scratch buffer + precomputed pointer tables for the fused decode modules |
| `EXL3_INT8_PREFILL` | 1 | INT8-PREFILL: W8A8 int8 tensor-core GEMMs for prefill rows (Hadamard-basis activations, per-token / per-column scales) |
| `EXL3_INT8_MIN_ROWS` | default (256) | INT8-PREFILL: minimum rows per call |
| `EXL3_RECON_SLICE_N` | default (16384) | INT8-PREFILL: output columns reconstructed per slice |
| `EXL3_INT8_HAD_EPI` | default (1) | INT8-PREFILL: output Hadamard fused into the GEMM epilogue |
| `EXL3_INT8_HAD_QUANT / _MIN` | default (1 / 2048) | INT8-PREFILL: fused input Hadamard + activation quant from this many rows |
| `EXL3_INT8_RECON_Q8` | default (1) | INT8-PREFILL: trellis decoded straight to transposed int8 (bit-exact) |
| `EXL3_INT8_L2SPLIT` | 1 | L2SPLIT: faster bit-identical trellis decode into L2-sized int8 slices (fused_kernels.l2_split_linear) |
| `EXL3_INT8_L2_BUDGET_MB` | default (64) | L2SPLIT slice budget |
| `EXL3_KV_BLOCK_ALIGN` | default (64) | KV-BLOCK-ALIGN: attention block size of the hybrid GDN model rounded up to a multiple of this (same blocks as the XPU plugin) |
| `EXL3_NOMMAP_LOAD` | default (1) | weight loading without mmap through `llmt_kv.nommap_loader` (skipped when `llmt_kv` is not installed) |
| `EXL3_GRAPH_DUMP` | unset | diagnosis: dump every captured CUDA graph as .dot to /root/.cache/graph-dump/ |
| `EXL3_WHX` | unset | optional external weight-keeper loader (module `exl3cuda.whx_loader`, not part of this package) |

## Licenses

Third-party code in this directory keeps its own license: see `THIRD_PARTY_NOTICES.md`.
