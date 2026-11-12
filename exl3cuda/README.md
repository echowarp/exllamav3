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
| `gdn_rc.py` | Triton GDN recompute kernels |
| `csrc/fa16/` | FA16 FlashAttention 2 variant (patch + binding + build) |

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
| `EXL3_GDN_RECOMPUTE` | 1 | GDN recompute: MTP spec decoding with one committed SSM state per request (no 1 + k state pages); needs the companion vLLM gpu_model_runner hooks |
| `EXL3_GDN_RC_SLOTS` | default (16) | GDN recompute pending-pool slots |
| `EXL3_RC_FUSED_IN` | 1 | RC-FUSED-IN: q/k/v split, gating and output write inside the recompute kernel |
| `EXL3_RC_SLICE` | 1 | RC-SLICE: mixed steps use views for the non-spec rows instead of gather/scatter (bit-identical) |
| `EXL3_RC_BV / EXL3_RC_WARPS` | default (32 / 4) | recompute kernel tile / warps |
| `EXL3_KSMOOTH` | path to a K-mean table | KSMOOTH: store K - mean (per layer / KV head / channel) in the quantized KV cache; softmax unchanged |
| `EXL3_KSMOOTH_CALIB` | unset | KSMOOTH calibration: accumulate key means of real traffic (eager engine) and save the table to this path |
| `EXL3_KSMOOTH_ROBUST` | default (1) | calibration ignores token-heads with key norm > 3x the median (attention sinks) |
| `EXL3_INT8KV_FA` | default (1) | INT8KV-FA: prefill rows (> EXL3_VERIFY_Q per request) on block-dequantized FA2 instead of Triton unified attention (int8_per_token_head KV) |
| `EXL3_INT8KV_KB` | default (32768) | INT8KV-FA / I4FA: context tokens dequantized per FA2 call (merged by log-sum-exp) |
| `EXL3_VERIFY_Q` | 8 | query rows per request treated as decode / verify (k = 7 verify = 8) |
| `EXL3_FA16` | f | FA16: route those FA2 calls to the fp16-accumulate build (b = P.V fp16, f = b + Q.K fp16); unset = stock FA2 |

## Building the kernels

All kernels are sm_120 (RTX 5090) builds through `torch.utils.cpp_extension` (no GPU needed to compile). If your CUDA
toolkit lacks headers that torch includes (cuSPARSE / cuSOLVER / cuFFT, e.g. in a runtime container), pass their
directory in `EXL3_EXTRA_INC` (FA16 / I8FA builds).

### FA16 (`csrc/fa16`)

FlashAttention 2 forward (hdim 256, fp16, 64 x 64 tile) with fp16-accumulate MMAs, built against
vllm-project/flash-attention `caaa4eb5` (vLLM 0.26.0's pin) + CUTLASS, patched with `srcmod.diff`:

    cd exl3cuda/csrc/fa16
    ./fetch_and_patch.sh            # clones FA2 + cutlass, writes ./srcmod
    python3 build.py fa_f 1 2       # production variant f -> exl3cuda/lib/fa_f/fa_f.so  (fa_b: 1 0)
    export PYTHONPATH=$PWD/../../lib/fa_f:$PYTHONPATH   # the plugin imports module fa_f (or from /fa16ext/fa_f)

One nvcc needs ~12 GB RAM (`MAX_JOBS=1`); `build-seq.sh fa_f:1:2 fa_b:1:0` builds variants one at a time.

## Data files you supply

These are derived from a deployment's own traffic or weights and are not part of the package:

- **K smoothing table** (`EXL3_KSMOOTH`): not shipped. Generate it on your own traffic: run the engine eagerly
  (`--enforce-eager`) with `EXL3_KSMOOTH_CALIB=<out.pt>` (and int8 KV) over representative prompts; the plugin saves
  `{layer_name: mean[H, D]}` every 64 calls. Then serve with `EXL3_KSMOOTH=<out.pt>`.

## Licenses

Third-party code in this directory keeps its own license: see `THIRD_PARTY_NOTICES.md`.
