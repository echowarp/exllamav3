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

Built kernels go to `exl3cuda/lib/` (ignored by git, installed as package data).

## Environment variables

Production = the Qwen3.8-27B RTX 5090 service (int4_per_token_head KV, MTP k = 7); "default (x)" = not set, code default x.

| Variable | Production | What it does |
|---|---|---|
| `EXL3_INT8_FUSED_MAX_M` | default (512) | fused_kernels.fused_linear: F1 (fused decode + GEMM) up to this many rows, F2 above |
| `EXL3_INT8_FUSED_ASM` | default (1) | fused_kernels: inline-PTX decode (0 = pure-Triton spelling, same bits) |
