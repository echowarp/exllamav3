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
| `i8fa_prefill.py`, `csrc/i8fa/` | I8FA int8-KV prefill helpers and kernels |
| `csrc/fa16/` | FA16 FlashAttention 2 variant (patch + binding + build) |
| `csrc/va/` | VA verify attention kernel (int8 KV) |
| `csrc/va4/`, `csrc/kvw4/` | VA4 verify attention and KVW4 KV write (int4 KV) |
| `gdn_port.py`, `gdn_fast.py`, `csrc/gdnchunk/` | GDN prefill via the Magnitude kernel; fast GDN decode metadata |
| `draftdyn.py`, `tools/quantize_draft.py` | DRAFT-DYN pieces; draft-head re-quantization tool |

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
| `EXL3_I8FA` | 1 | I8FA: CUDA gather-dequant + LSE merge around the same FA2 calls (bitwise equal), paged in-kernel-dequant FA2 for few rows |
| `EXL3_I8FA_DIR` | default (`<package>/lib`) | directory with `i8kv/`, `i8fa_fp/`, `i8fa_a/` |
| `EXL3_I8FA_PAGED_MAXQ` | auto | paged route: auto = Q <= 128 and >= 3 parts, N = Q <= N, 0 = off |
| `EXL3_VA` | 1 | VA: custom CUDA verify / decode attention (<= 8 query rows) for int8 KV; also enables VA4 |
| `EXL3_VA_SO` | default (`<package>/lib/va5090.so`) | VA library |
| `EXL3_VA_WAVE` | default (42) | VA CTA-segments per KV head and sequence (one wave on 170 SMs) |
| `EXL3_VA4` | default (1) | VA4: int4_per_token_head verify / decode attention (with EXL3_VA=1) |
| `EXL3_VA4_SO` | `<package>/lib/va4c.so` | VA4 library (the code default names an older `va4b.so` variant that is not shipped: set this) |
| `EXL3_VA4_OP` | va4c | VA4 op namespace (code default `va4b`: set `va4c` for the kernel in csrc/va4) |
| `EXL3_VA4_PREFILL_Q` | default (64) | VA4-PSEUDO: prefills up to this many rows go through VA4 as 8-row pseudo-sequences |
| `EXL3_I4FA` | default (1) | I4FA: int4 KV prefill = gather-dequant + rotated q + the same FA2 / FA16 calls |
| `EXL3_KVW4` | default (1) | KVW4: fused int4 KV-cache write |
| `EXL3_KVW4_SO` | default (`<package>/lib/kvw4.so`) | KVW4 library |
| `EXL3_KSMOOTH_INT4` | default (1) | KS4: K smoothing applied to int4 KV too |
| `EXL3_GDN_PORT` | 1 | GDN-PORT: GDN prefill through the Magnitude WY kernel (csrc/gdnchunk) instead of FLA |
| `EXL3_GDN_PORT_SO` | default (`<package>/lib/gdn_chunk_mag.so`) | GDN-PORT extension |
| `EXL3_GDN_PORT_ROWS` | default (64) | GDN-PORT scan tile |
| `EXL3_GDN_FAST` | 1 | GDN-FAST: one Triton launch for the decode-step GDN metadata (2 = check every field against the original) |
| `EXL3_DRAFT_VOCAB` | path to a 512-block draft vocab json | pruned MTP draft head: the drafter proposes from these 128-token vocab blocks (target verifies with the full head) |
| `EXL3_DRAFT_VOCAB_FAR` | same file | GRADED-DRAFT: vocab for draft positions >= EXL3_DRAFT_FAR_FROM (same file = one shared head copy) |
| `EXL3_DRAFT_FAR_FROM` | 2 | first draft position (0-based) that uses the far vocab |
| `EXL3_DRAFT_WEIGHTS` | path to a 4-bit draft head | re-quantized draft head / MTP linears (tools/quantize_draft.py) |
| `EXL3_DRAFT_DYN / _NEAR` | 256 / 128 | DRAFT-DYN: up to E extra draft-head rows per request (its most frequent prompt tokens), far / near passes |
| `EXL3_DRAFT_BIAS` | 0 | DRAFT-DYN: drafter logit bias on prompt tokens |
| `EXL3_DRAFT_DYN_GS / _BG / _DELAY_US` | default (64 / 1 / 2000) | DRAFT-DYN host int4 group size, background uploads, upload delay |
| `EXL3_DRAFT_WINDOW` | unset | DRAFT-WINDOW: sliding window for the MTP layer's attention (drafts only) |
| `EXL3_CAPTURE_MTP_H / _EVERY` | unset | capture X^T X of the MTP linears' inputs for calibrated draft re-quantization (eager engine) |

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

### I8FA (`csrc/i8fa`)

    cd exl3cuda/csrc/i8fa
    python3 build.py i8kv            # gather-dequant + merge -> lib/i8kv/i8kv.so
    python3 build.py i8fa_fp 1 2     # paged FA2, FA16 f numerics (needs ../fa16/fetch_and_patch.sh first)
    python3 build.py i8fa_a 0 0      # paged FA2, stock numerics (used when EXL3_FA16 is unset)

### VA (`csrc/va`)

    python3 exl3cuda/csrc/va/build.py        # -> lib/va5090.so (torch.ops.va5090)

### VA4 and KVW4 (`csrc/va4`, `csrc/kvw4`)

    python3 exl3cuda/csrc/va4/build.py       # -> lib/va4c.so (torch.ops.va4c); extra nvcc flags are passed through
    python3 exl3cuda/csrc/kvw4/build.py      # -> lib/kvw4.so (torch.ops.kvw4)

### GDN chunk (`csrc/gdnchunk`)

    python3 exl3cuda/csrc/gdnchunk/build.py  # -> lib/gdn_chunk_mag.so (no GPU needed)

See `csrc/gdnchunk/README.md` for the kernel, its argument contract and measurements.

## Data files you supply

These are derived from a deployment's own traffic or weights and are not part of the package:

- **K smoothing table** (`EXL3_KSMOOTH`): not shipped. Generate it on your own traffic: run the engine eagerly
  (`--enforce-eager`) with `EXL3_KSMOOTH_CALIB=<out.pt>` (and int8 KV) over representative prompts; the plugin saves
  `{layer_name: mean[H, D]}` every 64 calls. Then serve with `EXL3_KSMOOTH=<out.pt>`.

- **Draft vocabulary** (`EXL3_DRAFT_VOCAB`): not shipped. A json `{"block_size": 128, "blocks": [...]}` listing the
  128-token vocab blocks the drafter may propose (block_size must be a multiple of 128). Rank blocks by token
  frequency on text like your own model outputs and keep the top N (production uses 512 blocks); force-include
  special tokens. The exl3xpu project's `scripts/draft_vocab.py` (MIT) builds one from a text corpus.
- **Draft weights** (`EXL3_DRAFT_WEIGHTS`): not shipped. `tools/quantize_draft.py --bf16 <BF16 checkpoint dir>
  --vocab <draft vocab json> --no-mtp --k-head 4 --out head4.safetensors` re-quantizes the draft head rows to 4 bpw
  (drafts only: outputs stay exact). Needs exllamav3 and the BF16 weights of the model.

## Licenses

Third-party code in this directory keeps its own license: see `THIRD_PARTY_NOTICES.md`.
