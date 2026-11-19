# exllamav3

A fork of [exllamav3](https://github.com/turboderp-org/exllamav3) that serves Qwen3.8-27B (EXL3 4.0 bpw) in vLLM on one
RTX 5090 (32 GB), for several tool-calling agents at once. It adds `exl3cuda/`, a vLLM plugin that runs EXL3
checkpoints on exllamav3's CUDA kernels plus this fork's own (int4 KV attention, MTP drafting, GDN, fused decode), and
three small kernel changes to the library. With [echowarp/vllm](https://github.com/echowarp/vllm) and
[AllBroker](https://github.com/echowarp/AllBroker), on shipped defaults: 1,262.5 tok/s aggregate at 16 concurrent
requests (vLLM 0.28: 546.6; SparkInfer v0.5.5: 345.1), 337.6 tok/s single stream (SparkInfer + DSpark v2: 264.8), and
479,954 tokens of int4 KV cache. 5-task benchmark mean: 93.3 with int8 KV, 92.9 with int4 KV (Q4_K_M 93.2, BF16
93.5; differences of a few tenths are inside each task's error bar).

![Total throughput vs. concurrent users](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/5090-aggregate.png)

This fork is tuned and tested for Qwen3.8-27B, EXL3 4.0 bpw, on the RTX 5090 (sm_120) with vLLM 0.26.0 and CUDA 13.
The other architectures exllamav3 supports come from upstream unchanged and are not tested here.

## What's different from upstream

`exl3cuda/` (every feature has its own `EXL3_*` switch; all of them, the kernel builds and the data files are in
[exl3cuda/README.md](exl3cuda/README.md); release notes in [exl3cuda/RELEASE.md](exl3cuda/RELEASE.md)):

| Feature | What it does | Measured effect |
|---|---|---|
| int4 KV attention (VA4, KVW4, I4FA) | verify / decode attention, fused KV write and prefill for `int4_per_token_head` KV | KV cache 479,954 tokens vs 264,241 with int8 KV (1.82x); KL vs BF16 0.0080 vs 0.0060 (int8 KV); 5-task mean 92.9 vs 93.3 (1); single-stream decode +14-16 % at 128k-190k context (2) |
| K smoothing (KSMOOTH, KS4) | stores K minus a per-channel mean in the quantized KV cache; softmax unchanged | int4 KV KL 0.0044 with a table vs 0.0051 without (2) |
| MTP drafting | pruned draft vocabulary, per-request draft rows from the prompt (DRAFT-DYN), 4-bit draft head, full CUDA graphs for every draft depth | single stream, tokens per step: chat 3.36, code 6.93, math 4.83, JSON 6.42 vs SparkInfer + DSpark v2's accept 2.32 / 6.78 / 4.36 / 4.60, at about the same steps per second (1) |
| DRAFT-GRAPH | MTP draft passes 2..k as one CUDA graph | step rate +14 % (1 stream), +10 % (2), +1 % (4); greedy output bit-identical (2) |
| GDN recompute, GDN prefill port | MTP with one committed SSM state per request; GDN prefill on the Magnitude WY kernel | not published separately |
| Prefill and decode GEMMs (INT8-PREFILL, fused decode, L2SPLIT, TAIL-FUSE) | W8A8 prefill GEMMs, fused decode GEMMs, bit-exact decode-tail trimming | not published separately; whole stack: an 8K-token prompt prefilled in about 1.1 s (1) |
| Weight-keeper callbacks | `whx_registry` / `whx_finish` so AllBroker's keeper can hold the weights across engine restarts | restart attaches 15 GiB of weights in about 3-5 s instead of a minute or more of loading (3) |

The exllamav3 library itself (upstream fea92fd7 plus three commits, each meant for upstream):

| Change | Measured effect (RTX 5090, Qwen3.8-27B) |
|---|---|
| fp16 MMA accumulation on sm_120 (upstream enables it on sm_86 only) | all linears, rows 17 / 32 / 48: 16.1 -> 15.1, 16.5 -> 15.4, 20.5 -> 17.7 ms; +6 % throughput at 8 concurrent requests; RMS error vs fp32 accumulation 0.07-0.17 % |
| gemv3: transposed block mapping for `down_proj` at m = 8 | 40.5 -> 36.6 us, bitwise identical output |
| mgemm `c_row_stride` / `c_base`: fused linears write into one concatenated output row | bitwise identical; removes the copy kernels after the GEMM |

Sources: (1) AllRunner [benchmarks/README.md](https://github.com/echowarp/AllRunner/blob/main/benchmarks/README.md);
(2) [exl3cuda/RELEASE.md](exl3cuda/RELEASE.md); (3) [AllBroker README](https://github.com/echowarp/AllBroker#readme);
library rows: the commit messages.

MTP drafting per workload, single stream (SparkInfer's harness shape: 256 tokens, greedy, thinking off; shipped
defaults, median of 3 reps):

![RTX 5090, single stream by workload](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/5090-single-stream.png)

int4 KV and EXL3 4.0 bpw against BF16, Q4_K_M and NVFP4: 5-task mean on the published rows' harness, KL vs BF16, and
GPQA Diamond / IFBench with thinking on (RTX 5090 and Arc B70 runs pooled):

![Q4_K_M-level quality](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/quality.png)

At the 32k thinking budget of the published NVFP4 runs (RTX 5090 production engine, int4 KV; GPQA 5 runs, AIME 4 runs,
mean ± 1 s.e.):

![Same 32k thinking budget as the published NVFP4 runs](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/quality-32k-vs-nvfp4.png)

## Use it

[AllRunner](https://github.com/echowarp/AllRunner) builds the RTX 5090 image from this repo, echowarp/vllm and AllBroker
at pinned commits, and starts it:

```sh
git clone https://github.com/echowarp/AllRunner && cd AllRunner
docker build -f docker/Dockerfile.rtx5090 -t allrunner-rtx5090 .     # about 70 min at MAX_JOBS=1; no GPU needed
launchers/rtx5090.sh        # engine on 127.0.0.1:8001
```

Read AllRunner's [launchers/README.md](https://github.com/echowarp/AllRunner/blob/main/launchers/README.md) first:
weights (`turboderp/Qwen3.8-27B-exl3`, branch `4.00bpw`), the required hash-seed file, start order, and the router in
front.

By hand, in an environment with vLLM 0.26.0 (CUDA 13) and `torch` already installed:

```sh
pip install --no-deps --no-build-isolation .     # exllamav3 and its extension (no prebuilt wheels for this fork)
# build the exl3cuda kernels you want: exl3cuda/README.md, "Building the kernels"
pip install --no-deps --no-build-isolation ./exl3cuda
vllm serve <EXL3 checkpoint dir> ...             # quant_method "exl3" in its config.json selects the plugin
```

Several features need the companion vLLM build (below); with stock vLLM they log a warning and keep the stock path.
CPU-only checks: `python -m pytest -q exl3cuda/tests`.

## Requires / works with

- [echowarp/vllm](https://github.com/echowarp/vllm): vLLM 0.26 with our scheduler, speculative-decoding and KV-offload
  commits and the extension hooks the plugin uses (CUDA build on its default branch).
- [echowarp/AllBroker](https://github.com/echowarp/AllBroker): the memory layer (weight keeper, KV lending, lazy vision
  tower, host-RAM KV tier shared with the Arc B70), and the no-mmap loader the plugin imports when installed. Optional.
- [echowarp/AllRunner](https://github.com/echowarp/AllRunner): images, launchers, router and benchmarks.
- Arc Pro B70 counterpart: [echowarp/exl3xpu](https://github.com/echowarp/exl3xpu).

## Upstream

exllamav3 is by turboderp: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). For everything else
(other models and architectures, quantizing with `convert.py`, the Python generator, TabbyAPI, installation from
wheels), see [upstream's README](https://github.com/turboderp-org/exllamav3#readme) and [doc/](doc/).

## License

MIT, as upstream ([LICENSE](LICENSE)). `exl3cuda/` is MIT ([exl3cuda/LICENSE](exl3cuda/LICENSE)); third-party code in
it keeps its own license ([exl3cuda/THIRD_PARTY_NOTICES.md](exl3cuda/THIRD_PARTY_NOTICES.md)).
