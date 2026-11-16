# exl3cuda (echowarp/exllamav3), 2026-11-16: first public release

Branch `publish-20261113`. Features and switches: `exl3cuda/README.md`.

## What is in it

- vLLM plugin serving EXL3 checkpoints on sm_120 (RTX 5090) with exllamav3's kernels plus this fork's: fused decode
  GEMMs, int8 / int4 per-token-head KV attention (VA / VA4, KVW4, I4FA prefill), GDN prefill port and recompute, MTP
  draft rows (DRAFT-DYN / DYN-ROW, TAIL-FUSE), FAST-LAUNCH (off by default).
- DRAFT-GRAPH (`EXL3_DRAFT_GRAPH=1`): MTP draft passes 2..k as one CUDA graph.
- Weight-keeper hooks (`whx_registry` / `whx_finish`): policy-free loader callbacks; the keeper itself is AllBroker's.

## How it was validated

- CPU tests (`exl3cuda/tests`): 20 passed.
- int4 KV vs int8 KV on the same RTX 5090 engine (`configs/int4.env`): KV capacity 479,954 vs 254,803 tokens;
  single-stream decode +14-16 % at 128k-190k context; KL vs an fp16-KV reference 0.0044.
- DRAFT-GRAPH vs the same build without it: step rate +14 % (1 stream), +10 % (2), +1 % (4), +11 % (long context, 1);
  greedy output bit-identical to eager; agent simulation 0 errors. In production since 2026-11-16.
- Production engine (this code, 2026-11-16, KV 482,846 tokens): aggregate decode 300.4 / 492.8 / 684.8 / 926.7 /
  1,285.2 / 1,359.9 tok/s; single stream 241.0 (chat), 470.1 (code), 322.0 (math), 442.3 (JSON) tok/s.

## Known limitations

- The numbers above used production's traffic-ranked draft vocabulary and its 4-bit draft head; with a public
  vocabulary (AllRunner `data/draft-vocab/`) and no re-quantized head (about -3 % decode) they are not measured yet.
- Two starts of the same engine are not bit-identical (4/6 greedy prompts identical, max |dlp| 0.091).
- No K-smoothing table ships (int4 KL 0.0051 without vs 0.0044 with, measured on agent tokens).
- Kernels are specialised for Qwen3.8-27B attention shapes on sm_120; other shapes fall back to vLLM's stock paths.
- The Docker image (AllRunner `docker/Dockerfile.rtx5090`) has not been rebuilt from the published commits yet.
