# gdnchunk: Magnitude's GDN prefill kernel, ported to vLLM's layouts (RTX 5090, sm_120)

Used by `exl3cuda/gdn_port.py` (`EXL3_GDN_PORT=1`) in place of vLLM's `causal_conv1d_fn` + `fused_post_conv_prep` +
FLA `chunk_gated_delta_rule` for the prefill rows of Qwen3.x GDN layers.

Source: github.com/magnitudedev/magnitude @ `7f029060a37ed1f7884db038222dc8f36430a08e`
(`inference/engine/kernels/kernels/cuda/gated_delta_chunk.cu`, `lib/recurrent/recurrent.cuh`). Apache-2.0: see
`magnitude/LICENSE` and `magnitude/NOTICE`; `gdn_chunk.cu`'s header lists the modifications.

| File | What |
|---|---|
| `gdn_chunk.cu` | The port: 4 kernels + torch binding (`forward`, `scratch_bytes`) |
| `magnitude/` | Unmodified Magnitude helpers (`cuda_prelude.cuh`: PTX wrappers for mma/ldmatrix/cp.async; `element.cuh`), `LICENSE`, `NOTICE` |
| `gdnext.py` | `load_ext()`: `torch.utils.cpp_extension.load`, sm_120, `-O3 -fmad=false` |
| `build.py` | Compile (no GPU needed, ~1 min) and copy the module to `../../lib/gdn_chunk_mag.so` |

## What was ported

Magnitude's CUDA `gated_delta_chunk` has three launches. The port keeps the first two almost line for line and drops
the third:
- **prepare**: one block per (16-row piece, key head): causal conv (C=4) + SiLU + L2 norm (q scaled by K^-1/2) of the
  piece's q/k and of the v rows of the key head's 3 value heads, in F32; K K^T and Q K^T in F32; per value head
  A = beta·exp(G_t−G_s)·KK^T, Tb = (I+A)^-1 diag(beta) by forward substitution, D = exp(G_t−G_s)·QK^T, and the gammas.
  Results go to a per-piece scratch record (q, k, k−f16(k), v, Tb, D as f16; gammas F32).
- **scan**: one block per (ROWS state rows, value head, sequence). The state lives in m16n8k16 accumulator registers
  (F32) for the whole sequence; pieces are applied in order on tensor cores, the next pieces staged with cp.async
  (3 stages). Per piece: U = Tb(V − diag(γ)K Sᵀ), O = diag(γ)Q Sᵀ + D U, S ← γ_last S + (U diag(γ_last/γ))ᵀ K.
- **gate** (RMSNormGated × act(z)): not ported; vLLM does it in `_output_projection`, outside the core op.

The Seismic ABI macros became a plain `Params` struct and compile-time NK/NV/W/C (`-DGDN_NK=...`, defaults
16/48/128/4).

## Argument contract (vLLM layouts, no transposes)

| | port | vLLM |
|---|---|---|
| qkv | `[M, CH]` f16, raw (before conv), CH = 10240 | `mixed_qkv` before `causal_conv1d_fn` |
| a, b | `[M, 48]` f16 | `a`, `b` |
| conv | `[CH, 4]` **f32** | `conv1d.weight.view(CH, 4)` (converted once at load) |
| rate, time_bias | `-exp(A_log)`, `dt_bias` f32 | `A_log`, `dt_bias` |
| state | `[S, 48, V, K]` f32 | `ssm_state` `[blocks, HV, V, K]` |
| window | `[S, 3, CH]` f16 | conv state, "SD" layout `[blocks, 3, CH]`, row 2 = newest |
| out | `[M, 48, 128]` f16 | `core_attn_out` |

Per sequence: `cu` (query start locations), `pos0` (absolute position of the first new row; pieces are cut at absolute
multiples of 16), `src` (state slot to resume from, −1 = zero state and window) and any number of checkpoints
`(pos, dst slot, kind)`: kind 0 "exact" (state after row `pos`, the piece is cut there; what decode needs at the prompt
end), kind 1 "aligned + tail" (state at the last 16-aligned row plus up to 15 raw tail rows; no cut, so resuming from it
is bit-identical to a one-shot prefill wherever the split falls). The scan reads its own state rows before writing
them and a separate last launch publishes windows and tails, so `dst == src` (vLLM's in-place slot use) is safe.

## Differences vs vLLM FLA

| | port | FLA (vLLM 0.26, Triton) |
|---|---|---|
| chunk | 16 rows (WY), Tb by F32 forward substitution, stored f16 | 64 rows; `solve_tril` → f16 |
| state in the chunk loop | F32 registers, never stored between pieces | every chunk's h stored as fp16 |
| state operand precision | split f16 high + low (~22 bits) | fp16 |
| conv + norm | fused, F32, one rounding to f16 | conv → fp16, then norm → fp16 |
| FMA | `-fmad=false` (as Magnitude) | Triton default |

## Measured (RTX 5090, per layer)

Accuracy vs fp64 (rel-RMS): out 4.3e-4 (FLA 6.9e-4), state 2.9e-4 (FLA 6.2e-4). Run-to-run, ROWS 32/64/128 and batch
invariant bit-identical; resume from aligned+tail checkpoints == one-shot (FLA split ≠ one-shot).

| case | FLA (conv+prep+chunk) | port ROWS=64 |
|---|---|---|
| tiny resume (5 rows) | 0.789 ms | 0.024 ms |
| 2k rows | 0.854 ms | 0.388 ms (2.20×) |
| 8k rows | 1.675 ms | 1.464 ms (1.14×) |

FLA has a fixed ~0.8 ms/layer cost (≈38 ms per step over 48 layers) for any prefill row; the port removes it, which
shows up as −20–50 % TTFT for prompts up to ~1k tokens in the engine (neutral at ≥ 2k).
