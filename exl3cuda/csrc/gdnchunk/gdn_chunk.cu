// SPDX-License-Identifier: Apache-2.0
//
// gdn_chunk: the gated delta rule prefill in the chunked WY form over pieces of 16 rows on tensor cores
// (f16 m16n8k16, F32 accumulation, state and keys as high + low f16 pairs), as a torch extension.
//
// Port of Magnitude's `gated_delta_chunk` CUDA kernel
//   inference/engine/kernels/kernels/cuda/gated_delta_chunk.cu and lib/recurrent/recurrent.cuh
//   (https://github.com/magnitudedev/magnitude, commit 7f029060a37ed1f7884db038222dc8f36430a08e)
// Copyright 2026 Magnitude AI Inc., licensed under the Apache License, Version 2.0 (csrc/magnitude/LICENSE).
// Modifications (vLLM layouts, RTX 5090), see README.md in this directory:
//  * Seismic ABI macros (SEISMIC_PTR, SEISMIC_DIM_*, strides, slab tables) replaced by a plain `Params` struct and
//    compile-time NK/NV/W/C (-DGDN_NK=...); the activation element is fixed to f16.
//  * Inputs in vLLM's layout: mixed_qkv [M, CH] (raw, before the convolution), a [M, NV], b [M, NV] separate;
//    conv weights f32 [CH, C]; rate = -exp(A_log), time_bias = dt_bias; SSM state f32 [S, NV, V, K] (vLLM's
//    [HV, V, K] per slot), conv window f16 [S, C-1, CH] (vLLM's "SD" conv-state layout); raw core output
//    [M, NV, W] (vLLM's core_attn_out). The output gate (RMSNormGated) is not part of this port.
//  * No tape, no MTP-verify sequential path (the row-sequential `advance_rows` is removed): every slot, also a
//    1-row one, goes through the WY pieces, so the bits of a row never depend on how its sequence was split.
//  * Partial-block reloads: pieces are aligned to ABSOLUTE token positions (pos0); a slot may resume from a
//    checkpoint that holds the state at an aligned position plus up to 15 raw "tail" rows, which the kernel
//    replays (no outputs) before the new rows; any number of checkpoints per slot, each published to its own
//    state slot, either "exact" (state at that row; the piece is cut there) or "aligned" (state at the last
//    16-aligned row + the raw tail rows up to the checkpoint; no cut, so the arithmetic is the one-shot run's).
//  * The successor conv window (and tail) is published by a separate launch, so in-place state slots are safe.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "magnitude/cuda_prelude.cuh"
#include "magnitude/element.cuh"

#ifndef GDN_NK
#define GDN_NK 16
#endif
#ifndef GDN_NV
#define GDN_NV 48
#endif
#ifndef GDN_W
#define GDN_W 128
#endif
#ifndef GDN_C
#define GDN_C 4
#endif

namespace gdn {

using element::u8;
using element::u16;
using element::u32;
using element::u64;
typedef element::F16 Act;
typedef long long i64;

constexpr int NK = GDN_NK;
constexpr int NV = GDN_NV;
constexpr int W = GDN_W;
constexpr int C = GDN_C;
constexpr int CH = (2 * NK + NV) * W;  // convolved channels q | k | v
constexpr int RAW = CH + 2 * NV;       // a raw (tail) row: q | k | v channels, alpha, beta
constexpr int PIECE = 16;
constexpr int TAIL = PIECE - 1;        // most raw rows an aligned checkpoint carries
constexpr int GROUP = NV / NK;
constexpr int PREPARE_THREADS = 256;
constexpr int PREPARE_WARPS = PREPARE_THREADS / 32;
constexpr int CPL = W / 32;  // channels per lane

struct Params {
    const u16 *qkv;    // [M, >= CH] f16, row stride qkv_stride
    i64 qkv_stride;
    const u16 *alpha;  // [M, NV] f16 (vLLM `a`)
    i64 alpha_stride;
    const u16 *beta;   // [M, NV] f16 (vLLM `b`)
    i64 beta_stride;
    const float *conv;       // [CH, C] f32
    const float *rate;       // [NV] f32: -exp(A_log)
    const float *time_bias;  // [NV] f32: dt_bias
    const int *cu;           // [B + 1] slot rows in qkv / out
    const int *pos0;         // [B] absolute position of the slot's first new row
    const int *src;          // [B] state slot to resume from (-1: zero state and window, no tail)
    const int *ck_cu;        // [B + 1] the slot's checkpoint entries
    const int *ck_pos;       // [n] absolute position (pos0 <= pos <= pos0 + rows)
    const int *ck_dst;       // [n] state slot the checkpoint is published to
    const int *ck_kind;      // [n] 0 = exact, 1 = aligned + tail
    float *state;            // [S, NV, W(v), W(k)] f32
    i64 st_slot, st_head, st_row;
    u16 *window;             // [S, C - 1, CH] f16, channel stride 1
    i64 win_slot, win_tap;
    u16 *tail;               // [S, TAIL, RAW] f16
    i64 tail_slot;
    int *tail_count;         // [S] raw tail rows of the slot's checkpoint
    u16 *out;                // [M, NV, W] f16
    // launch scratch
    int *slot_start;         // [B] absolute first row the scan processes (pos0 - tail rows of src)
    int *piece_cu;           // [B + 1]
    int4 *pieces;            // [max_pieces] (slot, first absolute row, count, -)
    u8 *records;             // [max_pieces] * PIECE_BYTES
    int B;
    int max_pieces;
    float epsilon;
    int grouped;
};

// ------------------------------------------------------------------------------------------------------------
// Slots, raw rows, gates
// ------------------------------------------------------------------------------------------------------------

struct Slot {
    int lo, hi;      // rows in qkv / out
    int pos0;        // absolute position of row lo
    int start;       // absolute first processed row (pos0 - replayed tail rows)
    int end;         // absolute end: pos0 + hi - lo
    int src;
    int ck_lo, ck_hi;
};

__device__ __forceinline__ Slot slot_of(const Params &p, int s) {
    Slot r;
    r.lo = p.cu[s];
    r.hi = p.cu[s + 1];
    r.pos0 = p.pos0[s];
    r.start = p.slot_start[s];
    r.end = r.pos0 + (r.hi - r.lo);
    r.src = p.src[s];
    r.ck_lo = p.ck_cu[s];
    r.ck_hi = p.ck_cu[s + 1];
    return r;
}

// The absolute row a checkpoint's state is taken at: the row itself (exact) or the last 16-aligned row at or
// before it, but not before the slot's first processed row (aligned + tail).
__device__ __forceinline__ int checkpoint_base(const Slot &s, int pos, int kind) {
    return kind == 0 ? pos : max((pos / PIECE) * PIECE, s.start);
}

// CPL raw channels from `channel` of absolute row x (x >= start - (C - 1)): new rows from qkv, replayed rows from
// the source slot's tail, earlier rows from the source slot's window (zero without a source).
template <bool NC>
__device__ __forceinline__ void raw_span(const Params &p, const Slot &s, int x, int channel, float (&out)[CPL]) {
    if (x >= s.pos0) {
        element::span<Act, CPL, NC>(reinterpret_cast<const u8 *>(p.qkv),
                                    static_cast<u64>(s.lo + x - s.pos0) * p.qkv_stride + channel, out);
    } else if (s.src < 0) {
#pragma unroll
        for (int c = 0; c < CPL; ++c) out[c] = 0.0f;
    } else if (x >= s.start) {
        element::span<Act, CPL, NC>(reinterpret_cast<const u8 *>(p.tail),
                                    static_cast<u64>(s.src) * p.tail_slot +
                                        static_cast<u64>(x - s.start) * RAW + channel,
                                    out);
    } else {
        element::span<Act, CPL, NC>(reinterpret_cast<const u8 *>(p.window),
                                    static_cast<u64>(s.src) * p.win_slot +
                                        static_cast<u64>(C - 1 - (s.start - x)) * p.win_tap + channel,
                                    out);
    }
}

// The raw bits of column `column` (< RAW) of absolute row x, as above (the window holds only the CH channels).
__device__ __forceinline__ u16 raw_bits(const Params &p, const Slot &s, int x, int column) {
    if (x >= s.pos0) {
        const u64 row = s.lo + x - s.pos0;
        if (column < CH) return p.qkv[row * p.qkv_stride + column];
        if (column < CH + NV) return p.alpha[row * p.alpha_stride + column - CH];
        return p.beta[row * p.beta_stride + column - CH - NV];
    }
    if (s.src < 0) return 0;
    if (x >= s.start) return p.tail[static_cast<u64>(s.src) * p.tail_slot + static_cast<u64>(x - s.start) * RAW + column];
    if (column >= CH) return 0;
    return p.window[static_cast<u64>(s.src) * p.win_slot + static_cast<u64>(C - 1 - (s.start - x)) * p.win_tap + column];
}

// beta = sigmoid(b) and log_decay = rate * softplus(alpha + time_bias).
struct Gates {
    float beta;
    float log_decay;
};
__device__ __forceinline__ Gates gates(const Params &p, const Slot &s, int x, int head) {
    const float alpha = seismic_f16_to_f32(raw_bits(p, s, x, CH + head));
    const float beta_input = seismic_f16_to_f32(raw_bits(p, s, x, CH + NV + head));
    const float shifted = alpha + p.time_bias[head];
    const float softplus = fmaxf(shifted, 0.0f) + logf(1.0f + expf(-fabsf(shifted)));
    return Gates{1.0f / (1.0f + expf(-beta_input)), p.rate[head] * softplus};
}

__device__ __forceinline__ int key_head(const Params &p, int head) { return p.grouped ? head * NK / NV : head % NK; }
__device__ __forceinline__ int member_head(const Params &p, int key, int member) {
    return p.grouped ? key * GROUP + member : key + member * NK;
}

__device__ __forceinline__ float *state_row(const Params &p, int slot, int head, int row) {
    return p.state + static_cast<u64>(slot) * p.st_slot + static_cast<u64>(head) * p.st_head +
           static_cast<u64>(row) * p.st_row;
}

// ------------------------------------------------------------------------------------------------------------
// L0 gdn_pieces: the piece table. Pieces of a slot cover [start, end); they are cut at every absolute multiple of
// 16 and at every exact checkpoint. One block; one thread per slot.
// ------------------------------------------------------------------------------------------------------------

template <bool WRITE>
__device__ __forceinline__ int enumerate_pieces(const Params &p, const Slot &s, int s_index, int base) {
    int n = 0;
    for (int cur = s.start; cur < s.end;) {
        int next = min(s.end, (cur / PIECE + 1) * PIECE);
        for (int k = s.ck_lo; k < s.ck_hi; ++k) {
            const int pos = p.ck_pos[k];
            if (p.ck_kind[k] == 0 && pos > cur && pos < next) next = pos;
        }
        if (WRITE && base + n < p.max_pieces) p.pieces[base + n] = make_int4(s_index, cur, next - cur, 0);
        ++n;
        cur = next;
    }
    return n;
}

__global__ void gdn_pieces(Params p) {
    __shared__ int offsets[256];
    __shared__ int total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    for (int s0 = 0; s0 < p.B; s0 += blockDim.x) {
        const int s = s0 + threadIdx.x;
        int count = 0;
        Slot slot{};
        if (s < p.B) {
            const int src = p.src[s];
            p.slot_start[s] = p.pos0[s] - (src >= 0 ? p.tail_count[src] : 0);
            slot = Slot{p.cu[s], p.cu[s + 1], p.pos0[s], p.slot_start[s], p.pos0[s] + p.cu[s + 1] - p.cu[s], src,
                        p.ck_cu[s], p.ck_cu[s + 1]};
            count = enumerate_pieces<false>(p, slot, s, 0);
        }
        offsets[threadIdx.x] = count;
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = total;
            for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
                const int c = offsets[i];
                offsets[i] = acc;
                acc += c;
            }
            total = acc;
        }
        __syncthreads();
        if (s < p.B) {
            p.piece_cu[s] = offsets[threadIdx.x];
            enumerate_pieces<true>(p, slot, s, offsets[threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) p.piece_cu[p.B] = min(total, p.max_pieces);
}

// ------------------------------------------------------------------------------------------------------------
// L1 gdn_chunk_prepare (Magnitude's gated_delta_chunk_prepare), one block per (piece, key head).
// ------------------------------------------------------------------------------------------------------------

constexpr int PARTS = 2 + GROUP;
__host__ __device__ constexpr int prepare_row_run() {
    int best = PIECE;
    for (int run = PIECE; run >= 1; run /= 2) {
        const int runs = PARTS * (PIECE / run);
        const int rows = (runs + PREPARE_WARPS - 1) / PREPARE_WARPS * run;
        const int best_runs = PARTS * (PIECE / best);
        if (rows < (best_runs + PREPARE_WARPS - 1) / PREPARE_WARPS * best) best = run;
    }
    return best;
}
constexpr int ROW_RUN = prepare_row_run();
constexpr int RUNS = PIECE / ROW_RUN;
static_assert(NV % NK == 0 && W % 32 == 0, "chunk geometry");
static_assert(PIECE * PIECE == PREPARE_THREADS, "one prepare thread per (t, s) pair");
static_assert(GROUP <= 8, "the prepare block's static shared memory holds up to eight value heads per key head");

constexpr int QK_WIDTH = 3 * W;
constexpr u64 QK_BYTES = static_cast<u64>(NK) * PIECE * QK_WIDTH * 2;
constexpr u64 RECORD_V = 0;                                 // f16 [16][W]
constexpr u64 RECORD_TB = static_cast<u64>(PIECE) * W * 2;  // f16 [16][16]
constexpr u64 RECORD_D = RECORD_TB + PIECE * PIECE * 2;     // f16 [16][16]
constexpr u64 RECORD_GAMMA = RECORD_D + PIECE * PIECE * 2;  // f32 [16]
constexpr u64 RECORD_TAIL = RECORD_GAMMA + PIECE * 4;       // f32 [16]
constexpr u64 RECORD_LAST = RECORD_TAIL + PIECE * 4;        // f32, 3 pad
constexpr u64 RECORD_BYTES = RECORD_LAST + 16;
constexpr u64 PIECE_BYTES = QK_BYTES + NV * RECORD_BYTES;
static_assert(RECORD_BYTES % 16 == 0 && QK_BYTES % 16 == 0, "16-byte scratch records");

__device__ __forceinline__ u8 *qk_rows(u8 *records, int piece, int key) {
    return records + static_cast<u64>(piece) * PIECE_BYTES + static_cast<u64>(key) * PIECE * QK_WIDTH * 2;
}
__device__ __forceinline__ u8 *record(u8 *records, int piece, int head) {
    return records + static_cast<u64>(piece) * PIECE_BYTES + QK_BYTES + static_cast<u64>(head) * RECORD_BYTES;
}

__device__ __forceinline__ u32 pack(float lo, float hi) { return seismic_pack_f16x2(lo, hi); }

struct Split {
    u32 high;
    u32 low;
};
__device__ __forceinline__ Split split(float first, float second) {
    const u32 high = pack(first, second);
    const float2 rounded = seismic_unpack_f16x2(high);
    return Split{high, pack(first - rounded.x, second - rounded.y)};
}
__device__ __forceinline__ void split_tiles(const float (&left)[4], const float (&right)[4], u32 (&high)[4],
                                            u32 (&low)[4]) {
    const Split parts[4] = {split(left[0], left[1]), split(left[2], left[3]), split(right[0], right[1]),
                            split(right[2], right[3])};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        high[i] = parts[i].high;
        low[i] = parts[i].low;
    }
}

__global__ void __launch_bounds__(PREPARE_THREADS) gdn_chunk_prepare(Params p) {
    if (static_cast<int>(blockIdx.x) >= p.piece_cu[p.B]) return;
    const int4 entry = p.pieces[blockIdx.x];
    const Slot slot = slot_of(p, entry.x);
    const int piece_first = entry.y;  // absolute
    const int n = entry.z;
    const int key = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    u8 *records = p.records;

    __shared__ float queries[PIECE][W + 1];
    __shared__ float keys[PIECE][W + 1];
    __shared__ float key_gram[PIECE][PIECE + 1];
    __shared__ float query_gram[PIECE][PIECE + 1];
    __shared__ float cumulative[GROUP][PIECE];
    __shared__ float beta[GROUP][PIECE];
    __shared__ float system[GROUP][PIECE][PIECE + 1];

    u16 *qk = reinterpret_cast<u16 *>(qk_rows(records, blockIdx.x, key));
    for (int item = warp; item < PARTS * RUNS; item += PREPARE_WARPS) {
        const int part = item % PARTS;
        const int first = ROW_RUN * (item / PARTS);
        const int column = lane * CPL;
        const int channel = (part == 0   ? key * W
                             : part == 1 ? (NK + key) * W
                                         : (2 * NK + member_head(p, key, part - 2)) * W) +
                            column;
        float weights[C][CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            float taps[C];
            element::f32_span<C>(p.conv + static_cast<u64>(channel + c) * C, taps);
#pragma unroll
            for (int tap = 0; tap < C; ++tap) weights[tap][c] = taps[tap];
        }
        const int x0 = piece_first + first;  // absolute row of the run
        float inputs[C][CPL];
#pragma unroll
        for (int tap = 0; tap < C - 1; ++tap) {
            if (first < n) {
                raw_span<true>(p, slot, x0 + tap - (C - 1), channel, inputs[tap]);
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[tap][c] = 0.0f;
            }
        }
        u16 *v = part < 2 ? nullptr
                          : reinterpret_cast<u16 *>(record(records, blockIdx.x, member_head(p, key, part - 2)) +
                                                    RECORD_V);
        const float scale = part == 0 ? rsqrtf(static_cast<float>(W)) : 1.0f;
#pragma unroll
        for (int i = 0; i < ROW_RUN; ++i) {
            const int t = first + i;
            if (t < n) {
                raw_span<true>(p, slot, x0 + i, channel, inputs[C - 1]);
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[C - 1][c] = 0.0f;
            }
            float values[CPL];
            float squares = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                float sum = weights[C - 1][c] * inputs[C - 1][c];
#pragma unroll
                for (int tap = 0; tap < C - 1; ++tap) sum = __fmaf_rn(weights[tap][c], inputs[tap][c], sum);
                values[c] = t < n ? sum / (1.0f + expf(-sum)) : 0.0f;
                squares = __fmaf_rn(values[c], values[c], squares);
            }
#pragma unroll
            for (int tap = 0; tap < C - 1; ++tap)
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[tap][c] = inputs[tap + 1][c];
            if (part < 2) {
                const float inverse = t < n ? rsqrtf(seismic_warp_sum_f32(squares) + p.epsilon) * scale : 0.0f;
                float (*rows)[W + 1] = part == 0 ? queries : keys;
#pragma unroll
                for (int c = 0; c < CPL; ++c) {
                    values[c] *= inverse;
                    rows[t][column + c] = values[c];
                    const u16 high = seismic_f32_to_f16(values[c]);
                    qk[t * QK_WIDTH + part * W + column + c] = high;
                    if (part == 1)
                        qk[t * QK_WIDTH + 2 * W + column + c] = seismic_f32_to_f16(values[c] - seismic_f16_to_f32(high));
                }
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) v[t * W + column + c] = seismic_f32_to_f16(values[c]);
            }
        }
    }
    if (threadIdx.x < GROUP * PIECE) {
        const int member = threadIdx.x / PIECE;
        const int t = threadIdx.x % PIECE;
        float log_decay = 0.0f, b = 0.0f;
        if (t < n) {
            const Gates g = gates(p, slot, piece_first + t, member_head(p, key, member));
            log_decay = g.log_decay;
            b = g.beta;
        }
        cumulative[member][t] = log_decay;
        beta[member][t] = b;
    }
    __syncthreads();

    {
        const int t = threadIdx.x / PIECE;
        const int s = threadIdx.x % PIECE;
        float kk[2] = {0.0f, 0.0f}, qk_dot[2] = {0.0f, 0.0f};
#pragma unroll 8
        for (int c = 0; c < W; c += 2) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                kk[h] = __fmaf_rn(keys[t][c + h], keys[s][c + h], kk[h]);
                qk_dot[h] = __fmaf_rn(queries[t][c + h], keys[s][c + h], qk_dot[h]);
            }
        }
        key_gram[t][s] = kk[0] + kk[1];
        query_gram[t][s] = qk_dot[0] + qk_dot[1];
    }
    if (threadIdx.x < GROUP) {
        float sum = 0.0f;
        for (int t = 0; t < PIECE; ++t) {
            sum += cumulative[threadIdx.x][t];
            cumulative[threadIdx.x][t] = sum;
        }
    }
    __syncthreads();

    {
        const int t = threadIdx.x / PIECE;
        const int s = threadIdx.x % PIECE;
#pragma unroll
        for (int member = 0; member < GROUP; ++member) {
            const float *G = cumulative[member];
            const float decay = s <= t ? expf(G[t] - G[s]) : 0.0f;
            system[member][t][s] = s < t ? beta[member][t] * decay * key_gram[t][s] : 0.0f;
            u8 *out = record(records, blockIdx.x, member_head(p, key, member));
            reinterpret_cast<u16 *>(out + RECORD_D)[t * PIECE + s] =
                seismic_f32_to_f16(t < n ? decay * query_gram[t][s] : 0.0f);
            if (threadIdx.x < PIECE) {
                const float last = G[PIECE - 1];
                reinterpret_cast<float *>(out + RECORD_GAMMA)[s] = expf(G[s]);
                reinterpret_cast<float *>(out + RECORD_TAIL)[s] = expf(last - G[s]);
                if (s == 0) reinterpret_cast<float *>(out + RECORD_LAST)[0] = expf(last);
            }
        }
    }
    __syncthreads();

    for (int member = warp; member < GROUP; member += PREPARE_WARPS) {
        if (lane >= PIECE) continue;
        u16 *tb = reinterpret_cast<u16 *>(record(records, blockIdx.x, member_head(p, key, member)) + RECORD_TB);
        float x[PIECE];
#pragma unroll
        for (int i = 0; i < PIECE; ++i) {
            float value = i == lane ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < i; ++k) value = __fmaf_rn(-system[member][i][k], x[k], value);
            x[i] = value;
        }
        const float b = beta[member][lane];
#pragma unroll
        for (int i = 0; i < PIECE; ++i) tb[i * PIECE + lane] = seismic_f32_to_f16(x[i] * b);
    }
}

// ------------------------------------------------------------------------------------------------------------
// L2 gdn_chunk_scan (Magnitude's gated_delta_chunk_scan), one block per (ROWS state rows, value head, slot).
// ------------------------------------------------------------------------------------------------------------

constexpr int QK_PITCH = QK_WIDTH + 8;
template <unsigned ROWS>
struct Stage {
    static constexpr int V_PITCH = ROWS + 8;
    u16 qk[PIECE * QK_PITCH];
    u16 v[PIECE * V_PITCH];
    u16 td[2 * PIECE * PIECE];
    float gates[2 * PIECE + 4];
};
constexpr int STAGES = 3;

template <unsigned ROWS>
constexpr size_t scan_shared_bytes() {
    return STAGES * sizeof(Stage<ROWS>) + ROWS / 16 * 16 * 17 * 4;
}

__device__ __forceinline__ void store_row8(u16 *base, u64 index, const float (&values)[8]) {
    u32 words[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) words[k] = Act::pack2(values[2 * k], values[2 * k + 1]);
    *reinterpret_cast<uint4 *>(base + index) = make_uint4(words[0], words[1], words[2], words[3]);
}

template <unsigned ROWS>
__device__ __forceinline__ void stage_piece(Stage<ROWS> &stage, u8 *records, int piece, int key, int head,
                                            int row0) {
    constexpr int V_PITCH = Stage<ROWS>::V_PITCH;
    constexpr int SCAN_THREADS = ROWS * 2;
    const u8 *qk = qk_rows(records, piece, key);
    const u8 *rec_bytes = record(records, piece, head);
    constexpr int QK_CHUNKS = PIECE * QK_WIDTH * 2 / 16;
    constexpr int V_CHUNKS = PIECE * ROWS * 2 / 16;
    constexpr int TD_CHUNKS = 2 * PIECE * PIECE * 2 / 16;
    constexpr int GATE_CHUNKS = (2 * PIECE + 4) * 4 / 16;
    for (int index = threadIdx.x; index < QK_CHUNKS + V_CHUNKS + TD_CHUNKS + GATE_CHUNKS; index += SCAN_THREADS) {
        if (index < QK_CHUNKS) {
            const int t = index / (QK_WIDTH / 8);
            const int c = 8 * (index % (QK_WIDTH / 8));
            seismic_cp_async_16(stage.qk + t * QK_PITCH + c, qk + (t * QK_WIDTH + c) * 2);
        } else if (index < QK_CHUNKS + V_CHUNKS) {
            const int local = index - QK_CHUNKS;
            const int t = local / (ROWS / 8);
            const int c = 8 * (local % (ROWS / 8));
            seismic_cp_async_16(stage.v + t * V_PITCH + c, rec_bytes + RECORD_V + (static_cast<u64>(t) * W + row0 + c) * 2);
        } else if (index < QK_CHUNKS + V_CHUNKS + TD_CHUNKS) {
            const int local = index - QK_CHUNKS - V_CHUNKS;
            seismic_cp_async_16(stage.td + 8 * local, rec_bytes + RECORD_TB + 16 * local);
        } else {
            const int local = index - QK_CHUNKS - V_CHUNKS - TD_CHUNKS;
            seismic_cp_async_16(stage.gates + 4 * local, rec_bytes + RECORD_GAMMA + 16 * local);
        }
    }
}

template <unsigned ROWS>
__global__ void __launch_bounds__(ROWS * 2) gdn_chunk_scan(Params p) {
    constexpr int V_PITCH = Stage<ROWS>::V_PITCH;
    static_assert(ROWS % 16 == 0 && W % ROWS == 0, "chunk scan geometry");
    static_assert(sizeof(Stage<ROWS>) == 96 * W + 32 * ROWS + 1680, "Magnitude's declared shared_bytes");
    u8 *records = p.records;
    const int head = blockIdx.y;
    const int row0 = blockIdx.x * ROWS;
    const int slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t4 = lane % 4;
    const int warp_row = row0 + 16 * warp;
    const Slot slot = slot_of(p, slot_index);
    extern __shared__ __align__(16) unsigned char scan_shared[];
    const int key = key_head(p, head);

    // The warp's 16 state rows as accumulators: tile j holds S[row g, g + 8][key columns 8j + 2t4, + 1].
    float state[W / 8][4];
    if (slot.src >= 0) {
        const float *upper = state_row(p, slot.src, head, warp_row + g);
        const float *lower = state_row(p, slot.src, head, warp_row + g + 8);
#pragma unroll
        for (int j = 0; j < W / 8; ++j) {
            const float2 top = *reinterpret_cast<const float2 *>(upper + 8 * j + 2 * t4);
            const float2 bottom = *reinterpret_cast<const float2 *>(lower + 8 * j + 2 * t4);
            state[j][0] = top.x;
            state[j][1] = top.y;
            state[j][2] = bottom.x;
            state[j][3] = bottom.y;
        }
    } else {
#pragma unroll
        for (int j = 0; j < W / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) state[j][e] = 0.0f;
    }
    // Publish the state to every checkpoint whose base row is `boundary` (an absolute piece boundary).
    auto publish = [&](int boundary) {
        for (int k = slot.ck_lo; k < slot.ck_hi; ++k) {
            if (checkpoint_base(slot, p.ck_pos[k], p.ck_kind[k]) != boundary) continue;
            const int dst = p.ck_dst[k];
            float *upper = state_row(p, dst, head, warp_row + g);
            float *lower = state_row(p, dst, head, warp_row + g + 8);
#pragma unroll
            for (int j = 0; j < W / 8; ++j) {
                *reinterpret_cast<float2 *>(upper + 8 * j + 2 * t4) = make_float2(state[j][0], state[j][1]);
                *reinterpret_cast<float2 *>(lower + 8 * j + 2 * t4) = make_float2(state[j][2], state[j][3]);
            }
        }
    };
    publish(slot.start);

    Stage<ROWS> *stages = reinterpret_cast<Stage<ROWS> *>(scan_shared);
    float (*transpose)[17] = reinterpret_cast<float (*)[17]>(scan_shared + STAGES * sizeof(Stage<ROWS>)) + 16 * warp;
    const int base = p.piece_cu[slot_index];
    const int count = p.piece_cu[slot_index + 1] - base;
#pragma unroll
    for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
        if (ahead < count) stage_piece<ROWS>(stages[ahead], records, base + ahead, key, head, row0);
        seismic_cp_async_commit();
    }
    for (int local = 0; local < count; ++local) {
        const int4 piece = p.pieces[base + local];
        const Stage<ROWS> &stage = stages[local % STAGES];
        const int ahead = local + STAGES - 1;
        if (ahead < count) stage_piece<ROWS>(stages[ahead % STAGES], records, base + ahead, key, head, row0);
        seismic_cp_async_commit();
        seismic_cp_async_wait<STAGES - 1>();
        __syncthreads();
        const u16 *queries = stage.qk;
        const u16 *keys = stage.qk + W;
        const u16 *keys_low = stage.qk + 2 * W;

        float removed[2][4] = {}, removed_low[2][4] = {}, removed_key_low[2][4] = {};
        float output[2][4] = {};
#pragma unroll
        for (int step = 0; step < W / 16; ++step) {
            u32 a[4], a_low[4];
            split_tiles(state[2 * step], state[2 * step + 1], a, a_low);
            const int t = (lane % 8) + 8 * (lane / 16);
            const int c = 16 * step + 8 * ((lane / 8) % 2);
            u32 kb[4], qb[4], lb[4];
            seismic_ldmatrix_x4(kb, keys + t * QK_PITCH + c);
            seismic_ldmatrix_x4(qb, queries + t * QK_PITCH + c);
            seismic_ldmatrix_x4(lb, keys_low + t * QK_PITCH + c);
            const u32 k0[2] = {kb[0], kb[1]}, k1[2] = {kb[2], kb[3]};
            const u32 l0[2] = {lb[0], lb[1]}, l1[2] = {lb[2], lb[3]};
            const u32 q0[2] = {qb[0], qb[1]}, q1[2] = {qb[2], qb[3]};
            seismic_mma_m16n8k16_f16(removed[0], a, k0);
            seismic_mma_m16n8k16_f16(removed[1], a, k1);
            seismic_mma_m16n8k16_f16(output[0], a, q0);
            seismic_mma_m16n8k16_f16(output[1], a, q1);
            seismic_mma_m16n8k16_f16(removed_low[0], a_low, k0);
            seismic_mma_m16n8k16_f16(removed_low[1], a_low, k1);
            seismic_mma_m16n8k16_f16(removed_key_low[0], a, l0);
            seismic_mma_m16n8k16_f16(removed_key_low[1], a, l1);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) removed[j][e] += removed_low[j][e] + removed_key_low[j][e];

        // Z^T = V^T - X^T diag(gamma); U^T = Z^T Tb^T; O^T = Y^T diag(gamma) + U^T D^T.
        const float *gamma = stage.gates;
        const float *tail = stage.gates + PIECE;
        const float last = stage.gates[2 * PIECE];
        const int local_row = 16 * warp + g;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * j + 2 * t4 + (e & 1);
                const int r = local_row + 8 * (e / 2);
                const float v = seismic_f16_to_f32(stage.v[t * V_PITCH + r]);
                removed[j][e] = v - removed[j][e] * gamma[t];
                output[j][e] *= gamma[t];
            }
        }
        u32 za[4], za_low[4];
        split_tiles(removed[0], removed[1], za, za_low);
        float update[2][4] = {};
        const u32 *td = reinterpret_cast<const u32 *>(stage.td);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const u32 tb[2] = {td[((8 * j + g) * PIECE + 2 * t4) / 2], td[((8 * j + g) * PIECE + 2 * t4 + 8) / 2]};
            seismic_mma_m16n8k16_f16(update[j], za, tb);
            seismic_mma_m16n8k16_f16(update[j], za_low, tb);
        }
        const u32 ua[4] = {pack(update[0][0], update[0][1]), pack(update[0][2], update[0][3]),
                           pack(update[1][0], update[1][1]), pack(update[1][2], update[1][3])};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const u32 d[2] = {td[(PIECE * PIECE + (8 * j + g) * PIECE + 2 * t4) / 2],
                              td[(PIECE * PIECE + (8 * j + g) * PIECE + 2 * t4 + 8) / 2]};
            seismic_mma_m16n8k16_f16(output[j], ua, d);
#pragma unroll
            for (int e = 0; e < 4; ++e) transpose[8 * j + 2 * t4 + (e & 1)][g + 8 * (e / 2)] = output[j][e];
        }
        __syncwarp();
        {
            const int t = lane / 2;
            const int x = piece.y + t;  // absolute row
            if (t < piece.z && x >= slot.pos0) {
                float values[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) values[k] = transpose[t][8 * (lane % 2) + k];
                const u64 row = slot.lo + x - slot.pos0;
                store_row8(p.out, (row * NV + head) * W + warp_row + 8 * (lane % 2), values);
            }
        }

        // S <- gamma_last S + (U diag(tail))^T K.
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) update[j][e] *= tail[8 * j + 2 * t4 + (e & 1)];
        u32 wa[4], wa_low[4];
        split_tiles(update[0], update[1], wa, wa_low);
#pragma unroll
        for (int pair = 0; pair < W / 16; ++pair) {
            const int t = (lane % 8) + 8 * ((lane / 8) % 2);
            const int c = 16 * pair + 8 * (lane / 16);
            u32 kb[4], lb[4];
            seismic_ldmatrix_x4_trans(kb, keys + t * QK_PITCH + c);
            seismic_ldmatrix_x4_trans(lb, keys_low + t * QK_PITCH + c);
            const u32 k0[2] = {kb[0], kb[1]}, k1[2] = {kb[2], kb[3]};
            const u32 l0[2] = {lb[0], lb[1]}, l1[2] = {lb[2], lb[3]};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                state[2 * pair][e] *= last;
                state[2 * pair + 1][e] *= last;
            }
            seismic_mma_m16n8k16_f16(state[2 * pair], wa, k0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa, k1);
            seismic_mma_m16n8k16_f16(state[2 * pair], wa_low, k0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa_low, k1);
            seismic_mma_m16n8k16_f16(state[2 * pair], wa, l0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa, l1);
        }
        publish(piece.y + piece.z);
        __syncthreads();
    }
    seismic_cp_async_wait<0>();
}

// ------------------------------------------------------------------------------------------------------------
// L3 gdn_publish_window: per checkpoint, the conv window at its base row (raw rows base - (C-1) .. base - 1) and,
// for an aligned checkpoint, the raw tail rows base .. pos - 1 (q | k | v | alpha | beta) and the tail count.
// One thread per (raw column, slot); the thread reads everything a checkpoint needs before it writes, and handles
// a checkpoint that overwrites the slot's own source last, so in-place state slots are safe.
// ------------------------------------------------------------------------------------------------------------

__global__ void gdn_publish_window(Params p) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= RAW) return;
    const Slot slot = slot_of(p, blockIdx.y);
    for (int pass = 0; pass < 2; ++pass) {
        for (int k = slot.ck_lo; k < slot.ck_hi; ++k) {
            const int dst = p.ck_dst[k];
            if ((dst == slot.src) != (pass == 1)) continue;
            const int pos = p.ck_pos[k];
            const int kind = p.ck_kind[k];
            const int base = checkpoint_base(slot, pos, kind);
            const int tail_rows = kind == 0 ? 0 : pos - base;
            u16 win[C - 1];
            u16 tail[TAIL];
#pragma unroll
            for (int r = 0; r < C - 1; ++r) win[r] = column < CH ? raw_bits(p, slot, base - (C - 1) + r, column) : 0;
#pragma unroll
            for (int r = 0; r < TAIL; ++r) tail[r] = r < tail_rows ? raw_bits(p, slot, base + r, column) : 0;
            if (column < CH) {
#pragma unroll
                for (int r = 0; r < C - 1; ++r)
                    p.window[static_cast<u64>(dst) * p.win_slot + static_cast<u64>(r) * p.win_tap + column] = win[r];
            }
            if (p.tail != nullptr) {
#pragma unroll
                for (int r = 0; r < TAIL; ++r)
                    if (r < tail_rows) p.tail[static_cast<u64>(dst) * p.tail_slot + static_cast<u64>(r) * RAW + column] = tail[r];
            }
            if (column == 0 && p.tail_count != nullptr) p.tail_count[dst] = tail_rows;
        }
    }
}

}  // namespace gdn

// ------------------------------------------------------------------------------------------------------------
// Host
// ------------------------------------------------------------------------------------------------------------

namespace {

using gdn::i64;

#define CHECK_DEV(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).scalar_type() == (dt), #t ": wrong device or dtype")
#define CHECK_ALIGN(t, bytes) \
    TORCH_CHECK(reinterpret_cast<uintptr_t>((t).data_ptr()) % (bytes) == 0, #t ": not " #bytes "-byte aligned")

template <unsigned ROWS>
void launch_scan(const gdn::Params &p, cudaStream_t stream) {
    constexpr size_t smem = gdn::scan_shared_bytes<ROWS>();
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn::gdn_chunk_scan<ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        configured = true;
    }
    gdn::gdn_chunk_scan<ROWS><<<dim3(gdn::W / ROWS, gdn::NV, p.B), ROWS * 2, smem, stream>>>(p);
}

// Upper bound on the pieces of a launch (no device sync): per slot ceil((rows + 15 tail) / 16) + 1 for a misaligned
// start, plus one cut per exact checkpoint (generous rounding).
int64_t max_pieces(int64_t M, int64_t B, int64_t checkpoints) {
    return (M + 2 * gdn::TAIL * B) / gdn::PIECE + 3 * B + checkpoints + 1;
}

}  // namespace

int64_t scratch_bytes(int64_t M, int64_t B, int64_t checkpoints) {
    const int64_t P = max_pieces(M, B, checkpoints);
    const int64_t tables = ((B + (B + 1)) * 4 + 255) / 256 * 256 + (P * 16 + 255) / 256 * 256;
    return tables + P * static_cast<int64_t>(gdn::PIECE_BYTES);
}

// Prefill of B slots (sequences) in one launch sequence. See README.md (this directory) for the argument contract.
void forward(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor conv, torch::Tensor rate,
             torch::Tensor time_bias, torch::Tensor cu, torch::Tensor pos0, torch::Tensor src, torch::Tensor ck_cu,
             torch::Tensor ck_pos, torch::Tensor ck_dst, torch::Tensor ck_kind, torch::Tensor state,
             torch::Tensor window, torch::Tensor tail, torch::Tensor tail_count, torch::Tensor out,
             torch::Tensor scratch, double epsilon, bool grouped, int64_t rows) {
    using namespace gdn;
    const c10::cuda::OptionalCUDAGuard guard(qkv.device());
    CHECK_DEV(qkv, at::kHalf);
    CHECK_DEV(a, at::kHalf);
    CHECK_DEV(b, at::kHalf);
    CHECK_DEV(conv, at::kFloat);
    CHECK_DEV(rate, at::kFloat);
    CHECK_DEV(time_bias, at::kFloat);
    for (auto *t : {&cu, &pos0, &src, &ck_cu, &ck_pos, &ck_dst, &ck_kind, &tail_count}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous(), "index tensors: int32 CUDA");
    }
    CHECK_DEV(state, at::kFloat);
    CHECK_DEV(window, at::kHalf);
    CHECK_DEV(tail, at::kHalf);
    CHECK_DEV(out, at::kHalf);
    CHECK_DEV(scratch, at::kByte);
    const int64_t M = qkv.size(0);
    const int64_t B = cu.size(0) - 1;
    const int64_t n_ck = ck_pos.size(0);
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) >= CH && qkv.stride(1) == 1 && qkv.stride(0) % 4 == 0, "qkv [M, CH]");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == M && a.size(1) == NV && a.stride(1) == 1, "a [M, NV]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == M && b.size(1) == NV && b.stride(1) == 1, "b [M, NV]");
    TORCH_CHECK(conv.is_contiguous() && conv.numel() == static_cast<int64_t>(CH) * C, "conv [CH, C] f32");
    TORCH_CHECK(rate.numel() == NV && time_bias.numel() == NV && rate.is_contiguous() && time_bias.is_contiguous(),
                "rate / time_bias [NV]");
    TORCH_CHECK(pos0.numel() == B && src.numel() == B && ck_cu.numel() == B + 1, "slot tensors");
    TORCH_CHECK(ck_dst.numel() == n_ck && ck_kind.numel() == n_ck, "checkpoint tensors");
    TORCH_CHECK(state.dim() == 4 && state.size(1) == NV && state.size(2) == W && state.size(3) == W &&
                    state.stride(3) == 1 && state.stride(2) % 4 == 0 && state.stride(1) % 4 == 0 &&
                    state.stride(0) % 4 == 0,
                "state [S, NV, V, K] f32, K contiguous");
    TORCH_CHECK(window.dim() == 3 && window.size(1) == C - 1 && window.size(2) == CH && window.stride(2) == 1 &&
                    window.stride(1) % 4 == 0 && window.stride(0) % 4 == 0,
                "window [S, C-1, CH] f16, channel contiguous");
    TORCH_CHECK(tail.dim() == 3 && tail.size(1) == TAIL && tail.size(2) == RAW && tail.is_contiguous(),
                "tail [S, 15, CH + 2 NV] f16 contiguous");
    TORCH_CHECK(tail_count.numel() == state.size(0), "tail_count [S]");
    TORCH_CHECK(out.is_contiguous() && out.numel() == M * NV * W, "out [M, NV, W] contiguous");
    TORCH_CHECK(scratch.numel() >= scratch_bytes(M, B, n_ck), "scratch too small (scratch_bytes)");
    TORCH_CHECK(rows == 32 || rows == 64 || rows == 128, "rows in {32, 64, 128}");
    CHECK_ALIGN(qkv, 16);
    CHECK_ALIGN(state, 16);
    CHECK_ALIGN(window, 16);
    CHECK_ALIGN(tail, 16);
    CHECK_ALIGN(out, 16);
    CHECK_ALIGN(conv, 16);
    CHECK_ALIGN(scratch, 256);
    if (B == 0) return;

    const int64_t P = max_pieces(M, B, n_ck);
    u8 *base = scratch.data_ptr<uint8_t>();
    Params p{};
    p.qkv = reinterpret_cast<const u16 *>(qkv.data_ptr());
    p.qkv_stride = qkv.stride(0);
    p.alpha = reinterpret_cast<const u16 *>(a.data_ptr());
    p.alpha_stride = a.stride(0);
    p.beta = reinterpret_cast<const u16 *>(b.data_ptr());
    p.beta_stride = b.stride(0);
    p.conv = conv.data_ptr<float>();
    p.rate = rate.data_ptr<float>();
    p.time_bias = time_bias.data_ptr<float>();
    p.cu = cu.data_ptr<int>();
    p.pos0 = pos0.data_ptr<int>();
    p.src = src.data_ptr<int>();
    p.ck_cu = ck_cu.data_ptr<int>();
    p.ck_pos = ck_pos.data_ptr<int>();
    p.ck_dst = ck_dst.data_ptr<int>();
    p.ck_kind = ck_kind.data_ptr<int>();
    p.state = state.data_ptr<float>();
    p.st_slot = state.stride(0);
    p.st_head = state.stride(1);
    p.st_row = state.stride(2);
    p.window = reinterpret_cast<u16 *>(window.data_ptr());
    p.win_slot = window.stride(0);
    p.win_tap = window.stride(1);
    p.tail = reinterpret_cast<u16 *>(tail.data_ptr());
    p.tail_slot = tail.stride(0);
    p.tail_count = tail_count.data_ptr<int>();
    p.out = reinterpret_cast<u16 *>(out.data_ptr());
    p.slot_start = reinterpret_cast<int *>(base);
    p.piece_cu = reinterpret_cast<int *>(base) + B;
    const int64_t tables = ((B + (B + 1)) * 4 + 255) / 256 * 256;
    p.pieces = reinterpret_cast<int4 *>(base + tables);
    p.records = base + tables + (P * 16 + 255) / 256 * 256;
    p.B = static_cast<int>(B);
    p.max_pieces = static_cast<int>(P);
    p.epsilon = static_cast<float>(epsilon);
    p.grouped = grouped ? 1 : 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gdn::gdn_pieces<<<1, 256, 0, stream>>>(p);
    gdn::gdn_chunk_prepare<<<dim3(static_cast<unsigned>(P), NK), PREPARE_THREADS, 0, stream>>>(p);
    if (rows == 32) launch_scan<32>(p, stream);
    else if (rows == 64) launch_scan<64>(p, stream);
    else launch_scan<128>(p, stream);
    gdn::gdn_publish_window<<<dim3((RAW + 127) / 128, static_cast<unsigned>(B)), 128, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "gated delta rule prefill (Magnitude WY port)");
    m.def("scratch_bytes", &scratch_bytes, "scratch bytes for (M, B, checkpoints)");
    m.attr("NK") = gdn::NK;
    m.attr("NV") = gdn::NV;
    m.attr("W") = gdn::W;
    m.attr("C") = gdn::C;
    m.attr("PIECE") = gdn::PIECE;
    m.attr("TAIL") = gdn::TAIL;
    m.attr("RAW") = gdn::RAW;
    m.attr("PIECE_BYTES") = static_cast<int64_t>(gdn::PIECE_BYTES);
}
