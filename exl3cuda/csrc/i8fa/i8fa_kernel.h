// i8fa: FA2 forward (hdim 256, fp16, 64x64 tile, 4 warps) that reads the paged int8_per_token_head
// KV cache directly. Each K/V tile is staged as raw int8 with cp.async into the fp16 smem slot it will occupy (every
// thread stages and later converts only its own bytes, so no extra barrier), and the per-row f32 scales go to a small
// smem area. The int8 -> fp16 conversion uses the plugin's dequant arithmetic, fp16(float(x) * scale) with RNE, so the
// tile in smem is bit-identical to what the FA2 cp.async of the dequantized fp16 buffer would load. The conversion of
// V(n) is spread over the second half of the Q.K GEMM of tile n and the conversion of K(n-1) over the P.V GEMM of tile n
// (independent ALU work next to the MMAs). The math (MMA order, masking, softmax, epilogue) is the srcmod (FlashAttention
// 2, Copyright (c) 2024 Tri Dao, BSD-3-Clause; see ../../THIRD_PARTY_NOTICES.md) compute_attn_1rowblock with FA16_PV / FA16_QK = 0/0 (A: stock) or 1/2 (F).
// One launch runs several "parts" (key ranges of one request against the same query rows) as grid.y: the plugin's
// 32k-token prefix chunks (non-causal) and the causal tail; each part writes its own o / lse like the separate FA calls.
#pragma once
#include "flash_fwd_kernel.h"

#ifndef I8_VW
#define I8_VW 8      // S-GEMM k-step at which V(n) is waited for and conversion starts (16 k-steps)
#endif
#ifndef I8_KW
#define I8_KW 2      // P.V GEMM k-step at which K(n-1) is waited for (4 k-steps)
#endif
#ifndef I8_NOCONV
#define I8_NOCONV 0  // timing experiment only: skip conversions (wrong results)
#endif
#ifndef I8_NOSTAGE
#define I8_NOSTAGE 0 // timing experiment only: skip the int8 cp.async (wrong results)
#endif
#ifndef I8_KPOST
#define I8_KPOST 0   // 1: convert K(n-1) after the P.V GEMM instead of inside it (less register pressure)
#endif

namespace FLASH_NAMESPACE {

constexpr int kI8MaxParts = 8;
struct I8Part { int k0, len, causal, pad; };

struct I8Params {
    const cutlass::half_t *q; int64_t q_row_stride, q_head_stride;            // elements
    cutlass::half_t *o; int64_t o_row_stride, o_head_stride, o_part_stride;  // elements
    float *lse; int64_t lse_part_stride;                                      // lse[part][h][seqlen_q]
    const int8_t *kc, *vc;                                                    // K / V data of slot 0 head 0 (scale at +256)
    int64_t s_blk, s_slot, s_head;                                            // bytes
    const int *pages; int bs;
    int seqlen_q, h, h_k, h_h_k_ratio;
    float scale_softmax, scale_softmax_log2;
    int nparts;
    I8Part parts[kI8MaxParts];
};

__device__ __forceinline__ void i8_cp4(uint32_t s, const void *g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void i8_cp8(uint32_t s, const void *g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(g));
}

// byte k of wx (= raw ^ 0x80808080) -> float(int8) exactly: 2^23 + (x + 128) - (2^23 + 128)
template<int k> __device__ __forceinline__ float i8_b2f(uint32_t wx) {
    return __fsub_rn(__uint_as_float(__byte_perm(wx, 0x4B000000u, 0x7540 | k)), 8388736.0f);
}
__device__ __forceinline__ uint32_t i8_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&h);
}
// 4 int8 -> 4 fp16 = fp16(float(x) * s)
__device__ __forceinline__ uint2 i8_cvt4(uint32_t raw, float s) {
    const uint32_t wx = raw ^ 0x80808080u;
    uint2 r;
    r.x = i8_h2(__fmul_rn(i8_b2f<0>(wx), s), __fmul_rn(i8_b2f<1>(wx), s));
    r.y = i8_h2(__fmul_rn(i8_b2f<2>(wx), s), __fmul_rn(i8_b2f<3>(wx), s));
    return r;
}

// Rows of the 64-row KV tile handled by a warp: w*16 + i (i = 0..15), consecutive tokens. Pages: a 16-token run spans
// at most 2 pages (block size >= 16).
struct I8Rows {
    int64_t b0, b1;   // byte offsets of (page, slot 0, kv head) for the two pages
    int off0, nv;     // slot of row 0 in page 0, number of valid rows (0..16)
    __device__ __forceinline__ int64_t row(int i, int64_t s_slot, int bs) const {
        const int o = off0 + i;
        return o < bs ? b0 + int64_t(o) * s_slot : b1 + int64_t(o - bs) * s_slot;
    }
};
__device__ __forceinline__ I8Rows i8_rows(const I8Params &p, int k0, int seqlen_k, int n_block, int warp, int kvh) {
    I8Rows r;
    const int rel = n_block * 64 + warp * 16;
    r.nv = max(0, min(16, seqlen_k - rel));
    const int t0 = k0 + rel;
    const int pi = t0 / p.bs;
    r.off0 = t0 - pi * p.bs;
    const int64_t hoff = int64_t(kvh) * p.s_head;
    if (r.nv > 0) {
        const int pg0 = p.pages[pi];
        const int pg1 = (r.off0 + r.nv - 1 >= p.bs) ? p.pages[pi + 1] : pg0;
        r.b0 = int64_t(pg0) * p.s_blk + hoff;
        r.b1 = int64_t(pg1) * p.s_blk + hoff;
    } else {
        r.b0 = r.b1 = 0;
    }
    return r;
}

// Stage tile n of K (IsV = false: 8 B per lane per row) or V (4 B per lane per half row) + scales.
template<typename SmemLayout, bool IsV>
__device__ __forceinline__ void i8_stage(const I8Params &p, const int8_t *base, char *sm, float *ssc, int k0,
                                         int seqlen_k, int n_block, int kvh) {
    if (I8_NOSTAGE) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const I8Rows rr = i8_rows(p, k0, seqlen_k, n_block, warp, kvh);
    SmemLayout L;
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < rr.nv) {
            const int r = warp * 16 + i;
            const int8_t *g = base + rr.row(i, p.s_slot, p.bs);
            if constexpr (!IsV) {
                i8_cp8(sbase + 2 * int(L(r, 8 * lane)), g + 8 * lane);
            } else {
                i8_cp4(sbase + 2 * int(L(r, 4 * lane)), g + 4 * lane);
                i8_cp4(sbase + 2 * int(L(r, 128 + 4 * lane)), g + 128 + 4 * lane);
            }
        }
    }
    if (lane < rr.nv) {
        i8_cp4(static_cast<uint32_t>(__cvta_generic_to_shared(ssc + warp * 16 + lane)),
               base + rr.row(lane, p.s_slot, p.bs) + 256);
    }
}

// Convert rows [i0, i1) of this warp's 16 rows of a staged tile (call after cp_async_wait + __syncwarp).
template<typename SmemLayout, bool IsV>
__device__ __forceinline__ void i8_convert(char *sm, const float *ssc, int nv, int i0, int i1) {
    if (I8_NOCONV) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    SmemLayout L;
    #pragma unroll
    for (int i = i0; i < i1; ++i) {
        const int r = warp * 16 + i;
        const bool ok = i < nv;
        const float s = ok ? ssc[r] : 0.f;
        if constexpr (!IsV) {
            uint4 *d = reinterpret_cast<uint4 *>(sm + 2 * int(L(r, 8 * lane)));
            uint4 o = make_uint4(0, 0, 0, 0);
            if (ok) {
                const uint2 raw = *reinterpret_cast<const uint2 *>(d);
                const uint2 a = i8_cvt4(raw.x, s), b = i8_cvt4(raw.y, s);
                o = make_uint4(a.x, a.y, b.x, b.y);
            }
            *d = o;
        } else {
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                uint2 *d = reinterpret_cast<uint2 *>(sm + 2 * int(L(r, 128 * hh + 4 * lane)));
                uint2 o = make_uint2(0, 0);
                if (ok) { o = i8_cvt4(*reinterpret_cast<const uint32_t *>(d), s); }
                *d = o;
            }
        }
    }
}

// ---- GEMMs of utils.h / srcmod with a hook after each k-step's MMAs (same MMA sequence) ----
template<bool A_in_regs, typename Hook, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void i8_gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                        Tensor4 const &tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
                                        TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
                                        ThrCopyB smem_thr_copy_B, Hook &&hook) {
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        hook(i);
    }
}

template<bool A_in_regs, int kNWarps, typename Hook, typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
         typename ThrCopyB>
__forceinline__ __device__ void i8_gemm_qk_chunk(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                                 Tensor4 const &tCsB, TiledCopyA smem_tiled_copy_A,
                                                 TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
                                                 ThrCopyB smem_thr_copy_B, Hook &&hook) {
    TiledMmaH16<kNWarps> mh;
    Tensor acc_h = make_fragment_like<cutlass::half_t>(acc);
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        if (i % 4 == 0) clear(acc_h);
        cute::gemm(mh, tCrA(_, _, i), tCrB(_, _, i), acc_h);
        if (i % 4 == 3) {
            #pragma unroll
            for (int j = 0; j < size(acc); ++j) acc(j) += float(acc_h(j));
        }
        hook(i);
    }
}

template<typename Hook, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma,
         typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void i8_gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                           TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B,
                                           Hook &&hook) {
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        hook(i);
    }
}

template<int kNWarps, typename Hook, typename AccO, typename TP, typename TV, typename SV, typename CopyV,
         typename ThrV>
__forceinline__ __device__ void i8_pv(AccO &acc_o, TP &tOrP, TV &tOrVt, SV const &tOsVt, CopyV cv, ThrV tv,
                                      Hook &&hook) {
#if FA16_PV == 0
    // tiled_mma passed via the stock path (see caller)
#elif FA16_PV == 1
    TiledMmaH16<kNWarps> mh;
    Tensor acc_h = make_fragment_like<cutlass::half_t>(acc_o);
    clear(acc_h);
    i8_gemm_rs(acc_h, tOrP, tOrVt, tOsVt, mh, cv, tv, hook);
    #pragma unroll
    for (int i = 0; i < size(acc_o); ++i) acc_o(i) += float(acc_h(i));
#else
#error "FA16_PV 2 not supported by i8fa"
#endif
}

template<typename Kernel_traits, bool Is_causal>
inline __device__ void compute_attn_i8(const I8Params &params, const int part, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    static_assert(!Kernel_traits::Share_Q_K_smem && !Kernel_traits::Is_Q_in_regs);
    constexpr bool Is_even_MN = false, Is_even_K = true, Is_local = false, Has_alibi = false;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    static_assert(kBlockN == 64 && kHeadDim == 256 && kNWarps == 4 && kBlockM == 64);

    const int seqlen_q = params.seqlen_q;
    const int seqlen_k = params.parts[part].len;
    const int k0 = params.parts[part].k0;
    const int kvh = bidh / params.h_h_k_ratio;
    if (m_block * kBlockM >= seqlen_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q + 0, kBlockN));
    }
    Element *const o_base = params.o + part * params.o_part_stride;
    float *const lse_base = params.lse + part * params.lse_part_stride + int64_t(bidh) * seqlen_q;
    Tensor mO = make_tensor(make_gmem_ptr(o_base), make_shape(seqlen_q, params.h, Int<kHeadDim>{}),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    if (n_block_max <= n_block_min) {
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                                     seqlen_q - m_block * kBlockM);
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { lse_base[m_block * kBlockM + row] = INFINITY; }
        }
        return;
    }

    Tensor mQ = make_tensor(make_gmem_ptr(params.q), make_shape(seqlen_q, params.h, Int<kHeadDim>{}),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    char *const smK = smem_ + Kernel_traits::kSmemQSize;
    char *const smV = smK + size(sK) * sizeof(Element);
    float *const sScK = reinterpret_cast<float *>(smem_ + Kernel_traits::kSmemSize);
    float *const sScV = sScK + kBlockN;

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // Prologue: Q (fp16, as stock) + K of the last tile (int8 staged)
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;
    i8_stage<SmemLayoutKV, false>(params, params.kc, smK, sScK, k0, seqlen_k, n_block, kvh);
    cute::cp_async_fence();

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(seqlen_k, seqlen_q, Is_causal ? seqlen_k : -1,
                                                               Is_causal ? 0 : -1, 0.f);
    const int warp = tidx / 32;
    auto nv_of = [&](int nb) { return max(0, min(16, seqlen_k - (nb * 64 + warp * 16))); };

    // first K tile: converted before the first barrier (the only exposed conversion)
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncwarp();
    i8_convert<SmemLayoutKV, false>(smK, sScK, nv_of(n_block), 0, 16);

    constexpr int kSK = kHeadDim / 16;   // S-GEMM k-steps (16)
    constexpr int kPV = kBlockN / 16;    // P.V GEMM k-steps (4)
    static_assert(I8_VW >= 0 && I8_VW < kSK && I8_KW >= 0 && I8_KW < kPV);

    // one iteration of the stock loop; Masking selects the masked variant, first = masking_step == 0
    auto iteration = [&](auto masking_c, bool first, int nb) {
        constexpr bool Masking = decltype(masking_c)::value;
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        // V(nb): staged int8; OOB rows become zeros at conversion (= stock Clear_OOB_MN on the first block)
        i8_stage<SmemLayoutKV, true>(params, params.vc, smV, sScV, k0, seqlen_k, nb, kvh);
        cute::cp_async_fence();
        const int nvV = nv_of(nb);
        auto hookV = [&](int i) {
            if (i == I8_VW) { FLASH_NAMESPACE::cp_async_wait<0>(); __syncwarp(); }
            if (i >= I8_VW) {
                constexpr int nst = kSK - I8_VW;
                const int j = i - I8_VW;
                i8_convert<SmemLayoutKV, true>(smV, sScV, nvV, (16 * j) / nst, (16 * (j + 1)) / nst);
            }
        };
        if constexpr (FA16_QK == 0) {
            i8_gemm<false>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                           smem_thr_copy_Q, smem_thr_copy_K, hookV);
        } else if constexpr (FA16_QK == 2) {
            i8_gemm_qk_chunk<false, kNWarps>(acc_s, tSrQ, tSrK, tSsQ, tSsK, smem_tiled_copy_Q, smem_tiled_copy_K,
                                             smem_thr_copy_Q, smem_thr_copy_K, hookV);
        } else {
            static_assert(FA16_QK == 0 || FA16_QK == 2, "i8fa: FA16_QK 0 or 2");
        }
        if constexpr (Masking) {
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, nb * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
        }
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        const bool k_pending = nb > n_block_min;
        if (k_pending) {
            i8_stage<SmemLayoutKV, false>(params, params.kc, smK, sScK, k0, seqlen_k, nb - 1, kvh);
            cute::cp_async_fence();
        }
        if constexpr (!Masking) {
            mask.template apply_mask</*Causal_mask=*/false>(
                acc_s, nb * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
        }
        if constexpr (Masking) {
            first ? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2)
                  : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);
        } else {
            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        }
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        const int nvK = nv_of(nb - 1);
        auto hookK = [&](int i) {
            if (!I8_KPOST && k_pending) {
                if (i == I8_KW) { FLASH_NAMESPACE::cp_async_wait<0>(); __syncwarp(); }
                if (i >= I8_KW) {
                    constexpr int nst = kPV - I8_KW;
                    const int j = i - I8_KW;
                    i8_convert<SmemLayoutKV, false>(smK, sScK, nvK, (16 * j) / nst, (16 * (j + 1)) / nst);
                }
            }
        };
        if constexpr (FA16_PV == 0) {
            i8_gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V, hookK);
        } else {
            i8_pv<kNWarps>(acc_o, tOrP, tOrVt, tOsVt, smem_tiled_copy_V, smem_thr_copy_V, hookK);
        }
        if (I8_KPOST && k_pending) {
            FLASH_NAMESPACE::cp_async_wait<0>(); __syncwarp();
            i8_convert<SmemLayoutKV, false>(smK, sScK, nvK, 0, 16);
        }
    };

    constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        iteration(cute::true_type{}, masking_step == 0, n_block);
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
    for (; n_block >= n_block_min; --n_block) {
        iteration(cute::false_type{}, false, n_block);
    }

    // Epilogue (stock)
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(acc_o, params.scale_softmax, 1.0f);
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    __syncthreads();
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < seqlen_q - m_block * kBlockM) { lse_base[m_block * kBlockM + row] = lse(mi); }
        }
    }
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                                 seqlen_q - m_block * kBlockM);
}

// MODE 0: all parts non-causal, 1: all causal, 2: per-part flag
template<typename Kernel_traits, int MODE>
__global__ void i8fa_kernel(const __grid_constant__ I8Params params) {
    const int m_block = blockIdx.x, part = blockIdx.y, bidh = blockIdx.z;
    if constexpr (MODE == 0) {
        compute_attn_i8<Kernel_traits, false>(params, part, bidh, m_block);
    } else if constexpr (MODE == 1) {
        compute_attn_i8<Kernel_traits, true>(params, part, bidh, m_block);
    } else {
        if (params.parts[part].causal) compute_attn_i8<Kernel_traits, true>(params, part, bidh, m_block);
        else compute_attn_i8<Kernel_traits, false>(params, part, bidh, m_block);
    }
}

}  // namespace FLASH_NAMESPACE
