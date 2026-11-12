#pragma once
// gemv3: exl3_gemv_kernel (4 bpw, register path) with the block mapping transposed for DRAM locality.
// Stock: block = one 32/64-column group, its warps = the WK k-chunks of that group (each warp reads 256-512 B per
// k-row at a full-row stride). Here: work item = (k-chunk j, column block cb of WK adjacent groups), warp w = group
// cb * WK + w, so a block reads WK * 256-512 B (4-8 KB) contiguous per k-row. Per (group, chunk) the arithmetic is the
// stock one (same chunk bounds, prefetch ring, fp16 MMA chain, fold cadence); the cross-chunk fp32 sum goes through a
// global scratch [WK][ROWS][n] (DevCtx::get_gv3_scratch, via the locks argument) and is summed in the stock order j = 0..WK-1 from 0.0f after a grid
// sync, rounded to half into C, then the unchanged output Hadamard -> bitwise identical to exl3_gemv_kernel.
#include "exl3_gemv_kernel.cuh"

// The scratch arrives in the locks argument (the GEMV path never uses locks; the launcher passes
// DevCtx::get_gv3_scratch instead), so no device symbol has to be set and graph capture is unaffected

template <int bits, bool c_fp32, int cb, int MMODE, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256, 1)
void exl3_gemv_kernel_t(EXL3_GEMM_ARGS)
{
    static_assert(bits == 4 && !c_fp32, "gemv3: 4 bpw, fp16 output");
    constexpr int WK   = CFG == 0 ? 16 : 8;
    constexpr int WNT  = CFG == 0 ? 2 : 4;
    constexpr int PF   = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = WNT;
    constexpr int LSTRIDE = 32;

    auto grid = cooperative_groups::this_grid();

    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128, suh + (this_warp * 128) % size_k, 0.088388347648f);
        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;
    const int num_cb = CEIL_DIVIDE(num_groups, WK);
    const int chunk = CEIL_DIVIDE(kslices, WK);

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;
    float* scratch = (float*) locks;

    for (int item = blockIdx.x; item < num_cb * WK; item += gridDim.x)
    {
        const int cbk = item / WK;           // column block
        const int j = item % WK;             // k-chunk (stock: the warp index)
        const int group = cbk * WK + warp;
        if (group >= num_groups) continue;   // no block-level sync inside the item loop
        const int ks0 = j * chunk;
        const int myn = max(0, min(chunk, kslices - ks0));

        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
        auto ld_b = [&] (int i, int l) -> uint32_t { return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE); };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }
            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        // Partial of (group, chunk j) -> scratch[j][row][col], the slot stock keeps in sh_red[j][row][col]
        const int c0 = 2 * (lane & 3);
        const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
        const int sr0 = MMODE == 0 ? 0 : r0;
        if (store0)
        {
            float* sp = scratch + ((size_t) j * ROWS + sr0) * size_n + group * COLS;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    const int col = t * 16 + f * 8 + c0;
                    sp[col + 0] = acc0[t][f].x;
                    sp[col + 1] = acc0[t][f].y;
                }
        }
    }

    grid.sync();

    // Cross-chunk sum in the stock order, rounded to half into C
    {
        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        const int total = rows_out * size_n;
        for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < total; idx += gridDim.x * THREADS)
        {
            const int r = idx / size_n;
            const int col = idx % size_n;
            float sum = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < WK; ++jj)
                sum += __ldcg(scratch + ((size_t) jj * ROWS + r) * size_n + col);
            ((half*) C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
    }

    // Output scales and Hadamard transform, as in exl3_gemv_kernel
    {
        grid.sync();
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<false, true>(((const half*) C) + this_warp * 128, ((half*) C) + this_warp * 128,
                                            svh + (this_warp * 128) % size_n, 0.088388347648f);
    }
}
