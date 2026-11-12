// i8fa binding. fwd(q, kc, vc, pages, parts, scale, o, lse, mode):
//   q     [Q, H, 256] fp16 (row / head strides free, last dim contiguous)
//   kc/vc key_cache / value_cache views [blocks, BS, Hk, 260] int8 of the per-token-head cache (bytes 256..259 = f32 scale)
//   pages int32 block-table row of the request
//   parts list of (k0, len, causal): key ranges [k0, k0 + len) of the request, all against the same Q rows
//   o     [n, Q, H, 256] fp16 (or [Q, H, 256] for one part), lse [n, H, Q] f32
//   mode  0 = all parts non-causal, 1 = all causal, 2 = mixed (per-part flag); -1 = choose
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include "namespace_config.h"
#include "flash.h"
#include "kernel_traits.h"
#include "i8fa_kernel.h"
using namespace FLASH_NAMESPACE;

using KT = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::half_t>;
static constexpr int kSmem = KT::kSmemSize + 2 * 64 * sizeof(float);

template<int MODE> static void launch(const I8Params &p, dim3 grid, cudaStream_t st) {
    auto k = &i8fa_kernel<KT, MODE>;
    static bool once = false;
    if (!once) { C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem)); once = true; }
    k<<<grid, KT::kNThreads, kSmem, st>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fwd(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor pages, std::vector<int64_t> parts, double scale,
         at::Tensor o, at::Tensor lse, int64_t mode) {
    at::cuda::CUDAGuard g{q.device()};
    TORCH_CHECK(q.dtype() == torch::kFloat16 && q.dim() == 3 && q.size(2) == 256 && q.stride(2) == 1, "q");
    TORCH_CHECK(kc.dtype() == torch::kInt8 && vc.dtype() == torch::kInt8 && kc.dim() == 4 && kc.size(3) == 260 &&
                kc.stride(3) == 1 && vc.stride(3) == 1, "kv cache view");
    TORCH_CHECK(kc.stride(0) == vc.stride(0) && kc.stride(1) == vc.stride(1) && kc.stride(2) == vc.stride(2), "kv strides");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kc.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(vc.data_ptr()) % 4 == 0 &&
                kc.stride(0) % 8 == 0 && kc.stride(1) % 8 == 0 && kc.stride(2) % 8 == 0, "kv alignment");
    TORCH_CHECK(pages.dtype() == torch::kInt32 && pages.is_contiguous(), "pages");
    TORCH_CHECK(o.dtype() == torch::kFloat16 && lse.dtype() == torch::kFloat32 && lse.is_contiguous(), "o / lse");
    const int n = parts.size() / 3;
    TORCH_CHECK(n >= 1 && n <= kI8MaxParts && (int)parts.size() == 3 * n, "parts");
    const int Q = q.size(0), H = q.size(1), Hk = kc.size(2);
    TORCH_CHECK(H % Hk == 0 && lse.numel() == (int64_t)n * H * Q, "heads / lse size");
    I8Params p{};
    p.q = reinterpret_cast<const cutlass::half_t *>(q.data_ptr()); p.q_row_stride = q.stride(0); p.q_head_stride = q.stride(1);
    if (o.dim() == 3) {
        TORCH_CHECK(n == 1 && o.size(0) == Q && o.size(1) == H && o.size(2) == 256 && o.stride(2) == 1, "o [Q,H,256]");
        p.o_row_stride = o.stride(0); p.o_head_stride = o.stride(1); p.o_part_stride = 0;
    } else {
        TORCH_CHECK(o.dim() == 4 && o.size(0) == n && o.size(1) == Q && o.size(2) == H && o.size(3) == 256 && o.stride(3) == 1, "o [n,Q,H,256]");
        p.o_row_stride = o.stride(1); p.o_head_stride = o.stride(2); p.o_part_stride = o.stride(0);
    }
    p.o = reinterpret_cast<cutlass::half_t *>(o.data_ptr());
    p.lse = lse.data_ptr<float>(); p.lse_part_stride = (int64_t)H * Q;
    p.kc = kc.data_ptr<int8_t>(); p.vc = vc.data_ptr<int8_t>();
    p.s_blk = kc.stride(0); p.s_slot = kc.stride(1); p.s_head = kc.stride(2);
    p.pages = pages.data_ptr<int>(); p.bs = kc.size(1);
    p.seqlen_q = Q; p.h = H; p.h_k = Hk; p.h_h_k_ratio = H / Hk;
    // as set_params_fprop: float scale, log2 scale = float * double -> float
    const float sc = (float)scale;
    p.scale_softmax = sc; p.scale_softmax_log2 = sc * M_LOG2E;
    p.nparts = n;
    bool anyc = false, allc = true;
    for (int i = 0; i < n; ++i) {
        p.parts[i].k0 = parts[3 * i]; p.parts[i].len = parts[3 * i + 1]; p.parts[i].causal = parts[3 * i + 2] != 0;
        TORCH_CHECK(p.parts[i].len > 0 && (int64_t)(p.parts[i].k0 + p.parts[i].len + p.bs - 1) / p.bs <= pages.numel(), "part range");
        anyc |= p.parts[i].causal; allc &= p.parts[i].causal != 0;
    }
    if (mode < 0) mode = allc ? 1 : (anyc ? 2 : 0);
    TORCH_CHECK(mode == 2 || (mode == 1 && allc) || (mode == 0 && !anyc), "mode does not match parts");
    dim3 grid((Q + 63) / 64, n, H);
    auto st = at::cuda::getCurrentCUDAStream().stream();
    if (mode == 0) launch<0>(p, grid, st); else if (mode == 1) launch<1>(p, grid, st); else launch<2>(p, grid, st);
}

// fused gather-dequant (same arithmetic as the plugin's torch path) for comparison / fallback: out [n, Hk, 256] fp16
__global__ void deq_kernel(const int8_t *c, int64_t s_blk, int64_t s_slot, int64_t s_head, const int *pages, int bs,
                           int start, int n, int Hk, cutlass::half_t *out) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;   // one thread per 4 values
    const int64_t total = (int64_t)n * Hk * 64;
    if (idx >= total) return;
    const int d4 = idx % 64; const int64_t th = idx / 64; const int hh = th % Hk; const int t = th / Hk + start;
    const int8_t *row = c + (int64_t)pages[t / bs] * s_blk + (int64_t)(t % bs) * s_slot + hh * s_head;
    const float s = *reinterpret_cast<const float *>(row + 256);
    const uint32_t raw = *reinterpret_cast<const uint32_t *>(row + 4 * d4);
    *reinterpret_cast<uint2 *>(out + idx * 4) = i8_cvt4(raw, s);
}

at::Tensor deq(at::Tensor c, at::Tensor pages, int64_t start, int64_t end) {
    at::cuda::CUDAGuard g{c.device()};
    const int n = end - start, Hk = c.size(2);
    auto out = torch::empty({n, Hk, 256}, c.options().dtype(torch::kFloat16));
    const int64_t total = (int64_t)n * Hk * 64;
    deq_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        c.data_ptr<int8_t>(), c.stride(0), c.stride(1), c.stride(2), pages.data_ptr<int>(), c.size(1), start, n, Hk,
        reinterpret_cast<cutlass::half_t *>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd);
    m.def("deq", &deq);
}
