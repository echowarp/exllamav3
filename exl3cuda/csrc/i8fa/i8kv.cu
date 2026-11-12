// i8kv: bit-exact helpers for the plugin's int8-KV FA2 prefill (EXL3_I8FA).
//  deq2(kc, vc, pages, start, end) -> (K, V) fp16 [n, Hk, 256]: the plugin's _i8_gather_dequant for K and V in one
//      launch, fp16(float(x) * scale) per element (RNE), reading the paged per-token-head cache directly.
//  merge(parts, w, ws, out): the plugin's log-sum-exp merge as one kernel (see below).
// Built WITHOUT fast math; explicit _rn intrinsics (no contraction).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

struct DeqArgs { const int8_t *c[2]; __half *o[2]; };

__device__ __forceinline__ float b2f(uint32_t wx, int k) {
    return __fsub_rn(__uint_as_float(__byte_perm(wx, 0x4B000000u, 0x7540 | k)), 8388736.0f);
}

__global__ void deq2_kernel(DeqArgs a, int64_t s_blk, int64_t s_slot, int64_t s_head, const int *pages, int bs,
                            int start, int n, int Hk) {
    const int64_t per = (int64_t)n * Hk * 64;                  // one thread per 4 values; K then V
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= 2 * per) return;
    const int which = idx >= per; idx -= which * per;
    const int d4 = idx % 64; const int64_t th = idx / 64; const int hh = th % Hk; const int t = th / Hk + start;
    const int8_t *row = a.c[which] + (int64_t)pages[t / bs] * s_blk + (int64_t)(t % bs) * s_slot + hh * s_head;
    const float s = *reinterpret_cast<const float *>(row + 256);
    const uint32_t wx = *reinterpret_cast<const uint32_t *>(row + 4 * d4) ^ 0x80808080u;
    __half2 lo = __floats2half2_rn(__fmul_rn(b2f(wx, 0), s), __fmul_rn(b2f(wx, 1), s));
    __half2 hi = __floats2half2_rn(__fmul_rn(b2f(wx, 2), s), __fmul_rn(b2f(wx, 3), s));
    uint2 r; r.x = *reinterpret_cast<uint32_t *>(&lo); r.y = *reinterpret_cast<uint32_t *>(&hi);
    *reinterpret_cast<uint2 *>(a.o[which] + idx * 4) = r;
}

std::vector<at::Tensor> deq2(at::Tensor kc, at::Tensor vc, at::Tensor pages, int64_t start, int64_t end) {
    at::cuda::CUDAGuard g{kc.device()};
    TORCH_CHECK(kc.dtype() == torch::kInt8 && vc.dtype() == torch::kInt8 && kc.dim() == 4 && kc.size(3) == 260 &&
                kc.stride(3) == 1 && vc.stride(3) == 1 && kc.strides() == vc.strides(), "kv cache view");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kc.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(vc.data_ptr()) % 4 == 0 &&
                kc.stride(0) % 4 == 0 && kc.stride(1) % 4 == 0 && kc.stride(2) % 4 == 0, "kv alignment");
    TORCH_CHECK(pages.dtype() == torch::kInt32 && pages.is_contiguous(), "pages");
    const int n = end - start, Hk = kc.size(2), bs = kc.size(1);
    TORCH_CHECK(n > 0 && start >= 0 && (end + bs - 1) / bs <= pages.numel(), "range");
    auto k16 = torch::empty({n, Hk, 256}, kc.options().dtype(torch::kFloat16));
    auto v16 = torch::empty({n, Hk, 256}, kc.options().dtype(torch::kFloat16));
    DeqArgs a{{kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>()},
              {reinterpret_cast<__half *>(k16.data_ptr()), reinterpret_cast<__half *>(v16.data_ptr())}};
    const int64_t total = 2 * (int64_t)n * Hk * 64;
    deq2_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        a, kc.stride(0), kc.stride(1), kc.stride(2), pages.data_ptr<int>(), bs, start, n, Hk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {k16, v16};
}

struct OPtrs { const __half *p[8]; };

__global__ void merge_kernel(OPtrs o, int64_t o_row, int64_t o_head, const float *w,
                             const float *ws, int n, int Q, int H, __half *out, int64_t out_row, int64_t out_head) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;    // one thread per 8 values
    const int64_t total = (int64_t)Q * H * 32;
    if (idx >= total) return;
    const int d8 = idx % 32; const int64_t qh = idx / 32; const int h = qh % H; const int q = qh / H;
    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
    for (int p = 0; p < n; ++p) {
        const float wp = w[((int64_t)p * H + h) * Q + q];
        const uint4 raw = *reinterpret_cast<const uint4 *>(o.p[p] + q * o_row + h * o_head + d8 * 8);
        const __half *hv = reinterpret_cast<const __half *>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = __fadd_rn(acc[j], __fmul_rn(__half2float(hv[j]), wp));
    }
    const float den = ws[(int64_t)h * Q + q];
    uint4 res;
    __half *rv = reinterpret_cast<__half *>(&res);
    #pragma unroll
    for (int j = 0; j < 8; ++j) rv[j] = __float2half_rn(__fdiv_rn(acc[j], den));
    *reinterpret_cast<uint4 *>(out + q * out_row + h * out_head + d8 * 8) = res;
}

void merge(std::vector<at::Tensor> os, at::Tensor w, at::Tensor ws, at::Tensor out) {
    TORCH_CHECK(os.size() >= 1 && os.size() <= 8, "1..8 parts");
    at::cuda::CUDAGuard g{os[0].device()};
    OPtrs P{};
    for (size_t i = 0; i < os.size(); ++i) {
        TORCH_CHECK(os[i].dim() == 3 && os[i].size(2) == 256 && os[i].stride(2) == 1 && os[i].dtype() == torch::kFloat16 &&
                    os[i].sizes() == os[0].sizes() && os[i].strides() == os[0].strides(), "o parts");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(os[i].data_ptr()) % 16 == 0, "o align");
        P.p[i] = reinterpret_cast<const __half *>(os[i].data_ptr());
    }
    const at::Tensor &o = os[0];
    TORCH_CHECK(out.dim() == 3 && out.stride(2) == 1 && out.dtype() == torch::kFloat16, "out");
    TORCH_CHECK(w.is_contiguous() && ws.is_contiguous() && w.dtype() == torch::kFloat32 && ws.dtype() == torch::kFloat32, "w");
    const int n = os.size(), Q = o.size(0), H = o.size(1);
    TORCH_CHECK(w.numel() == (int64_t)n * H * Q && ws.numel() == (int64_t)H * Q, "w size");
    TORCH_CHECK(o.stride(0) % 8 == 0 && o.stride(1) % 8 == 0 && out.stride(0) % 8 == 0 && out.stride(1) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "alignment");
    const int64_t total = (int64_t)Q * H * 32;
    if (total == 0) return;
    merge_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        P, o.stride(0), o.stride(1), w.data_ptr<float>(),
        ws.data_ptr<float>(), n, Q, H, reinterpret_cast<__half *>(out.data_ptr()), out.stride(0), out.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("merge", &merge); m.def("deq2", &deq2); }
