// kvw4_kernel.cu (port of the Intel Arc XPU int4 KV write): fused int4_per_token_head KV write. One 256-thread block
// per (token, KV head, K|V): x (fp16) * signs, unnormalized Sylvester Hadamard in fp32 (= vLLM fast_hadamard_transform),
// fp16 round (as single_rht(..).to(fp16) and the XPU KV write), then vLLM's _reshape_cache_int4_kernel math: scale =
// max((max - min) / 15, 1e-6), zp = clamp(round_half_away(-min / scale), 0, 15), q = clamp(round_half_away(x / scale *
// ... + zp)) (x * (1 / scale) + zp as vLLM), nibble pack (dim 2b low, 2b + 1 high), scale word = (bits & ~15) | zp.
// Record (264 B): K nibbles +0, K scale +128, V nibbles +132, V scale +260. slot < 0 = skip.
// Difference to stock CUDA: stock rotates through hadacore in bf16 (input rounded to bf16 first); this keeps fp32.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float rha(float x) { return x >= 0.f ? truncf(x + 0.5f) : truncf(x - 0.5f); }  // as (x +- 0.5).to(int32)

__global__ void kvw4(const __half* key, const __half* val, long long k_s0, long long k_s1, long long v_s0, long long v_s1,
                     uint8_t* cache, long long s_blk, long long s_slot, long long s_head, const int64_t* slots, int block_size,
                     const float* signs) {
  __shared__ float x[256];
  __shared__ float rmn[8], rmx[8];
  __shared__ uint8_t q[256];
  const int t = blockIdx.x, h = blockIdx.y, isv = blockIdx.z, d = threadIdx.x;
  const long long slot = slots[t];
  if (slot < 0) return;
  const __half* src = isv ? val + t * v_s0 + h * v_s1 : key + t * k_s0 + h * k_s1;
  x[d] = __half2float(src[d]) * signs[d];
  __syncthreads();
#pragma unroll
  for (int s = 1; s < 256; s <<= 1) {
    if (d < 128) {
      const int i = (d / s) * 2 * s + (d % s);
      const float a = x[i], c = x[i + s];
      x[i] = a + c; x[i + s] = a - c;
    }
    __syncthreads();
  }
  const float v = __half2float(__float2half_rn(x[d]));
  float mn = v, mx = v;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
  if ((d & 31) == 0) { rmn[d >> 5] = mn; rmx[d >> 5] = mx; }
  __syncthreads();
  mn = rmn[0]; mx = rmx[0];
#pragma unroll
  for (int i = 1; i < 8; i++) { mn = fminf(mn, rmn[i]); mx = fmaxf(mx, rmx[i]); }
  const float scale = fmaxf((mx - mn) / 15.f, 1e-6f);
  const float zpf = fminf(fmaxf(rha(-mn / scale), 0.f), 15.f);
  const float inv = 1.f / scale;
  q[d] = (uint8_t)fminf(fmaxf(rha(v * inv + zpf), 0.f), 15.f);
  __syncthreads();
  const long long blk = slot / block_size, off = slot % block_size;
  uint8_t* rec = cache + blk * s_blk + off * s_slot + (long long)h * s_head + (isv ? 132 : 0);
  if (d < 128) rec[d] = q[2 * d] | (q[2 * d + 1] << 4);
  if (d == 0) {
    const uint32_t sb = (__float_as_uint(scale) & ~15u) | ((uint32_t)zpf & 15u);
    *reinterpret_cast<uint32_t*>(rec + 128) = sb;
  }
}

// key / value [T, H, 256] fp16 (dims contiguous); cache = uint8 view whose (blk, slot, head, 0) is the record start
void kvw4_write(torch::Tensor key, torch::Tensor value, torch::Tensor cache, int64_t s_blk, int64_t s_slot, int64_t s_head,
                torch::Tensor slots, int64_t block_size, torch::Tensor signs) {
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1 && key.size(2) == 256, "kvw4: [T, H, 256] with contiguous dims");
  const int T = key.size(0), H = key.size(1);
  if (T == 0) return;
  auto stream = c10::cuda::getCurrentCUDAStream();
  kvw4<<<dim3(T, H, 2), 256, 0, stream>>>(reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                                         reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), key.stride(0), key.stride(1),
                                         value.stride(0), value.stride(1), reinterpret_cast<uint8_t*>(cache.data_ptr()),
                                         s_blk, s_slot, s_head, slots.data_ptr<int64_t>(), (int)block_size, signs.data_ptr<float>());
}

TORCH_LIBRARY(kvw4, m) {
  m.def("write(Tensor key, Tensor value, Tensor(a!) cache, int s_blk, int s_slot, int s_head, Tensor slots, int block_size, Tensor signs) -> ()");
}
TORCH_LIBRARY_IMPL(kvw4, CUDA, m) { m.impl("write", &kvw4_write); }
