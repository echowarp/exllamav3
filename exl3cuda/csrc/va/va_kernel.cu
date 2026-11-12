// va_kernel.cu: verify / decode attention (long-context decode) for Qwen3.8-27B on the RTX 5090.
// Shape: GQA 24 query / 4 KV heads, head 256, int8 per-token-head KV interleaved per token as
//   [K int8 x256 | K scale f32 | V int8 x256 | V scale f32] = 520 B (vLLM INT8_PER_TOKEN_HEAD layout),
// up to 8 query rows per sequence (k=7 verify). FlashDecoding split over the context: one CTA per
// (segment, KV head, sequence); 6 warps = 3 row groups (16 of the 48 query vectors each) x 2 halves of the head dim.
// QK: each warp computes a partial dot over its 128 dims, the pair exchanges partials through shared memory.
// K is read straight from the int8 staging buffer: the k order inside each mma k-step is permuted
// (logical 2q,2q+1,2q+8,2q+9 <-> physical 4q..4q+3) identically for Q and K, so one 32-bit load feeds a B fragment.
// V is dequantized to fp16 in shared memory per tile and read with ldmatrix.trans. Scales: K's folded into the scores,
// V's into the probabilities. fp32 accumulation (same precision class as the Triton kernel). Partial results per segment
// (unnormalized O, running max in log2 units, sum) are merged by va_merge.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdint>

#define HD 256
#define GQA 6
#define ROWS 48
#define TILE 32
#define NST 3
#define TOKB 528          // smem bytes per staged token (520 + 8 pad: conflict-free 32-bit K reads)
#define TOKREC 520
#define VHS 264           // fp16 V row stride (halves) in smem: 528 B, ldmatrix conflict-free
#define NW 6
#define NT (NW * 32)

__device__ __forceinline__ void cp_async8(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int sz = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 4 signed int8 (little endian) -> two half2 {x0,x1}, {x2,x3}; exact (1024 + (x+128) - 1152)
__device__ __forceinline__ void i8x4_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  w ^= 0x80808080u;
  uint32_t a = __byte_perm(w, 0x64646464u, 0x4140);
  uint32_t b = __byte_perm(w, 0x64646464u, 0x4342);
  const uint32_t k = 0x64806480u;
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(lo) : "r"(a), "r"(k));
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(hi) : "r"(b), "r"(k));
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(s));
}

struct VaParams {
  const __half* q;          // [Tq, 24, 256], token / head strides q_s0 / q_s1 (elements), dims contiguous
  long long q_s0, q_s1, o_s0, o_s1;
  const int8_t* kv;         // token record base (K at +0, K scale +256, V +260, V scale +516)
  long long s_blk, s_head, s_slot;  // bytes; vLLM NHD layout [blk, slot, head, 520]: s_slot = 4 heads x 520 = 2080
  const int* bt;
  int bt_stride;
  const int* seq_lens;
  const int* qsl;           // query_start_loc [nseq + 1]
  float* po;                // [nseq, 4, S, 48, 256]
  float* pm;                // [nseq, 4, S, 48]
  float* pl;
  int S, block_size;
  float scale_log2;         // softmax_scale * log2(e)
};

__global__ void __launch_bounds__(NT, 1) va_main(VaParams P) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* stg = smem;                                                    // NST x TILE x TOKB
  __half* vh = reinterpret_cast<__half*>(smem + NST * TILE * TOKB);       // TILE x VHS
  float4* xb = reinterpret_cast<float4*>(smem + NST * TILE * TOKB + TILE * VHS * 2);  // [2][NW][4][32]

  const int seg = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
  const int rg = w % 3, dh = w / 3, partner = rg + 3 * (1 - dh);
  const int L = P.seq_lens[seq];
  const int q0 = P.qsl[seq], QL = P.qsl[seq + 1] - q0;
  const int ntiles = (L + TILE - 1) / TILE;
  const int tps = (ntiles + P.S - 1) / P.S;
  const int t_begin = seg * tps, t_end = min(ntiles, t_begin + tps);
  const int r0 = rg * 16 + g, r1 = r0 + 8;
  const int qp0 = r0 / GQA, qp1 = r1 / GQA;
  const bool v0 = qp0 < QL, v1 = qp1 < QL;
  const long long obase = (((long long)seq * 4 + kvh) * P.S + seg) * ROWS;

  if (t_begin >= t_end) {
    if (dh == 0 && qd == 0) {
      P.pm[obase + r0] = -INFINITY; P.pl[obase + r0] = 0.f;
      P.pm[obase + r1] = -INFINITY; P.pl[obase + r1] = 0.f;
    }
    return;
  }

  // Q fragments (8 k-steps over this warp's 128 dims), permuted k order
  uint32_t qa[8][4];
  {
    const __half* qr0 = P.q + (long long)(q0 + qp0) * P.q_s0 + (long long)(kvh * GQA + r0 % GQA) * P.q_s1 + dh * 128 + 4 * qd;
    const __half* qr1 = P.q + (long long)(q0 + qp1) * P.q_s0 + (long long)(kvh * GQA + r1 % GQA) * P.q_s1 + dh * 128 + 4 * qd;
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      uint2 x = v0 ? *reinterpret_cast<const uint2*>(qr0 + ks * 16) : make_uint2(0, 0);
      uint2 y = v1 ? *reinterpret_cast<const uint2*>(qr1 + ks * 16) : make_uint2(0, 0);
      qa[ks][0] = x.x; qa[ks][2] = x.y; qa[ks][1] = y.x; qa[ks][3] = y.y;
    }
  }
  float oacc[16][4];
#pragma unroll
  for (int n = 0; n < 16; n++) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  // causal limits: row with query position qp sees tokens <= L - QL + qp
  const int lim0 = v0 ? L - QL + qp0 : L - 1, lim1 = v1 ? L - QL + qp1 : L - 1;

  auto issue = [&](int t, int st) {
    uint8_t* dst = stg + st * TILE * TOKB;
    for (int c = tid; c < TILE * (TOKREC / 8); c += NT) {
      int i = c / (TOKREC / 8), ch = c - i * (TOKREC / 8);
      int tok = t * TILE + i;
      bool ok = tok < L;
      int tk = ok ? tok : 0;
      int blk = P.bt[(long long)seq * P.bt_stride + tk / P.block_size];
      const int8_t* src = P.kv + (long long)blk * P.s_blk + (long long)kvh * P.s_head +
                          (long long)(tk % P.block_size) * P.s_slot + ch * 8;
      cp_async8(dst + i * TOKB + ch * 8, src, ok);
    }
  };

#pragma unroll
  for (int i = 0; i < NST - 1; i++) {
    if (t_begin + i < t_end) issue(t_begin + i, i);
    cp_async_commit();
  }
  for (int t = t_begin; t < t_end; t++) {
    const int it = t - t_begin, st = it % NST;
    cp_async_wait<NST - 2>();
    __syncthreads();
    if (t + NST - 1 < t_end) issue(t + NST - 1, (it + NST - 1) % NST);
    cp_async_commit();
    const uint8_t* S8 = stg + st * TILE * TOKB;
    // V -> fp16 smem
    for (int c = tid; c < TILE * 64; c += NT) {
      int tok = c >> 6, wd = c & 63;
      uint32_t x = *reinterpret_cast<const uint32_t*>(S8 + tok * TOKB + 260 + wd * 4);
      uint32_t lo, hi;
      i8x4_h2(x, lo, hi);
      *reinterpret_cast<uint2*>(vh + tok * VHS + wd * 4) = make_uint2(lo, hi);
    }
    // partial QK over this warp's 128 dims
    float sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t x = *reinterpret_cast<const uint32_t*>(S8 + (8 * j + g) * TOKB + dh * 128 + ks * 16 + 4 * qd);
        uint32_t b0, b1;
        i8x4_h2(x, b0, b1);
        mma16816(sacc[j], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
      }
    }
    float4* mine = xb + ((it & 1) * NW + w) * 4 * 32;
#pragma unroll
    for (int j = 0; j < 4; j++) mine[j * 32 + lane] = make_float4(sacc[j][0], sacc[j][1], sacc[j][2], sacc[j][3]);
    __syncthreads();
    const float4* his = xb + ((it & 1) * NW + partner) * 4 * 32;
    float p[4][4];
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float4 o = his[j * 32 + lane];
      float s4[4] = {sacc[j][0] + o.x, sacc[j][1] + o.y, sacc[j][2] + o.z, sacc[j][3] + o.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int tl = 8 * j + 2 * qd + (e & 1);
        int tok = t * TILE + tl;
        float ks = *reinterpret_cast<const float*>(S8 + tl * TOKB + 256);
        int lim = (e < 2) ? lim0 : lim1;
        float s = (tok <= lim && tok < L) ? s4[e] * ks * P.scale_log2 : -INFINITY;
        p[j][e] = s;
        if (e < 2) mx0 = fmaxf(mx0, s); else mx1 = fmaxf(mx1, s);
      }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float mu0 = (mn0 == -INFINITY) ? 0.f : mn0, mu1 = (mn1 == -INFINITY) ? 0.f : mn1;
    const float al0 = exp2f(m0 - mu0), al1 = exp2f(m1 - mu1);
    m0 = mn0; m1 = mn1;
    l0 *= al0; l1 *= al1;
#pragma unroll
    for (int n = 0; n < 16; n++) { oacc[n][0] *= al0; oacc[n][1] *= al0; oacc[n][2] *= al1; oacc[n][3] *= al1; }
    float vsc[4][2];
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        float x = exp2f(p[j][e] - ((e < 2) ? mu0 : mu1));
        if (e < 2) l0 += x; else l1 += x;
        p[j][e] = x;
      }
      vsc[j][0] = *reinterpret_cast<const float*>(S8 + (8 * j + 2 * qd) * TOKB + 516);
      vsc[j][1] = *reinterpret_cast<const float*>(S8 + (8 * j + 2 * qd + 1) * TOKB + 516);
    }
    // PV over 2 k-steps of 16 tokens, this warp's 128 dims (8 x ldmatrix.x4.trans = 16 n-tiles)
#pragma unroll
    for (int s = 0; s < 2; s++) {
      const int ja = 2 * s, jb = 2 * s + 1;
      uint32_t a0 = pack_h2(p[ja][0] * vsc[ja][0], p[ja][1] * vsc[ja][1]);
      uint32_t a1 = pack_h2(p[ja][2] * vsc[ja][0], p[ja][3] * vsc[ja][1]);
      uint32_t a2 = pack_h2(p[jb][0] * vsc[jb][0], p[jb][1] * vsc[jb][1]);
      uint32_t a3 = pack_h2(p[jb][2] * vsc[jb][0], p[jb][3] * vsc[jb][1]);
      const int tokr = 16 * s + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int np = 0; np < 8; np++) {
        const int dim = dh * 128 + np * 16 + (lane >> 4) * 8;
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, vh + tokr * VHS + dim);
        mma16816(oacc[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(oacc[2 * np + 1], a0, a1, a2, a3, b2, b3);
      }
    }
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  float* o0 = P.po + (obase + r0) * HD + dh * 128 + 2 * qd;
  float* o1 = P.po + (obase + r1) * HD + dh * 128 + 2 * qd;
#pragma unroll
  for (int n = 0; n < 16; n++) {
    *reinterpret_cast<float2*>(o0 + 8 * n) = make_float2(oacc[n][0], oacc[n][1]);
    *reinterpret_cast<float2*>(o1 + 8 * n) = make_float2(oacc[n][2], oacc[n][3]);
  }
  if (dh == 0 && qd == 0) {
    P.pm[obase + r0] = m0; P.pl[obase + r0] = l0;
    P.pm[obase + r1] = m1; P.pl[obase + r1] = l1;
  }
}

__global__ void va_merge(const float* po, const float* pm, const float* pl, const int* qsl, __half* out, int S,
                         long long o_s0, long long o_s1) {
  const int r = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z, d = threadIdx.x;
  const int q0 = qsl[seq], QL = qsl[seq + 1] - q0;
  if (r >= QL * GQA) return;
  const long long b = ((long long)seq * 4 + kvh) * S * ROWS + r;
  float M = -INFINITY;
  for (int s = 0; s < S; s++) if (pl[b + (long long)s * ROWS] > 0.f) M = fmaxf(M, pm[b + (long long)s * ROWS]);
  float Ls = 0.f, o = 0.f;
  for (int s = 0; s < S; s++) {
    const long long i = b + (long long)s * ROWS;
    const float l = pl[i];
    if (l > 0.f) {
      const float f = exp2f(pm[i] - M);
      Ls += l * f;
      o += po[i * HD + d] * f;
    }
  }
  out[(long long)(q0 + r / GQA) * o_s0 + (long long)(kvh * GQA + r % GQA) * o_s1 + d] = __float2half_rn(Ls > 0.f ? o / Ls : 0.f);
}

// q [Tq,24,256] f16; key = int8 view whose element (blk, slot, head, 0) is the token record start; block_table int32;
// seq_lens int32 [nseq]; qsl int32 [nseq+1]; out f16 like q; scratch po/pm/pl sized [nseq*4*S*48(*256)].
void va_forward(torch::Tensor q, torch::Tensor key, int64_t s_blk, int64_t s_head, int64_t s_slot, torch::Tensor block_table,
                torch::Tensor seq_lens, torch::Tensor qsl, torch::Tensor out, torch::Tensor po, torch::Tensor pm,
                torch::Tensor pl, int64_t S, int64_t block_size, double softmax_scale, int64_t qlmax) {
  const int nseq = seq_lens.size(0);
  VaParams P;
  TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1 && q.stride(1) % 4 == 0 && q.stride(0) % 4 == 0,
              "va: head dim must be contiguous and 8-byte aligned rows");
  P.q = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  P.q_s0 = q.stride(0); P.q_s1 = q.stride(1); P.o_s0 = out.stride(0); P.o_s1 = out.stride(1);
  P.kv = reinterpret_cast<const int8_t*>(key.data_ptr<int8_t>());
  P.s_blk = s_blk; P.s_head = s_head; P.s_slot = s_slot;
  P.bt = block_table.data_ptr<int>(); P.bt_stride = block_table.stride(0);
  P.seq_lens = seq_lens.data_ptr<int>(); P.qsl = qsl.data_ptr<int>();
  P.po = po.data_ptr<float>(); P.pm = pm.data_ptr<float>(); P.pl = pl.data_ptr<float>();
  P.S = S; P.block_size = block_size;
  P.scale_log2 = (float)(softmax_scale * 1.4426950408889634);
  const int smem = NST * TILE * TOKB + TILE * VHS * 2 + 2 * NW * 4 * 32 * 16;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(va_main, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  auto stream = c10::cuda::getCurrentCUDAStream();
  va_main<<<dim3(S, 4, nseq), NT, smem, stream>>>(P);
  va_merge<<<dim3(qlmax * GQA, 4, nseq), HD, 0, stream>>>(P.po, P.pm, P.pl, P.qsl,
                                                          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), S,
                                                          P.o_s0, P.o_s1);
}

TORCH_LIBRARY(va5090, m) {
  m.def("forward(Tensor q, Tensor key, int s_blk, int s_head, int s_slot, Tensor block_table, Tensor seq_lens, Tensor qsl, Tensor(a!) out, Tensor(b!) po, Tensor(c!) pm, Tensor(d!) pl, int S, int block_size, float softmax_scale, int qlmax) -> ()");
}
TORCH_LIBRARY_IMPL(va5090, CUDA, m) { m.impl("forward", &va_forward); }
