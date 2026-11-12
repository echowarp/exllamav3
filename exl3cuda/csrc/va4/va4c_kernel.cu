// va4c_kernel.cu: int4-KV verify / decode attention for Qwen3.8-27B on the RTX 5090 (op namespace va4c).
// Third revision of the int4 version of csrc/va/va_kernel.cu (same strategies as the Intel Arc XPU kernels):
//  * LAZY rescale: rescale O / l only when a row max grows by > 8 in log2 units (p <= 256, safe in fp16); optional
//    -DHPV (PV accumulated in fp16 per 32-token tile, then added to fp32; not used by default).
//  * QK on int8 tensor cores (the fp16 version measured ~107 TFLOPS = the 5090's fp16/fp32-acc mma ceiling): prerot
//    also quantizes q_r to int8 per (token, head) row; K = (n - zp) as exact signed bytes (__vsub4); one 32-bit K word
//    = one m16n8k32 s8 k-step. PV fp16 x fp16 -> fp32.
// vLLM 0.26 int4_per_token_head record per (token, head) = [K 128 B packed nibbles | K scale f32 | V 128 B | V scale f32]
// = 264 B; byte b holds dim 2b (low nibble) and 2b+1 (high); value = (n - zp) * s with zp = the scale's low 4 mantissa bits
// and s = the scale with those bits cleared; K and V are stored rotated by the unnormalized Hadamard H(D x).
//  * va4c_prerot: q_r = H(D q) / 256 (fp32 butterflies, fp16 out) -> q_r . k_rot = q . k (pre-rotated q)
//  * va4c_main: one 32-bit K word = 8 dims = two mma k-steps (logical order 2qd,2qd+1,2qd+8,2qd+9 <-> dims
//    {0,2,4,6} / {1,3,5,7} of the word), the same permutation for Q; zp removed in-register by subtracting (1024+zp)
//    from the 1024+n fp16 trick (exact, same instruction count as int8). V -> (n - zp) fp16 in smem (exact).
//  * va4c_merge: segment merge + o = D H(o_r) / 256 (un-rotating merge).
// Structure otherwise as va_kernel.cu: FlashDecoding split, 6 warps = 3 row groups x 2 head-dim halves, fp32 accum.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdint>

#define HD 256
#define GQA 6
#define ROWS 48
#define TILE 32
#define NST 3
#define TOKB 272          // smem bytes per staged token (264 + 8 pad: banks 4g+qd, conflict-free K words)
#define TOKREC 264
#define KSO 128           // K scale offset
#define VO 132            // V offset
#define VSO 260           // V scale offset
#define VHS 264           // fp16 V row stride (halves) in smem: 528 B, ldmatrix conflict-free
#define NW 6
#define NT (NW * 32)
#ifndef LZT
#define LZT 8.f           // lazy rescale threshold (log2 units); 0 = eager. HPV needs a small one (fp16 PV range)
#endif

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
// byte word {b0,b1,b2,b3} (values 0..15) -> half2 (b0,b1), (b2,b3) minus z2 (= half2 1024+zp): exact n - zp
__device__ __forceinline__ void n4_h2(uint32_t w, uint32_t z2, uint32_t& lo, uint32_t& hi) {
  uint32_t a = __byte_perm(w, 0x64646464u, 0x4140);
  uint32_t b = __byte_perm(w, 0x64646464u, 0x4342);
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(lo) : "r"(a), "r"(z2));
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(hi) : "r"(b), "r"(z2));
}
__device__ __forceinline__ uint32_t zp2(uint32_t sbits) { uint32_t z = 0x6400u + (sbits & 0xFu); return z | (z << 16); }
__device__ __forceinline__ float sval(uint32_t sbits) { return __uint_as_float(sbits & ~0xFu); }
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void imma16832(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma16816(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(c[0]), "+r"(c[1])
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
  const float* qs;          // [Tq, 24] int8 q row scales
  const __half* q;          // [Tq, 24, 256], token / head strides q_s0 / q_s1 (elements), dims contiguous
  long long q_s0, q_s1, o_s0, o_s1;
  const int8_t* kv;         // token record base (K at +0, K scale +256, V +260, V scale +516)
  long long s_blk, s_head, s_slot;  // bytes; e.g. int8 NHD [blk, slot, head, 520]: s_slot = 4 heads x 520 = 2080
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

__global__ void __launch_bounds__(NT, 1) va4c_main(VaParams P) {
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

  // Q fragments, int8: k-step u (32 dims dh*128 + 32u .. +31) for thread qd = slots 4qd..4qd+3 <-> dims 8qd + {0,2,4,6},
  // slots 16+4qd.. <-> 8qd + {1,3,5,7} (the K word's low / high nibbles)
  uint32_t qa[4][4];
  float qsc0, qsc1;
  {
    const int8_t* Q8 = reinterpret_cast<const int8_t*>(P.q);
    const long long i0 = (long long)(q0 + qp0) * 24 + kvh * GQA + r0 % GQA, i1 = (long long)(q0 + qp1) * 24 + kvh * GQA + r1 % GQA;
    qsc0 = v0 ? P.qs[i0] : 0.f; qsc1 = v1 ? P.qs[i1] : 0.f;
    const int8_t* qr0 = Q8 + i0 * HD + dh * 128 + 8 * qd;
    const int8_t* qr1 = Q8 + i1 * HD + dh * 128 + 8 * qd;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      uint2 x = v0 ? *reinterpret_cast<const uint2*>(qr0 + u * 32) : make_uint2(0, 0);
      uint2 y = v1 ? *reinterpret_cast<const uint2*>(qr1 + u * 32) : make_uint2(0, 0);
      qa[u][0] = __byte_perm(x.x, x.y, 0x6420); qa[u][2] = __byte_perm(x.x, x.y, 0x7531);
      qa[u][1] = __byte_perm(y.x, y.y, 0x6420); qa[u][3] = __byte_perm(y.x, y.y, 0x7531);
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
    // V -> (n - zp) fp16 smem, natural dim order (word wd = dims 8wd..8wd+7)
    for (int c = tid; c < TILE * 32; c += NT) {
      int tok = c >> 5, wd = c & 31;
      uint32_t x = *reinterpret_cast<const uint32_t*>(S8 + tok * TOKB + VO + wd * 4);
      uint32_t z2 = zp2(*reinterpret_cast<const uint32_t*>(S8 + tok * TOKB + VSO));
      uint32_t L4 = x & 0x0F0F0F0Fu, H4 = (x >> 4) & 0x0F0F0F0Fu;
      uint32_t h0, h1, h2, h3;
      n4_h2(__byte_perm(L4, H4, 0x5140), z2, h0, h1);
      n4_h2(__byte_perm(L4, H4, 0x7362), z2, h2, h3);
      *reinterpret_cast<uint4*>(vh + tok * VHS + wd * 8) = make_uint4(h0, h1, h2, h3);
    }
    // partial QK over this warp's 128 dims
    float sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
    uint32_t kz[4];
#pragma unroll
    for (int j = 0; j < 4; j++) kz[j] = (*reinterpret_cast<const uint32_t*>(S8 + (8 * j + g) * TOKB + KSO) & 0xFu) * 0x01010101u;
    int iacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) iacc[j][0] = iacc[j][1] = iacc[j][2] = iacc[j][3] = 0;
#pragma unroll
    for (int u = 0; u < 4; u++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t x = *reinterpret_cast<const uint32_t*>(S8 + (8 * j + g) * TOKB + dh * 64 + u * 16 + 4 * qd);
        uint32_t L4 = __vsub4(x & 0x0F0F0F0Fu, kz[j]), H4 = __vsub4((x >> 4) & 0x0F0F0F0Fu, kz[j]);
#ifndef NOQK
        imma16832(iacc[j], qa[u][0], qa[u][1], qa[u][2], qa[u][3], L4, H4);
#else
        iacc[j][0] += L4 ^ H4;
#endif
      }
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      sacc[j][0] = (float)iacc[j][0] * qsc0; sacc[j][1] = (float)iacc[j][1] * qsc0;
      sacc[j][2] = (float)iacc[j][2] * qsc1; sacc[j][3] = (float)iacc[j][3] * qsc1;
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
        float ks = sval(*reinterpret_cast<const uint32_t*>(S8 + tl * TOKB + KSO));
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
    {
      const bool n0 = mx0 > m0 + LZT, n1 = mx1 > m1 + LZT;   // m = -inf at the start: the first valid tile sets it
      if (__any_sync(0xffffffffu, n0 || n1)) {
        const float mn0 = n0 ? mx0 : m0, mn1 = n1 ? mx1 : m1;
        const float al0 = n0 ? exp2f(m0 - mn0) : 1.f, al1 = n1 ? exp2f(m1 - mn1) : 1.f;
        m0 = mn0; m1 = mn1;
        l0 *= al0; l1 *= al1;
#pragma unroll
        for (int n = 0; n < 16; n++) { oacc[n][0] *= al0; oacc[n][1] *= al0; oacc[n][2] *= al1; oacc[n][3] *= al1; }
      }
    }
    const float mu0 = (m0 == -INFINITY) ? 0.f : m0, mu1 = (m1 == -INFINITY) ? 0.f : m1;
    float vsc[4][2];
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        float x = exp2f(p[j][e] - ((e < 2) ? mu0 : mu1));
        if (e < 2) l0 += x; else l1 += x;
        p[j][e] = x;
      }
      vsc[j][0] = sval(*reinterpret_cast<const uint32_t*>(S8 + (8 * j + 2 * qd) * TOKB + VSO));
      vsc[j][1] = sval(*reinterpret_cast<const uint32_t*>(S8 + (8 * j + 2 * qd + 1) * TOKB + VSO));
    }
    // PV over 2 k-steps of 16 tokens, this warp's 128 dims (8 x ldmatrix.x4.trans = 16 n-tiles)
#ifdef HPV
    uint32_t hacc[16][2];
#pragma unroll
    for (int n = 0; n < 16; n++) hacc[n][0] = hacc[n][1] = 0u;
#endif
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
#if defined(HPV)
        hmma16816(hacc[2 * np], a0, a1, a2, a3, b0, b1);
        hmma16816(hacc[2 * np + 1], a0, a1, a2, a3, b2, b3);
#elif !defined(NOPV)
        mma16816(oacc[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(oacc[2 * np + 1], a0, a1, a2, a3, b2, b3);
#else
        oacc[2 * np][0] += __uint_as_float(a0 ^ b0 ^ b1 ^ a3); oacc[2 * np + 1][0] += __uint_as_float(a1 ^ b2 ^ b3 ^ a2);
#endif
      }
    }
#ifdef HPV
#pragma unroll
    for (int n = 0; n < 16; n++) {
      float2 x = __half22float2(*reinterpret_cast<__half2*>(&hacc[n][0])), y = __half22float2(*reinterpret_cast<__half2*>(&hacc[n][1]));
      oacc[n][0] += x.x; oacc[n][1] += x.y; oacc[n][2] += y.x; oacc[n][3] += y.y;
    }
#endif
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

__global__ void va4c_merge(const float* po, const float* pm, const float* pl, const int* qsl, const float* signs,
                          __half* out, int S, long long o_s0, long long o_s1) {
  __shared__ float x[HD];
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
  x[d] = Ls > 0.f ? o / Ls : 0.f;
  __syncthreads();
#pragma unroll
  for (int h = 1; h < HD; h <<= 1) {      // unnormalized Sylvester Hadamard (vLLM fast_hadamard_transform order)
    if (d < HD / 2) {
      const int i = (d / h) * 2 * h + (d % h);
      const float a = x[i], c = x[i + h];
      x[i] = a + c; x[i + h] = a - c;
    }
    __syncthreads();
  }
  out[(long long)(q0 + r / GQA) * o_s0 + (long long)(kvh * GQA + r % GQA) * o_s1 + d] = __float2half_rn(x[d] * signs[d] * (1.f / HD));
}

// q_r[t, h, :] = H(D q[t, h, :]) / 256 in fp32, then int8 per row: q8 = rint(q_r * 127 / amax), qs = amax / 127
__global__ void va4c_prerot(const __half* q, long long q_s0, long long q_s1, const float* signs, int8_t* qr, float* qs) {
  __shared__ float x[HD];
  __shared__ float red[8];
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  x[d] = __half2float(q[(long long)t * q_s0 + (long long)h * q_s1 + d]) * signs[d];
  __syncthreads();
#pragma unroll
  for (int s = 1; s < HD; s <<= 1) {
    if (d < HD / 2) {
      const int i = (d / s) * 2 * s + (d % s);
      const float a = x[i], c = x[i + s];
      x[i] = a + c; x[i + s] = a - c;
    }
    __syncthreads();
  }
  const float v = x[d] * (1.f / HD);
  float m = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((d & 31) == 0) red[d >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < 8; i++) m = fmaxf(m, red[i]);
  const float inv = m > 0.f ? 127.f / m : 0.f;
  qr[((long long)t * 24 + h) * HD + d] = (int8_t)__float2int_rn(v * inv);
  if (d == 0) qs[(long long)t * 24 + h] = m / 127.f;
}

// q [Tq,24,256] f16; key = uint8 view whose element (blk, slot, head, 0) is the 264 B record start; signs f32 [256];
// qr = f16 scratch [>= Tq, 24, 256]; block_table int32; seq_lens int32 [nseq]; qsl int32 [nseq+1]; out f16 like q.
void va4c_forward(torch::Tensor q, torch::Tensor key, int64_t s_blk, int64_t s_head, int64_t s_slot, torch::Tensor block_table,
                 torch::Tensor seq_lens, torch::Tensor qsl, torch::Tensor out, torch::Tensor po, torch::Tensor pm,
                 torch::Tensor pl, torch::Tensor signs, torch::Tensor qr, int64_t S, int64_t block_size, double softmax_scale,
                 int64_t qlmax) {
  const int nseq = seq_lens.size(0), Tq = q.size(0);
  VaParams P;
  TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1, "va4: head dim must be contiguous");
  TORCH_CHECK(qr.is_contiguous() && qr.size(0) >= Tq, "va4: qr scratch");
  auto stream = c10::cuda::getCurrentCUDAStream();
  // qr scratch (f16 [>= Tq, 24, 256], 2 B/elem): int8 q in its first Tq*24*256 bytes, row scales after Tq*24*256*... (end half)
  int8_t* q8 = reinterpret_cast<int8_t*>(qr.data_ptr());
  float* qsp = reinterpret_cast<float*>(q8 + (long long)qr.size(0) * 24 * HD);
  va4c_prerot<<<dim3(Tq, 24), HD, 0, stream>>>(reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), q.stride(0), q.stride(1),
                                              signs.data_ptr<float>(), q8, qsp);
  P.q = reinterpret_cast<const __half*>(q8); P.qs = qsp;
  P.q_s0 = 24 * HD; P.q_s1 = HD; P.o_s0 = out.stride(0); P.o_s1 = out.stride(1);
  P.kv = reinterpret_cast<const int8_t*>(key.data_ptr());
  P.s_blk = s_blk; P.s_head = s_head; P.s_slot = s_slot;
  P.bt = block_table.data_ptr<int>(); P.bt_stride = block_table.stride(0);
  P.seq_lens = seq_lens.data_ptr<int>(); P.qsl = qsl.data_ptr<int>();
  P.po = po.data_ptr<float>(); P.pm = pm.data_ptr<float>(); P.pl = pl.data_ptr<float>();
  P.S = S; P.block_size = block_size;
  P.scale_log2 = (float)(softmax_scale * 1.4426950408889634);
  const int smem = NST * TILE * TOKB + TILE * VHS * 2 + 2 * NW * 4 * 32 * 16;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(va4c_main, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  va4c_main<<<dim3(S, 4, nseq), NT, smem, stream>>>(P);
  va4c_merge<<<dim3(qlmax * GQA, 4, nseq), HD, 0, stream>>>(P.po, P.pm, P.pl, P.qsl, signs.data_ptr<float>(),
                                                           reinterpret_cast<__half*>(out.data_ptr<at::Half>()), S,
                                                           P.o_s0, P.o_s1);
}

TORCH_LIBRARY(va4c, m) {
  m.def("forward(Tensor q, Tensor key, int s_blk, int s_head, int s_slot, Tensor block_table, Tensor seq_lens, Tensor qsl, Tensor(a!) out, Tensor(b!) po, Tensor(c!) pm, Tensor(d!) pl, Tensor signs, Tensor(e!) qr, int S, int block_size, float softmax_scale, int qlmax) -> ()");
}
TORCH_LIBRARY_IMPL(va4c, CUDA, m) { m.impl("forward", &va4c_forward); }
