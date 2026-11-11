/* simd/matmul_neon.c — AArch64 NEON GEMM and elementwise kernels.
 *
 * Built only for 64-bit ARM, where NEON and fused multiply-add are part of the
 * base architecture, so no run-time feature check is needed. 32-bit ARM (and the
 * Pi Zero's ARMv6) keeps the scalar backend.
 */
#include "matmul_neon.h"

#if defined(__aarch64__)
#include <arm_neon.h>

/*
 * Micro-kernel: a 6 x 16 block of C in 24 q-register accumulators (6 rows x 4
 * vectors of 4 floats). With the four B vectors that is 28 of AArch64's 32 vector
 * registers; nothing spills. Per step of k: four B loads, six scalar A loads and
 * twenty-four fused multiply-adds by scalar (fmla by element).
 */
static inline void kernel_6x16_neon(const float *A_blk, int lda, const float *Bp, int kc,
                                    float *C_blk, int ldc) {
    float32x4_t c[6][4];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++)
            c[i][j] = vld1q_f32(C_blk + i * ldc + 4 * j);

    const float *a0 = A_blk, *a1 = A_blk + lda, *a2 = A_blk + 2 * lda;
    const float *a3 = A_blk + 3 * lda, *a4 = A_blk + 4 * lda, *a5 = A_blk + 5 * lda;

    for (int k = 0; k < kc; k++) {
        const float *bk = Bp + k * 16;
        float32x4_t b0 = vld1q_f32(bk), b1 = vld1q_f32(bk + 4);
        float32x4_t b2 = vld1q_f32(bk + 8), b3 = vld1q_f32(bk + 12);
#define ROW(i, ai)                                         \
        do {                                               \
            float av = (ai)[k];                            \
            c[i][0] = vfmaq_n_f32(c[i][0], b0, av);        \
            c[i][1] = vfmaq_n_f32(c[i][1], b1, av);        \
            c[i][2] = vfmaq_n_f32(c[i][2], b2, av);        \
            c[i][3] = vfmaq_n_f32(c[i][3], b3, av);        \
        } while (0)
        ROW(0, a0); ROW(1, a1); ROW(2, a2); ROW(3, a3); ROW(4, a4); ROW(5, a5);
#undef ROW
    }

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++)
            vst1q_f32(C_blk + i * ldc + 4 * j, c[i][j]);
}

#define GEMM_KERNEL kernel_6x16_neon
#include "gemm_driver.h"

void fe_gemm_neon(const float *a, int lda, const float *b, int ldb,
                  float *c, int ldc, int M, int N, int K) {
    gemm_blocked(a, lda, b, ldb, c, ldc, M, N, K);
}

FeStatus fe_matmul_neon(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C) return FE_ERR_NULL;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;
    int M = A->shape[0], K = A->shape[1], N = B->shape[1];
    if (B->shape[0] != K || C->shape[0] != M || C->shape[1] != N) return FE_ERR_SHAPE;
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (!fe_tensor_is_contiguous(A) || !fe_tensor_is_contiguous(B) ||
        !fe_tensor_is_contiguous(C)) return FE_ERR_SHAPE;
    gemm_blocked((const float *)A->data, K, (const float *)B->data, N,
                 (float *)C->data, N, M, N, K);
    return FE_OK;
}

/* Elementwise: 4-wide main loop, scalar tail. Exact arithmetic, as the scalar
 * kernels compute it. */
void fe_relu_neon(const float *in, float *out, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmaxq_f32(vld1q_f32(in + i), zero));
    for (; i < n; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void fe_add_neon(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] + b[i];
}

void fe_mul_neon(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
}

#endif /* __aarch64__ */
