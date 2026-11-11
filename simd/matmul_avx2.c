#include "matmul_avx2.h"
#include <immintrin.h>
#include <cpuid.h>

int fe_cpu_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return 0;
    if (eax < 7) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx >> 5) & 1;
}

/*
 * AVX2 micro-kernel: a 6 x 16 block of C in 12 ymm accumulators (6 rows x 2
 * vectors), which with the two B vectors and one broadcast A value fits AVX2's 16
 * registers. Per step of k: two B loads, six broadcasts, twelve FMAs, no spills.
 * The cache blocking around it is shared with the NEON backend (gemm_driver.h).
 */
static inline void kernel_6x16_avx2(const float *A_blk, int lda, const float *Bp, int kc,
                                    float *C_blk, int ldc) {
    __m256 c00 = _mm256_loadu_ps(C_blk + 0 * ldc), c01 = _mm256_loadu_ps(C_blk + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C_blk + 1 * ldc), c11 = _mm256_loadu_ps(C_blk + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C_blk + 2 * ldc), c21 = _mm256_loadu_ps(C_blk + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C_blk + 3 * ldc), c31 = _mm256_loadu_ps(C_blk + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C_blk + 4 * ldc), c41 = _mm256_loadu_ps(C_blk + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C_blk + 5 * ldc), c51 = _mm256_loadu_ps(C_blk + 5 * ldc + 8);

    const float *a0 = A_blk, *a1 = A_blk + lda, *a2 = A_blk + 2 * lda;
    const float *a3 = A_blk + 3 * lda, *a4 = A_blk + 4 * lda, *a5 = A_blk + 5 * lda;

    for (int k = 0; k < kc; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + k * 16);
        __m256 b1 = _mm256_loadu_ps(Bp + k * 16 + 8);
        __m256 a;
        a = _mm256_broadcast_ss(a0 + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(a1 + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(a2 + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(a3 + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(a4 + k); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(a5 + k); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C_blk + 0 * ldc, c00); _mm256_storeu_ps(C_blk + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C_blk + 1 * ldc, c10); _mm256_storeu_ps(C_blk + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C_blk + 2 * ldc, c20); _mm256_storeu_ps(C_blk + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C_blk + 3 * ldc, c30); _mm256_storeu_ps(C_blk + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C_blk + 4 * ldc, c40); _mm256_storeu_ps(C_blk + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C_blk + 5 * ldc, c50); _mm256_storeu_ps(C_blk + 5 * ldc + 8, c51);
}

#define GEMM_KERNEL kernel_6x16_avx2
#include "gemm_driver.h"

void fe_gemm_avx2(const float *a, int lda, const float *b, int ldb,
                  float *c, int ldc, int M, int N, int K) {
    gemm_blocked(a, lda, b, ldb, c, ldc, M, N, K);
}

void fe_gemm_packed_avx2(const float *a, int lda, FeGemmPackB pack, const void *ctx,
                         float *c, int ldc, int M, int N, int K) {
    gemm_blocked_pack(a, lda, pack, ctx, c, ldc, M, N, K);
}

FeStatus fe_matmul_avx2(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C)          return FE_ERR_NULL;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K) return FE_ERR_SHAPE;
    if (C->shape[0] != M) return FE_ERR_SHAPE;
    if (C->shape[1] != N) return FE_ERR_SHAPE;

    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    /* The kernel walks raw row-major memory. A strided view (a transpose, a
     * column slice) is not that, so leave it to the scalar fallback. */
    if (!fe_tensor_is_contiguous(A) || !fe_tensor_is_contiguous(B) ||
        !fe_tensor_is_contiguous(C)) return FE_ERR_SHAPE;

    gemm_blocked((const float *)A->data, K, (const float *)B->data, N,
                 (float *)C->data, N, M, N, K);
    return FE_OK;
}
