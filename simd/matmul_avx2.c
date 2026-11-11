#include "matmul_avx2.h"
#include <string.h>
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
 * Register-blocked GEMM: C = A @ B, all row-major and contiguous.
 *
 * The micro-kernel computes an MR x NR = 6 x 16 block of C held in 12 ymm
 * accumulators (6 rows x 2 vectors), which together with the two B vectors and
 * one broadcast A value fit in AVX2's 16 registers. Per step of k it does two
 * B loads, six broadcasts and twelve FMAs, with nothing spilled to memory.
 *
 * (The previous kernel kept a whole 64-row tile of accumulators "in registers";
 * with 16 registers available almost all of them lived in memory, so every FMA
 * also paid a load and a store.)
 *
 * Blocking: K is taken KC at a time so a packed B panel (KC x NR) stays in L1.
 * B is packed per 16-column strip and zero-padded at the right edge, so the
 * kernel never branches on N. A is read in place: six rows, each sequential in
 * k. Ragged edges (fewer than 6 rows or 16 columns left) run the same kernel
 * on a small local copy of C.
 */

#define MR 6
#define NR 16
#define KC 256

/* C_blk (row stride ldc) += A_blk (6 rows, row stride lda, kc long) @ Bp (kc x 16 packed). */
static inline void kernel_6x16(const float *A_blk, int lda, const float *Bp, int kc,
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
        __m256 b0 = _mm256_loadu_ps(Bp + k * NR);
        __m256 b1 = _mm256_loadu_ps(Bp + k * NR + 8);
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

    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float       *c = (float *)C->data;

    memset(c, 0, (size_t)M * N * sizeof(float));
    if (M == 0 || N == 0 || K == 0) return FE_OK;

    float Bp[KC * NR];           /* one packed B panel: 16 KB */
    float A_edge[MR * KC];       /* the last few rows of A, zero-padded to 6 */
    float C_edge[MR * NR];       /* a ragged block of C */

    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = K - k0 < KC ? K - k0 : KC;

        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = N - n0 < NR ? N - n0 : NR;

            /* Pack B[k0 : k0+kc, n0 : n0+nr], padding the strip to 16 columns. */
            for (int k = 0; k < kc; k++) {
                const float *src = b + (size_t)(k0 + k) * N + n0;
                float *dst = Bp + k * NR;
                memcpy(dst, src, (size_t)nr * sizeof(float));
                if (nr < NR) memset(dst + nr, 0, (size_t)(NR - nr) * sizeof(float));
            }

            for (int m0 = 0; m0 < M; m0 += MR) {
                int mr = M - m0 < MR ? M - m0 : MR;
                const float *A_blk = a + (size_t)m0 * K + k0;
                int lda = K;
                if (mr < MR) {
                    /* Fewer than six rows left: copy them, zero the rest. */
                    memset(A_edge, 0, sizeof(A_edge));
                    for (int i = 0; i < mr; i++)
                        memcpy(A_edge + i * KC, A_blk + (size_t)i * K, (size_t)kc * sizeof(float));
                    A_blk = A_edge;
                    lda = KC;
                }

                float *C_blk = c + (size_t)m0 * N + n0;
                if (mr == MR && nr == NR) {
                    kernel_6x16(A_blk, lda, Bp, kc, C_blk, N);
                } else {
                    memset(C_edge, 0, sizeof(C_edge));
                    for (int i = 0; i < mr; i++)
                        memcpy(C_edge + i * NR, C_blk + (size_t)i * N, (size_t)nr * sizeof(float));
                    kernel_6x16(A_blk, lda, Bp, kc, C_edge, NR);
                    for (int i = 0; i < mr; i++)
                        memcpy(C_blk + (size_t)i * N, C_edge + i * NR, (size_t)nr * sizeof(float));
                }
            }
        }
    }

    return FE_OK;
}
