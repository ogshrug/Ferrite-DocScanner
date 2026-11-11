/* simd/gemm_driver.h — the cache-blocking loop shared by the vector GEMM kernels.
 *
 * Not a normal header: a backend defines
 *
 *     static inline void GEMM_KERNEL(const float *A_blk, int lda, const float *Bp,
 *                                    int kc, float *C_blk, int ldc);
 *
 * which adds a 6 x 16 block (A_blk: 6 rows of length kc, row stride lda; Bp: kc rows
 * of 16 packed floats) into C_blk (row stride ldc), then includes this file to get
 *
 *     static void gemm_blocked(const float *a, int lda, const float *b, int ldb,
 *                              float *c, int ldc, int M, int N, int K);
 *
 * which overwrites the M x N block of C with A @ B. A, B and C are row-major with
 * the given leading dimensions, so a caller can hand it a column range of a larger
 * matrix (that is how the threaded path splits the work).
 *
 * K is taken GEMM_KC at a time so a packed B panel stays in L1. B is packed per
 * 16-column strip and zero-padded at the right edge, so the kernel never branches on
 * N; A is read in place. Ragged edges (fewer than 6 rows or 16 columns left) run the
 * same kernel on small local copies. Every element of C is summed over k in the same
 * order whatever the block sizes, so splitting C by columns gives identical results.
 */
#ifndef GEMM_KERNEL
#error "define GEMM_KERNEL before including gemm_driver.h"
#endif

#include <string.h>

#define GEMM_MR 6
#define GEMM_NR 16
#define GEMM_KC 256

static void gemm_blocked(const float *a, int lda, const float *b, int ldb,
                         float *c, int ldc, int M, int N, int K) {
    for (int i = 0; i < M; i++) memset(c + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
    if (M <= 0 || N <= 0 || K <= 0) return;

    float Bp[GEMM_KC * GEMM_NR];            /* one packed B panel: 16 KB */
    float A_edge[GEMM_MR * GEMM_KC];        /* the last few rows of A, zero-padded */
    float C_edge[GEMM_MR * GEMM_NR];        /* a ragged block of C */

    for (int k0 = 0; k0 < K; k0 += GEMM_KC) {
        int kc = K - k0 < GEMM_KC ? K - k0 : GEMM_KC;

        for (int n0 = 0; n0 < N; n0 += GEMM_NR) {
            int nr = N - n0 < GEMM_NR ? N - n0 : GEMM_NR;

            for (int k = 0; k < kc; k++) {
                const float *src = b + (size_t)(k0 + k) * ldb + n0;
                float *dst = Bp + k * GEMM_NR;
                memcpy(dst, src, (size_t)nr * sizeof(float));
                if (nr < GEMM_NR) memset(dst + nr, 0, (size_t)(GEMM_NR - nr) * sizeof(float));
            }

            for (int m0 = 0; m0 < M; m0 += GEMM_MR) {
                int mr = M - m0 < GEMM_MR ? M - m0 : GEMM_MR;
                const float *A_blk = a + (size_t)m0 * lda + k0;
                int a_ld = lda;
                if (mr < GEMM_MR) {
                    memset(A_edge, 0, sizeof(A_edge));
                    for (int i = 0; i < mr; i++)
                        memcpy(A_edge + i * GEMM_KC, A_blk + (size_t)i * lda,
                               (size_t)kc * sizeof(float));
                    A_blk = A_edge;
                    a_ld = GEMM_KC;
                }

                float *C_blk = c + (size_t)m0 * ldc + n0;
                if (mr == GEMM_MR && nr == GEMM_NR) {
                    GEMM_KERNEL(A_blk, a_ld, Bp, kc, C_blk, ldc);
                } else {
                    memset(C_edge, 0, sizeof(C_edge));
                    for (int i = 0; i < mr; i++)
                        memcpy(C_edge + i * GEMM_NR, C_blk + (size_t)i * ldc,
                               (size_t)nr * sizeof(float));
                    GEMM_KERNEL(A_blk, a_ld, Bp, kc, C_edge, GEMM_NR);
                    for (int i = 0; i < mr; i++)
                        memcpy(C_blk + (size_t)i * ldc, C_edge + i * GEMM_NR,
                               (size_t)nr * sizeof(float));
                }
            }
        }
    }
}
