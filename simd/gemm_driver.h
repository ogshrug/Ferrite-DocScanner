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
 *     gemm_blocked_pack(a, lda, pack, ctx, c, ldc, M, N, K)
 *         C[M x N] = A[M x K] @ B[K x N], where B is never read directly: pack(ctx, k0,
 *         kc, n0, nr, Bp) writes the kc x nr block of B at (k0, n0) into Bp as kc rows of
 *         16, zero-padding columns nr..15. A convolution passes a pack function that
 *         reads its input image directly, so no im2col matrix is ever built.
 *     gemm_blocked(a, lda, b, ldb, c, ldc, M, N, K)
 *         the same with B an ordinary row-major matrix.
 *
 * Blocking (GotoBLAS-style): N is taken GEMM_NC columns at a time, so that block of C
 * stays in cache while K is walked GEMM_KC at a time; for each (n, k) block every
 * 16-column panel of B is packed once and reused by all M rows. Each C element is
 * summed over k in the same order whatever the block sizes, so splitting C by
 * columns (the threaded path) gives identical results.
 */
#ifndef GEMM_KERNEL
#error "define GEMM_KERNEL before including gemm_driver.h"
#endif

#include <string.h>
#include "backend.h"

#define GEMM_MR 6
#define GEMM_NR 16
#define GEMM_KC 256
#define GEMM_NC 128                         /* 8 panels: 128 KB of packed B */

static void gemm_blocked_pack(const float *a, int lda, FeGemmPackB pack, const void *ctx,
                              float *c, int ldc, int M, int N, int K) {
    for (int i = 0; i < M; i++) memset(c + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
    if (M <= 0 || N <= 0 || K <= 0) return;

    float Bp[GEMM_NC / GEMM_NR][GEMM_KC * GEMM_NR];
    float A_edge[GEMM_MR * GEMM_KC];
    float C_edge[GEMM_MR * GEMM_NR];

    for (int nb = 0; nb < N; nb += GEMM_NC) {
        int ncols = N - nb < GEMM_NC ? N - nb : GEMM_NC;
        int panels = (ncols + GEMM_NR - 1) / GEMM_NR;

        for (int k0 = 0; k0 < K; k0 += GEMM_KC) {
            int kc = K - k0 < GEMM_KC ? K - k0 : GEMM_KC;
            for (int p = 0; p < panels; p++) {
                int n0 = nb + p * GEMM_NR;
                int nr = N - n0 < GEMM_NR ? N - n0 : GEMM_NR;
                pack(ctx, k0, kc, n0, nr, Bp[p]);
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
                for (int p = 0; p < panels; p++) {
                    int n0 = nb + p * GEMM_NR;
                    int nr = N - n0 < GEMM_NR ? N - n0 : GEMM_NR;
                    float *C_blk = c + (size_t)m0 * ldc + n0;
                    if (mr == GEMM_MR && nr == GEMM_NR) {
                        GEMM_KERNEL(A_blk, a_ld, Bp[p], kc, C_blk, ldc);
                    } else {
                        memset(C_edge, 0, sizeof(C_edge));
                        for (int i = 0; i < mr; i++)
                            memcpy(C_edge + i * GEMM_NR, C_blk + (size_t)i * ldc,
                                   (size_t)nr * sizeof(float));
                        GEMM_KERNEL(A_blk, a_ld, Bp[p], kc, C_edge, GEMM_NR);
                        for (int i = 0; i < mr; i++)
                            memcpy(C_blk + (size_t)i * ldc, C_edge + i * GEMM_NR,
                                   (size_t)nr * sizeof(float));
                    }
                }
            }
        }
    }
}

typedef struct { const float *b; int ldb; } GemmMatrixB;

static void gemm_pack_matrix(const void *ctx, int k0, int kc, int n0, int nr, float *Bp) {
    const GemmMatrixB *m = (const GemmMatrixB *)ctx;
    for (int k = 0; k < kc; k++) {
        const float *src = m->b + (size_t)(k0 + k) * m->ldb + n0;
        float *dst = Bp + k * GEMM_NR;
        memcpy(dst, src, (size_t)nr * sizeof(float));
        if (nr < GEMM_NR) memset(dst + nr, 0, (size_t)(GEMM_NR - nr) * sizeof(float));
    }
}

static void gemm_blocked(const float *a, int lda, const float *b, int ldb,
                         float *c, int ldc, int M, int N, int K) {
    GemmMatrixB m = { b, ldb };
    gemm_blocked_pack(a, lda, gemm_pack_matrix, &m, c, ldc, M, N, K);
}
