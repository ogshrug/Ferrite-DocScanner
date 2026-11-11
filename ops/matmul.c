#include "ops.h"
#include "backend.h"
#include <string.h>
#include <assert.h>

/*
 * Naive matmul: C = A @ B
 *
 * This is the reference implementation — correct but not optimized.
 * It exists to validate correctness before we add SIMD kernels, and
 * serves as the fallback path on hardware without vector support.
 *
 * Memory access pattern:
 *   A[i][k] — row-major access, stride 1 in inner loop. Cache-friendly.
 *   B[k][j] — column access, stride N in inner loop. Cache-unfriendly.
 *
 * This is the fundamental matmul performance problem.
 * The active SIMD backend (simd/backend.h) fixes this with tiling.
 */
FeStatus fe_matmul_scalar(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C) return FE_ERR_NULL;

    /* Shape validation */
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K)   return FE_ERR_SHAPE;
    if (C->shape[0] != M)   return FE_ERR_SHAPE;
    if (C->shape[1] != N)   return FE_ERR_SHAPE;

    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float       *c = (float *)C->data;

    /* Zero output — caller may pass uninitialized buffer */
    memset(c, 0, M * N * sizeof(float));

    /*
     * Standard triple loop.
     * Loop order i-k-j (not i-j-k) improves B access slightly
     * by keeping the k-stride on B in the innermost position,
     * but column access on B is still the bottleneck.
     */
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = a[i * K + k];
            for (int j = 0; j < N; j++) {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }

    return FE_OK;
}

/*
 * Public entry point: C = A @ B
 *
 * Dispatches through the SIMD backend table (simd/backend.h): the active
 * backend's matmul kernel runs when present, falling back to the scalar
 * reference implementation either on a scalar-only build or if the vector
 * path returns anything other than FE_OK. New backends (NEON) slot in via
 * the table without touching ops/.
 */
/*
 * The vector GEMM, spread over the thread pool by columns of C. Each task owns a
 * run of columns that is a multiple of 16 wide (the kernel's block width), and
 * every element is summed in the same order whoever computes it, so the result
 * does not depend on the thread count. Small products stay on one thread: below
 * roughly a million multiply-adds, waking workers costs more than it saves.
 */
typedef struct {
    const FeSimdOps *ops;
    const float *a, *b;
    float *c;
    int M, N, K, cols_per_task;
} GemmSplit;

static void gemm_task(void *p, int t) {
    const GemmSplit *g = (const GemmSplit *)p;
    int n0 = t * g->cols_per_task;
    int n = g->N - n0 < g->cols_per_task ? g->N - n0 : g->cols_per_task;
    if (n > 0)
        g->ops->gemm(g->a, g->K, g->b + n0, g->N, g->c + n0, g->N, g->M, n, g->K);
}

static void gemm_split(const FeSimdOps *ops, const FeTensor *A, const FeTensor *B,
                       FeTensor *C) {
    GemmSplit g = { ops, (const float *)A->data, (const float *)B->data,
                    (float *)C->data, A->shape[0], B->shape[1], A->shape[1], 0 };
    int threads = fe_get_num_threads();
    double work = (double)g.M * g.N * g.K;
    if (threads <= 1 || work < 1e6 || g.N < 64) {
        ops->gemm(g.a, g.K, g.b, g.N, g.c, g.N, g.M, g.N, g.K);
        return;
    }
    /* About four tasks per thread, so a slow core doesn't hold everyone up. */
    int tasks = threads * 4;
    int per = (g.N + tasks - 1) / tasks;
    per = (per + 15) / 16 * 16;
    g.cols_per_task = per;
    fe_parallel_for((g.N + per - 1) / per, gemm_task, &g);
}

FeStatus fe_matmul(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    /*
     * The active SIMD backend's tiled kernel packs an MC=64-row tile of A
     * and an NR=8-col tile of B once, then reuses the packed B tile across
     * every row in that A-tile — the packing cost is amortized over up to
     * 64 rows of compute. With M == 1 there is exactly one row: zero
     * reuse, so packing is pure overhead with nothing to amortize it
     * against.
     *
     * Measured on AcousticLeakNet's post-flatten FC layer
     * (M=1, K=65536, N=128): naive 1.79 ms vs. AVX2 5.65 ms — a 3.2x
     * *regression*, not a speedup. Route M==1 straight to scalar, which
     * is faster here and has no packing overhead to begin with.
     */
    if (A && A->ndim == 2 && A->shape[0] == 1) {
        return fe_matmul_scalar(A, B, C);
    }

    const FeSimdOps *ops = fe_simd_ops();
    if (ops->gemm && A && B && C && A->ndim == 2 && B->ndim == 2 && C->ndim == 2 &&
        A->dtype == DTYPE_FLOAT32 && B->dtype == DTYPE_FLOAT32 &&
        C->dtype == DTYPE_FLOAT32 &&
        B->shape[0] == A->shape[1] && C->shape[0] == A->shape[0] &&
        C->shape[1] == B->shape[1] &&
        fe_tensor_is_contiguous(A) && fe_tensor_is_contiguous(B) &&
        fe_tensor_is_contiguous(C)) {
        gemm_split(ops, A, B, C);
        return FE_OK;
    }
    if (ops->matmul) {
        FeStatus s = ops->matmul(A, B, C);
        if (s == FE_OK) return s;
        /* fall through to scalar on any vector-path failure */
    }
    return fe_matmul_scalar(A, B, C);
}

FeStatus fe_linear(const FeTensor *A, const FeTensor *W,
                   const FeTensor *b, FeTensor *C) {
    /* C = A @ W */
    FeStatus s = fe_matmul(A, W, C);
    if (s != FE_OK) return s;

    /* C += b (broadcast) */
    return fe_bias_add(C, b, C);
}
