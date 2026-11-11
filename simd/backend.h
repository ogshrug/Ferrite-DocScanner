// simd/backend.h — generalized SIMD dispatch (Section 2.5).
//
// Every op-family's SIMD path resolves through one table: buffer-level
// kernels for relu/add/mul, tensor-level for matmul (which returns an
// FeStatus so a failed vector path can fall back to the reference scalar
// kernel in ops/). The active backend is decided once by fe_simd_backend()
// (CPUID / compile-time constant) and cached; a NEON or scalar-only backend
// replaces simd/backend.c — or scalar.c below — without touching ops/.
#ifndef FERRITE_SIMD_BACKEND_H
#define FERRITE_SIMD_BACKEND_H

#include "types.h"
#include "tensor.h"

typedef enum {
    FE_SIMD_SCALAR = 0,   /* reference/portable kernels everywhere        */
    FE_SIMD_AVX2,         /* x86-64 AVX2 + FMA, CPUID-gated               */
    FE_SIMD_NEON          /* AArch64 NEON, compile-time (always present)  */
} FeSimdBackend;

/* One dispatch table per op family. `matmul` validates and runs the fast
 * kernel; callers fall back to fe_matmul_scalar when it returns non-FE_OK.
 * The buffer-level kernels are hot path — callers already validated. */
/* Fills Bp with the kc x nr block of a GEMM's B operand at (k0, n0), as kc rows of
 * 16 floats with columns nr..15 zeroed. Lets a caller supply B without storing it
 * (a convolution reads it from the input image). */
typedef void (*FeGemmPackB)(const void *ctx, int k0, int kc, int n0, int nr, float *Bp);

typedef struct {
    FeStatus (*matmul)(const FeTensor *A, const FeTensor *B, FeTensor *C);
    /* Raw-pointer GEMM with leading dimensions: overwrites the M x N block of c
     * with a @ b. NULL in the scalar table. Lets callers run a column range of
     * a larger product, which is how fe_matmul spreads work across threads. */
    void (*gemm)(const float *a, int lda, const float *b, int ldb,
                 float *c, int ldc, int M, int N, int K);
    /* The same with B supplied by a packing callback instead of a matrix. */
    void (*gemm_packed)(const float *a, int lda, FeGemmPackB pack, const void *ctx,
                        float *c, int ldc, int M, int N, int K);
    void (*relu)(const float *in, float *out, int n);
    void (*add) (const float *a, const float *b, float *out, int n);
    void (*mul) (const float *a, const float *b, float *out, int n);
} FeSimdOps;

/* Active backend for this CPU/build, selected once and cached. */
FeSimdBackend fe_simd_backend(void);

/* The active table. Never NULL. */
const FeSimdOps *fe_simd_ops(void);

/* The scalar-only table (simd/scalar.c), for fallback and scalar builds. */
const FeSimdOps *fe_simd_scalar_ops(void);

#endif /* FERRITE_SIMD_BACKEND_H */