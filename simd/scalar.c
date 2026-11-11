// simd/scalar.c — scalar-only SIMD backend (Section 2.5).
//
// The reference fallback. No vector instructions, correct on every target.
// Used when AVX2 is compiled out (FERRITE_NO_AVX2) or the CPU lacks it; the
// matmul slot routes to ops/fe_matmul_scalar, so the table stays symmetric
// with the vectorized backends.

#include "backend.h"

static void relu_scalar(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

static void add_scalar(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

static void mul_scalar(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

static const FeSimdOps k_scalar_ops = {
    .matmul = NULL,          /* resolved to fe_matmul_scalar by the caller */
    .gemm   = NULL,
    .gemm_packed = NULL,
    .relu   = relu_scalar,
    .add    = add_scalar,
    .mul    = mul_scalar,
};

const FeSimdOps *fe_simd_scalar_ops(void) {
    return &k_scalar_ops;
}