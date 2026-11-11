// simd/backend.c — SIMD backend selection (Section 2.5).
//
// Chooses the active FeSimdOps table once and caches it:
//   - on AArch64 with NEON compiled in, the NEON kernels (NEON is part of the
//     64-bit ARM base architecture, so there is nothing to probe);
//   - on x86 with AVX2 compiled in AND reported by the CPU, the AVX2 kernels;
//   - otherwise the scalar table, whose matmul slot is NULL so the ops layer
//     resolves it to fe_matmul_scalar.
// Compile-out (FERRITE_NO_AVX2 / FERRITE_NO_NEON) is handled by the
// preprocessor, so a scalar-only build never references vector symbols.

#include "backend.h"

#if !defined(FERRITE_NO_NEON) && defined(__aarch64__)
#define FE_HAVE_NEON 1
#include "matmul_neon.h"
#endif

#if !defined(FERRITE_NO_AVX2) && !defined(FE_HAVE_NEON)
#define FE_HAVE_AVX2 1
#include "matmul_avx2.h"
#include "elementwise_avx2.h"
#endif

#ifdef FE_HAVE_NEON
static const FeSimdOps k_neon_ops = {
    .matmul = fe_matmul_neon,
    .gemm   = fe_gemm_neon,
    .gemm_packed = fe_gemm_packed_neon,
    .relu   = fe_relu_neon,
    .add    = fe_add_neon,
    .mul    = fe_mul_neon,
};
#endif

#ifdef FE_HAVE_AVX2
static const FeSimdOps k_avx2_ops = {
    .matmul = fe_matmul_avx2,
    .gemm   = fe_gemm_avx2,
    .gemm_packed = fe_gemm_packed_avx2,
    .relu   = fe_relu_avx2,
    .add    = fe_add_avx2,
    .mul    = fe_mul_avx2,
};
#endif

FeSimdBackend fe_simd_backend(void) {
#if defined(FE_HAVE_NEON)
    return FE_SIMD_NEON;
#elif defined(FE_HAVE_AVX2)
    static int cached = -1;
    if (cached < 0)
        cached = fe_cpu_has_avx2() ? (int)FE_SIMD_AVX2 : (int)FE_SIMD_SCALAR;
    return (FeSimdBackend)cached;
#else
    return FE_SIMD_SCALAR;
#endif
}

const FeSimdOps *fe_simd_ops(void) {
#if defined(FE_HAVE_NEON)
    return &k_neon_ops;
#elif defined(FE_HAVE_AVX2)
    static const FeSimdOps *table = NULL;
    if (!table) {
        if (fe_simd_backend() == FE_SIMD_AVX2)
            table = &k_avx2_ops;
        else
            table = fe_simd_scalar_ops();
    }
    return table;
#else
    return fe_simd_scalar_ops();
#endif
}
