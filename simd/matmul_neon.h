// simd/matmul_neon.h
#ifndef FERRITE_MATMUL_NEON_H
#define FERRITE_MATMUL_NEON_H

#include "types.h"
#include "tensor.h"
#include "backend.h"

/*
 * AArch64 NEON kernels (simd/matmul_neon.c). Compiled to nothing on other targets;
 * the backend table only references them when __aarch64__ is defined.
 *
 * fe_gemm_neon overwrites the M x N block of C with A @ B; all three are row-major
 * with leading dimensions lda / ldb / ldc. fe_matmul_neon is the tensor form, for
 * contiguous float32 tensors (anything else returns an error so the caller falls
 * back to the scalar kernel).
 */
void     fe_gemm_neon(const float *a, int lda, const float *b, int ldb,
                      float *c, int ldc, int M, int N, int K);
void fe_gemm_packed_neon(const float *a, int lda, FeGemmPackB pack, const void *ctx,
                          float *c, int ldc, int M, int N, int K);
FeStatus fe_matmul_neon(const FeTensor *A, const FeTensor *B, FeTensor *C);
void     fe_relu_neon(const float *in, float *out, int n);
void     fe_add_neon (const float *a, const float *b, float *out, int n);
void     fe_mul_neon (const float *a, const float *b, float *out, int n);

#endif // FERRITE_MATMUL_NEON_H
