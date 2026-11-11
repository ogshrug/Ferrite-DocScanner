#include "bench.h"
#include <stdio.h>
#include <time.h>

/* CPUID and the AVX2 kernels exist only on x86; on other targets (ARM) the
 * header reports the NEON or scalar backend instead. */
#if defined(__x86_64__) || defined(__i386__)
#define FE_BENCH_X86 1
#include <cpuid.h>
#ifndef FERRITE_NO_AVX2
#include "matmul_avx2.h"
#endif
#endif

double fe_bench_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

double fe_bench_mean(int runs, void (*fn)(void *), void *arg) {
    fn(arg);  /* warmup */
    double start = fe_bench_ms();
    for (int i = 0; i < runs; i++) fn(arg);
    return (fe_bench_ms() - start) / runs;
}

double fe_bench_gflops(double ms, double n_flops) {
    return n_flops / (ms * 1e6);
}

/*
 * CPU brand string via CPUID leaf 0x80000002..0x80000004 (GCC/Clang only).
 * Falls back to "(unknown)" otherwise.
 */
void fe_bench_header(const char *name) {
    char brand[49] = {0};
#ifdef FE_BENCH_X86
    unsigned int regs[4];
    if (__get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]) &&
        regs[0] >= 0x80000004) {
        unsigned offset = 0;
        for (unsigned leaf = 0x80000002; leaf <= 0x80000004; leaf++) {
            __get_cpuid(leaf, &regs[0], &regs[1], &regs[2], &regs[3]);
            for (int r = 0; r < 4; r++)
                for (int b = 0; b < 4; b++)
                    brand[offset++] = (char)(regs[r] >> (8 * b));
        }
        for (int i = 48; i >= 0 && brand[i] == ' '; i--) brand[i] = '\0';
    } else {
        snprintf(brand, sizeof(brand), "unknown");
    }
#else
    snprintf(brand, sizeof(brand), "unknown (not x86)");
#endif

    printf("=== %s ===\n", name);
    printf("CPU:   %s\n", brand);
#if defined(FE_BENCH_X86) && !defined(FERRITE_NO_AVX2)
    printf("AVX2:  %s\n", fe_cpu_has_avx2() ? "yes" : "no");
#elif defined(__aarch64__) && !defined(FERRITE_NO_NEON)
    printf("SIMD:  NEON (AArch64)\n");
#else
    printf("SIMD:  scalar\n");
#endif
#ifdef __AVX2__
    printf("Build: compiled with -mavx2 (SIMD-kernel translations enabled)\n");
#else
    printf("Build: scalar build (SIMD library still AVX2)\n");
#endif
    printf("\n");
}