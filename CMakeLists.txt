# CMakeLists.txt — Ferrite build.
#
# Mirrors the Makefile: same sources, same binaries, same flags.
# Canonical build story (roadmap Phase 0); requires GCC or Clang —
# MSVC lacks clock_gettime/_POSIX_C_SOURCE used by tools/profiler.c.
#
#   cmake -S . -B build -G Ninja
#   cmake --build build              # all tests
#   ctest --test-dir build           # run the suite
#
# Benches and demo are opt-in:  cmake --build build --target bench_avx2

cmake_minimum_required(VERSION 3.16)
project(Ferrite C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Threads: core/platform.c uses pthread for the thread seam; the planner,
# parallel GEMM, and scheduler use it when enabled. Resolve once up front so
# every library can reference Threads::Threads regardless of gating.
find_package(Threads REQUIRED)

if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
        "Ferrite requires GCC or Clang (POSIX clock + sanitizers). "
        "Use CLion's bundled MinGW, MSYS2, or WSL.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)   # tests run under ASan/UBSan by default
endif()

option(FERRITE_SANITIZE "Build everything with ASan+UBSan" ON)

# --- Subsystem gating (Section 2.3) ---
#
# Defaults are all ON so the existing desktop build is unchanged. Device
# presets (ferrite-runtime-only) turn the host-only subsystems OFF; every
# library and its consumers are conditionally defined below, so turning a
# subsystem off excludes its sources, its link deps, and the code that
# references it. The compile definition is what lets in-tree code (e.g. the
# profiler hooks in the exec plan) compile out cleanly.
option(FERRITE_ENABLE_IMPORTER  "Host: ONNX protobuf loading"     ON)
option(FERRITE_ENABLE_OPTIM     "Host: optimization passes"       ON)
option(FERRITE_ENABLE_COMPILER  "Host: stage-14 linear IR"        ON)
option(FERRITE_ENABLE_PARALLEL  "Host: thread pool + DAG"         ON)
option(FERRITE_ENABLE_QUANT     "Quantization kernels + engine"   ON)
option(FERRITE_ENABLE_PROFILER  "Timing profiler (POSIX clock)"   ON)
# The vector backend follows the target: AVX2 on x86-64, NEON on AArch64.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" FERRITE_PROC_LOWER)
if(FERRITE_PROC_LOWER MATCHES "^(x86_64|amd64|x64|i[3-6]86)$")
    set(FERRITE_AVX2_DEFAULT ON)
else()
    set(FERRITE_AVX2_DEFAULT OFF)
endif()
if(FERRITE_PROC_LOWER MATCHES "^(aarch64|arm64)$")
    set(FERRITE_NEON_DEFAULT ON)
else()
    set(FERRITE_NEON_DEFAULT OFF)
endif()
option(FERRITE_ENABLE_AVX2      "x86 AVX2 SIMD backend"           ${FERRITE_AVX2_DEFAULT})
option(FERRITE_ENABLE_NEON      "AArch64 NEON SIMD backend"       ${FERRITE_NEON_DEFAULT})
option(FERRITE_ENABLE_FLOAT64   "Double-precision tensor support" ON)

foreach(subsys IMPORTER OPTIM COMPILER PARALLEL QUANT PROFILER AVX2 NEON)
    if(NOT ${FERRITE_ENABLE_${subsys}})
        add_compile_definitions(FERRITE_NO_${subsys})
    endif()
endforeach()

if(NOT FERRITE_ENABLE_FLOAT64)
    add_compile_definitions(FERRITE_NO_FLOAT64)
endif()

# x86 check (Section 3): the only SIMD backend today is AVX2. Requesting it on
# a non-x86 target is a configuration error, not a silent scalar downgrade.
if(FERRITE_ENABLE_NEON AND NOT FERRITE_PROC_LOWER MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "FERRITE_ENABLE_NEON=ON needs an AArch64 target, "
                        "not ${CMAKE_SYSTEM_PROCESSOR}.")
endif()
if(FERRITE_ENABLE_AVX2)
    if(FERRITE_PROC_LOWER MATCHES "aarch64|arm|riscv|mips|ppc")
        message(FATAL_ERROR
            "FERRITE_ENABLE_AVX2=ON on ${CMAKE_SYSTEM_PROCESSOR}, but AVX2 is "
            "x86-only. Use the target's toolchain file (see "
            "cmake/raspberry-pi-zero-w.toolchain.cmake) which sets it OFF.")
    endif()
endif()

add_compile_options(-Wall -Wextra -g)
add_compile_definitions(_POSIX_C_SOURCE=199309L)

# Probe whether the toolchain actually ships the sanitizer runtimes
# (CLion's bundled MinGW does not; MSYS2 gcc and Linux gcc do).
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined")
check_c_source_compiles("int main(void){return 0;}" FERRITE_HAVE_SANITIZERS)
unset(CMAKE_REQUIRED_FLAGS)

if(FERRITE_SANITIZE AND FERRITE_HAVE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    set(FERRITE_SANITIZE_ENABLED ON)
elseif(FERRITE_SANITIZE)
    message(STATUS "ASan/UBSan requested but unavailable in this toolchain - building without")
endif()

option(FERRITE_COVERAGE "Build with gcov coverage instrumentation" OFF)
if(FERRITE_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# On Windows, executables need libwinpthread-1.dll (clock_gettime in the
# profiler). This toolchain cannot link its static winpthread against its
# own CRT, so instead ship the correct DLL beside every executable: the
# Windows loader searches the exe directory before PATH, which defeats
# stale copies of the DLL found there.
if(WIN32)
    get_filename_component(FERRITE_MINGW_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
    set(FERRITE_PTHREAD_DLL "${FERRITE_MINGW_BIN}/libwinpthread-1.dll")
endif()

function(ferrite_exe target)
    if(WIN32 AND EXISTS "${FERRITE_PTHREAD_DLL}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FERRITE_PTHREAD_DLL}" "$<TARGET_FILE_DIR:${target}>")
    endif()
endfunction()

if(UNIX)
    link_libraries(m)
endif()

# --- Configurable capacity ceilings (Section 2.4) ---
#
# Compile-time constants, not runtime knobs: the "fixed capacity" invariant
# is preserved, but each target can set its own ceiling at configure time.
# Device builds must size these to the models they ship — see the port docs,
# not the desktop defaults.
set(FERRITE_MAX_DIMS   8    CACHE STRING "Tensor rank ceiling")
set(FE_MAX_NODES       512  CACHE STRING "Graph node ceiling")
set(FE_MAX_TENSORS     1024 CACHE STRING "Tensor registry ceiling")
set(FE_MAX_ALLOCS      1024 CACHE STRING "Planner allocation ceiling")

# Planner tables are indexed by tensor index, so the allocation ceiling can
# never be smaller than the registry ceiling. A device preset that overrides
# one without the other is a configure-time error, not an OOB runtime
# surprise. (The same invariant is enforced in planner/memory_planner.h by
# a _Static_assert; this catches a misconfigured preset before compilation.)
if(FE_MAX_ALLOCS LESS FE_MAX_TENSORS)
    message(FATAL_ERROR "FE_MAX_ALLOCS (${FE_MAX_ALLOCS}) must be "
                        ">= FE_MAX_TENSORS (${FE_MAX_TENSORS}): planner "
                        "tables are indexed by tensor index")
endif()

# Activation-buffer alignment (Section 4.3): 64 keeps SIMD backends' loads/
# stores aligned; a scalar-only device build (no SIMD backend) drops to the
# natural scalar alignment to stop wasting padding. Must be a power of two.
set(FERRITE_PLANNER_ALIGN 64 CACHE STRING "Activation-buffer alignment")

# Generate core/config.h into the build tree (ahead of the source core/ dir in
# the include path so <config.h> resolves to the configured values, not the
# committed Makefile defaults).
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(core/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/config.h @ONLY)

# Flat include path, matching the Makefile's global -I flags: sources
# include headers by bare name ("tensor.h", "profiler.h"). The generated
# config.h dir sits first so <config.h> resolves to the configured values.
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
set(FERRITE_DIRS core graph ops simd planner runtime importer quantization optim tools)
foreach(dir ${FERRITE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endforeach()

# --- Libraries (one per subsystem; dependencies point downward only) ---

add_library(ferrite_core STATIC
    core/tensor.c
    core/tensor_ser.c
    core/allocator.c
    core/log.c
    core/fp16.c
    core/platform.c
)
target_link_libraries(ferrite_core PUBLIC Threads::Threads)

add_library(ferrite_graph STATIC graph/graph.c)
target_link_libraries(ferrite_graph PUBLIC ferrite_core)

# SIMD kernels are compiled with AVX2+FMA when the backend is enabled;
# runtime CPUID detection (fe_simd_backend) decides whether they execute.
# A scalar-only build (FERRITE_ENABLE_AVX2=OFF) compiles the dispatch table
# without the AVX2 sources; the preprocessor guards inside backend.c resolve
# the table to scalar kernels.
add_library(ferrite_simd STATIC
    simd/backend.c
    simd/scalar.c
)
if(FERRITE_ENABLE_AVX2)
    target_sources(ferrite_simd PRIVATE
        simd/matmul_avx2.c
        simd/elementwise_avx2.c
    )
    target_compile_options(ferrite_simd PRIVATE -O3 -mavx2 -mfma)
endif()
if(FERRITE_ENABLE_NEON)
    target_sources(ferrite_simd PRIVATE simd/matmul_neon.c)
    target_compile_options(ferrite_simd PRIVATE -O3)
endif()
target_link_libraries(ferrite_simd PUBLIC ferrite_core)

add_library(ferrite_ops STATIC
    ops/matmul.c
    ops/activations.c
    ops/conv1d.c
    ops/conv2d.c
    ops/elementwise.c
    ops/gemm.c
    ops/math.c
    ops/norm.c
    ops/pool.c
    ops/rand.c
    ops/reduce.c
    ops/sequence.c
    ops/stability.c
    ops/vision.c
    ops/threads.c
)
target_link_libraries(ferrite_ops PUBLIC ferrite_core ferrite_simd)

add_library(ferrite_planner STATIC planner/memory_planner.c)
target_link_libraries(ferrite_planner PUBLIC ferrite_graph)

# Device-runtime library (Section 2.2): kernel dispatch + the FEMD artifact
# loader/runner. This is the ONLY library a target device compiles — it pulls
# core/graph/ops/simd/planner (and, conditionally, quant) but never
# optim's shape inference, and no ONNX parser. The host runtime links it too
# (kernels + artifact both live here), so there is exactly one TU providing
# fe_kernels_get.
add_library(ferrite_device STATIC
    runtime/kernels.c
    runtime/model_ser.c
)
target_link_libraries(ferrite_device PUBLIC
    ferrite_core ferrite_graph ferrite_ops ferrite_simd ferrite_planner
)
if(FERRITE_ENABLE_QUANT)
    target_link_libraries(ferrite_device PUBLIC ferrite_quant)
endif()
if(FERRITE_ENABLE_PROFILER)
    target_link_libraries(ferrite_device PUBLIC ferrite_tools)
endif()

# Engine calls the profiler (engine.c); Tools here tofix the Makefile's
# old test_engine rule. Exec plan (Stage 13) lives alongside the engine;
# quant.h supplies the engine-path INT8 kernels (Stage 15). Kernels live in
# their own library (ferrite_device) so the device runtime links them
# without optim's shape inference.
add_library(ferrite_runtime STATIC
    runtime/engine.c
    runtime/exec_plan.c
)
target_link_libraries(ferrite_runtime PUBLIC ferrite_device)
if(FERRITE_ENABLE_OPTIM)
    target_link_libraries(ferrite_runtime PUBLIC ferrite_optim)
endif()

if(FERRITE_ENABLE_IMPORTER)
    add_library(ferrite_importer STATIC importer/onnx.c)
    target_link_libraries(ferrite_importer PUBLIC ferrite_graph ferrite_optim)
endif()

if(FERRITE_ENABLE_QUANT)
    add_library(ferrite_quant STATIC quantization/quant.c)
    target_link_libraries(ferrite_quant PUBLIC ferrite_core ferrite_graph)
endif()

# Optimization passes (Stage 12): shape infer, folding, fusion, CSE, DCE.
if(FERRITE_ENABLE_OPTIM)
    add_library(ferrite_optim STATIC
        optim/shape_infer.c
        optim/constant_fold.c
        optim/dead_elim.c
        optim/simplify.c
        optim/cse.c
        optim/fusion.c
        optim/optim.c
    )
    target_link_libraries(ferrite_optim PUBLIC ferrite_graph ferrite_ops)
endif()

if(FERRITE_ENABLE_PROFILER)
    add_library(ferrite_tools STATIC tools/profiler.c tools/bench.c)
endif()

if(FERRITE_ENABLE_PARALLEL)
    # Stage 11 parallel runtime: worker pool, row-parallel GEMM, DAG scheduler.
    add_library(ferrite_parallel STATIC
        parallel/threadpool.c
        parallel/parallel_gemm.c
        parallel/scheduler.c
    )
    target_link_libraries(ferrite_parallel PUBLIC ferrite_core ferrite_graph Threads::Threads)
endif()

if(FERRITE_ENABLE_COMPILER)
    # Stage 14 compiler: linear IR, lowering, kernel selection, scheduling.
    add_library(ferrite_compiler STATIC
        compiler/lower.c
        compiler/codegen.c
        compiler/schedule.c
    )
    target_link_libraries(ferrite_compiler PUBLIC ferrite_graph ferrite_ops ferrite_core)
endif()

# --- Tests (built by default, registered with CTest) ---

enable_testing()

add_executable(test_tensor      tests/test_tensor.c)
add_executable(test_allocator   tests/test_allocator.c)
add_executable(test_ops         tests/test_ops.c)
add_executable(test_ops3        tests/test_ops3.c)
add_executable(test_vision      tests/test_vision.c)
if(FERRITE_ENABLE_AVX2)
    add_executable(test_simd        tests/test_simd.c)
endif()
add_executable(test_graph       tests/test_graph.c)
add_executable(test_engine      tests/test_engine.c)
add_executable(test_onnx        tests/test_onnx.c)
add_executable(test_planner     tests/test_planner.c)
add_executable(test_conv1d      tests/test_conv1d.c)
add_executable(test_profiler    tests/test_profiler.c)
add_executable(test_quant       tests/test_quant.c)
add_executable(test_tensor_ser  tests/test_tensor_ser.c)
add_executable(test_opt         tests/test_opt.c)

add_executable(test_parallel   tests/test_parallel.c)
add_executable(test_compiler   tests/test_compiler.c)
add_executable(test_artifact   tests/test_artifact.c)
add_executable(test_error_contract tests/test_error_contract.c)

# Continuous-fuzzing seam (Section 1.3): the libFuzzer entry point
# (fuzz/fuzz_onnx.c) plus a standalone corpus-replay driver (fuzz_main.c)
# for toolchains without clang's fuzzer runtime. Both are host-only — the
# importer is compiled out of device builds (2.3), so so is the fuzzer.
if(FERRITE_ENABLE_IMPORTER)
    add_executable(fuzz_runner fuzz/fuzz_onnx.c fuzz/fuzz_main.c)
    target_link_libraries(fuzz_runner ferrite_core ferrite_graph ferrite_importer)

    include(CheckCSourceCompiles)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        # Probe for the libFuzzer runtime. check_c_source_compiles links the
        # probe as an executable, and -fsanitize=fuzzer supplies its own main(),
        # so the probe must NOT define main() (a plain "int main(...)" body
        # yields a duplicate-symbol link error and the target is never built).
        # A mainless TU that only defines the fuzzer entry point links exactly
        # when the runtime is present.
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer,address,undefined")
        check_c_source_compiles(
            "#include <stddef.h>\nint LLVMFuzzerTestOneInput(const unsigned char *data, size_t size){ (void)data; (void)size; return 0; }"
            FERRITE_HAVE_FUZZER)
        unset(CMAKE_REQUIRED_FLAGS)
        if(FERRITE_HAVE_FUZZER)
            add_executable(fuzz_onnx_fuzzer EXCLUDE_FROM_ALL
                fuzz/fuzz_onnx.c fuzz/fuzz_onnx_libfuzzer.c)
            target_link_libraries(fuzz_onnx_fuzzer ferrite_core ferrite_graph ferrite_importer)
            target_compile_options(fuzz_onnx_fuzzer PRIVATE
                -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -g)
            target_link_options(fuzz_onnx_fuzzer PRIVATE
                -fsanitize=fuzzer,address,undefined)
        endif()
    endif()
else()
    message(STATUS "ferrite-fuzz targets disabled: FERRITE_ENABLE_IMPORTER=OFF")
endif()

target_link_libraries(test_tensor      ferrite_core)
target_link_libraries(test_allocator   ferrite_core)
target_link_libraries(test_ops         ferrite_core ferrite_ops ferrite_simd)
target_link_libraries(test_ops3        ferrite_core ferrite_ops ferrite_simd)
target_link_libraries(test_vision      ferrite_core ferrite_ops ferrite_simd)
if(FERRITE_ENABLE_AVX2)
    target_link_libraries(test_simd    ferrite_core ferrite_ops ferrite_simd)
endif()
target_link_libraries(test_graph       ferrite_core ferrite_graph)
target_link_libraries(test_engine      ferrite_runtime ferrite_importer)
target_link_libraries(test_onnx        ferrite_core ferrite_graph ferrite_importer)
target_link_libraries(test_planner     ferrite_core ferrite_graph ferrite_planner)
target_link_libraries(test_conv1d      ferrite_core ferrite_ops)
target_link_libraries(test_profiler    ferrite_core ferrite_graph ferrite_ops ferrite_simd ferrite_runtime ferrite_tools)
target_link_libraries(test_quant       ferrite_core ferrite_quant ferrite_runtime ferrite_importer)
target_link_libraries(test_tensor_ser  ferrite_core)
target_link_libraries(test_opt         ferrite_core ferrite_graph ferrite_optim ferrite_runtime)
target_link_libraries(test_parallel    ferrite_core ferrite_graph ferrite_ops ferrite_parallel)
target_link_libraries(test_compiler    ferrite_core ferrite_graph ferrite_ops ferrite_compiler ferrite_runtime)
target_link_libraries(test_artifact    ferrite_runtime ferrite_device ferrite_importer)
target_link_libraries(test_error_contract ferrite_runtime ferrite_importer)
if(FERRITE_ENABLE_IMPORTER)
    ferrite_exe(fuzz_runner)
    add_test(NAME fuzz_runner COMMAND fuzz_runner
        fuzz/corpus/seed_tiny_mlp.onnx tests/tiny_mlp.onnx
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

foreach(t test_tensor test_allocator test_ops test_ops3 test_vision test_graph test_engine
        test_onnx test_planner test_conv1d test_profiler test_quant
        test_tensor_ser test_opt test_parallel test_compiler test_artifact
        test_error_contract)
    ferrite_exe(${t})
    target_compile_options(${t} PRIVATE -UNDEBUG)
    # Run from the source root: tests reference fixtures by relative
    # path ("tests/tiny_mlp.onnx"), matching the Makefile's cwd.
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
if(FERRITE_ENABLE_AVX2)
    ferrite_exe(test_simd)
    target_compile_options(test_simd PRIVATE -UNDEBUG)
    add_test(NAME test_simd COMMAND test_simd WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# --- Benchmarks and demo (opt-in, excluded from the default build) ---

# Artifact compiler (Section 2.2): ONNX model -> FEMD artifact (+ --emit-header).
# A host tool; the device never sees ONNX or the optimizer.
add_executable(ferrite_compile tools/ferrite_compile.c)
target_link_libraries(ferrite_compile PRIVATE
    ferrite_device ferrite_importer ferrite_planner ferrite_optim)
ferrite_exe(ferrite_compile)

add_executable(bench_matmul EXCLUDE_FROM_ALL tools/bench_matmul.c)
target_link_libraries(bench_matmul PRIVATE ferrite_core ferrite_ops ferrite_simd)
target_compile_options(bench_matmul PRIVATE -O2)
ferrite_exe(bench_matmul)

# The AVX2 benches are meaningless without the backend — exclude them from
# cross builds (Section 3) rather than tripping the FATAL_ERROR above.
if(FERRITE_ENABLE_AVX2)
    add_executable(bench_matmul_avx2 EXCLUDE_FROM_ALL tools/bench_matmul.c)
    target_link_libraries(bench_matmul_avx2 PRIVATE ferrite_core ferrite_ops ferrite_simd)
    target_compile_options(bench_matmul_avx2 PRIVATE -O3 -mavx2 -mfma)
    ferrite_exe(bench_matmul_avx2)

    add_executable(bench_avx2 EXCLUDE_FROM_ALL tools/bench_avx2.c)
    target_link_libraries(bench_avx2 PRIVATE ferrite_core ferrite_ops ferrite_simd)
    target_compile_options(bench_avx2 PRIVATE -O3 -mavx2 -mfma)
    ferrite_exe(bench_avx2)
endif()

add_executable(demo EXCLUDE_FROM_ALL tools/demo_acousticleaknet.c)
target_link_libraries(demo PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_runtime ferrite_importer ferrite_tools)
target_compile_options(demo PRIVATE -O2)
ferrite_exe(demo)

# MNIST demo. Needs tools/mnist_samples.h, generated by tools/train_mnist.py,
# so it is opt-in like the other demos rather than part of the default build.
add_executable(demo_mnist EXCLUDE_FROM_ALL tools/demo_mnist.c)
target_link_libraries(demo_mnist PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_runtime ferrite_importer)
target_compile_options(demo_mnist PRIVATE -O2)
ferrite_exe(demo_mnist)

# Measures Ferrite's own accuracy over the full MNIST test set.
add_executable(mnist_eval EXCLUDE_FROM_ALL tools/mnist_eval.c)
target_link_libraries(mnist_eval PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_runtime ferrite_importer)
target_compile_options(mnist_eval PRIVATE -O2)
ferrite_exe(mnist_eval)

# Model-relevant benchmark: end-to-end latency + per-op profile + memory
# + naive-vs-AVX2 table derived from the loaded model's MatMul shapes.
add_executable(bench_model EXCLUDE_FROM_ALL tools/bench_model.c)
target_link_libraries(bench_model PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_planner ferrite_runtime ferrite_importer ferrite_tools)
target_compile_options(bench_model PRIVATE -O2)
ferrite_exe(bench_model)

# Golden runner (Phase D): load an ONNX model, feed one input .bin, write
# one output .bin. Drives tools/golden_compare.py's Ferrite side.
add_executable(run_model EXCLUDE_FROM_ALL tools/run_model.c)
target_link_libraries(run_model PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_planner ferrite_runtime ferrite_importer)
target_compile_options(run_model PRIVATE -O2)
ferrite_exe(run_model)

# Image-model runner with heap buffers and timing; drives tools/vision_compare.py.
add_executable(run_vision EXCLUDE_FROM_ALL tools/run_vision.c)
target_link_libraries(run_vision PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_planner ferrite_runtime ferrite_importer ferrite_tools)
target_compile_options(run_vision PRIVATE -O2)
ferrite_exe(run_vision)
