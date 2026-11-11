# idf/ferrite/CMakeLists.txt — ESP-IDF component wrapping the Ferrite device
# runtime (Section 4.1).
#
# Compiles the scalar-only subset of Ferrite (core + graph + ops + simd-scalar
# + planner + quant + runtime). optim/shape_infer.c is the one optimizer file
# that comes along: the runtime re-infers shapes on every plan build, so the
# rule table must be on the device too; the pass pipeline (fold/fusion/CSE/DCE)
# stays compiled out via FERRITE_NO_OPTIM. It is an ESP-IDF component: add this
# directory to EXTRA_COMPONENT_DIRS (or IDF_COMPONENT_MANAGER managed-components)
# and `PRIV_REQUIRES ferrite` from your app. See idf/README.md for the minimal
# project layout.

# Resolve the Ferrite repository root. This file normally lives at
# <repo>/idf/ferrite/, so two levels up from CMAKE_CURRENT_LIST_DIR is the
# repo root (matches both the documented `cd idf/demo` flow and the CI docker
# mount layout that puts the repo at /github/workspace). Fall back one more
# level for setups that copy the component one directory deeper.
get_filename_component(FERRITE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT EXISTS "${FERRITE_ROOT}/core/types.h")
    get_filename_component(FERRITE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

idf_component_register(
    SRCS
        "${FERRITE_ROOT}/core/allocator.c"
        "${FERRITE_ROOT}/core/tensor.c"
        "${FERRITE_ROOT}/core/log.c"
        "${FERRITE_ROOT}/core/fp16.c"
        "${FERRITE_ROOT}/core/platform_esp32.c"
        "${FERRITE_ROOT}/graph/graph.c"
        "${FERRITE_ROOT}/optim/shape_infer.c"
        "${FERRITE_ROOT}/ops/matmul.c"
        "${FERRITE_ROOT}/ops/activations.c"
        "${FERRITE_ROOT}/ops/conv1d.c"
        "${FERRITE_ROOT}/ops/conv2d.c"
        "${FERRITE_ROOT}/ops/elementwise.c"
        "${FERRITE_ROOT}/ops/gemm.c"
        "${FERRITE_ROOT}/ops/math.c"
        "${FERRITE_ROOT}/ops/norm.c"
        "${FERRITE_ROOT}/ops/pool.c"
        "${FERRITE_ROOT}/ops/sequence.c"
        "${FERRITE_ROOT}/ops/vision.c"
        "${FERRITE_ROOT}/simd/backend.c"
        "${FERRITE_ROOT}/simd/scalar.c"
        "${FERRITE_ROOT}/planner/memory_planner.c"
        "${FERRITE_ROOT}/quantization/quant.c"
        "${FERRITE_ROOT}/tools/profiler.c"
        "${FERRITE_ROOT}/runtime/kernels.c"
        "${FERRITE_ROOT}/runtime/exec_plan.c"
        "${FERRITE_ROOT}/runtime/engine.c"
        "${FERRITE_ROOT}/runtime/model_ser.c"
    INCLUDE_DIRS
        config          # FIRST: device-sized core/config.h override
        "${FERRITE_ROOT}/core"
        "${FERRITE_ROOT}/graph"
        "${FERRITE_ROOT}/optim"
        "${FERRITE_ROOT}/ops"
        "${FERRITE_ROOT}/simd"
        "${FERRITE_ROOT}/planner"
        "${FERRITE_ROOT}/quantization"
        "${FERRITE_ROOT}/tools"
        "${FERRITE_ROOT}/runtime"
    PRIV_REQUIRES
        esp_timer       # fe_platform_now_ns() monotonic clock
)

# Host-only subsystems are compiled out; FLOAT64 is rejected at FEMD load.
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    FERRITE_NO_IMPORTER
    FERRITE_NO_OPTIM
    FERRITE_NO_COMPILER
    FERRITE_NO_PARALLEL
    FERRITE_NO_AVX2
    FERRITE_NO_PROFILER
    FERRITE_NO_FLOAT64
)

# The Xtensa port requires no pthread/libm beyond what ESP-IDF provides;
# esp_timer linkage is handled by the PRIV_REQUIRES dependency graph above.