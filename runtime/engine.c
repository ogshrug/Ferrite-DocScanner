#include "engine.h"
#include "exec_plan.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * The engine owns the runtime lifecycle (arena binding, weight allocation,
 * validation). Inference itself is delegated to the static execution plan
 * (Stage 13, runtime/exec_plan.c): nodes are resolved to kernel function
 * pointers at plan-build time and run as a flat array, with the memory
 * plan cached and re-applied per run.
 */

FeStatus fe_runtime_init(FeRuntime *rt, FeGraph *graph,
                          void *weight_buf,     size_t weight_size,
                          void *activation_buf, size_t activation_size) {
    if (!rt || !graph || !weight_buf || !activation_buf) return FE_ERR_NULL;

    rt->graph    = graph;
#ifndef FERRITE_NO_PROFILER
    rt->profiler = NULL;
#endif

    FeStatus s;
    s = fe_arena_init(&rt->weight_arena,     weight_buf,     weight_size);
    if (s != FE_OK) return s;
    s = fe_arena_init(&rt->activation_arena, activation_buf, activation_size);
    if (s != FE_OK) return s;

    memset(&rt->exec, 0, sizeof(rt->exec));

    if (!graph->topo_valid) {
        s = fe_graph_topo_sort(graph);
        if (s != FE_OK) return s;
    }
    s = fe_graph_validate(graph);
    if (s != FE_OK) return s;
    return FE_OK;
}

FeStatus fe_runtime_alloc_weights(FeRuntime *rt) {
    FeGraph *g = rt->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight || e->tensor) continue;
        e->tensor = fe_arena_alloc_tensor(&rt->weight_arena,
                                           e->dtype, e->ndim, e->shape);
        if (!e->tensor) return FE_ERR_NOMEM;
    }
    return FE_OK;
}

FeStatus fe_runtime_run(FeRuntime *rt,
                         const FeTensor *input,
                         FeTensor *output) {
    if (!rt) return FE_ERR_NULL;
    return fe_exec_run(&rt->exec, rt, input, output);
}

FeStatus fe_runtime_required_bytes(FeRuntime *rt, const FeTensor *input,
                                   size_t *bytes) {
    if (!rt) return FE_ERR_NULL;
    return fe_exec_required_bytes(&rt->exec, rt, input, bytes);
}

FeStatus fe_runtime_run_batch(FeRuntime *rt,
                              FeTensor *const *inputs,
                              FeTensor *const *outputs,
                              int n) {
    if (!rt || !inputs || !outputs || n < 1) return FE_ERR_NULL;
    for (int i = 0; i < n; i++) {
        if (!inputs[i] || !outputs[i]) return FE_ERR_NULL;
        FeStatus s = fe_runtime_run(rt, inputs[i], outputs[i]);
        if (s != FE_OK) return s;
    }
    return FE_OK;
}

FeStatus fe_runtime_calibrate(FeRuntime *rt,
                              FeTensor *const *inputs, int n,
                              FeTensor *output, float *ranges) {
    if (!rt || !inputs || n < 1 || !output || !ranges) return FE_ERR_NULL;

    FeGraph *g = rt->graph;
    int nt = g->n_tensors;
    for (int t = 0; t < nt; t++) ranges[t] = 0.0f;

    for (int s = 0; s < n; s++) {
        if (!inputs[s]) return FE_ERR_NULL;
        FeStatus st = fe_runtime_run(rt, inputs[s], output);
        if (st != FE_OK) return st;
        for (int t = 0; t < nt; t++) {
            FeTensorEntry *e = &g->tensors[t];
            if (e->is_weight || !e->tensor || !e->tensor->data) continue;
            if (e->tensor->dtype != DTYPE_FLOAT32) continue;
            int m = fe_tensor_numel(e->tensor);
            const float *d = (const float *)e->tensor->data;
            for (int i = 0; i < m; i++) {
                float a = fabsf(d[i]);
                if (a > ranges[t]) ranges[t] = a;
            }
        }
    }
    return FE_OK;
}

/* qsort comparator for ascending floats. */
static int cmp_float_asc(const void *pa, const void *pb) {
    float a = *(const float *)pa, b = *(const float *)pb;
    return (a > b) - (a < b);
}

FeStatus fe_runtime_calibrate_pct(FeRuntime *rt,
                                  FeTensor *const *inputs, int n,
                                  FeTensor *output, float pct,
                                  float *ranges) {
    if (!rt || !inputs || n < 1 || !output || !ranges) return FE_ERR_NULL;
    if (pct < 0.0f || pct > 1.0f) return FE_ERR_SHAPE;

    FeGraph *g = rt->graph;
    int nt = g->n_tensors;
    for (int t = 0; t < nt; t++) ranges[t] = 0.0f;

    /* Per-sample maxima: column-major samples[t * n + s]. Collected so the
     * pct-th percentile of sample maxima can be taken per tensor instead of
     * the global max. Calibration is offline, so malloc is allowed. */
    float *samples = (float *)malloc((size_t)nt * (size_t)n * sizeof(float));
    if (!samples) return FE_ERR_NOMEM;

    FeStatus res = FE_OK;
    for (int s = 0; s < n && res == FE_OK; s++) {
        if (!inputs[s]) { res = FE_ERR_NULL; break; }
        res = fe_runtime_run(rt, inputs[s], output);
        if (res != FE_OK) break;
        for (int t = 0; t < nt; t++) {
            FeTensorEntry *e = &g->tensors[t];
            float peak = 0.0f;
            if (!e->is_weight && e->tensor && e->tensor->data &&
                e->tensor->dtype == DTYPE_FLOAT32) {
                int m = fe_tensor_numel(e->tensor);
                const float *d = (const float *)e->tensor->data;
                for (int i = 0; i < m; i++) {
                    float a = fabsf(d[i]);
                    if (a > peak) peak = a;
                }
            }
            samples[t * n + s] = peak;
        }
    }

    if (res == FE_OK) {
        for (int t = 0; t < nt; t++) {
            float *col = &samples[t * n];
            qsort(col, (size_t)n, sizeof(float), cmp_float_asc);
            size_t idx = (size_t)(pct * (float)(n - 1) + 0.5f);
            if (idx >= (size_t)n) idx = (size_t)n - 1;
            ranges[t] = col[idx];
        }
    }

    free(samples);
    return res;
}

void fe_runtime_print_trace(const FeRuntime *rt) {
    FeGraph *g = rt->graph;
    printf("Execution trace (%d nodes):\n", g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        int idx = g->topo_valid ? g->topo_order[i] : i;
        printf("  step %d: %s\n", i, g->nodes[idx].name);
    }
}