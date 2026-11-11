/* runtime/exec_plan.c — static execution plan (Stage 13). */
#include "engine.h"
#include "exec_plan.h"
#include "kernels.h"
#include "memory_planner.h"
#include "shape_infer.h"
#include "log.h"
#include <string.h>
#include <stdio.h>

/*
 * Plan management only: kernels (ex_* + the FeOpType->FeExecFn table) live in
 * kernels.c so the device runtime can resolve dispatch without pulling in
 * shape inference. Here we find the graph input, seed/infer shapes, cache the
 * memory plan, and walk the pre-resolved step array.
 */

/* Locate the graph's input tensor (shared helper in graph.c). */
static int find_graph_input(const FeGraph *g) {
    return fe_graph_find_input(g);
}

/* Clear every produced (non-weight) tensor shape so shape inference
 * recomputes them from scratch; the graph input's own shape is saved first
 * and restored unchanged so a static (ONNX value_info) input keeps it. */
static void reset_produced_shapes(FeGraph *g, int in_tidx,
                                  int *save_ndim,
                                  int save_shape[FERRITE_MAX_DIMS]) {
    FeTensorEntry *ie = &g->tensors[in_tidx];
    *save_ndim = ie->ndim;
    for (int i = 0; i < ie->ndim; i++) save_shape[i] = ie->shape[i];

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight) continue;
        e->ndim = 0;
        memset(e->shape, 0, sizeof(e->shape));
    }

    ie->ndim = *save_ndim;
    for (int i = 0; i < ie->ndim; i++) ie->shape[i] = save_shape[i];
}

/*
 * Infer output shapes for dynamic tensors. A dynamic graph input ({0,...})
 * is seeded from the caller's tensor shape; once identified as dynamic it is
 * re-seeded on every plan build (force_dynamic), because the first seed
 * overwrites the zero marker. Static inputs (ONNX value_info) are untouched.
 * Sets *is_dynamic = 1 when the graph input is dynamic. Shape-inference
 * failure propagates loudly instead of leaving produced tensors at ndim == 0
 * (a 0-dim tensor would make the planner write strides[-1]).
 */
static FeStatus infer_shapes(FeRuntime *rt, const FeTensor *input,
                             int force_dynamic, int *is_dynamic) {
    FeGraph *g = rt->graph;
    FeStatus s;
    *is_dynamic = 0;

    int in_tidx = find_graph_input(g);
    if (in_tidx >= 0) {
        FeTensorEntry *e = &g->tensors[in_tidx];
        int dynamic = !e->is_weight && e->ndim > 0 &&
                      (force_dynamic || e->shape[0] == 0);
        if (dynamic) {
            e->ndim = input->ndim;
            memcpy(e->shape, input->shape, input->ndim * sizeof(int));
        }
        s = fe_infer_shapes(g);
        if (s != FE_OK) return s;
        *is_dynamic = dynamic;
        return FE_OK;
    }
    s = fe_infer_shapes(g);
    if (s != FE_OK) return s;
    return FE_OK;
}

/* Copy the final output into the caller's buffer. Order: OUTPUT-node input,
 * else the last planned step's output (the engine's documented behavior). */
static FeStatus copy_output(FeRuntime *rt, FeTensor *output) {
    FeGraph *g = rt->graph;
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_OUTPUT && g->nodes[i].n_inputs > 0) {
            int in_idx = g->nodes[i].inputs[0];
            if (!g->tensors[in_idx].tensor) return FE_ERR_SHAPE;
            return fe_tensor_copy(output, g->tensors[in_idx].tensor);
        }
    }
    FeNode *last = &g->nodes[g->topo_order[g->n_nodes - 1]];
    if (last->n_outputs > 0)
        return fe_tensor_copy(output, g->tensors[last->outputs[0]].tensor);
    return FE_ERR_SHAPE;
}

FeStatus fe_exec_build(FeExecPlan *ep, FeRuntime *rt) {
    if (!ep || !rt) return FE_ERR_NULL;
    FeGraph *g = rt->graph;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }
    ep->n_steps = g->n_nodes;
    for (int i = 0; i < g->n_nodes; i++) {
        int idx = g->topo_order[i];
        FeExecFn fn = fe_kernels_get(g->nodes[idx].op);
        if (!fn) {
            fprintf(stderr, "Engine: unimplemented op %d\n", g->nodes[idx].op);
            return FE_ERR_SHAPE;
        }
        ep->steps[i].fn = fn;
        ep->steps[i].node_index = idx;
    }
    ep->steps_valid = 1;
    return FE_OK;
}

FeStatus fe_exec_run(FeExecPlan *ep, FeRuntime *rt,
                     const FeTensor *input, FeTensor *output) {
    if (!ep || !rt || !input || !output) return FE_ERR_NULL;
    FeGraph *g = rt->graph;
    FeStatus s;

    if (!ep->steps_valid) {
        s = fe_exec_build(ep, rt);
        if (s != FE_OK) return s;
        ep->plan_valid = 0;
    }

    /* Shape change detection: the one piece of per-run analysis. */
    int shape_changed = !ep->plan_valid ||
        ep->bound_input_ndim != input->ndim ||
        memcmp(ep->bound_input_shape, input->shape,
               input->ndim * sizeof(int)) != 0;

    if (shape_changed) {
        /* Shape inference only fills unknown shapes — it never shrinks an
         * already-known one. A dynamic-batch rebuild must forget every
         * produced shape and recompute from the input, so the graph input's
         * own (authoritative) shape is saved and restored first. */
        int in_tidx = find_graph_input(g);
        if (in_tidx < 0) return FE_ERR_SHAPE;
        int save_ndim;
        int save_shape[FERRITE_MAX_DIMS];
        reset_produced_shapes(g, in_tidx, &save_ndim, save_shape);

        int dynamic = 0;
        s = infer_shapes(rt, input, ep->input_dynamic, &dynamic);
        if (s != FE_OK) return s;
        ep->input_dynamic |= dynamic;
        s = fe_plan_memory(g, &ep->plan);
        if (s != FE_OK) return s;
        ep->n_plan_builds++;
        ep->bound_input_ndim = input->ndim;
        memcpy(ep->bound_input_shape, input->shape,
               input->ndim * sizeof(int));
        ep->plan_valid = 1;
    }

    fe_arena_reset(&rt->activation_arena);
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) e->tensor = NULL;
    }

    s = fe_plan_apply(g, &ep->plan,
                      rt->activation_arena.base,
                      rt->activation_arena.size,
                      &rt->activation_arena);
    if (s != FE_OK) return s;

    int in_tidx = find_graph_input(g);
    if (in_tidx < 0) return FE_ERR_SHAPE;
    g->tensors[in_tidx].tensor = (FeTensor *)input;

    /* Allocate the output tensor when it has no producer (planner skips
     * producer-less tensors); hand-built graphs use an OUTPUT node. */
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op != FE_OP_OUTPUT || g->nodes[i].n_inputs == 0)
            continue;
        int in_idx = g->nodes[i].inputs[0];
        if (g->tensors[in_idx].tensor) continue;
        g->tensors[in_idx].tensor = fe_arena_alloc_tensor(
            &rt->activation_arena, g->tensors[in_idx].dtype,
            g->tensors[in_idx].ndim, g->tensors[in_idx].shape);
        if (!g->tensors[in_idx].tensor) return FE_ERR_NOMEM;
        break;
    }

    /* The flat execution loop: no dispatch switch, no lifetime analysis. */
    for (int i = 0; i < ep->n_steps; i++) {
        s = ep->steps[i].fn(rt, &g->nodes[ep->steps[i].node_index]);
        if (s != FE_OK) {
            fprintf(stderr, "Engine: node '%s' failed with %d\n",
                    g->nodes[ep->steps[i].node_index].name, s);
            return s;
        }
    }

    return copy_output(rt, output);
}

FeStatus fe_exec_required_bytes(FeExecPlan *ep, FeRuntime *rt,
                                const FeTensor *input, size_t *bytes) {
    if (!ep || !rt || !input || !bytes) return FE_ERR_NULL;
    FeGraph *g = rt->graph;
    int in_tidx = find_graph_input(g);
    if (in_tidx < 0) return FE_ERR_SHAPE;
    int save_ndim;
    int save_shape[FERRITE_MAX_DIMS];
    reset_produced_shapes(g, in_tidx, &save_ndim, save_shape);
    int dynamic = 0;
    FeStatus s = infer_shapes(rt, input, ep->input_dynamic, &dynamic);
    if (s != FE_OK) return s;
    ep->input_dynamic |= dynamic;
    s = fe_plan_memory(g, &ep->plan);
    if (s != FE_OK) return s;
    *bytes = ep->plan.total_activation_bytes;
    ep->plan_valid = 0;          /* fe_exec_run re-plans against its own buffer */
    return FE_OK;
}
