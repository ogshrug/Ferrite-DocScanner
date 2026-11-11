// runtime/exec_plan.h — static execution plan (Stage 13).
#ifndef FERRITE_EXEC_PLAN_H
#define FERRITE_EXEC_PLAN_H

#include "types.h"
#include "graph.h"
#include "allocator.h"
#include "memory_planner.h"

/*
 * A static execution plan turns "walk topo_order and dispatch" into
 * "execute a precomputed array of kernel function pointers".
 *
 * At plan-build time each node is resolved to a concrete kernel wrapper
 * (table lookup, no per-run switch) and the memory plan is computed once.
 * At run time the engine resets the activation arena, re-applies the cached
 * byte offsets, binds the caller's input, and walks the flat step array.
 * The only per-run analysis is an input-shape comparison: when the shape
 * changes, a rare rebuild re-runs shape inference + memory planning.
 *
 * The naive topo-walk engine stays the reference path (tests compare them).
 */

typedef struct FeRuntime FeRuntime;
typedef struct FeNode     FeNode;

typedef FeStatus (*FeExecFn)(FeRuntime *rt, FeNode *node);

/* One flat execution step: a resolved kernel and the node it executes. */
typedef struct {
    FeExecFn fn;
    int      node_index;
} FeExecStep;

typedef struct {
    FeExecStep steps[FE_MAX_NODES];
    int        n_steps;
    int        steps_valid;   /* kernel resolution done for current graph */

    FePlan     plan;          /* cached memory layout (byte offsets)    */
    int        plan_valid;    /* plan matches the last observed shapes  */

    /* Last input shape observed — a mismatch invalidates the plan. */
    int        bound_input_ndim;
    int        bound_input_shape[FERRITE_MAX_DIMS];

    /* The graph input was declared dynamic ({0,...}); every rebuild then
     * re-seeds it from the caller's current tensor shape. */
    int        input_dynamic;

    /* Number of times the memory plan was (re)computed. Steady-state runs
     * with unchanged shapes leave this untouched; tests assert == 1 after
     * N same-shaped runs. */
    uint64_t   n_plan_builds;
} FeExecPlan;

/* Resolve every node in topo order to its kernel and cache the plan.
 * Runs shape-agnostic work only; dynamic shapes are handled by
 * fe_exec_run's rebuild path. */
FeStatus fe_exec_build(FeExecPlan *ep, FeRuntime *rt);

/* Execute one inference. If the input shape differs from the last planned
 * shapes, the memory plan and (if needed) steps are rebuilt. */
FeStatus fe_exec_run(FeExecPlan *ep, FeRuntime *rt,
                     const FeTensor *input, FeTensor *output);

/* Plans memory for an input of this shape without running, and reports the
 * activation buffer size fe_exec_run will need in *bytes. */
FeStatus fe_exec_required_bytes(FeExecPlan *ep, FeRuntime *rt,
                                const FeTensor *input, size_t *bytes);

#endif /* FERRITE_EXEC_PLAN_H */