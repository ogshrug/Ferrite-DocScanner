/* optim/optim.c — pass orchestration (Stage 12). */
#include "optim.h"
#include "shape_infer.h"
#include <string.h>

FeStatus fe_optimize(FeGraph *g, FeArena *weight_arena) {
    if (!g || !weight_arena) return FE_ERR_NULL;

    FeStatus s = fe_infer_shapes(g);          /* resolve shapes first       */
    if (s != FE_OK) return s;

    s = fe_pass_simplify(g);                  /* identity transpose/flatten */
    if (s != FE_OK) return s;

    s = fe_graph_topo_sort(g);
    if (s != FE_OK) return s;

    s = fe_pass_fold_constants(g, weight_arena);   /* fold weight subgraphs */
    if (s != FE_OK) return s;

    s = fe_pass_fuse_conv_bn(g, weight_arena);     /* Conv+BN -> Conv        */
    if (s != FE_OK) return s;

    s = fe_pass_fuse_conv_relu(g);                 /* Conv+Relu -> Conv      */
    if (s != FE_OK) return s;

    s = fe_pass_cse(g);                       /* common subexpressions      */
    if (s != FE_OK) return s;

    s = fe_pass_dead_elim(g);                 /* compact garbage away       */
    if (s != FE_OK) return s;

    s = fe_graph_topo_sort(g);
    if (s != FE_OK) return s;
    return fe_graph_validate(g);
}