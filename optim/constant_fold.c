/* optim/constant_fold.c — fold subgraphs whose inputs are all constants. */
#include "optim.h"
#include "ops.h"
#include "shape_infer.h"
#include <string.h>

/*
 * Constant folding (Stage 12.2/12.5).
 *
 * A node whose every input carries known data (a weight tensor, or a tensor
 * already folded into a weight) is itself constant: compute it now, at load,
 * into the weight arena, then rewrite the node into an INPUT feed that simply
 * hands the constant through. Folding runs in topo order so folded constants
 * feed later folds; each fold resolves more tensors, so we sweep until no
 * node changes.
 *
 * Rewriting to INPUT (rather than deleting) keeps consumer indices valid;
 * fe_pass_dead_elim compacts the leftovers away afterwards.
 */

static int foldable_op(FeOpType op) {
    switch (op) {
        case FE_OP_TRANSPOSE: case FE_OP_FLATTEN:
        case FE_OP_RELU: case FE_OP_SIGMOID: case FE_OP_TANH:
        case FE_OP_EXP: case FE_OP_LOG: case FE_OP_NEG:
        case FE_OP_GELU: case FE_OP_SOFTMAX:
        case FE_OP_LEAKY_RELU: case FE_OP_ELU: case FE_OP_SWISH:
        case FE_OP_ADD: case FE_OP_SUB: case FE_OP_MUL:
        case FE_OP_DIV: case FE_OP_POW:
        case FE_OP_MATMUL: case FE_OP_GEMM:
        case FE_OP_CONV1D: case FE_OP_CONV2D:
        case FE_OP_MAXPOOL: case FE_OP_AVGPOOL:
        case FE_OP_BATCHNORM: case FE_OP_LAYERNORM: case FE_OP_GROUPNORM:
        case FE_OP_EMBEDDING:
            return 1;
        default:
            return 0;
    }
}

/* A tensor is fold-readable iff it carries data: a weight or a folded-in
 * INPUT feed. INPUT nodes produce the tensor; their outputs never hold data
 * unless a pass gave them a weight tensor. So: data means is_weight && tensor. */
static int is_constant(const FeGraph *g, int tidx) {
    const FeTensorEntry *e = &g->tensors[tidx];
    return e->is_weight && e->tensor && e->tensor->data;
}

static FeStatus alloc_out(FeArena *arena, const FeTensorEntry *e,
                          FeTensor **out) {
    *out = fe_arena_alloc_tensor(arena, e->dtype, e->ndim, e->shape);
    return *out ? FE_OK : FE_ERR_NOMEM;
}

static FeStatus fold_node(FeGraph *g, FeNode *node, FeArena *arena) {
    FeTensor *outs[FE_MAX_NODE_OUTPUTS] = {0};
    for (int o = 0; o < node->n_outputs; o++) {
        FeStatus s = alloc_out(arena, &g->tensors[node->outputs[o]], &outs[o]);
        if (s != FE_OK) return s;
    }

    #define IN(i)  (g->tensors[node->inputs [i]].tensor)
    #define OUT(i) (outs[i])

    FeStatus s = FE_OK;
    switch (node->op) {
        case FE_OP_TRANSPOSE:    s = fe_transpose(IN(0), OUT(0)); break;
        case FE_OP_RELU:         s = fe_relu(IN(0), OUT(0)); break;
        case FE_OP_SIGMOID:      s = fe_sigmoid(IN(0), OUT(0)); break;
        case FE_OP_TANH:         s = fe_tanh(IN(0), OUT(0)); break;
        case FE_OP_EXP:          s = fe_exp(IN(0), OUT(0)); break;
        case FE_OP_LOG:          s = fe_ln(IN(0), OUT(0)); break;
        case FE_OP_NEG:          s = fe_neg(IN(0), OUT(0)); break;
        case FE_OP_GELU:         s = fe_gelu(IN(0), OUT(0)); break;
        case FE_OP_SOFTMAX:      s = fe_softmax(IN(0), OUT(0)); break;
        case FE_OP_LEAKY_RELU:
            s = fe_leaky_relu(IN(0), OUT(0),
                              node->attrs.leaky_relu.negative_slope); break;
        case FE_OP_ELU:
            s = fe_elu(IN(0), OUT(0), node->attrs.elu.alpha); break;
        case FE_OP_SWISH:        s = fe_swish(IN(0), OUT(0)); break;
        case FE_OP_ADD:          s = fe_add(IN(0), IN(1), OUT(0)); break;
        case FE_OP_SUB:          s = fe_sub(IN(0), IN(1), OUT(0)); break;
        case FE_OP_MUL:          s = fe_mul(IN(0), IN(1), OUT(0)); break;
        case FE_OP_DIV:          s = fe_div(IN(0), IN(1), OUT(0)); break;
        case FE_OP_POW:          s = fe_pow(IN(0), IN(1), OUT(0)); break;
        case FE_OP_MATMUL:       s = fe_matmul(IN(0), IN(1), OUT(0)); break;
        case FE_OP_GEMM:
            s = fe_gemm(IN(0), node->attrs.gemm.transA,
                        IN(1), node->attrs.gemm.transB,
                        OUT(0), node->attrs.gemm.alpha,
                        node->attrs.gemm.beta); break;
        case FE_OP_CONV1D:
            s = fe_conv1d(IN(0), IN(1),
                          node->n_inputs > 2 ? IN(2) : NULL, OUT(0),
                          node->attrs.conv1d.stride, node->attrs.conv1d.pad);
            break;
        case FE_OP_CONV2D:
            s = fe_conv2d_ex(IN(0), IN(1),
                             node->n_inputs > 2 ? IN(2) : NULL, OUT(0),
                             node->attrs.conv2d.stride_h, node->attrs.conv2d.stride_w,
                             node->attrs.conv2d.pad_h,    node->attrs.conv2d.pad_w,
                             node->attrs.conv2d.dil_h > 0 ? node->attrs.conv2d.dil_h : 1,
                             node->attrs.conv2d.dil_w > 0 ? node->attrs.conv2d.dil_w : 1,
                             node->attrs.conv2d.groups > 0 ? node->attrs.conv2d.groups : 1,
                             node->attrs.conv2d.relu);
            break;
        case FE_OP_MAXPOOL:
            s = fe_maxpool_pad(IN(0), OUT(0), node->attrs.pool.kh, node->attrs.pool.kw,
                               node->attrs.pool.sh, node->attrs.pool.sw,
                               node->attrs.pool.ph, node->attrs.pool.pw); break;
        case FE_OP_AVGPOOL:
            s = fe_avgpool_pad(IN(0), OUT(0), node->attrs.pool.kh, node->attrs.pool.kw,
                               node->attrs.pool.sh, node->attrs.pool.sw,
                               node->attrs.pool.ph, node->attrs.pool.pw); break;
        case FE_OP_BATCHNORM:
            s = fe_batchnorm(IN(0), IN(1), IN(2), IN(3), IN(4), OUT(0),
                             node->attrs.batchnorm.eps); break;
        case FE_OP_LAYERNORM:
            s = fe_layernorm(IN(0), IN(1), IN(2), OUT(0),
                             node->attrs.layernorm.eps); break;
        case FE_OP_GROUPNORM:
            s = fe_groupnorm(IN(0), IN(1), IN(2), OUT(0),
                             node->attrs.groupnorm.groups,
                             node->attrs.groupnorm.eps); break;
        case FE_OP_FLATTEN: {
            FeTensor *in = IN(0);
            if (in->ndim <= 0 || in->shape[0] <= 0) { s = FE_ERR_SHAPE; break; }
            int total = fe_tensor_numel(in);
            if (total % in->shape[0] != 0) { s = FE_ERR_SHAPE; break; }
            int shape[2] = { in->shape[0], total / in->shape[0] };
            FeTensor *view = fe_tensor_reshape(in, 2, shape);
            if (!view) { s = FE_ERR_SHAPE; break; }
            s = fe_tensor_copy(OUT(0), view);
            break;
        }
        case FE_OP_EMBEDDING:    s = fe_embedding(IN(0), IN(1), OUT(0)); break;
        default:
            return FE_OK;   /* not foldable; leave alone */
    }

    #undef IN
    #undef OUT

    if (s != FE_OK) return s;

    /* Publish the folded result as a constant and neuter the node. */
    for (int o = 0; o < node->n_outputs; o++) {
        FeTensorEntry *e = &g->tensors[node->outputs[o]];
        e->is_weight = 1;
        e->tensor    = outs[o];
    }
    node->op = FE_OP_INPUT;
    node->n_inputs = 0;
    return FE_OK;
}

FeStatus fe_pass_fold_constants(FeGraph *g, FeArena *arena) {
    if (!g || !arena) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    int changed = 1;
    for (int pass = 0; changed && pass <= g->n_nodes; pass++) {
        changed = 0;
        for (int i = 0; i < g->n_nodes; i++) {
            FeNode *node = &g->nodes[g->topo_order[i]];
            if (node->op == FE_OP_INPUT || node->op == FE_OP_OUTPUT) continue;
            if (node->n_outputs == 0 || !foldable_op(node->op)) continue;

            int all_const = 1;
            for (int k = 0; k < node->n_inputs; k++)
                if (!is_constant(g, node->inputs[k])) { all_const = 0; break; }
            if (!all_const) continue;
            /* All inputs constant, so a whole-tensor-output op is foldable. */
            FeStatus s = fold_node(g, node, arena);
            if (s != FE_OK) return s;
            changed = 1;
        }
    }
    return FE_OK;
}