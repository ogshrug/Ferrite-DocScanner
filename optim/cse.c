/* optim/cse.c — common-subexpression elimination (Stage 12.7). */
#include "optim.h"
#include <string.h>

/*
 * Two computations are identical when they use the same operator, the same
 * input tensors in the same order, and the same operator attributes. When
 * they do, keep the earlier one and relink every consumer of the later
 * node's outputs to the canonical node's outputs, neutering the later node.
 * Single-output nodes (the overwhelmingly common case) collapse outright;
 * multi-output nodes collapse per-output.
 */

static int attrs_equal(FeOpType op, const FeNode *a, const FeNode *b) {
    switch (op) {
        case FE_OP_SOFTMAX:    return a->attrs.softmax.axis == b->attrs.softmax.axis;
        case FE_OP_CONV1D:     return a->attrs.conv1d.stride == b->attrs.conv1d.stride &&
                                      a->attrs.conv1d.pad    == b->attrs.conv1d.pad;
        case FE_OP_LEAKY_RELU: return a->attrs.leaky_relu.negative_slope == b->attrs.leaky_relu.negative_slope;
        case FE_OP_ELU:        return a->attrs.elu.alpha == b->attrs.elu.alpha;
        case FE_OP_CONV2D:     return memcmp(&a->attrs.conv2d, &b->attrs.conv2d,
                                             sizeof(a->attrs.conv2d)) == 0;
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:    return memcmp(&a->attrs.pool, &b->attrs.pool,
                                             sizeof(a->attrs.pool)) == 0;
        case FE_OP_LAYERNORM:  return a->attrs.layernorm.eps == b->attrs.layernorm.eps;
        case FE_OP_GROUPNORM:  return a->attrs.groupnorm.groups == b->attrs.groupnorm.groups &&
                                      a->attrs.groupnorm.eps    == b->attrs.groupnorm.eps;
        case FE_OP_BATCHNORM:  return a->attrs.batchnorm.eps == b->attrs.batchnorm.eps;
        case FE_OP_GEMM:       return memcmp(&a->attrs.gemm, &b->attrs.gemm,
                                             sizeof(a->attrs.gemm)) == 0;
        case FE_OP_MULTIHEAD_ATTN:
            return a->attrs.multihead.num_heads == b->attrs.multihead.num_heads;
        case FE_OP_CONCAT:     return a->attrs.concat.axis == b->attrs.concat.axis;
        case FE_OP_RESIZE:     return memcmp(&a->attrs.resize, &b->attrs.resize,
                                             sizeof(a->attrs.resize)) == 0;
        case FE_OP_CLIP:       return a->attrs.clip.min == b->attrs.clip.min &&
                                      a->attrs.clip.max == b->attrs.clip.max;
        case FE_OP_PAD:        return memcmp(&a->attrs.pad, &b->attrs.pad,
                                             sizeof(a->attrs.pad)) == 0;
        default:               return 1;   /* op carries no attributes */
    }
}

static int nodes_equivalent(const FeNode *a, const FeNode *b) {
    if (a->op != b->op) return 0;
    if (a->n_inputs  != b->n_inputs  || a->n_outputs != b->n_outputs) return 0;
    for (int k = 0; k < a->n_inputs; k++)
        if (a->inputs[k] != b->inputs[k]) return 0;
    return attrs_equal(a->op, a, b);
}

FeStatus fe_pass_cse(FeGraph *g) {
    if (!g) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *a = &g->nodes[i];
        if (a->op == FE_OP_INPUT || a->op == FE_OP_OUTPUT) continue;

        for (int j = 0; j < i; j++) {
            FeNode *b = &g->nodes[j];
            if (b->op == FE_OP_INPUT || b->op == FE_OP_OUTPUT) continue;
            if (!nodes_equivalent(a, b)) continue;

            /* Duplicate: rewrite every consumer of a's outputs to b's. */
            for (int o = 0; o < a->n_outputs; o++)
                for (int c = 0; c < g->n_nodes; c++) {
                    FeNode *n = &g->nodes[c];
                    if (n == a) continue;
                    for (int k = 0; k < n->n_inputs; k++)
                        if (n->inputs[k] == a->outputs[o])
                            n->inputs[k] = b->outputs[o];
                }
            a->op = FE_OP_INPUT;          /* a's outputs now unused -> dead-elim */
            a->n_inputs = 0;
            break;
        }
    }
    return FE_OK;
}