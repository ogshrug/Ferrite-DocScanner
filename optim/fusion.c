/* optim/fusion.c — Conv + BatchNorm fusion (Stage 12.6). */
#include "optim.h"
#include <math.h>
#include <string.h>

/*
 * Fuse a Conv followed by a BatchNorm into an equivalent Conv, at load time.
 *
 *   conv:   y[c]   = W[c] * x + bias[c]
 *   bn:     z[c]   = gamma[c] * (y[c] - mean[c]) * inv_std[c] + beta[c]
 *                  = scale[c] * y[c] + shift[c]
 *   fused:  z[c]   = (scale[c] * W[c]) * x + (scale[c] * bias[c] + shift[c])
 *
 * with inv_std[c] = 1/sqrt(var[c]+eps), scale[c] = gamma[c] * inv_std[c],
 * shift[c] = beta[c] - mean[c] * scale[c]. The fused scales/shifts are
 * folded straight into the conv kernel tensors in the weight arena, the BN
 * node is neutered, and downstream consumers read BN'd data from the conv
 * output as before.
 *
 * Fires only when the conv's output feeds exactly the BN (no other reader),
 * and all BN coefficients are known weights with shape [C_out].
 */

static int count_consumers(const FeGraph *g, int tidx) {
    int c = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_inputs; k++)
            if (n->inputs[k] == tidx) c++;
    }
    return c;
}

static const float *weight_data(const FeGraph *g, int tidx) {
    const FeTensorEntry *e = &g->tensors[tidx];
    return (e->is_weight && e->tensor && e->tensor->data)
           ? (const float *)e->tensor->data : NULL;
}

FeStatus fe_pass_fuse_conv_bn(FeGraph *g, FeArena *arena) {
    if (!g || !arena) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *bn = &g->nodes[i];
        if (bn->op != FE_OP_BATCHNORM || bn->n_inputs < 5) continue;

        int conv_out = bn->inputs[0];
        if (count_consumers(g, conv_out) != 1) continue;

        /* Find the single producer of conv_out; must be a conv. */
        FeNode *conv = NULL;
        for (int p = 0; p < g->n_nodes; p++) {
            for (int o = 0; o < g->nodes[p].n_outputs; o++)
                if (g->nodes[p].outputs[o] == conv_out) { conv = &g->nodes[p]; break; }
            if (conv) break;
        }
        if (!conv || (conv->op != FE_OP_CONV1D && conv->op != FE_OP_CONV2D))
            continue;

        /* BN coefficients: gamma[1] beta[2] mean[3] var[4], each [C_out]. */
        int C_out = g->tensors[conv->inputs[1]].shape[0];
        const float *gamm = weight_data(g, bn->inputs[1]);
        const float *beta = weight_data(g, bn->inputs[2]);
        const float *mean = weight_data(g, bn->inputs[3]);
        const float *var  = weight_data(g, bn->inputs[4]);
        if (!gamm || !beta || !mean || !var) continue;
        if (g->tensors[bn->inputs[1]].shape[0] != C_out ||
            g->tensors[bn->inputs[2]].shape[0] != C_out ||
            g->tensors[bn->inputs[3]].shape[0] != C_out ||
            g->tensors[bn->inputs[4]].shape[0] != C_out) continue;

        float eps = bn->attrs.batchnorm.eps;

        /* Fused weight: W'[c] = scale[c] * W[c]. */
        const FeTensorEntry *we = &g->tensors[conv->inputs[1]];
        FeTensor *fused_w = fe_arena_alloc_tensor(arena, we->dtype, we->ndim,
                                                  we->shape);
        if (!fused_w) return FE_ERR_NOMEM;
        int w_per_channel = fe_tensor_numel(we->tensor) / C_out;
        const float *w = (const float *)we->tensor->data;
        float *fw = (float *)fused_w->data;
        for (int c = 0; c < C_out; c++) {
            float scale = gamm[c] * (1.0f / (float)sqrt(var[c] + eps));
            for (int e = 0; e < w_per_channel; e++)
                fw[c * w_per_channel + e] = w[c * w_per_channel + e] * scale;
        }

        /* Fused bias: bias'[c] = scale[c]*bias[c] + beta[c] - mean[c]*scale[c]. */
        int has_bias = conv->n_inputs > 2;
        int C_out_d = C_out;
        FeTensor *fused_b = fe_arena_alloc_tensor(arena, DTYPE_FLOAT32, 1,
                                                  &C_out_d);
        if (!fused_b) return FE_ERR_NOMEM;
        float *fb = (float *)fused_b->data;
        for (int c = 0; c < C_out; c++) {
            float scale  = gamm[c] * (1.0f / (float)sqrt(var[c] + eps));
            float shift  = beta[c] - mean[c] * scale;
            float b_conv = has_bias
                ? ((const float *)g->tensors[conv->inputs[2]].tensor->data)[c]
                : 0.0f;
            fb[c] = b_conv * scale + shift;
        }

        /* Rewrite the conv to read the fused weights and always carry bias. */
        int fw_idx = fe_graph_add_tensor(g, "fused_w", DTYPE_FLOAT32,
                                         we->ndim, we->shape, 1);
        int fb_idx = fe_graph_add_tensor(g, "fused_b", DTYPE_FLOAT32,
                                         1, &C_out_d, 1);
        if (fw_idx < 0 || fb_idx < 0) return FE_ERR_NOMEM;
        FeTensorEntry *fw_e = &g->tensors[fw_idx];
        FeTensorEntry *fb_e = &g->tensors[fb_idx];
        fw_e->is_weight = 1;
        fw_e->tensor = fused_w;
        fb_e->is_weight = 1;
        fb_e->tensor = fused_b;

        conv->inputs[1] = fw_idx;
        conv->inputs[2] = fb_idx;
        if (conv->n_inputs < 3) conv->n_inputs = 3;

        /* Neuter the BN and relink its consumers to the conv, whose output
         * now carries the batch-normed values. With no consumers left and
         * no inputs, dead-elim later reclaims the BN node and all four
         * coefficient tensors. */
        for (int c = 0; c < g->n_nodes; c++) {
            FeNode *n = &g->nodes[c];
            if (n == bn) continue;
            for (int k = 0; k < n->n_inputs; k++)
                if (n->inputs[k] == bn->outputs[0])
                    n->inputs[k] = conv->outputs[0];
        }
        bn->op = FE_OP_INPUT;
        bn->n_inputs = 0;
    }
    return FE_OK;
}

/*
 * Conv + Relu -> one Conv that clamps as it writes its output.
 *
 * Saves a full read and write of the activation per fused pair (image models put a
 * Relu after most convolutions). Fires only when the conv's output is read by the
 * Relu alone and the Relu's output is read by something (a Relu that is the graph's
 * output keeps its node, so the output tensor keeps its producer).
 */
FeStatus fe_pass_fuse_conv_relu(FeGraph *g) {
    if (!g) return FE_ERR_NULL;
    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *relu = &g->nodes[i];
        if (relu->op != FE_OP_RELU || relu->n_inputs != 1 || relu->n_outputs != 1) continue;
        int conv_out = relu->inputs[0];
        int relu_out = relu->outputs[0];
        if (count_consumers(g, conv_out) != 1 || count_consumers(g, relu_out) == 0) continue;

        FeNode *conv = NULL;
        for (int p = 0; p < g->n_nodes && !conv; p++)
            for (int o = 0; o < g->nodes[p].n_outputs; o++)
                if (g->nodes[p].outputs[o] == conv_out) { conv = &g->nodes[p]; break; }
        if (!conv || conv->op != FE_OP_CONV2D || conv->attrs.conv2d.relu) continue;

        conv->attrs.conv2d.relu = 1;
        conv->outputs[0] = relu_out;
        relu->op = FE_OP_INPUT;      /* neutered; dead-elim removes it */
        relu->n_inputs = 0;
        relu->n_outputs = 0;
    }
    return fe_graph_topo_sort(g);
}
