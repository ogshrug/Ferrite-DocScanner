/* optim/shape_infer.c — per-op shape-inference pass (Stage 12.1). */
#include "shape_infer.h"
#include "types.h"
#include <string.h>

static int is_known(const FeTensorEntry *e) {
    if (e->ndim <= 0) return 0;
    for (int i = 0; i < e->ndim; i++)
        if (e->shape[i] <= 0) return 0;
    return 1;
}

/* Last-dims broadcast (NumPy rules). Returns 0 if incompatible/unknown. */
static int broadcast_shape(const FeTensorEntry *a, const FeTensorEntry *b,
                           FeTensorEntry *out) {
    if (!is_known(a) || !is_known(b)) return 0;
    int nd = a->ndim > b->ndim ? a->ndim : b->ndim;
    int rs[FERRITE_MAX_DIMS];
    for (int i = 0; i < nd; i++) {
        int da = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
        int db = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
        if (da == db)            rs[nd - 1 - i] = da;
        else if (da == 1)        rs[nd - 1 - i] = db;
        else if (db == 1)        rs[nd - 1 - i] = da;
        else return 0;           /* e.g. [4] vs [3] */
    }
    out->ndim = nd;
    memcpy(out->shape, rs, nd * sizeof(int));
    out->dtype = DTYPE_FLOAT32;
    return 1;
}

static void copy_shape(const FeTensorEntry *in, FeTensorEntry *out) {
    out->ndim = in->ndim;
    memcpy(out->shape, in->shape, in->ndim * sizeof(int));
    out->dtype = in->dtype;
}

static void set_shape(FeTensorEntry *out, int ndim, const int *shape) {
    out->ndim = ndim;
    memcpy(out->shape, shape, ndim * sizeof(int));
    out->dtype = DTYPE_FLOAT32;
}

/* Same-shape producers: activations, norms, attention, positional encoding. */
static int rule_same_shape(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in)) return 0;
    copy_shape(in, out);
    return 1;
}

static int rule_broadcast(FeGraph *g, FeNode *node) {
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out)) return 0;
    FeTensorEntry *a = &g->tensors[node->inputs[0]];
    FeTensorEntry *b = &g->tensors[node->inputs[1]];
    if (!is_known(a) || !is_known(b)) return 0;
    return broadcast_shape(a, b, out);
}

static int rule_matmul(FeGraph *g, FeNode *node) {
    FeTensorEntry *A   = &g->tensors[node->inputs[0]];
    FeTensorEntry *B   = &g->tensors[node->inputs[1]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(A) || !is_known(B)) return 0;
    if (A->ndim < 2 || B->ndim < 2 || A->shape[1] != B->shape[0])
        return -1;                       /* K mismatch is a real error */
    int rs[2] = {A->shape[0], B->shape[1]};
    set_shape(out, 2, rs);
    return 1;
}

static int rule_linear(FeGraph *g, FeNode *node) {
    FeTensorEntry *A   = &g->tensors[node->inputs[0]];
    FeTensorEntry *W   = &g->tensors[node->inputs[1]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(A) || !is_known(W)) return 0;
    if (A->shape[1] != W->shape[0]) return -1;
    int rs[2] = {A->shape[0], W->shape[1]};
    set_shape(out, 2, rs);
    return 1;
}

static int rule_gemm(FeGraph *g, FeNode *node) {
    FeTensorEntry *A   = &g->tensors[node->inputs[0]];
    FeTensorEntry *B   = &g->tensors[node->inputs[1]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(A) || !is_known(B)) return 0;
    if (A->ndim != 2 || B->ndim != 2) return -1;

    int transA = node->attrs.gemm.transA;
    int transB = node->attrs.gemm.transB;
    int M = transA ? A->shape[1] : A->shape[0];
    int K_A = transA ? A->shape[0] : A->shape[1];
    int K_B = transB ? B->shape[1] : B->shape[0];
    int N = transB ? B->shape[0] : B->shape[1];
    if (K_A != K_B) return -1;

    int rs[2] = {M, N};
    set_shape(out, 2, rs);
    return 1;
}

static int rule_transpose(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in) || in->ndim != 2) return 0;
    int rs[2] = {in->shape[1], in->shape[0]};
    set_shape(out, 2, rs);
    return 1;
}

static int rule_conv1d(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *w   = &g->tensors[node->inputs[1]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in) || !is_known(w)) return 0;
    int L = in->shape[2];
    int C_out = w->shape[0];
    int K = w->shape[2];
    int pad    = node->attrs.conv1d.pad;
    int stride = node->attrs.conv1d.stride > 0 ? node->attrs.conv1d.stride : 1;
    int rs[3] = { in->shape[0], C_out, (L - K + 2 * pad) / stride + 1 };
    set_shape(out, 3, rs);
    return 1;
}

static int rule_conv2d(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *w   = &g->tensors[node->inputs[1]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in) || !is_known(w)) return 0;
    int H = in->shape[2], W = in->shape[3];
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    int sh = node->attrs.conv2d.stride_h, sw = node->attrs.conv2d.stride_w;
    int ph = node->attrs.conv2d.pad_h,    pw = node->attrs.conv2d.pad_w;
    int dh = node->attrs.conv2d.dil_h > 0 ? node->attrs.conv2d.dil_h : 1;
    int dw = node->attrs.conv2d.dil_w > 0 ? node->attrs.conv2d.dil_w : 1;
    if (sh < 1 || sw < 1) return -1;
    int rs[4] = { in->shape[0], C_out,
                  (H + 2 * ph - (dh * (KH - 1) + 1)) / sh + 1,
                  (W + 2 * pw - (dw * (KW - 1) + 1)) / sw + 1 };
    if (rs[2] < 1 || rs[3] < 1) return -1;
    set_shape(out, 4, rs);
    return 1;
}

static int rule_concat(FeGraph *g, FeNode *node) {
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || node->n_inputs < 1) return 0;
    FeTensorEntry *first = &g->tensors[node->inputs[0]];
    if (!is_known(first)) return 0;
    int nd = first->ndim;
    int axis = node->attrs.concat.axis < 0 ? node->attrs.concat.axis + nd
                                           : node->attrs.concat.axis;
    if (axis < 0 || axis >= nd) return -1;
    int rs[FERRITE_MAX_DIMS];
    memcpy(rs, first->shape, nd * sizeof(int));
    for (int i = 1; i < node->n_inputs; i++) {
        FeTensorEntry *e = &g->tensors[node->inputs[i]];
        if (!is_known(e)) return 0;
        if (e->ndim != nd) return -1;
        for (int d = 0; d < nd; d++)
            if (d != axis && e->shape[d] != rs[d]) return -1;
        rs[axis] += e->shape[axis];
    }
    set_shape(out, nd, rs);
    return 1;
}

static int rule_resize(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in)) return 0;
    if (in->ndim != 4) return -1;
    int oh = node->attrs.resize.out_h, ow = node->attrs.resize.out_w;
    if (oh <= 0 || ow <= 0) {
        /* ONNX: output = floor(input * scale). */
        oh = (int)((float)in->shape[2] * node->attrs.resize.scale_h);
        ow = (int)((float)in->shape[3] * node->attrs.resize.scale_w);
    }
    if (oh < 1 || ow < 1) return -1;
    int rs[4] = { in->shape[0], in->shape[1], oh, ow };
    set_shape(out, 4, rs);
    return 1;
}

static int rule_global_avgpool(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in)) return 0;
    if (in->ndim != 4) return -1;
    int rs[4] = { in->shape[0], in->shape[1], 1, 1 };
    set_shape(out, 4, rs);
    return 1;
}

static int rule_pad(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in)) return 0;
    if (in->ndim != 4) return -1;
    int rs[4] = { in->shape[0], in->shape[1],
                  in->shape[2] + node->attrs.pad.top  + node->attrs.pad.bottom,
                  in->shape[3] + node->attrs.pad.left + node->attrs.pad.right };
    /* Reflection needs a sample to mirror: pads must be smaller than the image. */
    if (node->attrs.pad.mode == 1 &&
        (node->attrs.pad.top >= in->shape[2] || node->attrs.pad.bottom >= in->shape[2] ||
         node->attrs.pad.left >= in->shape[3] || node->attrs.pad.right >= in->shape[3]))
        return -1;
    set_shape(out, 4, rs);
    return 1;
}

static int rule_pool(FeGraph *g, FeNode *node) {
    /* In [N,C,H,W] or [C,H,W]; pool the last two dims without padding. */
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in) || (in->ndim != 3 && in->ndim != 4))
        return 0;
    int H = in->shape[in->ndim - 2], W = in->shape[in->ndim - 1];
    int oh = (H + 2 * node->attrs.pool.ph - node->attrs.pool.kh) / node->attrs.pool.sh + 1;
    int ow = (W + 2 * node->attrs.pool.pw - node->attrs.pool.kw) / node->attrs.pool.sw + 1;
    int rs[FERRITE_MAX_DIMS];
    if (in->ndim == 4) {
        rs[0] = in->shape[0]; rs[1] = in->shape[1]; rs[2] = oh; rs[3] = ow;
    } else {
        rs[0] = in->shape[0]; rs[1] = oh; rs[2] = ow;
    }
    set_shape(out, in->ndim, rs);
    return 1;
}

static int rule_flatten(FeGraph *g, FeNode *node) {
    FeTensorEntry *in  = &g->tensors[node->inputs[0]];
    FeTensorEntry *out = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(in) || in->shape[0] == 0) return 0;
    int total = 1;
    for (int d = 0; d < in->ndim; d++) total *= in->shape[d];
    int rs[2] = { in->shape[0], total / in->shape[0] };
    set_shape(out, 2, rs);
    return 1;
}

static int rule_embedding(FeGraph *g, FeNode *node) {
    /* indices [..., idx], table [vocab, d_model] -> indices + [d_model]. */
    FeTensorEntry *idx   = &g->tensors[node->inputs[0]];
    FeTensorEntry *table = &g->tensors[node->inputs[1]];
    FeTensorEntry *out   = &g->tensors[node->outputs[0]];
    if (is_known(out) || !is_known(idx) || !is_known(table)) return 0;
    int rs[FERRITE_MAX_DIMS];
    memcpy(rs, idx->shape, idx->ndim * sizeof(int));
    rs[idx->ndim] = table->shape[1];
    set_shape(out, idx->ndim + 1, rs);
    return 1;
}

FeStatus fe_infer_shapes(FeGraph *g) {
    if (!g) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    int changed = 1;
    for (int pass = 0; changed && pass <= g->n_nodes; pass++) {
        changed = 0;
        for (int i = 0; i < g->n_nodes; i++) {
            FeNode *node = &g->nodes[g->topo_order[i]];
            int r = 0;
            switch (node->op) {
                case FE_OP_INPUT:
                case FE_OP_OUTPUT:
                    break;
                /* Same-shape: activations, norms, attention, pos-enc */
                case FE_OP_RELU:
                case FE_OP_SOFTMAX:
                case FE_OP_SIGMOID:
                case FE_OP_TANH:
                case FE_OP_GELU:
                case FE_OP_LEAKY_RELU:
                case FE_OP_ELU:
                case FE_OP_SWISH:
                case FE_OP_EXP:
                case FE_OP_LOG:
                case FE_OP_NEG:
                case FE_OP_BATCHNORM:
                case FE_OP_LAYERNORM:
                case FE_OP_GROUPNORM:
                case FE_OP_ATTENTION:
                case FE_OP_POSITIONAL_ENCOD:
                    r = rule_same_shape(g, node);
                    break;
                /* Elementwise broadcast */
                case FE_OP_ADD:
                case FE_OP_SUB:
                case FE_OP_MUL:
                case FE_OP_DIV:
                case FE_OP_POW:
                    r = rule_broadcast(g, node);
                    break;
                case FE_OP_MATMUL:      r = rule_matmul(g, node);     break;
                case FE_OP_LINEAR:      r = rule_linear(g, node);     break;
                case FE_OP_GEMM:        r = rule_gemm(g, node);       break;
                case FE_OP_TRANSPOSE:   r = rule_transpose(g, node);  break;
                case FE_OP_CONV1D:      r = rule_conv1d(g, node);     break;
                case FE_OP_CONV2D:      r = rule_conv2d(g, node);     break;
                case FE_OP_MAXPOOL:
                case FE_OP_AVGPOOL:     r = rule_pool(g, node);       break;
                case FE_OP_FLATTEN:     r = rule_flatten(g, node);    break;
                case FE_OP_EMBEDDING:   r = rule_embedding(g, node);  break;
                case FE_OP_MULTIHEAD_ATTN:
                case FE_OP_CLIP:
                case FE_OP_PRELU:
                    r = rule_same_shape(g, node);
                    break;
                case FE_OP_CONCAT:         r = rule_concat(g, node);         break;
                case FE_OP_RESIZE:         r = rule_resize(g, node);         break;
                case FE_OP_GLOBAL_AVGPOOL: r = rule_global_avgpool(g, node); break;
                case FE_OP_PAD:            r = rule_pad(g, node);            break;
            }
            if (r < 0) return FE_ERR_SHAPE;   /* geometry contradiction */
            if (r > 0) changed = 1;
        }
    }
    return FE_OK;
}