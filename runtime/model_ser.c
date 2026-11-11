/* runtime/model_ser.c — FEMD model artifact serialization (Section 2.2).
 *
 * The counterpart to core/tensor_ser.c's strict little-endian field I/O.
 * Format, all fields little-endian fixed-width:
 *
 *   magic[4] = "FEMD"
 *   u32 version
 *   u32 n_tensors, u32 n_nodes
 *   u32 in_tidx, u32 out_tidx
 *   u8  input_ndim; input shape[]
 *   per tensor (n_tensors), in registry order:
 *       u8 is_weight
 *       u8 dtype
 *       u8 ndim
 *       shape[]
 *       f32 act_scale           (0 = dynamic)
 *       u32 n_scales            (per-channel weight scales)
 *       f32 scales[]
 *       if is_weight: u64 nbytes + raw data   (weights carry their payload
 *           inline so the reader stream is self-describing and no blob
 *           offsets need computing)
 *   per node (n_nodes, in topological order):
 *       u8  op
 *       u8  n_inputs, n_outputs
 *       u32 inputs[], outputs[]
 *       attrs (op-dependent)
 *   u64 total_activation_bytes
 *   u32 n_lifetimes
 *   per lifetime (n_lifetimes): i32 tensor_idx, i32 first_use,
 *       i32 last_use, i32 size_bytes   (fe_plan_apply re-reads these)
 *   u64 offsets[n_tensors]      (the resolved memory plan)
 *   trailing check: EOF required
 *
 * Attrs encoding (per node, after inputs/outputs):
 *   CONV1D       i32 stride, i32 pad
 *   CONV2D       i32 sh, sw, ph, pw
 *   MAXPOOL/AVGPOOL i32 kh, kw, sh, sw
 *   BATNORM      f32 eps
 *   LAYERNORM    f32 eps
 *   GROUPNORM    i32 groups, f32 eps
 *   MULTIHEAD    i32 num_heads
 *   GEMM         i32 transA, transB, f32 alpha, f32 beta
 *   LEAKY_RELU   f32 negative_slope
 *   ELU          f32 alpha
 *   others: no attrs
 *
 * The reader is deliberately strict: magic + version check, every count and
 * extent bounds-checked against config.h ceilings, and a trailing-byte
 * check. This file is read by devices with no filesystem sandboxing, so it
 * is at least as defensive as the ONNX parser.
 */

#include "model_ser.h"
#include "kernels.h"
#include "engine.h"
#include <stdio.h>
#include <string.h>

/* --- Little-endian fixed-width field I/O (format is LE on all hosts) --- */

static FeStatus write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, 1, 1, f) == 1 ? FE_OK : FE_ERR_IO;
}
static FeStatus write_u32le(FILE *f, uint32_t v) {
    unsigned char b[4] = {
        (unsigned char)(v        & 0xFF),
        (unsigned char)((v >> 8)  & 0xFF),
        (unsigned char)((v >> 16) & 0xFF),
        (unsigned char)((v >> 24) & 0xFF),
    };
    return fwrite(b, 1, 4, f) == 4 ? FE_OK : FE_ERR_IO;
}
static FeStatus write_u64le(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    return fwrite(b, 1, 8, f) == 8 ? FE_OK : FE_ERR_IO;
}
static FeStatus write_f32le(FILE *f, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return write_u32le(f, u);
}

static FeStatus read_u8(FILE *f, uint8_t *v) {
    int c = fgetc(f);
    if (c == EOF) return FE_ERR_IO;
    *v = (uint8_t)c;
    return FE_OK;
}
static FeStatus read_u32le(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return FE_ERR_IO;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return FE_OK;
}
static FeStatus read_u64le(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) return FE_ERR_IO;
    uint64_t out = 0;
    for (int i = 0; i < 8; i++) out |= (uint64_t)b[i] << (8 * i);
    *v = out;
    return FE_OK;
}
static FeStatus read_f32le(FILE *f, float *v) {
    uint32_t u;
    FeStatus s = read_u32le(f, &u);
    if (s != FE_OK) return s;
    memcpy(v, &u, 4);
    return FE_OK;
}

/* Sanity cap for per-tensor scale counts and dimension extents. */
#define FEMD_MAX_SCALES 64

static FeStatus write_attrs(FILE *f, const FeNode *n) {
    FeStatus s;
    /* The artifact format predates dilation, groups, pool padding and the
     * vision operators and has no fields for them. Writing such a node would
     * silently drop what makes it correct, so refuse. */
    if (n->op >= FE_OP_CONCAT) return FE_ERR_DTYPE;
    if (n->op == FE_OP_CONV2D &&
        (n->attrs.conv2d.dil_h > 1 || n->attrs.conv2d.dil_w > 1 ||
         n->attrs.conv2d.groups > 1)) return FE_ERR_DTYPE;
    if ((n->op == FE_OP_MAXPOOL || n->op == FE_OP_AVGPOOL) &&
        (n->attrs.pool.ph > 0 || n->attrs.pool.pw > 0)) return FE_ERR_DTYPE;
    switch (n->op) {
        case FE_OP_CONV1D:
            s = write_u32le(f, (uint32_t)n->attrs.conv1d.stride);
            if (s != FE_OK) return s;
            return write_u32le(f, (uint32_t)n->attrs.conv1d.pad);
        case FE_OP_CONV2D:
            s = write_u32le(f, (uint32_t)n->attrs.conv2d.stride_h);
            if (s != FE_OK) return s;
            s = write_u32le(f, (uint32_t)n->attrs.conv2d.stride_w);
            if (s != FE_OK) return s;
            s = write_u32le(f, (uint32_t)n->attrs.conv2d.pad_h);
            if (s != FE_OK) return s;
            return write_u32le(f, (uint32_t)n->attrs.conv2d.pad_w);
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            s = write_u32le(f, (uint32_t)n->attrs.pool.kh);
            if (s != FE_OK) return s;
            s = write_u32le(f, (uint32_t)n->attrs.pool.kw);
            if (s != FE_OK) return s;
            s = write_u32le(f, (uint32_t)n->attrs.pool.sh);
            if (s != FE_OK) return s;
            return write_u32le(f, (uint32_t)n->attrs.pool.sw);
        case FE_OP_BATCHNORM:
            return write_f32le(f, n->attrs.batchnorm.eps);
        case FE_OP_LAYERNORM:
            return write_f32le(f, n->attrs.layernorm.eps);
        case FE_OP_GROUPNORM:
            s = write_u32le(f, (uint32_t)n->attrs.groupnorm.groups);
            if (s != FE_OK) return s;
            return write_f32le(f, n->attrs.groupnorm.eps);
        case FE_OP_MULTIHEAD_ATTN:
            return write_u32le(f, (uint32_t)n->attrs.multihead.num_heads);
        case FE_OP_GEMM:
            s = write_u32le(f, (uint32_t)n->attrs.gemm.transA);
            if (s != FE_OK) return s;
            s = write_u32le(f, (uint32_t)n->attrs.gemm.transB);
            if (s != FE_OK) return s;
            s = write_f32le(f, n->attrs.gemm.alpha);
            if (s != FE_OK) return s;
            return write_f32le(f, n->attrs.gemm.beta);
        case FE_OP_LEAKY_RELU:
            return write_f32le(f, n->attrs.leaky_relu.negative_slope);
        case FE_OP_ELU:
            return write_f32le(f, n->attrs.elu.alpha);
        default:
            return FE_OK;
    }
}

static FeStatus read_attrs(FILE *f, FeNode *n) {
    FeStatus s;
    switch (n->op) {
        case FE_OP_CONV1D: {
            uint32_t stride, pad;
            if ((s = read_u32le(f, &stride)) != FE_OK) return s;
            if ((s = read_u32le(f, &pad)) != FE_OK) return s;
            n->attrs.conv1d.stride = (int)stride;
            n->attrs.conv1d.pad    = (int)pad;
            return FE_OK;
        }
        case FE_OP_CONV2D: {
            uint32_t sh, sw, ph, pw;
            if ((s = read_u32le(f, &sh)) != FE_OK) return s;
            if ((s = read_u32le(f, &sw)) != FE_OK) return s;
            if ((s = read_u32le(f, &ph)) != FE_OK) return s;
            if ((s = read_u32le(f, &pw)) != FE_OK) return s;
            n->attrs.conv2d.stride_h = (int)sh;
            n->attrs.conv2d.stride_w = (int)sw;
            n->attrs.conv2d.pad_h    = (int)ph;
            n->attrs.conv2d.pad_w    = (int)pw;
            return FE_OK;
        }
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL: {
            uint32_t kh, kw, sh, sw;
            if ((s = read_u32le(f, &kh)) != FE_OK) return s;
            if ((s = read_u32le(f, &kw)) != FE_OK) return s;
            if ((s = read_u32le(f, &sh)) != FE_OK) return s;
            if ((s = read_u32le(f, &sw)) != FE_OK) return s;
            n->attrs.pool.kh = (int)kh; n->attrs.pool.kw = (int)kw;
            n->attrs.pool.sh = (int)sh; n->attrs.pool.sw = (int)sw;
            return FE_OK;
        }
        case FE_OP_BATCHNORM:
            return read_f32le(f, &n->attrs.batchnorm.eps);
        case FE_OP_LAYERNORM:
            return read_f32le(f, &n->attrs.layernorm.eps);
        case FE_OP_GROUPNORM: {
            uint32_t groups;
            if ((s = read_u32le(f, &groups)) != FE_OK) return s;
            if ((s = read_f32le(f, &n->attrs.groupnorm.eps)) != FE_OK) return s;
            n->attrs.groupnorm.groups = (int)groups;
            return FE_OK;
        }
        case FE_OP_MULTIHEAD_ATTN: {
            uint32_t heads;
            if ((s = read_u32le(f, &heads)) != FE_OK) return s;
            n->attrs.multihead.num_heads = (int)heads;
            return FE_OK;
        }
        case FE_OP_GEMM: {
            uint32_t ta, tb;
            if ((s = read_u32le(f, &ta)) != FE_OK) return s;
            if ((s = read_u32le(f, &tb)) != FE_OK) return s;
            if ((s = read_f32le(f, &n->attrs.gemm.alpha)) != FE_OK) return s;
            if ((s = read_f32le(f, &n->attrs.gemm.beta)) != FE_OK) return s;
            n->attrs.gemm.transA = (int)ta;
            n->attrs.gemm.transB = (int)tb;
            return FE_OK;
        }
        case FE_OP_LEAKY_RELU:
            return read_f32le(f, &n->attrs.leaky_relu.negative_slope);
        case FE_OP_ELU:
            return read_f32le(f, &n->attrs.elu.alpha);
        default:
            return FE_OK;
    }
}

/* --- Save --- */

FeStatus fe_model_save(FILE *f, const FeGraph *g, const FePlan *plan,
                       int in_tidx, int out_tidx) {
    if (!f || !g || !plan) return FE_ERR_NULL;
    if (in_tidx < 0 || in_tidx >= g->n_tensors ||
        out_tidx < 0 || out_tidx >= g->n_tensors) return FE_ERR_BOUNDS;
    if (!g->topo_valid) return FE_ERR_SHAPE;

    FeStatus s = FE_OK;
    if (s == FE_OK) s = (fwrite(FERRITE_MODEL_MAGIC, 1, 4, f) == 4)
                            ? FE_OK : FE_ERR_IO;
    if (s == FE_OK) s = write_u32le(f, FERRITE_MODEL_VERSION);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)g->n_tensors);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)g->n_nodes);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)in_tidx);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)out_tidx);

    /* Fixed input shape the artifact is planned for. */
    const FeTensorEntry *ie = &g->tensors[in_tidx];
    if (s == FE_OK) s = write_u8(f, (uint8_t)ie->ndim);
    for (int d = 0; s == FE_OK && d < ie->ndim; d++)
        s = write_u32le(f, (uint32_t)ie->shape[d]);

    /* Per-tensor entries, weights carrying their data inline. */
    for (int i = 0; i < g->n_tensors && s == FE_OK; i++) {
        const FeTensorEntry *e = &g->tensors[i];
        if (e->ndim < 0 || e->ndim > FERRITE_MAX_DIMS) return FE_ERR_SHAPE;
        s = write_u8(f, (uint8_t)e->is_weight);
        if (s != FE_OK) break;
        s = write_u8(f, (uint8_t)e->dtype);
        if (s != FE_OK) break;
        s = write_u8(f, (uint8_t)e->ndim);
        if (s != FE_OK) break;
        for (int d = 0; s == FE_OK && d < e->ndim; d++)
            s = write_u32le(f, (uint32_t)e->shape[d]);
        s = write_f32le(f, e->act_scale);
        if (s != FE_OK) break;
        if (e->n_scales < 0 || e->n_scales > FEMD_MAX_SCALES)
            return FE_ERR_SHAPE;
        s = write_u32le(f, (uint32_t)e->n_scales);
        if (s != FE_OK) break;
        for (int k = 0; s == FE_OK && k < e->n_scales; k++)
            s = write_f32le(f, e->scales[k]);
        if (s != FE_OK) break;
        if (e->is_weight) {
            if (!e->tensor || !e->tensor->data) return FE_ERR_NULL;
            s = write_u64le(f, (uint64_t)e->tensor->nbytes);
            if (s != FE_OK) break;
            if (e->tensor->nbytes > 0 &&
                fwrite(e->tensor->data, 1, e->tensor->nbytes, f) !=
                    e->tensor->nbytes) {
                s = FE_ERR_IO;
            }
        }
    }

    /* Per-node entries (topological order). */
    for (int i = 0; s == FE_OK && i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[g->topo_order[i]];
        if (n->n_inputs < 0 || n->n_inputs > FE_MAX_NODE_INPUTS ||
            n->n_outputs < 0 || n->n_outputs > FE_MAX_NODE_OUTPUTS)
            return FE_ERR_SHAPE;
        s = write_u8(f, (uint8_t)n->op);
        if (s != FE_OK) break;
        s = write_u8(f, (uint8_t)n->n_inputs);
        if (s != FE_OK) break;
        s = write_u8(f, (uint8_t)n->n_outputs);
        if (s != FE_OK) break;
        for (int k = 0; s == FE_OK && k < n->n_inputs; k++)
            s = write_u32le(f, (uint32_t)n->inputs[k]);
        for (int k = 0; s == FE_OK && k < n->n_outputs; k++)
            s = write_u32le(f, (uint32_t)n->outputs[k]);
        s = write_attrs(f, n);
    }

    /* Resolved memory plan (byte offsets + arena total + lifetimes). */
    if (s == FE_OK)
        s = write_u64le(f, (uint64_t)plan->total_activation_bytes);
    if (s == FE_OK)
        s = write_u32le(f, (uint32_t)plan->n_lifetimes);
    for (int i = 0; s == FE_OK && i < plan->n_lifetimes; i++) {
        const FeTensorLifetime *lt = &plan->lifetimes[i];
        s = write_u32le(f, (uint32_t)lt->tensor_idx);
        if (s != FE_OK) break;
        s = write_u32le(f, (uint32_t)lt->first_use);
        if (s != FE_OK) break;
        s = write_u32le(f, (uint32_t)lt->last_use);
        if (s != FE_OK) break;
        s = write_u32le(f, (uint32_t)lt->size_bytes);
    }
    for (int i = 0; s == FE_OK && i < g->n_tensors; i++)
        s = write_u64le(f, (uint64_t)plan->offsets[i]);

    return s;
}

/* --- Load --- */

FeStatus fe_model_load(FILE *f, FeArena *weight_arena, FeModel *m) {
    if (!f || !weight_arena || !m) return FE_ERR_NULL;
    memset(m, 0, sizeof(*m));

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, FERRITE_MODEL_MAGIC, 4) != 0) return FE_ERR_IO;

    uint32_t version, n_tensors, n_nodes, in_tidx, out_tidx;
    FeStatus s;
    if ((s = read_u32le(f, &version)) != FE_OK) return s;
    if (version != FERRITE_MODEL_VERSION) return FE_ERR_IO;
    if ((s = read_u32le(f, &n_tensors)) != FE_OK) return s;
    if ((s = read_u32le(f, &n_nodes)) != FE_OK) return s;
    if ((s = read_u32le(f, &in_tidx)) != FE_OK) return s;
    if ((s = read_u32le(f, &out_tidx)) != FE_OK) return s;
    if (n_tensors == 0 || n_tensors > FE_MAX_TENSORS ||
        n_nodes == 0 || n_nodes > FE_MAX_NODES ||
        in_tidx >= n_tensors || out_tidx >= n_tensors)
        return FE_ERR_SHAPE;
    m->in_tidx = (int)in_tidx;
    m->out_tidx = (int)out_tidx;

    uint8_t in_ndim;
    if ((s = read_u8(f, &in_ndim)) != FE_OK) return s;
    if (in_ndim < 1 || in_ndim > FERRITE_MAX_DIMS) return FE_ERR_SHAPE;
    m->in_ndim = (int)in_ndim;
    for (int d = 0; d < m->in_ndim; d++) {
        uint32_t dim;
        if ((s = read_u32le(f, &dim)) != FE_OK) return s;
        if (dim < 1 || dim > (uint32_t)-1 / 2) return FE_ERR_SHAPE;
        m->in_shape[d] = (int)dim;
    }

    /* Tensor registry. Weights get arena-allocated buffers immediately
     * (shape/dtype known); activations stay metadata-only. */
    for (uint32_t i = 0; i < n_tensors; i++) {
        FeTensorEntry *e = &m->graph.tensors[i];
        uint8_t is_w, dt, nd;
        if ((s = read_u8(f, &is_w)) != FE_OK) return s;
        if ((s = read_u8(f, &dt)) != FE_OK) return s;
        if ((s = read_u8(f, &nd)) != FE_OK) return s;
        if (nd > FERRITE_MAX_DIMS) return FE_ERR_SHAPE;
        if (fe_dtype_size((FeDtype)dt) == 0) return FE_ERR_DTYPE;
#ifdef FERRITE_NO_FLOAT64
        /* Device kernels (Section 4.2) ship FP32/FP16/BF16/INT8/INT16 only;
         * a double-precision model is a load-time error, not a silent
         * downcast at inference time. */
        if (dt == DTYPE_FLOAT64) return FE_ERR_DTYPE;
#endif
        e->is_weight = is_w ? 1 : 0;
        e->dtype = (FeDtype)dt;
        e->ndim = (int)nd;
        int numel = 1;
        for (int d = 0; d < e->ndim; d++) {
            uint32_t dim;
            if ((s = read_u32le(f, &dim)) != FE_OK) return s;
            if (dim < 1 || dim > (uint32_t)-1 / 2) return FE_ERR_SHAPE;
            /* Overflow guard on the running element count. */
            if ((uint64_t)numel * dim > (uint64_t)0x7FFFFFFF)
                return FE_ERR_SHAPE;
            e->shape[d] = (int)dim;
            numel *= (int)dim;
        }
        if ((s = read_f32le(f, &e->act_scale)) != FE_OK) return s;
        uint32_t n_scales;
        if ((s = read_u32le(f, &n_scales)) != FE_OK) return s;
        if (n_scales > FEMD_MAX_SCALES) return FE_ERR_SHAPE;
        e->n_scales = (int)n_scales;
        if (e->n_scales > 0) {
            e->scales = (float *)fe_arena_alloc(weight_arena,
                                                (size_t)e->n_scales *
                                                    sizeof(float),
                                                sizeof(float));
            if (!e->scales) return FE_ERR_NOMEM;
            for (int k = 0; k < e->n_scales; k++)
                if ((s = read_f32le(f, &e->scales[k])) != FE_OK) return s;
        }
        if (e->is_weight) {
            e->tensor = fe_arena_alloc_tensor(weight_arena, e->dtype,
                                              e->ndim, e->shape);
            if (!e->tensor) return FE_ERR_NOMEM;
            uint64_t nbytes;
            if ((s = read_u64le(f, &nbytes)) != FE_OK) return s;
            if (nbytes != (uint64_t)e->tensor->nbytes) return FE_ERR_IO;
            if (nbytes > 0 &&
                fread(e->tensor->data, 1, (size_t)nbytes, f) !=
                    (size_t)nbytes)
                return FE_ERR_IO;
        }
    }
    m->graph.n_tensors = (int)n_tensors;

    /* Nodes in topological order: node i is topo_order[i]. */
    for (uint32_t i = 0; i < n_nodes; i++) {
        FeNode *n = &m->graph.nodes[i];
        uint8_t op, ni, no;
        if ((s = read_u8(f, &op)) != FE_OK) return s;
        if ((s = read_u8(f, &ni)) != FE_OK) return s;
        if ((s = read_u8(f, &no)) != FE_OK) return s;
        if (ni < 1 || ni > FE_MAX_NODE_INPUTS ||
            no < 1 || no > FE_MAX_NODE_OUTPUTS) return FE_ERR_SHAPE;
        if ((int)op > FE_OP_POSITIONAL_ENCOD) return FE_ERR_DTYPE;
        n->op = (FeOpType)op;
        n->n_inputs = (int)ni;
        n->n_outputs = (int)no;
        for (int k = 0; k < ni; k++) {
            uint32_t idx;
            if ((s = read_u32le(f, &idx)) != FE_OK) return s;
            if (idx >= n_tensors) return FE_ERR_BOUNDS;
            n->inputs[k] = (int)idx;
        }
        for (int k = 0; k < no; k++) {
            uint32_t idx;
            if ((s = read_u32le(f, &idx)) != FE_OK) return s;
            if (idx >= n_tensors) return FE_ERR_BOUNDS;
            n->outputs[k] = (int)idx;
        }
        if ((s = read_attrs(f, n)) != FE_OK) return s;
    }
    m->graph.n_nodes = (int)n_nodes;
    m->graph.topo_valid = 1;             /* artifacts are pre-sorted */
    for (int i = 0; i < (int)n_nodes; i++) m->graph.topo_order[i] = i;

    /* Resolved memory plan: total, lifetimes (fe_plan_apply re-reads them),
     * then per-tensor byte offsets. */
    uint64_t total;
    if ((s = read_u64le(f, &total)) != FE_OK) return s;
    m->plan.total_activation_bytes = (size_t)total;
    uint32_t n_lifetimes;
    if ((s = read_u32le(f, &n_lifetimes)) != FE_OK) return s;
    if (n_lifetimes > FE_MAX_ALLOCS) return FE_ERR_SHAPE;
    m->plan.n_lifetimes = (int)n_lifetimes;
    for (uint32_t i = 0; i < n_lifetimes; i++) {
        FeTensorLifetime *lt = &m->plan.lifetimes[i];
        uint32_t tidx, first, last, bytes;
        if ((s = read_u32le(f, &tidx)) != FE_OK) return s;
        if ((s = read_u32le(f, &first)) != FE_OK) return s;
        if ((s = read_u32le(f, &last)) != FE_OK) return s;
        if ((s = read_u32le(f, &bytes)) != FE_OK) return s;
        if (tidx >= n_tensors || first > last) return FE_ERR_SHAPE;
        lt->tensor_idx = (int)tidx;
        lt->first_use  = (int)first;
        lt->last_use   = (int)last;
        lt->size_bytes = (int)bytes;
    }
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint64_t off;
        if ((s = read_u64le(f, &off)) != FE_OK) return s;
        m->plan.offsets[i] = (size_t)off;
    }

    /* Strict v1: nothing may follow the payload. */
    if (fgetc(f) != EOF) return FE_ERR_IO;

    /* Resolve kernels after the whole payload parsed (fail loudly on an op
     * the device build ships no kernel for). */
    m->n_steps = (int)n_nodes;
    for (int i = 0; i < m->n_steps; i++) {
        FeExecFn fn = fe_kernels_get(m->graph.nodes[i].op);
        if (!fn) {
            fprintf(stderr, "femload: no kernel for op %d\n",
                    m->graph.nodes[i].op);
            return FE_ERR_SHAPE;
        }
        m->steps[i].fn = fn;
        m->steps[i].node_index = i;
    }

    return FE_OK;
}

/* --- Device run loop --- */

static int shape_matches(const FeTensor *t, int ndim, const int *shape) {
    if (!t) return 0;
    if (t->ndim != ndim) return 0;
    for (int d = 0; d < ndim; d++)
        if (t->shape[d] != shape[d]) return 0;
    return 1;
}

FeStatus fe_model_run(FeModel *m, FeArena *weight_arena,
                      FeArena *activation_arena,
                      const FeTensor *input, FeTensor *output) {
    if (!m || !weight_arena || !activation_arena || !input || !output)
        return FE_ERR_NULL;

    /* Fixed-shape artifact: any deviation is a loud error, not a replan. */
    if (!shape_matches(input, m->in_ndim, m->in_shape)) return FE_ERR_SHAPE;

    FeGraph *g = &m->graph;
    FeStatus s;

    fe_arena_reset(activation_arena);
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) e->tensor = NULL;
    }

    s = fe_plan_apply(g, &m->plan,
                      activation_arena->base,
                      activation_arena->size,
                      activation_arena);
    if (s != FE_OK) return s;

    g->tensors[m->in_tidx].tensor = (FeTensor *)input;

    /* Output with no producer (planner skips producer-less tensors). */
    FeTensorEntry *oe = &g->tensors[m->out_tidx];
    if (!oe->tensor) {
        oe->tensor = fe_arena_alloc_tensor(activation_arena, oe->dtype,
                                           oe->ndim, oe->shape);
        if (!oe->tensor) return FE_ERR_NOMEM;
    }

    /* The flat execution loop — identical to fe_exec_run's. Kernels are the
     * same FeExecFn signature; profiling compiles out of kernels.c on
     * FERRITE_NO_PROFILER builds, leaving the profiler slot unused here. */
    FeRuntime rt;
    memset(&rt, 0, sizeof(rt));
    rt.graph         = g;
    rt.weight_arena  = *weight_arena;
    rt.activation_arena = *activation_arena;

    for (int i = 0; i < m->n_steps; i++) {
        s = m->steps[i].fn(&rt, &g->nodes[m->steps[i].node_index]);
        if (s != FE_OK) {
            fprintf(stderr, "femodel run: node '%s' failed (%d)\n",
                    g->nodes[m->steps[i].node_index].name, s);
            return s;
        }
    }

    return fe_tensor_copy(output, oe->tensor);
}