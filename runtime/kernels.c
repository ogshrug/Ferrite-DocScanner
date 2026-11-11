/* runtime/kernels.c — per-op execution kernel wrappers (Stage 13 split).
 *
 * Holds every ex_* wrapper and the FeOpType -> FeExecFn resolution table.
 * Deliberately free of any dependency on optim/ (shape inference): the
 * device runtime links this file for kernel dispatch and never needs the
 * optimizer, so a "host-only" subsystem stays genuinely host-only.
 */

#include "kernels.h"
#include "engine.h"
#include "ops.h"
#include "log.h"

#ifndef FERRITE_NO_PROFILER
#include "profiler.h"
#endif
#ifndef FERRITE_NO_QUANT
#include "quant.h"
#endif

#ifndef FERRITE_NO_PROFILER
#define EXEC_BEGIN() uint64_t t0 = rt->profiler ? fe_profiler_now_ns() : 0
#define EXEC_END()                                              \
    do {                                                        \
        if (rt->profiler && t0)                                 \
            fe_profiler_record(rt->profiler, node->name,        \
                               fe_profiler_now_ns() - t0);      \
    } while (0)
#else
#define EXEC_BEGIN() (void)0
#define EXEC_END()   do {} while (0)
#endif

#define IN(i)  (g->tensors[node->inputs [i]].tensor)
#define OUT(i) (g->tensors[node->outputs[i]].tensor)

static FeStatus ex_noop(FeRuntime *rt, FeNode *node) {
    (void)rt; (void)node;
    return FE_OK;
}

static FeStatus ex_matmul(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    /* A per-channel-quantized INT8/INT16 weight (fe_quantize_model[-_int16])
     * routes to the matching engine-path kernel; a half-precision weight is
     * upconverted to an FP32 shadow once and runs the float kernel; anything
     * else stays float. The activation input's recorded static scale selects
     * the single-pass static kernel over per-run dynamic quantization. */
    FeTensorEntry *we  = &g->tensors[node->inputs[1]];
    FeTensorEntry *ae  = &g->tensors[node->inputs[0]];
    FeTensor       *C   = OUT(0);
    FeStatus s;

    if (we->tensor && we->tensor->dtype == DTYPE_INT8) {
#ifndef FERRITE_NO_QUANT
        if (!we->scales) {
            fe_log_error("Exec plan: INT8 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        if (ae->act_scale > 0.0f)
            s = fe_matmul_int8_static(IN(0), we->tensor, we->scales,
                                      ae->act_scale, C);
        else
            s = fe_matmul_int8_dyn(IN(0), we->tensor, we->scales, C);
#else
        fe_log_error("Exec plan: INT8 weight '%s' but quantization "
                     "is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#endif
    } else if (we->tensor && we->tensor->dtype == DTYPE_INT16) {
#ifndef FERRITE_NO_QUANT
        if (!we->scales) {
            fe_log_error("Exec plan: INT16 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        if (ae->act_scale > 0.0f)
            s = fe_matmul_int16_static(IN(0), we->tensor, we->scales,
                                       ae->act_scale, C);
        else
            s = fe_matmul_int16_dyn(IN(0), we->tensor, we->scales, C);
#else
        fe_log_error("Exec plan: INT16 weight '%s' but quantization "
                     "is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#endif
    } else if (we->tensor &&
               (we->tensor->dtype == DTYPE_FLOAT16 ||
                we->tensor->dtype == DTYPE_BFLOAT16)) {
#ifdef FERRITE_NO_QUANT
        fe_log_error("Exec plan: half-precision weight '%s' but FP32 "
                     "upconversion is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#else
        s = fe_ensure_f32_weight(g, &rt->weight_arena, node->inputs[1]);
        if (s != FE_OK) { EXEC_END(); return s; }
        s = fe_matmul(IN(0), we->shadow, C);
#endif
    } else {
        s = fe_matmul(IN(0), IN(1), C);
    }
    EXEC_END();
    return s;
}

static FeStatus ex_linear(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeTensorEntry *we  = &g->tensors[node->inputs[1]];
    FeTensorEntry *ae  = &g->tensors[node->inputs[0]];
    FeTensor       *C   = OUT(0);
    FeStatus s;
    if (we->tensor && we->tensor->dtype == DTYPE_INT8) {
#ifndef FERRITE_NO_QUANT
        if (!we->scales) {
            fe_log_error("Exec plan: INT8 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        if (ae->act_scale > 0.0f)
            s = fe_linear_int8_static(IN(0), we->tensor, we->scales, IN(2),
                                      ae->act_scale, C);
        else
            s = fe_linear_int8(IN(0), we->tensor, we->scales, IN(2), C);
#else
        fe_log_error("Exec plan: INT8 weight '%s' but quantization "
                     "is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#endif
    } else if (we->tensor && we->tensor->dtype == DTYPE_INT16) {
#ifndef FERRITE_NO_QUANT
        if (!we->scales) {
            fe_log_error("Exec plan: INT16 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        if (ae->act_scale > 0.0f)
            s = fe_linear_int16_static(IN(0), we->tensor, we->scales, IN(2),
                                       ae->act_scale, C);
        else
            s = fe_linear_int16(IN(0), we->tensor, we->scales, IN(2), C);
#else
        fe_log_error("Exec plan: INT16 weight '%s' but quantization "
                     "is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#endif
    } else if (we->tensor &&
               (we->tensor->dtype == DTYPE_FLOAT16 ||
                we->tensor->dtype == DTYPE_BFLOAT16)) {
#ifdef FERRITE_NO_QUANT
        fe_log_error("Exec plan: half-precision weight '%s' but FP32 "
                     "upconversion is compiled out", we->name);
        EXEC_END();
        return FE_ERR_DTYPE;
#else
        s = fe_ensure_f32_weight(g, &rt->weight_arena, node->inputs[1]);
        if (s != FE_OK) { EXEC_END(); return s; }
        s = fe_linear(IN(0), we->shadow, IN(2), C);
#endif
    } else {
        s = fe_linear(IN(0), IN(1), IN(2), C);
    }
    EXEC_END();
    return s;
}

static FeStatus ex_relu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_relu(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_softmax(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_softmax(IN(0), OUT(0));
    EXEC_END();
    return s;
}

/* ONNX Add: two same-shaped tensors, or any pair that broadcasts (a bias on the
 * last dimension is the common case). fe_add handles both. */
static FeStatus ex_add(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_add(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_conv1d(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    int stride = node->attrs.conv1d.stride > 0 ? node->attrs.conv1d.stride : 1;
    FeStatus s = fe_conv1d(IN(0), IN(1), node->n_inputs > 2 ? IN(2) : NULL,
                           OUT(0), stride, node->attrs.conv1d.pad);
    EXEC_END();
    return s;
}

static FeStatus ex_flatten(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeTensor *src = IN(0);
    int total = fe_tensor_numel(src);
    int flat_shape[] = { (src->ndim > 0 && src->shape[0]) ? src->shape[0] : 1,
                         total / ((src->ndim > 0 && src->shape[0]) ? src->shape[0] : 1) };
    FeTensor *view = fe_tensor_reshape(src, 2, flat_shape);
    FeStatus s = view ? fe_tensor_copy(OUT(0), view) : FE_ERR_SHAPE;
    if (view) fe_tensor_free(view);
    EXEC_END();
    return s;
}

static FeStatus ex_sub(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_sub(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_mul(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_mul(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_div(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_div(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_neg(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_neg(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_exp(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_exp(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_log(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_ln(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_pow(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_pow(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_sigmoid(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_sigmoid(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_tanh(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_tanh(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_gelu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_gelu(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_leaky_relu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_leaky_relu(IN(0), OUT(0),
                               node->attrs.leaky_relu.negative_slope);
    EXEC_END();
    return s;
}

static FeStatus ex_elu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_elu(IN(0), OUT(0), node->attrs.elu.alpha);
    EXEC_END();
    return s;
}

static FeStatus ex_swish(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_swish(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_gemm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_gemm(IN(0), node->attrs.gemm.transA,
                         IN(1), node->attrs.gemm.transB,
                         OUT(0), node->attrs.gemm.alpha,
                         node->attrs.gemm.beta);
    EXEC_END();
    return s;
}

static FeStatus ex_transpose(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_transpose(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_conv2d(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    int dh = node->attrs.conv2d.dil_h > 0 ? node->attrs.conv2d.dil_h : 1;
    int dw = node->attrs.conv2d.dil_w > 0 ? node->attrs.conv2d.dil_w : 1;
    int groups = node->attrs.conv2d.groups > 0 ? node->attrs.conv2d.groups : 1;
    FeStatus s = fe_conv2d_ex(IN(0), IN(1), node->n_inputs > 2 ? IN(2) : NULL,
                              OUT(0),
                              node->attrs.conv2d.stride_h, node->attrs.conv2d.stride_w,
                              node->attrs.conv2d.pad_h,    node->attrs.conv2d.pad_w,
                              dh, dw, groups, node->attrs.conv2d.relu);
    EXEC_END();
    return s;
}

static FeStatus ex_maxpool(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_maxpool_pad(IN(0), OUT(0), node->attrs.pool.kh,
                                node->attrs.pool.kw, node->attrs.pool.sh,
                                node->attrs.pool.sw, node->attrs.pool.ph,
                                node->attrs.pool.pw);
    EXEC_END();
    return s;
}

static FeStatus ex_avgpool(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_avgpool_pad(IN(0), OUT(0), node->attrs.pool.kh,
                                node->attrs.pool.kw, node->attrs.pool.sh,
                                node->attrs.pool.sw, node->attrs.pool.ph,
                                node->attrs.pool.pw);
    EXEC_END();
    return s;
}

static FeStatus ex_batchnorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_batchnorm(IN(0), IN(1), IN(2), IN(3), IN(4), OUT(0),
                              node->attrs.batchnorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_layernorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_layernorm(IN(0), IN(1), IN(2), OUT(0),
                              node->attrs.layernorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_groupnorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_groupnorm(IN(0), IN(1), IN(2), OUT(0),
                              node->attrs.groupnorm.groups,
                              node->attrs.groupnorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_attention(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_attention(IN(0), IN(1), IN(2), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_multihead(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_multihead_attention(IN(0), IN(1), IN(2), IN(3), IN(4),
                                        OUT(0), node->attrs.multihead.num_heads);
    EXEC_END();
    return s;
}

static FeStatus ex_embedding(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_embedding(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_posenc(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_positional_encoding(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_concat(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    const FeTensor *ins[FE_MAX_NODE_INPUTS] = {0};
    for (int i = 0; i < node->n_inputs; i++) ins[i] = IN(i);
    FeStatus s = fe_concat(ins, node->n_inputs, node->attrs.concat.axis, OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_resize(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = node->attrs.resize.mode == FE_RESIZE_LINEAR
        ? fe_resize_bilinear(IN(0), OUT(0), node->attrs.resize.coord)
        : fe_resize_nearest(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_clip(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_clip(IN(0), OUT(0), node->attrs.clip.min, node->attrs.clip.max);
    EXEC_END();
    return s;
}

static FeStatus ex_global_avgpool(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_global_avgpool(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_prelu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_prelu(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_pad(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_pad2d(IN(0), OUT(0), node->attrs.pad.top, node->attrs.pad.left,
                          node->attrs.pad.mode, node->attrs.pad.value);
    EXEC_END();
    return s;
}

#undef IN
#undef OUT

/* Resolver: FeOpType -> kernel wrapper. Indexed by the enum value so the
 * plan build is a table load, not a scan. Unlisted ops map to NULL. */
static FeExecFn const k_fns[] = {
    [FE_OP_INPUT]              = ex_noop,
    [FE_OP_OUTPUT]             = ex_noop,
    [FE_OP_MATMUL]             = ex_matmul,
    [FE_OP_LINEAR]             = ex_linear,
    [FE_OP_RELU]               = ex_relu,
    [FE_OP_SOFTMAX]            = ex_softmax,
    [FE_OP_CONV1D]             = ex_conv1d,
    [FE_OP_BATCHNORM]          = ex_batchnorm,
    [FE_OP_ADD]                = ex_add,
    [FE_OP_FLATTEN]            = ex_flatten,
    [FE_OP_SUB]                = ex_sub,
    [FE_OP_MUL]                = ex_mul,
    [FE_OP_DIV]                = ex_div,
    [FE_OP_NEG]                = ex_neg,
    [FE_OP_EXP]                = ex_exp,
    [FE_OP_LOG]                = ex_log,
    [FE_OP_POW]                = ex_pow,
    [FE_OP_SIGMOID]            = ex_sigmoid,
    [FE_OP_TANH]               = ex_tanh,
    [FE_OP_GELU]               = ex_gelu,
    [FE_OP_LEAKY_RELU]         = ex_leaky_relu,
    [FE_OP_ELU]                = ex_elu,
    [FE_OP_SWISH]              = ex_swish,
    [FE_OP_GEMM]               = ex_gemm,
    [FE_OP_TRANSPOSE]          = ex_transpose,
    [FE_OP_CONV2D]             = ex_conv2d,
    [FE_OP_MAXPOOL]            = ex_maxpool,
    [FE_OP_AVGPOOL]            = ex_avgpool,
    [FE_OP_LAYERNORM]          = ex_layernorm,
    [FE_OP_GROUPNORM]          = ex_groupnorm,
    [FE_OP_ATTENTION]          = ex_attention,
    [FE_OP_MULTIHEAD_ATTN]     = ex_multihead,
    [FE_OP_EMBEDDING]          = ex_embedding,
    [FE_OP_POSITIONAL_ENCOD]   = ex_posenc,
    [FE_OP_CONCAT]             = ex_concat,
    [FE_OP_RESIZE]             = ex_resize,
    [FE_OP_CLIP]               = ex_clip,
    [FE_OP_GLOBAL_AVGPOOL]     = ex_global_avgpool,
    [FE_OP_PRELU]              = ex_prelu,
    [FE_OP_PAD]                = ex_pad,
};

FeExecFn fe_kernels_get(int op) {
    if (op < 0 || (size_t)op >= sizeof(k_fns) / sizeof(k_fns[0]))
        return NULL;
    if (k_fns[op] == NULL)
        return NULL;
    return k_fns[op];
}