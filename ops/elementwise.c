/* ops/elementwise.c (new file) */
#include "ops.h"
#include "backend.h"
#include <string.h>

typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV } ElemOp;

static float apply_op(float x, float y, int op) {
    switch (op) {
        case OP_ADD: return x + y;
        case OP_SUB: return x - y;
        case OP_MUL: return x * y;
        default:     return x / y;
    }
}

FeStatus fe_broadcast_binary(const FeTensor *a, const FeTensor *b,
                             FeTensor *out, int op) {
    if (!a || !b || !out) return FE_ERR_NULL;
    if (op < OP_ADD || op > OP_DIV) return FE_ERR_SHAPE;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    int nd = out->ndim;
    if (nd < 1 || a->ndim > nd || b->ndim > nd) return FE_ERR_SHAPE;

    /* Element step of each operand per output dimension: 0 where it stretches. */
    int sa[FERRITE_MAX_DIMS], sb[FERRITE_MAX_DIMS];
    int acc_a = 1, acc_b = 1;
    for (int d = nd - 1; d >= 0; d--) {
        int ia = d - (nd - a->ndim), ib = d - (nd - b->ndim);
        int da = ia >= 0 ? a->shape[ia] : 1;
        int db = ib >= 0 ? b->shape[ib] : 1;
        int od = out->shape[d];
        if ((da != od && da != 1) || (db != od && db != 1)) return FE_ERR_SHAPE;
        sa[d] = da == 1 ? 0 : acc_a;
        sb[d] = db == 1 ? 0 : acc_b;
        acc_a *= da;
        acc_b *= db;
    }

    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    float *po = (float *)out->data;

    /* Odometer over every dimension but the last; the last runs as a tight loop. */
    int idx[FERRITE_MAX_DIMS] = {0};
    int last = out->shape[nd - 1];
    int step_a = sa[nd - 1], step_b = sb[nd - 1];
    size_t rows = 1;
    for (int d = 0; d < nd - 1; d++) rows *= (size_t)out->shape[d];

    for (size_t r = 0; r < rows; r++) {
        size_t oa = 0, ob = 0;
        for (int d = 0; d < nd - 1; d++) {
            oa += (size_t)idx[d] * sa[d];
            ob += (size_t)idx[d] * sb[d];
        }
        const float *x = pa + oa, *y = pb + ob;
        float *o = po + r * (size_t)last;
        if (step_b == 0) {
            float yv = y[0];
            if (step_a == 0) {
                float v = apply_op(x[0], yv, op);
                for (int i = 0; i < last; i++) o[i] = v;
            } else if (op == OP_ADD) for (int i = 0; i < last; i++) o[i] = x[i] + yv;
            else if (op == OP_SUB)   for (int i = 0; i < last; i++) o[i] = x[i] - yv;
            else if (op == OP_MUL)   for (int i = 0; i < last; i++) o[i] = x[i] * yv;
            else                     for (int i = 0; i < last; i++) o[i] = x[i] / yv;
        } else if (step_a == 0) {
            float xv = x[0];
            for (int i = 0; i < last; i++) o[i] = apply_op(xv, y[i], op);
        } else {
            for (int i = 0; i < last; i++) o[i] = apply_op(x[i], y[i], op);
        }
        for (int d = nd - 2; d >= 0; d--) {
            if (++idx[d] < out->shape[d]) break;
            idx[d] = 0;
        }
    }
    return FE_OK;
}

static FeStatus elementwise_binary(const FeTensor *a, const FeTensor *b,
                                    FeTensor *out, ElemOp op) {
    if (!a || !b || !out) return FE_ERR_NULL;
    int same = a->ndim == b->ndim && a->ndim == out->ndim;
    for (int i = 0; same && i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i])
            same = 0;
    }
    /* Different shapes are legal when they broadcast; that path validates them. */
    if (!same) return fe_broadcast_binary(a, b, out, (int)op);
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = fe_tensor_numel(a);
    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    float       *po = (float *)out->data;

    const FeSimdOps *ops = fe_simd_ops();
    switch (op) {
        case OP_ADD:
            if (ops->add) ops->add(pa, pb, po, n);
            else          for (int i = 0; i < n; i++) po[i] = pa[i] + pb[i];
            break;
        case OP_SUB: for (int i = 0; i < n; i++) po[i] = pa[i] - pb[i]; break;
        case OP_MUL:
            if (ops->mul) ops->mul(pa, pb, po, n);
            else          for (int i = 0; i < n; i++) po[i] = pa[i] * pb[i];
            break;
        case OP_DIV: for (int i = 0; i < n; i++) po[i] = pa[i] / pb[i]; break;
    }
    return FE_OK;
}

FeStatus fe_add(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_ADD);
}
FeStatus fe_sub(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_SUB);
}
FeStatus fe_mul(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_MUL);
}
FeStatus fe_div(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_DIV);
}

/* in ops/elementwise.c, alongside the tensor-tensor versions */

typedef enum { SOP_ADD, SOP_SUB, SOP_MUL, SOP_DIV } ScalarOp;

static FeStatus scalar_binary(const FeTensor *a, float scalar,
                               FeTensor *out, ScalarOp op) {
    if (!a || !out) return FE_ERR_NULL;
    if (a->ndim != out->ndim) return FE_ERR_SHAPE;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != out->shape[i]) return FE_ERR_SHAPE;
    }
    if (a->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int n = fe_tensor_numel(a);
    const float *pa = (const float *)a->data;
    float       *po = (float *)out->data;

    switch (op) {
        case SOP_ADD: for (int i = 0; i < n; i++) po[i] = pa[i] + scalar; break;
        case SOP_SUB: for (int i = 0; i < n; i++) po[i] = pa[i] - scalar; break;
        case SOP_MUL: for (int i = 0; i < n; i++) po[i] = pa[i] * scalar; break;
        case SOP_DIV: for (int i = 0; i < n; i++) po[i] = pa[i] / scalar; break;
    }
    return FE_OK;
}

FeStatus fe_add_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_ADD);
}
FeStatus fe_sub_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_SUB);
}
FeStatus fe_mul_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_MUL);
}
FeStatus fe_div_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_DIV);
}

/* Also worth adding while we're here — negation is just mul by -1,
 * but a dedicated function reads clearer at call sites */
FeStatus fe_neg(const FeTensor *a, FeTensor *out) {
    return scalar_binary(a, -1.0f, out, SOP_MUL);
}