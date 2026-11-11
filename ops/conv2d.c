/* ops/conv2d.c — 2D convolution (Stage 3, CNN family).
 *
 * Three paths, picked per call, each split across the thread pool (ops/threads.c):
 *   - depthwise (groups == C_in, one input channel per filter): direct accumulation,
 *     one kernel tap at a time, with no scratch buffer;
 *   - 1x1, stride 1, no padding: the input already is the im2col matrix, so it goes
 *     straight to GEMM;
 *   - everything else: im2col -> GEMM, one group at a time.
 *
 * The one exception to the "no allocation inside kernels" rule, mirroring
 * fe_conv1d: the im2col scratch buffer is malloc'd here and freed before
 * returning. Everything else follows the kernel contract.
 */
#include "ops.h"
#include <stdlib.h>
#include <string.h>

/* First and one-past-last output index whose input coordinate o*stride + off lies in
 * [0, n). Empty when *lo >= *hi. */
static void valid_range(int out_n, int stride, int off, int n, int *lo, int *hi) {
    int a = 0;
    if (off < 0) a = (-off + stride - 1) / stride;
    int b = out_n;
    if (off >= n) b = 0;
    else {
        int last = (n - 1 - off) / stride;      /* largest o with o*stride+off <= n-1 */
        if (last + 1 < b) b = last + 1;
    }
    *lo = a;
    *hi = b;
}

/* Build the im2col matrix for C channels starting at `src`:
 *   col: [C*KH*KW, OH*OW], col[r, oh*OW+ow] = src[c, oh*sh + i*dh - ph, ow*sw + j*dw - pw]
 * out-of-bounds reads (padding) yield 0. */
typedef struct {
    const float *src;
    float *col;
    int H, W, KH, KW, OH, OW, sh, sw, ph, pw, dh, dw;
} Im2col;

/* The KH*KW rows of the im2col matrix that come from input channel c. */
static void im2col_channel(void *p, int c) {
    const Im2col *q = (const Im2col *)p;
    int H = q->H, W = q->W, KH = q->KH, KW = q->KW, OH = q->OH, OW = q->OW;
    int sh = q->sh, sw = q->sw, ph = q->ph, pw = q->pw, dh = q->dh, dw = q->dw;
    size_t plane = (size_t)OH * OW;
    float *col = q->col;
    memset(col + (size_t)c * KH * KW * plane, 0, (size_t)KH * KW * plane * sizeof(float));
    size_t r = (size_t)c * KH * KW;
    {
        const float *chan = q->src + (size_t)c * H * W;
        for (int i = 0; i < KH; i++) {
            int oh_lo, oh_hi;
            valid_range(OH, sh, i * dh - ph, H, &oh_lo, &oh_hi);
            for (int j = 0; j < KW; j++, r++) {
                int ow_lo, ow_hi;
                valid_range(OW, sw, j * dw - pw, W, &ow_lo, &ow_hi);
                if (ow_lo >= ow_hi) continue;
                float *row = col + r * plane;
                for (int oh = oh_lo; oh < oh_hi; oh++) {
                    const float *s = chan + (size_t)(oh * sh + i * dh - ph) * W
                                          + (ow_lo * sw + j * dw - pw);
                    float *d = row + (size_t)oh * OW + ow_lo;
                    int n = ow_hi - ow_lo;
                    if (sw == 1) memcpy(d, s, (size_t)n * sizeof(float));
                    else for (int k = 0; k < n; k++) d[k] = s[(size_t)k * sw];
                }
            }
        }
    }
}

/* A non-owning 2-D view over contiguous float data. */
static void view2d(FeTensor *t, float *data, int rows, int cols) {
    t->data = data; t->dtype = DTYPE_FLOAT32; t->ndim = 2;
    t->shape[0] = rows; t->shape[1] = cols;
    t->strides[0] = cols; t->strides[1] = 1;
    t->nbytes = (size_t)rows * cols * sizeof(float);
    t->owns_data = false;
}

/* Depthwise: every output channel reads exactly one input channel. Each kernel tap
 * adds a scaled, shifted copy of the input plane to the output plane, which the
 * compiler vectorises when the stride is 1. */
typedef struct {
    const float *in, *w, *bias;
    float *out;
    int mult, H, W, KH, KW, OH, OW, sh, sw, ph, pw, dh, dw, relu;
} Depthwise;

/* One input channel and the mult output channels it feeds. */
static void depthwise_channel(void *p, int c) {
    const Depthwise *q = (const Depthwise *)p;
    const float *in = q->in, *w = q->w, *bias = q->bias;
    float *out = q->out;
    int mult = q->mult, H = q->H, W = q->W, KH = q->KH, KW = q->KW;
    int OH = q->OH, OW = q->OW, sh = q->sh, sw = q->sw, ph = q->ph, pw = q->pw;
    int dh = q->dh, dw = q->dw;
    size_t plane = (size_t)OH * OW;
    {
        const float *chan = in + (size_t)c * H * W;
        for (int m = 0; m < mult; m++) {
            int oc = c * mult + m;
            float *o = out + (size_t)oc * plane;
            float b = bias ? bias[oc] : 0.0f;
            for (size_t k = 0; k < plane; k++) o[k] = b;

            const float *kern = w + (size_t)oc * KH * KW;
            for (int i = 0; i < KH; i++) {
                int oh_lo, oh_hi;
                valid_range(OH, sh, i * dh - ph, H, &oh_lo, &oh_hi);
                for (int j = 0; j < KW; j++) {
                    int ow_lo, ow_hi;
                    valid_range(OW, sw, j * dw - pw, W, &ow_lo, &ow_hi);
                    float kv = kern[i * KW + j];
                    int n = ow_hi - ow_lo;
                    if (n <= 0 || kv == 0.0f) continue;
                    for (int oh = oh_lo; oh < oh_hi; oh++) {
                        const float *s = chan + (size_t)(oh * sh + i * dh - ph) * W
                                              + (ow_lo * sw + j * dw - pw);
                        float *d = o + (size_t)oh * OW + ow_lo;
                        if (sw == 1) for (int k = 0; k < n; k++) d[k] += kv * s[k];
                        else for (int k = 0; k < n; k++) d[k] += kv * s[(size_t)k * sw];
                    }
                }
            }
            if (q->relu)
                for (size_t k = 0; k < plane; k++) o[k] = o[k] > 0.0f ? o[k] : 0.0f;
        }
    }
}

FeStatus fe_conv2d_ex(const FeTensor *input, const FeTensor *weight,
                      const FeTensor *bias, FeTensor *out,
                      int sh, int sw, int ph, int pw,
                      int dh, int dw, int groups, int relu) {
    if (!input || !weight || !out) return FE_ERR_NULL;
    if (input->ndim  != 4) return FE_ERR_SHAPE;
    if (weight->ndim != 4) return FE_ERR_SHAPE;
    if (out->ndim    != 4) return FE_ERR_SHAPE;

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2], W = input->shape[3];

    int C_out = weight->shape[0];
    int Cg = weight->shape[1];           /* input channels per group */
    int KH = weight->shape[2], KW = weight->shape[3];
    if (sh < 1 || sw < 1 || ph < 0 || pw < 0) return FE_ERR_SHAPE;
    if (dh < 1 || dw < 1 || groups < 1) return FE_ERR_SHAPE;
    if (Cg * groups != C_in || C_out % groups != 0) return FE_ERR_SHAPE;

    int OH = (H + 2 * ph - (dh * (KH - 1) + 1)) / sh + 1;
    int OW = (W + 2 * pw - (dw * (KW - 1) + 1)) / sw + 1;
    if (OH < 1 || OW < 1) return FE_ERR_SHAPE;
    if (out->shape[0] != N || out->shape[1] != C_out ||
        out->shape[2] != OH || out->shape[3] != OW) return FE_ERR_SHAPE;
    if (input->dtype != DTYPE_FLOAT32 || weight->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (bias && (bias->dtype != DTYPE_FLOAT32 || fe_tensor_numel(bias) != C_out))
        return FE_ERR_SHAPE;

    const float *in_data = (const float *)input->data;
    const float *w_data  = (const float *)weight->data;
    const float *b_data  = bias ? (const float *)bias->data : NULL;
    float *out_data = (float *)out->data;
    size_t in_plane = (size_t)H * W, out_plane = (size_t)OH * OW;

    if (Cg == 1 && groups == C_in) {
        for (int n = 0; n < N; n++) {
            Depthwise q = { in_data + (size_t)n * C_in * in_plane, w_data, b_data,
                            out_data + (size_t)n * C_out * out_plane,
                            C_out / C_in, H, W, KH, KW, OH, OW,
                            sh, sw, ph, pw, dh, dw, relu };
            fe_parallel_for(C_in, depthwise_channel, &q);
        }
        return FE_OK;
    }

    int Og = C_out / groups;             /* output channels per group */
    int K = Cg * KH * KW;                /* rows of the im2col matrix */
    int pointwise = KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0;

    float *col = NULL;
    if (!pointwise) {
        col = malloc((size_t)K * out_plane * sizeof(float));
        if (!col) return FE_ERR_NOMEM;
    }

    FeStatus s = FE_OK;
    for (int n = 0; n < N && s == FE_OK; n++) {
        for (int g = 0; g < groups && s == FE_OK; g++) {
            const float *src = in_data + ((size_t)n * C_in + (size_t)g * Cg) * in_plane;
            FeTensor colT, Wview, out_slice;
            if (pointwise) {
                /* Read-only: GEMM never writes its inputs. */
                view2d(&colT, (float *)(size_t)src, K, (int)out_plane);
            } else {
                Im2col q = { src, col, H, W, KH, KW, OH, OW, sh, sw, ph, pw, dh, dw };
                fe_parallel_for(Cg, im2col_channel, &q);
                view2d(&colT, col, K, (int)out_plane);
            }
            view2d(&Wview, (float *)(size_t)(w_data + (size_t)g * Og * K), Og, K);
            float *o = out_data + ((size_t)n * C_out + (size_t)g * Og) * out_plane;
            view2d(&out_slice, o, Og, (int)out_plane);

            s = fe_matmul(&Wview, &colT, &out_slice);
            if (s != FE_OK) break;

            if (b_data || relu) {
                for (int oc = 0; oc < Og; oc++) {
                    float b = b_data ? b_data[g * Og + oc] : 0.0f;
                    float *row = o + (size_t)oc * out_plane;
                    if (relu)
                        for (size_t px = 0; px < out_plane; px++) {
                            float v = row[px] + b;
                            row[px] = v > 0.0f ? v : 0.0f;
                        }
                    else
                        for (size_t px = 0; px < out_plane; px++) row[px] += b;
                }
            }
        }
    }

    free(col);
    return s;
}

FeStatus fe_conv2d(const FeTensor *input, const FeTensor *weight,
                   const FeTensor *bias, FeTensor *out,
                   int sh, int sw, int ph, int pw) {
    return fe_conv2d_ex(input, weight, bias, out, sh, sw, ph, pw, 1, 1, 1, 0);
}
