/* ops/pool.c — max and average pooling over 2D spatial maps (Stage 3). */
#include "ops.h"
#include <float.h>

typedef struct {
    const float *src;
    float *dst;
    int H, W, OH, OW, kh, kw, sh, sw, ph, pw, is_max;
} Pool;

/* One channel plane. Windows fully inside the image skip the bounds checks. */
static void pool_plane(void *p, int plane) {
    const Pool *q = (const Pool *)p;
    int H = q->H, W = q->W, OH = q->OH, OW = q->OW, kh = q->kh, kw = q->kw;
    int sh = q->sh, sw = q->sw, ph = q->ph, pw = q->pw, is_max = q->is_max;
    const float *src = q->src + (size_t)plane * H * W;
    float *dst = q->dst + (size_t)plane * OH * OW;
    for (int oh = 0; oh < OH; oh++) {
        int y0 = oh * sh - ph;
        int inside_y = y0 >= 0 && y0 + kh <= H;
        for (int ow = 0; ow < OW; ow++) {
            int x0 = ow * sw - pw;
            float acc = is_max ? -FLT_MAX : 0.0f;
            if (inside_y && x0 >= 0 && x0 + kw <= W) {
                for (int i = 0; i < kh; i++) {
                    const float *r = src + (size_t)(y0 + i) * W + x0;
                    for (int j = 0; j < kw; j++) {
                        float v = r[j];
                        if (is_max) acc = v > acc ? v : acc;
                        else        acc += v;
                    }
                }
                if (!is_max) acc /= (float)(kh * kw);
            } else {
                int count = 0;
                for (int i = 0; i < kh; i++) {
                    int ih = y0 + i;
                    if (ih < 0 || ih >= H) continue;
                    for (int j = 0; j < kw; j++) {
                        int iw = x0 + j;
                        if (iw < 0 || iw >= W) continue;
                        float v = src[(size_t)ih * W + iw];
                        if (is_max) acc = v > acc ? v : acc;
                        else        acc += v;
                        count++;
                    }
                }
                /* ph < kh and pw < kw, so every window holds a real sample. */
                if (!is_max) acc /= (float)count;
            }
            dst[(size_t)oh * OW + ow] = acc;
        }
    }
}

/* Shared pooling driver; `is_max` selects the reduction. */
static FeStatus pool2d(const FeTensor *in, FeTensor *out,
                       int kh, int kw, int sh, int sw, int ph, int pw,
                       int is_max) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (kh < 1 || kw < 1 || sh < 1 || sw < 1) return FE_ERR_SHAPE;
    if (ph < 0 || pw < 0 || ph >= kh || pw >= kw) return FE_ERR_SHAPE;

    int N = in->shape[0], C = in->shape[1];
    int H = in->shape[2], W = in->shape[3];
    int OH = (H + 2 * ph - kh) / sh + 1;
    int OW = (W + 2 * pw - kw) / sw + 1;

    if (OH < 1 || OW < 1) return FE_ERR_SHAPE;
    if (out->ndim != 4 || out->shape[0] != N || out->shape[1] != C ||
        out->shape[2] != OH || out->shape[3] != OW) return FE_ERR_SHAPE;

    Pool q = { (const float *)in->data, (float *)out->data,
               H, W, OH, OW, kh, kw, sh, sw, ph, pw, is_max };
    fe_parallel_for(N * C, pool_plane, &q);
    return FE_OK;
}

FeStatus fe_maxpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int sh, int sw) {
    return pool2d(in, out, kh, kw, sh, sw, 0, 0, 1);
}

FeStatus fe_avgpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int sh, int sw) {
    return pool2d(in, out, kh, kw, sh, sw, 0, 0, 0);
}

FeStatus fe_maxpool_pad(const FeTensor *in, FeTensor *out, int kh, int kw,
                        int sh, int sw, int ph, int pw) {
    return pool2d(in, out, kh, kw, sh, sw, ph, pw, 1);
}

FeStatus fe_avgpool_pad(const FeTensor *in, FeTensor *out, int kh, int kw,
                        int sh, int sw, int ph, int pw) {
    return pool2d(in, out, kh, kw, sh, sw, ph, pw, 0);
}
