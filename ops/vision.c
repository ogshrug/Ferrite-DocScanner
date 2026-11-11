/* ops/vision.c — kernels for image models: concat, resize, clip, global average pool,
 * PReLU and spatial padding.
 *
 * All follow the kernel contract: caller-allocated output of the right shape, float32
 * only, no allocation, no logging. Tensors must be contiguous.
 */
#include "ops.h"
#include <string.h>
#include <math.h>

static int all_f32(const FeTensor *a, const FeTensor *b) {
    return a->dtype == DTYPE_FLOAT32 && b->dtype == DTYPE_FLOAT32;
}

FeStatus fe_concat(const FeTensor *const *ins, int n, int axis, FeTensor *out) {
    if (!ins || !out || n < 1) return FE_ERR_NULL;
    int nd = out->ndim;
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd) return FE_ERR_SHAPE;
    if (out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    /* Elements before the axis repeat the copy; elements after it move as one run. */
    size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= (size_t)out->shape[d];
    for (int d = axis + 1; d < nd; d++) inner *= (size_t)out->shape[d];

    int total = 0;
    for (int i = 0; i < n; i++) {
        const FeTensor *t = ins[i];
        if (!t) return FE_ERR_NULL;
        if (t->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
        if (t->ndim != nd) return FE_ERR_SHAPE;
        for (int d = 0; d < nd; d++)
            if (d != axis && t->shape[d] != out->shape[d]) return FE_ERR_SHAPE;
        total += t->shape[axis];
    }
    if (total != out->shape[axis]) return FE_ERR_SHAPE;

    float *dst = (float *)out->data;
    for (size_t o = 0; o < outer; o++) {
        for (int i = 0; i < n; i++) {
            size_t run = (size_t)ins[i]->shape[axis] * inner;
            memcpy(dst, (const float *)ins[i]->data + o * run, run * sizeof(float));
            dst += run;
        }
    }
    return FE_OK;
}

/* Source coordinate for output index o along one axis. */
static float resize_coord(int o, int in, int out, int coord_mode) {
    if (coord_mode == FE_RESIZE_ALIGN_CORNERS)
        return out > 1 ? (float)o * (float)(in - 1) / (float)(out - 1) : 0.0f;
    float scale = (float)in / (float)out;
    if (coord_mode == FE_RESIZE_ASYMMETRIC) return (float)o * scale;
    return ((float)o + 0.5f) * scale - 0.5f;          /* half_pixel */
}

FeStatus fe_resize_bilinear(const FeTensor *in, FeTensor *out, int coord_mode) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4 || out->ndim != 4) return FE_ERR_SHAPE;
    if (!all_f32(in, out)) return FE_ERR_DTYPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;
    if (coord_mode < FE_RESIZE_HALF_PIXEL || coord_mode > FE_RESIZE_ASYMMETRIC)
        return FE_ERR_SHAPE;

    int planes = in->shape[0] * in->shape[1];
    int H = in->shape[2], W = in->shape[3];
    int OH = out->shape[2], OW = out->shape[3];
    if (H < 1 || W < 1 || OH < 1 || OW < 1) return FE_ERR_SHAPE;

    const float *src = (const float *)in->data;
    float *dst = (float *)out->data;

    for (int oy = 0; oy < OH; oy++) {
        float fy = resize_coord(oy, H, OH, coord_mode);
        if (fy < 0.0f) fy = 0.0f;
        if (fy > (float)(H - 1)) fy = (float)(H - 1);
        int y0 = (int)fy;
        int y1 = y0 + 1 < H ? y0 + 1 : y0;
        float wy = fy - (float)y0;
        for (int ox = 0; ox < OW; ox++) {
            float fx = resize_coord(ox, W, OW, coord_mode);
            if (fx < 0.0f) fx = 0.0f;
            if (fx > (float)(W - 1)) fx = (float)(W - 1);
            int x0 = (int)fx;
            int x1 = x0 + 1 < W ? x0 + 1 : x0;
            float wx = fx - (float)x0;
            float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
            float w10 = wy * (1.0f - wx),          w11 = wy * wx;
            for (int p = 0; p < planes; p++) {
                const float *s = src + (size_t)p * H * W;
                dst[((size_t)p * OH + oy) * OW + ox] =
                    w00 * s[y0 * W + x0] + w01 * s[y0 * W + x1] +
                    w10 * s[y1 * W + x0] + w11 * s[y1 * W + x1];
            }
        }
    }
    return FE_OK;
}

FeStatus fe_resize_nearest(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4 || out->ndim != 4) return FE_ERR_SHAPE;
    if (!all_f32(in, out)) return FE_ERR_DTYPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;

    int planes = in->shape[0] * in->shape[1];
    int H = in->shape[2], W = in->shape[3];
    int OH = out->shape[2], OW = out->shape[3];
    if (H < 1 || W < 1 || OH < 1 || OW < 1) return FE_ERR_SHAPE;

    const float *src = (const float *)in->data;
    float *dst = (float *)out->data;
    for (int p = 0; p < planes; p++) {
        const float *s = src + (size_t)p * H * W;
        float *d = dst + (size_t)p * OH * OW;
        for (int oy = 0; oy < OH; oy++) {
            int y = (int)((long long)oy * H / OH);
            for (int ox = 0; ox < OW; ox++)
                d[oy * OW + ox] = s[y * W + (int)((long long)ox * W / OW)];
        }
    }
    return FE_OK;
}

FeStatus fe_clip(const FeTensor *in, FeTensor *out, float lo, float hi) {
    if (!in || !out) return FE_ERR_NULL;
    if (!all_f32(in, out)) return FE_ERR_DTYPE;
    int n = fe_tensor_numel(in);
    if (n != fe_tensor_numel(out)) return FE_ERR_SHAPE;
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) {
        float v = p[i];
        o[i] = v < lo ? lo : (v > hi ? hi : v);
    }
    return FE_OK;
}

FeStatus fe_global_avgpool(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4 || out->ndim != 4) return FE_ERR_SHAPE;
    if (!all_f32(in, out)) return FE_ERR_DTYPE;
    if (out->shape[0] != in->shape[0] || out->shape[1] != in->shape[1] ||
        out->shape[2] != 1 || out->shape[3] != 1) return FE_ERR_SHAPE;

    int planes = in->shape[0] * in->shape[1];
    int area = in->shape[2] * in->shape[3];
    if (area < 1) return FE_ERR_SHAPE;
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int c = 0; c < planes; c++) {
        /* Summed in double so large maps keep float32 accuracy. */
        double acc = 0.0;
        const float *plane = p + (size_t)c * area;
        for (int i = 0; i < area; i++) acc += plane[i];
        o[c] = (float)(acc / area);
    }
    return FE_OK;
}

FeStatus fe_prelu(const FeTensor *in, const FeTensor *slope, FeTensor *out) {
    if (!in || !slope || !out) return FE_ERR_NULL;
    if (!all_f32(in, out) || slope->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    int n = fe_tensor_numel(in);
    if (n != fe_tensor_numel(out)) return FE_ERR_SHAPE;

    const float *p = (const float *)in->data;
    const float *a = (const float *)slope->data;
    float *o = (float *)out->data;
    int ns = fe_tensor_numel(slope);

    if (ns == 1) {
        float s = a[0];
        for (int i = 0; i < n; i++) o[i] = p[i] >= 0.0f ? p[i] : s * p[i];
        return FE_OK;
    }
    /* One slope per channel of an [N, C, ...] input. */
    if (in->ndim < 2 || ns != in->shape[1]) return FE_ERR_SHAPE;
    int C = in->shape[1];
    int inner = n / (in->shape[0] * C);
    for (int b = 0; b < in->shape[0]; b++)
        for (int c = 0; c < C; c++) {
            size_t base = ((size_t)b * C + c) * inner;
            float s = a[c];
            for (int i = 0; i < inner; i++)
                o[base + i] = p[base + i] >= 0.0f ? p[base + i] : s * p[base + i];
        }
    return FE_OK;
}

/* Source index for a padded coordinate, or -1 for "use the constant". */
static int pad_index(int i, int n, int mode) {
    if (i >= 0 && i < n) return i;
    if (mode == FE_PAD_CONSTANT) return -1;
    if (mode == FE_PAD_EDGE) return i < 0 ? 0 : n - 1;
    /* reflect, without repeating the edge sample: -1 -> 1, n -> n - 2 */
    if (n == 1) return 0;
    int period = 2 * (n - 1);
    i %= period;
    if (i < 0) i += period;
    return i < n ? i : period - i;
}

FeStatus fe_pad2d(const FeTensor *in, FeTensor *out, int top, int left,
                  int mode, float value) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4 || out->ndim != 4) return FE_ERR_SHAPE;
    if (!all_f32(in, out)) return FE_ERR_DTYPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;
    if (mode < FE_PAD_CONSTANT || mode > FE_PAD_EDGE) return FE_ERR_SHAPE;

    int planes = in->shape[0] * in->shape[1];
    int H = in->shape[2], W = in->shape[3];
    int OH = out->shape[2], OW = out->shape[3];
    if (H < 1 || W < 1 || OH < 1 || OW < 1) return FE_ERR_SHAPE;

    const float *src = (const float *)in->data;
    float *dst = (float *)out->data;
    for (int p = 0; p < planes; p++) {
        const float *s = src + (size_t)p * H * W;
        float *d = dst + (size_t)p * OH * OW;
        for (int oy = 0; oy < OH; oy++) {
            int y = pad_index(oy - top, H, mode);
            for (int ox = 0; ox < OW; ox++) {
                int x = pad_index(ox - left, W, mode);
                d[oy * OW + ox] = (y < 0 || x < 0) ? value : s[y * W + x];
            }
        }
    }
    return FE_OK;
}
