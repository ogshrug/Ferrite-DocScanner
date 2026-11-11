/* tests/test_vision.c — the image-model kernels against straightforward references. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/tensor.h"
#include "../ops/ops.h"

static FeTensor *make(int n, int c, int h, int w) {
    int shape[4] = {n, c, h, w};
    return fe_tensor_alloc(DTYPE_FLOAT32, 4, shape);
}

static void fill(FeTensor *t, unsigned seed) {
    float *d = (float *)t->data;
    int n = fe_tensor_numel(t);
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        d[i] = (float)((seed >> 9) & 0x3FFF) / 8192.0f - 1.0f;
    }
}

static float at(const FeTensor *t, int n, int c, int y, int x) {
    return ((const float *)t->data)[((n * t->shape[1] + c) * t->shape[2] + y) * t->shape[3] + x];
}

static float max_diff(const FeTensor *a, const FeTensor *b) {
    float m = 0.0f;
    int n = fe_tensor_numel(a);
    for (int i = 0; i < n; i++) {
        float d = fabsf(((const float *)a->data)[i] - ((const float *)b->data)[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Direct convolution, the definition. */
static void conv_ref(const FeTensor *in, const FeTensor *w, const FeTensor *b, FeTensor *out,
                     int sh, int sw, int ph, int pw, int dh, int dw, int groups) {
    int N = in->shape[0], C = in->shape[1], H = in->shape[2], W = in->shape[3];
    int O = w->shape[0], Cg = w->shape[1], KH = w->shape[2], KW = w->shape[3];
    int OH = out->shape[2], OW = out->shape[3], Og = O / groups;
    (void)C;
    for (int n = 0; n < N; n++)
        for (int o = 0; o < O; o++)
            for (int oy = 0; oy < OH; oy++)
                for (int ox = 0; ox < OW; ox++) {
                    double acc = b ? ((const float *)b->data)[o] : 0.0;
                    int g = o / Og;
                    for (int c = 0; c < Cg; c++)
                        for (int i = 0; i < KH; i++)
                            for (int j = 0; j < KW; j++) {
                                int y = oy * sh + i * dh - ph, x = ox * sw + j * dw - pw;
                                if (y < 0 || y >= H || x < 0 || x >= W) continue;
                                acc += (double)at(in, n, g * Cg + c, y, x) * at(w, o, c, i, j);
                            }
                    ((float *)out->data)[((n * O + o) * OH + oy) * OW + ox] = (float)acc;
                }
}

static void check_conv(int C, int H, int W, int O, int K, int s, int p, int d, int groups,
                       const char *what) {
    FeTensor *in = make(1, C, H, W), *w = make(O, C / groups, K, K);
    int bshape[1] = {O};
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, bshape);
    fill(in, 1); fill(w, 2); fill(b, 3);
    int OH = (H + 2 * p - (d * (K - 1) + 1)) / s + 1, OW = (W + 2 * p - (d * (K - 1) + 1)) / s + 1;
    FeTensor *got = make(1, O, OH, OW), *ref = make(1, O, OH, OW);
    assert(fe_conv2d_ex(in, w, b, got, s, s, p, p, d, d, groups) == FE_OK);
    conv_ref(in, w, b, ref, s, s, p, p, d, d, groups);
    float diff = max_diff(got, ref);
    if (diff > 1e-4f) printf("  %s: max diff %g\n", what, diff);
    assert(diff <= 1e-4f);
    fe_tensor_free(in); fe_tensor_free(w); fe_tensor_free(b);
    fe_tensor_free(got); fe_tensor_free(ref);
}

static void test_conv_paths(void) {
    check_conv(3, 17, 19, 8, 3, 2, 1, 1, 1, "dense 3x3 stride 2");
    check_conv(8, 12, 13, 16, 1, 1, 0, 1, 1, "pointwise 1x1");
    check_conv(16, 15, 14, 16, 3, 1, 1, 1, 16, "depthwise 3x3");
    check_conv(8, 15, 14, 16, 5, 2, 2, 1, 8, "depthwise x2 multiplier 5x5 stride 2");
    check_conv(6, 20, 21, 7, 3, 1, 4, 4, 1, "dilated 3x3 (d=4)");
    check_conv(5, 26, 24, 6, 5, 1, 6, 3, 1, "dilated 5x5 (d=3)");
    check_conv(8, 11, 9, 12, 3, 1, 1, 1, 4, "grouped (4 groups)");
    check_conv(4, 6, 6, 3, 3, 1, 9, 9, 1, "padding wider than the image");
    printf("PASS test_conv_paths\n");
}

static void test_conv_rejects_bad_groups(void) {
    FeTensor *in = make(1, 6, 5, 5), *w = make(4, 2, 3, 3), *out = make(1, 4, 5, 5);
    fill(in, 4); fill(w, 5);
    assert(fe_conv2d_ex(in, w, NULL, out, 1, 1, 1, 1, 1, 1, 2) == FE_ERR_SHAPE);  /* 2*2 != 6 */
    assert(fe_conv2d_ex(in, w, NULL, out, 1, 1, 1, 1, 0, 1, 3) == FE_ERR_SHAPE);  /* dilation 0 */
    fe_tensor_free(in); fe_tensor_free(w); fe_tensor_free(out);
    printf("PASS test_conv_rejects_bad_groups\n");
}

static void test_pool_padding(void) {
    FeTensor *in = make(1, 2, 5, 6);
    fill(in, 6);
    FeTensor *mx = make(1, 2, 3, 3), *av = make(1, 2, 3, 3);
    assert(fe_maxpool_pad(in, mx, 3, 3, 2, 2, 1, 1) == FE_OK);
    assert(fe_avgpool_pad(in, av, 3, 3, 2, 2, 1, 1) == FE_OK);
    for (int c = 0; c < 2; c++)
        for (int oy = 0; oy < 3; oy++)
            for (int ox = 0; ox < 3; ox++) {
                float best = -1e30f, sum = 0.0f;
                int count = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) {
                        int y = oy * 2 + i - 1, x = ox * 2 + j - 1;
                        if (y < 0 || y >= 5 || x < 0 || x >= 6) continue;
                        float v = at(in, 0, c, y, x);
                        if (v > best) best = v;
                        sum += v;
                        count++;
                    }
                assert(at(mx, 0, c, oy, ox) == best);
                assert(fabsf(at(av, 0, c, oy, ox) - sum / count) < 1e-6f);
            }
    fe_tensor_free(in); fe_tensor_free(mx); fe_tensor_free(av);
    printf("PASS test_pool_padding\n");
}

static void test_broadcast(void) {
    FeTensor *a = make(2, 3, 4, 5);
    int bs[3] = {3, 1, 5};
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 3, bs);
    FeTensor *out = make(2, 3, 4, 5);
    fill(a, 7); fill(b, 8);
    for (int op = 0; op < 4; op++) {
        assert(fe_broadcast_binary(a, b, out, op) == FE_OK);
        for (int n = 0; n < 2; n++) for (int c = 0; c < 3; c++)
            for (int y = 0; y < 4; y++) for (int x = 0; x < 5; x++) {
                float x0 = at(a, n, c, y, x), y0 = ((const float *)b->data)[c * 5 + x];
                float want = op == 0 ? x0 + y0 : op == 1 ? x0 - y0 : op == 2 ? x0 * y0 : x0 / y0;
                assert(fabsf(at(out, n, c, y, x) - want) <= 1e-5f * (1.0f + fabsf(want)));
            }
    }
    /* A scalar on the left. */
    int ss[1] = {1};
    FeTensor *s = fe_tensor_alloc(DTYPE_FLOAT32, 1, ss);
    ((float *)s->data)[0] = 2.0f;
    assert(fe_broadcast_binary(s, a, out, 1) == FE_OK);
    assert(at(out, 1, 2, 3, 4) == 2.0f - at(a, 1, 2, 3, 4));
    /* Shapes that do not broadcast. */
    int bad[1] = {4};
    FeTensor *c = fe_tensor_alloc(DTYPE_FLOAT32, 1, bad);
    assert(fe_broadcast_binary(a, c, out, 0) == FE_ERR_SHAPE);
    fe_tensor_free(a); fe_tensor_free(b); fe_tensor_free(out); fe_tensor_free(s); fe_tensor_free(c);
    printf("PASS test_broadcast\n");
}

static void test_concat(void) {
    FeTensor *a = make(1, 2, 3, 4), *b = make(1, 3, 3, 4), *out = make(1, 5, 3, 4);
    fill(a, 9); fill(b, 10);
    const FeTensor *ins[2] = {a, b};
    assert(fe_concat(ins, 2, 1, out) == FE_OK);
    assert(memcmp(out->data, a->data, 2 * 12 * sizeof(float)) == 0);
    assert(memcmp((float *)out->data + 24, b->data, 3 * 12 * sizeof(float)) == 0);
    FeTensor *w = make(1, 2, 3, 7);
    FeTensor *a2 = make(1, 2, 3, 3);
    fill(a2, 11);
    const FeTensor *ins2[2] = {a2, a};
    assert(fe_concat(ins2, 2, -1, w) == FE_OK);
    assert(at(w, 0, 1, 2, 2) == at(a2, 0, 1, 2, 2) && at(w, 0, 1, 2, 3) == at(a, 0, 1, 2, 0));
    assert(fe_concat(ins, 2, 2, out) == FE_ERR_SHAPE);
    fe_tensor_free(a); fe_tensor_free(b); fe_tensor_free(out); fe_tensor_free(w); fe_tensor_free(a2);
    printf("PASS test_concat\n");
}

static void test_resize(void) {
    /* Doubling a 2x2 ramp with half-pixel centres, as ONNX "linear" does. */
    FeTensor *in = make(1, 1, 2, 2), *out = make(1, 1, 4, 4);
    float v[4] = {0, 1, 2, 3};
    memcpy(in->data, v, sizeof v);
    assert(fe_resize_bilinear(in, out, FE_RESIZE_HALF_PIXEL) == FE_OK);
    float want[16] = {0, 0.25f, 0.75f, 1, 0.5f, 0.75f, 1.25f, 1.5f,
                      1.5f, 1.75f, 2.25f, 2.5f, 2, 2.25f, 2.75f, 3};
    for (int i = 0; i < 16; i++) assert(fabsf(((float *)out->data)[i] - want[i]) < 1e-6f);
    assert(fe_resize_bilinear(in, out, FE_RESIZE_ALIGN_CORNERS) == FE_OK);
    assert(fabsf(((float *)out->data)[1] - 1.0f / 3) < 1e-6f);
    assert(fe_resize_nearest(in, out) == FE_OK);
    assert(((float *)out->data)[5] == 0 && ((float *)out->data)[15] == 3);
    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_resize\n");
}

static void test_pad(void) {
    FeTensor *in = make(1, 1, 1, 4), *out = make(1, 1, 1, 8);
    float v[4] = {1, 2, 3, 4};
    memcpy(in->data, v, sizeof v);
    float want_reflect[8] = {3, 2, 1, 2, 3, 4, 3, 2};
    assert(fe_pad2d(in, out, 0, 2, FE_PAD_REFLECT, 0) == FE_OK);
    assert(memcmp(out->data, want_reflect, sizeof want_reflect) == 0);
    float want_edge[8] = {1, 1, 1, 2, 3, 4, 4, 4};
    assert(fe_pad2d(in, out, 0, 2, FE_PAD_EDGE, 0) == FE_OK);
    assert(memcmp(out->data, want_edge, sizeof want_edge) == 0);
    float want_const[8] = {9, 9, 1, 2, 3, 4, 9, 9};
    assert(fe_pad2d(in, out, 0, 2, FE_PAD_CONSTANT, 9) == FE_OK);
    assert(memcmp(out->data, want_const, sizeof want_const) == 0);
    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_pad\n");
}

static void test_small_ops(void) {
    FeTensor *in = make(1, 2, 2, 2), *out = make(1, 2, 2, 2), *g = make(1, 2, 1, 1);
    float v[8] = {-2, -1, 0, 1, 2, 3, 4, 5};
    memcpy(in->data, v, sizeof v);
    assert(fe_clip(in, out, 0.0f, 3.0f) == FE_OK);
    assert(((float *)out->data)[0] == 0 && ((float *)out->data)[7] == 3);
    int ss[1] = {2};
    FeTensor *slope = fe_tensor_alloc(DTYPE_FLOAT32, 1, ss);
    ((float *)slope->data)[0] = 0.5f;
    ((float *)slope->data)[1] = 0.25f;
    assert(fe_prelu(in, slope, out) == FE_OK);
    assert(((float *)out->data)[0] == -1.0f && ((float *)out->data)[3] == 1.0f);
    assert(fe_global_avgpool(in, g) == FE_OK);
    assert(((float *)g->data)[0] == -0.5f && ((float *)g->data)[1] == 3.5f);
    fe_tensor_free(in); fe_tensor_free(out); fe_tensor_free(g); fe_tensor_free(slope);
    printf("PASS test_small_ops\n");
}

int main(void) {
    test_conv_paths();
    test_conv_rejects_bad_groups();
    test_pool_padding();
    test_broadcast();
    test_concat();
    test_resize();
    test_pad();
    test_small_ops();
    return 0;
}
