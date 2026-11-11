/*
 * tools/run_vision.c — runs an image model end to end and times it.
 *
 *   run_vision <model.onnx> <input.bin> <output.bin> <out_dims.cfg> [runs] [threads]
 *
 * Same file formats as run_model (FRT1 tensors, whitespace-separated output dims),
 * but the buffers are heap-allocated and sized for image models: tens of megabytes
 * of weights and hundreds of megabytes of activations at full resolution, far more
 * than run_model's static buffers hold. Prints load time and the best and median
 * of `runs` timed inferences (default 5) after one warm-up run, on `threads`
 * threads (default 1).
 *
 * Drives tools/vision_compare.py. With FERRITE_PROFILE=1 in the environment it also
 * prints the time spent per operator type over the timed runs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"
#include "../ops/ops.h"
#include "../tools/bench.h"
#include "../tools/profiler.h"

#define WEIGHT_BUF_SIZE ((size_t)160 << 20)
#define ACT_BUF_SIZE    ((size_t)1536 << 20)

static FeTensor *read_bin(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    int32_t version = 0, ndim = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FRT1", 4) != 0 ||
        fread(&version, 4, 1, f) != 1 || version != 1 ||
        fread(&ndim, 4, 1, f) != 1 || ndim < 1 || ndim > FERRITE_MAX_DIMS) {
        fclose(f);
        return NULL;
    }
    int dims[FERRITE_MAX_DIMS];
    for (int i = 0; i < ndim; i++) {
        int32_t d;
        if (fread(&d, 4, 1, f) != 1 || d < 1) { fclose(f); return NULL; }
        dims[i] = d;
    }
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, ndim, dims);
    if (!t) { fclose(f); return NULL; }
    size_t n = (size_t)fe_tensor_numel(t);
    if (fread(t->data, sizeof(float), n, f) != n) { fe_tensor_free(t); t = NULL; }
    fclose(f);
    return t;
}

static int write_bin(const char *path, const FeTensor *t) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t version = 1, ndim = t->ndim;
    fwrite("FRT1", 1, 4, f);
    fwrite(&version, 4, 1, f);
    fwrite(&ndim, 4, 1, f);
    for (int i = 0; i < t->ndim; i++) {
        int32_t d = t->shape[i];
        fwrite(&d, 4, 1, f);
    }
    fwrite(t->data, sizeof(float), (size_t)fe_tensor_numel(t), f);
    fclose(f);
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: run_vision <model.onnx> <input.bin> <output.bin> "
                        "<out_dims.cfg> [runs] [threads]\n");
        return 2;
    }
    int runs = argc >= 6 ? atoi(argv[5]) : 5;
    if (runs < 1) runs = 1;
    fe_set_num_threads(argc >= 7 ? atoi(argv[6]) : 1);

    FeTensor *input = read_bin(argv[2]);
    if (!input) { fprintf(stderr, "run_vision: cannot read %s\n", argv[2]); return 1; }

    int out_dims[FERRITE_MAX_DIMS], out_ndim = 0;
    FILE *cfg = fopen(argv[4], "r");
    if (!cfg) { fprintf(stderr, "run_vision: cannot read %s\n", argv[4]); return 1; }
    while (out_ndim < FERRITE_MAX_DIMS && fscanf(cfg, "%d", &out_dims[out_ndim]) == 1) out_ndim++;
    fclose(cfg);

    void *weight_buf = malloc(WEIGHT_BUF_SIZE);
    void *act_buf = malloc(ACT_BUF_SIZE);
    FeGraph *g = malloc(sizeof(FeGraph));
    if (!weight_buf || !act_buf || !g) { fprintf(stderr, "run_vision: out of memory\n"); return 1; }

    FeArena weights;
    fe_arena_init(&weights, weight_buf, WEIGHT_BUF_SIZE);
    double t_load = fe_bench_ms();
    if (fe_onnx_load(g, &weights, argv[1]) != FE_OK) {
        fprintf(stderr, "run_vision: failed to load %s\n", argv[1]);
        return 1;
    }
    t_load = fe_bench_ms() - t_load;

    /* The importer used the front of the weight buffer; the runtime gets the rest. */
    size_t used = weights.offset;
    FeRuntime rt;
    if (fe_runtime_init(&rt, g, (char *)weight_buf + used, WEIGHT_BUF_SIZE - used,
                        act_buf, ACT_BUF_SIZE) != FE_OK) {
        fprintf(stderr, "run_vision: runtime init failed\n");
        return 1;
    }

    /* Profile by operator type: the profiler keys on node names, so name each node
     * after its op (it has room for 64 names, fewer than most image models' nodes). */
    static FeProfiler prof;
    const char *want_profile = getenv("FERRITE_PROFILE");
    int profile = want_profile && want_profile[0] == '1';
    if (profile) {
        fe_profiler_init(&prof);
        for (int i = 0; i < g->n_nodes; i++) {
            char *name = g->nodes[i].name, *cut = strrchr(name, '_');
            if (cut) *cut = '\0';
        }
    }

    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, out_ndim, out_dims);
    double *times = malloc(sizeof(double) * (size_t)runs);
    for (int r = -1; r < runs; r++) {           /* r == -1 is the warm-up */
        double t = fe_bench_ms();
        FeStatus s = fe_runtime_run(&rt, input, output);
        t = fe_bench_ms() - t;
        if (s != FE_OK) {
            fprintf(stderr, "run_vision: inference failed with status %d\n", s);
            return 1;
        }
        if (r >= 0) times[r] = t;
        if (r == -1 && profile) rt.profiler = &prof;   /* time the measured runs only */
    }
    if (profile) fe_profiler_print(&prof);
    qsort(times, (size_t)runs, sizeof(double), cmp_double);
    printf("nodes %d  load %.0f ms  best %.1f ms  median %.1f ms  (%d runs, %d threads)\n",
           g->n_nodes, t_load, times[0], times[runs / 2], runs, fe_get_num_threads());

    int rc = write_bin(argv[3], output);
    free(times);
    fe_tensor_free(input);
    fe_tensor_free(output);
    return rc == 0 ? 0 : 1;
}
