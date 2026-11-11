// importer/onnx.c
#include "ops.h"
#include "fp16.h"
#include "onnx.h"
#include "tensor.h"
#include "optim.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/* Protobuf wire format parser                                          */
/* ------------------------------------------------------------------ */

void fe_pb_init(FePbReader *r, const unsigned char *data, size_t len) {
    r->data = data;
    r->pos  = 0;
    r->len  = len;
}

int fe_pb_done(const FePbReader *r) {
    return r->pos >= r->len;
}

uint64_t fe_pb_varint(FePbReader *r) {
    uint64_t result = 0;
    int      shift  = 0;
    while (r->pos < r->len) {
        unsigned char b = r->data[r->pos++];
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;   /* MSB clear = last byte */
        shift += 7;
        if (shift >= 63) return 0;   /* > 9 bytes: runaway varint */
    }
    return result;
}

uint32_t fe_pb_fixed32(FePbReader *r) {
    if (r->pos + 4 > r->len) return 0;   /* truncated field */
    uint32_t v = (uint32_t)r->data[r->pos]
               | ((uint32_t)r->data[r->pos + 1] << 8)
               | ((uint32_t)r->data[r->pos + 2] << 16)
               | ((uint32_t)r->data[r->pos + 3] << 24);
    r->pos += 4;
    return v;
}

int fe_pb_tag(FePbReader *r, int *field_number, int *wire_type) {
    if (fe_pb_done(r)) return 0;
    uint64_t tag = fe_pb_varint(r);
    *wire_type    = (int)(tag & 0x07);
    *field_number = (int)(tag >> 3);
    return 1;
}

void fe_pb_skip(FePbReader *r, int wire_type) {
    switch (wire_type) {
        case 0: fe_pb_varint(r); break;                          /* varint */
        case 1: r->pos += 8; break;                             /* 64-bit  */
        case 2: { size_t len = (size_t)fe_pb_varint(r);         /* length-delimited */
                  if (len > r->len - r->pos) r->pos = r->len;   /* clamp: never run past EOF */
                  else r->pos += len;
                  break; }
        case 5: r->pos += 4; break;                             /* 32-bit  */
        default: r->pos = r->len; break;                        /* corrupt */
    }
    if (r->pos > r->len) r->pos = r->len;
}

/*
 * Hand out a bounds-checked view of a length-delimited field.
 *
 * The declared length is validated against the bytes actually remaining
 * BEFORE the view is published, and neither out_data nor out_len is written
 * unless the whole field is present. A caller that ignores the return value
 * therefore cannot be handed a pointer/length pair describing memory past the
 * end of the buffer — the earlier version published the view first and only
 * then reported failure, so a truncated raw_data field became an over-read.
 *
 * Advancing r->pos to r->len on failure matches fe_pb_skip's clamp: a message
 * that lies about a length is finished, not resynchronised.
 */
int fe_pb_bytes(FePbReader *r,
                const unsigned char **out_data, size_t *out_len) {
    if (fe_pb_done(r)) return 0;
    uint64_t len       = fe_pb_varint(r);
    size_t   remaining = r->len - r->pos;   /* fe_pb_done ⇒ pos < len */
    if (len > (uint64_t)remaining) {
        r->pos = r->len;
        return 0;
    }
    *out_len  = (size_t)len;
    *out_data = r->data + r->pos;
    r->pos   += *out_len;
    return 1;
}

/* Read a length-delimited field into a NUL-terminated string.
 * Returns 1 on success. buf must be at least FE_NAME_LEN bytes. */
static int pb_string(FePbReader *r, char *buf, size_t buflen) {
    const unsigned char *data;
    size_t len;
    if (!fe_pb_bytes(r, &data, &len)) return 0;
    size_t copy = len < buflen - 1 ? len : buflen - 1;
    memcpy(buf, data, copy);
    buf[copy] = '\0';
    return 1;
}

/* ------------------------------------------------------------------ */
/* ONNX op_type string → FeOpType                                      */
/* ------------------------------------------------------------------ */

static FeOpType op_type_from_string(const char *s) {
    if (strcmp(s, "MatMul")             == 0) return FE_OP_MATMUL;
    if (strcmp(s, "Gemm")               == 0) return FE_OP_LINEAR;
    if (strcmp(s, "Relu")               == 0) return FE_OP_RELU;
    if (strcmp(s, "Softmax")            == 0) return FE_OP_SOFTMAX;
    if (strcmp(s, "Conv")               == 0) return FE_OP_CONV1D;   /* upgraded to CONV2D from kernel_shape */
    if (strcmp(s, "BatchNormalization") == 0) return FE_OP_BATCHNORM;
    if (strcmp(s, "Add")                == 0) return FE_OP_ADD;
    if (strcmp(s, "Flatten")            == 0) return FE_OP_FLATTEN;
    /* ---- Basic math ---- */
    if (strcmp(s, "Sub") == 0) return FE_OP_SUB;
    if (strcmp(s, "Mul") == 0) return FE_OP_MUL;
    if (strcmp(s, "Div") == 0) return FE_OP_DIV;
    if (strcmp(s, "Neg") == 0) return FE_OP_NEG;
    if (strcmp(s, "Exp") == 0) return FE_OP_EXP;
    if (strcmp(s, "Log") == 0) return FE_OP_LOG;
    if (strcmp(s, "Pow") == 0) return FE_OP_POW;
    /* ---- Activations ---- */
    if (strcmp(s, "Sigmoid") == 0) return FE_OP_SIGMOID;
    if (strcmp(s, "Tanh")    == 0) return FE_OP_TANH;
    if (strcmp(s, "Gelu")    == 0) return FE_OP_GELU;
    if (strcmp(s, "LeakyRelu") == 0) return FE_OP_LEAKY_RELU;
    if (strcmp(s, "Elu")     == 0) return FE_OP_ELU;
    if (strcmp(s, "Swish")   == 0) return FE_OP_SWISH;
    /* ---- Linear algebra / views ---- */
    if (strcmp(s, "Transpose") == 0) return FE_OP_TRANSPOSE;
    /* ---- CNN ---- */
    if (strcmp(s, "MaxPool")      == 0) return FE_OP_MAXPOOL;
    if (strcmp(s, "AveragePool")  == 0) return FE_OP_AVGPOOL;
    /* ---- Norms ---- */
    if (strcmp(s, "LayerNormalization")       == 0) return FE_OP_LAYERNORM;
    if (strcmp(s, "GroupNormalization")       == 0) return FE_OP_GROUPNORM;
    /* ---- Sequence (ONNX contrib opset 'MultiHeadAttention') ---- */
    if (strcmp(s, "MultiHeadAttention")       == 0) return FE_OP_MULTIHEAD_ATTN;
    /* ---- Vision ---- */
    if (strcmp(s, "Concat")            == 0) return FE_OP_CONCAT;
    if (strcmp(s, "Resize")            == 0) return FE_OP_RESIZE;
    if (strcmp(s, "Clip")              == 0) return FE_OP_CLIP;
    if (strcmp(s, "GlobalAveragePool") == 0) return FE_OP_GLOBAL_AVGPOOL;
    if (strcmp(s, "ReduceMean")        == 0) return FE_OP_GLOBAL_AVGPOOL; /* axes checked in parse_node */
    if (strcmp(s, "PRelu")             == 0) return FE_OP_PRELU;
    if (strcmp(s, "Pad")               == 0) return FE_OP_PAD;
    return (FeOpType)-1;   /* unsupported */
}

/* ------------------------------------------------------------------ */
/* Tensor name → graph tensor index lookup                             */
/* ------------------------------------------------------------------ */

static int find_or_add_tensor(FeGraph *g, const char *name) {
    /* Search existing tensors by name */
    for (int i = 0; i < g->n_tensors; i++) {
        if (strncmp(g->tensors[i].name, name, FE_NAME_LEN) == 0)
            return i;
    }
    /* Not found — add a placeholder (shape filled in later) */
    int shape[] = {0};
    int idx = fe_graph_add_tensor(g, name, DTYPE_FLOAT32, 1, shape, 0);
    if (idx < 0)
        fprintf(stderr,
                "ONNX: tensor registry full at %d entries while adding '%s' "
                "— raise FE_MAX_TENSORS to load this model\n",
                FE_MAX_TENSORS, name);
    return idx;
}

/* ------------------------------------------------------------------ */
/* Parse TensorProto (initializer / weight)                            */
/* ------------------------------------------------------------------ */

/*
 * TensorProto field numbers:
 *   1 = dims      (repeated int64)
 *   2 = data_type (int32)
 *   4 = float_data (repeated float, packed or not — used by some exporters
 *                   for small weight tensors instead of raw_data)
 *   7 = int64_data (repeated int64, packed or not)
 *   8 = name      (string)
 *   9 = raw_data  (bytes — values packed little-endian)
 *
 * FLOAT (1) loads as is, and FLOAT16 (10, raw_data only) is widened to float32. INT32 (6) and INT64 (7) tensors, which ONNX uses for
 * shapes, pads and axes, are converted to float32 so that the rest of the
 * runtime has one weight type; they are small integers, which float32 holds
 * exactly. A scalar (no dims) loads as a one-element vector.
 *
 * Fixed32 values in field 7 are also read as floats: earlier versions of this
 * parser took field 7 for float_data, and files written against that keep
 * loading.
 */
/* Appends one value to a growing float buffer. Returns 0 when out of memory. */
static int push_value(float **vals, size_t *n, size_t *cap, float v) {
    if (*n == *cap) {
        size_t nc = *cap ? *cap * 2 : 64;
        float *nf = (float *)realloc(*vals, nc * sizeof(float));
        if (!nf) return 0;
        *vals = nf;
        *cap = nc;
    }
    (*vals)[(*n)++] = v;
    return 1;
}

static FeStatus parse_initializer(FePbReader *r, FeGraph *g,
                                   FeArena *weight_arena) {
    char    name[FE_NAME_LEN] = {0};
    int     dims[FERRITE_MAX_DIMS];
    int     ndim     = 0;
    int     data_type = 0;
    const unsigned char *raw_data = NULL;
    size_t  raw_len  = 0;

    /* float_data (field 7) is repeated float — accumulate in a growing
     * buffer. Sparse path; exporter-generated small weights only. */
    float  *fvals     = NULL;
    size_t  n_fvals   = 0;
    size_t  cap_fvals = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: /* dims — repeated int64 */
                /* Dropping an extent silently would load a rank-9 tensor as a
                 * rank-8 one with a different shape and wrong results. Refuse
                 * instead, and name the knob that lifts the ceiling. */
                if (ndim >= FERRITE_MAX_DIMS) {
                    fprintf(stderr,
                            "ONNX: initializer '%s' has more than %d dimensions "
                            "— raise FERRITE_MAX_DIMS to load this model\n",
                            name[0] ? name : "(unnamed)", FERRITE_MAX_DIMS);
                    free(fvals);
                    return FE_ERR_SHAPE;
                }
                dims[ndim++] = (int)fe_pb_varint(r);
                break;
            case 2: data_type = (int)fe_pb_varint(r); break;
            case 4: /* float_data */
            case 7: /* int64_data; fixed32 here is legacy float_data */
                if (wtype == 5) {
                    uint32_t bits = fe_pb_fixed32(r);
                    float v;
                    memcpy(&v, &bits, 4);
                    if (!push_value(&fvals, &n_fvals, &cap_fvals, v)) {
                        free(fvals);
                        return FE_ERR_NOMEM;
                    }
                } else if (wtype == 0 && field == 7) {
                    float v = (float)(int64_t)fe_pb_varint(r);
                    if (!push_value(&fvals, &n_fvals, &cap_fvals, v)) {
                        free(fvals);
                        return FE_ERR_NOMEM;
                    }
                } else if (wtype == 2) {
                    /* Packed: one length-delimited run of values. */
                    const unsigned char *pd;
                    size_t pl;
                    if (!fe_pb_bytes(r, &pd, &pl)) { free(fvals); return FE_ERR_SHAPE; }
                    FePbReader pr;
                    fe_pb_init(&pr, pd, pl);
                    while (!fe_pb_done(&pr)) {
                        float v;
                        if (field == 4) {
                            if (pr.pos + 4 > pr.len) { free(fvals); return FE_ERR_SHAPE; }
                            uint32_t bits = fe_pb_fixed32(&pr);
                            memcpy(&v, &bits, 4);
                        } else {
                            v = (float)(int64_t)fe_pb_varint(&pr);
                        }
                        if (!push_value(&fvals, &n_fvals, &cap_fvals, v)) {
                            free(fvals);
                            return FE_ERR_NOMEM;
                        }
                    }
                } else {
                    fe_pb_skip(r, wtype);
                }
                break;
            case 8: pb_string(r, name, FE_NAME_LEN); break;
            case 9: /* raw_data — a truncated field is fatal, never partial */
                if (!fe_pb_bytes(r, &raw_data, &raw_len)) {
                    fprintf(stderr, "ONNX: initializer '%s' declares more "
                            "raw_data than the file holds\n",
                            name[0] ? name : "(unnamed)");
                    free(fvals);
                    return FE_ERR_SHAPE;
                }
                break;
            default: fe_pb_skip(r, wtype); break;
        }
    }

    /* A scalar has no dims; give it one so it is an ordinary tensor. An
     * initializer with neither dims nor data is still malformed. */
    if (ndim == 0) {
        if (!raw_data && n_fvals == 0) { free(fvals); return FE_ERR_SHAPE; }
        dims[0] = 1;
        ndim = 1;
    }

    if (data_type != 0 && data_type != 1 && data_type != 6 && data_type != 7 &&
        !(data_type == 10 && raw_data)) {
        fprintf(stderr, "ONNX: initializer '%s' has unsupported data type %d "
                "(float32, raw float16, int32 and int64 load)\n",
                name[0] ? name : "(unnamed)", data_type);
        free(fvals);
        return FE_ERR_DTYPE;
    }

    /*
     * Bounds: reject absurd dims before integer overflow can bite.
     *
     * Per-dim limits alone are not enough — 4 dims of 2^20 each are all legal
     * individually but their product is 2^80, which overflows int64_t (signed
     * overflow is undefined, not merely wrapping). Cap the running product
     * too. 2^28 elements is 1 GiB of float32, far past any arena this runtime
     * binds, so no real model is turned away.
     */
    enum { FE_MAX_INIT_ELEMS = 1 << 28 };
    int64_t numel = 1;
    for (int d = 0; d < ndim; d++) {
        if (dims[d] <= 0 || dims[d] > (1 << 20)) { free(fvals); return FE_ERR_SHAPE; }
        numel *= dims[d];
        if (numel > FE_MAX_INIT_ELEMS) {
            fprintf(stderr, "ONNX: initializer '%s' claims %lld elements "
                    "(limit %d)\n", name[0] ? name : "(unnamed)",
                    (long long)numel, FE_MAX_INIT_ELEMS);
            free(fvals);
            return FE_ERR_SHAPE;
        }
    }

    /* Find or register this tensor in the graph */
    int tidx = find_or_add_tensor(g, name);
    if (tidx < 0) { free(fvals); return FE_ERR_NOMEM; }
    FeTensorEntry *e = &g->tensors[tidx];
    e->ndim      = ndim;
    e->dtype     = DTYPE_FLOAT32;
    e->is_weight = 1;
    memcpy(e->shape, dims, ndim * sizeof(int));

    /* Allocate and copy weight data into the arena */
    e->tensor = fe_arena_alloc_tensor(weight_arena,
                                       DTYPE_FLOAT32, ndim, dims);
    if (!e->tensor) { free(fvals); return FE_ERR_NOMEM; }

    if (raw_data && data_type == 10) {
        /* Half-precision storage, widened once here: every kernel works in float32. */
        if (raw_len != (size_t)(numel * 2)) { free(fvals); return FE_ERR_SHAPE; }
        float *dst = (float *)e->tensor->data;
        for (int64_t k = 0; k < numel; k++) {
            uint16_t h = (uint16_t)(raw_data[k * 2] | (raw_data[k * 2 + 1] << 8));
            dst[k] = fe_fp16_to_f32(h);
        }
    } else if (raw_data && data_type == 7) {
        if (raw_len != (size_t)(numel * 8)) { free(fvals); return FE_ERR_SHAPE; }
        float *dst = (float *)e->tensor->data;
        for (int64_t k = 0; k < numel; k++) {
            uint64_t bits = 0;
            for (int b = 0; b < 8; b++)
                bits |= (uint64_t)raw_data[k * 8 + b] << (8 * b);
            dst[k] = (float)(int64_t)bits;
        }
    } else if (raw_data && data_type == 6) {
        if (raw_len != (size_t)(numel * 4)) { free(fvals); return FE_ERR_SHAPE; }
        float *dst = (float *)e->tensor->data;
        for (int64_t k = 0; k < numel; k++) {
            uint32_t bits = 0;
            for (int b = 0; b < 4; b++)
                bits |= (uint32_t)raw_data[k * 4 + b] << (8 * b);
            dst[k] = (float)(int32_t)bits;
        }
    } else if (raw_data) {
        if (raw_len != (size_t)(numel * 4)) { free(fvals); return FE_ERR_SHAPE; }
        memcpy(e->tensor->data, raw_data, raw_len);
    } else if (n_fvals > 0) {
        if ((int64_t)n_fvals != numel) { free(fvals); return FE_ERR_SHAPE; }
        memcpy(e->tensor->data, fvals, n_fvals * sizeof(float));
    } else {
        free(fvals);
        return FE_ERR_SHAPE;
    }

    free(fvals);
    return FE_OK;
}

/* ------------------------------------------------------------------ */
/* Node attributes                                                     */
/* ------------------------------------------------------------------ */

#define FE_MAX_ATTRIBUTES 8

/*
 * AttributeProto field numbers:
 *   1 = name    (string)
 *   2 = f       (float)
 *   3 = i       (int64)
 *   4 = s       (bytes — a short string such as a mode name)
 *   7 = floats  (repeated float)
 *   8 = ints    (repeated int64)
 */
typedef struct {
    char name[64];
    int  ints[8];
    int  n_ints;
    int  int_val;
    float float_val;
    int  has_i;
    int  has_f;
    char str[32];
    int  has_s;
} FeAttr;

/* Set by apply_attr when an attribute asks for something the kernels cannot
 * do; parse_node turns it into a load failure. */
static int s_attr_unsupported;

/* Where each kept input of each node sat in the ONNX node's input list. ONNX
 * marks an omitted optional input with an empty name, which is dropped here,
 * so position is the only way to tell Resize's `scales` from its `sizes`.
 * Used only between parsing and resolve_constant_inputs. */
static unsigned char s_input_pos[FE_MAX_NODES][FE_MAX_NODE_INPUTS];

/* Parse a single AttributeProto sub-message into *a. */
static void parse_attr(FePbReader *r, FeAttr *a) {
    memset(a, 0, sizeof(*a));

    while (!fe_pb_done(r)) {
        int af, awt;
        if (!fe_pb_tag(r, &af, &awt)) break;
        switch (af) {
            case 1: pb_string(r, a->name, sizeof(a->name)); break;
            case 2: { /* f — 32-bit little-endian float. */
                if (r->pos + 4 <= r->len) {
                    const unsigned char *p = r->data + r->pos;
                    unsigned bits = (unsigned)p[0]
                                   | ((unsigned)p[1] << 8)
                                   | ((unsigned)p[2] << 16)
                                   | ((unsigned)p[3] << 24);
                    memcpy(&a->float_val, &bits, 4);
                    a->has_f = 1;
                }
                r->pos += 4;
                break;
            }
            case 3: a->int_val = (int)fe_pb_varint(r); a->has_i = 1; break;
            case 4: pb_string(r, a->str, sizeof(a->str)); a->has_s = 1; break;
            case 8: /* ints — repeated varint */
                if (a->n_ints < 8)
                    a->ints[a->n_ints++] = (int)fe_pb_varint(r);
                else
                    fe_pb_varint(r);
                break;
            default: fe_pb_skip(r, awt); break;
        }
    }
}

/* Apply one parsed attribute to a node whose op is already known.
 * Unknown attributes on a supported op are a warning, never an error. */
static void apply_attr(FeNode *node, FeOpType op,
                       const FeAttr *a) {
    if (strcmp(a->name, "pads") == 0 || strcmp(a->name, "strides") == 0 ||
        strcmp(a->name, "kernel_shape") == 0) {
        switch (op) {
            case FE_OP_CONV1D:
                if (strcmp(a->name, "pads") == 0 && a->n_ints >= 1)
                    node->attrs.conv1d.pad = a->ints[0];
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 1)
                    node->attrs.conv1d.stride = a->ints[0];
                break;
            case FE_OP_CONV2D:
                if (strcmp(a->name, "pads") == 0 && a->n_ints >= 4) {
                    /* ONNX order: top, left, bottom, right. */
                    node->attrs.conv2d.pad_h = a->ints[0];
                    node->attrs.conv2d.pad_w = a->ints[1];
                    if (a->ints[2] != a->ints[0] || a->ints[3] != a->ints[1]) {
                        fprintf(stderr, "ONNX: Conv with different padding on "
                                "opposite sides is not supported\n");
                        s_attr_unsupported = 1;
                    }
                } else if (strcmp(a->name, "pads") == 0 && a->n_ints >= 2) {
                    node->attrs.conv2d.pad_h = a->ints[0];
                    node->attrs.conv2d.pad_w = a->ints[1];
                }
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 2) {
                    node->attrs.conv2d.stride_h = a->ints[0];
                    node->attrs.conv2d.stride_w = a->ints[1];
                } else if (strcmp(a->name, "strides") == 0 && a->n_ints >= 1) {
                    node->attrs.conv2d.stride_h = a->ints[0];
                    node->attrs.conv2d.stride_w = a->ints[0];
                }
                break;
            case FE_OP_MAXPOOL:
            case FE_OP_AVGPOOL:
                if (strcmp(a->name, "kernel_shape") == 0 && a->n_ints >= 2) {
                    node->attrs.pool.kh = a->ints[0];
                    node->attrs.pool.kw = a->ints[1];
                } else if (strcmp(a->name, "kernel_shape") == 0 && a->n_ints == 1) {
                    node->attrs.pool.kh = a->ints[0];
                    node->attrs.pool.kw = a->ints[0];
                }
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 2) {
                    node->attrs.pool.sh = a->ints[0];
                    node->attrs.pool.sw = a->ints[1];
                } else if (strcmp(a->name, "strides") == 0 && a->n_ints == 1) {
                    node->attrs.pool.sh = a->ints[0];
                    node->attrs.pool.sw = a->ints[0];
                }
                if (strcmp(a->name, "pads") == 0 && a->n_ints >= 4) {
                    node->attrs.pool.ph = a->ints[0];
                    node->attrs.pool.pw = a->ints[1];
                    if (a->ints[2] != a->ints[0] || a->ints[3] != a->ints[1]) {
                        fprintf(stderr, "ONNX: pooling with different padding "
                                "on opposite sides is not supported\n");
                        s_attr_unsupported = 1;
                    }
                }
                break;
            case FE_OP_PAD:
                /* Opset < 11 carries pads as an attribute; later ones as an
                 * input, read in resolve_constant_inputs. Order: N, C, H, W
                 * begins, then the four ends. */
                if (strcmp(a->name, "pads") == 0 && a->n_ints == 8) {
                    if (a->ints[0] || a->ints[1] || a->ints[4] || a->ints[5]) {
                        fprintf(stderr, "ONNX: Pad on the batch or channel "
                                "dimension is not supported\n");
                        s_attr_unsupported = 1;
                    }
                    node->attrs.pad.top    = a->ints[2];
                    node->attrs.pad.left   = a->ints[3];
                    node->attrs.pad.bottom = a->ints[6];
                    node->attrs.pad.right  = a->ints[7];
                }
                break;
            default: break;
        }
        return;
    }

    switch (op) {
        case FE_OP_CONV2D:
            if (strcmp(a->name, "dilations") == 0 && a->n_ints >= 2) {
                node->attrs.conv2d.dil_h = a->ints[0];
                node->attrs.conv2d.dil_w = a->ints[1];
            }
            if (strcmp(a->name, "group") == 0 && a->has_i)
                node->attrs.conv2d.groups = a->int_val;
            if (strcmp(a->name, "auto_pad") == 0 && a->has_s &&
                strcmp(a->str, "NOTSET") != 0) {
                fprintf(stderr, "ONNX: Conv auto_pad=%s is not supported "
                        "(export with explicit pads)\n", a->str);
                s_attr_unsupported = 1;
            }
            break;
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            if ((strcmp(a->name, "ceil_mode") == 0 && a->has_i && a->int_val != 0) ||
                (strcmp(a->name, "count_include_pad") == 0 && a->has_i && a->int_val != 0) ||
                (strcmp(a->name, "auto_pad") == 0 && a->has_s &&
                 strcmp(a->str, "NOTSET") != 0)) {
                fprintf(stderr, "ONNX: pooling attribute '%s' is not supported\n",
                        a->name);
                s_attr_unsupported = 1;
            }
            if (strcmp(a->name, "dilations") == 0)
                for (int k = 0; k < a->n_ints; k++)
                    if (a->ints[k] != 1) {
                        fprintf(stderr, "ONNX: dilated pooling is not supported\n");
                        s_attr_unsupported = 1;
                    }
            break;
        case FE_OP_CONCAT:
            if (strcmp(a->name, "axis") == 0 && a->has_i)
                node->attrs.concat.axis = a->int_val;
            break;
        case FE_OP_RESIZE:
            if (strcmp(a->name, "mode") == 0 && a->has_s) {
                if (strcmp(a->str, "linear") == 0)
                    node->attrs.resize.mode = FE_RESIZE_LINEAR;
                else if (strcmp(a->str, "nearest") == 0)
                    node->attrs.resize.mode = FE_RESIZE_NEAREST;
                else {
                    fprintf(stderr, "ONNX: Resize mode '%s' is not supported\n", a->str);
                    s_attr_unsupported = 1;
                }
            }
            if (strcmp(a->name, "coordinate_transformation_mode") == 0 && a->has_s) {
                if (strcmp(a->str, "half_pixel") == 0)
                    node->attrs.resize.coord = FE_RESIZE_HALF_PIXEL;
                else if (strcmp(a->str, "align_corners") == 0)
                    node->attrs.resize.coord = FE_RESIZE_ALIGN_CORNERS;
                else if (strcmp(a->str, "asymmetric") == 0)
                    node->attrs.resize.coord = FE_RESIZE_ASYMMETRIC;
                else {
                    fprintf(stderr, "ONNX: Resize coordinate mode '%s' is not "
                            "supported\n", a->str);
                    s_attr_unsupported = 1;
                }
            }
            break;
        case FE_OP_CLIP:           /* opset < 11 keeps the bounds as attributes */
            if (strcmp(a->name, "min") == 0 && a->has_f) node->attrs.clip.min = a->float_val;
            if (strcmp(a->name, "max") == 0 && a->has_f) node->attrs.clip.max = a->float_val;
            break;
        case FE_OP_PAD:
            if (strcmp(a->name, "mode") == 0 && a->has_s) {
                if (strcmp(a->str, "constant") == 0)     node->attrs.pad.mode = FE_PAD_CONSTANT;
                else if (strcmp(a->str, "reflect") == 0) node->attrs.pad.mode = FE_PAD_REFLECT;
                else if (strcmp(a->str, "edge") == 0)    node->attrs.pad.mode = FE_PAD_EDGE;
                else {
                    fprintf(stderr, "ONNX: Pad mode '%s' is not supported\n", a->str);
                    s_attr_unsupported = 1;
                }
            }
            if (strcmp(a->name, "value") == 0 && a->has_f)
                node->attrs.pad.value = a->float_val;
            break;
        case FE_OP_LEAKY_RELU:
            if (strcmp(a->name, "alpha") == 0 && a->has_f)
                node->attrs.leaky_relu.negative_slope = a->float_val;
            break;
        case FE_OP_ELU:
            if (strcmp(a->name, "alpha") == 0 && a->has_f)
                node->attrs.elu.alpha = a->float_val;
            break;
        case FE_OP_BATCHNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.batchnorm.eps = a->float_val;
            break;
        case FE_OP_LAYERNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.layernorm.eps = a->float_val;
            break;
        case FE_OP_GROUPNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.groupnorm.eps = a->float_val;
            if (strcmp(a->name, "num_groups") == 0 && a->has_i)
                node->attrs.groupnorm.groups = a->int_val;
            break;
        case FE_OP_MULTIHEAD_ATTN:
            if (strcmp(a->name, "num_heads") == 0 && a->has_i)
                node->attrs.multihead.num_heads = a->int_val;
            break;
        case FE_OP_SOFTMAX:
            /* Engine computes softmax over the last axis always. ONNX writes
             * axis=-1 for "last", or the explicit last-axis index. Record the
             * axis here and judge it in check_softmax_axes() once shape
             * inference has run — output ranks are not yet known at parse
             * time, and guessing off an unresolved rank reports valid models
             * as unsupported. */
            if (strcmp(a->name, "axis") == 0 && a->has_i)
                node->attrs.softmax.axis = (int)a->int_val;
            break;
        case FE_OP_LINEAR:
            /* Gemm maps to C = A @ W + b (alpha=1, beta=1, no transpose).
             * Non-default Gemm attributes would change the result — flag
             * them rather than silently computing something else. */
            if (strcmp(a->name, "transA") == 0 && a->has_i && a->int_val != 0)
                fprintf(stderr, "ONNX: unsupported Gemm transA (mapped to "
                        "Linear without transpose)\n");
            if (strcmp(a->name, "transB") == 0 && a->has_i && a->int_val != 0)
                fprintf(stderr, "ONNX: unsupported Gemm transB (mapped to "
                        "Linear without transpose)\n");
            if (strcmp(a->name, "alpha") == 0 && a->has_f &&
                a->float_val != 1.0f)
                fprintf(stderr, "ONNX: unsupported Gemm alpha (mapped to "
                        "Linear with alpha=1)\n");
            if (strcmp(a->name, "beta") == 0 && a->has_f &&
                a->float_val != 1.0f)
                fprintf(stderr, "ONNX: unsupported Gemm beta (mapped to "
                        "Linear with bias included)\n");
            break;
        default:
            break;
    }
}

/* ------------------------------------------------------------------ */
/* Parse NodeProto                                                      */
/* ------------------------------------------------------------------ */

/*
 * NodeProto field numbers:
 *   1 = input     (repeated string)
 *   2 = output    (repeated string)
 *   3 = name      (string)
 *   4 = op_type   (string)
 *   5 = attribute (repeated AttributeProto)
 */
static FeStatus parse_node(FePbReader *r, FeGraph *g) {
    char     node_name[FE_NAME_LEN] = {0};
    char     op_str   [FE_NAME_LEN] = {0};
    int      inputs [FE_MAX_NODE_INPUTS];
    int      outputs[FE_MAX_NODE_OUTPUTS];
    int      n_in = 0, n_out = 0;
    int      in_pos[FE_MAX_NODE_INPUTS];
    int      n_seen_in = 0;         /* inputs listed, including omitted ones */
    FeAttr   attrs[FE_MAX_ATTRIBUTES];
    int      n_attrs = 0;
    int      kernel_dims = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                int pos = n_seen_in++;
                if (tname[0] == '\0') break;        /* omitted optional input */
                if (n_in >= FE_MAX_NODE_INPUTS) {
                    /* Dropping an input would run a different computation. */
                    fprintf(stderr, "ONNX: a node has more than %d inputs\n",
                            FE_MAX_NODE_INPUTS);
                    return FE_ERR_SHAPE;
                }
                {
                    int t = find_or_add_tensor(g, tname);
                    if (t < 0) return FE_ERR_NOMEM;   /* registry exhausted */
                    in_pos[n_in] = pos;
                    inputs[n_in++] = t;
                }
                break;
            }
            case 2: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                if (n_out < FE_MAX_NODE_OUTPUTS) {
                    int t = find_or_add_tensor(g, tname);
                    if (t < 0) return FE_ERR_NOMEM;   /* registry exhausted */
                    outputs[n_out++] = t;
                }
                break;
            }
            case 3: pb_string(r, node_name, FE_NAME_LEN); break;
            case 4: pb_string(r, op_str,    FE_NAME_LEN); break;
            case 5: {
                const unsigned char *attr_data;
                size_t attr_len;
                if (!fe_pb_bytes(r, &attr_data, &attr_len)) break;
                if (n_attrs < FE_MAX_ATTRIBUTES) {
                    FePbReader ar;
                    fe_pb_init(&ar, attr_data, attr_len);
                    parse_attr(&ar, &attrs[n_attrs]);
                    if (strcmp(attrs[n_attrs].name, "kernel_shape") == 0)
                        kernel_dims = attrs[n_attrs].n_ints;
                    n_attrs++;
                }
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }

    FeOpType op = op_type_from_string(op_str);
    if ((int)op == -1) {
        fprintf(stderr, "ONNX: unsupported op '%s'\n", op_str);
        return FE_ERR_SHAPE;
    }

    /* ONNX "Conv" is N-D; pick 1D or 2D from kernel_shape's rank. */
    if (op == FE_OP_CONV1D && kernel_dims >= 2) op = FE_OP_CONV2D;

    if (node_name[0] == '\0') snprintf(node_name, FE_NAME_LEN, "%s", op_str);

    int idx = fe_graph_add_node(g, node_name, op,
                                 inputs, n_in, outputs, n_out);
    if (idx < 0) {
        fprintf(stderr,
                "ONNX: graph node limit of %d reached while adding '%s' "
                "— raise FE_MAX_NODES to load this model\n",
                FE_MAX_NODES, node_name[0] ? node_name : op_str);
        return FE_ERR_NOMEM;
    }

    FeNode *node = &g->nodes[idx];
    for (int i = 0; i < n_in; i++)
        s_input_pos[idx][i] = (unsigned char)in_pos[i];

    /* Per-op default attribute values (attrs union is zeroed at init). */
    switch (op) {
        case FE_OP_CONV2D:
            node->attrs.conv2d.stride_h = 1;
            node->attrs.conv2d.stride_w = 1;
            node->attrs.conv2d.dil_h = 1;
            node->attrs.conv2d.dil_w = 1;
            node->attrs.conv2d.groups = 1;
            break;
        case FE_OP_CONCAT:
            node->attrs.concat.axis = 1;
            break;
        case FE_OP_RESIZE:
            node->attrs.resize.mode  = FE_RESIZE_NEAREST;      /* ONNX defaults */
            node->attrs.resize.coord = FE_RESIZE_HALF_PIXEL;
            break;
        case FE_OP_CLIP:
            node->attrs.clip.min = -FLT_MAX;
            node->attrs.clip.max =  FLT_MAX;
            break;
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            node->attrs.pool.sh = 1;
            node->attrs.pool.sw = 1;
            break;
        case FE_OP_LEAKY_RELU:
            node->attrs.leaky_relu.negative_slope = 0.01f;
            break;
        case FE_OP_ELU:
            node->attrs.elu.alpha = 1.0f;
            break;
        case FE_OP_SOFTMAX:
            /* ONNX opset-13 default. -1 also encodes "no axis attribute",
             * which is exactly what the engine does (last axis). */
            node->attrs.softmax.axis = -1;
            break;
        case FE_OP_BATCHNORM:
            node->attrs.batchnorm.eps = 1e-5f;
            break;
        case FE_OP_LAYERNORM:
            node->attrs.layernorm.eps = 1e-5f;
            break;
        case FE_OP_GROUPNORM:
            node->attrs.groupnorm.eps = 1e-5f;
            break;
        default: break;
    }

    s_attr_unsupported = 0;
    for (int i = 0; i < n_attrs; i++)
        apply_attr(node, op, &attrs[i]);
    if (s_attr_unsupported) return FE_ERR_SHAPE;

    /* ReduceMean is only the global average pool: over H and W, keeping them. */
    if (strcmp(op_str, "ReduceMean") == 0) {
        int ok = 0;
        for (int i = 0; i < n_attrs; i++) {
            if (strcmp(attrs[i].name, "axes") == 0 && attrs[i].n_ints == 2 &&
                ((attrs[i].ints[0] == 2 && attrs[i].ints[1] == 3) ||
                 (attrs[i].ints[0] == -2 && attrs[i].ints[1] == -1)))
                ok = 1;
        }
        for (int i = 0; i < n_attrs; i++)
            if (strcmp(attrs[i].name, "keepdims") == 0 && attrs[i].has_i &&
                attrs[i].int_val != 1)
                ok = 0;
        if (!ok || n_in != 1) {
            fprintf(stderr, "ONNX: ReduceMean is supported only over axes [2, 3] "
                    "with keepdims=1 (a global average pool)\n");
            return FE_ERR_SHAPE;
        }
    }
    if (op == FE_OP_RESIZE && node->attrs.resize.mode == FE_RESIZE_NEAREST &&
        node->attrs.resize.coord != FE_RESIZE_ASYMMETRIC) {
        fprintf(stderr, "ONNX: nearest Resize is supported only with "
                "coordinate_transformation_mode=asymmetric\n");
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_PRELU && n_in != 2) {
        fprintf(stderr, "ONNX: PRelu requires a slope input\n");
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_CONV2D &&
        (node->attrs.conv2d.dil_h < 1 || node->attrs.conv2d.dil_w < 1 ||
         node->attrs.conv2d.groups < 1 || node->attrs.conv2d.stride_h < 1 ||
         node->attrs.conv2d.stride_w < 1 || node->attrs.conv2d.pad_h < 0 ||
         node->attrs.conv2d.pad_w < 0)) {
        fprintf(stderr, "ONNX: Conv has an out-of-range stride, pad, dilation "
                "or group\n");
        return FE_ERR_SHAPE;
    }

    /* Required attributes and inputs must be present — fail loudly. */
    switch (op) {
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            if (node->attrs.pool.kh == 0 || node->attrs.pool.kw == 0) {
                fprintf(stderr, "ONNX: %s missing required 'kernel_shape'\n",
                        op_str);
                return FE_ERR_SHAPE;
            }
            break;
        case FE_OP_GROUPNORM:
            if (node->attrs.groupnorm.groups <= 0) {
                fprintf(stderr, "ONNX: GroupNormalization missing required "
                        "'num_groups'\n");
                return FE_ERR_SHAPE;
            }
            break;
        default: break;
    }

    /* GEMM/LINEAR and the norm ops have optional extra ONNX inputs
     * (bias / beta) that our kernels require — refuse without them. */
    if (op == FE_OP_LINEAR && n_in < 3) {
        fprintf(stderr, "ONNX: Gemm requires a bias input (C)\n");
        return FE_ERR_SHAPE;
    }
    if ((op == FE_OP_LAYERNORM || op == FE_OP_GROUPNORM) && n_in < 3) {
        fprintf(stderr, "ONNX: %s requires a bias (B) input\n", op_str);
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_BATCHNORM && n_in != 5) {
        fprintf(stderr, "ONNX: BatchNormalization requires 5 inputs\n");
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_MULTIHEAD_ATTN && n_in != 5) {
        fprintf(stderr, "ONNX: MultiHeadAttention requires 5 inputs\n");
        return FE_ERR_SHAPE;
    }

    return FE_OK;
}

/* ------------------------------------------------------------------ */
/* Parse ValueInfoProto (graph inputs / outputs / value_info)          */
/* ------------------------------------------------------------------ */

/*
 * GraphProto fields 11 (input), 12 (output) and 13 (value_info) are all
 * repeated ValueInfoProto describing a tensor's name and shape:
 *
 *   ValueInfoProto: name = 1, type = 2 (TypeProto)
 *   TypeProto:      tensor_type = 1 (TypeProto.Tensor)
 *   Tensor:         elem_type = 1, shape = 2 (TensorShapeProto)
 *   TensorShape:    dim = 1 (repeated Dimension)
 *   Dimension:      dim_value = 1 (int64), dim_param = 2 (string)
 *
 * Shapes are filled into the tensor registry. A dim_param (dynamic
 * dimension) parses as 0, preserving the graph's "shape unknown, fill
 * later" contract used by the runtime's infer_shapes.
 */
static FeStatus parse_value_info(FePbReader *r, FeGraph *g) {
    char name[FE_NAME_LEN] = {0};
    int  dims[FERRITE_MAX_DIMS];
    int  ndim = 0;
    int  have_shape = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: pb_string(r, name, FE_NAME_LEN); break;
            case 2: {
                /* TypeProto */
                const unsigned char *td; size_t tl;
                if (!fe_pb_bytes(r, &td, &tl)) break;
                FePbReader tr;
                fe_pb_init(&tr, td, tl);
                while (!fe_pb_done(&tr)) {
                    int tf, twt;
                    if (!fe_pb_tag(&tr, &tf, &twt)) break;
                    if (tf == 1) {
                        /* tensor_type */
                        const unsigned char *ts; size_t sl;
                        if (!fe_pb_bytes(&tr, &ts, &sl)) break;
                        FePbReader sr;
                        fe_pb_init(&sr, ts, sl);
                        while (!fe_pb_done(&sr)) {
                            int sf, swt;
                            if (!fe_pb_tag(&sr, &sf, &swt)) break;
                            if (sf == 2) {
                                /* shape */
                                const unsigned char *sh; size_t hl;
                                if (!fe_pb_bytes(&sr, &sh, &hl)) break;
                                FePbReader hr;
                                fe_pb_init(&hr, sh, hl);
                                while (!fe_pb_done(&hr)) {
                                    int hf, hwt;
                                    if (!fe_pb_tag(&hr, &hf, &hwt)) break;
                                    if (hf == 1) {
                                        /* dim */
                                        const unsigned char *dd; size_t dl;
                                        if (!fe_pb_bytes(&hr, &dd, &dl)) break;
                                        FePbReader dr;
                                        fe_pb_init(&dr, dd, dl);
                                        int dim_val = 0;
                                        while (!fe_pb_done(&dr)) {
                                            int df, dwt;
                                            if (!fe_pb_tag(&dr, &df, &dwt)) break;
                                            if (df == 1)
                                                dim_val = (int)fe_pb_varint(&dr);
                                            else
                                                fe_pb_skip(&dr, dwt);
                                        }
                                        if (ndim >= FERRITE_MAX_DIMS) {
                                            fprintf(stderr,
                                                "ONNX: tensor '%s' declares more "
                                                "than %d dimensions — raise "
                                                "FERRITE_MAX_DIMS to load this "
                                                "model\n",
                                                name[0] ? name : "(unnamed)",
                                                FERRITE_MAX_DIMS);
                                            return FE_ERR_SHAPE;
                                        }
                                        dims[ndim++] = dim_val;
                                    } else {
                                        fe_pb_skip(&hr, hwt);
                                    }
                                }
                                have_shape = 1;
                            } else {
                                fe_pb_skip(&sr, swt);
                            }
                        }
                    } else {
                        fe_pb_skip(&tr, twt);
                    }
                }
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }

    /* Fill the graph entry for this name. */
    if (name[0] != '\0') {
        int tidx = find_or_add_tensor(g, name);
        if (tidx < 0) return FE_ERR_NOMEM;
        FeTensorEntry *e = &g->tensors[tidx];
        if (have_shape && ndim > 0) {
            e->ndim = ndim;
            memcpy(e->shape, dims, ndim * sizeof(int));
        }
    }

    return FE_OK;
}

static FeStatus parse_graph(FePbReader *r, FeGraph *g,
                              FeArena *weight_arena) {
    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        const unsigned char *sub_data;
        size_t               sub_len;

        switch (field) {
            case 1: { /* node */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                FePbReader sub;
                fe_pb_init(&sub, sub_data, sub_len);
                FeStatus s = parse_node(&sub, g);
                if (s != FE_OK) return s;
                break;
            }
            case 5: { /* initializer — weight tensor */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                FePbReader sub;
                fe_pb_init(&sub, sub_data, sub_len);
                FeStatus s = parse_initializer(&sub, g, weight_arena);
                if (s != FE_OK) return s;
                break;
            }
            case 11: /* input — ValueInfoProto */
            case 12: /* output — ValueInfoProto */
            case 13: /* value_info — ValueInfoProto */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                {
                    FePbReader sub;
                    fe_pb_init(&sub, sub_data, sub_len);
                    FeStatus s = parse_value_info(&sub, g);
                    if (s != FE_OK) return s;
                }
                break;
            default:
                fe_pb_skip(r, wtype);
                break;
        }
    }
    return FE_OK;
}

/* The constant tensor wired to ONNX input position `pos` of node `idx`, or NULL
 * when that input was omitted. Sets *bad when it is present but not constant. */
static const FeTensor *constant_input(const FeGraph *g, int idx, int pos, int *bad) {
    const FeNode *n = &g->nodes[idx];
    for (int k = 0; k < n->n_inputs; k++) {
        if (s_input_pos[idx][k] != pos) continue;
        const FeTensorEntry *e = &g->tensors[n->inputs[k]];
        if (!e->is_weight || !e->tensor) { *bad = 1; return NULL; }
        return e->tensor;
    }
    return NULL;
}

/*
 * Resize, Pad and Clip take their sizes, pads and bounds as inputs. The kernels
 * need them before any data flows (they fix output shapes), so they must be
 * constants; this copies them into the node's attributes and leaves the node
 * with its one data input. Runs once every initializer has been read.
 */
static FeStatus resolve_constant_inputs(FeGraph *g) {
    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *n = &g->nodes[i];
        int bad = 0;
        if (n->op == FE_OP_RESIZE) {
            const FeTensor *scales = constant_input(g, i, 2, &bad);
            const FeTensor *sizes  = constant_input(g, i, 3, &bad);
            if (!bad && sizes && fe_tensor_numel(sizes) == 4) {
                const float *v = (const float *)sizes->data;
                n->attrs.resize.out_h = (int)v[2];
                n->attrs.resize.out_w = (int)v[3];
                if (n->attrs.resize.out_h < 1 || n->attrs.resize.out_w < 1) bad = 1;
            } else if (!bad && scales && fe_tensor_numel(scales) == 4) {
                const float *v = (const float *)scales->data;
                if (v[0] != 1.0f || v[1] != 1.0f || !(v[2] > 0.0f) || !(v[3] > 0.0f))
                    bad = 1;
                n->attrs.resize.scale_h = v[2];
                n->attrs.resize.scale_w = v[3];
            } else {
                bad = 1;
            }
            if (bad) {
                fprintf(stderr, "ONNX: Resize '%s' needs constant 4-element "
                        "sizes or scales that leave batch and channels alone\n",
                        n->name);
                return FE_ERR_SHAPE;
            }
            n->n_inputs = 1;
        } else if (n->op == FE_OP_PAD) {
            const FeTensor *pads  = constant_input(g, i, 1, &bad);
            const FeTensor *value = constant_input(g, i, 2, &bad);
            if (!bad && constant_input(g, i, 3, &bad)) bad = 1;   /* axes input */
            if (!bad && pads) {
                const float *v = (const float *)pads->data;
                if (fe_tensor_numel(pads) != 8 || v[0] != 0 || v[1] != 0 ||
                    v[4] != 0 || v[5] != 0) bad = 1;
                else {
                    n->attrs.pad.top    = (int)v[2];
                    n->attrs.pad.left   = (int)v[3];
                    n->attrs.pad.bottom = (int)v[6];
                    n->attrs.pad.right  = (int)v[7];
                }
            }
            if (!bad && value) n->attrs.pad.value = ((const float *)value->data)[0];
            if (n->attrs.pad.top < 0 || n->attrs.pad.left < 0 ||
                n->attrs.pad.bottom < 0 || n->attrs.pad.right < 0) bad = 1;
            if (bad) {
                fprintf(stderr, "ONNX: Pad '%s' needs constant, non-negative pads "
                        "on H and W only\n", n->name);
                return FE_ERR_SHAPE;
            }
            n->n_inputs = 1;
        } else if (n->op == FE_OP_CLIP) {
            const FeTensor *lo = constant_input(g, i, 1, &bad);
            const FeTensor *hi = constant_input(g, i, 2, &bad);
            if (bad) {
                fprintf(stderr, "ONNX: Clip '%s' needs constant bounds\n", n->name);
                return FE_ERR_SHAPE;
            }
            if (lo) n->attrs.clip.min = ((const float *)lo->data)[0];
            if (hi) n->attrs.clip.max = ((const float *)hi->data)[0];
            n->n_inputs = 1;
        }
    }
    return FE_OK;
}

/*
 * Flag Softmax nodes whose axis is provably not the last one.
 *
 * The engine always reduces over the last axis, so any other axis would
 * compute a different result. Run this only after shape inference: at parse
 * time an output's rank is usually still unknown, and an axis that merely
 * looks wrong against an unresolved rank is the common case for valid models
 * (axis=1 on a rank-2 output is the last axis, not an unsupported one).
 *
 * A rank that is still unknown here is left alone rather than guessed at.
 */
static void check_softmax_axes(const FeGraph *g) {
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        if (n->op != FE_OP_SOFTMAX) continue;
        int axis = n->attrs.softmax.axis;
        if (axis == -1 || n->n_outputs < 1) continue;
        int rank = g->tensors[n->outputs[0]].ndim;
        if (rank <= 0) continue;                 /* unresolved — never guess */
        if (axis < 0) axis += rank;              /* normalise negative axes */
        if (axis != rank - 1)
            fprintf(stderr,
                    "ONNX: unsupported Softmax axis=%d on a rank-%d output "
                    "(only the last axis is supported)\n",
                    n->attrs.softmax.axis, rank);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

FeStatus fe_onnx_load(FeGraph *graph, FeArena *weight_arena,
                       const char *path) {
    /* Read entire file into memory */
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ONNX: cannot open '%s'\n", path);
        return FE_ERR_NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    unsigned char *buf = malloc((size_t)fsize);
    if (!buf) { fclose(f); return FE_ERR_NOMEM; }

    if ((long)fread(buf, 1, (size_t)fsize, f) != fsize) {
        free(buf); fclose(f); return FE_ERR_NULL;
    }
    fclose(f);

    FeStatus status = fe_onnx_load_buffer(graph, weight_arena, buf, (size_t)fsize);
    free(buf);
    if (status == FE_OK)
        fe_log_info("loaded %d nodes, %d tensors from %s",
                    graph->n_nodes, graph->n_tensors, path);
    return status;
}

FeStatus fe_onnx_load_buffer(FeGraph *graph, FeArena *weight_arena,
                             const void *data, size_t len) {
    if (!graph || !weight_arena || !data) return FE_ERR_NULL;
    const unsigned char *buf = (const unsigned char *)data;
    size_t fsize = len;

    fe_graph_init(graph);

    /*
     * ModelProto field numbers:
     *   7 = graph (GraphProto)
     */
    FePbReader r;
    fe_pb_init(&r, buf, fsize);

    FeStatus status = FE_OK;
    while (!fe_pb_done(&r)) {
        int field, wtype;
        if (!fe_pb_tag(&r, &field, &wtype)) break;

        if (field == 7) {  /* graph */
            const unsigned char *sub_data;
            size_t               sub_len;
            if (!fe_pb_bytes(&r, &sub_data, &sub_len)) {
                status = FE_ERR_SHAPE;
                break;
            }
            FePbReader sub;
            fe_pb_init(&sub, sub_data, sub_len);
            status = parse_graph(&sub, graph, weight_arena);
            if (status != FE_OK) break;
        } else {
            fe_pb_skip(&r, wtype);
        }
    }

    if (status == FE_OK)
        status = resolve_constant_inputs(graph);
    if (status == FE_OK)
        status = fe_graph_topo_sort(graph);
    if (status == FE_OK)
        status = fe_graph_validate(graph);
    if (status == FE_OK)
        status = fe_optimize(graph, weight_arena);
    if (status == FE_OK)
        check_softmax_axes(graph);
    /*
     * Report success through the logger, not stdout. A library that prints
     * unconditionally corrupts any caller whose stdout carries data (run_model
     * piping a tensor, a tool emitting JSON), and core/log.h already sets the
     * policy: INFO compiles out at the default WARN level, so release builds
     * stay silent. Callers wanting the node dump call fe_graph_print.
     */
    return status;
}