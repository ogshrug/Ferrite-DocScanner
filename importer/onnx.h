// importer/onnx.h
#ifndef FERRITE_ONNX_H
#define FERRITE_ONNX_H

#include "types.h"
#include "graph.h"
#include "allocator.h"
#include <stddef.h>

/*
 * Load an ONNX model from a file.
 *
 * Parses the protobuf binary, builds an FeGraph, and loads all
 * weight tensors into the provided weight arena.
 *
 * graph:        caller-provided, will be initialised by this function
 * weight_arena: weights are allocated here; must outlive the graph
 * path:         path to the .onnx file
 *
 * Returns FE_OK on success.
 * On failure, graph state is undefined — do not use it.
 */
FeStatus fe_onnx_load(FeGraph *graph, FeArena *weight_arena,
                       const char *path);

/*
 * The same, from a model already in memory (an Android asset, an embedded array).
 * Weights are copied into weight_arena, so data may be freed once this returns.
 */
FeStatus fe_onnx_load_buffer(FeGraph *graph, FeArena *weight_arena,
                             const void *data, size_t len);

/*
 * Internal protobuf reader — exposed for testing.
 *
 * A cursor into a byte buffer. All parse functions advance
 * the cursor and return parsed values.
 */
typedef struct {
    const unsigned char *data;
    size_t               pos;
    size_t               len;
} FePbReader;

void     fe_pb_init  (FePbReader *r, const unsigned char *data, size_t len);
int      fe_pb_done  (const FePbReader *r);

/* Read a varint. Returns 0 on buffer exhaustion (or a 10-byte runaway). */
uint64_t fe_pb_varint(FePbReader *r);

/* Read a raw little-endian fixed32 (wire type 5). Returns 0 if truncated. */
uint32_t fe_pb_fixed32(FePbReader *r);

/* Read tag: sets *field_number and *wire_type. Returns 0 on exhaustion. */
int      fe_pb_tag   (FePbReader *r, int *field_number, int *wire_type);

/* Skip a field whose wire type is known. */
void     fe_pb_skip  (FePbReader *r, int wire_type);

/* Read a length-delimited field into *out_data, *out_len. */
int      fe_pb_bytes (FePbReader *r,
                       const unsigned char **out_data, size_t *out_len);

#endif // FERRITE_ONNX_H