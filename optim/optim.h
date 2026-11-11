// optim/optim.h
#ifndef FERRITE_OPTIM_H
#define FERRITE_OPTIM_H

#include "graph.h"
#include "allocator.h"

/*
 * Graph optimization entry point (Stage 12).
 *
 *   graph -> shape-infer -> simplify -> fold-constants -> fuse-ConvBN
 *         -> CSE -> dead-elim
 *
 * Runs in place on `g`; any new constant tensors (folding, fused ConvBN
 * weights) are arena-allocated into `arena` (the weight arena). Always
 * re-sorts topologically and validates before returning, so the result is
 * immediately runnable. Each pass is individually testable via its own
 * fe_*_pass function; fe_optimize just orders them.
 *
 * The constant/feed convention: an INPUT node whose output is a weight is
 * a constant source. Optimization passes rewrite folded or dead producers
 * into such INPUT nodes rather than deleting them, so consumer indices stay
 * valid; dead elimination then compacts the arrays to drop real garbage.
 */
FeStatus fe_optimize(FeGraph *g, FeArena *weight_arena);

/* Individual passes (each idempotent, each documented in its .c). */
FeStatus fe_pass_simplify(FeGraph *g);
FeStatus fe_pass_fold_constants(FeGraph *g, FeArena *arena);
FeStatus fe_pass_fuse_conv_bn(FeGraph *g, FeArena *arena);
FeStatus fe_pass_fuse_conv_relu(FeGraph *g);
FeStatus fe_pass_cse(FeGraph *g);
FeStatus fe_pass_dead_elim(FeGraph *g);

#endif // FERRITE_OPTIM_H