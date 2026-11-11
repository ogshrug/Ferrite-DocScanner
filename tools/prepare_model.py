#!/usr/bin/env python3
"""Rewrites an ONNX model into the subset Ferrite loads, without changing what it computes.

    python tools/prepare_model.py in.onnx out.onnx [--fp16-weights]

Steps:
  1. ONNX Runtime's basic graph optimisation folds constants (Cast, Constant, Shape
     arithmetic), merges BatchNormalization into Conv and removes identities.
  2. A weighted sum written as Unsqueeze + Concat(axis 0) + Einsum("i,i...->...")
     becomes Mul by each weight and Add.
  3. Tensors get short names (Ferrite keeps 63 characters), initializers are stored as
     raw float32 / int64, and value_info is dropped.

The result is checked against the original on random input before it is written.

--fp16-weights then stores as float16 every weight that float16 holds exactly (Ferrite
widens them to float32 when it loads the model), so the file shrinks and nothing about
the result changes. ONNX Runtime cannot run such a
file, since the weights no longer match their operators' types; it is for Ferrite only.
Needs: onnx, onnxruntime, numpy. None of this touches the runtime.
"""
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def optimise(src: str) -> onnx.ModelProto:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "basic.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = out
        ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
        return onnx.load(out)


def rewrite_weighted_sums(graph: onnx.GraphProto) -> int:
    inits = {i.name: i for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    drop, add, count = set(), [], 0
    for node in graph.node:
        if node.op_type != "Einsum":
            continue
        eq = helper.get_attribute_value(node.attribute[0]).decode().replace(" ", "")
        if eq != "i,i...->..." or node.input[0] not in inits:
            continue
        concat = producer.get(node.input[1])
        if concat is None or concat.op_type != "Concat":
            continue
        if helper.get_attribute_value(concat.attribute[0]) != 0:
            continue
        parts = [producer.get(i) for i in concat.input]
        if any(p is None or p.op_type != "Unsqueeze" for p in parts):
            continue
        if any(numpy_helper.to_array(inits[p.input[1]]).tolist() != [0] for p in parts):
            continue
        weights = numpy_helper.to_array(inits[node.input[0]]).astype(np.float32)
        if weights.shape != (len(parts),):
            continue

        total = None
        for k, part in enumerate(parts):
            wname = f"{node.output[0]}_w{k}"
            graph.initializer.append(numpy_helper.from_array(weights[k:k + 1], wname))
            scaled = f"{node.output[0]}_s{k}"
            add.append((node, helper.make_node("Mul", [part.input[0], wname], [scaled])))
            if total is None:
                total = scaled
            else:
                out = node.output[0] if k == len(parts) - 1 else f"{node.output[0]}_a{k}"
                add.append((node, helper.make_node("Add", [total, scaled], [out])))
                total = out
        drop.add(id(node))
        count += 1

    if count:
        nodes = []
        for n in graph.node:
            nodes.extend(new for at, new in add if at is n)
            if id(n) not in drop:
                nodes.append(n)
        del graph.node[:]
        graph.node.extend(nodes)
        prune(graph)
    return count


def prune(graph: onnx.GraphProto) -> None:
    """Removes nodes whose outputs nothing reads (the Unsqueeze and Concat left behind)."""
    while True:
        read = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
        live = [n for n in graph.node if any(o in read for o in n.output)]
        if len(live) == len(graph.node):
            return
        del graph.node[:]
        graph.node.extend(live)


def tidy(model: onnx.ModelProto) -> None:
    graph = model.graph
    used = {i for n in graph.node for i in n.input}
    kept = [i for i in graph.initializer if i.name in used]
    names = {}

    def short(name: str, prefix: str) -> str:
        if name and name not in names:
            names[name] = f"{prefix}{len(names)}"
        return names.get(name, "")

    init_names = {i.name for i in kept}
    for vi in graph.input:
        if vi.name not in init_names:
            vi.name = short(vi.name, "in")
    fresh = []
    for init in kept:
        arr = numpy_helper.to_array(init)
        if arr.dtype in (np.float16, np.float64):
            arr = arr.astype(np.float32)
        elif arr.dtype in (np.int32,):
            arr = arr.astype(np.int64)
        fresh.append(numpy_helper.from_array(arr, short(init.name, "w")))
    del graph.initializer[:]
    graph.initializer.extend(fresh)
    for k, node in enumerate(graph.node):
        node.name = f"{node.op_type}_{k}"
        for i, name in enumerate(node.input):
            node.input[i] = short(name, "t")
        for i, name in enumerate(node.output):
            node.output[i] = short(name, "t")
    for vi in graph.output:
        vi.name = names[vi.name]
    inputs = [vi for vi in graph.input if vi.name.startswith("in")]
    del graph.input[:]
    graph.input.extend(inputs)
    del graph.value_info[:]
    # Only the default domain is left in use.
    opsets = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
    del model.opset_import[:]
    model.opset_import.extend(opsets)


def check(src: str, model: onnx.ModelProto) -> float:
    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in a.get_inputs()[0].shape]
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x})[0]
    return float(np.abs(ya - yb).max())


def store_half(model: onnx.ModelProto) -> int:
    count = 0
    fresh = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 16:
            half = arr.astype(np.float16)
            # Only where nothing is lost: weights that came from a float16 model
            # round-trip exactly, while ones the optimiser rewrote (a BatchNorm
            # folded into a Conv) usually don't, and stay float32.
            if np.array_equal(half.astype(np.float32), arr):
                arr = half
                count += 1
        fresh.append(numpy_helper.from_array(arr, init.name))
    del model.graph.initializer[:]
    model.graph.initializer.extend(fresh)
    return count


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    half = "--fp16-weights" in sys.argv
    src, dst = args[0], args[1]
    model = optimise(src)
    sums = rewrite_weighted_sums(model.graph)
    tidy(model)
    onnx.checker.check_model(model)
    diff = check(src, model)
    if diff > 1e-4:
        sys.exit(f"prepared model disagrees with the original by {diff}")
    halved = store_half(model) if half else 0
    onnx.save(model, dst)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"{dst}: {len(model.graph.node)} nodes, {sums} weighted sums rewritten, "
          f"max difference from original {diff:.2e}"
          + (f", {halved} weights stored as float16" if half else ""))
    print("operators:", ", ".join(ops))


if __name__ == "__main__":
    main()
