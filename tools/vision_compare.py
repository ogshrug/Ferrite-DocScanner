#!/usr/bin/env python3
"""Runs an image model in Ferrite and in ONNX Runtime on the same input, and compares.

    python tools/vision_compare.py --run-vision build-rel/run_vision model.onnx [model.onnx ...]

The input is a synthetic photo-like image (a bright page on a dark, textured
background) at the model's own input size, in 0..1. For each model it prints
Ferrite's and ONNX Runtime's single-threaded time and the largest absolute and
relative differences in the output, and fails if they disagree by more than
--atol. --reference points ONNX Runtime at a different copy of each model. Run tools/prepare_model.py on a model first.

Needs: numpy, onnx, onnxruntime.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def write_frt1(path: Path, a: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"FRT1" + struct.pack("<ii", 1, a.ndim))
        f.write(struct.pack(f"<{a.ndim}i", *a.shape))
        f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def read_frt1(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:4] == b"FRT1"
    _, ndim = struct.unpack_from("<ii", data, 4)
    shape = struct.unpack_from(f"<{ndim}i", data, 12)
    return np.frombuffer(data, dtype="<f4", offset=12 + 4 * ndim).reshape(shape)


def photo(shape) -> np.ndarray:
    n, c, h, w = shape
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w]
    img = 0.15 + 0.05 * np.sin(x / 9.0) * np.cos(y / 13.0) + 0.03 * rng.random((h, w))
    page = (x > 0.18 * w + 0.05 * y) & (x < 0.82 * w + 0.03 * y) & (y > 0.12 * h) & (y < 0.9 * h)
    img = np.where(page, 0.85 + 0.05 * rng.random((h, w)), img)
    lines = page & ((y // 7) % 3 == 0) & ((x // 5) % 4 != 0)
    img = np.where(lines, 0.2, img)
    out = np.stack([img * k for k in (1.0, 0.97, 0.92)][:c])
    return np.clip(out, 0, 1).astype(np.float32)[None].repeat(n, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-vision", required=True)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1, help="threads for both runtimes")
    ap.add_argument("--reference", help="directory of the models ONNX Runtime should run instead, "
                    "matched by file name (for files made with prepare_model.py --fp16-weights, "
                    "which only Ferrite can load)")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    failed = False
    for model in args.models:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        ref_model = str(Path(args.reference) / Path(model).name) if args.reference else model
        sess = ort.InferenceSession(ref_model, opts, providers=["CPUExecutionProvider"])
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[0].shape]
        x = photo(shape)
        name = sess.get_inputs()[0].name
        ref = sess.run(None, {name: x})[0]
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            sess.run(None, {name: x})
            times.append((time.perf_counter() - t) * 1000)
        times.sort()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_frt1(tmp / "in.bin", x)
            (tmp / "dims.cfg").write_text(" ".join(map(str, ref.shape)))
            proc = subprocess.run(
                [args.run_vision, model, str(tmp / "in.bin"), str(tmp / "out.bin"),
                 str(tmp / "dims.cfg"), str(args.runs), str(args.threads)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"FAIL {Path(model).name}: Ferrite exited {proc.returncode}\n{proc.stderr.strip()}")
                failed = True
                continue
            got = read_frt1(tmp / "out.bin")

        diff = np.abs(got.astype(np.float64) - ref)
        scale = float(np.abs(ref).max()) or 1.0
        ok = got.shape == ref.shape and float(diff.max()) <= args.atol
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {Path(model).name}: output {list(ref.shape)}, "
              f"max abs diff {diff.max():.2e} ({diff.max() / scale:.1e} of the output range), "
              f"mean {diff.mean():.2e}")
        print(f"     Ferrite      {proc.stdout.strip()}")
        print(f"     ONNX Runtime best {times[0]:.1f} ms  median {times[len(times) // 2]:.1f} ms  ({args.threads} threads)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
