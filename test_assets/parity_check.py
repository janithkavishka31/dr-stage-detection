# Verifies that docs/pipeline.js reproduces the Python preprocessing and
# prediction. Feeds identical RGBA pixels to both, then compares:
#   - Stage 1 image (380x380 uint8)    - pixel differences
#   - Stage 2 model input (3x300x300)  - max abs difference
#   - TTA batch + model probabilities  - max abs difference, same grade
#
# Usage: .venv/Scripts/python test_assets/parity_check.py [image ...]

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dr_preprocess as dp  # noqa: E402

TMP = os.path.join(HERE, "_parity_tmp")


def bordered_variant(path):
    # Non-square copy with uneven black borders to exercise crop_black_border.
    img = dp.load_rgb(path)
    h, w = img.shape[:2]
    canvas = np.zeros((h + 90, w + 260, 3), np.uint8)
    canvas[30:30 + h, 170:170 + w] = img
    out = os.path.join(TMP, "bordered_" + os.path.basename(path))
    Image.fromarray(canvas).save(out)
    return out


def check(path, sess, class_names):
    rgb = dp.load_rgb(path)
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    stem = os.path.join(TMP, os.path.splitext(os.path.basename(path))[0])
    rgba.tofile(stem + "_rgba.bin")
    subprocess.run(["node", os.path.join(HERE, "parity_test.js"),
                    stem + "_rgba.bin", str(w), str(h), stem], check=True)

    py_s1 = dp.preprocess_fundus(rgb)
    py_chw = dp.to_model_input(py_s1)
    py_tta = dp.tta_batch(py_chw)
    js_s1 = np.fromfile(stem + "_stage1.bin", np.uint8).reshape(py_s1.shape)
    js_chw = np.fromfile(stem + "_chw.bin", np.float32).reshape(py_chw.shape)
    js_tta = np.fromfile(stem + "_tta.bin", np.float32).reshape(py_tta.shape)

    d1 = np.abs(py_s1.astype(int) - js_s1.astype(int))
    py_p = sess.run(None, {"input": py_tta})[0].mean(0)
    js_p = sess.run(None, {"input": js_tta})[0].mean(0)

    print(f"\n{os.path.basename(path)} ({w}x{h})")
    print(f"  Stage 1 (380x380x3): {int((d1 > 0).sum())} of {d1.size} values differ, max diff {int(d1.max())}")
    print(f"  Stage 2 tensor     : max |diff| = {np.abs(py_chw - js_chw).max():.2e}")
    print(f"  TTA batch          : max |diff| = {np.abs(py_tta - js_tta).max():.2e}")
    print(f"  Probabilities      : max |diff| = {np.abs(py_p - js_p).max():.2e}")
    print(f"  Python -> {class_names[py_p.argmax()]} {py_p.max():.2%} | "
          f"JS -> {class_names[js_p.argmax()]} {js_p.max():.2%}")
    return py_p.argmax() == js_p.argmax()


def main():
    import json
    os.makedirs(TMP, exist_ok=True)
    meta = json.load(open(os.path.join(HERE, "..", "docs", "meta.json")))
    sess = ort.InferenceSession(os.path.join(HERE, "model.onnx"), providers=["CPUExecutionProvider"])
    images = sys.argv[1:] or [os.path.join(HERE, "synthetic_fundus.png")]
    images += [bordered_variant(images[0])]
    ok = all([check(p, sess, meta["class_names"]) for p in images])
    print("\nPARITY OK" if ok else "\nPARITY FAILED: predicted grades differ")


if __name__ == "__main__":
    main()
