# Export the trained DR model (dr_model.pt) to ONNX for in-browser inference.
#
# Produces in docs/:
#   model.onnx.part0, part1, ...  - the ONNX file split into <25 MB chunks
#   meta.json                     - classes, preprocessing params, TTA views
# and in test_assets/:
#   synthetic_fundus.png          - deterministic test image
#   reference.json                - Python preprocessing + PyTorch TTA result,
#                                   used to check the browser pipeline
#
# Usage:  .venv/Scripts/python export_onnx.py [optional_test_image]

import hashlib
import json
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

import dr_preprocess as dp

CKPT = "dr_model.pt"
OUT_DIR = "docs"
TEST_DIR = "test_assets"
ONNX_PATH = os.path.join(TEST_DIR, "model.onnx")   # full file kept out of docs/
CHUNK_BYTES = 20 * 1024 * 1024                     # 20 MiB, safely under 25 MB
OPSET = 17
REFERRAL_THRESHOLD = 2


class DRClassifier(nn.Module):
    # EfficientNet-B3 + softmax, so the ONNX graph outputs probabilities
    # directly (TTA averages softmax outputs).

    def __init__(self, num_classes=5):
        super().__init__()
        # weights=None: the ImageNet init is irrelevant, every tensor is
        # overwritten by the fine-tuned state_dict below.
        self.net = efficientnet_b3(weights=None)
        self.net.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1536, num_classes))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def make_synthetic_fundus(path, size=512, seed=0):
    # A crude but deterministic fundus-like image: orange disc on black with
    # vessels, an optic disc and a few red/yellow spots.
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size * 0.45)
    disc = r < 1
    shade = np.clip(1.0 - 0.5 * r ** 2, 0, 1)
    img[..., 0] = np.where(disc, 200 * shade + 30, 0)
    img[..., 1] = np.where(disc, 90 * shade + 15, 0)
    img[..., 2] = np.where(disc, 40 * shade + 5, 0)
    for _ in range(14):
        pts = np.cumsum(rng.normal(0, 18, (12, 2)), axis=0) + [c * 1.3, c]
        cv2.polylines(img, [pts.astype(np.int32)], False, (120, 25, 15), 3)
    cv2.circle(img, (int(c * 1.35), c), 28, (245, 210, 150), -1)
    for _ in range(20):
        p = tuple(int(v) for v in rng.uniform(c * 0.5, c * 1.5, 2))
        cv2.circle(img, p, int(rng.integers(2, 6)), (90, 10, 10), -1)
    for _ in range(8):
        p = tuple(int(v) for v in rng.uniform(c * 0.6, c * 1.4, 2))
        cv2.circle(img, p, 4, (230, 220, 90), -1)
    img[~disc] = 0
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise * disc[..., None], 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 1. Rebuild the architecture and load the trained weights (strict).
    ckpt = torch.load(CKPT, map_location="cpu")
    assert ckpt["arch"] == "efficientnet_b3", ckpt["arch"]
    assert ckpt["img_size"] == dp.INPUT_SIZE, ckpt["img_size"]
    class_names = list(ckpt["class_names"])
    model = DRClassifier(len(class_names))
    model.net.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"[1] Loaded {CKPT}: {ckpt['arch']}, classes={class_names}")

    # 2. Export to ONNX with a dynamic batch axis (the browser sends 5 TTA views).
    dummy = torch.randn(1, 3, dp.INPUT_SIZE, dp.INPUT_SIZE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        opset_version=OPSET,
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        dynamo=False,
    )
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"[2] Exported {ONNX_PATH} (opset {onnx_model.opset_import[0].version}, "
          f"{os.path.getsize(ONNX_PATH) / 1e6:.1f} MB), checker passed")

    # 3. Verify ONNX == PyTorch on a real preprocessed image (5 TTA views).
    test_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEST_DIR, "synthetic_fundus.png")
    if len(sys.argv) <= 1:
        make_synthetic_fundus(test_path)
    rgb = dp.load_rgb(test_path)
    stage1 = dp.preprocess_fundus(rgb)
    batch = dp.tta_batch(dp.to_model_input(stage1))

    with torch.no_grad():
        torch_probs = model(torch.from_numpy(batch)).numpy()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_probs = sess.run(["probs"], {"input": batch})[0]
    max_diff = float(np.abs(torch_probs - onnx_probs).max())

    avg = onnx_probs.mean(axis=0)
    grade = int(avg.argmax())
    print(f"[3] Test image: {test_path} ({rgb.shape[1]}x{rgb.shape[0]})")
    print(f"    Max |PyTorch - ONNX| over 5 TTA views x 5 classes: {max_diff:.2e}")
    print(f"    TTA-averaged prediction: grade {grade} = {class_names[grade]} "
          f"({avg[grade]:.1%})")
    for name, p in zip(class_names, avg):
        print(f"      {name:<14}{p:7.2%}")
    if max_diff > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch - stopping.")

    # 4. Split the ONNX file into chunks for GitHub Pages.
    for old in os.listdir(OUT_DIR):
        if old.startswith("model.onnx.part"):
            os.remove(os.path.join(OUT_DIR, old))
    data = open(ONNX_PATH, "rb").read()
    parts = []
    for i in range(0, len(data), CHUNK_BYTES):
        name = f"model.onnx.part{len(parts)}"
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(data[i:i + CHUNK_BYTES])
        parts.append(name)
    rejoined = b"".join(open(os.path.join(OUT_DIR, p), "rb").read() for p in parts)
    assert rejoined == data
    print(f"[4] Split into {len(parts)} chunks: "
          + ", ".join(f"{p} ({os.path.getsize(os.path.join(OUT_DIR, p)) / 1e6:.1f} MB)" for p in parts))

    # 5. meta.json for the website.
    meta = {
        "arch": ckpt["arch"],
        "class_names": class_names,
        "referral": {"threshold_grade": REFERRAL_THRESHOLD,
                     "refer": "Refer to ophthalmologist",
                     "routine": "Routine re-screening"},
        "model": {"parts": parts, "total_bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest(),
                  "input_name": "input", "output_name": "probs",
                  "output_is_softmax": True, "opset": OPSET},
        "preprocess": {
            "stage1_size": dp.STAGE1_SIZE,
            "input_size": dp.INPUT_SIZE,
            "crop_black_border": {"tol": 7, "min_pixels": 100},
            "stage1_resize": "INTER_AREA",
            "ben_graham": {"sigma_frac": 30.0, "gain": 4.0, "offset": 128},
            "clahe": {"clip": 2.0, "grid": 8, "channel": "L of LAB"},
            "bilateral": {"d": 5, "sigma_color": 40, "sigma_space": 40},
            "circle_mask": {"frac": 0.97},
            "stage2_resize": "PIL bilinear (antialiased)",
            "mean": dp.MEAN.tolist(),
            "std": dp.STD.tolist(),
        },
        "tta_views": dp.TTA_VIEWS,
        "test_metrics": {k: v for k, v in ckpt["test_metrics"].items()},
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[5] Wrote {OUT_DIR}/meta.json")

    # 6. Reference outputs for checking the JavaScript pipeline later.
    cv2.imwrite(os.path.join(TEST_DIR, "reference_stage1.png"),
                cv2.cvtColor(stage1, cv2.COLOR_RGB2BGR))
    ref = {"image": os.path.basename(test_path),
           "per_view_probs": onnx_probs.tolist(),
           "avg_probs": avg.tolist(), "grade": grade,
           "stage_name": class_names[grade]}
    with open(os.path.join(TEST_DIR, "reference.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print(f"[6] Wrote {TEST_DIR}/reference.json and reference_stage1.png")


if __name__ == "__main__":
    main()
