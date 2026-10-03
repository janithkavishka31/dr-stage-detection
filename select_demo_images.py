# Pick clear demo images for the presentation from the HELD-OUT TEST SPLIT.
#
# 1. Reproduces the notebook's stratified 70/15/15 split (seed 42) from the
#    APTOS 2019 train.csv, so only images the model never trained on are used.
# 2. Downloads a set of candidates per grade from a public Hugging Face mirror
#    of the original Kaggle files.
# 3. Runs the exact deployment pipeline (preprocessing + ONNX + 5-view TTA).
# 4. Keeps the correctly classified, most confident images per grade and
#    zips them with a CSV describing each one.
#
# Usage: .venv/Scripts/python select_demo_images.py

import csv
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.model_selection import train_test_split

import dr_preprocess as dp

MIRROR = "https://huggingface.co/datasets/RohanAi/Aptos-blindness-detection/resolve/main"
WORK = "test_images_work"
IMG_DIR = os.path.join(WORK, "candidates")
ZIP_PATH = "DR_demo_test_images.zip"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CANDIDATES = {0: 15, 1: 30, 2: 60, 3: 29, 4: 25}   # images examined per grade
PER_GRADE = 3                                       # images kept per grade


def reproduce_test_split():
    csv_path = os.path.join(WORK, "train.csv")
    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(f"{MIRROR}/train.csv", csv_path)
    df = pd.read_csv(csv_path)
    _, temp = train_test_split(df, test_size=0.30, stratify=df["diagnosis"], random_state=42)
    _, test = train_test_split(temp, test_size=0.5, stratify=temp["diagnosis"], random_state=42)
    return test.reset_index(drop=True)


def download(id_code):
    path = os.path.join(IMG_DIR, f"{id_code}.png")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{MIRROR}/train_images/{id_code}.png", path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    test = reproduce_test_split()
    print(f"Test split: {len(test)} images {test.diagnosis.value_counts().sort_index().to_dict()}")

    cand = pd.concat([test[test.diagnosis == g].head(n) for g, n in CANDIDATES.items()])
    print(f"Downloading {len(cand)} candidate images…")
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(download, cand.id_code))

    sess = ort.InferenceSession(os.path.join("test_assets", "model.onnx"),
                                providers=["CPUExecutionProvider"])
    rows = []
    for (_, r), path in zip(cand.iterrows(), paths):
        rgb = dp.load_rgb(path)
        batch = dp.tta_batch(dp.to_model_input(dp.preprocess_fundus(rgb)))
        probs = sess.run(None, {"input": batch})[0].mean(0)
        pred = int(probs.argmax())
        rows.append({"id_code": r.id_code, "true": int(r.diagnosis), "pred": pred,
                     "confidence": float(probs[pred]), "probs": probs, "path": path,
                     "size": f"{rgb.shape[1]}x{rgb.shape[0]}"})
    res = pd.DataFrame(rows)

    print("\nCandidates per grade: correct / examined, and the best confidences")
    picks = []
    for g in range(5):
        sub = res[res.true == g]
        ok = sub[sub.pred == g].sort_values("confidence", ascending=False)
        print(f"  {g} {CLASS_NAMES[g]:<14} {len(ok):>2}/{len(sub):<3} "
              f"top: {', '.join(f'{c:.0%}' for c in ok.confidence.head(PER_GRADE))}")
        picks.append(ok.head(PER_GRADE))
    picks = pd.concat(picks)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
        lines = [["file", "aptos_id_code", "true_grade", "true_stage", "model_prediction",
                  "confidence"] + [f"p_{n}" for n in CLASS_NAMES]]
        for g in range(5):
            for k, (_, r) in enumerate(picks[picks.true == g].iterrows(), 1):
                name = f"grade{g}_{CLASS_NAMES[g].replace(' ', '')}/{g}_{CLASS_NAMES[g].replace(' ', '')}_{k}_{r.id_code}.png"
                zf.write(r.path, name)
                lines.append([name, r.id_code, g, CLASS_NAMES[g], CLASS_NAMES[r.pred],
                              f"{r.confidence:.4f}"] + [f"{p:.4f}" for p in r.probs])
        csv_path = os.path.join(WORK, "labels.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(lines)
        zf.write(csv_path, "labels.csv")
        zf.writestr("README.txt", README)
    print(f"\nWrote {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 1e6:.1f} MB, {len(picks)} images)")
    res.drop(columns=["probs"]).to_csv(os.path.join(WORK, "all_candidates.csv"), index=False)


README = """DR demo test images
===================
Source : APTOS 2019 Blindness Detection (Kaggle), original train_images files.
Split  : every image is from the HELD-OUT TEST SPLIT reproduced from the
         notebook (stratified 70/15/15, random_state=42). None of these images
         were used for training or validation.
Labels : true_grade is the ophthalmologist grade from APTOS train.csv.

Selection: for each grade, the model was run on a set of test images and the
correctly classified images with the highest confidence were kept. They are
therefore chosen to demonstrate the prototype clearly and are NOT a random or
representative sample - see the notebook's test-set metrics for real
performance (accuracy 76.7%, QWK 0.864).

labels.csv lists the true grade, the model's prediction and all five
probabilities as computed by the Python reference pipeline; the website
should show the same values (to within rounding).

Research prototype, not a diagnostic device.
"""

if __name__ == "__main__":
    main()
