# Diabetic Retinopathy Stage Detection: Web Prototype

**Live demo:** https://janithkavishka31.github.io/dr-stage-detection/

A static website that grades retinal fundus photographs into five diabetic
retinopathy stages (No DR, Mild, Moderate, Severe, Proliferative). Everything
runs in the browser: images never leave the user's device.

> **Research prototype, not a diagnostic device.** All outputs require review by
> a qualified clinician.

## How it works

| Part | Details |
|---|---|
| Model | EfficientNet-B3 (ImageNet transfer learning), `Dropout(0.4) → Linear(1536, 5)`, trained on APTOS 2019 |
| Test-set results | accuracy 76.7%, quadratic weighted κ 0.864 (held-out 15% split) |
| Preprocessing | crop black border → resize 380 (INTER_AREA) → Ben Graham illumination correction → CLAHE on LAB L-channel → bilateral denoise → circular mask → resize 300 → ImageNet normalisation |
| Test-time augmentation | identity, horizontal flip, vertical flip, rotate 90°, rotate 270°; softmax averaged |
| In-browser runtime | [ONNX Runtime Web](https://onnxruntime.ai/) (WebAssembly) + [OpenCV.js 4.13](https://docs.opencv.org/4.13.0/) |
| Referral rule | grade ≥ 2 → "Refer to ophthalmologist", otherwise "Routine re-screening" |

The ONNX model is split into chunks under 25 MB (`docs/model.onnx.part*`) and
reassembled in the browser, with a SHA-256 check.

## Repository layout

```
docs/                      the website (served by GitHub Pages)
  index.html, style.css
  pipeline.js              preprocessing, TTA and inference in the browser
  meta.json                class names, preprocessing parameters, model checksum
  model.onnx.part0..2      ONNX model chunks
  vendor/opencv.js         OpenCV.js 4.13.0
export_onnx.py             PyTorch checkpoint -> ONNX (opset 17), chunks, meta.json
dr_preprocess.py           reference Python implementation of the preprocessing
repack_model.py            rebuilds dr_model.pt from an extracted checkpoint folder
select_demo_images.py      picks demo images from the held-out test split
test_assets/parity_check.py, parity_test.js   JS-vs-Python verification
```

## Verification

- ONNX vs PyTorch: max probability difference 4e-6 over all 5 TTA views.
- JavaScript vs Python preprocessing (`test_assets/parity_check.py`): the
  Pillow resize port is bit-exact. The full Stage 1 image differs in 3 of 433,200
  values by ±1 (OpenCV WebAssembly vs native rounding). Final probabilities agree
  to 5e-6.

## Run locally

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000
```

To regenerate the model files (requires `dr_model.pt`, not included):

```bash
pip install torch torchvision onnx onnxruntime onnxscript opencv-python pillow numpy
python export_onnx.py
```
