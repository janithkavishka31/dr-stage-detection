# Reference (Python/OpenCV) implementation of the production preprocessing
# pipeline from the training notebook's predict_fundus(). The browser version
# in docs/pipeline.js mirrors this step for step; this file is the ground truth
# used to verify both the ONNX export and the JavaScript port.

import cv2
import numpy as np
from PIL import Image

STAGE1_SIZE = 380
INPUT_SIZE = 300
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TTA_VIEWS = ["identity", "hflip", "vflip", "rot90", "rot270"]


# ---- Stage 1: fundus enhancement (uint8 RGB in, uint8 RGB out) ----

def crop_black_border(img, tol=7):
    # Crop to the bounding box of pixels brighter than the background.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() < 100:
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def ben_graham(img, sigma_frac=30.0, gain=4.0):
    # Subtract local average colour to even out illumination.
    sigma = max(img.shape[0] / sigma_frac, 1.0)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, gain, blurred, -gain, 128)


def apply_clahe(img, clip=2.0, grid=8):
    # Local contrast enhancement on the lightness channel only.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def denoise(img):
    return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)


def apply_circle_mask(img, frac=0.97):
    # Zero everything outside a centred circle to hide the bright rim that
    # Ben Graham's filter creates at the edge of the fundus.
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * frac), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess_fundus(img, size=STAGE1_SIZE):
    img = crop_black_border(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = ben_graham(img)
    img = apply_clahe(img)
    img = denoise(img)
    return apply_circle_mask(img)


# ---- Stage 2: model input (deterministic torchvision transform path) ----

def to_model_input(img380):
    # transforms.Resize(300) on a PIL image (bilinear, antialiased), then
    # ToTensor + Normalize. Returns float32 CHW.
    small = Image.fromarray(img380).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1).copy()


# ---- Stage 3: test-time augmentation views (on a CHW array) ----

def tta_batch(chw):
    views = [
        chw,
        chw[:, :, ::-1],              # horizontal flip
        chw[:, ::-1, :],              # vertical flip
        np.rot90(chw, 1, axes=(1, 2)),  # rotate 90
        np.rot90(chw, 3, axes=(1, 2)),  # rotate 270
    ]
    return np.stack([np.ascontiguousarray(v) for v in views]).astype(np.float32)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))
