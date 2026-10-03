# Repack the browser-extracted dr_model/ folder back into a PyTorch .pt file.
#
# A .pt file (PyTorch >= 1.6) is an uncompressed zip whose entries all live
# under one top-level folder ("archive/"). This script rebuilds that zip
# without touching the extracted folder.

import os
import zipfile

SRC_DIR = "dr_model"
OUT_FILE = "dr_model.pt"
PREFIX = "archive"

if os.path.exists(OUT_FILE):
    raise SystemExit(f"{OUT_FILE} already exists - refusing to overwrite it.")

# Collect every file, relative to the extracted folder, using forward slashes.
files = []
for root, _, names in os.walk(SRC_DIR):
    for name in names:
        full = os.path.join(root, name)
        rel = os.path.relpath(full, SRC_DIR).replace(os.sep, "/")
        files.append((full, rel))

# Put data.pkl first (as torch.save does); keep the rest in a stable order.
files.sort(key=lambda f: (f[1] != "data.pkl", f[1]))

with zipfile.ZipFile(OUT_FILE, "w", compression=zipfile.ZIP_STORED) as zf:
    for full, rel in files:
        zf.write(full, arcname=f"{PREFIX}/{rel}")

print(f"Wrote {OUT_FILE}: {len(files)} files, "
      f"{os.path.getsize(OUT_FILE) / 1e6:.1f} MB")
