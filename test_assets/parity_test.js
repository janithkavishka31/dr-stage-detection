// Node side of the JS-vs-Python parity check (driven by parity_check.py).
// Runs docs/pipeline.js preprocessing on raw RGBA bytes and writes the
// Stage 1 image (380x380x3 uint8) and model input (3x300x300 float32).
//
// Usage: node parity_test.js <rgba.bin> <width> <height> <out_prefix>

const fs = require("fs");
const path = require("path");

const docs = path.join(__dirname, "..", "docs");
const DR = require(path.join(docs, "pipeline.js"));
const meta = JSON.parse(fs.readFileSync(path.join(docs, "meta.json"), "utf8"));

const [rgbaPath, w, h, outPrefix] = process.argv.slice(2);

DR.waitForCv(require(path.join(docs, "vendor", "opencv.js"))).then((cv) => {
  const rgba = new Uint8Array(fs.readFileSync(rgbaPath));
  const p = meta.preprocess;
  const stage1 = DR.preprocessFundus(cv, rgba, Number(w), Number(h), p);
  const chw = DR.toModelInput(stage1, p);
  fs.writeFileSync(outPrefix + "_stage1.bin", Buffer.from(stage1.buffer, stage1.byteOffset, stage1.byteLength));
  fs.writeFileSync(outPrefix + "_chw.bin", Buffer.from(chw.buffer));
  // Also emit the TTA batch so the Python side can check the view ordering.
  const batch = DR.ttaBatch(chw, p.input_size, meta.tta_views);
  fs.writeFileSync(outPrefix + "_tta.bin", Buffer.from(batch.buffer));
  console.log("node: ok");
});
