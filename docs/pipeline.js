// Diabetic Retinopathy stage detection - in-browser pipeline.
//
// Mirrors predict_fundus() from the training notebook (reference copy:
// dr_preprocess.py). Image processing uses OpenCV.js 4.13.0, the same OpenCV
// version and C++ code as the Python pipeline; inference uses ONNX Runtime Web.
//
// The preprocessing functions are environment-agnostic so they can be
// verified against Python in Node (see test_assets/parity_test.js).

(function (root) {
  "use strict";

  const DR = {};

  // ---------------------------------------------------------------------
  // OpenCV.js readiness. The Emscripten module has its own .then(), which
  // loops forever if awaited directly, so strip it once resolved.
  // ---------------------------------------------------------------------
  DR.waitForCv = function (mod) {
    return new Promise((resolve) => {
      if (mod.Mat) return resolve(mod);
      mod.then((cv) => { delete cv.then; resolve(cv); });
    });
  };

  // ---------------------------------------------------------------------
  // Stage 1: fundus enhancement. RGBA bytes in, 380x380 RGB bytes out.
  // ---------------------------------------------------------------------
  DR.preprocessFundus = function (cv, rgba, width, height, p) {
    const S = p.stage1_size;
    const mats = [];
    const keep = (m) => { mats.push(m); return m; };
    try {
      const src = keep(cv.matFromArray(height, width, cv.CV_8UC4, rgba));
      const rgb = keep(new cv.Mat());
      cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);

      // 1. crop_black_border: bounding box of pixels with gray > tol.
      const gray = keep(new cv.Mat());
      const mask = keep(new cv.Mat());
      cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
      cv.threshold(gray, mask, p.crop_black_border.tol, 255, cv.THRESH_BINARY);
      let cropped = rgb;
      if (cv.countNonZero(mask) >= p.crop_black_border.min_pixels) {
        cropped = keep(rgb.roi(cv.boundingRect(mask)));
      }

      // 2. Resize to 380x380 with INTER_AREA.
      const img = keep(new cv.Mat());
      cv.resize(cropped, img, new cv.Size(S, S), 0, 0, cv.INTER_AREA);

      // 3. Ben Graham: 4*img - 4*GaussianBlur(img) + 128.
      const sigma = Math.max(S / p.ben_graham.sigma_frac, 1.0);
      const blurred = keep(new cv.Mat());
      const bg = keep(new cv.Mat());
      cv.GaussianBlur(img, blurred, new cv.Size(0, 0), sigma, 0, cv.BORDER_DEFAULT);
      cv.addWeighted(img, p.ben_graham.gain, blurred, -p.ben_graham.gain,
                     p.ben_graham.offset, bg);

      // 4. CLAHE on the L channel of LAB.
      const lab = keep(new cv.Mat());
      cv.cvtColor(bg, lab, cv.COLOR_RGB2Lab);
      const planes = new cv.MatVector();
      const clahe = new cv.CLAHE(p.clahe.clip, new cv.Size(p.clahe.grid, p.clahe.grid));
      const lEq = keep(new cv.Mat());
      const enhanced = keep(new cv.Mat());
      try {
        cv.split(lab, planes);
        const l = planes.get(0);
        clahe.apply(l, lEq);
        l.delete();
        planes.set(0, lEq);
        cv.merge(planes, lab);
      } finally {
        planes.delete();
        clahe.delete();
      }
      cv.cvtColor(lab, enhanced, cv.COLOR_Lab2RGB);

      // 5. Bilateral denoise.
      const den = keep(new cv.Mat());
      cv.bilateralFilter(enhanced, den, p.bilateral.d, p.bilateral.sigma_color,
                         p.bilateral.sigma_space, cv.BORDER_DEFAULT);

      // 6. Circular mask: zero outside radius = int(min(h,w)/2 * frac).
      const circle = keep(cv.Mat.zeros(S, S, cv.CV_8UC1));
      cv.circle(circle, new cv.Point(Math.floor(S / 2), Math.floor(S / 2)),
                Math.trunc((S / 2) * p.circle_mask.frac), new cv.Scalar(255), -1);
      const out = keep(new cv.Mat());
      cv.bitwise_and(den, den, out, circle);

      return new Uint8Array(out.data);   // copy out of the wasm heap
    } finally {
      mats.forEach((m) => m.delete());
    }
  };

  // ---------------------------------------------------------------------
  // Stage 2a: Pillow's Image.resize(BILINEAR) - what torchvision's
  // transforms.Resize does on a PIL image. Bit-exact port of Pillow's
  // Resample.c (antialiased triangle filter, 22-bit fixed point,
  // horizontal pass then vertical pass).
  // ---------------------------------------------------------------------
  const PRECISION_BITS = 32 - 8 - 2;
  const ONE = 1 << PRECISION_BITS;

  function pilCoeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(scale, 1.0);
    const support = 1.0 * filterscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const kk = new Int32Array(outSize * ksize);
    const w = new Float64Array(ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const ss = 1.0 / filterscale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0.0;
      for (let x = 0; x < xmax; x++) {
        const t = Math.abs((x + xmin - center + 0.5) * ss);
        w[x] = t < 1.0 ? 1.0 - t : 0.0;
        ww += w[x];
      }
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0.0 ? w[x] / ww : w[x];
        kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { bounds, kk, ksize };
  }

  const clip8 = (v) => {
    const s = Math.floor(v / ONE);
    return s < 0 ? 0 : s > 255 ? 255 : s;
  };

  DR.pilResizeBilinear = function (src, sw, sh, ch, dw, dh) {
    const hc = pilCoeffs(sw, dw);
    const tmp = new Uint8Array(dw * sh * ch);
    for (let y = 0; y < sh; y++) {
      for (let xx = 0; xx < dw; xx++) {
        const xmin = hc.bounds[xx * 2], xmax = hc.bounds[xx * 2 + 1];
        for (let c = 0; c < ch; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let x = 0; x < xmax; x++) {
            ss += src[(y * sw + x + xmin) * ch + c] * hc.kk[xx * hc.ksize + x];
          }
          tmp[(y * dw + xx) * ch + c] = clip8(ss);
        }
      }
    }
    const vc = pilCoeffs(sh, dh);
    const out = new Uint8Array(dw * dh * ch);
    for (let yy = 0; yy < dh; yy++) {
      const ymin = vc.bounds[yy * 2], ymax = vc.bounds[yy * 2 + 1];
      for (let xx = 0; xx < dw; xx++) {
        for (let c = 0; c < ch; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let y = 0; y < ymax; y++) {
            ss += tmp[((y + ymin) * dw + xx) * ch + c] * vc.kk[yy * vc.ksize + y];
          }
          out[(yy * dw + xx) * ch + c] = clip8(ss);
        }
      }
    }
    return out;
  };

  // ---------------------------------------------------------------------
  // Stage 2b: ToTensor + Normalize -> float32 CHW. Math.fround after each
  // operation reproduces numpy's float32 arithmetic exactly.
  // ---------------------------------------------------------------------
  DR.toModelInput = function (rgb380, p) {
    const S1 = p.stage1_size, S = p.input_size;
    const small = DR.pilResizeBilinear(rgb380, S1, S1, 3, S, S);
    const f = Math.fround;
    const mean = p.mean.map(f), std = p.std.map(f);
    const chw = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) {
      for (let c = 0; c < 3; c++) {
        chw[c * S * S + i] = f(f(f(small[i * 3 + c] / 255) - mean[c]) / std[c]);
      }
    }
    return chw;
  };

  // ---------------------------------------------------------------------
  // Stage 3: TTA views of a square CHW tensor (same as numpy in
  // dr_preprocess.tta_batch: flips and np.rot90 over the H,W axes).
  // ---------------------------------------------------------------------
  const VIEW_INDEX = {
    identity: (y, x, N) => y * N + x,
    hflip:    (y, x, N) => y * N + (N - 1 - x),
    vflip:    (y, x, N) => (N - 1 - y) * N + x,
    rot90:    (y, x, N) => x * N + (N - 1 - y),
    rot270:   (y, x, N) => (N - 1 - x) * N + y,
  };

  DR.ttaView = function (chw, N, view) {
    const idx = VIEW_INDEX[view];
    if (!idx) throw new Error("Unknown TTA view: " + view);
    const out = new Float32Array(chw.length);
    const plane = N * N;
    for (let c = 0; c < 3; c++) {
      const o = c * plane;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) out[o + y * N + x] = chw[o + idx(y, x, N)];
      }
    }
    return out;
  };

  DR.ttaBatch = function (chw, N, views) {
    const batch = new Float32Array(views.length * chw.length);
    views.forEach((v, i) => batch.set(DR.ttaView(chw, N, v), i * chw.length));
    return batch;
  };

  // Runs every TTA view through the model and averages the softmax outputs.
  DR.predict = async function (session, ort, chw, meta, onView) {
    const S = meta.preprocess.input_size;
    const views = meta.tta_views;
    const k = meta.class_names.length;
    const avg = new Float64Array(k);
    for (let i = 0; i < views.length; i++) {
      if (onView) onView(i, views.length, views[i]);
      const x = new ort.Tensor("float32", DR.ttaView(chw, S, views[i]), [1, 3, S, S]);
      const res = await session.run({ [meta.model.input_name]: x });
      const probs = res[meta.model.output_name].data;
      for (let c = 0; c < k; c++) avg[c] += probs[c] / views.length;
    }
    let grade = 0;
    for (let c = 1; c < k; c++) if (avg[c] > avg[grade]) grade = c;
    return { probs: Array.from(avg), grade };
  };

  if (typeof module === "object" && module.exports) {
    module.exports = DR;
    return;
  }
  root.DRPipeline = DR;

  // =====================================================================
  // Browser app
  // =====================================================================
  const ORT_VERSION = "1.30.0";
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const SEVERITY_CLASS = ["sev-0", "sev-1", "sev-2", "sev-3", "sev-4"];
  const STAGE_INFO = [
    "No visible signs of diabetic retinopathy.",
    "Microaneurysms only.",
    "More than microaneurysms but less than severe NPDR.",
    "Extensive haemorrhages, venous beading or IRMA (severe NPDR).",
    "Neovascularisation and/or vitreous/pre-retinal haemorrhage.",
  ];

  const $ = (id) => document.getElementById(id);
  const state = { cv: null, session: null, meta: null, busy: false };

  function setStatus(text, kind) {
    $("status-text").textContent = text;
    $("status").dataset.kind = kind || "loading";
  }

  function setProgress(frac) {
    const bar = $("load-progress");
    bar.hidden = frac == null;
    if (frac != null) bar.firstElementChild.style.width = (frac * 100).toFixed(1) + "%";
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Failed to load " + src));
      document.head.appendChild(s);
    });
  }

  async function fetchModel(meta) {
    // Download the chunks, report progress, and join them into one buffer.
    const total = meta.model.total_bytes;
    const buf = new Uint8Array(total);
    let offset = 0;
    for (const part of meta.model.parts) {
      const resp = await fetch(part);
      if (!resp.ok) throw new Error(`Failed to download ${part}: HTTP ${resp.status}`);
      const reader = resp.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.length > total) throw new Error("Model chunks larger than expected");
        buf.set(value, offset);
        offset += value.length;
        setProgress(offset / total);
        setStatus(`Downloading model… ${(offset / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
      }
    }
    if (offset !== total) throw new Error(`Model size mismatch: got ${offset} bytes, expected ${total}`);
    if (crypto.subtle) {
      setStatus("Verifying model integrity…");
      const hash = await crypto.subtle.digest("SHA-256", buf);
      const hex = Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("");
      if (hex !== meta.model.sha256) throw new Error("Model checksum mismatch - a chunk is corrupt or stale");
    }
    return buf;
  }

  async function init() {
    try {
      setStatus("Loading configuration…");
      const metaResp = await fetch("meta.json");
      if (!metaResp.ok) throw new Error("Could not load meta.json");
      state.meta = await metaResp.json();
      renderBars(null);
      renderModelInfo(state.meta);

      // OpenCV.js (~11 MB) downloads in parallel with the model chunks.
      const cvPromise = loadScript("vendor/opencv.js").then(() => DR.waitForCv(root.cv));
      cvPromise.catch(() => {});

      const modelBytes = await fetchModel(state.meta);
      setProgress(null);
      setStatus("Starting inference engine…");
      ort.env.wasm.wasmPaths = ORT_BASE;
      state.session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      setStatus("Loading image-processing engine…");
      state.cv = await cvPromise;

      setStatus("Model ready", "ready");
      $("file-input").disabled = false;
      $("dropzone").classList.remove("disabled");
    } catch (err) {
      console.error(err);
      setProgress(null);
      setStatus("Error: " + err.message, "error");
    }
  }

  async function decodeImage(file) {
    // Match PIL: no colour-profile conversion, no alpha premultiplication.
    const bmp = await createImageBitmap(file, {
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
    const canvas = document.createElement("canvas");
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  function drawRGB(canvas, rgb, S) {
    canvas.width = S;
    canvas.height = S;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      img.data[i * 4] = rgb[i * 3];
      img.data[i * 4 + 1] = rgb[i * 3 + 1];
      img.data[i * 4 + 2] = rgb[i * 3 + 2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  async function analyse(file) {
    if (state.busy || !state.session) return;
    if (!file.type.startsWith("image/")) {
      showError("Please choose an image file (JPEG or PNG).");
      return;
    }
    state.busy = true;
    $("results").classList.add("busy");
    $("error").hidden = true;
    const t0 = performance.now();
    try {
      $("analysis-status").textContent = "Decoding image…";
      const original = $("preview-original");
      original.src = URL.createObjectURL(file);
      $("previews").hidden = false;

      const data = await decodeImage(file);
      $("analysis-status").textContent = "Preprocessing (crop, Ben Graham, CLAHE, denoise)…";
      await nextFrame();
      const p = state.meta.preprocess;
      const stage1 = DR.preprocessFundus(state.cv, data.data, data.width, data.height, p);
      drawRGB($("preview-processed"), stage1, p.stage1_size);
      const chw = DR.toModelInput(stage1, p);

      const result = await DR.predict(state.session, ort, chw, state.meta, (i, n, v) => {
        $("analysis-status").textContent = `Running model: view ${i + 1} of ${n} (${v})…`;
      });
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      renderResult(result, file.name, secs);
    } catch (err) {
      console.error(err);
      showError("Analysis failed: " + err.message);
    } finally {
      state.busy = false;
      $("results").classList.remove("busy");
      $("analysis-status").textContent = "";
    }
  }

  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

  function showError(msg) {
    $("error").textContent = msg;
    $("error").hidden = false;
  }

  function renderBars(probs, grade) {
    const meta = state.meta;
    const list = $("bars");
    list.innerHTML = "";
    meta.class_names.forEach((name, i) => {
      const pct = probs ? probs[i] * 100 : 0;
      const li = document.createElement("li");
      li.className = SEVERITY_CLASS[i] + (probs && i === grade ? " top" : "");
      li.innerHTML =
        `<span class="bar-label"><span class="grade">${i}</span>${name}</span>` +
        `<span class="bar-track"><span class="bar-fill" style="width:${pct.toFixed(1)}%"></span></span>` +
        `<span class="bar-value">${probs ? pct.toFixed(1) + "%" : "–"}</span>`;
      list.appendChild(li);
    });
  }

  function renderResult(result, filename, secs) {
    const meta = state.meta;
    const { grade, probs } = result;
    const refer = grade >= meta.referral.threshold_grade;
    const card = $("verdict");
    card.className = "verdict " + SEVERITY_CLASS[grade];
    card.hidden = false;
    $("verdict-stage").textContent = meta.class_names[grade];
    $("verdict-grade").textContent = `Grade ${grade} of ${meta.class_names.length - 1}`;
    $("verdict-confidence").textContent = (probs[grade] * 100).toFixed(1) + "%";
    $("verdict-desc").textContent = STAGE_INFO[grade] || "";
    const ref = $("referral");
    ref.textContent = refer ? meta.referral.refer : meta.referral.routine;
    ref.className = "referral " + (refer ? "refer" : "routine");
    $("scale").querySelectorAll("li").forEach((li, i) => li.classList.toggle("active", i === grade));
    renderBars(probs, grade);
    $("result-meta").textContent =
      `${filename} · averaged over ${meta.tta_views.length} test-time views · ${secs}s`;
  }

  function renderModelInfo(meta) {
    const m = meta.test_metrics || {};
    const pct = (v) => (v == null ? "–" : (v * 100).toFixed(1) + "%");
    $("model-info").textContent =
      `EfficientNet-B3 (transfer learning) · ${meta.preprocess.input_size}×${meta.preprocess.input_size} input · ` +
      `test accuracy ${pct(m.accuracy)} · macro F1 ${pct(m.f1_macro)} · ` +
      `quadratic weighted κ ${m.qwk != null ? m.qwk.toFixed(3) : "–"}`;
  }

  function wireUpload() {
    const input = $("file-input");
    const zone = $("dropzone");
    input.addEventListener("change", () => {
      if (input.files[0]) analyse(input.files[0]);
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); if (!input.disabled) zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
    zone.addEventListener("drop", (e) => {
      if (!input.disabled && e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0]);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireUpload();
    init();
  });
})(this);
