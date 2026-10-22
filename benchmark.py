"""benchmark.py  -  measures the TFLite models against the FP32 baseline and writes the report.  (run.py calls it.)

    python benchmark.py bench  --work results/abcd123/work --models results/abcd123/models --data put_test_pictures_here
    python benchmark.py report --result-dir results/abcd123
    (python benchmark.py child ...  is used internally: it measures one model in a fresh process)

WHAT IS MEASURED
================
  size      file size in MB.
  accuracy  needs test pictures WITH label files. The model is run on every test picture, the boxes are decoded and
            scored with the standard COCO method (pycocotools): mAP50-95 and mAP50, plus precision / recall / F1.
            "Kept" = INT8 mAP divided by FP32 mAP. If the test set has no labels, accuracy is skipped (never reported as 0).
  speed     milliseconds per picture, batch 1, 4 threads each pinned to its own physical CPU core. 20 warm-up runs and
            100 timed runs per round; the models are measured in interleaved, shuffled rounds (so slow drift of the
            computer hits all models alike); the result is the median of the rounds.
  memory    extra memory (peak RSS) the process needs for loading and running the model, median of 3 fresh processes.

HOW THE OUTPUT OF A MODEL BECOMES BOXES  (function decode)
  Each model returns N rows of [x, y, w, h, objectness, class scores...]. The boxes are fractions of the picture (0-1,
  see convert.py), so they are multiplied by the input size. Class score = objectness x class probability. Rows above
  the confidence threshold go through non-maximum suppression (NMS, per class), then the boxes are moved back from the
  letterboxed (resized + padded) picture to the original picture.
  mAP uses confidence 0.001 and IoU 0.6 like the standard YOLOv5 validation; precision/recall/F1 use confidence 0.25, IoU 0.5.
"""
import argparse
import contextlib
import csv
import gc
import html
import io
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================================================================================
# Pictures and box decoding
# =====================================================================================================================
def read_image(path):
    """Read a picture as BGR (also from non-ASCII paths); None if unreadable."""
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def letterbox(img, size, color=114):
    """Resize keeping the aspect ratio and pad with grey to size x size. Returns picture, scale, (pad_x, pad_y)."""
    import cv2
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def to_input(letterboxed_bgr):
    """Network input: float32, RGB, 0-1, NCHW (1 x 3 x S x S)."""
    return np.ascontiguousarray(letterboxed_bgr[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def decode(pred, nc, box_scale, conf, iou, r, pad, shape, multi_label=False, max_det=300):
    """pred (N, 5 + nc) -> (M, 6) detections [x1, y1, x2, y2, confidence, class] in the ORIGINAL picture's pixels.
    multi_label=True keeps every class above the threshold per box (as the standard validation does)."""
    import cv2
    p = np.ascontiguousarray(pred, dtype=np.float32).reshape(-1, 5 + nc)
    scores = p[:, 5:] * p[:, 4:5]                               # class score = class probability x objectness
    if multi_label:
        i, j = np.nonzero(scores > conf)
        sc, xywh = scores[i, j], p[i, :4] * box_scale
    else:
        j = scores.argmax(1)
        sc = scores[np.arange(len(p)), j]
        keep = sc > conf
        sc, j, xywh = sc[keep], j[keep], p[keep, :4] * box_scale
    if not sc.size:
        return np.zeros((0, 6), np.float32)
    if sc.size > 30000:                                         # keep NMS fast on very noisy outputs
        top = np.argpartition(-sc, 30000)[:30000]
        sc, j, xywh = sc[top], j[top], xywh[top]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1)    # x, y, w, h
    idx = np.array(cv2.dnn.NMSBoxesBatched(boxes.tolist(), sc.tolist(), j.tolist(), conf, iou), dtype=int).reshape(-1)
    if not idx.size:
        return np.zeros((0, 6), np.float32)
    idx = idx[np.argsort(-sc[idx])][:max_det]
    b = boxes[idx]
    out = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3], sc[idx], j[idx]], 1).astype(np.float32)
    out[:, [0, 2]] = ((out[:, [0, 2]] - pad[0]) / r).clip(0, shape[1])         # undo the padding and the resize
    out[:, [1, 3]] = ((out[:, [1, 3]] - pad[1]) / r).clip(0, shape[0])
    return out


# =====================================================================================================================
# Test set (YOLO format) and scoring
# =====================================================================================================================
class Dataset:
    """Labelled test pictures in the usual YOLO layout:
        <root>/images/<anything>/x.jpg      <root>/labels/<same sub-path>/x.txt     (each line: class x_center y_center width height, all 0-1)
    <root> can also be the path of a data.yaml (its path/val entries and class names are used)."""

    def __init__(self, source):
        source = Path(source)
        if source.is_dir() and not (source / "images").exists():           # a folder with a data.yaml inside
            ys = sorted(source.glob("*.y*ml"))
            if ys:
                source = ys[0]
        if source.suffix in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(source.read_text(encoding="utf8"))
            root = Path(cfg.get("path", source.parent))
            root = root if root.is_absolute() else (source.parent / root).resolve()
            val = cfg.get("val") or cfg.get("test") or "images"
            val = Path(val) if Path(val).is_absolute() else root / val
            self.images = list_images(val)
        else:
            self.images = list_images(source / "images") if (source / "images").exists() else list_images(source)
        if not self.images:
            raise FileNotFoundError(f"No pictures found for the test set at {source}")
        self.labels = [self._label(p) for p in self.images]
        self.n_label_files = sum(1 for p in self.images if self._label_file(p).exists())

    @staticmethod
    def _label_file(img_path):
        parts = list(img_path.parts)
        if "images" in parts:                                   # .../images/val/x.jpg -> .../labels/val/x.txt
            k = len(parts) - 1 - parts[::-1].index("images")
            return Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        return img_path.with_suffix(".txt")

    def _label(self, img_path):
        f = self._label_file(img_path)
        if not f.exists() or not f.stat().st_size:
            return np.zeros((0, 5), np.float32)
        return np.loadtxt(f, ndmin=2).astype(np.float32).reshape(-1, 5)

    def __len__(self):
        return len(self.images)

    def subset(self, n, seed=0):
        """n pictures after a fixed shuffle (the same subset in every run)."""
        if n and n < len(self):
            idx = sorted(np.random.RandomState(seed).permutation(len(self))[:n])
            self.images, self.labels = [self.images[i] for i in idx], [self.labels[i] for i in idx]
        return self


def coco_ground_truth(images, labels, shapes, nc):
    """YOLO labels -> a pycocotools ground-truth object (shapes = [(height, width)] of the original pictures)."""
    from pycocotools.coco import COCO
    ims, anns = [], []
    for i, (lab, (h, w)) in enumerate(zip(labels, shapes)):
        ims.append({"id": i, "height": int(h), "width": int(w), "file_name": Path(images[i]).name})
        for c, xc, yc, bw, bh in lab:
            x, y, ww, hh = (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1, "bbox": [float(x), float(y), float(ww), float(hh)],
                         "area": float(ww * hh), "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": ims, "annotations": anns, "categories": [{"id": k + 1, "name": str(k)} for k in range(nc)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def _iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def precision_recall(dets, gt, n_images, conf=0.25, iou=0.5):
    """Greedy matching per class by score: a detection is right if it overlaps a not yet used real box of its class by IoU >= 0.5."""
    tp = fp = n_gt = 0
    for i in range(n_images):
        g = [(a["category_id"] - 1, np.array([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]))
             for a in gt.imgToAnns.get(i, [])]
        n_gt += len(g)
        d = dets[i][dets[i][:, 4] >= conf]
        used = [False] * len(g)
        for x1, y1, x2, y2, c, k in d[np.argsort(-d[:, 4])]:
            best, bj = iou, -1
            for j, (gc, gb) in enumerate(g):
                if gc == int(k) and not used[j]:
                    v = _iou(np.array([x1, y1, x2, y2]), gb[None])[0]
                    if v >= best:
                        best, bj = v, j
            if bj >= 0:
                used[bj] = True
                tp += 1
            else:
                fp += 1
    p, r = tp / max(tp + fp, 1), tp / max(n_gt, 1)
    return {"precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-9), "n_detections_025": tp + fp}


def score_detections(dets, gt, n_images):
    """mAP numbers from COCOeval plus precision / recall / F1."""
    from pycocotools.cocoeval import COCOeval
    res = [{"image_id": i, "category_id": int(k) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(c)}
           for i, d in enumerate(dets) for x1, y1, x2, y2, c, k in d]
    out = {"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0}
    if res and gt.dataset["annotations"]:
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, gt.loadRes(res), "bbox")
            E.evaluate(); E.accumulate(); E.summarize()
        out.update({"mAP50-95": float(E.stats[0]), "mAP50": float(E.stats[1]), "mAP75": float(E.stats[2])})
    out.update(precision_recall(dets, gt, n_images))
    return out


# =====================================================================================================================
# Running a .tflite model
# =====================================================================================================================
def load_runtime():
    """LiteRT (pip: ai-edge-litert) is used if installed, else TensorFlow's own interpreter. Returns (Interpreter class, name)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "LiteRT"
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter, "TensorFlow Lite"


class Model:
    """One .tflite file. Call it with a 1 x 3 x S x S float picture (0-1); it returns the raw rows (N, 5 + nc).
    Models with integer input/output are handled with the model's own scale and zero point."""

    def __init__(self, path, threads=4, nc=80):
        Interpreter, self.runtime = load_runtime()
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.nc = nc

    def __call__(self, x_nchw):
        v = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1), dtype=np.float32)       # TFLite wants channels-last
        d = self.inp
        if d["dtype"] != np.float32:
            scale, zero = d["quantization"]
            info = np.iinfo(d["dtype"])
            v = np.clip(np.round(v / scale) + zero, info.min, info.max).astype(d["dtype"])
        self.it.set_tensor(d["index"], v)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.reshape(-1, 5 + self.nc)


# =====================================================================================================================
# Speed and memory of ONE model in a fresh process  (python benchmark.py child latency|memory ...)
# =====================================================================================================================
def child_pictures(list_file, size, n=10):
    items = []
    if list_file and Path(list_file).exists():
        for line in Path(list_file).read_text(encoding="utf8").splitlines():
            img = read_image(line.strip()) if line.strip() else None
            if img is not None:
                items.append(to_input(letterbox(img, size)[0]))
            if len(items) >= n:
                break
    if not items:                                       # no pictures given: random noise (speed does not depend on the content)
        items = [np.random.RandomState(i).rand(1, 3, size, size).astype(np.float32) for i in range(n)]
    return items


def child_latency(model, nc, size, threads, cpus, list_file, warmup=20, runs=100):
    import psutil
    p = psutil.Process()
    try:
        p.cpu_affinity([int(c) for c in cpus.split(",")])       # one thread per physical core: two threads never share a core
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass
    m = Model(model, threads, nc)
    items = child_pictures(list_file, size)
    for k in range(warmup):
        m(items[k % len(items)])
    t = []
    for k in range(runs):
        t0 = time.perf_counter()
        m(items[k % len(items)])
        t.append(time.perf_counter() - t0)
    print(json.dumps({"median": float(np.median(t) * 1e3), "p95": float(np.percentile(t, 95) * 1e3)}))


def child_memory(model, nc, size, threads):
    import psutil
    load_runtime()                                      # import the runtime first: the runtime itself is not counted
    x = np.random.RandomState(0).rand(1, 3, size, size).astype(np.float32)
    proc = psutil.Process()
    gc.collect()
    base = proc.memory_info().rss / 1e6
    peak, stop = [base], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss / 1e6)
            time.sleep(0.002)
    threading.Thread(target=sample, daemon=True).start()
    m = Model(model, threads, nc)
    loaded = proc.memory_info().rss / 1e6
    for _ in range(35):
        m(x)
    stop.set()
    print(json.dumps({"baseline_mb": base, "load_mb": loaded - base, "peak_mb": peak[0] - base}))


def run_child(*args):
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", *map(str, args)], capture_output=True, text=True, encoding="utf8", errors="replace")
    if r.returncode or not (r.stdout or "").strip():
        raise RuntimeError((r.stderr or "no output")[-400:])
    return json.loads(r.stdout.strip().splitlines()[-1])


# =====================================================================================================================
# The benchmark itself
# =====================================================================================================================
def cpu_plan(max_threads=4):
    """Number of threads and the logical CPUs to pin them to: one per physical core (skipping hyper-thread siblings when that can be told)."""
    import psutil
    phys, logical = psutil.cpu_count(logical=False) or 1, psutil.cpu_count(logical=True) or 1
    threads = max(1, min(max_threads, phys))
    step = 2 if logical >= 2 * phys else 1
    return threads, [i * step for i in range(threads)], phys, logical


def accuracy(path, nc, size, gt, shapes, preload, n_images):
    m = Model(path, 4, nc)
    dets = [decode(m(x), nc, size, 0.001, 0.6, r, pad, shp, multi_label=True) for shp, (x, r, pad) in zip(shapes, preload)]
    return score_detections(dets, gt, n_images)


def step_bench(a):
    work, models_dir = Path(a.work), Path(a.models)
    layout = json.loads((work / "layout.json").read_text())
    nc, size = layout["nc"], layout["imgsz"]
    info = json.loads((models_dir / "models.json").read_text())
    entries = [e for e in info["models"] if e.get("exists")]
    runtime = load_runtime()[1]
    warnings = [e["warning"] for e in entries if e.get("warning")]
    print(f"{len(entries)} models to measure; runtime: {runtime}", flush=True)

    # ---- test set
    ds = gt = shapes = preload = None
    n_objects = n_label_files = 0
    if a.data:
        ds = Dataset(a.data)
        if a.exclude and Path(a.exclude).exists():              # pictures that were used for calibration are not used for testing
            ex = {l.strip() for l in Path(a.exclude).read_text(encoding="utf8").splitlines() if l.strip()}
            keep = [k for k, im in enumerate(ds.images) if str(im) not in ex]
            if keep and len(keep) < len(ds.images):
                ds.images, ds.labels = [ds.images[k] for k in keep], [ds.labels[k] for k in keep]
        ds.subset(a.max_images)
        shapes, preload, good = [], [], []
        for k, p in enumerate(ds.images):
            img = read_image(p)
            if img is None:
                continue
            lb, r, pad = letterbox(img, size)
            shapes.append(img.shape[:2]); preload.append((to_input(lb), r, pad)); good.append(k)
        if len(good) != len(ds.images):
            warnings.append(f"{len(ds.images) - len(good)} test pictures could not be read and were skipped.")
            ds.images, ds.labels = [ds.images[k] for k in good], [ds.labels[k] for k in good]
        n_objects = int(sum(len(l) for l in ds.labels))
        n_label_files = sum(1 for p in ds.images if Dataset._label_file(p).exists())
        classes = sorted({int(c) for l in ds.labels for c in l[:, 0]})
        print(f"test set: {len(ds)} pictures, {n_label_files} with a label file, {n_objects} labelled objects, class numbers used: {classes[:30]}", flush=True)
        if n_objects == 0:
            warnings.append("The test pictures have NO labelled objects (no label files found, or they are empty), so accuracy was NOT measured. "
                            "Label files must be in a 'labels' folder next to 'images', with the same file names (x.jpg -> x.txt).")
            print("WARNING:", warnings[-1], flush=True)
            ds = None
        else:
            if classes and max(classes) >= nc:
                warnings.append(f"The labels use class number {max(classes)}, but the model has only {nc} classes (0 to {nc - 1}).")
                print("WARNING:", warnings[-1], flush=True)
            if n_label_files < len(ds) * 0.8:
                warnings.append(f"Only {n_label_files} of {len(ds)} test pictures have a label file; pictures without one count as 'nothing in it'.")
                print("WARNING:", warnings[-1], flush=True)
            gt = coco_ground_truth(ds.images, ds.labels, shapes, nc)

    # ---- accuracy
    res = {e["file"]: dict(e) for e in entries}
    threads, cpus, phys, logical = cpu_plan()
    if ds is not None:
        for e in entries:
            r, t0 = res[e["file"]], time.time()
            try:
                r.update(accuracy(models_dir / e["file"], nc, size, gt, shapes, preload, len(ds)))
                print(f"   accuracy {e['file']}: mAP50-95 {r['mAP50-95']:.4f}  mAP50 {r['mAP50']:.4f}  F1 {r['F1']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
            except Exception as ex:
                r["error"] = f"accuracy: {type(ex).__name__}: {ex}"
                print(f"   accuracy {e['file']}: FAILED {r['error']}", flush=True)
        fp = next((r for r in res.values() if r["design"] == "fp32"), {})
        if fp.get("mAP50") is not None and fp["mAP50"] < 0.001:
            warnings.append(f"The FP32 BASELINE itself scores about 0 (mAP50 {fp['mAP50']:.4f}; it makes {fp.get('n_detections_025', 0)} detections above confidence 0.25 on {len(ds)} pictures). "
                            "So the INT8 conversion is not the cause: check the labels (class numbers and folder layout), that the test pictures show the objects the model was trained on, "
                            "and that the model was really trained (a model trained on a few unlabelled pictures detects nothing).")
            print("WARNING:", warnings[-1], flush=True)

    # ---- speed and memory
    if not a.skip_speed:
        pics = work / "speed_pictures.txt"
        pics.write_text("\n".join(str(p) for p in (ds.images[:10] if ds is not None else [])), encoding="utf8")
        print(f"speed: {threads} threads pinned to logical CPUs {cpus} ({phys} physical, {logical} logical cores), {a.rounds} interleaved rounds", flush=True)
        rounds = {e["file"]: [] for e in entries}
        for k in range(a.rounds):
            order = [e["file"] for e in entries]
            random.Random(k).shuffle(order)
            for fn in order:
                try:
                    rounds[fn].append(run_child("latency", models_dir / fn, nc, size, threads, ",".join(map(str, cpus)), pics))
                except Exception as ex:
                    res[fn]["error"] = f"speed: {ex}"
            print(f"   round {k + 1}/{a.rounds} done", flush=True)
        for fn, rs in rounds.items():
            if rs:
                med = [x["median"] for x in rs]
                res[fn].update({"ms_median": float(np.median(med)), "ms_p95": float(np.median([x["p95"] for x in rs])),
                                "ms_spread_pct": float((max(med) - min(med)) / np.median(med) * 100)})
        print("memory (3 fresh processes per model, median)", flush=True)
        for e in entries:
            fn = e["file"]
            try:
                ms = [run_child("memory", models_dir / fn, nc, size, threads) for _ in range(3)]
                res[fn].update({k: float(np.median([m[k] for m in ms])) for k in ("baseline_mb", "load_mb", "peak_mb")})
            except Exception as ex:
                res[fn]["error"] = f"memory: {ex}"

    # ---- comparison with the FP32 baseline
    base = next((r for r in res.values() if r["design"] == "fp32"), {})
    for r in res.values():
        if base.get("size_mb") and r.get("size_mb"):
            r["compression"] = base["size_mb"] / r["size_mb"]
        if base.get("mAP50-95") and r.get("mAP50-95") is not None:
            r["retained_pct"] = r["mAP50-95"] / base["mAP50-95"] * 100
        if base.get("ms_median") and r.get("ms_median"):
            r["speedup"] = base["ms_median"] / r["ms_median"]
    out = {"name": layout["name"], "nc": nc, "imgsz": size, "runtime": runtime, "threads": threads, "cpus": cpus, "physical_cores": phys,
           "logical_cores": logical, "n_test_pictures": len(ds) if ds is not None else 0, "n_test_objects": n_objects, "n_calibration": info.get("n_calibration"),
           "rounds": a.rounds, "warnings": warnings, "results": list(res.values())}
    (models_dir.parent / "benchmark.json").write_text(json.dumps(out, indent=1))
    print("step 3 done")


# =====================================================================================================================
# The report (report.html + results.csv)
# =====================================================================================================================
ORDER = ["fp32", "m1", "m2", "m3"]
NAMES = {"fp32": "FP32 baseline", "m1": "M1 all INT8", "m2": "M2 backbone INT8", "m3": "M3 backbone + neck INT8"}
COLORS = {"fp32": "#6b7280", "m1": "#1f4e9e", "m2": "#1b7f5c", "m3": "#c0551b"}


def short(r):
    return NAMES[r["design"]] + (f", {r['granularity']}" if r["granularity"] in ("per-channel", "per-tensor") else "")


def fmt(v, f):
    return "-" if v is None else f.format(v)


def bar_chart(title, rows, unit, f):
    """rows: [(label, value, color)] -> an inline SVG with horizontal bars."""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return ""
    top = max(v for _, v, _ in rows) or 1
    h = 28 * len(rows) + 46
    out = [f'<figure><svg viewBox="0 0 760 {h}" width="100%" role="img" aria-label="{html.escape(title)}">'
           f'<text x="0" y="18" font-size="15" font-weight="700" fill="currentColor">{html.escape(title)}</text>']
    for i, (label, v, c) in enumerate(rows):
        y = 34 + i * 28
        w = max(2, v / top * 400)
        out.append(f'<text x="0" y="{y + 14}" font-size="12.5" fill="currentColor">{html.escape(label)}</text>'
                   f'<rect x="250" y="{y}" width="{w:.1f}" height="19" rx="2" fill="{c}"/>'
                   f'<text x="{250 + w + 6:.1f}" y="{y + 14}" font-size="12.5" fill="currentColor">{f.format(v)} {unit}</text>')
    out.append("</svg></figure>")
    return "".join(out)


def step_report(result_dir):
    rd = Path(result_dir)
    b = json.loads((rd / "benchmark.json").read_text())
    info = json.loads((rd / "run_info.json").read_text()) if (rd / "run_info.json").exists() else {}
    rows = sorted(b["results"], key=lambda r: (ORDER.index(r["design"]), r["granularity"]))
    has_acc = any(r.get("mAP50-95") is not None for r in rows)
    has_speed = any(r.get("ms_median") is not None for r in rows)

    with open(rd / "results.csv", "w", newline="", encoding="utf8") as fh:
        cols = ["file", "design", "granularity", "size_mb", "compression", "mAP50-95", "mAP50", "retained_pct", "precision", "recall", "F1",
                "ms_median", "ms_p95", "speedup", "peak_mb", "agreement_with_fp32", "error"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    th = ["Model", "Size (MB)", "Smaller by"]
    if has_acc:
        th += ["mAP50-95", "Kept", "mAP50", "Precision", "Recall", "F1"]
    if has_speed:
        th += ["ms per image (median)", "p95 ms", "Faster by", "Peak memory added (MB)"]
    th += ["Output agreement with FP32"]
    body = []
    for r in rows:
        tds = [short(r), fmt(r.get("size_mb"), "{:.2f}"), fmt(r.get("compression"), "{:.2f}x")]
        if has_acc:
            tds += [fmt(r.get("mAP50-95"), "{:.4f}"), fmt(r.get("retained_pct"), "{:.1f}%"), fmt(r.get("mAP50"), "{:.4f}"),
                    fmt(r.get("precision"), "{:.3f}"), fmt(r.get("recall"), "{:.3f}"), fmt(r.get("F1"), "{:.3f}")]
        if has_speed:
            tds += [fmt(r.get("ms_median"), "{:.1f}"), fmt(r.get("ms_p95"), "{:.1f}"), fmt(r.get("speedup"), "{:.2f}x"), fmt(r.get("peak_mb"), "{:.0f}")]
        tds += [fmt(r.get("agreement_with_fp32"), "{:.3f}")]
        body.append(f"<tr{' class=base' if r['design'] == 'fp32' else ''}>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
    table = "<div class='tablewrap'><table><thead><tr>" + "".join(f"<th>{t}</th>" for t in th) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    col = lambda r: COLORS[r["design"]]
    charts = ""
    if has_acc:
        charts += bar_chart("Accuracy (mAP@0.5:0.95, higher is better)", [(short(r), r.get("mAP50-95"), col(r)) for r in rows], "", "{:.4f}")
    if has_speed:
        charts += bar_chart("Time per image (milliseconds, lower is better)", [(short(r), r.get("ms_median"), col(r)) for r in rows], "ms", "{:.1f}")
        charts += bar_chart("Memory added while running (MB, lower is better)", [(short(r), r.get("peak_mb"), col(r)) for r in rows], "MB", "{:.0f}")
    charts += bar_chart("File size (MB, lower is better)", [(short(r), r.get("size_mb"), col(r)) for r in rows], "MB", "{:.2f}")

    fp = next((r for r in rows if r["design"] == "fp32"), {})
    best = None
    if has_acc and has_speed:
        cand = [r for r in rows if r["design"] != "fp32" and r.get("retained_pct") and r.get("speedup")]
        ok = [r for r in cand if r["retained_pct"] >= 95] or cand
        best = max(ok, key=lambda r: r["speedup"]) if ok else None
    summary = []
    if fp.get("size_mb"):
        summary.append(f"The FP32 baseline is {fp['size_mb']:.2f} MB" + (f" and scores mAP@0.5:0.95 {fp['mAP50-95']:.4f}" if fp.get("mAP50-95") is not None else "") +
                       (f"; it takes {fp['ms_median']:.1f} ms per image on this computer" if fp.get("ms_median") else "") + ".")
    if best:
        summary.append(f"The fastest INT8 variant that keeps at least 95% of the accuracy (or the closest to it) is <b>{short(best)}</b>: "
                       f"{best['retained_pct']:.1f}% of the baseline accuracy, {best['speedup']:.2f}x faster, {best['compression']:.2f}x smaller.")
    warn_items = info.get("warnings", []) + b.get("warnings", []) + [f"{r['file']}: {r['error']}" for r in rows if r.get("error")]
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in warn_items)
    setup = [("Model file", info.get("pt", b["name"] + ".pt")), ("Classes", b["nc"]), ("Input size", f"{b['imgsz']} x {b['imgsz']}"),
             ("Test pictures (accuracy)", f"{b['n_test_pictures']} pictures, {b.get('n_test_objects', 0)} labelled objects" if b["n_test_pictures"] else "none usable: accuracy not measured"),
             ("Calibration pictures", b.get("n_calibration")),
             ("Computer", f"{b['physical_cores']} physical / {b['logical_cores']} logical cores; {b['threads']} threads pinned to CPUs {b['cpus']}"),
             ("TFLite runtime", b["runtime"]), ("Speed protocol", f"batch 1, 20 warm-up + 100 timed calls per round, {b['rounds']} interleaved rounds, median of the rounds")]
    setup_html = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in setup)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(b['name'])}: TFLite variants</title>
<style>:root{{color-scheme:light dark}}body{{margin:0;font:16px/1.6 Georgia,serif;background:#fff;color:#1a1a1a}}@media(prefers-color-scheme:dark){{body{{background:#16181c;color:#e4e6ea}}th,td{{border-color:#444!important}}}}
.page{{max-width:1000px;margin:0 auto;padding:40px 22px 80px}}h1,h2,table{{font-family:-apple-system,"Segoe UI",Arial,sans-serif}}h1{{font-size:28px}}h2{{font-size:19px;margin-top:38px}}
.tablewrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:6px 10px 6px 0;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
tr.base td{{font-weight:700}}table.setup td{{text-align:left;white-space:normal}}figure{{margin:22px 0}}.warn{{background:#fff4e5;color:#1a1a1a;border-left:4px solid #e08a00;padding:8px 14px;margin:18px 0}}</style></head><body><main class="page">
<h1>{html.escape(b['name'])}: TFLite variants against the FP32 baseline</h1>
<p>{' '.join(summary)}</p>
{'<div class="warn"><b>Please note</b><ul>' + warnings + '</ul></div>' if warnings else ''}
<h2>Results</h2>{table}
<p style="font-size:14px">"Smaller by", "Kept" (share of the baseline accuracy) and "Faster by" are against the FP32 baseline in the same table. Per-channel = one INT8 scale per output channel of each layer (more accurate); per-tensor = one scale per layer.
"Output agreement" is the correlation between the model's scores and the baseline's on one picture (1.000 = identical).</p>
{charts}
<h2>What the variants are</h2>
<ul><li><b>FP32 baseline</b>: the model you provided, converted to TFLite without quantization. It is about twice the size of the .pt file because a YOLOv5 .pt stores its numbers in 16 bits.</li>
<li><b>M1 all INT8</b>: every layer, including the detection head, in 8-bit integers. Smallest and usually fastest, least accurate. Box coordinates are scaled to 0-1 inside the model so they survive 8-bit rounding.</li>
<li><b>M2 backbone INT8</b>: only the backbone is INT8; the neck and head stay float. Most accurate INT8 variant, largest file.</li>
<li><b>M3 backbone + neck INT8</b>: backbone and neck INT8, the detection head stays float. Usually the best balance.</li>
<li>The weights of the detection head are never changed in any variant.</li></ul>
<h2>Setup</h2><div class="tablewrap"><table class="setup"><tbody>{setup_html}</tbody></table></div>
<h2>How to read these numbers</h2>
<ul><li>mAP@0.5:0.95 is the standard detector score (0 to 1, higher is better) on your test pictures. A loss of a few percent is typical for 8-bit models.</li>
<li>Time and memory were measured on this computer's CPU. An NPU or another CPU ranks the variants differently: measure on the target.</li>
<li>Precision / recall / F1 are at confidence 0.25 and overlap 0.5; mAP uses confidence 0.001 as in the standard validation.</li>
<li>TFLite's converter only offers min-max calibration; unusual calibration pictures widen the ranges and cost accuracy, so use pictures that look like the real data.</li></ul>
</main></body></html>"""
    (rd / "report.html").write_text(page, encoding="utf8")
    return rd / "report.html"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--work", required=True)
    b.add_argument("--models", required=True)
    b.add_argument("--data", default="", help="test pictures (YOLO layout folder or data.yaml); without it accuracy is not measured")
    b.add_argument("--max-images", type=int, default=1000)
    b.add_argument("--rounds", type=int, default=9)
    b.add_argument("--exclude", default="", help="text file with pictures to leave out of the accuracy test (used for calibration)")
    b.add_argument("--skip-speed", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--result-dir", required=True)
    c = sub.add_parser("child")
    c.add_argument("rest", nargs="+")
    args = ap.parse_args()
    if args.step == "bench":
        step_bench(args)
    elif args.step == "report":
        print("wrote", step_report(args.result_dir))
    else:
        k, rest = args.rest[0], args.rest[1:]
        if k == "latency":
            child_latency(rest[0], int(rest[1]), int(rest[2]), int(rest[3]), rest[4], rest[5] if len(rest) > 5 else "")
        else:
            child_memory(rest[0], int(rest[1]), int(rest[2]), int(rest[3]))
