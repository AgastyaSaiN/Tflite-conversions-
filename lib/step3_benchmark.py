"""Step 3 (a Python with LiteRT or TensorFlow, OpenCV, pycocotools, psutil): accuracy, size, speed and memory of the FP32 baseline and the INT8 variants.

    python lib/step3_benchmark.py --work results/abcd123/work --models results/abcd123/models --data path/to/dataset --max-images 1000

Without --data only size, speed and memory are measured (accuracy needs pictures with labels).
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yolo_io  # noqa: E402
from tflite_run import RUNTIME, Model  # noqa: E402


def cpu_plan(max_threads=4):
    """Threads and the logical CPUs to pin them to: one thread per physical core (skipping hyper-thread siblings) when that can be told."""
    import psutil
    phys, logical = psutil.cpu_count(logical=False) or 1, psutil.cpu_count(logical=True) or 1
    threads = max(1, min(max_threads, phys))
    step = 2 if logical >= 2 * phys else 1
    return threads, [i * step for i in range(threads)], phys, logical


def accuracy(path, ds, nc, size, gt, shapes, preload):
    m = Model(path, 4, nc)
    dets = []
    for img_shape, (x, r, pad) in zip(shapes, preload):
        pred = m(x)
        dets.append(yolo_io.decode(pred, nc, size, 0.001, 0.6, r, pad, img_shape, multi_label=True))
    return yolo_io.score_detections(dets, gt, nc, len(ds))


def child(kind, *args):
    r = subprocess.run([sys.executable, str(HERE / "measure.py"), kind, *map(str, args)], capture_output=True, text=True, encoding="utf8", errors="replace")
    if r.returncode or not r.stdout.strip():
        raise RuntimeError((r.stderr or "no output")[-400:])
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--pictures", default="", help="text file with pictures used for the speed test (default: the test pictures, else random noise)")
    ap.add_argument("--exclude", default="", help="text file with pictures to leave out of the accuracy test (they were used for calibration)")
    ap.add_argument("--skip-speed", action="store_true")
    a = ap.parse_args()
    work, models_dir = Path(a.work), Path(a.models)
    layout = json.loads((work / "layout.json").read_text())
    nc, size = layout["nc"], layout["imgsz"]
    info = json.loads((models_dir / "models.json").read_text())
    entries = [e for e in info["models"] if e.get("exists")]
    print(f"{len(entries)} models to measure; runtime: {RUNTIME}", flush=True)

    ds = gt = shapes = preload = None
    if a.data:
        ds = yolo_io.Dataset(a.data, names=layout["names"])
        if a.exclude and Path(a.exclude).exists():
            ex = {l.strip() for l in Path(a.exclude).read_text(encoding="utf8").splitlines() if l.strip()}
            keep = [k for k, im in enumerate(ds.images) if str(im) not in ex]
            if keep and len(keep) < len(ds.images):
                ds.images, ds.labels = [ds.images[k] for k in keep], [ds.labels[k] for k in keep]
        ds.subset(a.max_images)
        shapes, preload, good = [], [], []
        for k, p in enumerate(ds.images):
            img = yolo_io.read_image(p)
            if img is None:
                continue
            lb, r, pad = yolo_io.letterbox(img, size)
            shapes.append(img.shape[:2]); preload.append((yolo_io.to_input(lb), r, pad)); good.append(k)
        if len(good) != len(ds.images):
            print(f"WARNING: {len(ds.images) - len(good)} test pictures could not be read and were skipped", flush=True)
            ds.images, ds.labels = [ds.images[k] for k in good], [ds.labels[k] for k in good]
        labels_max = int(max((l[:, 0].max() for l in ds.labels if len(l)), default=-1))
        if labels_max >= nc:
            print(f"WARNING: the labels use class number {labels_max}, but the model has only {nc} classes (0 to {nc - 1}).", flush=True)
        gt = yolo_io.coco_ground_truth(ds.images, ds.labels, shapes, nc)
        print(f"test set: {len(ds)} pictures, {len(gt.dataset['annotations'])} labelled objects", flush=True)

    res = {e["file"]: dict(e) for e in entries}
    threads, cpus, phys, logical = cpu_plan()
    if ds is not None:
        for e in entries:
            r, t0 = res[e["file"]], time.time()
            try:
                r.update(accuracy(models_dir / e["file"], ds, nc, size, gt, shapes, preload))
                print(f"   accuracy {e['file']}: mAP50-95 {r['mAP50-95']:.4f}  mAP50 {r['mAP50']:.4f}  F1 {r['F1']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
            except Exception as ex:
                r["error"] = f"accuracy: {type(ex).__name__}: {ex}"
                print(f"   accuracy {e['file']}: FAILED {r['error']}", flush=True)

    if not a.skip_speed:
        pics = Path(a.pictures) if a.pictures else work / "speed_pictures.txt"
        if not a.pictures:
            pics.write_text("\n".join(str(p) for p in (ds.images[:10] if ds is not None else [])), encoding="utf8")
        print(f"speed: {threads} threads pinned to logical CPUs {cpus} ({phys} physical, {logical} logical cores), {a.rounds} interleaved rounds", flush=True)
        rounds = {e["file"]: [] for e in entries}
        for k in range(a.rounds):
            order = [e["file"] for e in entries]
            random.Random(k).shuffle(order)
            for fn in order:
                try:
                    rounds[fn].append(child("latency", models_dir / fn, nc, size, threads, ",".join(map(str, cpus)), pics))
                except Exception as ex:
                    res[fn]["error"] = f"speed: {ex}"
            print(f"   round {k + 1}/{a.rounds} done", flush=True)
        for fn, rs in rounds.items():
            if rs:
                med = [x["median"] for x in rs]
                res[fn].update({"ms_median": float(np.median(med)), "ms_p95": float(np.median([x["p95"] for x in rs])), "ms_round_medians": med,
                                "ms_spread_pct": float((max(med) - min(med)) / np.median(med) * 100)})
        print("memory (3 fresh processes per model, median)", flush=True)
        for e in entries:
            fn = e["file"]
            try:
                ms = [child("memory", models_dir / fn, nc, size, threads) for _ in range(3)]
                res[fn].update({k: float(np.median([m[k] for m in ms])) for k in ("baseline_mb", "load_mb", "peak_mb")})
            except Exception as ex:
                res[fn]["error"] = f"memory: {ex}"
    base = res.get(next((e["file"] for e in entries if e["design"] == "fp32"), ""), {})
    for r in res.values():
        if base.get("size_mb") and r.get("size_mb"):
            r["compression"] = base["size_mb"] / r["size_mb"]
        if base.get("mAP50-95") and r.get("mAP50-95") is not None:
            r["retained_pct"] = r["mAP50-95"] / base["mAP50-95"] * 100
        if base.get("ms_median") and r.get("ms_median"):
            r["speedup"] = base["ms_median"] / r["ms_median"]
    out = {"name": layout["name"], "nc": nc, "imgsz": size, "runtime": RUNTIME, "threads": threads, "cpus": cpus, "physical_cores": phys, "logical_cores": logical,
           "n_test_pictures": len(ds) if ds is not None else 0, "n_calibration": info.get("n_calibration"), "rounds": a.rounds, "results": list(res.values())}
    (models_dir.parent / "benchmark.json").write_text(json.dumps(out, indent=1))
    print("step 3 done")
