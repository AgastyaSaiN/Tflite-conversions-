"""Speed and memory of one .tflite model in a fresh process (called by step3_benchmark.py; not meant to be run by hand).

    python lib/measure.py latency <model> <nc> <size> <threads> <cpus> <pictures.txt>      -> one JSON line
    python lib/measure.py memory  <model> <nc> <size> <threads>                            -> one JSON line
"""
import gc
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
WARMUP, RUNS = 20, 100


def pictures(list_file, size, n=10):
    import yolo_io
    items = []
    if list_file and Path(list_file).exists():
        for line in Path(list_file).read_text(encoding="utf8").splitlines():
            img = yolo_io.read_image(line.strip()) if line.strip() else None
            if img is not None:
                items.append(yolo_io.to_input(yolo_io.letterbox(img, size)[0]))
            if len(items) >= n:
                break
    if not items:                                               # no pictures given: random noise (speed does not depend on the content)
        items = [np.random.RandomState(i).rand(1, 3, size, size).astype(np.float32) for i in range(n)]
    return items


def latency(model, nc, size, threads, cpus, list_file):
    import psutil
    p = psutil.Process()
    try:
        p.cpu_affinity([int(c) for c in cpus.split(",")])          # one thread per physical core: no sharing of a core between two threads
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass
    from tflite_run import Model
    m = Model(model, threads, nc)
    items = pictures(list_file, size)
    for k in range(WARMUP):
        m(items[k % len(items)])
    t = []
    for k in range(RUNS):
        t0 = time.perf_counter(); m(items[k % len(items)]); t.append(time.perf_counter() - t0)
    print(json.dumps({"median": float(np.median(t) * 1e3), "p95": float(np.percentile(t, 95) * 1e3), "mean": float(np.mean(t) * 1e3)}))


def memory(model, nc, size, threads):
    import psutil
    from tflite_run import Model                                # imported before the baseline: the runtime itself is not counted
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


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "latency":
        latency(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7] if len(sys.argv) > 7 else "")
    else:
        memory(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
