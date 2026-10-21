"""Finds out which Python on this computer can do which job, and says in plain words what is missing (standard library only).

Jobs ("roles"):
    export   read the .pt and export it to ONNX         needs torch, torchvision, onnx, onnxruntime, ... and the bundled YOLOv5 code
    tf       convert ONNX to TFLite and quantize        needs tensorflow, onnx2tf, onnx, onnxruntime, ...
    bench    measure accuracy, speed and memory         needs numpy, opencv, pycocotools, psutil and LiteRT (or tensorflow)
One Python that has everything can do all three jobs; if the packages are split over two environments, that works too.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VENDOR = HERE / "vendor_yolov5"

# python import name -> (pip package, what it is used for)
PKG = {
    "numpy": ("numpy==1.26.4", "arrays (must be below 2.0 for PyTorch 2.3 and TensorFlow 2.15)"),
    "torch": ("torch==2.3.1+cpu", "reads the .pt file and exports ONNX"),
    "torchvision": ("torchvision==0.18.1+cpu", "needed by the bundled YOLOv5 code"),
    "onnx": ("onnx==1.16.2", "ONNX files"),
    "onnxruntime": ("onnxruntime==1.18.1", "checks the exported ONNX"),
    "cv2": ("opencv-python-headless==4.11.0.86", "reading pictures"),
    "yaml": ("pyyaml", "reading data.yaml and model configs"),
    "pandas": ("pandas", "needed by the bundled YOLOv5 code"),
    "tqdm": ("tqdm", "needed by the bundled YOLOv5 code"),
    "requests": ("requests", "needed by the bundled YOLOv5 code"),
    "PIL": ("pillow", "needed by the bundled YOLOv5 code"),
    "matplotlib": ("matplotlib", "needed by the bundled YOLOv5 code"),
    "scipy": ("scipy", "needed by the bundled YOLOv5 code"),
    "seaborn": ("seaborn", "needed by the bundled YOLOv5 code"),
    "IPython": ("ipython", "needed by the bundled YOLOv5 code"),
    "pkg_resources": ("setuptools<81", "needed by the bundled YOLOv5 code"),
    "tensorflow": ("tensorflow==2.15.1", "the TFLite converter"),
    "tf_keras": ("tf_keras==2.15.1", "needed by onnx2tf"),
    "onnx2tf": ("onnx2tf==1.28.8", "converts ONNX to TensorFlow"),
    "onnx_graphsurgeon": ("onnx_graphsurgeon==0.6.1", "needed by onnx2tf"),
    "sng4onnx": ("sng4onnx==2.0.1", "needed by onnx2tf"),
    "onnxsim": ("onnxsim==0.7.3", "needed by onnx2tf"),
    "pycocotools": ("pycocotools==2.0.11", "scores detections (mAP)"),
    "psutil": ("psutil", "pins the speed test to CPU cores and measures memory"),
    "ai_edge_litert": ("ai-edge-litert==2.3.0", "runs the TFLite models (LiteRT)"),
}
ROLES = {
    "export": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "cv2", "yaml", "pandas", "tqdm", "requests", "PIL", "matplotlib", "scipy", "seaborn", "IPython", "pkg_resources"],
    "tf": ["numpy", "tensorflow", "tf_keras", "onnx", "onnxruntime", "onnx2tf", "onnx_graphsurgeon", "sng4onnx", "onnxsim", "cv2"],
    "bench": ["numpy", "cv2", "pycocotools", "psutil", "yaml"],       # plus ai_edge_litert or tensorflow
}
PROBE = r"""
import importlib, json, sys
res = {}
for m in sys.argv[1:]:
    try:
        importlib.import_module(m); res[m] = None
    except BaseException as e:
        res[m] = f"{type(e).__name__}: {str(e)[:160]}"
print(json.dumps(res))
"""


def candidates():
    """Pythons to try, best guess first: the one running this, ones named in environment variables, and environments next to this folder."""
    seen, out = set(), []

    def add(p):
        p = str(p)
        if p and Path(p).exists() and p not in seen:
            seen.add(p); out.append(p)
    add(sys.executable)
    for k in ("BASEPT_PYTHON", "BASEPT_PYTHON_EXPORT", "BASEPT_PYTHON_TF", "BASEPT_PYTHON_BENCH"):
        add(os.environ.get(k, ""))
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for root in (HERE, HERE.parent / "Day 2"):
        for d in sorted(root.glob(".venv*")):
            add(d / sub)
    for p in os.environ.get("CONDA_PREFIX", "").split(os.pathsep):
        add(Path(p) / sub) if p else None
    return out


def probe(python, modules):
    """{module: None if it imports, else the error text}."""
    r = subprocess.run([python, "-c", PROBE, *modules], capture_output=True, text=True, encoding="utf8", errors="replace")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {m: "this Python could not be started" for m in modules}


def vendor_ok(python):
    """The bundled YOLOv5 code imports with this Python (it pulls in many packages); returns an error text or None."""
    code = f"import sys; sys.path.insert(0, r'{VENDOR}'); import models.yolo, export"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, encoding="utf8", errors="replace", env=dict(os.environ, YOLOv5_AUTOINSTALL="False"))
    if r.returncode:
        return (r.stderr.strip().splitlines() or ["import failed"])[-1][:200]
    return None


def find(role):
    """Returns (python, report) where python is the first interpreter that satisfies the role (or None); report lists what each candidate lacks."""
    report = []
    for py in candidates():
        need = ROLES[role]
        res = probe(py, need + (["ai_edge_litert", "tensorflow"] if role == "bench" else []))
        missing = [m for m in need if res.get(m)]
        if role == "bench" and res.get("ai_edge_litert") and res.get("tensorflow"):
            missing.append("ai_edge_litert")
        extra = None
        if role == "export" and not missing:
            extra = vendor_ok(py)
        report.append((py, missing, extra, res))
        if not missing and not extra:
            return py, report
    return None, report


def advice(missing):
    """Plain-language fix for missing modules."""
    lines = []
    for m in missing:
        pkg, why = PKG.get(m, (m, ""))
        lines.append(f"    - {m:<18} ({why}); install:  pip install --no-index --find-links linux_offline_packages/wheelhouse \"{pkg}\"")
    return lines
