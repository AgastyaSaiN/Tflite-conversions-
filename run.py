"""run.py  -  the factory. Every YOLOv5 .pt file in  put_pt_files_here  becomes TFLite variants plus a benchmark against its own FP32 baseline.

    python run.py              normal run   (or double-click RUN.bat / run ./RUN.sh)
    python run.py --check      only check that this computer is ready (CHECK_SETUP.bat / ./CHECK_SETUP.sh)
    python run.py --quick      short trial: 100 calibration pictures, 200 test pictures, 3 speed rounds
    python run.py --force      redo everything (otherwise finished steps are skipped, so an interrupted run can simply be started again)
    python run.py --skip-speed skip the speed and memory measurements (accuracy is still measured when test pictures exist)
    python run.py --only abcd123   only the file abcd123.pt

Input folders:  put_pt_files_here/ (models)   put_test_pictures_here/ (pictures + labels, optional)   put_calibration_pictures_here/ (pictures, recommended)
Output:         results/<model name>/report.html (open this), results.csv, models/*.tflite, logs/

THE STEPS (one Python file each, all explained inside):
    1  convert.py export   .pt  -> ONNX                          (needs PyTorch)
    2  convert.py tflite   ONNX -> FP32 baseline + 6 INT8 files  (needs TensorFlow; the long step)
    3  benchmark.py bench  accuracy, speed, memory of every file
    4  benchmark.py report report.html + results.csv
This file only finds a suitable Python for each step, runs the steps in order, keeps the logs and explains failures in plain words.

ENVIRONMENTS
    One Python 3.11 environment that has all the packages is enough (SETUP_LINUX.sh builds it on Linux). If the packages are split over
    several Pythons, this file finds the right one for each step by itself; to force one, set BASEPT_PYTHON (or BASEPT_PYTHON_EXPORT /
    BASEPT_PYTHON_TF / BASEPT_PYTHON_BENCH).
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor_yolov5"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------------------------------------------------
# Which Python can do which step?
# ---------------------------------------------------------------------------------------------------------------------
# import name -> (pip package, what it is for)
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
    "export": ("read the .pt file and export it to ONNX", ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "cv2", "yaml", "pandas", "tqdm", "requests",
                                                           "PIL", "matplotlib", "scipy", "seaborn", "IPython", "pkg_resources"]),
    "tf": ("convert to TFLite and quantize to INT8", ["numpy", "tensorflow", "tf_keras", "onnx", "onnxruntime", "onnx2tf", "onnx_graphsurgeon", "sng4onnx", "onnxsim", "cv2"]),
    "bench": ("measure accuracy, speed and memory", ["numpy", "cv2", "pycocotools", "psutil", "yaml"]),       # plus ai_edge_litert or tensorflow
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
RUN = dict(capture_output=True, text=True, encoding="utf8", errors="replace")


def candidates():
    """Pythons to try, best guess first: the one running this, ones named in environment variables, environments next to this folder."""
    seen, out = set(), []

    def add(p):
        p = str(p)
        if p and Path(p).exists() and p not in seen:
            seen.add(p)
            out.append(p)
    add(sys.executable)
    for k in ("BASEPT_PYTHON", "BASEPT_PYTHON_EXPORT", "BASEPT_PYTHON_TF", "BASEPT_PYTHON_BENCH"):
        add(os.environ.get(k, ""))
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for root in (HERE, HERE.parent / "Day 2"):
        for d in sorted(root.glob(".venv*")):
            add(d / sub)
    add(Path(os.environ.get("CONDA_PREFIX", "")) / sub) if os.environ.get("CONDA_PREFIX") else None
    return out


def probe(python, modules):
    r = subprocess.run([python, "-c", PROBE, *modules], **RUN)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {m: "this Python could not be started" for m in modules}


def vendor_ok(python):
    """The bundled YOLOv5 code imports with this Python (it pulls in many packages); error text or None."""
    r = subprocess.run([python, "-c", f"import sys; sys.path.insert(0, r'{VENDOR}'); import models.yolo, export"], env=dict(os.environ, YOLOv5_AUTOINSTALL="False"), **RUN)
    return None if not r.returncode else (r.stderr.strip().splitlines() or ["import failed"])[-1][:200]


def find_python(role):
    """(python, report): the first interpreter that has everything the role needs, or None plus what each candidate lacks."""
    report = []
    for py in candidates():
        need = ROLES[role][1]
        res = probe(py, need + (["ai_edge_litert", "tensorflow"] if role == "bench" else []))
        missing = [m for m in need if res.get(m)]
        if role == "bench" and res.get("ai_edge_litert") and res.get("tensorflow"):
            missing.append("ai_edge_litert")                    # needs LiteRT or TensorFlow; neither is there
        extra = vendor_ok(py) if role == "export" and not missing else None
        report.append((py, missing, extra))
        if not missing and not extra:
            return py, report
    return None, report


def pictures(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT) if Path(folder).exists() else []


def check_setup(quiet=False):
    """Prints what is ready and what is missing (with the offline install command). Returns (all_ok, {role: python})."""
    p = (lambda *a: None) if quiet else print
    ok_all, found = True, {}
    p("=" * 78 + "\nSetup check\n" + "=" * 78)
    p(f"Python running this check: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        p("  note: built with Python 3.11. Other versions may work if all packages install; 3.11 is the safe choice.")
    for role, (what, _) in ROLES.items():
        py, report = find_python(role)
        found[role] = py
        if py:
            p(f"[ OK ] {role:<7} {what}\n         uses: {py}")
            continue
        ok_all = False
        p(f"[ !! ] {role:<7} {what}: NOT READY")
        for cand, missing, extra in report[:3]:
            p(f"         tried {cand}")
            for m in missing:
                pkg, why = PKG.get(m, (m, ""))
                p(f"    - {m:<18} ({why}); install:  pip install --no-index --find-links linux_offline_packages/wheelhouse \"{pkg}\"")
            if extra:
                p(f"         the bundled YOLOv5 code does not import: {extra}")
        if not report:
            p("         no Python found at all")
    pts = sorted((HERE / "put_pt_files_here").glob("*.pt"))
    nt, nc = len(pictures(HERE / "put_test_pictures_here")), len(pictures(HERE / "put_calibration_pictures_here"))
    p(f"\nModels in put_pt_files_here: {len(pts)}" + ("  (" + ", ".join(x.name for x in pts) + ")" if pts else "  -> copy the .pt file(s) there"))
    p(f"Test pictures: {nt}" + ("" if nt else "  -> without them accuracy is not measured (size, speed and memory still are)"))
    p(f"Calibration pictures: {nc}" + ("" if nc else "  -> empty: part of the test pictures will be used instead"))
    p(f"Free disk space here: {shutil.disk_usage(HERE).free / 1e9:.0f} GB (a conversion needs about 3 GB per model);  CPU cores: {os.cpu_count()}\n")
    p("READY." if ok_all else "NOT READY. Fix the items marked [ !! ], then run the check again. Offline computer: see README.md, part 'If something is missing'.")
    return ok_all, found


# ---------------------------------------------------------------------------------------------------------------------
# Running the steps
# ---------------------------------------------------------------------------------------------------------------------
HINTS = {
    1: "Check that the file is a YOLOv5 checkpoint (saved by the YOLOv5 training script). YOLOv8 / YOLO11 files are a different family and are not supported.",
    2: "Most often: not enough memory (close other programs), or onnx2tf could not convert a layer of this model. The file named in the message above "
       "(onnx2tf_*.log inside the work folder) has the details; keep it together with ERROR.txt if you ask for help.",
    3: "Most often a problem reading the test pictures or labels, or LiteRT/TensorFlow could not load one of the .tflite files. The messages above name the model.",
}


def say(msg=""):
    print(msg, flush=True)


def run_stage(title, cmd, log, env):
    """Runs a command, shows its output and keeps it in a log file. Returns the exit code."""
    say(f"\n--- {title}")
    t0 = time.time()
    with open(log, "a", encoding="utf8") as fh:
        fh.write(f"\n=== {title}\n$ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace", env=env)
        for line in p.stdout:
            fh.write(line)
            line = line.rstrip()
            if line and not any(s in line for s in ("UserWarning", "pkg_resources", "WARNING: All log messages", "I0000", "W0000", "INFO: Created TensorFlow", "f32", "occurrences")):
                say("    " + line[:200])
        p.wait()
    say(f"    ({time.time() - t0:.0f} s)")
    return p.returncode


def last_lines(log, n=25):
    try:
        return "".join(Path(log).read_text(encoding="utf8", errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""


def has_labels(folder):
    return bool([f for f in Path(folder).rglob("*.txt") if f.name.lower() != "readme.txt"] or list(Path(folder).glob("*.y*ml")))


def process(pt, args, py, test_dir, calib_dir):
    """All four steps for one .pt file. Returns True when the report was written."""
    name = pt.stem
    rd = HERE / "results" / name
    work, models, logs = rd / "work", rd / "models", rd / "logs"
    if args.force and rd.exists():
        shutil.rmtree(rd)
    for d in (work, models, logs):
        d.mkdir(parents=True, exist_ok=True)
    log = logs / "run.log"
    env = dict(os.environ, YOLOv5_AUTOINSTALL="False", PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1")
    warnings = []
    say("\n" + "=" * 78 + f"\nModel: {pt.name}\n" + "=" * 78)

    def fail(stage, code):
        msg = f"STEP {stage} FAILED (exit code {code}).\n\nLast lines of the log:\n{last_lines(log)}\n\nWhat to try:\n{HINTS.get(stage, '')}\n\nFull log: {log}\n"
        (rd / "ERROR.txt").write_text(msg, encoding="utf8")
        say("\n" + "!" * 78 + f"\nSTEP {stage} FAILED for {pt.name}. Details are in {rd / 'ERROR.txt'}\n  {HINTS.get(stage, '')}\n" + "!" * 78)
        return False

    # ---- pictures: calibration (for INT8) and test (for accuracy)
    test_imgs = pictures(test_dir)
    test_ok = bool(test_imgs) and has_labels(test_dir)
    calib_files, exclude = pictures(calib_dir), []
    if not calib_files and test_imgs:                          # no calibration pictures: borrow some test pictures and keep them out of the accuracy test
        k = min(args.n_calib, max(len(test_imgs) // 4, 50 if len(test_imgs) > 50 else len(test_imgs)))
        calib_files = exclude = random.Random(0).sample(test_imgs, min(k, len(test_imgs)))
        warnings.append(f"No pictures in put_calibration_pictures_here: {len(calib_files)} of the test pictures were used for calibration and are left out of the accuracy test.")
    if not calib_files:
        say("ERROR: there are no pictures to calibrate the INT8 models with.\n       Put 100 to 300 pictures that look like the real data into  put_calibration_pictures_here  and start again.")
        (rd / "ERROR.txt").write_text("No calibration pictures. Put pictures into put_calibration_pictures_here.\n", encoding="utf8")
        return False
    calib_files = calib_files[:args.n_calib]
    (work / "calib_images.txt").write_text("\n".join(map(str, calib_files)), encoding="utf8")
    (work / "exclude.txt").write_text("\n".join(map(str, exclude)), encoding="utf8")
    if len(calib_files) < 50:
        warnings.append(f"Only {len(calib_files)} calibration pictures: 100 or more is better.")
    if not test_imgs:
        warnings.append("No test pictures were given, so accuracy (mAP) was NOT measured; only size, speed and memory.")
    elif not test_ok:
        warnings.append("The test folder has pictures but no label files (.txt), so accuracy was NOT measured.")

    # ---- step 1: export
    if not (work / "layout.json").exists():
        code = run_stage("Step 1 of 4: reading the .pt and exporting it to ONNX (about 1 minute)",
                         [py["export"], HERE / "convert.py", "export", "--pt", pt, "--work", work, "--imgsz", args.imgsz], log, env)
        if code:
            return fail(1, code)
    else:
        say("\n--- Step 1 of 4: already done (use --force to redo)")

    # ---- step 2: TFLite conversion
    info = models / "models.json"
    if not (info.exists() and all(e.get("exists") for e in json.loads(info.read_text())["models"])):
        code = run_stage(f"Step 2 of 4: converting to TFLite and quantizing ({len(calib_files)} calibration pictures; the long step, roughly 20-60 minutes)",
                         [py["tf"], HERE / "convert.py", "tflite", "--work", work, "--out", models, "--calib-list", work / "calib_images.txt", "--n-calib", args.n_calib], log, env)
        if code and not info.exists():
            return fail(2, code)
        if code:
            warnings.append("Some variants could not be built (listed below); the others were measured.")
    else:
        say("\n--- Step 2 of 4: already done (use --force to redo)")

    # ---- step 3: benchmark
    cmd = [py["bench"], HERE / "benchmark.py", "bench", "--work", work, "--models", models, "--max-images", args.max_test, "--rounds", args.rounds]
    if test_ok:
        cmd += ["--data", test_dir, "--exclude", work / "exclude.txt"]
    if args.skip_speed:
        cmd += ["--skip-speed"]
    if run_stage("Step 3 of 4: measuring accuracy, speed and memory (several minutes)", cmd, log, env):
        return fail(3, 1)

    # ---- step 4: report
    (rd / "run_info.json").write_text(json.dumps({"pt": pt.name, "warnings": warnings, "settings": vars(args)}, indent=1, default=str), encoding="utf8")
    if run_stage("Step 4 of 4: writing the report", [py["bench"], HERE / "benchmark.py", "report", "--result-dir", rd], log, env):
        return fail(4, 1)
    say(f"\nDONE: {rd / 'report.html'}")
    return True


def write_index():
    """results/index.html: one line per model with a link to its report."""
    rows = []
    for rd in sorted((HERE / "results").glob("*/")):
        if not (rd / "benchmark.json").exists():
            rows.append(f"<tr><td>{rd.name}</td><td colspan=3>not finished (see {rd.name}/ERROR.txt)</td></tr>")
            continue
        res = json.loads((rd / "benchmark.json").read_text())["results"]
        fp = next((r for r in res if r["design"] == "fp32"), {})
        cand = [r for r in res if r["design"] != "fp32" and r.get("speedup") and r.get("retained_pct")]
        best = max([r for r in cand if r["retained_pct"] >= 95] or cand, key=lambda r: r["speedup"], default=None)
        text = f"{best['file']}: {best['retained_pct']:.1f}% accuracy kept, {best['speedup']:.2f}x faster" if best else "-"
        rows.append(f"<tr><td><a href='{rd.name}/report.html'>{rd.name}</a></td><td>{fp.get('size_mb', 0):.2f}</td><td>{fp.get('mAP50-95', float('nan')):.4f}</td><td>{text}</td></tr>")
    (HERE / "results" / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Results</title><body style='font:16px Arial;margin:30px'><h1>Results</h1><table cellpadding=6 border=0>"
        "<tr><th align=left>Model</th><th>FP32 MB</th><th>FP32 mAP</th><th align=left>Best INT8 variant</th></tr>" + "".join(rows) + "</table></body>", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--imgsz", type=int, default=640, help="input size; keep 640 unless the model was trained at another size")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_calib, args.max_test, args.rounds = 100, 200, 3
    if args.check:
        sys.exit(0 if check_setup()[0] else 1)

    ok, py = check_setup(quiet=True)
    if not ok:
        say("This computer is not ready yet. Here is what is missing:\n")
        check_setup()
        sys.exit(1)
    say(f"Python for export: {py['export']}\nPython for TFLite conversion: {py['tf']}\nPython for measuring: {py['bench']}")
    pts = [p for p in sorted((HERE / "put_pt_files_here").glob("*.pt")) if not args.only or p.stem == args.only]
    if not pts:
        say("\nNo .pt file found in the folder  put_pt_files_here . Copy the model file(s) there and start again.")
        sys.exit(1)
    say(f"\nModels: {', '.join(p.name for p in pts)}")
    results = {}
    for pt in pts:
        try:
            results[pt.name] = process(pt, args, py, HERE / "put_test_pictures_here", HERE / "put_calibration_pictures_here")
        except Exception:
            say("\nUNEXPECTED ERROR (details saved to ERROR.txt):\n" + traceback.format_exc())
            (HERE / "results" / pt.stem).mkdir(parents=True, exist_ok=True)
            (HERE / "results" / pt.stem / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf8")
            results[pt.name] = False
    write_index()
    say("\n" + "=" * 78 + "\nSummary")
    for k, v in results.items():
        say(f"  {k}: {'done' if v else 'FAILED (see results/' + Path(k).stem + '/ERROR.txt)'}")
    say(f"Open  {HERE / 'results' / 'index.html'}  for the list of reports.")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
