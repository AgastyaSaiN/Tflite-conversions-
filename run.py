"""Converts every YOLOv5 .pt file in  put_pt_files_here  into TFLite variants and benchmarks them against the FP32 baseline.

    python run.py                 normal run (or double-click RUN.bat / run ./RUN.sh)
    python run.py --quick         small and fast (100 calibration pictures, 200 test pictures, 3 speed rounds): a trial of the whole chain
    python run.py --force         redo everything for the models (otherwise finished steps are skipped, so an interrupted run can simply be started again)
    python run.py --only abcd123  only the file abcd123.pt

Input folders:   put_pt_files_here/ (the models)   put_test_pictures_here/ (pictures with labels, YOLO format)   put_calibration_pictures_here/ (plain pictures)
Output:          results/<model name>/report.html   (open this),  results.csv,  models/*.tflite,  logs/
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
sys.path.insert(0, str(HERE / "lib"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HINTS = {
    1: "Check that the file is a YOLOv5 checkpoint (saved by the YOLOv5 training script). YOLOv8 / YOLO11 files are a different family and are not supported.",
    2: "Most often: not enough memory (close other programs, or use --quick for a smaller trial), or onnx2tf could not convert a layer of this model. "
       "The file named in the message above (onnx2tf_*.log inside the work folder) has the details; send it along with ERROR.txt if you ask for help.",
    3: "Most often a problem reading the test pictures or labels, or LiteRT/TensorFlow could not load one of the .tflite files. The messages above name the model.",
}


def say(msg=""):
    print(msg, flush=True)


def pictures(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT) if Path(folder).exists() else []


def find_dataset(folder):
    """The test set as (path for the tool, number of pictures, has_labels)."""
    folder = Path(folder)
    for y in sorted(folder.glob("*.y*ml")):
        return y, len(pictures(folder)), True
    imgs = pictures(folder)
    if not imgs:
        return None, 0, False
    labels = list(folder.rglob("*.txt"))
    return folder, len(imgs), bool(labels)


def run_stage(title, cmd, log, env, cwd=None):
    """Runs a command, shows its output and keeps it in a log file. Returns the exit code."""
    say(f"\n--- {title}")
    t0 = time.time()
    with open(log, "a", encoding="utf8") as fh:
        fh.write(f"\n=== {title}\n$ {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace", env=env, cwd=cwd)
        for line in p.stdout:
            fh.write(line)
            line = line.rstrip()
            if line and not any(s in line for s in ("UserWarning", "pkg_resources", "WARNING: All log messages", "I0000", "W0000", "INFO: Created TensorFlow")):
                say("    " + line[:200])
        p.wait()
    say(f"    ({time.time() - t0:.0f} s)")
    return p.returncode


def last_lines(log, n=25):
    try:
        return "".join(Path(log).read_text(encoding="utf8", errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""


def process(pt, args, py, data, calib_dir, warnings_base):
    name = pt.stem
    rd = HERE / "results" / name
    work, models = rd / "work", rd / "models"
    logs = rd / "logs"
    if args.force and rd.exists():
        shutil.rmtree(rd)
    for d in (work, models, logs):
        d.mkdir(parents=True, exist_ok=True)
    log = logs / "run.log"
    env = dict(os.environ, YOLOv5_AUTOINSTALL="False", PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1")
    warnings = list(warnings_base)
    say("\n" + "=" * 78 + f"\nModel: {pt.name}\n" + "=" * 78)

    def fail(stage, code, extra=""):
        msg = (f"STEP {stage} FAILED (exit code {code}).\n\nLast lines of the log:\n{last_lines(log)}\n\nWhat to try:\n{HINTS.get(stage, '')}\n{extra}\n"
               f"Full log: {log}\n")
        (rd / "ERROR.txt").write_text(msg, encoding="utf8")
        say("\n" + "!" * 78 + f"\nSTEP {stage} FAILED for {pt.name}. Details are in {rd / 'ERROR.txt'}\n  " + HINTS.get(stage, "").replace("\n", "\n  ") + "\n" + "!" * 78)
        return False

    # ---- calibration and test pictures
    calib_files = pictures(calib_dir)
    exclude = []
    if not calib_files and data[0] is not None:
        allp = pictures(data[0].parent if data[0].suffix in (".yaml", ".yml") else data[0])
        k = min(args.n_calib, max(len(allp) // 4, 50 if len(allp) > 50 else len(allp)))
        calib_files = random.Random(0).sample(allp, min(k, len(allp)))
        exclude = calib_files
        warnings.append(f"No pictures in put_calibration_pictures_here: {len(calib_files)} of the test pictures were used for calibration and are left out of the accuracy test.")
    if not calib_files:
        say("ERROR: there are no pictures to calibrate the INT8 models with.\n       Put 100 to 300 pictures that look like the real data into the folder  put_calibration_pictures_here  and start again.")
        (rd / "ERROR.txt").write_text("No calibration pictures. Put pictures into put_calibration_pictures_here.\n", encoding="utf8")
        return False
    calib_files = calib_files[:args.n_calib]
    (work / "calib_images.txt").write_text("\n".join(map(str, calib_files)), encoding="utf8")
    (work / "exclude.txt").write_text("\n".join(map(str, exclude)), encoding="utf8")
    if len(calib_files) < 50:
        warnings.append(f"Only {len(calib_files)} calibration pictures: 100 or more is better.")
    if data[0] is None:
        warnings.append("No test pictures with labels were given, so accuracy (mAP) was NOT measured; only size, speed and memory.")
    elif not data[2]:
        warnings.append("The test folder has pictures but no label files (.txt), so accuracy cannot be measured.")

    # ---- step 1: export
    if not (work / "layout.json").exists():
        code = run_stage("Step 1 of 4: reading the .pt and exporting it to ONNX (about 1 minute)",
                         [py["export"], HERE / "lib" / "step1_export.py", "--pt", pt, "--work", work, "--imgsz", args.imgsz], log, env)
        if code:
            return fail(1, code)
    else:
        say("\n--- Step 1 of 4: already done (use --force to redo)")
    layout = json.loads((work / "layout.json").read_text())

    # ---- step 2: TFLite conversion
    info = models / "models.json"
    done = info.exists() and all(e.get("exists") for e in json.loads(info.read_text())["models"])
    if not done:
        code = run_stage(f"Step 2 of 4: converting to TFLite and quantizing ({len(calib_files)} calibration pictures; this is the long step: roughly 20-60 minutes)",
                         [py["tf"], HERE / "lib" / "step2_tf_convert.py", "--work", work, "--out", models, "--calib-list", work / "calib_images.txt", "--n-calib", args.n_calib], log, env)
        if code and not info.exists():
            return fail(2, code)
        if code:
            warnings.append("Some variants could not be built (see the list below); the others were measured.")
    else:
        say("\n--- Step 2 of 4: already done (use --force to redo)")

    # ---- step 3: benchmark
    cmd = [py["bench"], HERE / "lib" / "step3_benchmark.py", "--work", work, "--models", models, "--max-images", args.max_test, "--rounds", args.rounds]
    if data[0] is not None and data[2]:
        cmd += ["--data", data[0], "--exclude", work / "exclude.txt"]
    if args.skip_speed:
        cmd += ["--skip-speed"]
    code = run_stage("Step 3 of 4: measuring accuracy, speed and memory (several minutes)", cmd, log, env)
    if code:
        return fail(3, code)

    # ---- step 4: report
    (rd / "run_info.json").write_text(json.dumps({"pt": pt.name, "warnings": warnings, "settings": vars(args)}, indent=1, default=str), encoding="utf8")
    code = run_stage("Step 4 of 4: writing the report", [sys.executable, HERE / "lib" / "step4_report.py", "--result-dir", rd], log, env)
    if code:
        return fail(4, code)
    say(f"\nDONE: {rd / 'report.html'}")
    return True


def write_index():
    rows = []
    for rd in sorted((HERE / "results").glob("*/")):
        b = rd / "benchmark.json"
        if not b.exists():
            rows.append(f"<tr><td>{rd.name}</td><td colspan=3>not finished (see {rd.name}/ERROR.txt)</td></tr>")
            continue
        res = json.loads(b.read_text())["results"]
        fp = next((r for r in res if r["design"] == "fp32"), {})
        bests = [r for r in res if r["design"] != "fp32" and r.get("speedup") and r.get("retained_pct")]
        best = max([r for r in bests if r["retained_pct"] >= 95] or bests, key=lambda r: r["speedup"], default=None)
        text = f"{best['file']}: {best['retained_pct']:.1f}% accuracy kept, {best['speedup']:.2f}x faster" if best else "-"
        rows.append(f"<tr><td><a href='{rd.name}/report.html'>{rd.name}</a></td><td>{fp.get('size_mb', 0):.2f}</td><td>{fp.get('mAP50-95', float('nan')):.4f}</td><td>{text}</td></tr>")
    (HERE / "results" / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Results</title><body style='font:16px Arial;margin:30px'><h1>Results</h1><table cellpadding=6 border=0>"
                                                  "<tr><th align=left>Model</th><th>FP32 MB</th><th>FP32 mAP</th><th align=left>Best INT8 variant</th></tr>" + "".join(rows) + "</table></body>", encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_calib, args.max_test, args.rounds = 100, 200, 3

    import check_setup
    ok, py = check_setup.main(quiet=True)
    if not ok:
        say("This computer is not ready yet. Here is what is missing:\n")
        check_setup.main(quiet=False)
        sys.exit(1)
    say(f"Python for export: {py['export']}\nPython for TFLite conversion: {py['tf']}\nPython for measuring: {py['bench']}")
    pts = sorted((HERE / "put_pt_files_here").glob("*.pt"))
    if args.only:
        pts = [p for p in pts if p.stem == args.only]
    if not pts:
        say("\nNo .pt file found in the folder  put_pt_files_here . Copy the model file(s) there and start again.")
        sys.exit(1)
    data = find_dataset(HERE / "put_test_pictures_here")
    say(f"\nModels: {', '.join(p.name for p in pts)}\nTest set: " + (f"{data[1]} pictures ({'with labels' if data[2] else 'NO labels'})" if data[0] else "none (accuracy will not be measured)"))
    results = {}
    for pt in pts:
        try:
            results[pt.name] = process(pt, args, py, data, HERE / "put_calibration_pictures_here", [])
        except Exception:
            say("\nUNEXPECTED ERROR (details saved to ERROR.txt):\n" + traceback.format_exc())
            rd = HERE / "results" / pt.stem
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf8")
            results[pt.name] = False
    write_index()
    say("\n" + "=" * 78 + "\nSummary")
    for k, v in results.items():
        say(f"  {k}: {'done' if v else 'FAILED (see results/' + Path(k).stem + '/ERROR.txt)'}")
    say(f"Open  {HERE / 'results' / 'index.html'}  for the list of reports.")
    sys.exit(0 if all(results.values()) else 1)
