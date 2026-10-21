"""Checks that this computer is ready (does not convert anything). Run:   python check_setup.py     (or double-click CHECK_SETUP.bat / run ./CHECK_SETUP.sh)

It finds which Python can do each job, lists what is missing with the exact offline install command, and shows what it found in the input folders.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import envfind  # noqa: E402

JOBS = {"export": "read the .pt file and export it to ONNX", "tf": "convert to TFLite and quantize to INT8", "bench": "measure accuracy, speed and memory"}


def count_pictures(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in Path(folder).rglob("*") if p.suffix.lower() in exts) if Path(folder).exists() else 0


def inputs():
    pts = sorted((HERE / "put_pt_files_here").glob("*.pt"))
    test = HERE / "put_test_pictures_here"
    calib = HERE / "put_calibration_pictures_here"
    return pts, test, calib


def main(quiet=False):
    p = (lambda *a: None) if quiet else print
    ok_all = True
    p("=" * 78)
    p("Setup check")
    p("=" * 78)
    p(f"Python running this check: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        p("  note: the tool was built with Python 3.11. Other versions may work if all packages install; 3.11 is the safe choice.")
    found = {}
    for role, what in JOBS.items():
        py, report = envfind.find(role)
        found[role] = py
        if py:
            p(f"[ OK ] {role:<7} {what}\n         uses: {py}")
        else:
            ok_all = False
            p(f"[ !! ] {role:<7} {what}: NOT READY")
            for cand, missing, extra, res in report[:3]:
                p(f"         tried {cand}")
                if missing:
                    p("         missing:")
                    for line in envfind.advice(missing):
                        p("      " + line)
                if extra:
                    p(f"         the bundled YOLOv5 code does not import: {extra}")
            if not report:
                p("         no Python found at all")
    p("")
    pts, test, calib = inputs()
    p(f"Models waiting in put_pt_files_here: {len(pts)}" + ("  (" + ", ".join(x.name for x in pts) + ")" if pts else "  -> copy the .pt file(s) there"))
    nt, nc = count_pictures(test), count_pictures(calib)
    has_yaml = any(test.glob("*.yaml")) or any(test.glob("*.yml")) if test.exists() else False
    p(f"Test pictures (put_test_pictures_here): {nt}" + ("  + data.yaml" if has_yaml else "") + ("" if nt else "  -> without them accuracy is not measured (size, speed and memory still are)"))
    p(f"Calibration pictures (put_calibration_pictures_here): {nc}" + ("" if nc else "  -> empty: part of the test pictures will be used instead"))
    try:
        free = shutil.disk_usage(HERE).free / 1e9
        p(f"Free disk space here: {free:.0f} GB (a conversion needs about 3 GB per model)")
        if free < 5:
            p("  warning: very little free space")
    except Exception:
        pass
    p(f"CPU cores seen by Python: {os.cpu_count()}")
    p("")
    if ok_all:
        p("READY. Start the conversion with RUN.bat (Windows) or ./RUN.sh (Linux).")
    else:
        p("NOT READY. Fix the items marked [ !! ] above, then run this check again.")
        p("If this computer has no internet: see README.md, part 'If a package is missing' (the Linux package folder is included).")
    return ok_all, found


if __name__ == "__main__":
    ok, _ = main()
    sys.exit(0 if ok else 1)
