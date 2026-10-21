"""Step 2 (the Python that has TensorFlow + onnx2tf): ONNX -> TFLite FP32 baseline and the six INT8 variants, one .tflite file each.

    python lib/step2_tf_convert.py --work results/abcd123/work --out results/abcd123/models --calib-list results/abcd123/work/calib_images.txt --n-calib 300

Designs (the Detect head's weights are never changed):
    m1  everything INT8                     m2  backbone INT8, neck + head float          m3  backbone + neck INT8, head float
each as  pc = per-channel weights ("channel-wise")   and   pt = per-tensor weights ("layer-wise").

How: the network is converted with onnx2tf, then quantized with the TensorFlow Lite converter (full integer, min-max calibration on the calibration
pictures). For M2 and M3 the layers that must stay float are excluded with the quantization debugger's denylist, so every variant stays ONE file.
The converted graph has lost the layer names, so the excluded operations are found from the structure: everything that is not an ancestor of the
backbone outputs (M2) or of the neck outputs (M3); the boundary tensors are located by comparing values with separately converted parts.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import yolo_io  # noqa: E402


def say(msg):
    print(msg, flush=True)


def layer_output(model, layer):
    """Output tensor of the last node (in graph order) that belongs to layer N."""
    last = None
    for n in model.graph.node:
        if n.name.startswith(f"/model.{layer}/"):
            last = n
    if last is None:
        raise RuntimeError(f"layer {layer} not found in the ONNX graph")
    return last.output[0]


def make_parts(onnx_path, layout, parts_dir):
    """Cut the ONNX graph at the backbone / neck boundaries: backbone (images -> backbone outputs) and backbone_neck (images -> neck outputs)."""
    import onnx
    import onnxruntime as ort
    from onnx.utils import extract_model
    parts_dir.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(onnx_path))
    B = [layer_output(m, i) for i in layout["B"]]
    N = [layer_output(m, i) for i in layout["N"]]
    shutil.copy(onnx_path, parts_dir / "full.onnx")
    extract_model(str(onnx_path), str(parts_dir / "backbone.onnx"), ["images"], B)
    extract_model(str(onnx_path), str(parts_dir / "backbone_neck.onnx"), ["images"], N)
    # check: the parts give the same boundary tensors as the full graph
    full = ort.InferenceSession(str(parts_dir / "full.onnx"), providers=["CPUExecutionProvider"])
    s = layout["imgsz"]
    x = np.random.RandomState(0).rand(1, 3, s, s).astype(np.float32)
    ref = full.run(None, {"images": x})[0]
    n_out = ort.InferenceSession(str(parts_dir / "backbone_neck.onnx"), providers=["CPUExecutionProvider"]).run(None, {"images": x})
    assert len(n_out) == len(N), "unexpected number of neck outputs"
    say(f"   cut the graph at layers {layout['B']} (backbone) and {layout['N']} (neck); output shape of the full graph {ref.shape}")


def run_onnx2tf(onnx_file, out_dir, cwd):
    """Convert one ONNX file to a TensorFlow SavedModel + float TFLite files with onnx2tf (needs no internet: a local placeholder file is provided)."""
    exe = Path(sys.executable).parent / ("onnx2tf.exe" if os.name == "nt" else "onnx2tf")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "onnx2tf"]
    np.save(cwd / "calibration_image_sample_data_20x128x128x3_float32.npy", np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))   # onnx2tf starts onnxsim, which lives next to this Python
    r = subprocess.run(cmd + ["-i", str(onnx_file), "-o", str(out_dir), "-osd"], cwd=cwd, capture_output=True, text=True, encoding="utf8", errors="replace", env=env)
    (cwd / f"onnx2tf_{onnx_file.stem}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf8")
    if r.returncode or not (out_dir / f"{onnx_file.stem}_float32.tflite").exists():
        raise RuntimeError(f"onnx2tf failed on {onnx_file.name} (see {cwd / f'onnx2tf_{onnx_file.stem}.log'}): {(r.stderr or r.stdout)[-600:]}")


def representative_dataset(files, size):
    def gen():
        for f in files:
            img = yolo_io.read_image(f)
            if img is not None:
                x, _, _ = yolo_io.letterbox(img, size)
                yield [yolo_io.to_input(x).transpose(0, 2, 3, 1).astype(np.float32)]            # NHWC
    return gen


def float_model_bytes(saved, cache):
    import tensorflow as tf
    if not cache.exists():
        cache.write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(saved)).convert())
    return cache.read_bytes()


def run_float(model_bytes, x_nhwc, preserve=False):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=model_bytes, experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x_nhwc)
    it.invoke()
    return it


def region_names(work, calib_files, size):
    """{design: [tensor names of the operations that must stay float]} found from the graph structure (cached)."""
    cache = work / "float_regions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    saved = work / "tf" / "full" / "saved"
    img = yolo_io.read_image(calib_files[0])
    x = yolo_io.to_input(yolo_io.letterbox(img, size)[0]).transpose(0, 2, 3, 1).astype(np.float32)
    full = run_float(float_model_bytes(saved, work / "float_analysis.tflite"), x, preserve=True)
    details = {d["index"]: d for d in full.get_tensor_details()}
    ops = full._get_ops_details()
    producer = {t: op["index"] for op in ops for t in op["outputs"]}
    by_index = {op["index"]: op for op in ops}

    def boundary(part):
        m = (work / "tf" / part / "saved" / f"{part}_float32.tflite").read_bytes()
        sub = run_float(m, x)
        found = []
        for o in sub.get_output_details():
            v = sub.get_tensor(o["index"])
            cand = [(float(np.abs(full.get_tensor(i) - v).max()), i) for i, d in details.items()
                    if tuple(d["shape"]) == v.shape and i in producer and d["dtype"] == np.float32]
            err, idx = min(cand)
            if err > 1e-2:
                raise RuntimeError(f"could not locate a boundary tensor of {part} in the converted graph (closest differs by {err:.3g})")
            found.append(idx)
        return found

    def ancestors(tensors):
        seen, stack = set(), list(tensors)
        while stack:
            t = stack.pop()
            if t in producer and producer[t] not in seen:
                seen.add(producer[t])
                stack.extend(by_index[producer[t]]["inputs"])
        return seen

    out = {}
    for design, part in (("m2", "backbone"), ("m3", "backbone_neck")):
        keep_int8 = ancestors(boundary(part))
        float_ops = [op for op in ops if op["index"] not in keep_int8]
        out[design] = sorted({details[t]["name"] for op in float_ops for t in op["outputs"]})
        say(f"   {design}: {len(keep_int8)} operations get quantized, {len(float_ops)} stay float")
    cache.write_text(json.dumps(out, indent=1))
    return out


def build_int8(saved, design, per_channel, calib_files, size, deny):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(calib_files, size)
    if not per_channel:
        conv._experimental_disable_per_channel = True                       # one scale per tensor ("layer-wise")
    if design == "m1":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]      # strict: any non-INT8 operation fails loudly
        return conv.convert()
    dbg = tf.lite.experimental.QuantizationDebugger(converter=conv, debug_dataset=representative_dataset(calib_files[:1], size),
                                                    debug_options=tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=deny))
    return dbg.get_nondebug_quantized_model()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-list", required=True, help="text file with one calibration picture path per line")
    ap.add_argument("--n-calib", type=int, default=300)
    a = ap.parse_args()
    work, out = Path(a.work).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    layout = json.loads((work / "layout.json").read_text())
    size, name = layout["imgsz"], layout["name"]
    calib = [Path(l.strip()) for l in Path(a.calib_list).read_text(encoding="utf8").splitlines() if l.strip()][:a.n_calib]
    if not calib:
        say("ERROR: no calibration pictures")
        sys.exit(2)
    t0 = time.time()
    tfdir = work / "tf"
    onnx_norm = work / layout["onnx_norm"]

    say("2a. cutting the network into parts (used to find which layers stay float)")
    parts = work / "onnx_parts"
    if not (parts / "backbone_neck.onnx").exists():
        make_parts(onnx_norm, layout, parts)
    say("2b. converting to TensorFlow with onnx2tf (a few minutes)")
    for part in ("full", "backbone", "backbone_neck"):
        d = tfdir / part
        if not (d / "saved" / f"{part}_float32.tflite").exists():
            d.mkdir(parents=True, exist_ok=True)
            run_onnx2tf(parts / f"{part}.onnx", d / "saved", d)
            say(f"   {part} done ({time.time() - t0:.0f} s)")
    saved_full = tfdir / "full" / "saved"

    say("2c. FP32 baseline")
    dst = out / f"{name}_fp32.tflite"
    if not dst.exists():
        shutil.copy(saved_full / "full_float32.tflite", dst)
    entries = [{"file": dst.name, "design": "fp32", "granularity": "-", "label": "FP32 baseline"}]

    say(f"2d. INT8 variants with {len(calib)} calibration pictures (several minutes each)")
    deny = region_names(work, calib, size)
    LABEL = {"m1": "M1 all INT8", "m2": "M2 backbone INT8, neck and head float", "m3": "M3 backbone + neck INT8, head float"}
    status = {}
    for design in ("m1", "m2", "m3"):
        for g, pc in (("pc", True), ("pt", False)):
            f = out / f"{name}_{design}_{g}.tflite"
            entries.append({"file": f.name, "design": design, "granularity": "per-channel" if pc else "per-tensor", "label": LABEL[design]})
            if f.exists():
                continue
            try:
                f.write_bytes(build_int8(saved_full, design, pc, calib, size, deny.get(design, [])))
                say(f"   {f.name}: {f.stat().st_size / 1e6:.2f} MB ({time.time() - t0:.0f} s)")
            except Exception as e:
                status[f.name] = f"{type(e).__name__}: {str(e)[:300]}"
                say(f"   {f.name}: FAILED ({status[f.name]})")
    for e in entries:
        e["exists"] = (out / e["file"]).exists()
        e["size_mb"] = (out / e["file"]).stat().st_size / 1e6 if e["exists"] else None
        e["error"] = status.get(e["file"])
    json.dump({"name": name, "n_calibration": len(calib), "models": entries}, open(out / "models.json", "w"), indent=1)
    say(f"step 2 done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(0 if all(e["exists"] for e in entries) else 3)
