"""convert.py  -  turns a YOLOv5 .pt file into TFLite models.  (You normally do not run this by hand: run.py calls it.)

    python convert.py export --pt put_pt_files_here/abcd123.pt --work results/abcd123/work --imgsz 640
    python convert.py tflite --work results/abcd123/work --out results/abcd123/models --calib-list <text file with picture paths>

THE BIG PICTURE
===============
    abcd123.pt  --export-->  ONNX  --onnx2tf-->  TensorFlow SavedModel  --TFLite converter-->  .tflite files

  "export"  (needs PyTorch)      reads the .pt, writes two ONNX files and layout.json
  "tflite"  (needs TensorFlow)   writes the FP32 baseline and the six INT8 variants:

        abcd123_fp32.tflite             FP32 baseline: no quantization (all numbers are 32-bit floats)
        abcd123_m1_pc / _pt.tflite      M1: every layer INT8
        abcd123_m2_pc / _pt.tflite      M2: backbone INT8, neck + head stay float
        abcd123_m3_pc / _pt.tflite      M3: backbone + neck INT8, head stays float

        pc = per-channel quantization (one INT8 scale per output channel of a convolution; more accurate)
        pt = per-tensor quantization  (one INT8 scale per convolution layer)
        The weights of the Detect head are never changed in any variant.

WHY THE FP32 BASELINE IS BIGGER THAN THE .pt FILE
  YOLOv5 saves .pt files in half precision (16-bit), so a YOLOv5n .pt is about 4 MB. The FP32 baseline stores every
  number in 32 bits, so it is about 8 MB. This is normal. The "same model, other format" comparison is FP32 vs INT8.

WHAT A YOLOv5 NETWORK LOOKS LIKE (the cut points)
  layers 0-9    backbone   finds features in the picture. Layers 4, 6 and 9 are handed on to the neck.
  layers 10-23  neck       mixes the features of the three scales. Layers 17, 20 and 23 are handed to the head.
  layer 24      Detect     the head: turns the three feature maps into boxes, objectness and class scores.
  These numbers are the same for the n, s, m, l and x sizes, but they are read from the model anyway (function layout()).
  M2 = INT8 up to the backbone outputs, M3 = INT8 up to the neck outputs.

WHY THE BOXES ARE SCALED TO 0-1 INSIDE THE MODEL
  Detect outputs [x, y, w, h, objectness, class scores]. Boxes are in pixels (0..640) but scores are 0..1. A fully INT8
  model (M1) stores the whole output tensor with ONE scale. With pixel boxes that scale is about 640/127 = 5 pixels per
  step, so every score (0..1) rounds to 0 and the model detects nothing (mAP 0). Dividing the box part by the input size
  makes everything 0..1 so one scale fits all. In export() this is done by dividing Detect.stride by the picture size
  (the grid step and the anchors are both multiplied by the stride, so boxes come out as fractions of the picture).
  benchmark.py multiplies the boxes by the picture size again when it reads the output.

HOW INT8 QUANTIZATION WORKS HERE
  TensorFlow Lite's converter looks at a few hundred real pictures ("calibration pictures"), records the smallest and
  largest value seen at every layer ("min-max calibration") and then stores weights and activations as 8-bit integers
  using those ranges. Pictures that look like the real data matter: odd pictures widen the ranges and cost accuracy.

HOW M2 AND M3 KEEP SOME LAYERS FLOAT INSIDE ONE FILE
  The converter has a "quantization debugger" with a deny-list: operations on the list stay float. We put every
  operation that is NOT part of the backbone (M2) or backbone + neck (M3) on the list. onnx2tf renames all layers, so the
  operations cannot be found by name. Instead (function region_names()):
    1. the ONNX graph is cut at the boundary layers into extra parts (backbone.onnx, backbone_neck.onnx),
    2. each part is converted to TFLite too and run on one picture,
    3. its output values are matched to a tensor inside the full converted graph (the tensor with the same numbers),
    4. every operation that is an ancestor of those tensors is quantized, everything after them stays float.
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
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor_yolov5"          # a copy of the YOLOv5 v7.0 code, used for reading the .pt and for the ONNX export
sys.path.insert(0, str(VENDOR))


def say(msg):
    print(msg, flush=True)


def fail(msg):
    print("ERROR:", msg, flush=True)
    sys.exit(2)


# =====================================================================================================================
# Pictures (the same three helpers exist in benchmark.py; kept here so this file stands alone)
# =====================================================================================================================
def read_image(path):
    """Read a picture as BGR; None if it cannot be read. (Works with non-ASCII paths.)"""
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def letterbox(img, size, color=114):
    """Resize keeping the aspect ratio, then pad with grey to size x size (this is how YOLOv5 prepares pictures)."""
    import cv2
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def to_input_nhwc(letterboxed_bgr):
    """What a TFLite model takes: float32, RGB, values 0-1, shape 1 x size x size x 3 (TensorFlow uses channels-last)."""
    return np.ascontiguousarray(letterboxed_bgr[:, :, ::-1][None], dtype=np.float32) / 255.0


# =====================================================================================================================
# STEP 1: export  (.pt -> ONNX)
# =====================================================================================================================
def load_yolov5(pt):
    """Open the checkpoint and return the model. Gives a plain-words error if it is not a YOLOv5 file."""
    import torch
    try:
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        if "ultralytics" in str(e):
            fail("This looks like an Ultralytics YOLOv8 / YOLO11 file. This tool converts YOLOv5 files only "
                 "(anchor-based, trained with the YOLOv5 repository). The newer families have a different head.")
        fail(f"Could not read the file: {e}")
    except Exception as e:
        fail(f"Could not read the file as a PyTorch checkpoint: {type(e).__name__}: {e}")
    model = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else None
    if model is None:
        fail("The file has no 'model' or 'ema' entry. Expected a YOLOv5 checkpoint saved by the YOLOv5 training script.")
    model = model.float().eval()
    if type(model).__name__ != "DetectionModel" or type(model.model[-1]).__name__ != "Detect":
        fail(f"Expected a YOLOv5 DetectionModel ending in a Detect layer, found {type(model).__name__} "
             f"ending in {type(model.model[-1]).__name__}.")
    return model


def layout(model):
    """The cut points (see the top of this file).
    B = backbone layers whose output is used by the neck/head part (read from the 'from' lists of the later layers).
    N = the layers that feed the Detect head."""
    n_backbone = len(model.yaml["backbone"])
    B = set()
    for layer in model.model:                      # layer.i = its number, layer.f = where its input comes from
        if layer.i >= n_backbone:
            for src in ([layer.f] if isinstance(layer.f, int) else layer.f):
                src = layer.i - 1 if src == -1 else src
                if src < n_backbone:
                    B.add(src)
    return sorted(B), list(model.model[-1].f)


def export_onnx(weights, imgsz):
    """Run YOLOv5's own export.py (opset 13) in the bundled vendor_yolov5 folder; it writes <weights>.onnx next to the weights."""
    r = subprocess.run([sys.executable, "export.py", "--weights", str(weights), "--include", "onnx", "--opset", "13",
                        "--imgsz", str(imgsz), "--device", "cpu"],
                       cwd=VENDOR, capture_output=True, text=True, encoding="utf8", errors="replace")
    out = Path(weights).with_suffix(".onnx")
    if r.returncode or not out.exists():
        fail("ONNX export failed:\n" + (r.stderr or r.stdout)[-1500:])
    return out


def step_export(a):
    import torch
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pt = Path(a.pt).resolve()
    name = pt.stem
    model = load_yolov5(pt)
    det = model.model[-1]
    nc = int(det.nc)
    names = model.names if isinstance(model.names, list) else [model.names[k] for k in sorted(model.names)]
    B, N = layout(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    say(f"{name}: YOLOv5 model, {nc} classes, strides {det.stride.tolist()}, {params:.2f} M parameters")
    say(f"cut points: backbone outputs {B}, neck outputs {N}")
    if a.imgsz % int(det.stride.max()):
        fail(f"The input size {a.imgsz} must be a multiple of the largest stride ({int(det.stride.max())}).")

    # Two exports: "pixel" = normal YOLOv5 (boxes in pixels, only used to check the other one),
    # "norm" = boxes as fractions of the picture (0-1), used for everything. Saved in float32 on purpose:
    # a half-precision stride would blur the 0-1 box scale.
    pixel_pt, norm_pt = work / f"{name}_pixel.pt", work / f"{name}_norm.pt"
    torch.save({"model": model, "ema": None, "epoch": -1}, pixel_pt)
    det.stride = det.stride / a.imgsz
    torch.save({"model": model, "ema": None, "epoch": -1}, norm_pt)
    for p in (pixel_pt, norm_pt):
        export_onnx(p, a.imgsz)

    # Check: boxes of the norm model x size must equal the boxes of the pixel model, scores must be equal.
    import onnxruntime as ort
    x = np.clip(np.random.RandomState(0).rand(1, 3, a.imgsz, a.imgsz).astype(np.float32) * 0.6 + 0.2, 0, 1)
    ya = ort.InferenceSession(str(pixel_pt.with_suffix(".onnx")), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    yb = ort.InferenceSession(str(norm_pt.with_suffix(".onnx")), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    err = float(np.abs(yb[..., :4] * a.imgsz - ya[..., :4]).max())
    err_rest = float(np.abs(yb[..., 4:] - ya[..., 4:]).max())
    say(f"check: boxes x {a.imgsz} match the pixel-box model within {err:.2e} px; scores within {err_rest:.2e}; output shape {tuple(yb.shape)}")
    if err > 0.05 or err_rest > 1e-4 or yb.shape[-1] != nc + 5:
        fail("The normalised export does not match the pixel export; stopping.")
    json.dump({"name": name, "nc": nc, "names": names, "imgsz": a.imgsz, "B": B, "N": N, "params_million": params,
               "onnx_norm": norm_pt.with_suffix(".onnx").name, "output_shape": list(yb.shape)},
              open(work / "layout.json", "w"), indent=1)
    say("step 1 done")


# =====================================================================================================================
# STEP 2: ONNX -> TensorFlow -> TFLite (FP32 baseline + INT8 variants)
# =====================================================================================================================
def layer_output(model, layer):
    """Name of the output tensor of the last ONNX node that belongs to layer N (nodes are named /model.N/...)."""
    last = None
    for n in model.graph.node:
        if n.name.startswith(f"/model.{layer}/"):
            last = n
    if last is None:
        raise RuntimeError(f"layer {layer} not found in the ONNX graph")
    return last.output[0]


def make_parts(onnx_path, lay, parts_dir):
    """Cut the ONNX graph: backbone.onnx (picture -> backbone outputs) and backbone_neck.onnx (picture -> neck outputs).
    They are only used to find out which operations of the full model belong to the backbone / neck."""
    import onnx
    import onnxruntime as ort
    from onnx.utils import extract_model
    parts_dir.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(onnx_path))
    B = [layer_output(m, i) for i in lay["B"]]
    N = [layer_output(m, i) for i in lay["N"]]
    shutil.copy(onnx_path, parts_dir / "full.onnx")
    extract_model(str(onnx_path), str(parts_dir / "backbone.onnx"), ["images"], B)
    extract_model(str(onnx_path), str(parts_dir / "backbone_neck.onnx"), ["images"], N)
    x = np.random.RandomState(0).rand(1, 3, lay["imgsz"], lay["imgsz"]).astype(np.float32)
    n_out = ort.InferenceSession(str(parts_dir / "backbone_neck.onnx"), providers=["CPUExecutionProvider"]).run(None, {"images": x})
    assert len(n_out) == len(N), "unexpected number of neck outputs"
    say(f"   cut the graph at layers {lay['B']} (backbone) and {lay['N']} (neck)")


def run_onnx2tf(onnx_file, out_dir, cwd):
    """ONNX -> TensorFlow SavedModel + float TFLite (onnx2tf). onnx2tf wants a small sample file in the working
    directory (otherwise it tries to download one), so a placeholder is written first: no internet is needed."""
    exe = Path(sys.executable).parent / ("onnx2tf.exe" if os.name == "nt" else "onnx2tf")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "onnx2tf"]
    np.save(cwd / "calibration_image_sample_data_20x128x128x3_float32.npy",
            np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # onnx2tf starts the helper program "onnxsim", which sits next to this Python: put that folder on PATH.
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run(cmd + ["-i", str(onnx_file), "-o", str(out_dir), "-osd"], cwd=cwd, capture_output=True, text=True,
                       encoding="utf8", errors="replace", env=env)
    (cwd / f"onnx2tf_{onnx_file.stem}.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf8")
    if r.returncode or not (out_dir / f"{onnx_file.stem}_float32.tflite").exists():
        raise RuntimeError(f"onnx2tf failed on {onnx_file.name} (see {cwd / f'onnx2tf_{onnx_file.stem}.log'}): "
                           f"{((r.stderr or '') + (r.stdout or ''))[-600:]}")


def representative_dataset(files, size):
    """The calibration pictures as the converter wants them: a function that yields [one float32 NHWC picture] at a time."""
    def gen():
        for f in files:
            img = read_image(f)
            if img is not None:
                yield [to_input_nhwc(letterbox(img, size))]
    return gen


def run_float(model_bytes, x_nhwc, preserve=False):
    """Run a float TFLite model once. preserve=True keeps every inner tensor so that it can be read afterwards."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=model_bytes, experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x_nhwc)
    it.invoke()
    return it


def region_names(work, calib_files, size):
    """{design: [names of the tensors whose operations must stay float]}, found from the graph structure (see top of file).
    The result is cached in float_regions.json."""
    cache = work / "float_regions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import tensorflow as tf
    saved = work / "tf" / "full" / "saved"
    x = to_input_nhwc(letterbox(read_image(calib_files[0]), size))
    # The full model as the TFLite converter itself makes it from the SavedModel. It MUST be this one (not onnx2tf's own float file):
    # the INT8 conversion starts from the same SavedModel, so only its tensor names match the deny-list. Every inner tensor is kept
    # so that it can be compared with the parts' outputs.
    fcache = work / "float_analysis.tflite"
    if not fcache.exists():
        fcache.write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(saved)).convert())
    full_bytes = fcache.read_bytes()
    full = run_float(full_bytes, x, preserve=True)
    details = {d["index"]: d for d in full.get_tensor_details()}
    ops = full._get_ops_details()
    producer = {t: op["index"] for op in ops for t in op["outputs"]}        # tensor -> the operation that makes it
    by_index = {op["index"]: op for op in ops}

    def boundary(part):
        """Find, inside the full model, the tensors that equal the outputs of a separately converted part."""
        sub = run_float((work / "tf" / part / "saved" / f"{part}_float32.tflite").read_bytes(), x)
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
        """All operations that the given tensors depend on."""
        seen, stack = set(), list(tensors)
        while stack:
            t = stack.pop()
            if t in producer and producer[t] not in seen:
                seen.add(producer[t])
                stack.extend(by_index[producer[t]]["inputs"])
        return seen

    out = {}
    for design, part in (("m2", "backbone"), ("m3", "backbone_neck")):
        quantized = ancestors(boundary(part))
        float_ops = [op for op in ops if op["index"] not in quantized]
        out[design] = sorted({details[t]["name"] for op in float_ops for t in op["outputs"]})
        say(f"   {design}: {len(quantized)} operations get quantized, {len(float_ops)} stay float")
    cache.write_text(json.dumps(out, indent=1))
    return out


def build_int8(saved, design, per_channel, calib_files, size, deny):
    """One INT8 model from the SavedModel. Input and output stay float32 (the model converts inside), so it can be
    used like the baseline."""
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(calib_files, size)       # min-max calibration happens here
    if not per_channel:
        conv._experimental_disable_per_channel = True                              # one scale per layer instead of per channel
    if design == "m1":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]     # strict: any non-INT8 operation fails loudly
        return conv.convert()
    # M2 / M3: same, but the operations on the deny-list are left in float
    dbg = tf.lite.experimental.QuantizationDebugger(
        converter=conv, debug_dataset=representative_dataset(calib_files[:1], size),
        debug_options=tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=deny))
    return dbg.get_nondebug_quantized_model()


def agreement(model_a, model_b, calib_file, size):
    """Quick sanity check of a finished model: how well do its outputs (objectness + class scores) follow the FP32 baseline's
    on one picture? 1.00 = identical. A value far below 0.9 means the conversion damaged the model."""
    import tensorflow as tf
    x = to_input_nhwc(letterbox(read_image(calib_file), size))
    outs = []
    for m in (model_a, model_b):
        it = tf.lite.Interpreter(model_path=str(m))
        it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], x)
        it.invoke()
        outs.append(it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1, it.get_output_details()[0]["shape"][-1])[:, 4:].ravel())
    return float(np.corrcoef(outs[0], outs[1])[0, 1])


def step_tflite(a):
    work, out = Path(a.work).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lay = json.loads((work / "layout.json").read_text())
    size, name = lay["imgsz"], lay["name"]
    calib = [Path(l.strip()) for l in Path(a.calib_list).read_text(encoding="utf8").splitlines() if l.strip()][:a.n_calib]
    if not calib:
        fail("no calibration pictures")
    t0 = time.time()
    tfdir = work / "tf"
    parts = work / "onnx_parts"

    say("2a. cutting the network into parts (used to find which layers stay float)")
    if not (parts / "backbone_neck.onnx").exists():
        make_parts(work / lay["onnx_norm"], lay, parts)

    say("2b. converting to TensorFlow with onnx2tf (a few minutes)")
    for part in ("full", "backbone", "backbone_neck"):
        d = tfdir / part
        if not (d / "saved" / f"{part}_float32.tflite").exists():
            d.mkdir(parents=True, exist_ok=True)
            run_onnx2tf(parts / f"{part}.onnx", d / "saved", d)
            say(f"   {part} done ({time.time() - t0:.0f} s)")
    saved_full = tfdir / "full" / "saved"

    say("2c. FP32 baseline")
    base = out / f"{name}_fp32.tflite"
    if not base.exists():
        shutil.copy(saved_full / "full_float32.tflite", base)
    entries = [{"file": base.name, "design": "fp32", "granularity": "-", "label": "FP32 baseline"}]

    say(f"2d. INT8 variants with {len(calib)} calibration pictures (several minutes each)")
    deny = region_names(work, calib, size)
    LABEL = {"m1": "M1 all INT8", "m2": "M2 backbone INT8, neck and head float", "m3": "M3 backbone + neck INT8, head float"}
    status, agree = {}, {}
    for design in ("m1", "m2", "m3"):
        for g, per_channel in (("pc", True), ("pt", False)):
            f = out / f"{name}_{design}_{g}.tflite"
            entries.append({"file": f.name, "design": design, "granularity": "per-channel" if per_channel else "per-tensor", "label": LABEL[design]})
            if f.exists():
                continue                                    # finished earlier: an interrupted run continues where it stopped
            try:
                f.write_bytes(build_int8(saved_full, design, per_channel, calib, size, deny.get(design, [])))
                say(f"   {f.name}: {f.stat().st_size / 1e6:.2f} MB ({time.time() - t0:.0f} s)")
            except Exception as e:
                status[f.name] = f"{type(e).__name__}: {str(e)[:300]}"
                say(f"   {f.name}: FAILED ({status[f.name]})")
    for e in entries:
        p = out / e["file"]
        e["exists"] = p.exists()
        e["size_mb"] = p.stat().st_size / 1e6 if p.exists() else None
        e["error"] = status.get(e["file"])
        if p.exists() and e["design"] != "fp32":
            try:
                e["agreement_with_fp32"] = agreement(base, p, calib[0], size)
                say(f"   check {p.name}: outputs follow the FP32 baseline with correlation {e['agreement_with_fp32']:.3f}")
            except Exception as ex:
                e["agreement_with_fp32"] = None
                say(f"   check {p.name}: could not be run ({type(ex).__name__}: {str(ex)[:150]})")
    # M2 / M3 keep float layers, so they must be clearly bigger than M1; if not, the deny-list did not work and they are really M1 again.
    size_of = {(e["design"], e["granularity"]): e["size_mb"] for e in entries}
    for e in entries:
        m1 = size_of.get(("m1", e["granularity"]))
        if e["design"] in ("m2", "m3") and e["size_mb"] and m1 and e["size_mb"] < m1 * 1.05:
            e["warning"] = f"{e['file']} is not bigger than M1 ({e['size_mb']:.2f} MB vs {m1:.2f} MB): the float layers were probably NOT kept."
            say("   WARNING: " + e["warning"])
    json.dump({"name": name, "n_calibration": len(calib), "models": entries}, open(out / "models.json", "w"), indent=1)
    say(f"step 2 done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(0 if all(e["exists"] for e in entries) else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    e = sub.add_parser("export")
    e.add_argument("--pt", required=True)
    e.add_argument("--work", required=True)
    e.add_argument("--imgsz", type=int, default=640)
    t = sub.add_parser("tflite")
    t.add_argument("--work", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--calib-list", required=True, help="text file with one calibration picture path per line")
    t.add_argument("--n-calib", type=int, default=300)
    args = ap.parse_args()
    step_export(args) if args.step == "export" else step_tflite(args)
