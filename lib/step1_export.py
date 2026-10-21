"""Step 1 (main Python): read a YOLOv5 .pt file, check that it can be converted, and export it to ONNX at the chosen size with the box output
normalised to 0-1 (the Detect head's weights are not touched).

    python lib/step1_export.py --pt put_pt_files_here/abcd123.pt --work results/abcd123/work --imgsz 640

Writes in --work:  <name>_norm.onnx (boxes 0-1, used for everything), <name>_pixel.onnx (boxes in pixels, only used for the check),
                   layout.json (classes, class names, input size and the layer numbers where the network is cut into backbone / neck / head).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
VENDOR = HERE / "vendor_yolov5"
sys.path.insert(0, str(VENDOR))


def fail(msg):
    print("ERROR:", msg)
    sys.exit(2)


def load_yolov5(pt):
    import torch
    try:
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        if "ultralytics" in str(e):
            fail("This looks like an Ultralytics YOLOv8 / YOLO11 file (it needs the 'ultralytics' package). This tool converts YOLOv5 files "
                 "(anchor-based, trained with the YOLOv5 repository). The newer families have a different head and need a different conversion.")
        fail(f"Could not read the file: {e}")
    except Exception as e:
        fail(f"Could not read the file as a PyTorch checkpoint: {type(e).__name__}: {e}")
    model = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else None
    if model is None:
        fail("The file has no 'model' or 'ema' entry. Expected a YOLOv5 checkpoint saved by the training script (it holds the whole model object).")
    model = model.float().eval()
    if type(model).__name__ != "DetectionModel" or type(model.model[-1]).__name__ != "Detect":
        fail(f"Expected a YOLOv5 DetectionModel ending in a Detect layer, found {type(model).__name__} ending in {type(model.model[-1]).__name__}.")
    return ckpt, model


def layout(model):
    """Layer numbers where the network is cut: B = backbone layers whose output the neck uses, N = the layers feeding the Detect head."""
    nb = len(model.yaml["backbone"])
    B = set()
    for layer in model.model:
        if layer.i >= nb:
            for s in ([layer.f] if isinstance(layer.f, int) else layer.f):
                s = layer.i - 1 if s == -1 else s
                if s < nb:
                    B.add(s)
    return sorted(B), list(model.model[-1].f)


def export_onnx(weights, imgsz):
    r = subprocess.run([sys.executable, "export.py", "--weights", str(weights), "--include", "onnx", "--opset", "13", "--imgsz", str(imgsz), "--device", "cpu"],
                       cwd=VENDOR, capture_output=True, text=True, encoding="utf8", errors="replace")
    out = Path(weights).with_suffix(".onnx")
    if r.returncode or not out.exists():
        fail("ONNX export failed:\n" + (r.stderr or r.stdout)[-1500:])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    import torch
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    a.pt = str(Path(a.pt).resolve())
    name = Path(a.pt).stem
    ckpt, model = load_yolov5(a.pt)
    det = model.model[-1]
    nc = int(det.nc)
    names = model.names if isinstance(model.names, list) else [model.names[k] for k in sorted(model.names)]
    B, N = layout(model)
    print(f"{name}: YOLOv5 model, {nc} classes, {len(det.f)} detection layers, strides {det.stride.tolist()}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    print(f"cut points: backbone outputs {B}, neck outputs {N}")
    if a.imgsz % int(det.stride.max()):
        fail(f"The input size {a.imgsz} must be a multiple of the largest stride ({int(det.stride.max())}).")

    pixel_pt, norm_pt = work / f"{name}_pixel.pt", work / f"{name}_norm.pt"
    torch.save({"model": model, "ema": None, "epoch": -1}, pixel_pt)           # kept in float32: a half-precision stride would blur the 0-1 box scale
    det.stride = det.stride / a.imgsz              # boxes come out as fractions of the picture: the grid step and the anchors are both scaled by the stride
    torch.save({"model": model, "ema": None, "epoch": -1}, norm_pt)
    for pt in (pixel_pt, norm_pt):
        export_onnx(pt, a.imgsz)

    import onnxruntime as ort
    rng = np.random.RandomState(0)
    x = np.clip(rng.rand(1, 3, a.imgsz, a.imgsz).astype(np.float32) * 0.6 + 0.2, 0, 1)
    ya = ort.InferenceSession(str(pixel_pt.with_suffix(".onnx")), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    yb = ort.InferenceSession(str(norm_pt.with_suffix(".onnx")), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    err = float(np.abs(yb[..., :4] * a.imgsz - ya[..., :4]).max())
    err_rest = float(np.abs(yb[..., 4:] - ya[..., 4:]).max())
    print(f"check: boxes x {a.imgsz} match the pixel-box model within {err:.2e} px; scores match within {err_rest:.2e}; output shape {tuple(yb.shape)}")
    if err > 0.05 or err_rest > 1e-4 or yb.shape[-1] != nc + 5:
        fail("The normalised export does not match the pixel export; stopping.")
    json.dump({"name": name, "nc": nc, "names": names, "imgsz": a.imgsz, "B": B, "N": N, "strides": det.stride.mul(a.imgsz).tolist(),
               "params_million": sum(p.numel() for p in model.parameters()) / 1e6, "onnx_norm": str(norm_pt.with_suffix(".onnx").name), "onnx_pixel": str(pixel_pt.with_suffix(".onnx").name),
               "output_shape": list(yb.shape)}, open(work / "layout.json", "w"), indent=1)
    print("step 1 done")
