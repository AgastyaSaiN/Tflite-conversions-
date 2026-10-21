"""Picture preparation, box decoding, dataset reading and scoring shared by the steps of the conversion tool (numpy, OpenCV, pycocotools; no PyTorch needed)."""
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- pictures
def letterbox(img, size, color=114):
    """Resize keeping the aspect ratio and pad to size x size with grey. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def read_image(path):
    """Read a picture (any path, also with non-ASCII characters) as BGR; None if unreadable."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def to_input(letterboxed_bgr):
    """The network input: float32, RGB, 0-1, NCHW (1 x 3 x S x S)."""
    return np.ascontiguousarray(letterboxed_bgr[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


# ----------------------------------------------------------------------------- decoding
def decode(pred, nc, box_scale, conf, iou, r, pad, shape, multi_label=False, max_det=300):
    """pred (N, 5 + nc): x, y, w, h, objectness, class scores -> (M, 6) detections x1, y1, x2, y2, confidence, class in the original picture.
    multi_label=True keeps every class above the threshold per box (as the standard validation does); False keeps the best class only."""
    p = np.ascontiguousarray(pred, dtype=np.float32).reshape(-1, 5 + nc)
    scores = p[:, 5:] * p[:, 4:5]
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
    if sc.size > 30000:
        top = np.argpartition(-sc, 30000)[:30000]
        sc, j, xywh = sc[top], j[top], xywh[top]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1)         # x, y, w, h
    idx = np.array(cv2.dnn.NMSBoxesBatched(boxes.tolist(), sc.tolist(), j.tolist(), conf, iou), dtype=int).reshape(-1)       # NMS per class
    if not idx.size:
        return np.zeros((0, 6), np.float32)
    idx = idx[np.argsort(-sc[idx])][:max_det]
    b = boxes[idx]
    out = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3], sc[idx], j[idx]], 1).astype(np.float32)
    out[:, [0, 2]] = ((out[:, [0, 2]] - pad[0]) / r).clip(0, shape[1])
    out[:, [1, 3]] = ((out[:, [1, 3]] - pad[1]) / r).clip(0, shape[0])
    return out


# ----------------------------------------------------------------------------- dataset (YOLO format: pictures + one .txt label file per picture)
class Dataset:
    """A labelled set of pictures in the usual YOLO layout:

        <root>/images/<anything>/*.jpg      and      <root>/labels/<same sub-path>/*.txt      (each line: class x_center y_center width height, 0-1)
    <root> can also be given as the path of a data.yaml (its `path:` / `val:` entries and `names:` are used)."""

    def __init__(self, source, names=None):
        source = Path(source)
        self.names = names
        if source.suffix in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(source.read_text(encoding="utf8"))
            root = Path(cfg.get("path", source.parent))
            root = root if root.is_absolute() else (source.parent / root).resolve()
            val = cfg.get("val") or cfg.get("test") or "images"
            val = Path(val) if Path(val).is_absolute() else root / val
            self.images = list_images(val)
            n = cfg.get("names")
            if isinstance(n, dict):
                n = [n[k] for k in sorted(n)]
            self.names = self.names or n
        else:
            self.images = list_images(source / "images") if (source / "images").exists() else list_images(source)
        if not self.images:
            raise FileNotFoundError(f"No pictures found for the dataset at {source}")
        self.labels = [self._label(p) for p in self.images]

    @staticmethod
    def _label(img_path):
        parts = list(img_path.parts)
        if "images" in parts:                                   # .../images/val/x.jpg -> .../labels/val/x.txt
            k = len(parts) - 1 - parts[::-1].index("images")
            f = Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        else:
            f = img_path.with_suffix(".txt")
        if not f.exists() or not f.stat().st_size:
            return np.zeros((0, 5), np.float32)
        return np.loadtxt(f, ndmin=2).astype(np.float32).reshape(-1, 5)

    def __len__(self):
        return len(self.images)

    def subset(self, n, seed=0):
        """The first n pictures after a fixed shuffle (same subset every run)."""
        if n and n < len(self):
            idx = sorted(np.random.RandomState(seed).permutation(len(self))[:n])
            self.images, self.labels = [self.images[i] for i in idx], [self.labels[i] for i in idx]
        return self


def coco_ground_truth(images, labels, shapes, nc):
    """Build a pycocotools ground-truth object from YOLO labels. shapes: [(height, width)]."""
    from pycocotools.coco import COCO
    ims, anns = [], []
    for i, (lab, (h, w)) in enumerate(zip(labels, shapes)):
        ims.append({"id": i, "height": int(h), "width": int(w), "file_name": Path(images[i]).name})
        for c, xc, yc, bw, bh in lab:
            x, y, ww, hh = (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1, "bbox": [float(x), float(y), float(ww), float(hh)], "area": float(ww * hh), "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": ims, "annotations": anns, "categories": [{"id": k + 1, "name": str(k)} for k in range(nc)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def score_detections(dets, gt, nc, n_images):
    """dets: list over pictures of (M, 6) arrays. Returns mAP and AP by size from COCOeval, and precision / recall / F1 at confidence 0.25, IoU 0.5."""
    from pycocotools.cocoeval import COCOeval
    res = []
    for i, d in enumerate(dets):
        for x1, y1, x2, y2, c, k in d:
            res.append({"image_id": i, "category_id": int(k) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(c)})
    out = {}
    if res and gt.dataset["annotations"]:
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, gt.loadRes(res), "bbox")
            E.evaluate(); E.accumulate(); E.summarize()
        s = E.stats
        out.update({"mAP50-95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]), "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5])})
    else:
        out.update({"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0})
    out.update(precision_recall(dets, gt, n_images, conf=0.25))
    return out


def _iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def precision_recall(dets, gt, n_images, conf=0.25, iou=0.5):
    """Greedy matching by score within each class (a detection is right if it overlaps an unmatched real box of its class by IoU >= 0.5)."""
    tp = fp = n_gt = 0
    for i in range(n_images):
        g = [(a["category_id"] - 1, np.array([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]])) for a in gt.imgToAnns.get(i, [])]
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
                used[bj] = True; tp += 1
            else:
                fp += 1
    p, r = tp / max(tp + fp, 1), tp / max(n_gt, 1)
    return {"precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-9), "n_gt": n_gt}
