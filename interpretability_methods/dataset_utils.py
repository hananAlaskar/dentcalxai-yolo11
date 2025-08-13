"""
Generic helpers: label parsing, IoU, box ops, misc masks.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Tuple


# ----------------------------------------------
#  Class‑name ↔ class‑id helper
# ----------------------------------------------
YOLO_NAMES = ["Cavity", "Implant", "Fillings", "Impacted Tooth"]  # keep central

def class_to_id(label: int | str) -> int:
    """
    Accepts class index (int) or class name (str) and returns an int id.
    Raises ValueError on unknown names / out‑of‑range ids.
    """
    if isinstance(label, int):
        if 0 <= label < len(YOLO_NAMES):
            return label
        raise ValueError(f"class_id {label} out of range 0‑{len(YOLO_NAMES)-1}")
    else:
        try:
            return YOLO_NAMES.index(label)
        except ValueError:
            raise ValueError(f"Unknown class name: '{label}'")

# ------------------------------------------------------------------
# YOLO txt → xyxy(,cls)
# ------------------------------------------------------------------
def yolo_txt_to_xyxy(label_path: Path, W: int, H: int) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), np.float32)
    out = []
    for ln in label_path.read_text().strip().splitlines():
        cls, xc, yc, w, h = map(float, ln.split())
        x1, y1 = (xc - w/2) * W, (yc - h/2) * H
        x2, y2 = (xc + w/2) * W, (yc + h/2) * H
        out.append([x1, y1, x2, y2, int(cls)])
    return np.array(out, np.float32)


# ------------------------------------------------------------------
# IoU helpers
# ------------------------------------------------------------------
def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    box shape (4,), boxes shape (N,4) – returns IoU for each box.
    """
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xB - xA, 0, None) * np.clip(yB - yA, 0, None)

    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


# ------------------------------------------------------------------
# Simple box & mask utilities
# ------------------------------------------------------------------
def expand_box(box: np.ndarray, img_shape: Tuple[int,int,int], margin: float = 0.00):
    x1, y1, x2, y2 = box.astype(float)
    w, h = x2 - x1, y2 - y1
    dx, dy = w * margin, h * margin
    H, W = img_shape[:2]
    return np.array([max(0,x1-dx), max(0,y1-dy),
                     min(W,x2+dx),  min(H,y2+dy)], dtype=int)

def mask_from_heatmap(hm: np.ndarray, pct: float) -> np.ndarray:
    thresh = np.percentile(hm, pct)
    return (hm >= thresh).astype(np.uint8)

def hot_frac(mask: np.ndarray, box: np.ndarray) -> float:
    total = mask.sum()
    return 0.0 if total == 0 else mask[box[1]:box[3], box[0]:box[2]].sum() / total

def coverage_frac(mask: np.ndarray, box: np.ndarray) -> float:
    x1,y1,x2,y2 = box
    area = (y2-y1)*(x2-x1)
    return 0.0 if area == 0 else mask[y1:y2, x1:x2].sum() / area

