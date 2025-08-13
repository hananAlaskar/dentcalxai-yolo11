"""
TP / FP / FN sampler with confidence filtering.
"""
from __future__ import annotations
import pandas as pd
import json, yaml, cv2, glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from ultralytics import YOLO

from config import (VAL_IMAGES_DIR, VAL_LABELS_DIR, DATA_YAML,
                     MODEL_CHECKPOINT, IOU_THRESH, DEVICE)
from dataset_utils import class_to_id, yolo_txt_to_xyxy, iou_one_to_many

# ------------------------------------------------------------
# Load class names, expose helper to look up id by name
# ------------------------------------------------------------
with open(DATA_YAML, "r") as f:
    _cfg = yaml.safe_load(f)
CLASS_NAMES: List[str] = _cfg["names"]

def class_id(name_or_idx):
    return name_or_idx if isinstance(name_or_idx, int) \
           else CLASS_NAMES.index(name_or_idx)




def load_dataset_cases() -> List[Dict]:
    """
    Run YOLO inference on every image in VAL_IMAGES_DIR and return
    a list of dicts with keys:
      - img_path (str)
      - box_ref (np.ndarray of shape (4,))
      - conf    (float)
      - boxes   (List[np.ndarray])
      - class_ids (List[int])
      - type    ("TP" by default—we’ll relabel later if you want FP/FN)
    """
    cases = []

    # ——— DEBUG: check VAL_IMAGES_DIR and what files we see ———
    p = Path(VAL_IMAGES_DIR)
    img_files = sorted(p.glob("*.jpg"))
    print(f"DEBUG: VAL_IMAGES_DIR = {p.resolve()}, exists? {p.exists()}")
    print(f"DEBUG: Found {len(img_files)} .jpg files")



    # 1.) load your trained model once
    model = YOLO(str(MODEL_CHECKPOINT))
    model.model.to(DEVICE).eval()

    # 2.) iterate over all validation images
    for img_path in sorted(Path(VAL_IMAGES_DIR).glob("*.jpg")):
        # read as BGR for OpenCV & Ultralytics
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print("DEBUG: cv2 failed to read", img_path)
            continue

        # 3.) run a single forward‑pass
        results = model(
            bgr,
            device=DEVICE,
            imgsz=640,
            conf=0.60,      
            verbose=False,
            save=False
        )[0]

        # 4.) extract numpy arrays
        if not len(results.boxes):
            continue

        xyxy    = results.boxes.xyxy.cpu().numpy()       # (N,4)
        confs   = results.boxes.conf.cpu().numpy()       # (N,)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # pick the highest‑confidence detection as reference
        best_idx = int(np.argmax(confs))

        cases.append({
            "img_path": str(img_path),
            "box_ref":  xyxy[best_idx],
            "conf":     float(confs[best_idx]),
            "boxes":    [xyxy[i] for i in range(len(confs))],
            "class_ids": [int(c) for c in cls_ids],
            "type":     "TP"   # you can later reassign TP/FP/FN via iou logic if you like
        })

    return cases


def select_cases_for_lime_boxes(case_type: str,
                                class_name: str,
                                conf_range: tuple,
                                n_samples: int) -> list[dict]:
    """
    Special select_cases function for LIME multi-box explanations.
    
    Parameters
    ----------
    case_type : str
        Indicates the type of case to select (e.g. "TP", "FP", "FN").
    class_name : str
        Target class name (e.g. "Implant" or "Fillings").
    conf_range : tuple
        Confidence range filter as (min_confidence, max_confidence).
    n_samples : int
        Maximum number of cases to return.
    
    Returns
    -------
    list of dict
        Each dictionary has the required keys:
            - "img_path": Path to the image file.
            - "box_ref": Numpy array reference box (e.g. in [x1,y1,x2,y2] format).
            - "conf": Confidence score (float) corresponding to the reference box.
            - "boxes": Optional list/array of all predicted boxes.
            - "class_ids": Optional list/array of class ids corresponding to the boxes.
            
    Notes
    -----
    Modify the internal filtering logic according to your dataset and annotation format.
    """
    
    # For example, assume `dataset_cases` is a list of all available cases loaded from somewhere.
    # You might load it from a JSON file or database instead.
    dataset_cases = load_dataset_cases()  # Replace with your dataset loader
    
    # Filter the dataset based on the provided parameters.
    filtered_cases = []
    for case in dataset_cases:
        # Ensure the case has the expected keys.
        required_keys = {"img_path", "box_ref", "conf"}
        if not required_keys.issubset(case.keys()):
            continue

        # Example filter: case_type might be determined by a field "type" in the case.
        # Adjust the logic if your dataset marks cases differently.
        if case.get("type", None) != case_type:
            continue

        # Check the predicted confidence falls within the desired range.
        conf = float(case["conf"])
        if conf < conf_range[0] or conf > conf_range[1]:
            continue

        # Optionally, you might also check that the case has boxes for the target class.
        # Assume "class_ids" exists if multiple boxes are available.
        if "class_ids" in case:
            # For the target class (convert class_name to class_id if required)
            target_class_id = class_to_id(class_name)
            if target_class_id not in case["class_ids"]:
                continue

        filtered_cases.append(case)
        
        # Stop if we've collected enough samples.
        if len(filtered_cases) >= n_samples:
            break
    
    # In case dataset_cases is empty or filtering is too strict,
    # you may return a dummy list or raise an error.
    if not filtered_cases:
        print(f">>> DEBUG: total cases before filter: {len(dataset_cases)}")
        print(f">>> DEBUG: after filtering by class={class_name}, type={case_type}, range={conf_range}: 0 cases")
        raise ValueError("No cases found matching the criteria. Please check your filters.")
    return filtered_cases

# ------------------------------------------------------------
# Main selection routine
# ------------------------------------------------------------
def select_cases(
    case_type   : str,                             # "TP" | "FP" | "FN"
    class_name  : str,
    conf_range  : Tuple[float,float],
    n_samples   : int
) -> List[Dict]:
    """
    Returns a list of up to `n_samples` dicts each describing one example.
    For FN we ignore `conf_range` (no detection → conf = 0).
    """
    yolo = YOLO(str(MODEL_CHECKPOINT))
    yolo.model.to("cpu").eval()

    idx_cls = class_id(class_name)
    img_paths = sorted(Path(VAL_IMAGES_DIR).glob("*.jpg"))

    bucket: List[Dict] = []
    for img_path in img_paths:
        pil  = Image.open(img_path).convert("RGB")
        bgr  = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        # inference
        det = yolo.predict(bgr, device="cpu", imgsz=640,
                           verbose=False, save=False)[0]
        p_boxes  = det.boxes.xyxy.cpu().numpy()
        p_confs  = det.boxes.conf.cpu().numpy()
        p_labels = det.boxes.cls.cpu().numpy().astype(int)

        # ground truth
        stem = img_path.stem
        gt_arr = yolo_txt_to_xyxy(Path(VAL_LABELS_DIR)/f"{stem}.txt", W, H)
        g_boxes, g_labels = (gt_arr[:, :4], gt_arr[:, 4].astype(int)) if gt_arr.size else (np.zeros((0,4)), np.zeros((0,)))


        # match
        tp_idx, fp_idx, fn_idx = _match_tp_fp_fn(
            p_boxes, p_labels, p_confs,
            g_boxes, g_labels,
            target_cls=idx_cls, iou_thresh=IOU_THRESH
        )


        _collect(bucket, img_path, p_boxes, p_confs, g_boxes,
                 tp_idx, fp_idx, fn_idx, case_type,
                 conf_range, n_samples)
        

        if len(bucket) >= n_samples:
            break

        
    return bucket


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def _match_tp_fp_fn(
    p_boxes, p_labels, p_confs,
    g_boxes, g_labels,
    target_cls, iou_thresh
):
    mask_p = (p_labels == target_cls)
    mask_g = (g_labels == target_cls)
    P, G   = p_boxes[mask_p], g_boxes[mask_g]
    pi, gi = np.where(mask_p)[0], np.where(mask_g)[0]

    matched_p, matched_g = set(), set()
    if P.size and G.size:
        iou = np.zeros((len(P), len(G)), np.float32)
        for i in range(len(P)):
            iou[i] = iou_one_to_many(P[i], G)
        while True:
            k = int(np.argmax(iou))
            i, j = divmod(k, iou.shape[1])
            if iou[i, j] < iou_thresh:
                break
            matched_p.add(i); matched_g.add(j)
            iou[i, :] = -1;  iou[:, j] = -1

    tp = [pi[i] for i in matched_p]
    fp = [pi[i] for i in range(len(P)) if i not in matched_p]
    fn = [gi[j] for j in range(len(G)) if j not in matched_g]
    return tp, fp, fn


def _collect(bucket, img_path, p_boxes, p_confs, g_boxes,
             tp_idx, fp_idx, fn_idx,
             case_type, conf_range, n_samples):
    lo, hi = conf_range
    if case_type == "TP":
        for i in tp_idx:
            if lo <= p_confs[i] <= hi and len(bucket) < n_samples:
                bucket.append(dict(
                    img_path = str(img_path),
                    box_ref  = p_boxes[i].copy(),
                    conf     = float(p_confs[i]),
                    kind     = "TP",
                ))
    elif case_type == "FP":
        for i in fp_idx:
            if lo <= p_confs[i] <= hi and len(bucket) < n_samples:
                bucket.append(dict(
                    img_path = str(img_path),
                    box_ref  = p_boxes[i].copy(),
                    conf     = float(p_confs[i]),
                    kind     = "FP",
                ))
    elif case_type == "FN":
        for i in fn_idx:
            if len(bucket) < n_samples:
                bucket.append(dict(
                    img_path = str(img_path),
                    box_ref  = g_boxes[i].copy(),
                    conf     = 0.0,
                    kind     = "FN",
                ))
