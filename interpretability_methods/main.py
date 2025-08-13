"""
One‑stop helper: pick examples → compute three overlays → save / show.

>>> from dental_interpretability import generate_examples
>>> generate_examples("TP", (0.6, 1.0), 10)
"""
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from config import (
    MODEL_CHECKPOINT, OUTPUT_DIR, DEVICE,
    OCC_PATCH, OCC_STRIDE, BASELINE_COLOR
)
from selector import select_cases, select_cases_for_lime_boxes
from heatmap_methods import rise_heatmap, occlusion_heatmap, lime_per_box,run_lime_for_multiple_boxes
from dataset_utils import class_to_id, mask_from_heatmap
from  visualize import overlay_and_save, overlay_and_save_multi, overlay_heatmap



def explain_multiple_boxes_lime(
    case_type: str = "TP",
    conf_range: tuple = (0.5, 1.0),
    n_samples: int = 20,
    class_name: str = "Fillings",
    show_inline: bool = False,
):
    """
    Run LIME (multi-box per class) for a sample of detections.
    
    Parameters
    ----------
    case_type : str
        "TP", "FP", "FN", etc. Used to select subset.
    conf_range : tuple
        Confidence range to filter predictions.
    n_samples : int
        Number of examples to process.
    class_name : str
        Target class name (e.g., "Implant").
    show_inline : bool
        If True, show overlays in notebook / UI.
    """

    # ------------------ Load model ------------------
    yolo = YOLO(str(MODEL_CHECKPOINT)).to(DEVICE).eval()
    class_id = class_to_id(class_name)

    # ------------------ Select cases ------------------
    cases = select_cases_for_lime_boxes(case_type, class_name, conf_range, n_samples)

    dst_root = Path(OUTPUT_DIR) / class_name / case_type
    dst_root.mkdir(parents=True, exist_ok=True)

    # ------------------ Loop over cases ------------------
    for idx, info in enumerate(cases, 1):
        img_bgr = cv2.imread(info["img_path"])
        box_ref = info["box_ref"]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- Estimate GT count for target class ---
        if "class_ids" in info:
            num_gt = sum(1 for c in info["class_ids"] if c == class_id)
        else:
            num_gt = 1  # fallback if class_ids missing

        # --- LIME (multi-box) ---
        lime_mask = run_lime_for_multiple_boxes(
            rgb_img=rgb,
            model=yolo,
            class_id=class_id,
            num_gt_boxes=num_gt
        )

        # --- Output name ---
        stub = f"{Path(info['img_path']).stem}_{idx:03d}"

        # --- draw *all* predicted boxes for this class ---
        boxes_to_draw = [
                b.astype(int)
                for b, cid in zip(info["boxes"], info["class_ids"])
                if cid == class_id
            ]
        title = f"{class_name}"
        overlay_and_save_multi(img_bgr, lime_mask, boxes_to_draw, title, dst_root, stub, show_inline)


def generate_examples(
    case_type   : str = "TP",
    conf_range  : tuple = (0.5, 1.0),
    n_samples   : int = 20,
    class_name  : str = "Fillings",
    occ_pct: float = 98.0,
    rise_pct: float = 99.5,
    show_inline : bool = False,
    heatmap_mode: str = "mask",        # "mask" | "raw"

):
    """
    end‑to‑end pipeline: pick TP/FP/FN cases, build three heat‑maps,
    convert to masks with user‑supplied percentiles and save overlays.

    Percentile arguments control mask_from_heatmap() for each method.

    """

    # ------------------------------------------------------------
    # 0) Model + cases
    # ------------------------------------------------------------
    yolo = YOLO(str(MODEL_CHECKPOINT)).to(DEVICE).eval()
    cases = select_cases(case_type, class_name, conf_range, n_samples)

    dst_root = Path(OUTPUT_DIR) / class_name / case_type
    dst_root.mkdir(parents=True, exist_ok=True)


    class_id = class_to_id(class_name)

    # ------------------------------------------------------------
    # 1) Loop over examples
    # ------------------------------------------------------------
    print(cases)
    for idx, info in enumerate(cases, 1):
        img_bgr = cv2.imread(info["img_path"])
        box_ref = info["box_ref"]; conf = info["conf"]


        # ---------------- Heat‑maps ----------------
        lime = lime_per_box(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), yolo, ref_box_xyxy=box_ref, class_id=class_id)
        rgb_for_rise = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rise_h = rise_heatmap(rgb_for_rise, yolo, box_ref, class_id=class_id)
        occ_h  = occlusion_heatmap(yolo, img_bgr, box_ref, conf)

        stub = f"{Path(info['img_path']).stem}_{idx:03d}"
        if heatmap_mode == "mask":
            rise = mask_from_heatmap(rise_h, rise_pct)
            occ  = mask_from_heatmap(occ_h,   occ_pct)
            # ---------- save binary overlays ----------
            overlay_and_save(img_bgr, occ,  box_ref, "Occlusion", dst_root, stub, show_inline)
            overlay_and_save(img_bgr, lime, box_ref, "LIME",      dst_root, stub, show_inline)
            overlay_and_save(img_bgr, rise, box_ref, "RISE",      dst_root, stub, show_inline)

        elif heatmap_mode == "raw":
            # normalise 0‑1 for display
            rise_hn = (rise_h - rise_h.min()) / (np.ptp(rise_h) + 1e-7)
            occ_hn = (occ_h - occ_h.min()) / (np.ptp(occ_h) + 1e-7)
            # ---------- save colour heat‑maps ----------
            overlay_heatmap(img_bgr, occ_hn,  box_ref, "Occlusion", dst_root, stub, show_inline)
            overlay_heatmap(img_bgr, lime,    box_ref, "LIME",      dst_root, stub, show_inline)
            overlay_heatmap(img_bgr, rise_hn, box_ref, "RISE",      dst_root, stub, show_inline)

        else:
            raise ValueError("heatmap_mode must be 'mask' or 'raw'")

