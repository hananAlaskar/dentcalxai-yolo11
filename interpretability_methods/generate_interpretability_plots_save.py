
from __future__ import annotations

import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import Optional, Sequence
from ultralytics import YOLO

from config import MODEL_CHECKPOINT
from dataset_utils import (
    class_to_id, coverage_frac, expand_box,
    hot_frac, mask_from_heatmap
)
from heatmap_methods import lime_per_box, occlusion_heatmap, rise_heatmap
from selector import select_cases
from visualize import overlay_and_save


def generate_interpretability_plots(
    output_dir: str | Path = "./interpretability_plots/all",
    class_names: Optional[Sequence[str]] = None,
    case_types: Optional[Sequence[str]] = None,
    conf_range: tuple[float, float] = (0.25, 1.0),
    n_samples: int = 330,
    occ_pct: float = 98.0,
    rise_pct: float = 99.5,
    device: Optional[str] = None,
    model: Optional[YOLO] = None,
) -> None:
    """Run the interpretability pipeline and save statistics/overlays.

    Parameters
    ----------
    output_dir
        Root directory where all overlay images and statistics CSV will be saved.
    class_names
        List of class names to iterate over. Defaults to common dental classes.
    case_types
        Case categories to consider (e.g. TP, FP, FN).
    conf_range
        Minimum and maximum confidence threshold for sample selection.
    n_samples
        Number of samples to draw per (class, case) pair.
    occ_pct
        Percentile cutoff for occlusion heatmap mask thresholding.
    rise_pct
        Percentile cutoff for RISE heatmap mask thresholding.
    device
        Torch device; if omitted, picks CUDA when available.
    model
        Pre‑loaded YOLO model; if omitted, loads from `MODEL_CHECKPOINT`.
    """
    # --- Defaults & setup -----------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    stats_csv = output_dir / "records_overlay_stats_all_classes.csv"

    if class_names is None:
        class_names = ["Cavity", "Impacted Tooth", "Implant", "Fillings"]
    if case_types is None:
        case_types = ["TP", "FP", "FN"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = YOLO(str(MODEL_CHECKPOINT)).to(device).eval()

    # --- Resume from previous run (if any) ------------------------------------
    if stats_csv.exists():
        df_existing = pd.read_csv(stats_csv)
        all_records = df_existing.to_dict(orient="records")
        processed = len(all_records)
        print(f"Resuming with {processed} records already in CSV")
    else:
        all_records = []
        processed = 0

    # --- Main loop ------------------------------------------------------------
    for cls in class_names:
        for case in case_types:
            # how many images already done for this (class,case)?
            done_for_pair = sum(
                1 for r in all_records if r["class"] == cls and r["case"] == case
            )

            print(f"→ {cls}/{case}: skipping first {done_for_pair} samples")
            cases = select_cases(case, cls, conf_range, n_samples)[done_for_pair:]

            # Process one image at a time
            for offset, info in enumerate(cases):
                idx = done_for_pair + offset + 1
                img_bgr = cv2.imread(info["img_path"])
                box_ref = info["box_ref"]
                conf = info["conf"]

                # a) heatmaps
                lime_h = lime_per_box(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    model,
                    ref_box_xyxy=box_ref,
                    class_id=class_to_id(cls),
                )
                rise_h = rise_heatmap(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    model,
                    box_ref,
                    class_id=class_to_id(cls),
                )
                occ_h = occlusion_heatmap(model, img_bgr, box_ref, conf)

                # b) masks
                mask_occ = mask_from_heatmap(occ_h, occ_pct)
                mask_lime = mask_from_heatmap(lime_h, 100.0)
                mask_rise = mask_from_heatmap(rise_h, rise_pct)

                # c) box → int
                box_int = expand_box(box_ref, img_bgr.shape)

                # d) metrics
                occ_hf, occ_cv = hot_frac(mask_occ, box_int), coverage_frac(
                    mask_occ, box_int
                )
                lime_hf, lime_cv = hot_frac(mask_lime, box_int), coverage_frac(
                    mask_lime, box_int
                )
                rise_hf, rise_cv = hot_frac(mask_rise, box_int), coverage_frac(
                    mask_rise, box_int
                )

                # e) overlay save
                stub = Path(info["img_path"]).stem + f"_{idx:03d}"
                for method, mask in [
                    ("Occlusion", mask_occ),
                    ("LIME", mask_lime),
                    ("RISE", mask_rise),
                ]:
                    out_dir = output_dir / cls / case / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    overlay_and_save(
                        img_bgr,
                        mask,
                        box_int,
                        method,
                        out_dir,
                        stub,
                        draw_box=False,
                    )

                # f) record
                all_records.append(
                    {
                        "class": cls,
                        "case": case,
                        "img": Path(info["img_path"]).name,
                        "idx": idx,
                        "conf": conf,
                        "occ_hf": occ_hf,
                        "occ_cv": occ_cv,
                        "lime_hf": lime_hf,
                        "lime_cv": lime_cv,
                        "rise_hf": rise_hf,
                        "rise_cv": rise_cv,
                    }
                )

                processed += 1

                # Flush every 2 new images
                if processed % 2 == 0:
                    pd.DataFrame(all_records).to_csv(stats_csv, index=False)
                    print(f"  [flushed {processed} records to CSV]")
    # --- Final save -----------------------------------------------------------
    pd.DataFrame(all_records).to_csv(stats_csv, index=False)
    print(f"Done! {len(all_records)} total records written to {stats_csv}")


if __name__ == "__main__":
    generate_interpretability_plots()
