#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract YOLO-σ detections (μ-box + σx,σy) for *both* VALID and TEST sets.

It writes:
    predictions_val_uncert.csv
    predictions_test_uncert.csv
Each row: image, det_idx, cls_id, confidence, x1 y1 x2 y2, sigma_x, sigma_y
"""

def extract_sigma_detections(
    ckpt: str,
    device: str,
    splits: dict
):
    # ─────────────────────────────────────────────────────────────
    # 1️⃣  Monkey-patch weight_norm for deepcopy-safe σ-head integration
    #     ▸ Must run before importing Ultralytics’ YOLO to avoid param issues
    # ─────────────────────────────────────────────────────────────
    import math
    import torch.nn.utils as nn_utils
    from torch.nn.utils import parametrizations as P

    nn_utils.weight_norm = P.weight_norm

    import os
    import glob
    import pandas as pd
    from ultralytics import YOLO

    # load model once
    model = YOLO(ckpt).to(device).eval()
    head = model.model.model[-1]       # your SigmaAwareDetect (σ-Head)
    strides = model.model.stride.tolist()

    # helper: map detection centre → (layer-idx, grid-x, grid-y)
    def layer_and_cell(box, strides, shape):
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        best, li, gx, gy = 1e9, -1, -1, -1
        h, w = shape
        for i, s in enumerate(strides):
            gx_i, gy_i = int(cx / s), int(cy / s)
            dx, dy = cx - (gx_i + 0.5) * s, cy - (gy_i + 0.5) * s
            d2 = dx * dx + dy * dy
            if d2 < best:
                best, li, gx, gy = d2, i, gx_i, gy_i
        return li, gx, gy

    # valid image extensions
    EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    for split, cfg in splits.items():
        source = cfg["SOURCE"]
        out_csv = cfg["OUT_CSV"]

        # gather image paths
        imgs = [
            p for p in glob.glob(os.path.join(source, "*"))
            if os.path.splitext(p)[1].lower() in EXT
        ]
        if not imgs:
            print(f"⚠️  No images found in {source}")
            continue

        rows = []
        for img_path in imgs:
            res = model.predict(
                img_path, imgsz=640, conf=0.25,
                save=False, device=device, 
                verbose=False
            )[0]
            sigma_maps = head._last_var_maps
            base = os.path.basename(img_path)
            h, w = res.orig_shape

            for k, box in enumerate(res.boxes):
                li, gx, gy = layer_and_cell(box, strides, (h, w))
                vm = sigma_maps[li][0]
                gx = max(0, min(gx, vm.shape[2] - 1))
                gy = max(0, min(gy, vm.shape[1] - 1))
                sigma_x, sigma_y = vm[:, gy, gx].exp().tolist()

                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                rows.append({
                    "image": base,
                    "det_idx": k,
                    "cls_id": int(box.cls),
                    "confidence": float(box.conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                })
            print(f"{split} | {base}: {len(res.boxes)} detections")

        # write split-specific CSV
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"✅  {split}: saved {len(rows)} rows  →  {out_csv}")


if __name__ == "__main__":
    # user-editable defaults
    CKPT = "/Volumes/L/L_PHAS0077/yolo/trying_yolov11_uncertainty/runs/detect/yolo11n_with_sigma2/weights/best.pt"
    DEVICE = "cpu"
    SPLITS = {
        "val": dict(
            SOURCE="/Volumes/L/L_PHAS0077/yolo/dental_radiography_yolo/valid/images",
            OUT_CSV="predictions_val_uncert.csv",
        ),
        "test": dict(
            SOURCE="/Volumes/L/L_PHAS0077/yolo/dental_radiography_yolo/test/images",
            OUT_CSV="predictions_test_uncert.csv",
        ),
    }

    extract_sigma_detections(CKPT, DEVICE, SPLITS)
