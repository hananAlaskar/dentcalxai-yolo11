
"""Utility for visualizing model predictions vs. ground truth."
"""
import random
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from ultralytics import YOLO

from config import TEST_IMAGES_DIR

def compute_iou(box1: Tuple[float,float,float,float],
                box2: Tuple[float,float,float,float]) -> float:
    """Compute intersection‑over‑union between two bboxes in (x1,y1,x2,y2)."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / (box1Area + box2Area - interArea + 1e-6)

def yolo_to_xyxy(box, w, h):
    """Convert YOLO (x_center,y_center,width,height) → (x1,y1,x2,y2)"""
    x_center, y_center, bw, bh = box
    x1 = int((x_center - bw / 2) * w)
    y1 = int((y_center - bh / 2) * h)
    x2 = int((x_center + bw / 2) * w)
    y2 = int((y_center + bh / 2) * h)
    return x1, y1, x2, y2

def show_random_predictions(
    model_path: str,
    conf_thres: float = 0.5,
    iou_threshold: float = 0.5,
    num_images: int = 10,
) -> None:
    """Draw predicted & GT boxes for *num_images* random test images."""
    model = YOLO(model_path)

    annotations_csv = Path(TEST_IMAGES_DIR) / "_annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(annotations_csv)

    df = pd.read_csv(annotations_csv)
    image_files = df['filename'].unique()
    choices = random.sample(list(image_files), min(num_images, len(image_files)))

    for image_name in choices:
        img_path = Path(TEST_IMAGES_DIR) / "images" / image_name
        img = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        # --- Ground truth boxes -------------------------------------------------
        gt_boxes = []
        for _, row in df[df['filename'] == image_name].iterrows():
            gt_boxes.append((row['xmin'], row['ymin'], row['xmax'], row['ymax']))

        # --- Predictions --------------------------------------------------------
        results = model(img_path, conf=conf_thres, iou=iou_threshold)[0]
        pred_boxes = []
        for box in results.boxes.xyxy.cpu().numpy():
            pred_boxes.append(tuple(map(int, box)))

        # --- Plot ----------------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image_rgb)
        ax.axis('off')

        # Draw GT boxes (green)
        for x1, y1, x2, y2 in gt_boxes:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        # Draw predictions (red) + IoU label
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Compute IoU against best matching GT box
            ious = [compute_iou(box, gt) for gt in gt_boxes] or [0.0]
            best_iou = max(ious)
            ax.text(x1, y1 - 5, f"IoU {best_iou:.2f}",
                    color='red', fontsize=10,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.show()
