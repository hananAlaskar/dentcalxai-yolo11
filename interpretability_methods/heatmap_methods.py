"""
Occlusion, LIME and RISE implementations in one module.
The functions are **model‑agnostic**: pass an ultralytics.YOLO instance.
"""
from __future__ import annotations
import math
import numpy as np, cv2, torch
from typing import List, Optional, Tuple
from ultralytics import YOLO
from torchvision.ops import box_iou
from lime import lime_image
from skimage.segmentation import slic

from dataset_utils import iou_one_to_many

from config import (
    DEVICE, LIME_SAMPLES, LIME_SUPERPIX, LIME_TOP_FEAT,
    RISE_NUM_MASKS, RISE_KEEP_PROB, RISE_LOW_CONF, RISE_BATCH_SZ,
    RISE_IMG_SIZE, RISE_MASK_CELL, OCC_PATCH, OCC_STRIDE, BASELINE_COLOR
)


# ----------------------------------------------------------------------
# Convenience – convert an RGB/uint8/float image to (1,3,H,W) tensor
# ----------------------------------------------------------------------
@torch.no_grad()
def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    if rgb.dtype != np.uint8:
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)

    # NEW – ensure positive, contiguous strides
    rgb = np.ascontiguousarray(rgb)         # or rgb = rgb.copy()

    return (torch.from_numpy(rgb)
                 .permute(2, 0, 1).float()[None].to(DEVICE) / 255.)


# ----------------------------------------------------------------------
# ------------------------------  R I S E  ------------------------------
# ----------------------------------------------------------------------
@torch.no_grad()
def _make_masks(N: int, H: int, W: int, keep: float, cell: int) -> torch.Tensor:
    h, w = H // cell, W // cell
    g = (torch.rand(N, 1, h, w, device=DEVICE) < keep).float()
    return torch.nn.functional.interpolate(g, size=(H, W),
                                           mode="bilinear", align_corners=False)

@torch.no_grad()
def rise_heatmap(
    rgb: np.ndarray,
    model: YOLO,
    ref_box_xyxy: Optional[np.ndarray],
    class_id: int,
) -> np.ndarray:
    """
    Implements D‑RISE (guided by IoU×confidence to reference box).
    """
    H, W, _ = rgb.shape
    ref_box = torch.as_tensor(ref_box_xyxy, device=DEVICE).view(1, 4)
    masks   = _make_masks(RISE_NUM_MASKS, H, W, RISE_KEEP_PROB, RISE_MASK_CELL)

    sal   = torch.zeros((H, W), device=DEVICE)
    img_t = _to_tensor(rgb)

    for b in range(0, RISE_NUM_MASKS, RISE_BATCH_SZ):
        m     = masks[b : b + RISE_BATCH_SZ]
        preds = model(img_t * m, imgsz=RISE_IMG_SIZE,
                      conf=RISE_LOW_CONF, verbose=False)

        for k, r in enumerate(preds):
            cls_mask = (r.boxes.cls.int() == class_id)
            if not cls_mask.any():
                continue
            iou = box_iou(r.boxes.xyxy[cls_mask].cpu(), ref_box.cpu())[0]
            score = (iou * r.boxes.conf[cls_mask].cpu()).max()
            sal  += score * m[k, 0]

    sal = sal / (RISE_NUM_MASKS * RISE_KEEP_PROB + 1e-7)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-7)
    return sal.cpu().numpy()


# ----------------------------------------------------------------------
# ------------------------------  L I M E  ------------------------------
# ----------------------------------------------------------------------

def run_lime_for_multiple_boxes(
    rgb_img: np.ndarray,
    model,
    class_id: int,
    num_gt_boxes: int,
    num_samples: int = 1000,
    n_segments: int = 100,
    compactness: float = 10.0
) -> np.ndarray:
    """
    Performs LIME for all predicted boxes of a class in one image.
    
    Parameters
    ----------
    rgb_img : np.ndarray
        Input RGB image (H×W×3, float32 or uint8).
    model : YOLO
        Ultralytics YOLO model instance.
    class_id : int
        Target class ID to explain.
    num_gt_boxes : int
        Number of ground-truth boxes for the target class.
    num_samples : int
        Number of LIME samples.
    n_segments : int
        Number of superpixels for segmentation.
    compactness : float
        Compactness factor for superpixel clustering.
    
    Returns
    -------
    mask : np.ndarray
        Binary mask (H×W) of top-contributing superpixels.
    """
    
    def _lime_score_fn(batch: list[np.ndarray]) -> np.ndarray:
        """Returns sum of confidence scores for the class_id across all boxes."""
        scores = []
        for img in batch:
            if img.max() > 1.0:
                img = img / 255.0
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            res = model.predict(img_bgr, device="cpu", imgsz=640, verbose=False, save=False)[0]

            if len(res.boxes):
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                confs   = res.boxes.conf.cpu().numpy()
                class_mask = cls_ids == class_id
                score = confs[class_mask].sum() if class_mask.any() else 0.0
            else:
                score = 0.0
            scores.append([score])
        return np.array(scores)

    # Ensure float32 RGB in [0,1]
    if rgb_img.dtype == np.uint8:
        rgb_img = rgb_img.astype(np.float32) / 255.0

    explainer = lime_image.LimeImageExplainer()
    top_feat  = math.ceil(num_gt_boxes * 1.1)

    explanation = explainer.explain_instance(
        rgb_img,
        classifier_fn=_lime_score_fn,
        labels=(0,),
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda im: slic(im, n_segments=n_segments, compactness=compactness)
    )

    _, mask = explanation.get_image_and_mask(
        label=0,
        positive_only=True,
        num_features=top_feat,
        hide_rest=False
    )
    
    return mask.astype(np.uint8)


# ----------------------------------------------------------------------
# ------------------------------  L I M E PER BOX ------------------------------
# ----------------------------------------------------------------------

def _yolo_scores_lime_per_box(
    batch: List[np.ndarray],
    model: YOLO,
    ref_box_xyxy: np.ndarray,
    class_id: int,
    iou_thresh: float = 0.5
) -> np.ndarray:
    """
    For each image in the batch, return the confidence of the box with highest IoU
    to the reference box, *only if* the class_id matches.
    """
    scores = []
    for img in batch:
        if img.max() > 1.0:
            img = img / 255.0
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        res = model.predict(img_bgr, device="cpu", imgsz=640, verbose=False, save=False)[0]
        conf = 0.0

        if len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            ious = box_iou(torch.tensor([ref_box_xyxy]), torch.tensor(boxes)).numpy()[0]
            best_idx = np.argmax(ious)
            if ious[best_idx] >= iou_thresh and classes[best_idx] == class_id:
                conf = confs[best_idx]
        scores.append([conf])

    return np.array(scores)


def lime_per_box(
    rgb: np.ndarray,
    model: YOLO,
    ref_box_xyxy: np.ndarray,
    class_id: int
) -> np.ndarray:
    """
    LIME per‑box variant — explanation mask for a specific box.
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb,
        classifier_fn=lambda b: _yolo_scores_lime_per_box(
            [x / 255. for x in b], model, ref_box_xyxy, class_id
        ),
        labels=(0,),  # dummy label, used internally by LIME
        hide_color=0,
        num_samples=LIME_SAMPLES,
        segmentation_fn=lambda im: slic(im, n_segments=LIME_SUPERPIX, compactness=10)
    )

    _, mask = explanation.get_image_and_mask(0, positive_only=True,
                                             num_features=LIME_TOP_FEAT,
                                             hide_rest=False)
    return mask.astype(np.uint8)


# ----------------------------------------------------------------------
# --------------------------  O c c l u s i o n  ------------------------
# ----------------------------------------------------------------------
def occlusion_heatmap(
    model: YOLO,
    img_bgr: np.ndarray,
    ref_box: np.ndarray,
    ref_conf: float,
    patch_size: int = OCC_PATCH,
    stride: int     = OCC_STRIDE,
    baseline: int   = BASELINE_COLOR,
    iou_thresh: float = 0.3
) -> np.ndarray:
    """
    Patch-wise occlusion sensitivity – returns heatmap ∝ drop in confidence.
    """
    H, W, _ = img_bgr.shape
    ny = (H - patch_size) // stride + 1
    nx = (W - patch_size) // stride + 1
    heat = np.zeros((ny, nx), np.float32)

    baseline_patch = np.full((patch_size, patch_size, 3), baseline, np.uint8)

    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * stride, ix * stride
            test = img_bgr.copy()
            test[y0:y0 + patch_size, x0:x0 + patch_size] = baseline_patch

            res = model(test, verbose=False, show=False)[0]
            if len(res.boxes):
                pb = res.boxes.xyxy.cpu().numpy()
                pc = res.boxes.conf.cpu().numpy()
                iou = iou_one_to_many(ref_box, pb)
                best = np.argmax(iou)
                conf = pc[best] if iou[best] >= iou_thresh else 0.0
            else:
                conf = 0.0
            heat[iy, ix] = ref_conf - conf

    # Resize and normalize AFTER filling the heatmap
    heat = cv2.resize(heat, (W, H))
    heat = (heat - heat.min()) / (np.ptp(heat) + 1e-7)
    return heat
