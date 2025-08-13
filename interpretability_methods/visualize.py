"""
Overlay binary masks onto the original image and save / display them.
"""
from pathlib import Path
from typing import Sequence
import cv2, matplotlib.pyplot as plt, numpy as np



def overlay_heatmap(
    img_bgr: np.ndarray,
    heat: np.ndarray,           # float32 in [0,1]  (H,W)
    box_xyxy: np.ndarray,
    label: str,
    out_dir: Path,
    fname_stub: str,
    show: bool = False,
):
    """
    Blend a continuous heat‑map (magma colormap) with the image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x1,y1,x2,y2 = box_xyxy.astype(int)

    H,W = img_bgr.shape[:2]
    heat_r = cv2.resize(heat, (W, H))
    heat_8 = (np.clip(heat_r, 0, 1) * 255).astype(np.uint8)
    heat_cm = cv2.applyColorMap(heat_8, cv2.COLORMAP_MAGMA)

    overlay = cv2.addWeighted(img_bgr, 0.5, heat_cm, 0.5, 0)
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    fname = f"{fname_stub}_{label.lower()}_raw.png"
    cv2.imwrite(str(out_dir / fname), overlay)

    if show:
        plt.figure(figsize=(6,3))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(fname_stub + " · " + label)
        plt.show()


def overlay_and_save(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    label: str,
    out_dir: Path,
    fname_stub: str,
    show: bool = False,
):
    """
    Draw green box, colour‑code mask (green inside box, red outside),
    save PNG to `out_dir`, optionally display inline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x1,y1,x2,y2 = box_xyxy.astype(int)

    vis = img_bgr.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

    mc = np.zeros_like(vis)
    inside = np.zeros(mask.shape, bool); inside[y1:y2, x1:x2] = True
    mc[ mask.astype(bool) &  inside] = (0,255,0)     # green
    mc[ mask.astype(bool) & ~inside] = (0,0,255)     # red

    ov = cv2.addWeighted(vis, 0.7, mc, 0.3, 0)
    fname = f"{fname_stub}_{label.lower()}_mask.png"
    cv2.imwrite(str(out_dir / fname), ov)

    if show:
        plt.figure(figsize=(6,3))
        plt.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(fname_stub + " · " + label)
        plt.show()



def overlay_and_save_multi(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    boxes_xyxy: Sequence[np.ndarray],
    label: str,
    out_dir: Path,
    fname_stub: str,
    show: bool = False
):
    """
    Like overlay_and_save, but draws *all* boxes in `boxes_xyxy`.
    
    - img_bgr: original BGR image
    - mask: either a binary mask or float heatmap (same logic as overlay_and_save)
    - boxes_xyxy: list/tuple of [x1,y1,x2,y2] numpy arrays
    - label: string to place in the title/filename
    - out_dir: Path to save into
    - fname_stub: base filename
    - show: whether to display inline
    """
    # 1) copy & draw all boxes
    vis = img_bgr.copy()
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2) prepare coloured overlay exactly as overlay_and_save does
    coloured = np.zeros_like(vis)
    if mask.dtype == np.bool_ or mask.max() <= 1:
        # binary mask or float in [0,1]
        coloured[..., 1] = (mask * 255).astype(np.uint8)   # green inside
        coloured[..., 2] = ((1 - mask) * 255).astype(np.uint8)  # red outside
    else:
        # assume mask is already a heatmap in [0,255]
        coloured = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

    # 3) blend & annotate
    blended = cv2.addWeighted(vis, 0.7, coloured, 0.3, 0)
    text = f"{fname_stub[:10]} - {label}"
    cv2.putText(
        blended, text, (10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA
    )

    # 4) save
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{fname_stub}.png"
    cv2.imwrite(str(save_path), blended)

    # 5) show inline if requested
    if show:
        from matplotlib import pyplot as plt
        # convert BGR→RGB for display
        disp = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(disp)
        plt.axis("off")
