#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute and visualize σ-Head detection statistics:

1. Load YOLO-σ predictions and GT data
2. Compute and display basic statistics and plots
3. Cluster detections and preview examples
"""

import os
import glob
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------------
def load_data(csv_path: str, img_folder: str, label_folder: str) -> (pd.DataFrame, pd.DataFrame):
    # Load predictions
    df = pd.read_csv(csv_path)
    df["pred_w"] = df["x2"] - df["x1"]
    df["pred_h"] = df["y2"] - df["y1"]
    df["pred_area"] = df["pred_w"] * df["pred_h"]
    df["sigma_norm"] = np.hypot(df.sigma_x, df.sigma_y)

    # Load GT data
    img_sizes = {}
    for img_path in glob.glob(os.path.join(img_folder, "*")):
        stem = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        img_sizes[stem] = (w, h)

    records = []
    for txt_path in glob.glob(os.path.join(label_folder, "*.txt")):
        stem = Path(txt_path).stem
        dims = img_sizes.get(stem)
        if not dims: continue
        w_img, h_img = dims
        with open(txt_path) as f:
            for line in f:
                cls, xc, yc, wn, hn = map(float, line.split())
                pix_w = wn * w_img
                pix_h = hn * h_img
                records.append({
                    "image": f"{stem}.jpg",
                    "gt_area": pix_w * pix_h
                })
    df_gt = pd.DataFrame.from_records(records)
    return df, df_gt

# ------------------------------------------------------------
# 2. Statistics and Visualization
# ------------------------------------------------------------
def visualize_stats(df: pd.DataFrame, df_gt: pd.DataFrame, csv_path: str) -> None:
    # Basic stats
    print(f"Loaded {len(df):,} predictions from {csv_path}")
    print(df[["sigma_x","sigma_y"]].describe(percentiles=[0.25,0.5,0.75]))
    print("Predicted-area stats (pixels²):")
    print(df["pred_area"].describe(percentiles=[0.25,0.5,0.75]))

    print(f"\nLoaded {len(df_gt):,} ground-truth boxes")
    print("GT-area stats (pixels²):")
    print(df_gt["gt_area"].describe(percentiles=[0.25,0.5,0.75]))

    # Histograms
    plt.figure(figsize=(7,4))
    plt.hist(df["sigma_x"], bins=50, alpha=0.6, label="σₓ")
    plt.hist(df["sigma_y"], bins=50, alpha=0.6, label="σᵧ")
    plt.xlim(0)
    plt.xlabel("σ (pixels)")
    plt.ylabel("Count")
    plt.title("Distribution of positional uncertainty")
    plt.legend()
    plt.tight_layout(); plt.show()

    # Scatter
    plt.figure(figsize=(5,5))
    plt.scatter(df["sigma_x"], df["sigma_y"], s=8, alpha=0.3)
    plt.xlabel("σₓ (pixels)")
    plt.ylabel("σᵧ (pixels)")
    plt.title("σₓ vs σᵧ per detection")
    plt.axis("equal")
    plt.grid(True, linestyle="--")
    plt.xlim(0); plt.ylim(0)
    plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 3. Clustering and Preview
# ------------------------------------------------------------
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def analyze_uncertainties_clusters(
    pred_csv,
    img_folder,
    label_folder,
    n_clusters=3,
    iou_thresh=0.30,
    k_sigma=3
):
    """
    Load σ-Head detections, match to ground truth boxes, cluster uncertainties,
    and visualize results—with clusters relabeled by centroid size,
    robust exemplar selection, and per-image legends.
    """
    # 1. load predictions and compute combined σ
    pred = pd.read_csv(pred_csv)
    pred["sigma_norm"] = np.hypot(pred.sigma_x, pred.sigma_y)
    print(f"Loaded {len(pred):,} detections from {pred_csv}")

    # 2. build GT map
    gt_boxes_map = {}
    for txt_path in glob.glob(os.path.join(label_folder, "*.txt")):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        img_file = base + ".jpg"
        img_path = os.path.join(img_folder, img_file)
        if not os.path.exists(img_path):
            continue
        H, W = cv2.imread(img_path).shape[:2]
        boxes = []
        with open(txt_path) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 5: 
                    continue
                _, xc, yc, bw, bh = map(float, parts[:5])
                x1 = (xc - bw/2) * W
                y1 = (yc - bh/2) * H
                x2 = (xc + bw/2) * W
                y2 = (yc + bh/2) * H
                boxes.append((x1, y1, x2, y2))
        if boxes:
            gt_boxes_map[img_file] = boxes
    print(f"Loaded GT for {len(gt_boxes_map)} images")

    # 3. IoU helper
    def iou_xyxy(a, b):
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        iw = max(0, min(xa2, xb2) - max(xa1, xb1))
        ih = max(0, min(ya2, yb2) - max(ya1, yb1))
        inter = iw * ih
        if inter == 0:
            return 0.0
        areaA = (xa2 - xa1) * (ya2 - ya1)
        areaB = (xb2 - xb1) * (yb2 - yb1)
        return inter / (areaA + areaB - inter)

    # 4. match preds → GT
    best_ious, best_gts = [], []
    for _, r in pred.iterrows():
        p_box = (r.x1, r.y1, r.x2, r.y2)
        bi, bg = 0.0, None
        for g in gt_boxes_map.get(r.image, []):
            i = iou_xyxy(p_box, g)
            if i > bi:
                bi, bg = i, g
        best_ious.append(bi)
        best_gts.append(bg)
    pred["best_iou"] = best_ious
    pred["best_gt"] = best_gts

    # 5. filter true positives
    pred_tp = pred[pred.best_iou >= iou_thresh].reset_index(drop=True)
    print(f"True positives: {len(pred_tp):,}")

    # 6. k-means on (σₓ,σᵧ)
    xy = pred_tp[["sigma_x", "sigma_y"]].to_numpy()
    cents = xy[np.random.choice(len(xy), n_clusters, replace=False)]
    for _ in range(50):
        d = np.stack([np.linalg.norm(xy - c, axis=1) for c in cents], axis=1)
        lbl = d.argmin(axis=1)
        for k in range(n_clusters):
            pts = xy[lbl == k]
            if len(pts):
                cents[k] = pts.mean(axis=0)

    # --- relabel clusters by ascending centroid norm ---
    centroids = np.vstack(cents)
    norms = np.linalg.norm(centroids, axis=1)
    new_order = np.argsort(norms)
    remap = {old: new for new, old in enumerate(new_order)}
    lbl = np.array([remap[l] for l in lbl], dtype=int)
    cents = centroids[new_order]
    # ------------------------------------------------

    pred_tp["cluster"] = lbl
    clusters = sorted(remap.values())

    # 7. pick cluster exemplars (robust)
    preview = []
    for c in clusters:
        sub = pred_tp[pred_tp.cluster == c]
        if sub.empty:
            continue
        for q in (0.25, 0.5, 0.75):
            tgt = sub.sigma_norm.quantile(q)
            idx = (sub.sigma_norm - tgt).abs().idxmin()
            preview.append(sub.loc[idx])

    # 8. scatter overview
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["blue", "orange", "purple", "cyan", "magenta", "gold"]
    for i, c in enumerate(clusters):
        col = colors[i]
        sub = pred_tp[pred_tp.cluster == c]
        ax.scatter(sub.sigma_x, sub.sigma_y,
                   s=30, alpha=0.6, color=col,
                   label=f"Cluster {c}")
        pts = [r for r in preview if r.cluster == c]
        ax.scatter([r.sigma_x for r in pts],
                   [r.sigma_y for r in pts],
                   s=150, marker="*",
                   facecolors=col, edgecolors="black",
                   label=f"Cluster {c} exemplar")

    ax.set_xlabel("σₓ (pixels)")
    ax.set_ylabel("σᵧ (pixels)")
    ax.set_title(f"σₓ vs σᵧ (TP only, {n_clusters} clusters)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

    # 9. image previews with per-image legend
    PURPLE = (255, 0, 255)
    def draw_dashed(img, pt1, pt2, color, th=2, dash=6, gap=4):
        (x1, y1), (x2, y2) = pt1, pt2
        segs = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
        for (sx, sy), (ex, ey) in segs:
            L = int(np.hypot(ex - sx, ey - sy))
            n = max(1, L // (dash + gap))
            for i in range(n):
                sf = i / n
                ef = min(1, (i*(dash+gap) + dash) / L)
                xs = int(sx + (ex - sx) * sf)
                ys = int(sy + (ey - sy) * sf)
                xe = int(sx + (ex - sx) * ef)
                ye = int(sy + (ey - sy) * ef)
                cv2.line(img, (xs, ys), (xe, ye), color, th)

    def shade_buffer(im, x1, y1, x2, y2, dx, dy, a=0.25):
        ov = np.zeros_like(im)
        cv2.rectangle(ov, (x1-dx, y1-dy), (x2+dx, y2+dy), PURPLE, -1)
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return cv2.addWeighted(ov, a, im, 1.0, 0)

    # prepare legend handles once
    legend_handles = [
        Patch(edgecolor='blue',    facecolor='none',   label='pred box',     linewidth=2),
        Patch(edgecolor='magenta', facecolor='magenta', alpha=0.25,         label='±3σ buffer'),
        Line2D([0],[0], color='magenta', lw=2, linestyle='--', label='buffer outline'),
        Patch(edgecolor='green',   facecolor='none',   label='GT box',       linewidth=2),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=8, label='pred center'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   markersize=8, label='GT center'),
    ]

    for c in clusters:
        for r in [x for x in preview if x.cluster == c]:
            img_p = os.path.join(img_folder, r.image)
            im = cv2.imread(img_p)
            if im is None:
                print("⚠️ missing", r.image)
                continue

            x1, y1, x2, y2 = map(int, (r.x1, r.y1, r.x2, r.y2))
            dx, dy = int(k_sigma * r.sigma_x), int(k_sigma * r.sigma_y)

            vis = shade_buffer(im.copy(), x1, y1, x2, y2, dx, dy)
            # predicted box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # buffer outline
            draw_dashed(vis, (x1-dx, y1-dy), (x2+dx, y2+dy), PURPLE, 2)
            # pred center
            mu_x, mu_y = (x1 + x2)//2, (y1 + y2)//2
            cv2.circle(vis, (mu_x, mu_y), 5, (0, 255, 0), -1)
            # GT box
            gx1, gy1, gx2, gy2 = map(int, r.best_gt)
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            # GT center
            gcx, gcy = (gx1 + gx2)//2, (gy1 + gy2)//2
            cv2.circle(vis, (gcx, gcy), 5, (0, 0, 255), -1)

            im_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            fig, ax2 = plt.subplots(figsize=(6,6))
            ax2.imshow(im_rgb)
            ax2.axis("off")
            ax2.set_title(f"{r.image}\n{k_sigma}σₓ={dx}px  σᵧ={dy}px  IoU={r.best_iou:.2f}")
            ax2.legend(handles=legend_handles,
                       loc='upper right', fontsize='small', framealpha=0.7)
            plt.tight_layout()
            plt.show()

    return pred_tp


# def analyze_uncertainties_clusters(
#     pred_csv,
#     img_folder,
#     label_folder,
#     n_clusters=3,
#     iou_thresh=0.30,
#     k_sigma=3
# ):
#     """
#     Load σ-Head detections, match to ground truth boxes, cluster uncertainties,
#     and visualize results, with clusters relabeled by centroid size.
#     """
#     # 1. load predictions and compute combined σ
#     pred = pd.read_csv(pred_csv)
#     pred["sigma_norm"] = np.hypot(pred.sigma_x, pred.sigma_y)
#     print(f"Loaded {len(pred):,} detections from {pred_csv}")

#     # 2. build GT map
#     gt_boxes_map = {}
#     for txt_path in glob.glob(os.path.join(label_folder, "*.txt")):
#         base = os.path.splitext(os.path.basename(txt_path))[0]
#         img_file = base + ".jpg"
#         img_path = os.path.join(img_folder, img_file)
#         if not os.path.exists(img_path):
#             continue
#         H, W = cv2.imread(img_path).shape[:2]
#         boxes = []
#         with open(txt_path) as f:
#             for ln in f:
#                 parts = ln.split()
#                 if len(parts) < 5: continue
#                 _, xc, yc, bw, bh = map(float, parts[:5])
#                 x1 = (xc - bw/2) * W
#                 y1 = (yc - bh/2) * H
#                 x2 = (xc + bw/2) * W
#                 y2 = (yc + bh/2) * H
#                 boxes.append((x1, y1, x2, y2))
#         if boxes:
#             gt_boxes_map[img_file] = boxes
#     print(f"Loaded GT for {len(gt_boxes_map)} images")

#     # 3. IoU helper
#     def iou_xyxy(a, b):
#         xa1, ya1, xa2, ya2 = a
#         xb1, yb1, xb2, yb2 = b
#         iw = max(0, min(xa2, xb2) - max(xa1, xb1))
#         ih = max(0, min(ya2, yb2) - max(ya1, yb1))
#         inter = iw * ih
#         if inter == 0: return 0.0
#         areaA = (xa2 - xa1)*(ya2 - ya1)
#         areaB = (xb2 - xb1)*(yb2 - yb1)
#         return inter/(areaA + areaB - inter)

#     # 4. match preds→GT
#     best_ious, best_gts = [], []
#     for _, r in pred.iterrows():
#         p_box = (r.x1, r.y1, r.x2, r.y2)
#         bi, bg = 0.0, None
#         for g in gt_boxes_map.get(r.image, []):
#             i = iou_xyxy(p_box, g)
#             if i > bi:
#                 bi, bg = i, g
#         best_ious.append(bi)
#         best_gts.append(bg)
#     pred["best_iou"] = best_ious
#     pred["best_gt"]  = best_gts

#     # 5. filter TPs
#     pred_tp = pred[pred.best_iou >= iou_thresh].reset_index(drop=True)
#     print(f"True positives: {len(pred_tp):,}")

#     # 6. k-means on (σₓ,σᵧ)
#     xy = pred_tp[["sigma_x", "sigma_y"]].to_numpy()
#     cents = xy[np.random.choice(len(xy), n_clusters, replace=False)]
#     for _ in range(50):
#         d = np.stack([np.linalg.norm(xy - c, axis=1) for c in cents], axis=1)
#         lbl = d.argmin(axis=1)
#         for k in range(n_clusters):
#             pts = xy[lbl == k]
#             if len(pts):
#                 cents[k] = pts.mean(axis=0)

#     # --- relabel clusters by ascending centroid norm -----------
#     centroids = np.vstack(cents)                     # shape (n_clusters, 2)
#     norms = np.linalg.norm(centroids, axis=1)        # size metric
#     new_order = np.argsort(norms)                    # indices sorted by size
#     remap = { old: new for new, old in enumerate(new_order) }
#     # apply to labels and reorder cents
#     lbl = np.array([remap[l] for l in lbl], dtype=int)
#     cents = centroids[new_order]
#     # -----------------------------------------------------------

#     pred_tp["cluster"] = lbl
#     clusters = sorted(remap.values())

#     # 7. pick cluster exemplars
#     preview = []
#     for c in clusters:
#         sub = pred_tp[pred_tp.cluster == c]
#         for q in (0.25, 0.5, 0.75):
#             tgt = sub.sigma_norm.quantile(q)
#             preview.append(sub.iloc[(sub.sigma_norm - tgt).abs().argmin()])

#     # 8. scatter overview
#     fig, ax = plt.subplots(figsize=(8, 8))
#     colors = ["blue", "orange", "purple", "cyan", "magenta", "gold"]
#     for i, c in enumerate(clusters):
#         col = colors[i]
#         sub = pred_tp[pred_tp.cluster == c]
#         ax.scatter(sub.sigma_x, sub.sigma_y,
#                    s=30, alpha=0.6, color=col,
#                    label=f"Cluster {c}")
#         # exemplars
#         pts = [r for r in preview if r.cluster == c]
#         ax.scatter([r.sigma_x for r in pts],
#                    [r.sigma_y for r in pts],
#                    s=150, marker="*",
#                    facecolors=col, edgecolors="black",
#                    label=f"Cluster {c} centroid")

#     ax.set_xlabel("σₓ (pixels)")
#     ax.set_ylabel("σᵧ (pixels)")
#     ax.set_title(f"σₓ vs σᵧ (TP only, {n_clusters} clusters)")
#     ax.set_aspect("equal")
#     ax.grid(True, linestyle="--", linewidth=0.5)
#     ax.legend(loc="upper left", bbox_to_anchor=(1.02,1))
#     plt.tight_layout()
#     plt.show()

#     # 9. image previews (unchanged)
#     PURPLE = (255,0,255)
#     def draw_dashed(img, pt1, pt2, color, th=2, dash=6, gap=4):
#         (x1,y1), (x2,y2) = pt1, pt2
#         segs = [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
#                 ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]
#         for (sx,sy),(ex,ey) in segs:
#             L = int(np.hypot(ex-sx,ey-sy))
#             n = max(1, L//(dash+gap))
#             for i in range(n):
#                 sf, ef = i/n, min(1,(i*(dash+gap)+dash)/L)
#                 xs = int(sx + (ex-sx)*sf)
#                 ys = int(sy + (ey-sy)*sf)
#                 xe = int(sx + (ex-sx)*ef)
#                 ye = int(sy + (ey-sy)*ef)
#                 cv2.line(img, (xs,ys), (xe,ye), color, th)

#     def shade_buffer(im, x1,y1,x2,y2, dx,dy, a=0.25):
#         ov = np.zeros_like(im)
#         cv2.rectangle(ov, (x1-dx,y1-dy), (x2+dx,y2+dy), PURPLE, -1)
#         cv2.rectangle(ov, (x1,y1), (x2,y2), (0,0,0), -1)
#         return cv2.addWeighted(ov, a, im, 1.0, 0)

#     for c in clusters:
#         print(f"\n=== Cluster {c} examples ===")
#         for r in [x for x in preview if x.cluster == c]:
#             img_p = os.path.join(img_folder, r.image)
#             im = cv2.imread(img_p)
#             if im is None:
#                 print("⚠️ missing", r.image); continue

#             x1,y1,x2,y2 = map(int,(r.x1,r.y1,r.x2,r.y2))
#             dx,dy = int(k_sigma*r.sigma_x), int(k_sigma*r.sigma_y)

#             vis = shade_buffer(im.copy(), x1,y1,x2,y2, dx,dy)
#             cv2.rectangle(vis, (x1,y1),(x2,y2),(255,0,0),2)
#             draw_dashed(vis, (x1-dx,y1-dy),(x2+dx,y2+dy), PURPLE, 2)

#             mu_x,mu_y = (x1+x2)//2, (y1+y2)//2
#             cv2.circle(vis, (mu_x,mu_y), 5, (0,255,0), -1)

#             gx1,gy1,gx2,gy2 = map(int, r.best_gt)
#             cv2.rectangle(vis, (gx1,gy1),(gx2,gy2),(0,255,0),2)
#             gcx,gcy = (gx1+gx2)//2, (gy1+gy2)//2
#             cv2.circle(vis, (gcx,gcy), 5, (0,0,255), -1)

#             im_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
#             fig, ax2 = plt.subplots(figsize=(6,6))
#             ax2.imshow(im_rgb); ax2.axis("off")
#             ax2.set_title(f"{r.image}\n"
#                           f"{k_sigma}σₓ={dx}px  σᵧ={dy}px  IoU={r.best_iou:.2f}")
#             plt.tight_layout(); plt.show()

#     return pred_tp


def load_and_cluster(pred_csv, label_folder, img_folder,
                     n_clusters=3, iou_thresh=0.30):
    """
    Load predictions, match to GT, filter TPs, cluster uncertainties,
    and relabel clusters by ascending centroid size.
    """
    pred = pd.read_csv(pred_csv)
    pred["sigma_norm"] = np.hypot(pred.sigma_x, pred.sigma_y)
    print(f"Loaded {len(pred):,} detections from {pred_csv}")

    # build GT map (same as above)…
    gt_boxes_map = {}
    for txt_path in glob.glob(os.path.join(label_folder, "*.txt")):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        img_file = base + ".jpg"
        img_path = os.path.join(img_folder, img_file)
        if not os.path.exists(img_path): continue
        H, W = cv2.imread(img_path).shape[:2]
        boxes = []
        with open(txt_path) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 5: continue
                _, xc, yc, bw, bh = map(float, parts[:5])
                x1 = (xc - bw/2)*W; y1 = (yc - bh/2)*H
                x2 = (xc + bw/2)*W; y2 = (yc + bh/2)*H
                boxes.append((x1,y1,x2,y2))
        if boxes: gt_boxes_map[img_file] = boxes
    print(f"Loaded GT for {len(gt_boxes_map)} images")

    # match preds→GT
    def iou_xyxy(a, b):
        xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
        iw = max(0, min(xa2, xb2) - max(xa1, xb1))
        ih = max(0, min(ya2, yb2) - max(ya1, yb1))
        inter = iw*ih
        if inter == 0: return 0.0
        return inter/((xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter)

    best_ious, best_gts = [], []
    for _, r in pred.iterrows():
        pbox = (r.x1, r.y1, r.x2, r.y2)
        bi, bg = 0.0, None
        for g in gt_boxes_map.get(r.image, []):
            i = iou_xyxy(pbox, g)
            if i > bi:
                bi, bg = i, g
        best_ious.append(bi); best_gts.append(bg)
    pred["best_iou"] = best_ious
    pred["best_gt"]  = best_gts

    pred_tp = pred[pred.best_iou >= iou_thresh].reset_index(drop=True)
    print(f"Kept {len(pred_tp):,} TPs (IoU ≥ {iou_thresh})")

    # k-means
    xy = pred_tp[["sigma_x","sigma_y"]].to_numpy()
    cent = xy[np.random.choice(len(xy), n_clusters, replace=False)]
    for _ in range(50):
        dists = np.stack([np.linalg.norm(xy - c, axis=1) for c in cent], axis=1)
        lbl   = np.argmin(dists, axis=1)
        for k in range(n_clusters):
            pts = xy[lbl==k]
            if len(pts):
                cent[k] = pts.mean(axis=0)

    # same relabel block as above
    centroids = np.vstack(cent)
    norms     = np.linalg.norm(centroids, axis=1)
    new_order = np.argsort(norms)
    remap     = { old: new for new, old in enumerate(new_order) }
    lbl = np.array([remap[l] for l in lbl], dtype=int)
    cent = centroids[new_order]

    pred_tp["cluster"] = lbl
    cluster_ids = sorted(remap.values())
    return pred_tp, cluster_ids


def plot_sigma_buffers(pred_tp, cluster_ids=None, buffer_multipliers=None,
                       buffer_box=(100, 100, 200, 150)):
    """
    Plot ±m·σ rectangles for each cluster around an example box.
    (No changes needed here—the relabeled clusters will just appear in the desired order.)
    """
    if buffer_multipliers is None:
        return

    if cluster_ids is None:
        cluster_ids = sorted(pred_tp['cluster'].unique())

    means = pred_tp.groupby('cluster')[['sigma_x','sigma_y']].mean().sort_index()
    x1,y1,x2,y2 = buffer_box
    rows, cols = len(buffer_multipliers), len(cluster_ids)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    for i, m in enumerate(buffer_multipliers):
        for j, c in enumerate(cluster_ids):
            ax = axes[i,j] if rows>1 and cols>1 else (axes[j] if rows==1 else axes[i])
            sx, sy = means.loc[c]
            bx1 = x1 - m*sx; by1 = y1 - m*sy
            bw  = (x2 + m*sx) - bx1; bh  = (y2 + m*sy) - by1
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                           edgecolor='blue', facecolor='none', lw=2))
            ax.add_patch(patches.Rectangle((bx1,by1), bw, bh,
                                           edgecolor='magenta', facecolor='none',
                                           linestyle='--', lw=2))
            ax.set_title(f"Cluster {c}\n{m}×σₓ={m*sx:.1f}px\n{m}×σᵧ={m*sy:.1f}px",
                         fontsize=10)
            ax.set_xlim(0, (x2 + max(buffer_multipliers)*means['sigma_x'].max())*1.1)
            ax.set_ylim((y2 + max(buffer_multipliers)*means['sigma_y'].max())*1.1, 0)
            ax.axis('off')

    fig.suptitle("Per‐cluster ±σ buffers", y=1.02)
    plt.tight_layout()
    plt.show()



