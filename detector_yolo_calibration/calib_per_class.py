#!/usr/bin/env python
"""
calib_per_class_methods_iou.py

Per-class calibration for YOLO detections across multiple IoU thresholds.

- Methods: Uncalibrated, Temperature Scaling, Platt Scaling
- IoUs: 0.3, 0.5, 0.7 (configurable)
- Class-level metrics:
    * ECE (fixed-width & adaptive/equal-frequency) + 95% bootstrap CI
    * NLL, Brier, AUROC
- Plots:
    * reliability_grid_fixed.png     (rows=IoU, cols=Method, lines=classes; fixed bins)
    * reliability_grid_adaptive.png  (rows=IoU, cols=Method, lines=classes; adaptive bins)
- Prints per-class results in your requested style.

Run:
python calib_per_class_methods_iou.py \
  --weights path/to/best.pt \
  --data path/to/data.yaml \
  --device cpu|cuda|mps \
  --fixed_bins 15 \
  --adaptive_bins 10 \
  --bootstrap \
  --out_dir calib_results_per_class_methods
"""

import argparse, yaml, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


# ========================== Calibration models ==========================
class BinaryTemperatureScaling:
    def __init__(self, T_init=1.0):
        self.T = T_init

    def fit(self, logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(logits) < 5:
            return

        def nll(T):
            T = np.clip(T, 1e-8, 100)
            probs = 1 / (1 + np.exp(-(logits / T)))
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        res = minimize(nll, x0=1.0, method="Nelder-Mead")
        if res.success:
            self.T = float(res.x[0])

    def transform(self, logit):
        return float(1 / (1 + np.exp(-(logit / self.T))))


class PlattScaling:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(logits) < 5:
            return

        def nll(params):
            a, b = params
            probs = 1 / (1 + np.exp(-(a * logits + b)))
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
        if res.success:
            self.a, self.b = float(res.x[0]), float(res.x[1])

    def transform(self, logit):
        return float(1 / (1 + np.exp(-(self.a * logit + self.b))))



# ========================== Matching ==========================
def greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_threshold=0.5):
    n_det, n_gt = len(p_boxes), len(gt_boxes)
    if n_gt == 0:
        return torch.zeros(n_det, dtype=torch.bool), torch.zeros(n_det)

    iou = box_iou(p_boxes, gt_boxes)
    candidates = [
        (iou[i, j].item(), i, j)
        for i in range(n_det)
        for j in range(n_gt)
        if p_cls[i] == gt_cls[j] and iou[i, j] >= iou_threshold
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    det_matched = torch.zeros(n_det, dtype=torch.bool)
    gt_matched = torch.zeros(n_gt, dtype=torch.bool)
    det_iou = torch.zeros(n_det)

    for iou_val, i, j in candidates:
        if not det_matched[i] and not gt_matched[j]:
            det_matched[i] = True
            gt_matched[j] = True
            det_iou[i] = iou_val

    return det_matched, det_iou



# ========================== ECE + helpers ==========================
def ece_fixed(probs, labels, n_bins=15):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(probs, edges) - 1
    ece = 0.0
    N = len(probs)
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        acc, conf = labels[mask].mean(), probs[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece), mids, edges


def adaptive_bin_edges(probs, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    # Ensure full [0,1] coverage
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def ece_adaptive(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = adaptive_bin_edges(probs, n_bins)
    bin_idx = np.digitize(probs, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    mids = []
    ece = 0.0
    N = len(probs)
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        mids.append(conf)
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece), np.array(mids), edges


def bootstrap_ece(probs, labels, ece_func, n_bins, n_boot=1000, seed=42):
    """Paired resampling (probs, labels) together."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    rng = np.random.default_rng(seed)

    samples = []
    N = len(probs)
    if N == 0:
        return float("nan"), float("nan"), float("nan")

    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        ece_val, *_ = ece_func(probs[idx], labels[idx], n_bins)
        samples.append(ece_val)

    samples = np.array(samples, dtype=float)
    return float(samples.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))



# ========================== Metrics ==========================
def calculate_nll(labels, probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-15, 1 - 1e-15)
    labels = np.asarray(labels, dtype=float)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def calculate_brier(labels, probs):
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def calculate_auroc(labels, probs):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) <= 1:
        return float("nan")
    return float(roc_auc_score(labels, probs))



# ========================== Plotting ==========================
def plot_reliability_grid(results_by_iou, class_names, methods_order, fixed_bins, adaptive_bins, out_dir):
    """
    results_by_iou[iou][class_idx] = {
        "labels": np.array(...),
        "Uncal": np.array(...),
        "TS": np.array(...),
        "Platt": np.array(...),
    }
    """
    ious = sorted(results_by_iou.keys())
    n_rows = len(ious)
    n_cols = len(methods_order)

    # ---------- FIXED ----------
    fig_f, axes_f = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.8*n_rows))
    if n_rows == 1:
        axes_f = np.expand_dims(axes_f, 0)
    if n_cols == 1:
        axes_f = np.expand_dims(axes_f, 1)

    for r, iou in enumerate(ious):
        for c, method in enumerate(methods_order):
            ax = axes_f[r, c]
            ax.plot([0, 1], [0, 1], "--", color="gray")
            for cls_idx, cls_name in enumerate(class_names):
                labels = np.asarray(results_by_iou[iou][cls_idx]["labels"], dtype=float)
                probs = np.asarray(results_by_iou[iou][cls_idx][method], dtype=float)
                if len(labels) == 0:
                    continue
                _, mids, edges = ece_fixed(probs, labels, fixed_bins)
                edges = np.asarray(edges, dtype=float)
                accs = [
                    np.mean(labels[(probs >= edges[b]) & (probs < edges[b + 1])])
                    if np.any((probs >= edges[b]) & (probs < edges[b + 1])) else np.nan
                    for b in range(len(mids))
                ]
                ax.plot(mids, accs, marker="o", label=cls_name)
            ax.set_title(f"IoU {iou} — {method} — Fixed ({fixed_bins} bins)")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle=":", alpha=0.7)
            if r == 0 and c == 0:
                ax.legend()

    plt.tight_layout()
    fixed_path = os.path.join(out_dir, "reliability_grid_fixed.png")
    plt.savefig(fixed_path, dpi=300)
    plt.close(fig_f)
    print(f"Saved {fixed_path}")

    # ---------- ADAPTIVE ----------
    fig_a, axes_a = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.8*n_rows))
    if n_rows == 1:
        axes_a = np.expand_dims(axes_a, 0)
    if n_cols == 1:
        axes_a = np.expand_dims(axes_a, 1)

    for r, iou in enumerate(ious):
        for c, method in enumerate(methods_order):
            ax = axes_a[r, c]
            ax.plot([0, 1], [0, 1], "--", color="gray")
            for cls_idx, cls_name in enumerate(class_names):
                labels = np.asarray(results_by_iou[iou][cls_idx]["labels"], dtype=float)
                probs = np.asarray(results_by_iou[iou][cls_idx][method], dtype=float)
                if len(labels) == 0:
                    continue
                _, mids, edges = ece_adaptive(probs, labels, adaptive_bins)
                edges = np.asarray(edges, dtype=float)
                accs = [
                    np.mean(labels[(probs >= edges[b]) & (probs <= edges[b + 1])])
                    if np.any((probs >= edges[b]) & (probs <= edges[b + 1])) else np.nan
                    for b in range(len(mids))
                ]
                ax.plot(mids, accs, marker="o", label=cls_name)
            ax.set_title(f"IoU {iou} — {method} — Adaptive ({adaptive_bins} bins)")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle=":", alpha=0.7)
            if r == 0 and c == 0:
                ax.legend()

    plt.tight_layout()
    adaptive_path = os.path.join(out_dir, "reliability_grid_adaptive.png")
    plt.savefig(adaptive_path, dpi=300)
    plt.close(fig_a)
    print(f"Saved {adaptive_path}")



# ========================== Data utils ==========================
def load_gt(lbl_path: Path):
    if not Path(lbl_path).exists():
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
    a = np.loadtxt(lbl_path, ndmin=2)
    cls = torch.tensor(a[:, 0], dtype=torch.long)
    x, y, w, h = a[:, 1:].T
    xyxy = torch.tensor(np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1))
    return xyxy, cls



# ========================== Main ==========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fixed_bins", type=int, default=15)
    p.add_argument("--adaptive_bins", type=int, default=10)
    p.add_argument("--ious", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--out_dir", default="calib_results_per_class_methods")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    model = YOLO(args.weights).to(device).eval()
    with open(args.data) as f:
        data = yaml.safe_load(f)

    val_dir = Path(data["val"])
    label_dir = val_dir.parent / "labels"
    img_files = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    nc = model.model.model[-1].nc
    class_names = [model.names[i] if hasattr(model, "names") else f"class_{i}" for i in range(nc)]
    EPS = 1e-12
    methods_order = ["Uncal", "TS", "Platt"]

    results_by_iou = {}  # iou -> list (len=nc) of dicts: {"labels", "Uncal", "TS", "Platt"}

    for iou_th in args.ious:
        print(f"\n=== IoU {iou_th} ===")
        # 1) gather for fitting (per class)
        fit_logits = [[] for _ in range(nc)]
        fit_labels = [[] for _ in range(nc)]

        for img_path in img_files:
            res = model.predict(img_path, imgsz=640, conf=0.001, verbose=False)[0]
            p_boxes = res.boxes.xyxyn.cpu()
            p_cls = res.boxes.cls.long().cpu()
            p_conf = res.boxes.conf.cpu()

            gt_boxes, gt_cls = load_gt(label_dir / img_path.with_suffix(".txt").name)
            matched, _ = greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_th)

            for i in range(len(p_boxes)):
                c = int(p_cls[i])
                p = float(p_conf[i])
                ok = int(matched[i])
                p_clip = np.clip(p, EPS, 1 - EPS)
                logit = float(np.log(p_clip / (1 - p_clip)))
                fit_logits[c].append(logit)
                fit_labels[c].append(ok)

        # 2) Fit per-class calibrators
        ts_scalers = [BinaryTemperatureScaling() for _ in range(nc)]
        platt_scalers = [PlattScaling() for _ in range(nc)]
        for c in range(nc):
            if len(fit_logits[c]) >= 10:
                ts_scalers[c].fit(fit_logits[c], fit_labels[c])
                platt_scalers[c].fit(fit_logits[c], fit_labels[c])

        # 3) Apply to full set (per class)
        labels_by_class = [[] for _ in range(nc)]
        raw_by_class = [[] for _ in range(nc)]
        ts_by_class = [[] for _ in range(nc)]
        platt_by_class = [[] for _ in range(nc)]

        for img_path in img_files:
            res = model.predict(img_path, imgsz=640, conf=0.001, verbose=False)[0]
            p_boxes = res.boxes.xyxyn.cpu()
            p_cls = res.boxes.cls.long().cpu()
            p_conf = res.boxes.conf.cpu()

            gt_boxes, gt_cls = load_gt(label_dir / img_path.with_suffix(".txt").name)
            matched, _ = greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_th)

            for i in range(len(p_boxes)):
                c = int(p_cls[i])
                p = float(p_conf[i])
                ok = int(matched[i])

                labels_by_class[c].append(ok)
                raw_by_class[c].append(p)

                p_clip = np.clip(p, EPS, 1 - EPS)
                logit = float(np.log(p_clip / (1 - p_clip)))

                ts_by_class[c].append(ts_scalers[c].transform(logit) if len(fit_logits[c]) >= 10 else p)
                platt_by_class[c].append(platt_scalers[c].transform(logit) if len(fit_logits[c]) >= 10 else p)

        # 4) Store
        results_by_iou[iou_th] = []
        for c in range(nc):
            results_by_iou[iou_th].append({
                "labels": np.array(labels_by_class[c], dtype=float),
                "Uncal": np.array(raw_by_class[c], dtype=float),
                "TS": np.array(ts_by_class[c], dtype=float),
                "Platt": np.array(platt_by_class[c], dtype=float),
            })

        # 5) Print per-class metrics in your exact style
        for c in range(nc):
            labels = results_by_iou[iou_th][c]["labels"]
            if len(labels) == 0:
                continue
            print(f"\nClass: {class_names[c]}")
            for meth, key in [("Uncal", "Uncal"), ("TS", "TS"), ("Platt", "Platt")]:
                probs = results_by_iou[iou_th][c][key]
                fece = ece_fixed(probs, labels, args.fixed_bins)[0]
                aece = ece_adaptive(probs, labels, args.adaptive_bins)[0]
                if args.bootstrap:
                    fmean, flow, fup = bootstrap_ece(probs, labels, ece_fixed, args.fixed_bins)
                    amean, alow, aup = bootstrap_ece(probs, labels, ece_adaptive, args.adaptive_bins)
                else:
                    fmean = flow = fup = fece
                    amean = alow = aup = aece

                # ECE prints
                print(f"{meth} Fixed ECE: {fece:.4f} | Adaptive ECE: {aece:.4f}")
                print(f"   Fixed 95% CI: [{flow:.4f}, {fup:.4f}] | Adaptive 95% CI: [{alow:.4f}, {aup:.4f}]")

                # Metrics prints (using the human-readable names)
                long_name = {"Uncal":"Uncalibrated", "TS":"TS", "Platt":"Platt"}[meth]
                print(f"{long_name} Metrics:")
                print(f"  NLL: {calculate_nll(labels, probs):.4f}")
                print(f"  Brier: {calculate_brier(labels, probs):.4f}")
                print(f"  AUROC: {calculate_auroc(labels, probs):.4f}")

    # 6) Big grid plots for visual comparison (fixed & adaptive)
    plot_reliability_grid(
        results_by_iou=results_by_iou,
        class_names=class_names,
        methods_order=["Uncal", "TS", "Platt"],
        fixed_bins=args.fixed_bins,
        adaptive_bins=args.adaptive_bins,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
