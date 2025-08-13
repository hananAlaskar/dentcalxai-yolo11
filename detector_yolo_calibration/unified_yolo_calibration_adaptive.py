#!/usr/bin/env python
"""
unified_yolo_calibration_stacked.py

- Runs calibration for multiple IoU thresholds
- Fixed-width and adaptive binning ECE with bootstrap CI
- Temperature Scaling and Platt Scaling
- Greedy matching for detections
- Stacked 3×2 reliability diagram (IoU rows, binning columns)
- CSV table of results
"""

import argparse, yaml, numpy as np, torch, os, matplotlib.pyplot as plt, csv
from pathlib import Path
from scipy.optimize import minimize
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# Calibration classes
# ---------------------------------------------------------
class BinaryTemperatureScaling:
    def __init__(self, T_init=1.0):
        self.T = T_init
        
    def fit(self, logits, labels):
        if len(logits) < 5:
            return
        def nll(T):
            T = np.clip(T, 1e-8, 100)
            scaled_logits = logits / T
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-15, 1-1e-15)
            return -np.mean(labels * np.log(probs) + (1-labels) * np.log(1 - probs))
        res = minimize(nll, x0=1.0, method='Nelder-Mead')
        if res.success:
            self.T = res.x[0]
            
    def transform(self, logit):
        return 1 / (1 + np.exp(-logit / self.T))

class PlattScaling:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        
    def fit(self, logits, labels):
        if len(logits) < 5:
            return
        def nll(params):
            a, b = params
            scaled = a * logits + b
            probs = 1 / (1 + np.exp(-scaled))
            probs = np.clip(probs, 1e-15, 1-1e-15)
            return -np.mean(labels * np.log(probs) + (1-labels) * np.log(1 - probs))
        res = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead')
        if res.success:
            self.a, self.b = res.x
            
    def transform(self, logit):
        return 1 / (1 + np.exp(-(self.a * logit + self.b)))

# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------
def greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_threshold=0.5):
    n_det, n_gt = len(p_boxes), len(gt_boxes)
    if n_gt == 0:
        return torch.zeros(n_det, dtype=torch.bool), torch.zeros(n_det)
    iou = box_iou(p_boxes, gt_boxes)
    candidates = [(iou[i, j].item(), i, j)
                  for i in range(n_det) for j in range(n_gt)
                  if p_cls[i] == gt_cls[j] and iou[i, j] >= iou_threshold]
    candidates.sort(key=lambda x: x[0], reverse=True)
    det_matched, gt_matched = torch.zeros(n_det, dtype=torch.bool), torch.zeros(n_gt, dtype=torch.bool)
    det_iou = torch.zeros(n_det)
    for iou_val, i, j in candidates:
        if not det_matched[i] and not gt_matched[j]:
            det_matched[i] = True
            gt_matched[j] = True
            det_iou[i] = iou_val
    return det_matched, det_iou

# ---------------------------------------------------------
# ECE + binning helpers
# ---------------------------------------------------------
def adaptive_bin_edges(probs, n_bins=10):
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = 0.0, 1.0
    return edges

def ece_fixed(probs, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(probs, edges) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        acc, conf = np.mean(labels[mask]), np.mean(probs[mask])
        weight = np.sum(mask) / len(probs)
        ece += weight * abs(acc - conf)
    return ece, mids, edges

def ece_adaptive(probs, labels, n_bins=10):
    edges = adaptive_bin_edges(probs, n_bins)
    mids = []
    ece = 0.0
    bin_idx = np.digitize(probs, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        mids.append(np.mean(probs[mask]))
        acc, conf = np.mean(labels[mask]), np.mean(probs[mask])
        weight = np.sum(mask) / len(probs)
        ece += weight * abs(acc - conf)
    return ece, mids, edges

def bootstrap_ece(probs, labels, ece_func, n_bins, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    samples = [ece_func(
        probs[rng.integers(0, len(probs), len(probs))],
        labels[rng.integers(0, len(labels), len(labels))],
        n_bins
    )[0] for _ in range(n_boot)]
    mean = np.mean(samples)
    ci_low, ci_up = np.percentile(samples, [2.5, 97.5])
    return mean, ci_low, ci_up

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def calculate_nll(labels, probs):
    probs = np.clip(probs, 1e-15, 1-1e-15)
    return -np.mean(labels * np.log(probs) + (1-labels) * np.log(1-probs))

def calculate_brier(labels, probs):
    return np.mean((probs - labels)**2)

def calculate_auroc(labels, probs):
    return roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')

# ---------------------------------------------------------
# Plot stacked reliability
# ---------------------------------------------------------
def plot_reliability_by_iou(probs_by_iou, labels_by_iou, methods, fixed_bins, adaptive_bins, out_path):
    ious = sorted(probs_by_iou.keys())
    fig, axes = plt.subplots(len(ious), 2, figsize=(12, 4*len(ious)))
    for row, iou in enumerate(ious):
        labels = labels_by_iou[iou]
        # Fixed
        ax_fixed = axes[row, 0]
        ax_fixed.plot([0, 1], [0, 1], "--", color="gray")
        for method in methods:
            probs = probs_by_iou[iou][method]
            _, mids, edges = ece_fixed(probs, labels, fixed_bins)
            accs = [np.mean(labels[(probs >= edges[b]) & (probs < edges[b+1])]) 
                    if np.any((probs >= edges[b]) & (probs < edges[b+1])) else np.nan
                    for b in range(fixed_bins)]
            ax_fixed.plot(mids, accs, marker="o", label=method)
        ax_fixed.set_title(f"IoU {iou} – Fixed-width")
        ax_fixed.set_xlabel("Predicted Probability")
        ax_fixed.set_ylabel("Empirical Accuracy")
        ax_fixed.grid(True, linestyle=":", alpha=0.7)
        if row == 0:
            ax_fixed.legend()
        # Adaptive
        ax_adapt = axes[row, 1]
        ax_adapt.plot([0, 1], [0, 1], "--", color="gray")
        for method in methods:
            probs = probs_by_iou[iou][method]
            _, mids, edges = ece_adaptive(probs, labels, adaptive_bins)
            accs = [np.mean(labels[(probs >= edges[b]) & (probs <= edges[b+1])]) 
                    if np.any((probs >= edges[b]) & (probs <= edges[b+1])) else np.nan
                    for b in range(len(mids))]
            ax_adapt.plot(mids, accs, marker="o", label=method)
        ax_adapt.set_title(f"IoU {iou} – Adaptive")
        ax_adapt.set_xlabel("Predicted Probability")
        ax_adapt.set_ylabel("Empirical Accuracy")
        ax_adapt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved stacked reliability plot to {out_path}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed_bins", type=int, default=15)
    parser.add_argument("--adaptive_bins", type=int, default=10)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--out_dir", default="calib_results_stacked")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model = YOLO(args.weights).to(device).eval()
    with open(args.data) as f:
        data = yaml.safe_load(f)
    val_dir = Path(data["val"])
    label_dir = val_dir.parent / "labels"
    img_files = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    iou_thresholds = [0.3, 0.5, 0.7]
    methods = ["Uncal", "TS", "Platt"]
    probs_by_iou = {}
    labels_by_iou = {}
    csv_rows = []

    for iou_th in iou_thresholds:
        nc = model.model.model[-1].nc
        ts_scalers, platt_scalers = [BinaryTemperatureScaling() for _ in range(nc)], [PlattScaling() for _ in range(nc)]
        fit_data, all_labels, all_raw, all_ts, all_platt = [[] for _ in range(nc)], [], [], [], []
        EPS = 1e-12
        # Collect calibration data
        for img_path in img_files:
            res = model.predict(img_path, imgsz=640, conf=0.001, verbose=False)[0]
            p_boxes, p_cls, p_conf = res.boxes.xyxyn.cpu(), res.boxes.cls.long().cpu(), res.boxes.conf.cpu()
            gt_boxes, gt_cls = load_gt(label_dir / img_path.with_suffix(".txt").name)
            matched, _ = greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_th)
            for i in range(len(p_boxes)):
                c, p, ok = int(p_cls[i]), float(p_conf[i]), int(matched[i])
                logit = np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
                fit_data[c].append((logit, ok))
        # Fit scalers
        for c in range(nc):
            if len(fit_data[c]) >= 10:
                logits, labels = zip(*fit_data[c])
                ts_scalers[c].fit(np.array(logits), np.array(labels))
                platt_scalers[c].fit(np.array(logits), np.array(labels))
        # Apply
        for img_path in img_files:
            res = model.predict(img_path, imgsz=640, conf=0.001, verbose=False)[0]
            p_boxes, p_cls, p_conf = res.boxes.xyxyn.cpu(), res.boxes.cls.long().cpu(), res.boxes.conf.cpu()
            gt_boxes, gt_cls = load_gt(label_dir / img_path.with_suffix(".txt").name)
            matched, _ = greedy_matching(p_boxes, p_cls, gt_boxes, gt_cls, iou_th)
            for i in range(len(p_boxes)):
                c, p, ok = int(p_cls[i]), float(p_conf[i]), int(matched[i])
                logit = np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
                all_labels.append(ok)
                all_raw.append(p)
                all_ts.append(ts_scalers[c].transform(logit) if len(fit_data[c]) >= 10 else p)
                all_platt.append(platt_scalers[c].transform(logit) if len(fit_data[c]) >= 10 else p)

        all_labels, all_raw, all_ts, all_platt = map(np.array, (all_labels, all_raw, all_ts, all_platt))
        labels_by_iou[iou_th] = all_labels
        probs_by_iou[iou_th] = {"Uncal": all_raw, "TS": all_ts, "Platt": all_platt}

        print(f"\n=== IoU {iou_th} ===")
        for name, probs in [("Uncal", all_raw), ("TS", all_ts), ("Platt", all_platt)]:
            fece = ece_fixed(probs, all_labels, args.fixed_bins)[0]
            aece = ece_adaptive(probs, all_labels, args.adaptive_bins)[0]
            if args.bootstrap:
                fmean, flow, fup = bootstrap_ece(probs, all_labels, ece_fixed, args.fixed_bins)
                amean, alow, aup = bootstrap_ece(probs, all_labels, ece_adaptive, args.adaptive_bins)
            else:
                fmean, flow, fup = fece, fece, fece
                amean, alow, aup = aece, aece, aece
            print(f"{name} Fixed ECE: {fece:.4f} | Adaptive ECE: {aece:.4f}")
            print(f"   Fixed 95% CI: [{flow:.4f}, {fup:.4f}] | Adaptive 95% CI: [{alow:.4f}, {aup:.4f}]")
            csv_rows.append([iou_th, name, fece, flow, fup, aece, alow, aup])
        # Global metrics
        print("\nGlobal Detection Metrics:")
        for name, probs in [("Uncalibrated", all_raw), ("Temperature Scaling", all_ts), ("Platt Scaling", all_platt)]:
            print(f"{name} Metrics:")
            print(f"  NLL: {calculate_nll(all_labels, probs):.4f}")
            print(f"  Brier: {calculate_brier(all_labels, probs):.4f}")
            print(f"  AUROC: {calculate_auroc(all_labels, probs):.4f}")

    plot_reliability_by_iou(probs_by_iou, labels_by_iou, methods, args.fixed_bins, args.adaptive_bins,
                            os.path.join(args.out_dir, "reliability_stacked.png"))
    # Save CSV
    with open(os.path.join(args.out_dir, "ece_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IoU", "Method", "Fixed ECE", "Fixed CI Low", "Fixed CI Up",
                         "Adaptive ECE", "Adaptive CI Low", "Adaptive CI Up"])
        writer.writerows(csv_rows)

def load_gt(lbl_path):
    if not Path(lbl_path).exists():
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)
    a = np.loadtxt(lbl_path, ndmin=2)
    cls = torch.tensor(a[:,0], dtype=torch.long)
    x,y,w,h = a[:,1:].T
    xyxy = torch.tensor(np.stack([x-w/2, y-h/2, x+w/2, y+h/2], 1))
    return xyxy, cls

if __name__ == "__main__":
    main()
