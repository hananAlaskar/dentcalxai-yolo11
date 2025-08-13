import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from math import erf, sqrt


def load_predictions(pred_csv: str) -> pd.DataFrame:
    """
    Load σ-Head detections CSV and verify required columns.
    """
    pred = pd.read_csv(pred_csv)
    expected = {"image","x1","y1","x2","y2","sigma_x","sigma_y"}
    missing = expected - set(pred.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")
    return pred


def build_gt_map(img_folder: str, label_folder: str) -> dict:
    """
    Build map from image filename to list of GT boxes in pixel coords.
    """
    gt_map = {}
    for txt_path in glob.glob(os.path.join(label_folder, "*.txt")):
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        img_file = stem + ".jpg"
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
            gt_map[img_file] = boxes
    return gt_map


def iou_xyxy(a: tuple, b: tuple) -> float:
    """
    Compute IoU between two xyxy boxes.
    """
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


def collect_err_sigma(pred: pd.DataFrame,
                      gt_map: dict,
                      iou_thresh: float = 0.30) -> list:
    """
    Match predictions to GT (IoU>=threshold) and collect (err, sigma).
    Returns list of tuples: (error, sigma) for valid matches.
    """
    rows = []
    for _, r in pred.iterrows():
        img = r.image
        if img not in gt_map:
            continue
        pb = (r.x1, r.y1, r.x2, r.y2)
        best_iou, best_gt = 0.0, None
        for g in gt_map[img]:
            iou = iou_xyxy(pb, g)
            if iou > best_iou:
                best_iou, best_gt = iou, g
        if best_iou < iou_thresh:
            continue
        mu_x = (r.x1 + r.x2) * 0.5
        mu_y = (r.y1 + r.y2) * 0.5
        gt_x = (best_gt[0] + best_gt[2]) * 0.5
        gt_y = (best_gt[1] + best_gt[3]) * 0.5
        err = np.hypot(mu_x - gt_x, mu_y - gt_y)
        sigma = np.mean([r.sigma_x, r.sigma_y])
        rows.append((err, sigma))
    return rows


def empirical_coverage(rows: list,
                       k_vals: np.ndarray) -> np.ndarray:
    """
    Compute coverage cov(k) = P[err <= k * sigma] over grid of k-values.
    """
    errs = np.array([e for e, s in rows])
    sigs = np.array([s for e, s in rows])
    mask = (sigs > 0) & ~np.isnan(sigs)
    errs, sigs = errs[mask], sigs[mask]
    return np.array([np.mean(errs <= k * sigs) for k in k_vals])


def fit_calibration(rows: list,
                    k_ref: float = 1.0,
                    cov_ref: float = 0.68) -> tuple:
    """
    Fit two calibration factors:
      - simple: scale sigmas so empirical coverage at k_ref equals cov_ref
      - mle: sqrt(mean((error/sigma)^2))
    Returns (c_simple, c_mle).
    """
    errs = np.array([e for e, s in rows])
    sigs = np.array([s for e, s in rows])
    mask = (sigs > 0) & ~np.isnan(sigs)
    errs, sigs = errs[mask], sigs[mask]
    cov_unc = np.mean(errs <= k_ref * sigs)
    c_simple = cov_ref / cov_unc if cov_unc > 0 else 1.0
    c_mle = np.sqrt(np.mean((errs / sigs) ** 2))
    return c_simple, c_mle

def plot_reliability(rows: list,
                     k_max: float = 3.0,
                     n_k: int = 200,
                     title: str = "Reliability Curve"):
    """
    Plot uncalibrated and calibrated reliability curves from matched rows.
    Also prints RMSSE for each case.
    """
    # fit calibration factors
    c_simple, c_mle = fit_calibration(rows)
    # compute and print RMSSE
    rmsse_unc, rmsse_simple, rmsse_mle = compute_rmsse(rows, c_simple, c_mle)
    print(f"RMSSE uncalibrated = {rmsse_unc:.3f}")
    print(f"RMSSE @k=1 fix      = {rmsse_simple:.3f}")
    print(f"RMSSE MLE          = {rmsse_mle:.3f}")

    # coverage curves
    k_vals = np.linspace(0, k_max, n_k)
    cov_unc = empirical_coverage(rows, k_vals)
    rows_simple = [(e, s * c_simple) for e, s in rows]
    rows_mle    = [(e, s * c_mle)    for e, s in rows]
    cov_simple = empirical_coverage(rows_simple, k_vals)
    cov_mle    = empirical_coverage(rows_mle,    k_vals)
    ideal = np.array([erf(k / sqrt(2)) for k in k_vals])

    plt.figure(figsize=(6,6))
    plt.plot(k_vals, cov_unc, label="Uncalibrated")
    plt.plot(k_vals, cov_simple, label=f"Calibrated @1σ (c={c_simple:.2f})")
    plt.plot(k_vals, cov_mle,    label=f"Calibrated MLE (c={c_mle:.2f})")
    plt.plot(k_vals, ideal, '--', label="Ideal Gaussian CDF")
    for k in (1,2,3): plt.axvline(k, linestyle=':', color='gray')
    plt.xlabel("Predicted σ-band (k)")
    plt.ylabel("Coverage")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()





def yolo_labels_from_csv(csv_path: str,
                         img_folder: str,
                         output_folder: str):
    """
    Generate YOLO TXT labels from CSV annotations.
    CSV columns: filename,xmin,ymin,xmax,ymax,width,height,class
    """
    df = pd.read_csv(csv_path)
    os.makedirs(output_folder, exist_ok=True)
    classes = sorted(df['class'].unique())
    cls2id = {c: i for i, c in enumerate(classes)}
    for fname, group in df.groupby('filename'):
        w, h = int(group['width'].iloc[0]), int(group['height'].iloc[0])
        lines = []
        for _, r in group.iterrows():
            cx = (r.xmin + r.xmax) / 2 / w
            cy = (r.ymin + r.ymax) / 2 / h
            bw = (r.xmax - r.xmin) / w
            bh = (r.ymax - r.ymin) / h
            lines.append(f"{cls2id[r['class']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_folder, stem + ".txt")
        with open(out_path, 'w') as f:
            f.write("\n".join(lines))




def compute_rmsse(rows, c_simple, c_mle):
    """
    Compute RMSSE for uncalibrated, fixed-at-1σ, and MLE-scaled uncertainties.
    Returns a tuple (rmsse_unc, rmsse_fix, rmsse_mle).
    """
    errs = np.array([e for e, s in rows])
    sigs = np.array([s for e, s in rows])
    mask = (sigs > 0) & ~np.isnan(sigs)
    errs, sigs = errs[mask], sigs[mask]
    rmsse_unc = np.sqrt(np.mean((errs / sigs) ** 2))
    rmsse_fix = np.sqrt(np.mean((errs / (sigs * c_simple)) ** 2))
    rmsse_mle = np.sqrt(np.mean((errs / (sigs * c_mle)) ** 2))
    return rmsse_unc, rmsse_fix, rmsse_mle


def evaluate_on_datasets(val_cfg: dict,
                         test_cfg: dict,
                         iou_thresh: float = 0.30,
                         k_max: float = 3.0,
                         n_k: int = 200):
    import numpy as np
    import matplotlib.pyplot as plt
    from math import erf, sqrt

    # ——— Validation ———
    print("Stage A – validation")
    pred_csv_v     = val_cfg.get('pred_csv')     or val_cfg.get('PRED_CSV')
    img_folder_v   = val_cfg.get('img_folder')   or val_cfg.get('IMG_FOLDER')
    label_folder_v = val_cfg.get('label_folder') or val_cfg.get('LABEL_FOLDER')
    if not (pred_csv_v and img_folder_v and label_folder_v):
        raise KeyError("val_cfg missing required keys")
    gt_val = build_gt_map(img_folder_v, label_folder_v)
    rows_v = collect_err_sigma(
        load_predictions(pred_csv_v),
        gt_val,
        iou_thresh
    )
    if not rows_v:
        print("⚠️ No TP matches in validation; aborting.")
        return

    # fit
    c_simple, c_mle = fit_calibration(rows_v)
    print(f"  fitted c_simple={c_simple:.3f}  c_mle={c_mle:.3f}")

    # RMSSE on val
    rmsse_unc, rmsse_fix, rmsse_mle = compute_rmsse(rows_v, c_simple, c_mle)
    print("\nValidation set RMSSE:")
    print(f"  Uncalibrated = {rmsse_unc:.3f}")
    print(f"  @k=1 fix     = {rmsse_fix:.3f}")
    print(f"  MLE          = {rmsse_mle:.3f}")

    # ——— Test ———
    print("\nStage B – test")
    pred_csv_t     = test_cfg.get('pred_csv')     or test_cfg.get('PRED_CSV')
    img_folder_t   = test_cfg.get('img_folder')   or test_cfg.get('IMG_FOLDER')
    label_folder_t = test_cfg.get('label_folder') or test_cfg.get('LABEL_FOLDER')
    if not (pred_csv_t and img_folder_t and label_folder_t):
        raise KeyError("test_cfg missing required keys")
    gt_test = build_gt_map(img_folder_t, label_folder_t)
    rows_t = collect_err_sigma(
        load_predictions(pred_csv_t),
        gt_test,
        iou_thresh
    )
    if not rows_t:
        print("⚠️ No TP matches in test; skipping.")
        return

    # RMSSE on test
    rmsse_unc_t, rmsse_fix_t, rmsse_mle_t = compute_rmsse(rows_t, c_simple, c_mle)
    print("\nTest set RMSSE:")
    print(f"  Uncalibrated = {rmsse_unc_t:.3f}")
    print(f"  @k=1 fix     = {rmsse_fix_t:.3f}")
    print(f"  MLE          = {rmsse_mle_t:.3f}")

    # ——— Plot reliability curves ———
    k_vals = np.linspace(0, k_max, n_k)
    ideal = np.array([erf(k / sqrt(2)) for k in k_vals])

    cov_v_unc = empirical_coverage(rows_v, k_vals)
    cov_v_fix = empirical_coverage([(e, s*c_simple) for e, s in rows_v], k_vals)
    cov_v_mle = empirical_coverage([(e, s*c_mle)    for e, s in rows_v], k_vals)

    cov_t_unc = empirical_coverage(rows_t, k_vals)
    cov_t_fix = empirical_coverage([(e, s*c_simple) for e, s in rows_t], k_vals)
    cov_t_mle = empirical_coverage([(e, s*c_mle)    for e, s in rows_t], k_vals)

    # —— NEW: print coverage at k = 2.0 ——
    k_target = 2.0
    idx = np.argmin(np.abs(k_vals - k_target))
    print(f"\nEmpirical coverage @ k={k_target:.1f}")
    print(f"  Validation:    Uncalibrated = {cov_v_unc[idx]:.3f}, "
          f"Fixed@1σ = {cov_v_fix[idx]:.3f}, "
          f"MLE = {cov_v_mle[idx]:.3f}")
    print(f"  Test:          Uncalibrated = {cov_t_unc[idx]:.3f}, "
          f"Fixed@1σ = {cov_t_fix[idx]:.3f}, "
          f"MLE = {cov_t_mle[idx]:.3f}")

    # —— now do the plotting exactly as before ——
    plt.figure(figsize=(10,5))
    ax = plt.subplot(1,2,1)
    ax.plot(k_vals, cov_v_unc, label="Val • Uncalibrated")
    ax.plot(k_vals, cov_v_fix, '--', label="Val • Fixed@1σ")
    ax.plot(k_vals, cov_v_mle, ':', label="Val • MLE")
    ax.plot(k_vals, ideal, label="Ideal", color='k')
    for k in (1,2,3): ax.axvline(k, linestyle=':', color='gray')
    ax.set(title="Validation Reliability", xlabel="k·σ", ylabel="Coverage")
    ax.legend()

    ax = plt.subplot(1,2,2)
    ax.plot(k_vals, cov_t_unc, label="Test • Uncalibrated")
    ax.plot(k_vals, cov_t_fix, '--', label="Test • Fixed@1σ")
    ax.plot(k_vals, cov_t_mle, ':', label="Test • MLE")
    ax.plot(k_vals, ideal, label="Ideal", color='k')
    for k in (1,2,3): ax.axvline(k, linestyle=':', color='gray')
    ax.set(title="Test Reliability", xlabel="k·σ")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # k_vals = np.linspace(0, k_max, n_k)
    # ideal = np.array([erf(k / sqrt(2)) for k in k_vals])

    # cov_v_unc = empirical_coverage(rows_v, k_vals)
    # cov_v_fix = empirical_coverage([(e, s*c_simple) for e, s in rows_v], k_vals)
    # cov_v_mle = empirical_coverage([(e, s*c_mle)    for e, s in rows_v], k_vals)

    # cov_t_unc = empirical_coverage(rows_t, k_vals)
    # cov_t_fix = empirical_coverage([(e, s*c_simple) for e, s in rows_t], k_vals)
    # cov_t_mle = empirical_coverage([(e, s*c_mle)    for e, s in rows_t], k_vals)

    # plt.figure(figsize=(10,5))
    # ax = plt.subplot(1,2,1)
    # ax.plot(k_vals, cov_v_unc, label="Val • Uncalibrated")
    # ax.plot(k_vals, cov_v_fix, '--', label="Val • Fixed@1σ")
    # ax.plot(k_vals, cov_v_mle, ':', label="Val • MLE")
    # ax.plot(k_vals, ideal, label="Ideal", color='k')
    # for k in (1,2,3): ax.axvline(k, linestyle=':', color='gray')
    # ax.set(title="Validation Reliability", xlabel="k·σ", ylabel="Coverage")
    # ax.legend()

    # ax = plt.subplot(1,2,2)
    # ax.plot(k_vals, cov_t_unc, label="Test • Uncalibrated")
    # ax.plot(k_vals, cov_t_fix, '--', label="Test • Fixed@1σ")
    # ax.plot(k_vals, cov_t_mle, ':', label="Test • MLE")
    # ax.plot(k_vals, ideal, label="Ideal", color='k')
    # for k in (1,2,3): ax.axvline(k, linestyle=':', color='gray')
    # ax.set(title="Test Reliability", xlabel="k·σ")
    # ax.legend()

    # plt.tight_layout()
    # plt.show()


