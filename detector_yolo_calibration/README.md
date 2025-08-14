# Detector Calibration (YOLO)

Calibrate a trained YOLO detector and visualize **reliability** (error vs. confidence) overall and **per-class**. This folder is documentation-only; you can run everything **without editing code** by providing the right paths as CLI flags.

---

## Contents
- **Notebook:** `view_dector_calibration.ipynb`
- **Scripts (called by the notebook):**
  - `unified_yolo_calibration_adaptive.py` — overall calibration
  - `calib_per_class.py` — per-class calibration

> Datasets and weights are not included.

---

## 1) Environment
- Python 3.10+
- Install once:
```bash
  pip install -r requirements.txt
````

* Use compatible versions of `ultralytics`, `torch`, `torchvision`.

**Hardware**

* The notebook uses `--device mps` (Apple Silicon).
  On Linux/Windows use `--device cuda` (e.g., `cuda:0`) or `--device cpu`.

---

## 2) Inputs (what the scripts expect)

* **Weights (`--weights`)**: path to your trained detector (e.g., `.../runs/detect/<run>/weights/best.pt`).
* **Dataset YAML (`--data`)**: e.g., `dental_data.yaml` that defines `train/val/test` paths & `names`.
* **Device (`--device`)**: `mps`, `cuda`, or `cpu`.
* **Optional**: `--bootstrap` to compute confidence intervals for calibration metrics.

> The example notebook uses:
>
> ```
> --weights "/Volumes/L/L_PHAS0077/yolo/runs/detect/train2/weights/best.pt"
> --data dental_data.yaml
> --device mps
> --bootstrap
> ```

**Zero-code path options**

* Update the flags to your absolute paths, **or**
* Create a symlink so `/Volumes/L/L_PHAS0077/...` resolves to your dataset/runs.

---

## 3) Run (same commands as the notebook)

**Overall calibration**

```bash
python unified_yolo_calibration_adaptive.py \
  --weights "/ABS/PATH/TO/runs/detect/<run>/weights/best.pt" \
  --data dental_data.yaml \
  --device mps \
  --bootstrap
```

**Per-class calibration**

```bash
python calib_per_class.py \
  --weights "/ABS/PATH/TO/runs/detect/<run>/weights/best.pt" \
  --data dental_data.yaml \
  --device mps \
  --bootstrap
```

---

## 4) What you get (outputs)

* **Reliability diagrams** (overall and per-class).
* **Calibration metrics** such as ECE/MCE (and, if enabled, bootstrap intervals).
* **CSV summaries** and/or images saved by each script to their default output folders (the scripts print the save path on completion).

> If you don’t see files where you expect them, check the script logs; outputs are written to the working directory or a script-defined subfolder (e.g., a `calibration/` or `runs/.../calibration/` directory).

---

## 5) Pipeline — expected inputs & outputs

### A) Overall calibration

* **Inputs:** `--weights`, `--data`, `--device`, optional `--bootstrap`
* **Outputs:** overall reliability plot(s), overall ECE/MCE (+ optional CI)

### B) Per-class calibration

* **Inputs:** same flags as above
* **Outputs:** per-class reliability plot(s), per-class ECE/MCE (+ optional CI), plus any CSV tables written by the script

---

## 6) Troubleshooting

* **“data.yaml not found / images or labels missing”**
  Ensure the paths in `dental_data.yaml` are correct and the folder layout exists.
* **“mps not available”**
  Use `--device cuda` (GPU) or `--device cpu` on non-Apple machines.
* **“weights not found”**
  Confirm the exact path to `best.pt` under your `runs/detect/<run>/weights/`.
* **“No plots/CSVs produced”**
  Re-run from the project root; check console output for the **save directory** each script reports.

---

## FAQ

**Do I need to edit any code to change paths?**
No. Pass your paths via `--weights` and `--data`, and choose `--device` for your machine.

**Can I run both overall and per-class calibration?**
Yes—run both commands above; they complement each other.

**Where are files saved?**
Where the scripts write them (printed at the end). If you prefer a specific directory, change into that directory before running, or use any script option that controls output (if provided in your version).



