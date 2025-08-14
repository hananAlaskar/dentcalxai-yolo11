# Sigma-Aware YOLO Detector

This module builds a **sigma-aware** YOLO model from a base architecture YAML (default inside the script: `yolo11s.yaml`), then evaluates it and exports **uncertainty-aware** predictions for downstream calibration and analysis.

**No code edits required** — follow the steps below to wire paths and run.

---

## Contents

- **Training script:** `train_with_sigma.py`  
  *(initializes the model from `yolo11s.yaml` internally and applies sigma modifications)*
- **Notebook:** `sigma_yolo_workflow.ipynb`  
  *(runs extraction of σ-aware predictions and performs calibration & uncertainty analysis)*
- **Outputs created when you run:**
  - `./runs/detect/<sigma_run_name>/…` (e.g., `weights/best.pt`, `results.csv`, plots)
  - `predictions_val_uncert.csv`, `predictions_test_uncert.csv` (CSV exports of predictions + σ)

> Datasets and large weights are not included in the repo.

---

## 1) Environment

- Python 3.10+
- Install the project dependencies:
```bash
  pip install -r requirements.txt
````

* Use Ultralytics/Torch versions compatible with your hardware and the training you intend to reproduce.

**Hardware**

* Apple Silicon works with `device="mps"`.
* NVIDIA GPUs use `device="cuda"` (e.g., `device=0`).
* CPU is fine for quick smoke tests (`device="cpu"`).

---

## 2) Dataset setup

Provide a standard YOLO **dataset YAML** (the notebook references `dental_data.yaml`):

```yaml
# dental_data.yaml (example)
path: /ABS/PATH/TO/dental_radiography_yolo
train: train/images
val:   valid/images
test:  test/images

names:
  0: Cavity
  1: Fillings
  2: Implant
  3: Impacted Tooth
```

Expected layout under `path`:

```
dental_radiography_yolo/
  train/images/ …    train/labels/ …
  valid/images/ …    valid/labels/ …
  test/images/  …    test/labels/  …
```

Place `dental_data.yaml` where the script/notebook can resolve it (repo root or use an absolute path).

---

## 3) Point to your data (no code edits)

**Option A — Dataset YAML (recommended):**
Edit `dental_data.yaml` so `path:` and the `train/val/test` subpaths resolve to your dataset on disk.

**Option B — Symlink to match absolute paths (macOS/Linux):**
If the notebook references absolute paths (e.g., under `/Volumes/...`), create symlinks so those paths resolve to your actual dataset without changing code.

---

## 4) Train (sigma-aware) — `train_with_sigma.py`

Run the training script **as-is**:

```bash
# from the repo root (or this subfolder)
python train_with_sigma.py
```

**What happens**

* The script **initializes from `yolo11s.yaml`** internally and applies the **sigma-aware** modifications.
* It trains and writes artifacts to:

  ```
  ./runs/detect/<sigma_run_name>/
  ```

  including `weights/best.pt`, `weights/last.pt`, `results.csv`, and standard PR/ROC/confusion plots.

> If your project uses env/config (e.g., `YOLO_PROJECT_ROOT`, `YOLO_DEVICE`, or `config.py`), set those first so the script picks up the correct dataset and device without editing code.

---

## 5) Extract σ-aware predictions — **in the notebook**

Open **`sigma_yolo_workflow.ipynb`** and run the extraction cells. The notebook will:

* Load the checkpoint produced in Step 4 (typically the run’s `best.pt`).
* For each split you choose (usually **val** and **test**), run inference and export:

  * `predictions_val_uncert.csv`
  * `predictions_test_uncert.csv`

**Inputs expected**

* The checkpoint path from your trained run under `./runs/detect/<sigma_run_name>/weights/best.pt`.
* The split image folders for **val** and/or **test** (as defined by `dental_data.yaml` or resolved via symlink).
* A device setting appropriate for your machine (`"mps"`, `"cuda"`, or `"cpu"`).

**Outputs expected**

* CSV files written to the notebook’s working directory (or the paths you set inside the extraction cell).

---

## 6) Calibration (in-notebook, high level + I/O)

The **Calibration** section assesses how well predicted **uncertainty (σ)** aligns with actual detection errors.

**What it does**

* Loads σ-aware prediction CSVs (val/test).
* Builds ground-truth maps from your YOLO labels.
* Matches predictions↔GT with an IoU threshold and aggregates **error vs. σ**.
* Plots **reliability diagrams** and reports **summary metrics** (e.g., ECE/MCE; optional per-class if enabled).

**Inputs**

* `predictions_val_uncert.csv` and/or `predictions_test_uncert.csv`.
* Matching `images/` and `labels/` folders for the same split.
* Calibration knobs such as `iou_thresh` (matching) and `n_bins` (diagram resolution).

**Outputs**

* Inline reliability plots and a small summary of calibration metrics.
* Batch evaluation is available to compare **val** and **test** in one go.

**Where in the notebook**

* Look for cells that call the helpers from `calibration_util`:
  `load_predictions`, `build_gt_map`, `collect_err_sigma`, `plot_reliability`, and `evaluate_on_datasets`.

---

## 7) Uncertainty analysis (in-notebook, high level + I/O)

The **Uncertainty Analysis** section explores σ behavior across detections and classes.

**What it does**

* Loads one split’s predictions + GT into analysis tables.
* Produces **σ histograms**, **confidence vs. σ** plots, and per-class summaries.
* Runs **clustering** (e.g., k-means) to group detections into low/medium/high-uncertainty regimes and summarizes them.

**Inputs**

* One prediction CSV (`predictions_val_uncert.csv` or `predictions_test_uncert.csv`).
* The aligned `images/` and `labels/` for the selected split.
* Analysis knobs such as `n_clusters` (e.g., 3), `iou_thresh`, and optional confidence filters.

**Outputs**

* Inline plots and cluster summaries with quick takeaways about where the model is over/under-confident.

**Where in the notebook**

* Look for cells that use `sigma_uncertainty_analysis`:
  `load_data`, `visualize_stats`, and `analyze_uncertainties_clusters`.

---

## 8) Pipeline at a glance — expected inputs & outputs (no code edits)

**Train (sigma-aware)**

* You run: `python train_with_sigma.py`.
* Inputs: base YAML inside the script (`yolo11s.yaml`), dataset via `dental_data.yaml`, your device setting.
* Outputs: a new run under `./runs/detect/<sigma_run_name>/` with weights, metrics, and plots.

**Extract σ-aware predictions**

* You run: the extraction cell(s) in `sigma_yolo_workflow.ipynb`.
* Inputs: the trained `best.pt`, the **val/test** image folders, and device.
* Outputs: `predictions_val_uncert.csv`, `predictions_test_uncert.csv`.

**Calibration**

* You run: the Calibration section in the notebook.
* Inputs: the prediction CSV(s) plus matching images/labels; set IoU/binning knobs as needed.
* Outputs: inline reliability plots and calibration metrics (ECE/MCE, optional per-class).

**Uncertainty analysis**

* You run: the Analysis section in the notebook.
* Inputs: one prediction CSV plus matching images/labels; set clustering/matching knobs.
* Outputs: inline σ plots and cluster summaries (low/med/high uncertainty regimes).

---

## 9) Troubleshooting

* **“data.yaml not found / images or labels missing”**
  Fix paths in `dental_data.yaml` and ensure the folder layout exists. If using symlinks, verify the link targets.
* **“mps device not available”**
  You’re not on Apple Silicon. Use a CUDA/CPU environment or adjust the device in your local notebook cells.
* **“runs/* not created” after training*\*
  Ensure `train_with_sigma.py` completed; check write permissions in the repo directory.
* **“No CSVs written”**
  Re-run the extraction cells in the notebook and confirm you’re writing to a folder you can access.
* **Plots look empty or mismatched**
  Make sure your CSV, images, and labels all point to the same split and that class names/IDs are consistent.

---

## FAQ

**Do I need a pre-trained `best.pt` to start?**
No. `train_with_sigma.py` starts from a base YAML (**`yolo11s.yaml`**) and applies sigma-aware modifications before training.

**Where do outputs go?**
Training artifacts go under `./runs/detect/<sigma_run_name>/…`.
CSV exports go to the notebook’s working directory (or the path you specify in the extraction cell).
