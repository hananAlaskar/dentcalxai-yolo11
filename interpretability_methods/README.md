
# Interpretability Methods

Generate visual and quantitative **interpretability** artifacts (overlays, Grad-CAM + LIME examples, SHAP overlays, and summary stats) for a trained YOLO detector.

This folder is **documentation-only**. You can run the notebook and scripts as they are; to use your own data, **only edit the top-of-cell variables in the notebook and the clearly marked constants in the two small scripts below** (no code rewrites needed).

---

## Contents

* **Notebook:** `interpretability_Result.ipynb`
* **Scripts used by the notebook / you may run directly:**

  * `run_gradcam_lime.py` → Grad-CAM + LIME side-by-side examples
  * `shap_yolo_explainer.py` → SHAP (KernelExplainer) overlays
* **Other helpers invoked by the notebook (not edited here):**

  * `generate_interpretability_plots_save.py`, `quantitative_analysis_plots.py`
  * `gradcam_yolo_dental.py` (device helpers, paths, utils)
  * `main.py` (example generators), etc.

> Datasets and weights aren’t included. Point paths to your own assets as described below.

---

## Environment

* Python 3.10+
* Install:

  ```bash
  pip install -r requirements.txt
  ```
* Typical libs: `ultralytics`, `torch`, `opencv-python`, `matplotlib`, `pandas`,
  `pytorch-grad-cam`, `lime`, `scikit-image`, `shap`.

**Device**
The notebook uses `DEVICE` from your `config.py` (e.g., `"mps"`, `"cuda"`, `"cpu"`). Set once and re-run.

---

## How the notebook is organized (high level)

1. **Overlay generation** (records + images) via `generate_interpretability_plots_save.generate_interpretability_plots(...)`

   * Outputs to `interpretability_plots/all/` (incl. `records_overlay_stats_all_classes.csv`).

2. **Quantitative analysis** with `quantitative_analysis_plots`

   * Consumes `interpretability_plots/all/records_overlay_stats_all_classes.csv` + `combined_embeddings_with_ratios_norm.csv`.
   * Outputs to `interpretability_plots/quantitative_analysis_plots/`.

3. **Example-level explainers** (Grad-CAM + LIME, SHAP)

   * Uses the two small scripts below; **these contain hardcoded selections** you should adjust.

---

## Hardcoded selections you may want to change

### A) `run_gradcam_lime.py` (Grad-CAM + LIME)

**Where the selections live**

* **Image root & model** come from `gradcam_yolo_dental.py`:

  * `IMG_ROOT` → base folder for images (e.g., `…/valid/images`)
  * `MODEL_CKPT` → YOLO checkpoint to load (e.g., `…/runs/detect/<run>/weights/best.pt`)
* **Class names** come from `DATA_YAML` (read in `run_gradcam_lime.py`):

  ```python
  from config import DATA_YAML
  with open(DATA_YAML) as f:
      names = yaml.safe_load(f)["names"]  # list or dict
  ```
* **Hardcoded group lists** (per class id) inside `run_gradcam_lime.py`:

  ```python
  GROUPS = {
      "0": ["0003_jpg.rf....jpg", "0078_jpg.rf....jpg", ...],  # class id 0
      "1": ["0021_jpg.rf....jpg", "0035_jpg.rf....jpg", ...],  # class id 1
      "3": ["0041_jpg.rf....jpg", "0068_jpg.rf....jpg", ...],  # class id 3
      "2": ["0021_jpg.rf....jpg", "0035_jpg.rf....jpg", ...],  # class id 2
  }
  ```

  **Note:** the script processes **only the first image** per list by default:

  ```python
  for fname in file_list[:1]:  # ← limited to one example per class id
      ...
  ```

**What to change (for your data)**

* Update **`IMG_ROOT`** in `gradcam_yolo_dental.py` to point to your split (e.g., `…/valid/images`).
* Replace the **filenames in `GROUPS`** with basenames that actually exist under `IMG_ROOT`.
* If you want more than one example per class, increase the slice `[:1]` to `[:k]` or remove the slice.

**Behavioral knobs**

* LIME runs on **CPU** inside the scoring function (by design). Use:

  * `lime_samples` (default 500), `n_segments` (default 100), `compactness` (default 10.0)
    to trade speed vs. quality.
* Grad-CAM target layer is taken as the last conv block (`net.model[-2:][0]`).

**Output**

* Shows two panels (Grad-CAM, LIME) **inline** via `plt.show()`.
  (No images are saved by default.)

---

### B) `shap_yolo_explainer.py` (SHAP KernelExplainer)

**Where the selections live**

* Hardcoded variables in `main()`:

  ```python
  model_path      = "/Volumes/.../runs/detect/train2/weights/best.pt"
  image_folder    = "/Volumes/.../dental_radiography_yolo/valid/images"
  target_class_id = 0
  image_path      = image_folder + "/0098_jpg.rf....jpg"
  ```

  The rest of the script constructs a SHAP explainability pipeline:

  * SLIC superpixels (`n_segments=250`, `compactness=10`, `sigma=1.0`)
  * A baseline image built by averaging superpixels
  * **CPU** predictions in the SHAP predict fn
  * `nsamples=300` for `KernelExplainer.shap_values(...)`

**What to change (for your data)**

* Set **`model_path`** to your run’s `best.pt`.
* Set **`image_folder`** and **`image_path`** to an existing image in that folder.
* Set **`target_class_id`** (index into your class list from `DATA_YAML`).

**Performance tips**

* To speed up: reduce `nsamples` (e.g., 100–200) or `n_segments`; SHAP here is **CPU-heavy**.
* Output is displayed **inline** via `plt.show()` (not saved).

---

## Minimal run steps

1. Open **`interpretability_Result.ipynb`** and run the overlay & analysis sections in order:
   **Overlay generation → Quantitative analysis → (Optional) Example-level explainers**.

2. For **Grad-CAM + LIME** examples:

   * Edit `IMG_ROOT` (in `gradcam_yolo_dental.py`) and the `GROUPS` filenames (in `run_gradcam_lime.py`).
   * Optionally increase the per-class slice from `[:1]` to see more examples.
   * Run:

     ```bash
     python run_gradcam_lime.py
     ```

3. For **SHAP** examples:

   * Edit `model_path`, `image_folder`/`image_path`, and `target_class_id` in `shap_yolo_explainer.py`.
   * Run:

     ```bash
     python shap_yolo_explainer.py
     ```

---

## What gets saved & where

* **Overlay generation** (from notebook):
  `interpretability_plots/all/` — overlay images + `records_overlay_stats_all_classes.csv`
* **Quantitative analysis** (from notebook):
  `interpretability_plots/quantitative_analysis_plots/` — plots & summary tables
* **Grad-CAM + LIME / SHAP**:

  * Displayed **inline** by default (no files saved unless you add a save path).

---

## Troubleshooting

* **Missing image errors**
  Your `GROUPS` filenames don’t exist under `IMG_ROOT`, or `image_path` is wrong—use basenames present in your actual split.
* **Wrong classes/labels**
  Make sure `DATA_YAML` has the right `names` order; class IDs used in scripts must match these indices.
* **Very slow runs**
  LIME/SHAP run on **CPU** in the current setup. Reduce `lime_samples`, `n_segments`, or SHAP `nsamples` for quicker iterations.
* **Blank/low-signal heatmaps**
  Lower `lime_samples`/segments too far or wrong `target_class_id` can cause sparse/empty masks; try more segments or check that detections exist for that class in the selected image.

