# Visual Dataset Analysis (`explore_data`)

A modular, results-first toolkit for **exploring dental detection datasets**: class distributions, example panels, and optional **mask-based** feature analysis (PCA, ratios, heatmaps).
**No code changes are required** to run this with your own images or models—just point the existing helpers to your paths.

## Contents

* **Notebook**

  * `visual_analysis.ipynb` — EDA, plots, example visualizations, and (optional) mask analysis.
* **Helpers** (imported by the notebook)

  * `label_parser.py` — read annotations CSV, compute class frequencies, filter images.
  * `annotation_visualizer.py` — sample panels, grouped boxes, class distribution plots.
  * `class_stats_utils.py` — per-class stats, ratio histograms/heatmaps.
  * `segmentation_utils.py` *(optional)* — batch mask inference & saving.
  * `mask_feature_analysis.py` *(optional)* — mask feature extraction, PCA, extremes.

> Launch Jupyter **from the repo root** so these imports resolve automatically.

## Environment

* Python 3.10+
* Install once:

  ```bash
  pip install -r requirements.txt
  ```

  If you don’t have a requirements file, typical deps are: `pandas numpy matplotlib scikit-learn opencv-python pillow`.


### Annotations CSV schema (`_annotations.csv`)

A **reference `_annotations.csv` is included in this folder**. If you want to run the notebook with your own images/annotations, make sure your CSV **matches this schema exactly**.

| Column     | Type | Description |
|------------|------|-------------|
| `filename` | str  | Image filename (e.g., `img001.png`). One row per box. |
| `width`    | int  | Image width in pixels. |
| `height`   | int  | Image height in pixels. |
| `class`    | str  | Category label (e.g., `Cavity`, `Fillings`, `Implant`, `Impacted Tooth`). |
| `xmin`     | int  | Left x (px), origin at top-left. |
| `ymin`     | int  | Top y (px). |
| `xmax`     | int  | Right x (px). |
| `ymax`     | int  | Bottom y (px). |

**Conventions:** pixel coordinates, axis-aligned boxes `[xmin,ymin,xmax,ymax]`, with  
`0 ≤ xmin < xmax ≤ width` and `0 ≤ ymin < ymax ≤ height`. `filename` must match the actual image file.

**Using your own data:** keep the exact columns; place images where helpers expect them **or** pass your paths directly, e.g.:

```python
df_images, df_masks, df_combined = run_mask_feature_analysis(
    source_folder="/PATH/TO/source_img",
    mask_folder="/PATH/TO/output_images_mask",
    output_dir="/PATH/TO/analysis_outputs",
)

```

## Choose how to point the notebook to **your** data (zero-code)

You have two ways to satisfy the paths the notebook expects:

### Option A — Symlink to match the author’s absolute paths (fastest; macOS/Linux)

```bash
# Replace right-hand sides with your actual locations
mkdir -p /Volumes/L/L_PHAS0077
ln -s /ABS/PATH/TO/dental_radiography   /Volumes/L/L_PHAS0077/dental_radiography

# Optional (images shown in preview panels)
mkdir -p /Volumes/L/L_PHAS0077/assets
ln -s /ABS/PATH/TO/your_images          /Volumes/L/L_PHAS0077/assets/source_img

# Optional (if you want mask analysis)
ln -s /ABS/PATH/TO/your_masks           /Volumes/L/L_PHAS0077/assets/output_images_mask

# Where figures/CSVs will be saved by default
mkdir -p /Volumes/L/L_PHAS0077/yolo/explore_data/analysis_outputs
```

### Option B — Pass your paths directly to the helpers (recommended; keeps code unchanged)

Most helpers accept **explicit path arguments**. For example, for mask-based analysis:

```python
from mask_feature_analysis import run_mask_feature_analysis

df_images, df_masks, df_combined = run_mask_feature_analysis(
    source_folder="/PATH/TO/source_img",          # your images
    mask_folder="/PATH/TO/output_images_mask",    # your masks (optional if skipping mask steps)
    output_dir="/PATH/TO/analysis_outputs"        # where plots/CSVs will be saved
)
```

> The same pattern applies to other helpers you’ll see in the notebook: look for arguments such as
> `images_dir`, `masks_dir`, `csv_path`, `output_dir` / `save_dir` — pass your own locations there.

## Expected dataset layout (typical)

If you use the symlink (Option A), the notebook expects something like:

```
/Volumes/L/L_PHAS0077/
 ├─ dental_radiography/
 │   ├─ train/
 │   │   ├─ images/...
 │   │   └─ _annotations.csv
 │   ├─ valid/ (optional)
 │   └─ test/  (optional)
 ├─ assets/
 │   ├─ source_img/             # images for visual previews
 │   └─ output_images_mask/     # optional: per-image masks (for mask analysis)
 └─ yolo/explore_data/analysis_outputs/   # figures & CSVs written here by default
```

> The annotations CSV schema is handled by `label_parser.py`. If you’re using Roboflow/YOLO-format exports, you should be fine.

## Quickstarts

### 1) EDA & preview panels (images only)

* Put the images you want to showcase under `assets/source_img/` (or pass your folder via a helper’s `images_dir`/`source_folder` argument).
* For dataset-wide plots, ensure your `_annotations.csv` is reachable (either under the symlinked layout or pass `csv_path="..."` to the parser).

Common helpers you’ll see used:

* `label_parser.load_annotations_csv(csv_path=...)`
* `class_stats_utils.plot_class_distribution(...)`
* `annotation_visualizer.visualize_single_class_examples(images_dir=..., output_dir=...)`
* `annotation_visualizer.visualize_multi_class_examples(images_dir=..., output_dir=...)`

### 2) Mask-based analysis (optional; with your model outputs)

You can either:

* **Pre-generate** masks to `assets/output_images_mask/`, or
* Use `segmentation_utils.batch_infer_and_save(...)` to create them (see its docstring for its own expected paths/weights).

Then run:

```python
df_images, df_masks, df_combined = run_mask_feature_analysis(
    source_folder="/PATH/TO/source_img",
    mask_folder="/PATH/TO/output_images_mask",
    output_dir="/PATH/TO/analysis_outputs"
)
```

This produces PCA plots, ratio histograms/heatmaps, and returns DataFrames you can save.

## Where are results saved?

* **Default (symlink layout):** `/Volumes/L/L_PHAS0077/yolo/explore_data/analysis_outputs`
* **When you pass `output_dir=...`:** saved under the folder you specify.
* Recommended structure inside `output_dir`:

  ```
  analysis_outputs/
    figures/
    tables/
    logs/
    cache/    # optional intermediates
  ```

### (Optional) Save DataFrames to CSV

Add a **new cell** in the notebook (no edits to existing code):

```python
from pathlib import Path
import pandas as pd

SAVE_DIR = Path("/PATH/TO/analysis_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

(df_images or pd.DataFrame()).to_csv(SAVE_DIR/"df_images.csv", index=False)
(df_masks or pd.DataFrame()).to_csv(SAVE_DIR/"df_masks.csv", index=False)
(df_combined or pd.DataFrame()).to_csv(SAVE_DIR/"df_combined.csv", index=False)
```

## Troubleshooting

* **File not found** → If you used symlinks, double-check the link targets and directory names; if you pass arguments, verify the exact paths you supplied.
* **Module not found** → Start Jupyter from the repo root so `explore_data/*.py` is importable.
* **Mask steps skipped** → Provide `mask_folder` with files whose basenames match the images, or skip mask analysis.

## FAQ

**Q: Do I need a trained detection model to run this?**
A: No. The notebook focuses on EDA and visualizations. A model is only needed if you want mask-based analysis—then either pre-generate masks or let `segmentation_utils` do it.

**Q: Can I use a completely different dataset?**
A: Yes. Either mirror/symlink to the expected layout or pass your own paths (`csv_path`, `images_dir`/`source_folder`, `mask_folder`, `output_dir`) to the helpers.

**Q: Can I change where results are saved without editing code?**
A: Yes—set `output_dir="..."` (or `save_dir="..."`) in the helper calls, e.g. in `run_mask_feature_analysis(...)`.

