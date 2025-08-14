# Model Training (YOLO11 / YOLOv8)

Train and validate YOLO models on the dental dataset using the project’s **existing** scripts/notebook. You can run with your own data **without changing code** by setting paths via env vars or symlinks.

## Contents
- Notebook: `run_yolo.ipynb` (training + evaluation overview)
- Snapshot: `validation_summary.csv`
- Helpers: `trainer.py`, `evaluator.py`, `visualizer.py`, `converter.py`, `config.py`

> This repo is a results-first showcase; large datasets and weights are not included.

---

## 1) Environment

- Python 3.10+
- Install:
```bash
  pip install -r requirements.txt
````

* Pin `ultralytics`, `torch`, and `torchvision` to your training versions.

**Hardware**

* GPU (CUDA) recommended; Apple Silicon (`device="mps"`) works where supported; CPU is OK for smoke tests.

---

## 2) Dataset setup

YOLO expects a **dataset YAML**:

```yaml
# dental_data.yaml
path: /ABS/PATH/TO/dental_radiography
train: train/images
val:   valid/images
test:  test/images  # optional

names:
  0: Cavity
  1: Fillings
  2: Implant
  3: Impacted Tooth
```

**Folder layout under `path`:**

```
dental_radiography/
  train/images/ ...    train/labels/ ...
  valid/images/ ...    valid/labels/ ...
  test/images/  ...    test/labels/  ...  # optional
```

### COCO → YOLO conversion (built-in)

If your annotations are in **COCO JSON**, convert them to YOLO txt labels to match the layout above:

```python
from pathlib import Path
from converter import coco_to_yolo
import os

ROOT = Path(os.getenv("YOLO_PROJECT_ROOT", "/ABS/PATH/TO/project_root"))

coco_json_path = ROOT/"dental_radiography/annotations/train.json"
images_dir     = ROOT/"dental_radiography/train"
output_dir     = ROOT/"dental_radiography/train/labels"

coco_to_yolo(coco_json_path, images_dir, output_dir)
```

* **Writes:** one `.txt` per image under the split’s `labels/` with normalized lines `<class_id> <cx> <cy> <w> <h>`.
* **Quick check:**

  ```bash
  find "$YOLO_PROJECT_ROOT/dental_radiography/train/labels" -type f | wc -l
  ```

---

## 3) Point the code to **your** files (no code edits)

**Option A — Env var (preferred):**

```bash
# macOS/Linux
export YOLO_PROJECT_ROOT=/ABS/PATH/TO/your_project_root
# Windows (PowerShell)
$env:YOLO_PROJECT_ROOT="C:\ABS\PATH\to\your_project_root"
```

**Option B — Symlink (macOS/Linux):**

```bash
mkdir -p /Volumes/L
ln -s /ABS/PATH/TO/your_project_root /Volumes/L/L_PHAS0077
```

> Launch Jupyter from the repo root so local imports resolve.

---

## 4) Train (using **our** `trainer.train`)

Minimal usage (matches `run_yolo.ipynb`). The trainer pulls dataset/model/device from **`config.py`**:

```python
from trainer import train
from config import DATA_YAML, DEFAULT_MODEL, DEVICE  # used internally

# Quick fine-tune using defaults from config.py
train(epochs=1, batch=16)
```

* Set **`DEFAULT_MODEL`** in `config.py` to your model name or **`.pt` checkpoint** (e.g., `yolo11s.pt`, `yolov8s.pt`, or a local `best.pt`).
* **`DATA_YAML`** (dataset YAML) and **`DEVICE`** are also defined in `config.py`.
* You can redirect the project root and device via env vars (e.g., `YOLO_PROJECT_ROOT`, `YOLO_DEVICE`) without changing code.

---

## 5) Evaluate all runs (using **our** `evaluator.evaluate_all_runs`)

Minimal usage (matches `run_yolo.ipynb`):

```python
import os
from pathlib import Path
from evaluator import evaluate_all_runs

ROOT = Path(os.getenv("YOLO_PROJECT_ROOT", "/Volumes/L/L_PHAS0077")).expanduser()
RUNS_DIR = ROOT/"yolo/model_training/runs/detect"

metrics = evaluate_all_runs(
    runs_dir=RUNS_DIR,
    output_csv="validation_summary.csv"
)
```

This produces/updates `validation_summary.csv` and per-run validation outputs (plots/metrics) under each run directory.

---

## 6) Visualize predictions (using **our** `visualizer.show_random_predictions`)

Quick **sanity check**: draw predicted boxes on a small random sample (shown **inline only**, **nothing is saved to disk**). Images are located via your dataset config (`config.DATA_YAML`), typically sampling from the validation split.

```python
from visualizer import show_random_predictions

show_random_predictions(
    model_path="./runs/detect/train2/weights/best.pt",  # your .pt checkpoint
    conf_thres=0.5,                                     # hide boxes below this confidence
    iou_threshold=0.5,                                  # NMS IoU threshold
    num_images=2,                                       # number of random images to show
)
```

**Parameters**

* `model_path` — path to the trained checkpoint (e.g., `…/runs/detect/<run>/weights/best.pt`).
* `conf_thres` — confidence cutoff in `[0,1]`; increase to view only high-confidence detections.
* `iou_threshold` — NMS IoU in `[0,1]`; lower values suppress more overlapping boxes.
* `num_images` — how many random samples to render inline.

**What it does**

* Loads the model at **`model_path`** (e.g., your run’s `best.pt`).
* Finds images using your dataset config (via `config.py` → `DATA_YAML`).
* Runs inference, applies **confidence filtering** and **NMS** (IoU threshold).
* Renders the selected images with **boxes + class labels + scores** for a quick visual check (use a higher `conf_thres` to see only the most confident detections).

---

## 7) Troubleshooting

* **“data.yaml not found / images/labels missing”** → Check **`DATA_YAML`** in `config.py` points to your dataset YAML, or set **`YOLO_PROJECT_ROOT`** so the path resolves. Ensure the folder layout matches the YAML.
* **GPU not detected** → Set **`DEVICE`** in `config.py` (e.g., `"cuda"`, `"cpu"`, `"mps"`) or export **`YOLO_DEVICE`**. Install a CUDA-enabled Torch if using `"cuda"`.
* **No runs found** → Verify **`RUNS_DIR`** in `config.py` and confirm training created `runs/detect/<run_name>` there.
* **Wrong model/weights used** → Ensure **`DEFAULT_MODEL`** in `config.py` is the intended `.pt` (or arch YAML).

---

## FAQ

**Do I need to edit code to change dataset, model, device, or outputs?**
No — adjust **`config.py`** (`DATA_YAML`, `DEFAULT_MODEL`, `DEVICE`, `RUNS_DIR`) or use env vars (e.g., **`YOLO_PROJECT_ROOT`**, **`YOLO_DEVICE`**).

**Where are my weights?**
Under **`RUNS_DIR`** → `detect/<run_name>/weights/{best.pt,last.pt}` (e.g., `$YOLO_PROJECT_ROOT/yolo/runs/detect/<run_name>/weights/…`).
