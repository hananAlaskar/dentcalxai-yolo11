# DentCalXAI-YOLO11 — Results-Only Showcase

A **results-first** tour of our dental object detection & explainability project built around **YOLO11**. You can **browse outputs on GitHub**, and—if you want to reproduce locally—follow each folder’s **README** to run with your own data.

> **Heads-up**
> - The **full dataset** and **large model weights** are **not included**.
> - Notebooks render saved outputs on GitHub. Local reproduction is possible via the **per-folder READMEs** (you provide data/weights and set paths).

---

## Quick Links (open to view results)

- **Dataset Exploration**  
  - Notebook: [`explore_data/visual_analysis.ipynb`](explore_data/visual_analysis.ipynb)  
  - PDF summary: [`explore_data/visual_analysis.pdf`](explore_data/visual_analysis.pdf)  
  - **Docs**: [`explore_data/README.md`](explore_data/README.md)

- **Model Training (YOLO11 / YOLOv8)**  
  - Notebook: [`model_training/run_yolo.ipynb`](model_training/run_yolo.ipynb)  
  - Metrics snapshot: [`model_training/validation_summary.csv`](model_training/validation_summary.csv)  
  - **Docs**: [`model_training/README.md`](model_training/README.md)

- **Detector Calibration (ECE, Reliability)**  
  - Notebook: [`detector_yolo_calibration/view_dector_calibration.ipynb`](detector_yolo_calibration/view_dector_calibration.ipynb)  
  - **Docs**: [`detector_yolo_calibration/README.md`](detector_yolo_calibration/README.md)

- **Sigma-Aware Uncertainty**  
  - Notebook: [`sigma_aware_yolo_detector/sigma_yolo_workflow.ipynb`](sigma_aware_yolo_detector/sigma_yolo_workflow.ipynb)  
  - **Docs**: [`sigma_aware_yolo_detector/README.md`](sigma_aware_yolo_detector/README.md)

- **Explainability (Grad-CAM / LIME / SHAP)**  
  - Notebook: [`interpretability_methods/interpretability_Result.ipynb`](interpretability_methods/interpretability_Result.ipynb)  
  - Saved figures: [`interpretability_methods/interpretability_plots/`](interpretability_methods/interpretability_plots/)  
  - **Docs**: [`interpretability_methods/README.md`](interpretability_methods/README.md)

---

## Table of Contents

- [Repository Purpose](#repository-purpose)
- [What’s Included vs Not Included](#whats-included-vs-not-included)
- [Folder-by-Folder Guide](#folder-by-folder-guide)
- [Local Reproduction (high level)](#local-reproduction-high-level)
- [License](#license)
- [Citation](#citation)

---

## Repository Purpose

- Provide **clear, browsable evidence** of dataset characteristics, model behavior, calibration quality, uncertainty patterns, and explainability overlays.
- Keep the repo **lightweight** and **viewable online**, avoiding large data files or private assets.
- Offer **per-folder READMEs** so you can reproduce locally with your own data/weights (no code rewrites; paths are configurable).

---

## What’s Included vs Not Included

**Included**
- Notebooks with **saved outputs** (plots, tables, image overlays).
- Lightweight CSV summaries and configuration stubs.
- Reference scripts showing methodology and pipeline structure.

**Not Included**
- **Full dataset** (private/large).
- **Model weights/checkpoints** (e.g., `*.pt`).

These exclusions keep the repository focused on **result viewing**.

---

## Folder-by-Folder Guide

Open the notebooks to view results. For local runs, read each folder’s **README**.

- `explore_data/`  
  - Start: `visual_analysis.ipynb` · `visual_analysis.pdf`  
  - Docs: `explore_data/README.md` (generalized run steps, path config, `_annotations.csv` schema)  
  - Highlights: class distributions, masks, embeddings/PCA snapshots.

- `model_training/`  
  - Start: `run_yolo.ipynb` · `validation_summary.csv`  
  - Docs: `model_training/README.md` (train via our `trainer.train`, **COCO→YOLO** converter, config-driven paths, visualize)  
  - Highlights: curves, eval tables, saved detections.

- `detector_yolo_calibration/`  
  - Start: `view_dector_calibration.ipynb`  
  - Docs: `detector_yolo_calibration/README.md` (overall & per-class calibration via CLI; ECE/MCE, reliability)  
  - Highlights: reliability diagrams, per-class calibration curves.

- `sigma_aware_yolo_detector/`  
  - Start: `sigma_yolo_workflow.ipynb`  
  - Docs: `sigma_aware_yolo_detector/README.md` (train via `train_with_sigma.py` from base YAML; extract σ-aware predictions; calibrate & analyze)  
  - Highlights: σ-histograms, confidence–σ trends, CSV exports for val/test.

- `interpretability_methods/`  
  - Start: `interpretability_Result.ipynb` · figures under `interpretability_plots/`  
  - Docs: `interpretability_methods/README.md` (Overlay generation → Quantitative analysis → optional example-level **Grad-CAM/LIME/SHAP**; notes on hardcoded image/path selections and how to change them)  
  - Highlights: overlay panels, σ-aware summaries, example explainers.

**Reference scripts (context only):**
- `model_training/{trainer.py,evaluator.py,config.py,visualizer.py}`
- `detector_yolo_calibration/{unified_yolo_calibration_adaptive.py,calib_per_class.py}`
- `sigma_aware_yolo_detector/train_with_sigma.py`
- `interpretability_methods/{generate_interpretability_plots_save.py,gradcam_yolo_dental.py,run_gradcam_lime.py,shap_yolo_explainer.py,quantitative_analysis_plots.py}`
- `explore_data/{segmentation_utils.py,annotation_visualizer.py}`

---

## Local Reproduction (high level)

- **Paths & config**: folders are **config-driven**. Most READMEs support:
  - `YOLO_PROJECT_ROOT` (env var) or simple **symlinks** to mirror author paths.
  - `config.py` values such as `DATA_YAML`, `DEFAULT_MODEL`, `DEVICE`, `RUNS_DIR`.
- **COCO→YOLO**: if your labels are in COCO JSON, convert with our `converter.coco_to_yolo` (see `model_training/README.md`).
- **Devices**: Apple Silicon uses `device="mps"`; otherwise choose `cuda`/`cpu`. Each README shows where to set this.

> For exact commands/arguments and per-stage inputs/outputs, see the **folder READMEs** linked above.

---

## License

TBD (e.g., MIT). A `LICENSE` will be added to clarify reuse. Until then, please treat this as **All Rights Reserved**.

---

## Citation

If you reference these results, please cite this repository:

```

@misc{DentCalXAI\_YOLO11\_Results,
title  = {DentCalXAI-YOLO11 — Results-Only Showcase},
author = {Alaskar, Hanan and collaborators},
year   = {2025},
note   = {GitHub repository: dentcalxai-yolo11}
}

```

A formal entry will be updated once the report/paper is finalized.