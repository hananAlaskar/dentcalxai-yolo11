
# DentCalXAI-YOLO11 — Results-Only Showcase

A **results-first, read-only** tour of our dental object detection & explainability project built around **YOLO11** . This repo is intentionally curated so you can **see the outputs in notebooks on GitHub** without needing the dataset or model weights.

> **Heads-up**
> - The **full dataset** and **all large model files/weights** are **not included**.
> - The code here is **not meant to be runnable** out of the box; scripts are kept for reference and context.

---

## Quick Links (Open to View Results)

- **Dataset Exploration**  
  - Notebook: [`explore_data/visual_analysis.ipynb`](explore_data/visual_analysis.ipynb)  
  - PDF summary: [`explore_data/visual_analysis.pdf`](explore_data/visual_analysis.pdf)

- **Model Training (YOLO11 / YOLOv8)**  
  - Notebook: [`model_training/run_yolo.ipynb`](model_training/run_yolo.ipynb)  
  - Metrics snapshot: [`model_training/validation_summary.csv`](model_training/validation_summary.csv)

- **Detector Calibration (ECE, Reliability)**  
  - Notebook: [`detector_yolo_calibration/view_dector_calibration.ipynb`](detector_yolo_calibration/view_dector_calibration.ipynb)

- **Sigma-Aware Uncertainty**  
  - Notebook: [`sigma_aware_yolo_detector/sigma_yolo_workflow.ipynb`](sigma_aware_yolo_detector/sigma_yolo_workflow.ipynb)

- **Explainability (Grad-CAM / LIME / SHAP)**  
  - Notebook: [`interpretability_methods/interpretability_Result.ipynb`](interpretability_methods/interpretability_Result.ipynb)  
  - Saved figures: [`interpretability_methods/interpretability_plots/`](interpretability_methods/interpretability_plots/)


---

## Table of Contents

- [Repository Purpose](#repository-purpose)
- [What’s Included vs Not Included](#whats-included-vs-not-included)
- [Folder-by-Folder Guide](#folder-by-folder-guide)
- [License](#license)
- [Citation](#citation)

---

## Repository Purpose

- Provide **clear, browsable evidence** of dataset characteristics, model behavior, calibration quality, uncertainty patterns, and explainability overlays.
- Keep the repo **lightweight** and **viewable online**, avoiding large data files or private assets.
- Maintain **context scripts** for readers who want to understand how figures/metrics were generated (even though they are not meant to be run here).

---

## What’s Included vs Not Included

**Included**
- Notebooks with **saved outputs** (plots, tables, image overlays).
- Lightweight CSV summaries and configuration stubs.
- Reference scripts to show methodology and pipeline structure.

**Not Included**
- **Full dataset** (private/large).
- **Model weights & checkpoints** (e.g., `yolo11s.pt`, `yolov8n.pt`).

These exclusions are intentional to keep the repository focused on **result viewing**.

---

## Folder-by-Folder Guide

**Open these notebooks first — they already include saved outputs (no execution needed).**

| Folder | Open first | Key visuals & metrics |
|---|---|---|
| `explore_data/` | [`visual_analysis.ipynb`](explore_data/visual_analysis.ipynb) · [`visual_analysis.pdf`](explore_data/visual_analysis.pdf) | Class distributions, sample images/masks, embeddings/PCA snapshots |
| `model_training/` | [`run_yolo.ipynb`](model_training/run_yolo.ipynb) · [`validation_summary.csv`](model_training/validation_summary.csv) | Train/val curves, eval tables, sample detections |
| `detector_yolo_calibration/` | [`view_dector_calibration.ipynb`](detector_yolo_calibration/view_dector_calibration.ipynb) | Reliability diagrams, ECE/MCE, per-class calibration curves |
| `sigma_aware_yolo_detector/` | [`sigma_yolo_workflow.ipynb`](sigma_aware_yolo_detector/sigma_yolo_workflow.ipynb) | σ-uncertainty histograms, confidence–σ trends, qualitative examples | 
| `interpretability_methods/` | [`interpretability_Result.ipynb`](interpretability_methods/interpretability_Result.ipynb) · [`interpretability_plots/`](interpretability_methods/interpretability_plots/) | Grad-CAM/LIME/SHAP overlays, aggregate plots |

**Reference scripts (context only, not meant to run here):**
- `model_training/{trainer.py,evaluator.py,config.py,visualizer.py}`
- `detector_yolo_calibration/{unified_yolo_calibration_adaptive.py,calib_per_class.py}`
- `interpretability_methods/{gradcam_yolo_dental.py,shap_yolo_explainer.py,quantitative_analysis_plots.py}`
- `explore_data/{segmentation_utils.py,annotation_visualizer.py}`


**At a glance — what to look for**
- **EDA**: category distributions, sample images, segmentation masks, embedding visualizations.
- **Training**: curves, evaluation tables, qualitative detections (saved as images in notebook outputs).
- **Calibration**: reliability diagrams, ECE/MCE metrics (before/after), per-class curves.
- **Uncertainty**: σ-aware training artifacts, histograms, confidence vs. uncertainty relationships.
- **Explainability**: Grad-CAM/LIME/SHAP overlays, per-class interpretability summaries.


## License

TBD (e.g., MIT). A `LICENSE` file will be added to clarify reuse. Until then, please treat this as **All Rights Reserved**.

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

---

