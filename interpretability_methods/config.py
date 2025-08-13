"""
All central paths and tunables in one place.
Edit these to fit your directory layout.
"""
from pathlib import Path
import torch

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
PROJECT_ROOT      = Path("/Volumes/L/L_PHAS0077/yolo")          # change once
MODEL_CHECKPOINT  = PROJECT_ROOT / "runs/detect/train2/weights/best.pt"
VAL_IMAGES_DIR    = PROJECT_ROOT / "dental_radiography_yolo/valid/images"
VAL_LABELS_DIR    = PROJECT_ROOT / "dental_radiography_yolo/valid/labels"
DATA_YAML         = PROJECT_ROOT / "dental_data.yaml"

OUTPUT_DIR        = PROJECT_ROOT / "interpretability_methods/interpretability_plots"     # sub‑dir per class

# --------------------------------------------------------
# Detection / matching
# --------------------------------------------------------
IOU_THRESH  : float = 0.50          # IoU for TP/FP/FN split
DEVICE      : str   = "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------------------------------------
# Heat‑map parameters
# --------------------------------------------------------
# Occlusion
OCC_PATCH   : int   = 16
OCC_STRIDE  : int   = 16
BASELINE_COLOR      = 127
# LIME
LIME_SAMPLES: int   = 2000
LIME_SUPERPIX: int  = 350
LIME_TOP_FEAT: int  = 4
# RISE
RISE_NUM_MASKS: int = 1200
RISE_KEEP_PROB: float = 0.5
RISE_LOW_CONF : float = 0.05
RISE_BATCH_SZ : int   = 32
RISE_IMG_SIZE : int   = 640
RISE_MASK_CELL: int   = 16
