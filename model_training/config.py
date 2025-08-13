
"""Project configuration constants.

You can change the `YOLO_PROJECT_ROOT` environment variable to point
to another root directory without touching the code.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Root directory that contains your YOLO data, YAML, and runs directory.
# Adjust this to your environment *once* and everything else will work.
# ---------------------------------------------------------------------
ROOT: Path = Path(os.getenv("YOLO_PROJECT_ROOT", "/Volumes/L/L_PHAS0077")).expanduser()

# ---------------------------------------------------------------------
# Paths to key files/folders (relative to ROOT)
# ---------------------------------------------------------------------
DATA_YAML          = ROOT / "yolo" / "dental_data.yaml"
TRAIN_IMAGES_DIR   = ROOT / "yolo" / "dental_radiography_yolo" / "train"
TEST_IMAGES_DIR    = ROOT / "yolo" / "dental_radiography_yolo" / "test"
RUNS_DIR           = ROOT / "yolo" / "runs" / "detect"

# ---------------------------------------------------------------------
# Hardware + model defaults
# ---------------------------------------------------------------------
DEFAULT_MODEL = "yolo11s.pt"   # change to 'yolov8s.pt', etc.
DEVICE        = os.getenv("YOLO_DEVICE", "mps")  # 'cuda', 'cpu', or 'mps'
