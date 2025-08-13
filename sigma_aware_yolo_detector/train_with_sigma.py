# ─────────────────────────────────────────────────────────────
# 1️⃣  Monkey-patch weight_norm for deepcopy-safe σ-head integration
#     ▸ Must run before importing Ultralytics’ YOLO to avoid param issues
# ─────────────────────────────────────────────────────────────

import math
import torch.nn.utils as nn_utils
from torch.nn.utils import parametrizations as P

nn_utils.weight_norm = P.weight_norm      # ← **add this three-liner**

# ─────────────────────────────────────────────────────────────
#  your custom head swap
# ─────────────────────────────────────────────────────────────
import custom_yolo11.sigma_aware_detect as sd
import ultralytics.nn.tasks as tasks
tasks.Detect = sd.SigmaAwareDetect       # ← now uses your renamed class

from ultralytics.utils import torch_utils as ul_tt
from ultralytics.engine import trainer as ul_tr      # ← add this import

class NoOpEMA:
    def __init__(self, model, *_, **__):
        self.ema = model
        self.enabled = False
        self.updates = 0
    def update(self, *_, **__):      pass
    def update_attr(self, *_, **__): pass

ul_tt.ModelEMA = NoOpEMA         # stub for future imports
ul_tr.ModelEMA = NoOpEMA         # overwrite Trainer’s cached copy

# ─────────────────────────────────────────────────────────────
#  normal Ultralytics workflow
# ─────────────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolo11s.yaml")  # using the original config

model.train(
    data          = "dental_data.yaml",     # train/val split identical to baseline
    # epochs        = 50,                     # same training length
    epochs        = 1,                     # same training length
    imgsz         = 640,
    batch         = 16,
    mosaic        = 1.0,
    mixup         = 0.5,
    cutmix        = 0.0,
    copy_paste    = 0.5,
    auto_augment  = "randaugment",
    erasing       = 0.4,
    lr0           = 0.01,                   # baseline LR & sched
    cos_lr        = False,
    momentum      = 0.937,
    weight_decay  = 5e-4,
    # name          = "yolo11n_with_sigma"
    name          = "yolo11n_with_sigma_temp"

)
