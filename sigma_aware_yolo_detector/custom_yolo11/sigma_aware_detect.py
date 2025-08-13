import torch
import torch.nn as nn
from ultralytics.nn.tasks import Detect

class SigmaAwareDetect(Detect):
    """
    Detect head that, in addition to the normal YOLO outputs, predicts
    log-σx and log-σy for every grid cell of every detection head.
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_convs = None          # nn.ModuleList built lazily (one per fmap)
        self._last_var_maps = None     # list of σ-maps cached at inference

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x):
        # ── stride-probe path (Ultralytics passes a dummy tensor) ─────────
        if isinstance(x, torch.Tensor):
            return super().forward(x)

        # ── normal detect path ────────────────────────────────────────────
        z = super().forward(x)         # standard YOLO boxes first

        # ── build a dedicated 1×1→ReLU→1×1→Softplus for EACH feature map ─
        if self.var_convs is None:
            self.var_convs = nn.ModuleList()
            for feat in x:             # x is a list of 3 fmaps for YOLO-11
                C = feat.shape[1]      # channel dim differs per fmap
                self.var_convs.append(
                    nn.Sequential(
                        nn.Conv2d(C, C, 1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(C, 2, 1, bias=True),   # σx, σy (A = 1)
                        nn.Softplus()
                    ).to(feat.device)
                )

        # ── run the matching conv on each fmap and cache the σ-maps ──────
        self._last_var_maps = [vc(f) for vc, f in zip(self.var_convs, x)]
        #  list length == nl (3); each element: [B, 2, Hi, Wi]

        return z
