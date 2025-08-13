
"""Training utilities for YOLOv8.

Example
-------
>>> from yolo_project.trainer import train
>>> train(epochs=100, batch=32)
"""
from ultralytics import YOLO
from config import DATA_YAML, DEFAULT_MODEL, DEVICE

def train(
    model_name: str = DEFAULT_MODEL,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    **kwargs,
):
    """Fine‑tune a YOLOv8 model.

    Parameters
    ----------
    model_name : str
        Path to the base model weights (.pt) or model alias (e.g. 'yolov8n.pt').
    epochs : int
        Number of training epochs.
    batch : int
        Batch size.
    imgsz : int
        Image size.
    kwargs :
        Any additional ultralytics `model.train` keyword arguments.
    """
    model = YOLO(model_name)
    print(f"[trainer] Starting training on {DATA_YAML} for {epochs} epochs …")

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.2,
        cos_lr=True,
        warmup_epochs=5,
        augment=True,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        half=True,
        workers=8,
        device=DEVICE,
        **kwargs,
    )

    print("[trainer] Done ✅")
    return model
