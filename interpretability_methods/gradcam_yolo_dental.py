"""
Grad-CAM  •  Ultralytics YOLO-v11  •  cavity class = 0
──────────────────────────────────────────────────────
• Renders a global Grad-CAM heat-map + bounding-boxes  
    • cavity (class-0) boxes → green  
    • all other classes      → grey  
• Works on Apple-silicon (MPS) with CPU fallback
• Groups images into TP / FP / FN figures
"""

import os, yaml, cv2, torch, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
import torch.nn as nn


# ────── Configuration ──────
MODEL_CKPT = "/Volumes/L/L_PHAS0077/yolo/runs/detect/train2/weights/best.pt"
DATA_YAML  = "/Volumes/L/L_PHAS0077/yolo/dental_data.yaml"
IMG_ROOT   = "/Volumes/L/L_PHAS0077/yolo/dental_radiography_yolo/valid/images"
CLASS_ID   = 0  # cavity

GROUPS = {
    "TP": ["0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
           "0078_jpg.rf.3df5e76aaf3853ffeb55283ed9666c1e.jpg",
           "0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg"],
    "FP": ["0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
           "0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
           "0056_jpg.rf.6719f5b67f55726778c11022c690ee7f.jpg"],
    "FN": ["0073_jpg.rf.0a3fc448611ebb614d068de0e7bf25a6.jpg",
           "0249_jpg.rf.5b54abe672fb6b47fa1399636d0d6aa0.jpg",
           "0463_jpg.rf.adc570a9de38c884cdf7820e5420b0e4.jpg"],
}


# ────── Helper Functions ──────
def best_device() -> str:
    if torch.backends.mps.is_available():
        try:
            torch.tensor(1., device="mps", requires_grad=True).backward()
            return "mps"
        except Exception:
            pass
    return "cpu"


def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    w, h = int(round(w0 * r)), int(round(h0 * r))
    pad_w, pad_h = new_shape[1] - w, new_shape[0] - h
    pad_w2, pad_h2 = pad_w // 2, pad_h // 2

    if (w, h) != (w0, h0):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    img = cv2.copyMakeBorder(img, pad_h2, pad_h - pad_h2,
                             pad_w2, pad_w - pad_w2,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (pad_w2, pad_h2), (w, h)


class TargetCavity:
    def __call__(self, outputs):
        raws = outputs[1] if isinstance(outputs, (list, tuple)) else [outputs]
        score = 0.
        for r in raws:
            if r.shape[1] - 5 <= CLASS_ID:
                continue
            obj = r[0, 4].sigmoid()
            cls = r[0, 5 + CLASS_ID].sigmoid()
            score += (obj * cls).sum()
        return score


def safe_cam(cam, x, target):
    try:
        return cam(x, targets=[target])[0]
    except Exception as e:
        if x.device.type == "mps":
            print("[WARN] Grad-CAM fell back to CPU →", type(e).__name__)
            cam_cpu = GradCAM(model=cam.model.to("cpu"), target_layers=cam.target_layers)
            return cam_cpu(x.cpu(), targets=[target])[0]
        raise


# ────── Main Entrypoint ──────
def main():
    device = best_device()
    print(f"Using device: {device}")

    yolo = YOLO(MODEL_CKPT)
    net = yolo.model.to(device).eval()
    target_layer = net.model[-2:][0]
    cam = GradCAM(model=net, target_layers=[target_layer])

    with open(DATA_YAML) as f:
        raw_names = yaml.safe_load(f)["names"]
    names = raw_names if isinstance(raw_names, list) else list(raw_names.values())
    cls_name = names[CLASS_ID] if CLASS_ID < len(names) else str(CLASS_ID)

    for group, file_list in GROUPS.items():
        fig, axes = plt.subplots(1, len(file_list), figsize=(5 * len(file_list), 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for idx, fname in enumerate(file_list):
            path = os.path.join(IMG_ROOT, fname)
            if not os.path.exists(path):
                axes[idx].set_title("missing"); axes[idx].axis("off"); continue

            orig = cv2.imread(path); H, W = orig.shape[:2]
            pad_img, r, (px, py), _ = letterbox(orig, 640, stride=int(net.stride.max()))
            rgb = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255
            tensor = tensor.to(device).requires_grad_(True)

            heat = safe_cam(cam, tensor, TargetCavity())
            heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = heat[py:py + int(640 - 2 * py), px:px + int(640 - 2 * px)]
            heat = cv2.resize(heat, (W, H), cv2.INTER_LINEAR)
            overlay = cv2.addWeighted(orig, 0.5, heat, 0.5, 0)

            det = yolo.predict(tensor, device=device, verbose=False)[0]
            for box, cls in zip(det.boxes.xyxy.cpu().numpy(),
                                det.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                x1 = int((x1 - px) / r); y1 = int((y1 - py) / r)
                x2 = int((x2 - px) / r); y2 = int((y2 - py) / r)
                color = (0, 255, 0) if cls == CLASS_ID else (160, 160, 160)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            axes[idx].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(fname, fontsize=8); axes[idx].axis("off")

        fig.suptitle(f"{group} — {cls_name} (Grad-CAM)", fontsize=15)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
