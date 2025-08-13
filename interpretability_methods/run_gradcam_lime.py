# run_gradcam_lime.py

import os
import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import slic

from config import DATA_YAML
from gradcam_yolo_dental import CLASS_ID, IMG_ROOT, MODEL_CKPT, TargetCavity, best_device, letterbox, safe_cam

# ─── Enhanced visualize function ─────────────────────────────────────────────

def visualize_gradcam_and_lime(
    img_path: str,
    yolo: YOLO,
    cam: GradCAM,
    class_id: int,
    device: str,
    lime_samples: int = 500,
    n_segments: int = 100,
    compactness: float = 10.0
):
    # 1) Load and letterbox
    orig = cv2.imread(img_path)
    H, W = orig.shape[:2]
    pad_img, r, (px, py), _ = letterbox(orig, 640, stride=int(yolo.model.stride.max()))
    rgb_pad = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(rgb_pad).permute(2, 0, 1)[None].to(device).requires_grad_(True)

    # 2) Grad-CAM
    heat = safe_cam(cam, tensor, TargetCavity())
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hpad = int(640 - 2 * py)
    wpad = int(640 - 2 * px)
    heat = heat[py : py + hpad, px : px + wpad]
    heat = cv2.resize(heat, (W, H), cv2.INTER_LINEAR)
    gradcam_vis = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)

    # 3) Run YOLO detection (to get boxes, classes, scores)
    det = yolo.predict(tensor, device=device, verbose=False)[0]
    boxes   = det.boxes.xyxy.cpu().numpy()
    cids    = det.boxes.cls.cpu().numpy().astype(int)
    confs   = det.boxes.conf.cpu().numpy()

    # draw boxes + scores on Grad-CAM
    for (x1,y1,x2,y2), cid, conf in zip(boxes, cids, confs):
        # map back from padded coords to original
        x1 = int((x1 - px)/r); y1 = int((y1 - py)/r)
        x2 = int((x2 - px)/r); y2 = int((y2 - py)/r)
        color = (0,255,0) if cid == class_id else (160,160,160)
        cv2.rectangle(gradcam_vis, (x1,y1), (x2,y2), color, 2)
        # put confidence above box
        cv2.putText(
            gradcam_vis,
            f"{conf:.2f}",
            (x1, max(15, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA
        )

    # 4) LIME
    def lime_score_fn(batch):
        out = []
        for im in batch:
            bgr = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            r = yolo.predict(bgr, device="cpu", imgsz=640, verbose=False)[0]
            if len(r.boxes):
                ids   = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                mask_ = ids == class_id
                out.append([confs[mask_].sum() if mask_.any() else 0.0])
            else:
                out.append([0.0])
        return np.array(out)

    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(
        rgb_pad,
        classifier_fn=lime_score_fn,
        num_samples=lime_samples,
        segmentation_fn=lambda im: slic(im, n_segments=n_segments, compactness=compactness)
    )

    _, mask = exp.get_image_and_mask(
        label=0,
        positive_only=True,
        num_features=int((cids==class_id).sum()*1.1),
        hide_rest=False
    )

    # crop & resize mask back to orig
    mask_crop = mask[py:py+hpad, px:px+wpad]
    heat2     = cv2.applyColorMap((mask_crop*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat2     = cv2.resize(heat2, (W, H), cv2.INTER_LINEAR)
    lime_vis  = cv2.addWeighted(orig, 0.7, heat2, 0.3, 0)

    # draw boxes + scores on LIME
    for (x1,y1,x2,y2), cid, conf in zip(boxes, cids, confs):
        x1 = int((x1 - px)/r); y1 = int((y1 - py)/r)
        x2 = int((x2 - px)/r); y2 = int((y2 - py)/r)
        color = (0,255,0) if cid == class_id else (160,160,160)
        cv2.rectangle(lime_vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            lime_vis,
            f"{conf:.2f}",
            (x1, max(15, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA
        )

    return gradcam_vis, lime_vis


# ─── Manual groups of three examples each ────────────────────────────────────
# [3 TP, 3 FP, 3 FN]
GROUPS = {
    "0":      ["0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
                       "0078_jpg.rf.3df5e76aaf3853ffeb55283ed9666c1e.jpg",
                       "0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg",
                       "0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
                       "0056_jpg.rf.6719f5b67f55726778c11022c690ee7f.jpg",
                       "0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg",
                       "0073_jpg.rf.0a3fc448611ebb614d068de0e7bf25a6.jpg",
                       "0249_jpg.rf.5b54abe672fb6b47fa1399636d0d6aa0.jpg",
                       "0463_jpg.rf.adc570a9de38c884cdf7820e5420b0e4.jpg"],

    "1":    ["0021_jpg.rf.0d2cd31b4786f6121d2241aa8c595bf9.jpg",
                       "0035_jpg.rf.386d686c55e9347823a6e3e75af5152e.jpg",
                       "0056_jpg.rf.6719f5b67f55726778c11022c690ee7f.jpg",
                       "0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg",
                       "0288_jpg.rf.7576a9c760c217036a5ab1a7db10d2c7.jpg",
                       "0319_jpg.rf.865635b4814640d241d4ca2349d1db42.jpg",
                       "0056_jpg.rf.6719f5b67f55726778c11022c690ee7f.jpg",
                       "0583_jpg.rf.4abd49cdba147a8819d61f27d68cad37.jpg",
                       "0995_jpg.rf.7e05c8d7d34d8a551eefafbf0f9520de.jpg"],

    "3":[ "0041_jpg.rf.45bc3049f99113cefd018e407df2415f.jpg",
                        "0068_jpg.rf.8d1f95a7bd401655c7768b48eb35add5.jpg",
                        "0092_jpg.rf.922f6021eb8dd014c8924dea310fb7eb.jpg",
                          "0041_jpg.rf.45bc3049f99113cefd018e407df2415f.jpg",
                        "0068_jpg.rf.8d1f95a7bd401655c7768b48eb35add5.jpg",
                        "0092_jpg.rf.922f6021eb8dd014c8924dea310fb7eb.jpg",
                          "0068_jpg.rf.8d1f95a7bd401655c7768b48eb35add5.jpg",
                        "0164_jpg.rf.b68fccb1be35095dc0411f28d42201a2.jpg",
                        "0481_jpg.rf.b2ecb150b2a86eba181ed234d7c2ef35.jpg"],

    "2":    ["0021_jpg.rf.0d2cd31b4786f6121d2241aa8c595bf9.jpg",
                       "0035_jpg.rf.386d686c55e9347823a6e3e75af5152e.jpg",
                       "0091_jpg.rf.63aed96fec3e4c67b25d53034ed4c12d.jpg",
                       "0003_jpg.rf.ca9d51a22c825737b49a5f73d3172d23.jpg",
                       "0056_jpg.rf.6719f5b67f55726778c11022c690ee7f.jpg",
                       "0078_jpg.rf.3df5e76aaf3853ffeb55283ed9666c1e.jpg",
                       "0021_jpg.rf.0d2cd31b4786f6121d2241aa8c595bf9.jpg",
                       "0035_jpg.rf.386d686c55e9347823a6e3e75af5152e.jpg",
                       "0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg"],
}

# ─── Main runner ─────────────────────────────────────────────────────────────

def main():
    device = best_device()
    print(f"Using device: {device}")

    # load YOLO + set up GradCAM
    yolo = YOLO(MODEL_CKPT)
    net  = yolo.model.to(device).eval()
    target_layer = net.model[-2:][0]
    cam = GradCAM(model=net, target_layers=[target_layer])

    # load class names
    with open(DATA_YAML) as f:
        names = yaml.safe_load(f)["names"]
    names = names if isinstance(names, list) else list(names.values())

    # iterate groups
    for group, file_list in GROUPS.items():
        for fname in file_list[:1]:
            cls_name = names[int(group)]

            path = os.path.join(IMG_ROOT, fname)
            if not os.path.exists(path):
                print("Missing:", path)
                continue

            gradcam_im, lime_im = visualize_gradcam_and_lime(
                path, yolo, cam, int(group), device
            )

            # display side-by-side
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
            ax1.imshow(cv2.cvtColor(gradcam_im, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"{cls_name} – {fname[:5]}\nGrad-CAM")
            ax1.axis("off")

            ax2.imshow(cv2.cvtColor(lime_im, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"{cls_name} – {fname[:5]}\nLIME")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    main()
