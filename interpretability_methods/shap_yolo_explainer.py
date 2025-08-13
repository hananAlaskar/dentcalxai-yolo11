import os, glob, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shap
from skimage.segmentation import slic
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam.utils.image import show_cam_on_image


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize the image."""
    pil_img = Image.open(path).convert("RGB")
    img_float = np.array(pil_img).astype(np.float32) / 255.0
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_float, img_uint8


def get_segments(img_uint8: np.ndarray, n_segments=250, compactness=10, sigma=1.0) -> tuple[np.ndarray, np.ndarray, int]:
    """SLIC segmentation + upsample to full size"""
    initial_segments = slic(img_uint8, n_segments=n_segments, compactness=compactness, sigma=sigma)
    unique_labels, inverse = np.unique(initial_segments, return_inverse=True)
    segments_small = inverse.reshape(initial_segments.shape)
    S = unique_labels.shape[0]

    H_img, W_img = img_uint8.shape[:2]
    H_seg, W_seg = segments_small.shape

    if (H_seg, W_seg) != (H_img, W_img):
        segments = cv2.resize(
            segments_small.astype(np.int32), (W_img, H_img), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
    else:
        segments = segments_small.copy()
    return segments, img_uint8, S


def build_baseline(img_uint8: np.ndarray, segments: np.ndarray, S: int) -> np.ndarray:
    """Build a baseline image with superpixels averaged."""
    baseline_img = img_uint8.copy()
    for i_sp in range(S):
        mask_sp = (segments == i_sp)
        if mask_sp.sum() == 0:
            continue
        mean_color = img_uint8[mask_sp].mean(axis=0).astype(np.uint8)
        baseline_img[mask_sp] = mean_color
    return baseline_img


def reconstruct_from_superpixels(z: np.ndarray, segments: np.ndarray, baseline_img: np.ndarray, img_uint8: np.ndarray) -> np.ndarray:
    """Return an image with superpixels included/excluded by z."""
    out = img_uint8.copy()
    for i_sp in range(len(z)):
        if z[i_sp] == 0:
            out[segments == i_sp] = baseline_img[segments == i_sp]
    return out


def build_predict_fn(model, segments, baseline_img, img_uint8, target_class_id):
    """Wraps the YOLO model with a SHAP-compatible predict function."""

    def shap_predict(z_batch: np.ndarray) -> np.ndarray:
        M = z_batch.shape[0]
        scores = np.zeros((M,), dtype=np.float32)
        for i in range(M):
            img_rec = reconstruct_from_superpixels(z_batch[i], segments, baseline_img, img_uint8)
            results = model.predict(
                source=img_rec,
                device="cpu",
                imgsz=640,
                augment=False,
                verbose=False,
                show=False,
                save=False
            )
            det = results[0]
            class_ids = det.boxes.cls.cpu().numpy().astype(int)
            confs = det.boxes.conf.cpu().numpy()
            mask = (class_ids == target_class_id)
            scores[i] = float(confs[mask].max()) if mask.sum() > 0 else 0.0
        return scores.reshape(-1, 1)

    return shap_predict


def compute_shap(model_path: str, img_path: str, target_class_id: int):
    print(f"[INFO] Explaining image: {os.path.basename(img_path)}")

    model = YOLO(model_path)
    img_float, img_uint8 = load_image(img_path)
    segments, img_uint8, S = get_segments(img_uint8)
    baseline_img = build_baseline(img_uint8, segments, S)

    background = np.vstack([
        np.zeros((1, S), dtype=int),
        (np.random.rand(10, S) > 0.5).astype(int)
    ])
    predict_fn = build_predict_fn(model, segments, baseline_img, img_uint8, target_class_id)

    explainer = shap.KernelExplainer(predict_fn, background, link="identity")

    start_time = time.time()
    z_orig = np.ones((1, S), dtype=int)
    shap_values = explainer.shap_values(z_orig, nsamples=300)
    elapsed = time.time() - start_time
    print(f"[INFO] SHAP explanation completed in {elapsed:.2f} seconds")

    phi_arr = np.array(shap_values[0]).flatten()
    heatmap_shap = np.zeros((img_float.shape[0], img_float.shape[1]), dtype=np.float32)
    valid_len = min(S, phi_arr.shape[0])
    for i_sp in range(valid_len):
        val = phi_arr[i_sp]
        if val > 0:
            heatmap_shap[segments == i_sp] = val
    if heatmap_shap.max() > 0:
        heatmap_shap /= heatmap_shap.max()

    overlay_shap = show_cam_on_image(img_float, heatmap_shap, use_rgb=True, image_weight=0.5)
    plt.figure(figsize=(6, 6))
    plt.title(f"SHAP (KernelExplainer) for class {target_class_id}")
    plt.imshow(overlay_shap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def find_first_image(folder: str, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> str:
    for ext in exts:
        files = glob.glob(os.path.join(folder, f"*{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError("No image found in folder.")


def main():

    model_path = "/Volumes/L/L_PHAS0077/yolo/runs/detect/train2/weights/best.pt"
    image_folder = "/Volumes/L/L_PHAS0077/yolo/dental_radiography_yolo/valid/images"
    target_class_id = 0  
    image_path = image_folder+"/0098_jpg.rf.66a48777bc22515364d7b419d5441796.jpg"
    compute_shap(model_path, image_path, target_class_id)


if __name__ == "__main__":
    main()
