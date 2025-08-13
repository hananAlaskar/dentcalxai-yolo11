# File: mask_clustering_analysis.py
"""
Color Histogram Feature Extraction + Embedding
for RGB dental images and their segmentation masks.

Steps:
1. Load images and masks
2. Extract color histograms
3. Reduce dimensionality with PCA, t-SNE, and UMAP
4. Build DataFrames
5. Save all outputs as CSV
4. Reduce dimensionality with PCA, t-SNE, and UMAP
5. Build DataFrames
6. Save all outputs as CSV
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_color_histogram(image_path: str | Path, bins=(8, 8, 8)) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256] * 3)
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_normalized_sum(embedding: np.ndarray) -> np.ndarray:
    emb_std = StandardScaler().fit_transform(embedding)
    summed = emb_std[:, :3].sum(axis=1)
    return MinMaxScaler().fit_transform(summed.reshape(-1, 1)).flatten()

def run_mask_feature_analysis(
    source_folder="assets/source_img",
    mask_folder="assets/output_images_mask",
    output_dir="analysis_outputs",
):
    # Paths
    IMAGE_FOLDER = Path(source_folder)
    MASK_IMAGE_FOLDER = Path(mask_folder)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load image paths
    image_paths = sorted([p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    mask_paths = sorted([p for p in MASK_IMAGE_FOLDER.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.name.startswith("._")])
    print(f"Found {len(image_paths)} source images and {len(mask_paths)} mask images.")

    # Extract histograms
    features_src = np.array([extract_color_histogram(p) for p in image_paths])
    features_mask = np.array([extract_color_histogram(p) for p in mask_paths])

    # Embedding
    pca_src = PCA(n_components=3, random_state=42)
    emb_pca = pca_src.fit_transform(features_src)
    emb_sum = compute_normalized_sum(emb_pca)
    emb_tsne = TSNE(n_components=3, random_state=42, perplexity=30).fit_transform(features_src)
    tsne_sum = compute_normalized_sum(emb_tsne)
    emb_umap = umap.UMAP(n_components=3, random_state=42).fit_transform(features_src)
    umap_sum = compute_normalized_sum(emb_umap)
    emb_mask_pca = PCA(n_components=3, random_state=42).fit_transform(features_mask)
    mask_sum = compute_normalized_sum(emb_mask_pca)

    # Build DataFrames
    df_images = pd.DataFrame({
        "PCA 1": emb_pca[:, 0], "PCA 2": emb_pca[:, 1], "PCA 3": emb_pca[:, 2],
        "tSNE 1": emb_tsne[:, 0], "tSNE 2": emb_tsne[:, 1], "tSNE 3": emb_tsne[:, 2],
        "UMAP 1": emb_umap[:, 0], "UMAP 2": emb_umap[:, 1], "UMAP 3": emb_umap[:, 2],
        "PCA Sum (Norm)": emb_sum, "tSNE Sum (Norm)": tsne_sum, "UMAP Sum (Norm)": umap_sum,
        "Image Name": [p.name for p in image_paths],
        "Image Path": [str(p.resolve()) for p in image_paths],
    })
    df_images["img_path"] = df_images["Image Path"]

    
    df_masks = pd.DataFrame({
        "PCA 1": emb_mask_pca[:, 0], "PCA 2": emb_mask_pca[:, 1], "PCA 3": emb_mask_pca[:, 2],
        "PCA Sum (Norm)": mask_sum, 
        "Image Name": [p.name for p in mask_paths],
        "Image Path": [str(p.resolve()) for p in mask_paths],
    })
    df_masks["img_path"] = df_masks["Image Path"]

    df_combined = pd.merge(
        df_images[["Image Name", "PCA 1", "img_path"]],
        df_masks[["Image Name", "PCA 1", "img_path"]],
        on="Image Name", suffixes=("_orig", "_mask")
    )

    # Export CSVs
    df_images.to_csv(OUTPUT_DIR / "images_embeddings.csv", index=False)
    df_masks.to_csv(OUTPUT_DIR / "masks_embeddings.csv", index=False)
    df_combined.to_csv(OUTPUT_DIR / "combined_embeddings.csv", index=False)
    print(f"✓ CSVs saved to {OUTPUT_DIR.resolve()}")

    return df_images, df_masks, df_combined


def compute_missing_teeth_ratio(mask_path):
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)
    total_pixels = mask_bin.size
    tooth_pixels = mask_bin.sum()
    return 1.0 - (tooth_pixels / total_pixels)


def compute_darkness_ratio(image_path, threshold=50):
    import cv2
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    dark_pixels = np.sum(img_gray < threshold)
    return dark_pixels / img_gray.size


def visualize_extremes(df, axes, image_column="img_path", name_column="Image Name", title_prefix=""):
    from PIL import Image
    import matplotlib.pyplot as plt

    for axis in axes:
        rows = {
            "Min": df.loc[df[axis].idxmin()],
            "Mid": df.loc[(df[axis] - df[axis].mean()).abs().idxmin()],
            "Max": df.loc[df[axis].idxmax()],
        }
        imgs = {k: Image.open(v[image_column]) for k, v in rows.items()}
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (k, v) in zip(axs, rows.items()):
            ax.imshow(imgs[k])
            ax.set_title(f"{axis} {k}\n{v[axis]:.4f}\n{v[name_column]}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{title_prefix}{axis}: Min vs. Mid vs. Max")
        plt.tight_layout()
        plt.show()


def visualize_combined_pca(df_combined):
    from PIL import Image
    import matplotlib.pyplot as plt

    specs = [("PCA 1_orig", "img_path_orig"), ("PCA 1_mask", "img_path_mask")]
    records = []
    for pca_col, path_col in specs:
        for kind, idx in zip(
            ["Min", "Mid", "Max"],
            [df_combined[pca_col].idxmin(),
             (df_combined[pca_col] - df_combined[pca_col].mean()).abs().idxmin(),
             df_combined[pca_col].idxmax()]
        ):
            records.append((pca_col, path_col, kind, df_combined.loc[idx]))

    fig, axs = plt.subplots(len(records), 1, figsize=(6, len(records) * 3))
    for ax, (pca_col, path_col, kind, row) in zip(axs, records):
        img = Image.open(row[path_col])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{pca_col.replace('_', ' ')} {kind}\n{row[pca_col]:.4f} — {row['Image Name']}", fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_ratios(df_combined):
    from PIL import Image
    import matplotlib.pyplot as plt

    df_combined["Missing Teeth Ratio"] = df_combined["img_path_mask"].apply(compute_missing_teeth_ratio)
    df_combined["Darkness Ratio"] = df_combined["img_path_orig"].apply(compute_darkness_ratio)

    for col in ["Missing Teeth Ratio", "Darkness Ratio"]:
        norm_col = f"{col} (Norm)"
        df_combined[norm_col] = (df_combined[col] - df_combined[col].min()) / (df_combined[col].max() - df_combined[col].min())

    specs = [
        ("Missing Teeth Ratio (Norm)", "img_path_mask", "Segmentation Mask"),
        ("Darkness Ratio (Norm)", "img_path_orig", "Original X-Ray"),
    ]
    for norm_col, path_col, label in specs:
        rows = {
            "Min": df_combined.loc[df_combined[norm_col].idxmin()],
            "Avg": df_combined.loc[(df_combined[norm_col] - df_combined[norm_col].mean()).abs().idxmin()],
            "Max": df_combined.loc[df_combined[norm_col].idxmax()],
        }
        imgs = {lbl: Image.open(r[path_col]) for lbl, r in rows.items()}
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (lbl, r) in zip(axs, rows.items()):
            ax.imshow(imgs[lbl], cmap=None if label == "Original X-Ray" else "gray")
            ax.axis("off")
            ax.set_title(f"{label} {lbl}\n{norm_col}: {r[norm_col]:.2f}\n{r['Image Name']}", fontsize=10)
        fig.suptitle(f"{label}: {norm_col} Extremes", fontsize=14)
        plt.tight_layout()
        plt.show()


def run_full_mask_feature_analysis(source_folder, mask_folder, output_dir, num_clusters=5):
    from mask_clustering_analysis import run_mask_image_clustering

    df_images, df_masks, df_combined = run_mask_image_clustering(
        source_folder=source_folder,
        mask_folder=mask_folder,
        output_dir=output_dir,
        num_clusters=num_clusters
    )

    visualize_extremes(df_images, axes=[
        "PCA 1", "PCA 2", "PCA 3",
        "tSNE 1", "tSNE 2", "tSNE 3",
        "UMAP 1", "UMAP 2", "UMAP 3"
    ], title_prefix="Original: ")

    visualize_extremes(df_masks, axes=["PCA 1", "PCA 2", "PCA 3"], title_prefix="Mask: ")

    visualize_combined_pca(df_combined)

    visualize_ratios(df_combined)

    return df_images, df_masks, df_combined


def save_ratio_histograms(df, output_dir="./analysis_outputs"):
    """
    Save annotated histograms of missing teeth ratio and darkness ratio.
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_edges = np.arange(0.0, 1.01, 0.1)
    total = len(df)

    hist_specs = [
        ("Missing Teeth Ratio (Norm)", "missing_teeth_ratio_hist.png"),
        ("Darkness Ratio (Norm)", "darkness_ratio_hist.png"),
    ]

    for col, fname in hist_specs:
        plt.figure(figsize=(6, 4))
        counts, edges, patches = plt.hist(
            df[col],
            bins=bin_edges,
        )
        plt.xticks(bin_edges)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")

        for count, left, patch in zip(counts, edges, patches):
            if count > 0:
                pct = count / total * 100
                x = left + patch.get_width() / 2
                y = count
                plt.text(
                    x, y + total * 0.001,
                    f"{int(count)}\n({pct:.1f}%)",
                    ha='center', va='bottom', fontsize=8
                )

        plt.margins(x=0.05, y=0.1)  #
        plt.tight_layout()
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path)
        plt.show()
        plt.close()
        print(f"Saved histogram for {col} at {out_path}")


def save_joint_ratio_heatmap(df, output_dir="./analysis_outputs"):
    """
    Save a 2D heatmap comparing Missing Teeth Ratio vs Darkness Ratio.
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_edges = np.arange(0.0, 1.01, 0.2)
    total = len(df)

    x = df["Missing Teeth Ratio (Norm)"]
    y = df["Darkness Ratio (Norm)"]

    plt.figure(figsize=(6, 5))
    heat_counts, xedges, yedges, img = plt.hist2d(
        x, y,
        bins=[bin_edges, bin_edges],
        cmap="YlOrRd"
    )
    plt.colorbar(img, label="Count")
    plt.xlabel("Missing Teeth Ratio (Norm)")
    plt.ylabel("Darkness Ratio (Norm)")
    plt.title("2D Density: Missing vs Darkness (Norm)")

    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            count = int(heat_counts[i, j])
            if count > 0:
                pct = count / total * 100
                xc = (xedges[i] + xedges[i + 1]) / 2
                yc = (yedges[j] + yedges[j + 1]) / 2
                plt.text(
                    xc, yc,
                    f"{count}\n{pct:.0f}%",
                    color="black",
                    ha='center', va='center',
                    fontsize=6
                )

    heatmap_path = os.path.join(output_dir, "missing_vs_darkness_heatmap.png")
    plt.savefig(heatmap_path)
    plt.show()
    plt.close()
    print(f"Saved joint-ratio heatmap at {heatmap_path}")


if __name__ == "__main__":
    run_mask_image_clustering()
