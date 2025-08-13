import torch
from transformers import AutoModelForSemanticSegmentation
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_segformer_inference(image_path: str | Path) -> tuple[Image.Image, np.ndarray]:
    """
    Load a dental image and run semantic segmentation using a pretrained SegFormer model.
    Returns the PIL image and the predicted segmentation mask (as NumPy array).
    """
    # Load model
    model = AutoModelForSemanticSegmentation.from_pretrained("vimassaru/segformer-b0-finetuned-teeth-segmentation")
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    segmentation_mask = outputs.logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return image, segmentation_mask

def plot_segformer_segmentation(image: Image.Image, mask: np.ndarray, cmap: str = "viridis"):
    """
    Plot the original image and SegFormer segmentation mask side by side.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original X-ray")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.title("Segmented Output")

    plt.tight_layout()
    plt.show()

def batch_infer_and_save(
    source_folder: str = "./assets/source_img",
    output_folder: str = "./assets/segformer_outputs",
    pattern: str = "*.jpg",  # change to "*.png" if needed
    max_images: int = 3
):
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(source_folder.glob(pattern))[:max_images]

    for img_path in image_paths:
        print(f"🦷 Processing: {img_path.name}")
        
        image, mask = run_segformer_inference(img_path)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(image)
        axs[0].set_title("Original X-ray")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="viridis")
        axs[1].set_title("Segmented Output")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
        
        # Save and close
        output_path = output_folder / f"{img_path.stem}_segformer_result.png"
        fig.savefig(output_path)
        print(f"✅ Saved: {output_path}")
        plt.close(fig)