"""
Prediction script for SegFormer on custom images.

Runs inference on all images in a given directory (recursively), and saves
visualizations showing the original image alongside the predicted segmentation
mask with class labels.

Usage:
    python helpers/predict_custom.py \
        --run_dir /path/to/run/directory \
        --image_dir sample_images/poder \
        --output_dir predictions/poder
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ============================================================================
# Checkpoint Selection (reused from evaluate_test.py)
# ============================================================================

def find_best_checkpoint(run_dir: str) -> str:
    """
    Find the best checkpoint from a training run directory.

    Looks at trainer_state.json to find the checkpoint with best metric.
    Checks both the run directory and inside checkpoint subdirectories.
    """
    # Check if this is already a checkpoint directory
    if os.path.exists(os.path.join(run_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        return run_dir

    # Look for trainer_state.json in run directory
    trainer_state_path = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)

        best_checkpoint = trainer_state.get("best_model_checkpoint")
        if best_checkpoint and os.path.exists(best_checkpoint):
            return best_checkpoint

    # Look for checkpoints in subdirectories
    checkpoint_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            if os.path.exists(os.path.join(item_path, "model.safetensors")) or \
               os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                checkpoint_dirs.append(item_path)

    # Check trainer_state.json inside the latest checkpoint
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoint_dirs[-1]

        trainer_state_in_ckpt = os.path.join(latest_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_in_ckpt):
            with open(trainer_state_in_ckpt, "r") as f:
                trainer_state = json.load(f)

            best_checkpoint = trainer_state.get("best_model_checkpoint")
            if best_checkpoint and os.path.exists(best_checkpoint):
                return best_checkpoint

        return latest_checkpoint

    # Fallback
    return run_dir


# ============================================================================
# Visualization
# ============================================================================

def create_color_palette(num_classes: int) -> np.ndarray:
    """Create a distinct color palette for visualization."""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background/no-label is black
    return palette


def save_prediction(
    image: Image.Image,
    prediction: np.ndarray,
    id2label: Dict[int, str],
    save_path: str,
    palette: np.ndarray,
    image_name: str = "",
):
    """
    Save a visualization with the original image, colored segmentation mask,
    and a blended overlay, plus a legend listing all detected classes.
    """
    # Create blended overlay (image + mask at 50% opacity)
    pred_rgb = palette[prediction]
    overlay = np.array(image.resize((prediction.shape[1], prediction.shape[0])))
    blended = (overlay * 0.5 + pred_rgb * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Segmentation mask
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Predicted Segmentation", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(blended)
    axes[2].set_title("Overlay", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Create legend for detected classes
    unique_classes = np.unique(prediction)
    patches = []
    for class_id in sorted(unique_classes):
        if class_id == 0:  # Skip background
            continue
        color = palette[class_id] / 255.0
        label = id2label.get(class_id, f"class_{class_id}")
        # Calculate percentage of image covered by this class
        pct = 100.0 * np.sum(prediction == class_id) / prediction.size
        patches.append(mpatches.Patch(color=color, label=f"{label} ({pct:.1f}%)"))

    if patches:
        fig.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=10,
            title="Detected Materials",
            title_fontsize=12,
        )

    if image_name:
        fig.suptitle(image_name, fontsize=12, y=0.02, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Image Collection
# ============================================================================

def collect_images(image_dir: str) -> List[str]:
    """Recursively collect all image files from a directory."""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
    return image_paths


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve best checkpoint
    checkpoint_path = find_best_checkpoint(args.run_dir)
    print(f"Run directory:  {args.run_dir}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)

    # Load processor
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        print(f"Loaded processor from checkpoint")
    except OSError:
        base_model = "nvidia/segformer-b5-finetuned-ade-640-640"
        print(f"Processor not in checkpoint, loading from: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)

    model = model.to(device)
    model.eval()

    # Label mappings
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    num_classes = len(id2label)
    print(f"Number of classes: {num_classes}")

    palette = create_color_palette(num_classes)

    # Collect images
    image_paths = collect_images(args.image_dir)
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    print("Running predictions...")
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting"):
            image = Image.open(img_path).convert("RGB")
            original_size = image.size[::-1]  # (H, W)

            # Preprocess
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # Upsample logits to original resolution and get prediction
            logits = outputs.logits  # [1, C, H/4, W/4]
            upsampled = nn.functional.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            prediction = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

            # Determine output filename, preserving subdirectory structure
            rel_path = os.path.relpath(img_path, args.image_dir)
            out_name = Path(rel_path).with_suffix(".png")
            save_path = os.path.join(args.output_dir, out_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save visualization
            save_prediction(
                image,
                prediction,
                id2label,
                save_path,
                palette,
                image_name=rel_path,
            )

    print(f"\nDone! {len(image_paths)} predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SegFormer predictions on custom images"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (best checkpoint selected automatically)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images to predict on (searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    main(args)
