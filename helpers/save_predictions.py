"""
Save raw predictions for SegFormer on custom images.

Runs inference on all images in a given directory (recursively), and saves
the raw prediction arrays, original images, and metadata so that a separate
plotting script can generate visualizations without requiring the model.

Usage:
    python helpers/save_predictions.py \
        --run_dir /path/to/run/directory \
        --image_dir sample_images/poder/paper \
        --output_dir predictions/poder/paper
"""

import os
import json
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ============================================================================
# Checkpoint Selection (reused from predict_custom.py)
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
# Color Palette (must match predict_custom.py exactly)
# ============================================================================

def create_color_palette(num_classes: int) -> np.ndarray:
    """Create a distinct color palette for visualization."""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background/no-label is black
    return palette


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve best checkpoint
    checkpoint_path = find_best_checkpoint(args.run_dir)
    print(f"Run directory:   {args.run_dir}")
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
        print("Loaded processor from checkpoint")
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

    # Save global metadata (shared across all predictions)
    global_meta = {
        "id2label": {str(k): v for k, v in id2label.items()},
        "palette": palette.tolist(),
        "num_classes": num_classes,
        "checkpoint_path": checkpoint_path,
    }
    global_meta_path = os.path.join(args.output_dir, "global_metadata.json")
    with open(global_meta_path, "w") as f:
        json.dump(global_meta, f, indent=2)
    print(f"Saved global metadata to {global_meta_path}")

    # Run inference and save raw predictions
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

            # Determine output subdirectory, preserving subdirectory structure
            rel_path = os.path.relpath(img_path, args.image_dir)
            stem = Path(rel_path).stem
            rel_dir = os.path.dirname(rel_path)
            out_subdir = os.path.join(args.output_dir, rel_dir, stem)
            os.makedirs(out_subdir, exist_ok=True)

            # Save prediction mask as numpy array
            np.save(os.path.join(out_subdir, "prediction.npy"), prediction.astype(np.uint8))

            # Save original image resized to prediction resolution
            image_resized = image.resize((prediction.shape[1], prediction.shape[0]))
            image_resized.save(os.path.join(out_subdir, "original.png"))

            # Save per-image metadata
            per_image_meta = {
                "image_name": rel_path,
                "original_path": img_path,
                "prediction_shape": list(prediction.shape),
            }
            with open(os.path.join(out_subdir, "metadata.json"), "w") as f:
                json.dump(per_image_meta, f, indent=2)

    print(f"\nDone! {len(image_paths)} predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save raw SegFormer predictions on custom images"
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
        default="sample_images/poder/paper",
        help="Directory containing images to predict on (searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions/poder/paper",
        help="Directory to save raw predictions",
    )

    args = parser.parse_args()
    main(args)
