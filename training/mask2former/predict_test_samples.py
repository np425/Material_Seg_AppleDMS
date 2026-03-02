"""
Qualitative evaluation script for Mask2Former on Apple-DMS test set.

Picks 5 random images (seeded for reproducibility) from the test split,
runs inference, and saves side-by-side visualizations:
  Original Image | Ground Truth Mask | Predicted Mask

Uses the SAME seed and sample-selection logic as the SegFormer version
(training/predict_test_samples.py) so that outputs are directly
comparable.

Usage:
    python training/mask2former/predict_test_samples.py \
        --run_dir /data/material_classification/checkpoints/mask2former/swin-large/run4 \
        --output_dir predictions/mask2former/run4 \
        --seed 42
"""

import os
import json
import random
import argparse
from typing import Dict

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets import load_dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)


# ============================================================================
# Checkpoint Selection
# ============================================================================

def find_best_checkpoint(run_dir: str) -> str:
    """
    Find the best checkpoint from a training run directory.
    Checks trainer_state.json in the run dir and inside checkpoint subdirs.
    """
    if os.path.exists(os.path.join(run_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        return run_dir

    trainer_state_path = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        best = trainer_state.get("best_model_checkpoint")
        if best and os.path.exists(best):
            return best

    checkpoint_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            if os.path.exists(os.path.join(item_path, "model.safetensors")) or \
               os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                checkpoint_dirs.append(item_path)

    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        latest = checkpoint_dirs[-1]

        state_in_ckpt = os.path.join(latest, "trainer_state.json")
        if os.path.exists(state_in_ckpt):
            with open(state_in_ckpt, "r") as f:
                trainer_state = json.load(f)
            best = trainer_state.get("best_model_checkpoint")
            if best and os.path.exists(best):
                return best
        return latest

    return run_dir


def get_base_model_name(checkpoint_path: str) -> str:
    """Get the base model name from a checkpoint's config."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "_name_or_path" in config:
            return config["_name_or_path"]
    return "facebook/mask2former-swin-large-ade-semantic"


# ============================================================================
# Visualization
# ============================================================================

def create_color_palette(num_classes: int) -> np.ndarray:
    """Create a distinct color palette for visualization."""
    rng = np.random.RandomState(42)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    return palette


def save_comparison(
    image: Image.Image,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    id2label: Dict[int, str],
    save_path: str,
    palette: np.ndarray,
    sample_idx: int,
):
    """Save a side-by-side comparison: original | ground truth | prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # --- Original image ---
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # --- Ground truth mask ---
    gt_rgb = palette[ground_truth]
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # --- Prediction mask ---
    pred_rgb = palette[prediction]
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # --- Legend with all classes from GT and prediction ---
    all_classes = np.unique(np.concatenate([ground_truth.flatten(), prediction.flatten()]))
    patches = []
    for class_id in sorted(all_classes):
        if class_id == 0:
            continue
        color = palette[class_id] / 255.0
        label = id2label.get(class_id, f"class_{class_id}")
        patches.append(mpatches.Patch(color=color, label=label))

    if patches:
        fig.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=10,
            title="Material Classes",
            title_fontsize=12,
        )

    fig.suptitle(f"Test Sample #{sample_idx}", fontsize=13, y=0.02, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve best checkpoint
    checkpoint_path = find_best_checkpoint(args.run_dir)
    print(f"Run directory:   {args.run_dir}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Num samples: {args.num_samples}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path)

    # Load processor – try checkpoint first, fall back to base model
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        print("Loaded processor from checkpoint")
    except OSError:
        base_model = get_base_model_name(checkpoint_path)
        print(f"Processor not in checkpoint, loading from: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)

    model = model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    num_classes = len(id2label)
    print(f"Number of classes: {num_classes}")

    palette = create_color_palette(num_classes)

    # Load test set
    print("\nLoading test dataset...")
    dataset = load_dataset("AllanK24/apple-dms-materials", split="test")
    print(f"Test set size: {len(dataset)} samples")

    # Pick random samples with fixed seed (same logic as SegFormer script)
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(len(dataset)), min(args.num_samples, len(dataset))))
    print(f"Selected indices: {sample_indices}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    print("\nRunning predictions...")
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]

            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            ground_truth = np.array(sample["label"])
            original_size = ground_truth.shape[:2]  # (H, W)

            # Preprocess
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # Post-process to get semantic segmentation at original resolution
            prediction = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[original_size],
            )[0].cpu().numpy()

            # Save visualization
            save_path = os.path.join(args.output_dir, f"sample_{idx:05d}.png")
            save_comparison(
                image,
                ground_truth,
                prediction,
                id2label,
                save_path,
                palette,
                sample_idx=idx,
            )
            print(f"  [{i+1}/{len(sample_indices)}] Saved: {save_path}")

    # Save metadata for reproducibility
    meta = {
        "model": "mask2former",
        "run_dir": args.run_dir,
        "checkpoint": checkpoint_path,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "sample_indices": sample_indices,
        "dataset": "AllanK24/apple-dms-materials",
        "split": "test",
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {len(sample_indices)} comparisons saved to: {args.output_dir}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mask2Former predictions on random Apple-DMS test samples"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (best checkpoint selected automatically)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random test samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )

    args = parser.parse_args()
    main(args)
