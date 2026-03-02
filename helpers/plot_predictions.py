"""
Generate paper-quality visualizations from saved SegFormer predictions.

Reads raw prediction arrays and metadata produced by save_predictions.py,
and produces publication-ready PDF figures with the same layout/colors as
predict_custom.py but with larger fonts, legends, and higher DPI.

Usage:
    python helpers/plot_predictions.py \
        --predictions_dir predictions/poder/paper \
        --output_dir plots/poder/paper
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# Visualization — paper-quality version of predict_custom.py
# ============================================================================

def save_paper_figure(
    image: np.ndarray,
    prediction: np.ndarray,
    id2label: Dict[int, str],
    save_path: str,
    palette: np.ndarray,
    image_name: str = "",
):
    """
    Save a paper-quality 3-panel figure (Input | Predicted Segmentation | Overlay).
    """
    # Coloured segmentation mask
    pred_rgb = palette[prediction]

    # Blended overlay (image + mask at 50 % opacity)
    blended = (image * 0.5 + pred_rgb * 0.5).astype(np.uint8)

    # --- GridSpec: Single row for images to prevent arbitrary scaling gaps ---
    from matplotlib.gridspec import GridSpec
    
    # Reduced figure height from 24 to 18 to better match the aspect ratio of 3 horizontal images
    fig = plt.figure(figsize=(54, 18))
    gs = GridSpec(
        1, 3,
        figure=fig,
        wspace=0.02,
        left=0.01, right=0.99,
        top=0.90, bottom=0.10, # Bbox tight will crop appropriately anyway
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(image)
    ax0.set_title("Input Image", fontsize=72, fontweight="bold", pad=18)
    ax0.axis("off")

    ax1.imshow(pred_rgb)
    ax1.set_title("Predicted Segmentation", fontsize=72, fontweight="bold", pad=18)
    ax1.axis("off")

    ax2.imshow(blended)
    ax2.set_title("Overlay", fontsize=72, fontweight="bold", pad=18)
    ax2.axis("off")

    # --- Legend tied strictly to the bottom of the middle image ---
    unique_classes = np.unique(prediction)
    patches = []
    for class_id in sorted(unique_classes):
        if class_id == 0:  # Skip background
            continue
        color = palette[class_id] / 255.0
        label = id2label.get(class_id, f"class_{class_id}")
        pct = 100.0 * np.sum(prediction == class_id) / prediction.size
        patches.append(mpatches.Patch(color=color, label=f"{label} ({pct:.1f}%)"))

    if patches:
        # Limit to 5 columns max so the legend doesn't overstretch horizontally
        ncol = min(5, len(patches))
        
        # We tie the legend to ax1 (the center image). 
        # loc="upper center" means the top middle of the legend aligns with the anchor.
        # bbox_to_anchor=(0.5, -0.02) means center horizontally, and exactly 2% below the bottom edge.
        ax1.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            title="Detected Materials",
            title_fontproperties={"weight": "bold", "size": 52},
            prop={"weight": "semibold", "size": 44},
            handleheight=2.5,   # reduced from 3.5 to tighten vertically
            handlelength=3.0,   # reduced from 4.0 to tighten horizontally
            borderpad=0.6,
            columnspacing=1.0,  # reduced from 2.0 to shrink total width
            handletextpad=0.5,  # reduced from 0.8 to tighten spacing
            framealpha=0.9,
            frameon=True,
        )

    # bbox_inches="tight" perfectly bounds the final PDF out to the edge of the legend
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(args):
    predictions_dir = args.predictions_dir
    output_dir = args.output_dir

    # Load global metadata
    global_meta_path = os.path.join(predictions_dir, "global_metadata.json")
    if not os.path.exists(global_meta_path):
        raise FileNotFoundError(
            f"global_metadata.json not found in {predictions_dir}. "
            "Run save_predictions.py first."
        )

    with open(global_meta_path, "r") as f:
        global_meta = json.load(f)

    id2label = {int(k): v for k, v in global_meta["id2label"].items()}
    palette = np.array(global_meta["palette"], dtype=np.uint8)

    # Discover prediction subdirectories (each contains prediction.npy, original.png, metadata.json)
    pred_dirs = []
    for root, dirs, files in os.walk(predictions_dir):
        if "prediction.npy" in files and "original.png" in files and "metadata.json" in files:
            pred_dirs.append(root)

    pred_dirs.sort()
    print(f"Found {len(pred_dirs)} predictions in {predictions_dir}")

    if not pred_dirs:
        print("No predictions found. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for pred_dir in pred_dirs:
        # Load per-image metadata
        with open(os.path.join(pred_dir, "metadata.json"), "r") as f:
            meta = json.load(f)

        prediction = np.load(os.path.join(pred_dir, "prediction.npy"))
        image = np.array(Image.open(os.path.join(pred_dir, "original.png")).convert("RGB"))

        image_name = meta.get("image_name", "")

        # Mirror the subdirectory structure relative to predictions_dir
        rel = os.path.relpath(pred_dir, predictions_dir)
        out_path = os.path.join(output_dir, rel + ".pdf")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        save_paper_figure(
            image=image,
            prediction=prediction,
            id2label=id2label,
            save_path=out_path,
            palette=palette,
            image_name=image_name,
        )
        print(f"Saved: {out_path}")

    print(f"\nDone! {len(pred_dirs)} figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper-quality plots from saved SegFormer predictions"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions/poder/paper",
        help="Directory containing saved predictions from save_predictions.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/poder/paper",
        help="Directory to save output PDF figures",
    )

    args = parser.parse_args()
    main(args)