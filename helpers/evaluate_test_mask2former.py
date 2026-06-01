"""
Distributed evaluation script for Mask2Former on Apple-DMS test set.

Computes:
- Mean IoU and per-class IoU
- Mean accuracy and per-class accuracy
- Boundary IoU (using morphological operations)

Saves:
- Metrics to JSON file
- 5 random sample predictions with visualizations

Usage:
    torchrun --nproc_per_node=8 helpers/evaluate_test_mask2former.py \
        --checkpoint /path/to/checkpoint \
        --output_dir /path/to/output
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)
from torch.utils.data import DataLoader, DistributedSampler

# Color palette for visualization (57 classes)
def create_color_palette(num_classes: int) -> np.ndarray:
    """Create a distinct color palette for visualization."""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background/no-label is black
    return palette


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Compute confusion matrix for a batch of predictions.
    
    Args:
        predictions: (H, W) or (B, H, W) predicted class indices
        labels: (H, W) or (B, H, W) ground truth class indices
        num_classes: Total number of classes
        ignore_index: Class index to ignore (not counted in matrix)
    
    Returns:
        (num_classes, num_classes) confusion matrix
    """
    # CRITICAL: Cast to int64 to avoid uint8 overflow when computing indices
    # Labels from datasets are often uint8, and label * num_classes overflows
    pred_flat = predictions.view(-1).to(torch.int64)
    label_flat = labels.view(-1).to(torch.int64)
    
    # Mask for valid pixels (not ignore_index)
    valid_mask = label_flat != ignore_index
    pred_flat = pred_flat[valid_mask]
    label_flat = label_flat[valid_mask]
    
    # Clamp to valid range
    pred_flat = torch.clamp(pred_flat, 0, num_classes - 1)
    label_flat = torch.clamp(label_flat, 0, num_classes - 1)
    
    # Compute confusion matrix using bincount
    indices = label_flat * num_classes + pred_flat
    conf_matrix = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    
    return conf_matrix.float()


def compute_boundary_mask(segmentation: np.ndarray, dilation_radius: int = 2) -> np.ndarray:
    """
    Compute boundary mask from a segmentation map.
    
    Args:
        segmentation: (H, W) segmentation map
        dilation_radius: Radius of the boundary region
    
    Returns:
        (H, W) binary boundary mask
    """
    # Create structure element for dilation
    struct = ndimage.generate_binary_structure(2, 1)
    
    # Find boundaries by comparing with dilated/eroded versions
    boundaries = np.zeros_like(segmentation, dtype=bool)
    
    for class_id in np.unique(segmentation):
        if class_id == 0:  # Skip background
            continue
        class_mask = segmentation == class_id
        
        # Dilate and erode
        dilated = ndimage.binary_dilation(class_mask, struct, iterations=dilation_radius)
        eroded = ndimage.binary_erosion(class_mask, struct, iterations=dilation_radius)
        
        # Boundary is the difference
        boundary = dilated & ~eroded
        boundaries |= boundary
    
    return boundaries


def compute_boundary_iou(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
    dilation_radius: int = 2,
) -> Dict[str, float]:
    """
    Compute Boundary IoU for predictions.
    
    Boundary IoU measures how well the model predicts class boundaries,
    which is important for segmentation quality.
    
    Args:
        predictions: (H, W) predicted segmentation
        labels: (H, W) ground truth segmentation
        num_classes: Number of classes
        ignore_index: Class to ignore
        dilation_radius: Boundary region width
    
    Returns:
        Dictionary with boundary IoU metrics
    """
    # Compute boundary masks
    pred_boundary = compute_boundary_mask(predictions, dilation_radius)
    label_boundary = compute_boundary_mask(labels, dilation_radius)
    
    # Create boundary region from labels (trimap)
    boundary_region = label_boundary
    
    # Mask predictions and labels to boundary region
    pred_boundary_classes = predictions.copy()
    pred_boundary_classes[~boundary_region] = ignore_index
    
    label_boundary_classes = labels.copy()
    label_boundary_classes[~boundary_region] = ignore_index
    
    # Compute per-class boundary IoU
    boundary_iou_per_class = {}
    
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        
        pred_class = (pred_boundary_classes == class_id)
        label_class = (label_boundary_classes == class_id)
        
        intersection = np.sum(pred_class & label_class)
        union = np.sum(pred_class | label_class)
        
        if union > 0:
            boundary_iou_per_class[class_id] = float(intersection / union)
    
    # Compute mean boundary IoU
    if boundary_iou_per_class:
        mean_boundary_iou = np.mean(list(boundary_iou_per_class.values()))
    else:
        mean_boundary_iou = 0.0
    
    return {
        "mean_boundary_iou": mean_boundary_iou,
        "boundary_iou_per_class": boundary_iou_per_class,
    }


def compute_iou_from_confusion_matrix(
    conf_matrix: torch.Tensor,
    id2label: Dict[int, str],
    ignore_index: int = 0,
) -> Dict[str, float]:
    """Compute IoU metrics from confusion matrix."""
    num_classes = conf_matrix.shape[0]
    
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    # IoU per class
    denominator = tp + fp + fn
    iou_per_class = torch.where(
        denominator > 0,
        tp / denominator,
        torch.zeros_like(tp)
    )
    
    # Accuracy per class
    class_totals = conf_matrix.sum(dim=1)
    acc_per_class = torch.where(
        class_totals > 0,
        tp / class_totals,
        torch.zeros_like(tp)
    )
    
    # Valid classes (have samples and not ignore_index)
    valid_classes = (class_totals > 0)
    valid_classes[ignore_index] = False
    
    # Mean metrics (excluding ignore_index and empty classes)
    mean_iou = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mean_accuracy = acc_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    
    # Overall accuracy
    total_correct = tp.sum().item()
    total_pixels = conf_matrix.sum().item()
    overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    metrics = {
        "mean_iou": mean_iou,
        "mean_accuracy": mean_accuracy,
        "overall_accuracy": overall_accuracy,
    }
    
    # Per-class metrics
    iou_per_class_dict = {}
    acc_per_class_dict = {}
    
    for i in range(num_classes):
        if i == ignore_index:
            continue
        label_name = id2label.get(i, f"class_{i}")
        iou_per_class_dict[label_name] = iou_per_class[i].item()
        acc_per_class_dict[label_name] = acc_per_class[i].item()
    
    metrics["iou_per_class"] = iou_per_class_dict
    metrics["accuracy_per_class"] = acc_per_class_dict
    
    return metrics


def save_prediction_visualization(
    image: Image.Image,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    id2label: Dict[int, str],
    save_path: str,
    palette: np.ndarray,
):
    """Save a visualization of the prediction vs ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")
    
    # Ground truth
    gt_rgb = palette[ground_truth]
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis("off")
    
    # Prediction
    pred_rgb = palette[prediction]
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=14)
    axes[2].axis("off")
    
    # Create legend for classes present in the image
    unique_classes = np.unique(np.concatenate([ground_truth.flatten(), prediction.flatten()]))
    patches = []
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        color = palette[class_id] / 255.0
        label = id2label.get(class_id, f"class_{class_id}")
        patches.append(mpatches.Patch(color=color, label=label))
    
    if patches:
        fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def collate_fn(batch):
    """Custom collate function for evaluation."""
    return batch


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def find_best_checkpoint(run_dir: str) -> str:
    """
    Find the best checkpoint from a training run directory.
    
    Looks at trainer_state.json to find the checkpoint with best metric.
    """
    trainer_state_path = os.path.join(run_dir, "trainer_state.json")
    
    # Check if this is already a checkpoint directory
    if os.path.exists(os.path.join(run_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        return run_dir
    
    # Look for trainer_state.json
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
            # Check if it has model files
            if os.path.exists(os.path.join(item_path, "model.safetensors")) or \
               os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                checkpoint_dirs.append(item_path)
    
    if checkpoint_dirs:
        # Return the checkpoint with highest step number
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoint_dirs[-1]
    
    # Fallback - return original path
    return run_dir


def get_base_model_name(checkpoint_path: str) -> str:
    """
    Get the base model name from a checkpoint's config.
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Check for common fields that indicate base model
        if "_name_or_path" in config:
            return config["_name_or_path"]
    
    # Default fallback
    return "facebook/mask2former-swin-large-ade-semantic"


def main(args):
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = args.checkpoint
    
    # Resolve checkpoint path if it's a local directory
    if os.path.isdir(checkpoint_path):
        resolved_checkpoint = find_best_checkpoint(checkpoint_path)
        if is_main_process():
            if resolved_checkpoint != checkpoint_path:
                print(f"Resolved best checkpoint: {resolved_checkpoint}")
        checkpoint_path = resolved_checkpoint
    
    if is_main_process():
        print(f"Running distributed evaluation with {world_size} GPUs")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
    
    # Synchronize before loading
    if dist.is_initialized():
        dist.barrier()
    
    # Load model
    if is_main_process():
        print("Loading model...")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path)
    
    # Load processor - try checkpoint first, then fall back to base model
    processor = None
    processor_path = args.processor if args.processor else checkpoint_path
    
    try:
        processor = AutoImageProcessor.from_pretrained(processor_path)
        if is_main_process():
            print(f"Loaded processor from: {processor_path}")
    except OSError:
        # Processor not in checkpoint, try base model
        base_model = get_base_model_name(checkpoint_path)
        if is_main_process():
            print(f"Processor not found in checkpoint, loading from base model: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)
    
    model = model.to(device)
    model.eval()
    
    # Get label mappings
    id2label = model.config.id2label
    num_classes = len(id2label)
    
    if is_main_process():
        print(f"Number of classes: {num_classes}")
    
    # Create color palette
    palette = create_color_palette(num_classes)
    
    # Load dataset
    if is_main_process():
        print("Loading dataset...")
    
    # Load dataset
    if is_main_process():
        print("Loading dataset...")
    
    raw_dataset = load_dataset("AllanK24/apple-dms-materials", split="test")

    # Define a rapid mapping function to tokenize features on your CPU workers
    def preprocess_function(examples):
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in examples["image"]]
        # This handles the normalization and resizing ahead of time
        model_inputs = processor(images=images, return_tensors="np")
        # Add labels manually so the collator can find them
        model_inputs["labels"] = [np.array(lbl) for lbl in examples["label"]]
        return model_inputs
    
    available_cpus = max(1, int(os.cpu_count() * 0.75)) 
    print(f"Sprinting pre-tokenization across {available_cpus} CPU threads...")

    # Use .map with multiple writers to cache the tensors
    dataset = raw_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=args.batch_size,
        num_proc=available_cpus # Automatically scales to ~90 cores on your system!
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    batch_size = args.batch_size
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )
    
    if is_main_process():
        print(f"Batch size per GPU: {batch_size}, Total: {batch_size * world_size}")
    
    # Try to compile model for faster inference (optional)
    if args.compile:
        if is_main_process():
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=device,
    )
    
    # Boundary IoU accumulators
    boundary_iou_sum = {i: 0.0 for i in range(num_classes)}
    boundary_iou_count = {i: 0 for i in range(num_classes)}
    
    # Select random samples for visualization (only on main process)
    num_vis_samples = args.num_samples
    if is_main_process():
        total_samples = len(dataset)
        sample_indices = set(random.sample(range(total_samples), min(num_vis_samples, total_samples)))
    else:
        sample_indices = set()
    
    # Broadcast sample indices to all processes
    if dist.is_initialized():
        if is_main_process():
            sample_indices_list = list(sample_indices)
            sample_indices_tensor = torch.tensor(sample_indices_list + [0] * (num_vis_samples - len(sample_indices_list)), device=device)
        else:
            sample_indices_tensor = torch.zeros(num_vis_samples, dtype=torch.long, device=device)
        dist.broadcast(sample_indices_tensor, src=0)
        sample_indices = set(sample_indices_tensor.cpu().numpy().tolist())
    
    # Evaluation loop
    if is_main_process():
        print("Starting evaluation...")
    
    samples_processed = 0
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating", disable=not is_main_process())
        
        for batch_idx, batch in enumerate(iterator):
            # Calculate global indices for this batch
            batch_start_idx = batch_idx * batch_size * world_size + rank * batch_size
            
            # The batch already contains tokenized arrays! Push them straight to VRAM
            pixel_values = torch.stack([torch.tensor(sample["pixel_values"]) for sample in batch]).to(device)
            pixel_mask = torch.stack([torch.tensor(sample["pixel_mask"]) for sample in batch]).to(device)
            
            labels = [np.array(sample["labels"]) for sample in batch]
            original_sizes = [lbl.shape[:2] for lbl in labels]
            
            # Pack inputs directly for the RTX 5090
            inputs = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
            
            # Batch forward pass (Runs at full speed on Blackwell Tensor Cores!)
            outputs = model(**inputs)
            
            # Post-process predictions
            pred_segs = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=original_sizes
            )
            
            # Process each sample in batch for metric updates
            for i, (pred_seg, label) in enumerate(zip(pred_segs, labels)):
                pred_tensor = pred_seg.to(device)
                label_tensor = torch.from_numpy(label).to(device)
                
                batch_conf = compute_confusion_matrix(
                    pred_tensor,
                    label_tensor,
                    num_classes,
                    ignore_index=0,
                )
                confusion_matrix += batch_conf
                
                # Compute boundary IoU for this sample
                # boundary_metrics = compute_boundary_iou(
                #     pred_seg_np,
                #     label,
                #     num_classes,
                #     ignore_index=0,
                #     dilation_radius=2,
                # )
                
                # for class_id, biou in boundary_metrics["boundary_iou_per_class"].items():
                #     boundary_iou_sum[class_id] += biou
                #     boundary_iou_count[class_id] += 1
                
                # Save sample if selected
                global_idx = batch_start_idx + i
                if global_idx in sample_indices and is_main_process():
                    save_path = os.path.join(
                        args.output_dir,
                        "predictions",
                        f"sample_{global_idx:04d}.png"
                    )
                    save_prediction_visualization(
                        image,
                        pred_seg_np,
                        label,
                        id2label,
                        save_path,
                        palette,
                    )
            
            samples_processed += len(batch)
    
    # Synchronize and aggregate confusion matrices
    if dist.is_initialized():
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        
        # Aggregate boundary IoU
        for class_id in range(num_classes):
            sum_tensor = torch.tensor([boundary_iou_sum[class_id]], device=device)
            count_tensor = torch.tensor([boundary_iou_count[class_id]], device=device)
            
            dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            
            boundary_iou_sum[class_id] = sum_tensor.item()
            boundary_iou_count[class_id] = count_tensor.item()
    
    # Compute final metrics on main process
    if is_main_process():
        print("\nComputing final metrics...")
        
        # Standard IoU metrics
        metrics = compute_iou_from_confusion_matrix(
            confusion_matrix,
            id2label,
            ignore_index=0,
        )
        
        # Boundary IoU metrics
        boundary_iou_per_class = {}
        for class_id in range(num_classes):
            if class_id == 0:
                continue
            if boundary_iou_count[class_id] > 0:
                biou = boundary_iou_sum[class_id] / boundary_iou_count[class_id]
                label_name = id2label.get(class_id, f"class_{class_id}")
                boundary_iou_per_class[label_name] = biou
        
        if boundary_iou_per_class:
            mean_boundary_iou = np.mean(list(boundary_iou_per_class.values()))
        else:
            mean_boundary_iou = 0.0
        
        metrics["mean_boundary_iou"] = mean_boundary_iou
        metrics["boundary_iou_per_class"] = boundary_iou_per_class
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Mean IoU:          {metrics['mean_iou']:.4f}")
        print(f"Mean Accuracy:     {metrics['mean_accuracy']:.4f}")
        print(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}")
        print(f"Mean Boundary IoU: {metrics['mean_boundary_iou']:.4f}")
        print("=" * 60)
        
        # Print per-class IoU (sorted by IoU)
        print("\nPer-Class IoU:")
        sorted_classes = sorted(
            metrics["iou_per_class"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, iou in sorted_classes:
            biou = boundary_iou_per_class.get(class_name, 0.0)
            print(f"  {class_name:30s}: IoU={iou:.4f}, BoundaryIoU={biou:.4f}")
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Save readable summary
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("MASK2FORMER EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: AllanK24/apple-dms-materials (test split)\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("AGGREGATE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean IoU:          {metrics['mean_iou']:.4f}\n")
            f.write(f"Mean Accuracy:     {metrics['mean_accuracy']:.4f}\n")
            f.write(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Mean Boundary IoU: {metrics['mean_boundary_iou']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Class':<30s} {'IoU':>8s} {'Acc':>8s} {'BIoU':>8s}\n")
            f.write("-" * 54 + "\n")
            
            for class_name, iou in sorted_classes:
                acc = metrics["accuracy_per_class"].get(class_name, 0.0)
                biou = boundary_iou_per_class.get(class_name, 0.0)
                f.write(f"{class_name:<30s} {iou:>8.4f} {acc:>8.4f} {biou:>8.4f}\n")
        
        print(f"Summary saved to: {summary_path}")
        print(f"Predictions saved to: {os.path.join(args.output_dir, 'predictions')}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former on Apple-DMS test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint, run directory, or HuggingFace Hub model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Path to image processor (defaults to checkpoint path, then base model)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per GPU for inference (default: 256)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference",
    )
    
    args = parser.parse_args()
    main(args)
