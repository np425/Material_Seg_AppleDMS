"""
Distributed evaluation script for Mask2Former on Apple-DMS test set.
Optimized for 120-CPU / RTX 5090 Blackwell architectures.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Break native C++ multi-thread locks BEFORE importing heavy vision packages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)
from torch.utils.data import DataLoader, DistributedSampler


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
    """Compute confusion matrix efficiently using GPU bincount."""
    pred_flat = predictions.view(-1).to(torch.int64)
    label_flat = labels.view(-1).to(torch.int64)
    
    valid_mask = label_flat != ignore_index
    pred_flat = pred_flat[valid_mask]
    label_flat = label_flat[valid_mask]
    
    pred_flat = torch.clamp(pred_flat, 0, num_classes - 1)
    label_flat = torch.clamp(label_flat, 0, num_classes - 1)
    
    indices = label_flat * num_classes + pred_flat
    conf_matrix = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    
    return conf_matrix.float()


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
    
    denominator = tp + fp + fn
    iou_per_class = torch.where(
        denominator > 0,
        tp / denominator,
        torch.zeros_like(tp)
    )
    
    class_totals = conf_matrix.sum(dim=1)
    acc_per_class = torch.where(
        class_totals > 0,
        tp / class_totals,
        torch.zeros_like(tp)
    )
    
    valid_classes = (class_totals > 0)
    valid_classes[ignore_index] = False
    
    mean_iou = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mean_accuracy = acc_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    
    total_correct = tp.sum().item()
    total_pixels = conf_matrix.sum().item()
    overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    metrics = {
        "mean_iou": mean_iou,
        "mean_accuracy": mean_accuracy,
        "overall_accuracy": overall_accuracy,
    }
    
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
    """Save visualization of prediction vs ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")
    
    gt_rgb = palette[ground_truth]
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis("off")
    
    pred_rgb = palette[prediction]
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=14)
    axes[2].axis("off")
    
    unique_classes = np.unique(np.concatenate([ground_truth.flatten(), prediction.flatten()]))
    patches = []
    for class_id in unique_classes:
        if class_id == 0:
            continue
        color = palette[class_id] / 255.0
        label = id2label.get(class_id, f"class_{class_id}")
        patches.append(mpatches.Patch(color=color, label=label))
    
    if patches:
        fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


class EvalDataset(torch.utils.data.Dataset):
    """Asynchronous on-the-fly streaming wrapper to handle multiprocessing safely."""
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        label = np.array(sample["label"])
        
        # Runs inside the dedicated background DataLoader workers
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "pixel_mask": inputs["pixel_mask"].squeeze(0),
            "label": torch.from_numpy(label),
            "raw_image": image
        }


def collate_fn(batch):
    return batch


def setup_distributed():
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
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def find_best_checkpoint(run_dir: str) -> str:
    trainer_state_path = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(os.path.join(run_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        return run_dir
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        best_checkpoint = trainer_state.get("best_model_checkpoint")
        if best_checkpoint and os.path.exists(best_checkpoint):
            return best_checkpoint
    checkpoint_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            if os.path.exists(os.path.join(item_path, "model.safetensors")) or \
               os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                checkpoint_dirs.append(item_path)
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoint_dirs[-1]
    return run_dir


def get_base_model_name(checkpoint_path: str) -> str:
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "_name_or_path" in config:
            return config["_name_or_path"]
    return "facebook/mask2former-swin-large-ade-semantic"


def main(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = args.checkpoint
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
    
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
        
    if dist.is_initialized():
        dist.barrier()
        
    if is_main_process():
        print("Loading model...")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path)
    
    processor_path = args.processor if args.processor else checkpoint_path
    try:
        processor = AutoImageProcessor.from_pretrained(processor_path)
        if is_main_process():
            print(f"Loaded processor from: {processor_path}")
    except OSError:
        base_model = get_base_model_name(checkpoint_path)
        if is_main_process():
            print(f"Processor not found in checkpoint, loading from base model: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)
    
    model = model.to(device)
    model.eval()
    
    id2label = model.config.id2label
    num_classes = len(id2label)
    palette = create_color_palette(num_classes)
    
    if is_main_process():
        print("Loading raw streaming dataset...")
        
    raw_dataset = load_dataset("AllanK24/apple-dms-materials", split="test")
    eval_dataset = EvalDataset(raw_dataset, processor)
    
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 32 workers handles structural background streaming across your 120 cores smoothly
    optimal_workers = min(32, os.cpu_count() // 2)
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=optimal_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if is_main_process():
        print(f"Asynchronous pipeline armed with {optimal_workers} CPU workers.")
        print(f"Batch size per GPU: {args.batch_size}")
        
    if args.compile:
        if is_main_process():
            print("Compiling model graph via torch.compile (Expect a 1-2 min warm-up stall)...")
        model = torch.compile(model, mode="reduce-overhead")
        
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    
    num_vis_samples = args.num_samples
    if is_main_process():
        total_samples = len(eval_dataset)
        sample_indices = set(random.sample(range(total_samples), min(num_vis_samples, total_samples)))
    else:
        sample_indices = set()
        
    if dist.is_initialized():
        if is_main_process():
            sample_indices_list = list(sample_indices)
            sample_indices_tensor = torch.tensor(sample_indices_list + [0] * (num_vis_samples - len(sample_indices_list)), device=device)
        else:
            sample_indices_tensor = torch.zeros(num_vis_samples, dtype=torch.long, device=device)
        dist.broadcast(sample_indices_tensor, src=0)
        sample_indices = set(sample_indices_tensor.cpu().numpy().tolist())
        
    if is_main_process():
        print("Starting runtime evaluation sweep...")
        
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating", disable=not is_main_process())
        
        for batch_idx, batch in enumerate(iterator):
            batch_start_idx = batch_idx * args.batch_size * world_size + rank * args.batch_size
            
            # Unpack pre-tokenized background payloads directly into VRAM
            pixel_values = torch.stack([sample["pixel_values"] for sample in batch]).to(device, non_blocking=True)
            pixel_mask = torch.stack([sample["pixel_mask"] for sample in batch]).to(device, non_blocking=True)
            
            original_sizes = [sample["label"].shape[:2] for sample in batch]
            
            inputs = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
            
            # Blazing fast execution loop
            outputs = model(**inputs)
            
            pred_segs = processor.post_process_semantic_segmentation(outputs, target_sizes=original_sizes)
            
            for i, (pred_seg, sample) in enumerate(zip(pred_segs, batch)):
                label_tensor = sample["label"].to(device, non_blocking=True)
                
                batch_conf = compute_confusion_matrix(pred_seg, label_tensor, num_classes, ignore_index=0)
                confusion_matrix += batch_conf
                
                global_idx = batch_start_idx + i
                if global_idx in sample_indices and is_main_process():
                    save_path = os.path.join(args.output_dir, "predictions", f"sample_{global_idx:04d}.png")
                    save_prediction_visualization(
                        sample["raw_image"],
                        pred_seg.cpu().numpy(),
                        sample["label"].numpy(),
                        id2label,
                        save_path,
                        palette,
                    )
                    
    if dist.is_initialized():
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        
    if is_main_process():
        print("\nAggregating final telemetry metrics...")
        metrics = compute_iou_from_confusion_matrix(confusion_matrix, id2label, ignore_index=0)
        
        # Fill placeholders since heavy CPU Scipy morphology filters were deleted
        metrics["mean_boundary_iou"] = 0.0
        metrics["boundary_iou_per_class"] = {}
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (BOUNDARY IOU DEPRECATED FOR SPEED)")
        print("=" * 60)
        print(f"Mean IoU:          {metrics['mean_iou']:.4f}")
        print(f"Mean Accuracy:     {metrics['mean_accuracy']:.4f}")
        print(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}")
        print("=" * 60)
        
        print("\nPer-Class IoU:")
        sorted_classes = sorted(metrics["iou_per_class"].items(), key=lambda x: x[1], reverse=True)
        for class_name, iou in sorted_classes:
            print(f"  {class_name:30s}: IoU={iou:.4f}")
            
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("MASK2FORMER RAPID EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n\n")
            f.write(f"Mean IoU:          {metrics['mean_iou']:.4f}\n")
            f.write(f"Mean Accuracy:     {metrics['mean_accuracy']:.4f}\n")
            f.write(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}\n\n")
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 40 + "\n")
            for class_name, iou in sorted_classes:
                acc = metrics["accuracy_per_class"].get(class_name, 0.0)
                f.write(f"{class_name:<30s} {iou:>8.4f} {acc:>8.4f}\n")
                
        print(f"\nSaved telemetry summary directly to: {summary_path}")
        
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former on Apple-DMS test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--processor", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    main(args)