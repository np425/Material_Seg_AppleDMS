"""
Utility functions for SegFormer training on Apple-DMS dataset.
"""

import torch
from torch import nn
import evaluate
import numpy as np
from typing import Dict, Any, Optional, Tuple


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Preprocess logits BEFORE they are accumulated during evaluation.
    
    This is CRITICAL for semantic segmentation to prevent RAM OOM:
    - Raw logits: [batch, 57, 512, 512] float32 = ~60 MB per image
    - After argmax: [batch, 512, 512] int64 = ~2 MB per image (~30× smaller)
    
    Args:
        logits: Model output logits [batch, num_classes, height, width]
        labels: Ground truth labels (used to get target size for upsampling)
    
    Returns:
        Predictions with argmax applied [batch, height, width]
    """
    # Upsample logits to match label size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    # Apply argmax to get predictions - huge memory reduction!
    return upsampled_logits.argmax(dim=1)


def create_compute_metrics(
    id2label: Dict[int, str],
    num_labels: int,
    ignore_index: int = 0,
    reduce_labels: bool = False,
):
    """
    Create a compute_metrics function for the HuggingFace Trainer.
    
    This factory function creates a closure with the proper references to
    id2label, num_labels, etc. so they don't need to be global variables.
    
    NOTE: This function expects that preprocess_logits_for_metrics is used
    to convert logits to predictions BEFORE accumulation.
    
    Args:
        id2label: Dictionary mapping label IDs to label names
        num_labels: Total number of labels
        ignore_index: Label index to ignore in metrics (default: 0 for "No label")
        reduce_labels: Whether labels were reduced during preprocessing
    
    Returns:
        compute_metrics function compatible with HF Trainer
    """
    # Load the metric once when creating the closure
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute mean IoU and per-class metrics for semantic segmentation.
        
        Args:
            eval_pred: EvalPrediction with predictions (already argmax'd) and label_ids
        
        Returns:
            Dictionary of metrics
        """
        # predictions are already post-argmax from preprocess_logits_for_metrics
        predictions, labels = eval_pred
        
        # Ensure predictions are numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Use _compute directly to bypass format validation
        # The public compute() method expects PIL Images, but _compute accepts numpy arrays
        metrics = metric._compute(
            predictions=predictions,
            references=labels,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=reduce_labels,
        )
        
        # Extract per-category metrics
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        # Add per-category metrics as individual key-value pairs
        for i, (acc, iou) in enumerate(zip(per_category_accuracy, per_category_iou)):
            if i < len(id2label):
                # Sanitize label name for metric key
                label_name = id2label[i].replace("/", "_").replace(" ", "_").replace(",", "")
                # Only add if not NaN
                if not np.isnan(acc):
                    metrics[f"accuracy_{label_name}"] = acc
                if not np.isnan(iou):
                    metrics[f"iou_{label_name}"] = iou
        
        return metrics
    
    return compute_metrics