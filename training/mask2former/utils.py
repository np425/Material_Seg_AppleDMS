"""
Utility functions for Mask2Former training on Apple-DMS dataset.

Mask2Former uses a query-based architecture where outputs are:
- class_queries_logits: (batch, num_queries, num_labels+1) - class for each query
- masks_queries_logits: (batch, num_queries, height, width) - mask for each query

This requires post-processing via image_processor.post_process_semantic_segmentation()
to convert query outputs to per-pixel semantic maps.

Evaluation uses confusion matrix accumulation to compute metrics at full resolution
without memory issues in distributed settings.
"""

import torch
import torch.distributed as dist
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
import time


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Compute confusion matrix for a batch of predictions and labels.
    
    Args:
        predictions: (N, H, W) tensor of predicted class IDs
        labels: (N, H, W) tensor of ground truth class IDs
        num_classes: Number of classes
        ignore_index: Class index to ignore (won't contribute to confusion matrix)
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    # Flatten predictions and labels
    pred_flat = predictions.view(-1)
    label_flat = labels.view(-1)
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = label_flat != ignore_index
    pred_flat = pred_flat[valid_mask]
    label_flat = label_flat[valid_mask]
    
    # Clamp to valid range to avoid indexing errors
    pred_flat = torch.clamp(pred_flat, 0, num_classes - 1)
    label_flat = torch.clamp(label_flat, 0, num_classes - 1)
    
    # Compute confusion matrix using bincount
    # Index: label * num_classes + pred
    indices = label_flat * num_classes + pred_flat
    conf_matrix = torch.bincount(
        indices.long(),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    
    return conf_matrix.float()


def compute_iou_from_confusion_matrix(
    conf_matrix: torch.Tensor,
    id2label: Dict[int, str],
    ignore_index: int = 0,
) -> Dict[str, float]:
    """
    Compute IoU metrics from a confusion matrix.
    
    Args:
        conf_matrix: (num_classes, num_classes) confusion matrix
        id2label: Dictionary mapping class IDs to names
        ignore_index: Class index that was ignored
    
    Returns:
        Dictionary of metrics including mean_iou, mean_accuracy, per-class IoU
    """
    num_classes = conf_matrix.shape[0]
    
    # Per-class metrics
    # IoU = TP / (TP + FP + FN)
    # TP = diagonal
    # FP = column sum - diagonal
    # FN = row sum - diagonal
    
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp  # Column sum minus diagonal
    fn = conf_matrix.sum(dim=1) - tp  # Row sum minus diagonal
    
    # Avoid division by zero
    denominator = tp + fp + fn
    iou_per_class = torch.where(
        denominator > 0,
        tp / denominator,
        torch.zeros_like(tp)
    )
    
    # Per-class accuracy
    class_totals = conf_matrix.sum(dim=1)  # Row sums = total GT pixels per class
    accuracy_per_class = torch.where(
        class_totals > 0,
        tp / class_totals,
        torch.zeros_like(tp)
    )
    
    # Create mask for valid classes (have GT pixels, not ignore index)
    valid_classes = (class_totals > 0)
    if ignore_index >= 0 and ignore_index < num_classes:
        valid_classes[ignore_index] = False
    
    # Mean IoU (only over classes that have GT pixels)
    valid_ious = iou_per_class[valid_classes]
    mean_iou = valid_ious.mean().item() if len(valid_ious) > 0 else 0.0
    
    # Mean accuracy
    valid_accs = accuracy_per_class[valid_classes]
    mean_accuracy = valid_accs.mean().item() if len(valid_accs) > 0 else 0.0
    
    # Overall accuracy (total correct / total pixels)
    total_correct = tp.sum().item()
    total_pixels = conf_matrix.sum().item()
    overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    # Build metrics dict
    metrics = {
        "mean_iou": mean_iou,
        "mean_accuracy": mean_accuracy,
        "overall_accuracy": overall_accuracy,
    }
    
    # Add per-class metrics
    iou_np = iou_per_class.cpu().numpy()
    acc_np = accuracy_per_class.cpu().numpy()
    
    for i in range(num_classes):
        if i in id2label and i != ignore_index:
            label_name = id2label[i].replace("/", "_").replace(" ", "_").replace(",", "")
            if not np.isnan(iou_np[i]) and class_totals[i] > 0:
                metrics[f"iou_{label_name}"] = float(iou_np[i])
            if not np.isnan(acc_np[i]) and class_totals[i] > 0:
                metrics[f"accuracy_{label_name}"] = float(acc_np[i])
    
    return metrics


@dataclass
class Mask2FormerDataCollator:
    """
    Custom data collator for Mask2Former training.
    
    Mask2Former expects:
    - pixel_values: (batch, channels, H, W) - can be stacked
    - mask_labels: List[Tensor] - each (num_masks_i, H, W), variable length - keep as list
    - class_labels: List[Tensor] - each (num_masks_i,), variable length - keep as list
    
    The standard data collator fails because mask_labels has variable first dimension
    (different number of masks per image).
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of features into a batch dict."""
        batch = {}
        
        # Stack pixel_values - these are all the same size
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
        # Keep mask_labels as a list (variable first dimension)
        if "mask_labels" in features[0]:
            batch["mask_labels"] = [f["mask_labels"] for f in features]
        
        # Keep class_labels as a list (variable first dimension) 
        if "class_labels" in features[0]:
            batch["class_labels"] = [f["class_labels"] for f in features]
        
        # Handle SegFormer-style labels if present (fallback for compatibility)
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        
        return batch


class Mask2FormerTrainer(Trainer):
    """
    Custom Trainer for Mask2Former with confusion matrix based evaluation.
    
    This trainer:
    1. Post-processes Mask2Former query outputs to semantic maps
    2. Accumulates confusion matrices at full resolution during evaluation
    3. Aggregates confusion matrices across DDP processes
    4. Computes IoU metrics from the final confusion matrix
    
    This approach avoids memory issues from gathering full-resolution predictions
    while maintaining metric accuracy.
    """
    
    def __init__(
        self,
        image_processor=None,
        num_classes: int = None,
        id2label: Dict[int, str] = None,
        ignore_index: int = 0,
        label_smoothing: float = 0.0,
        backbone_lr: float = None,
        min_lr: float = 0.0,
        **kwargs
    ):
        """
        Initialize the Mask2FormerTrainer.
        
        Args:
            image_processor: Mask2FormerImageProcessor for post-processing outputs
            num_classes: Number of classes for confusion matrix
            id2label: Dictionary mapping label IDs to label names
            ignore_index: Class index to ignore in metrics (default: 0)
            label_smoothing: Label smoothing factor for classification loss (default: 0.0)
            backbone_lr: Learning rate for backbone/encoder (default: None, uses main LR)
            min_lr: Minimum learning rate for cosine scheduler (default: 0.0)
            **kwargs: Arguments passed to the base Trainer
        """
        super().__init__(**kwargs)
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.id2label = id2label or {}
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.backbone_lr = backbone_lr
        self.min_lr = min_lr
    
    def create_optimizer(self):
        """
        Create optimizer with differential learning rates for backbone vs rest.
        
        The Swin backbone (pixel_level_module.encoder) gets a lower learning rate,
        while the decoder and other components get the standard learning rate.
        """
        if self.optimizer is not None:
            return self.optimizer
        
        model = self.model
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module
        
        # Get training arguments
        lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        backbone_lr = self.backbone_lr if self.backbone_lr is not None else lr
        
        # Separate parameters into backbone and non-backbone groups
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Mask2Former backbone is at model.pixel_level_module.encoder
            if 'pixel_level_module.encoder' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        optimizer_grouped_parameters = [
            {
                'params': backbone_params,
                'lr': backbone_lr,
                'weight_decay': weight_decay,
            },
            {
                'params': other_params,
                'lr': lr,
                'weight_decay': weight_decay,
            },
        ]
        
        # Log the parameter counts
        if self.args.local_rank <= 0:  # Only log on main process
            print(f"Differential LR: backbone ({len(backbone_params)} params) @ {backbone_lr}, "
                  f"other ({len(other_params)} params) @ {lr}")
        
        # Use AdamW optimizer
        from torch.optim import AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create cosine scheduler with min_lr support.
        
        The standard HuggingFace cosine scheduler decays to 0, but we want to
        decay to min_lr instead for better training stability.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        
        if optimizer is None:
            optimizer = self.optimizer
        
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        lr = self.args.learning_rate
        min_lr = self.min_lr
        backbone_lr = self.backbone_lr if self.backbone_lr is not None else lr
        
        # Compute min_lr ratios for each param group
        # Group 0: backbone, Group 1: other
        min_lr_ratios = [
            min_lr / backbone_lr if backbone_lr > 0 else 0,
            min_lr / lr if lr > 0 else 0,
        ]
        
        def lr_lambda(current_step: int, group_idx: int = 0):
            min_ratio = min_lr_ratios[group_idx] if group_idx < len(min_lr_ratios) else 0
            
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                # Cosine decay from 1.0 to min_ratio
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Scale cosine decay to go from 1.0 to min_ratio
                return min_ratio + (1.0 - min_ratio) * cosine_decay
        
        # Create separate lambda for each param group
        lr_lambdas = [lambda step, idx=i: lr_lambda(step, idx) for i in range(len(optimizer.param_groups))]
        
        self.lr_scheduler = LambdaLR(optimizer, lr_lambdas)
        
        if self.args.local_rank <= 0:
            print(f"Cosine scheduler with min_lr={min_lr}, warmup_steps={num_warmup_steps}")
        
        return self.lr_scheduler
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for Mask2Former with optional label smoothing.
        
        Mask2Former computes its own loss internally. When label_smoothing > 0,
        we recompute the classification loss component with label smoothing applied.
        """
        outputs = model(**inputs)
        
        if self.label_smoothing > 0.0 and "class_labels" in inputs:
            # Mask2Former's total loss includes:
            # - loss_ce: cross-entropy for class predictions
            # - loss_mask: binary cross-entropy for mask predictions  
            # - loss_dice: dice loss for mask predictions
            # 
            # We need to recompute loss_ce with label smoothing
            # The model outputs include class_queries_logits: (batch, num_queries, num_classes+1)
            
            class_queries_logits = outputs.class_queries_logits  # (B, num_queries, num_classes+1)
            
            # Apply label smoothing as a regularization term
            # We need to get the matching indices from the Hungarian matcher
            # This is complex because Mask2Former uses bipartite matching internally
            # 
            # Simpler approach: Apply label smoothing as a regularization term
            # by adding a uniform distribution penalty to the class predictions
            
            device = class_queries_logits.device
            num_classes = class_queries_logits.shape[-1]
            
            # Log-softmax of predictions
            log_probs = torch.nn.functional.log_softmax(class_queries_logits, dim=-1)
            
            # Uniform distribution target for smoothing regularization
            uniform_target = torch.ones_like(log_probs) / num_classes
            
            # KL divergence from uniform as smoothing regularization
            # This encourages less confident predictions (softer distributions)
            smoothing_loss = torch.nn.functional.kl_div(
                log_probs,
                uniform_target,
                reduction='batchmean'
            )
            
            # Add smoothing loss to the total loss
            # Scale by label_smoothing factor
            loss = outputs.loss + self.label_smoothing * smoothing_loss
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _reconstruct_semantic_labels(
        self,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Reconstruct semantic segmentation labels from instance mask format.
        
        Args:
            mask_labels: List of (num_masks, H, W) binary mask tensors
            class_labels: List of (num_masks,) class ID tensors
            device: Target device
        
        Returns:
            List of (H, W) semantic label tensors
        """
        semantic_labels = []
        for masks, classes in zip(mask_labels, class_labels):
            h, w = masks.shape[1], masks.shape[2]
            semantic_map = torch.zeros((h, w), dtype=torch.long, device=device)
            for mask, class_id in zip(masks, classes):
                semantic_map[mask.bool()] = class_id.item()
            semantic_labels.append(semantic_map)
        return semantic_labels
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Custom evaluation loop using confusion matrix accumulation.
        
        This computes metrics at full resolution by accumulating a confusion matrix
        during evaluation, then reducing across DDP processes.
        """
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)
        
        batch_size = self.args.per_device_eval_batch_size
        
        # Initialize confusion matrix on the model's device
        device = next(model.parameters()).device
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.float64,
            device=device
        )
        
        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        
        model.eval()
        
        start_time = time.time()
        
        for step, inputs in enumerate(dataloader):
            # Get labels before preparing inputs
            labels = inputs.get("labels", None)
            mask_labels = inputs.get("mask_labels", None)
            class_labels = inputs.get("class_labels", None)
            
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss.mean().detach()
                total_loss += loss.item()
                num_batches += 1
                
                # Post-process to get semantic segmentation maps
                if labels is not None:
                    batch_size_actual = labels.shape[0]
                    target_sizes = [(labels.shape[1], labels.shape[2])] * batch_size_actual
                    gt_labels = [lbl for lbl in labels]
                elif mask_labels is not None:
                    target_sizes = [(m.shape[1], m.shape[2]) for m in mask_labels]
                    batch_size_actual = len(mask_labels)
                    gt_labels = self._reconstruct_semantic_labels(
                        mask_labels, class_labels, device
                    )
                else:
                    continue
                
                # Post-process predictions
                segmentation_maps = self.image_processor.post_process_semantic_segmentation(
                    outputs,
                    target_sizes=target_sizes,
                )
                
                # Update confusion matrix for each sample
                for pred, gt in zip(segmentation_maps, gt_labels):
                    pred = pred.to(device)
                    gt = gt.to(device)
                    
                    # Add to confusion matrix
                    batch_conf = compute_confusion_matrix(
                        pred.unsqueeze(0),
                        gt.unsqueeze(0),
                        self.num_classes,
                        self.ignore_index,
                    )
                    confusion_matrix += batch_conf.to(torch.float64)
                    num_samples += 1
        
        # Aggregate confusion matrix across DDP processes
        if dist.is_initialized():
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
            # Also aggregate loss and sample count
            loss_tensor = torch.tensor([total_loss, num_batches], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
        
        # Compute metrics from confusion matrix
        metrics = compute_iou_from_confusion_matrix(
            confusion_matrix,
            self.id2label,
            self.ignore_index,
        )
        
        # Add loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics["loss"] = avg_loss
        
        # Add speed metrics
        runtime = time.time() - start_time
        metrics.update(speed_metrics(
            metric_key_prefix,
            start_time,
            num_samples=num_samples,
            num_steps=num_batches,
        ))
        
        # Prefix all metrics
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        
        # Log metrics
        self.log(metrics)
        
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )