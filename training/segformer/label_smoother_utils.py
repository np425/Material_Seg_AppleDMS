"""
Custom label smoothing utilities for semantic segmentation models.

The default HuggingFace LabelSmoother doesn't work with segmentation models because
it expects labels to match logit dimensions. SegFormer outputs logits at 1/4 resolution
(e.g., 160x160 for 640x640 input), but labels remain at full resolution.

This module provides a SegmentationTrainer that applies label smoothing correctly
by upsampling logits before computing the smoothed cross-entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Union, Any, Optional


class SegmentationLabelSmoother:
    """
    Label smoother for semantic segmentation that handles spatial resolution mismatch.
    
    Unlike the default HuggingFace LabelSmoother, this:
    1. Upsamples logits to match label resolution
    2. Applies label smoothing via soft cross-entropy
    3. Properly handles ignore_index for unlabeled pixels
    
    Args:
        epsilon: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
        ignore_index: Index to ignore in loss computation (default: 255 for segmentation)
    """
    
    def __init__(self, epsilon: float = 0.1, ignore_index: int = 255):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss for segmentation.
        
        Args:
            logits: Model output logits [B, C, H_out, W_out] (potentially lower resolution)
            labels: Ground truth labels [B, H, W] (full resolution)
            
        Returns:
            Scalar loss tensor
        """
        # Get dimensions
        batch_size, num_classes, h_out, w_out = logits.shape
        h_label, w_label = labels.shape[1], labels.shape[2]
        
        # Upsample logits to match label resolution if needed
        if h_out != h_label or w_out != w_label:
            logits = F.interpolate(
                logits,
                size=(h_label, w_label),
                mode="bilinear",
                align_corners=False,
            )
        
        # Reshape for loss computation: [B, C, H, W] -> [B*H*W, C]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
        labels_flat = labels.reshape(-1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = labels_flat != self.ignore_index
        
        if not valid_mask.any():
            # No valid pixels, return zero loss
            return logits.sum() * 0.0
        
        # Filter to valid pixels only
        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        # Compute log softmax
        log_probs = F.log_softmax(logits_valid, dim=-1)
        
        # Create smoothed targets
        # For label smoothing: target = (1 - epsilon) * one_hot + epsilon / num_classes
        num_valid = labels_valid.size(0)
        
        # One-hot encoding
        one_hot = torch.zeros_like(log_probs).scatter_(
            dim=-1,
            index=labels_valid.unsqueeze(-1),
            value=1.0,
        )
        
        # Apply label smoothing
        smoothed_targets = (1.0 - self.epsilon) * one_hot + self.epsilon / num_classes
        
        # Compute cross-entropy with soft targets: -sum(p * log(q))
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class SegmentationTrainer(Trainer):
    """
    Custom Trainer for semantic segmentation with proper label smoothing support
    and differential learning rates.
    
    This trainer overrides:
    - compute_loss: Apply label smoothing correctly for segmentation
    - create_optimizer: Support different learning rates for backbone vs decoder
    
    Usage:
        trainer = SegmentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            label_smoothing_factor=0.1,
            ignore_index=0,
            backbone_lr=1e-5,  # Lower LR for pretrained backbone
        )
    """
    
    def __init__(
        self,
        *args,
        label_smoothing_factor: float = 0.0,
        ignore_index: int = 255,
        backbone_lr: Optional[float] = None,  # If None, uses same LR for all params
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.seg_label_smoothing_factor = label_smoothing_factor
        self.seg_ignore_index = ignore_index
        self.backbone_lr = backbone_lr
        
        if label_smoothing_factor > 0:
            self.label_smoother = SegmentationLabelSmoother(
                epsilon=label_smoothing_factor,
                ignore_index=ignore_index,
            )
        else:
            self.label_smoother = None
    
    def create_optimizer(self):
        """
        Create optimizer with differential learning rates for backbone vs decoder.
        
        If backbone_lr is set, the encoder (backbone) parameters get backbone_lr,
        and the decode_head parameters get the base learning_rate from args.
        """
        if self.backbone_lr is None:
            # Use default behavior - same LR for all params
            return super().create_optimizer()
        
        model = self.model
        base_lr = self.args.learning_rate
        backbone_lr = self.backbone_lr
        
        # Separate parameters into backbone and decoder groups
        backbone_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # SegFormer structure:
            # - Backbone: segformer.encoder.* (includes patch_embeddings, block, layer_norm)
            # - Decoder: decode_head.* (includes linear_c, linear_fuse, batch_norm, classifier)
            if "segformer.encoder" in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        # Create parameter groups with different learning rates
        optimizer_grouped_parameters = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": decoder_params,
                "lr": base_lr,
                "name": "decoder",
            },
        ]
        
        # Get optimizer class and kwargs from Trainer
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        
        # Remove 'lr' from kwargs since we set it per group
        optimizer_kwargs.pop("lr", None)
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # Log the configuration
        if self.args.local_rank <= 0:
            print(f"Differential LR: backbone={backbone_lr}, decoder={base_lr}")
            print(f"  Backbone params: {len(backbone_params)}")
            print(f"  Decoder params: {len(decoder_params)}")
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create a cosine scheduler with min_lr support.
        
        Uses torch.optim.lr_scheduler.CosineAnnealingLR which supports eta_min (min_lr).
        Falls back to default HF scheduler if min_lr is not specified.
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
        
        if optimizer is None:
            optimizer = self.optimizer
        
        # Check if min_lr is specified in scheduler kwargs
        min_lr = self.args.lr_scheduler_kwargs.get("min_lr", None)
        
        if min_lr is None or self.args.lr_scheduler_type != "cosine":
            # Use default HF scheduler
            return super().create_scheduler(num_training_steps, optimizer)
        
        # Calculate warmup steps
        warmup_steps = int(num_training_steps * self.args.warmup_ratio) if self.args.warmup_ratio > 0 else self.args.warmup_steps
        
        # Create warmup scheduler (linear warmup)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-10 / self.args.learning_rate,  # Start from near-zero
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Create cosine annealing scheduler with min_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=min_lr,
        )
        
        # Combine warmup + cosine
        self.lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        
        if self.args.local_rank <= 0:
            print(f"Custom cosine scheduler with min_lr={min_lr}, warmup_steps={warmup_steps}")
        
        return self.lr_scheduler
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        Compute loss with optional segmentation-aware label smoothing.
        
        If label_smoothing_factor > 0, uses custom SegmentationLabelSmoother.
        Otherwise, uses the model's built-in loss computation.
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # If no label smoothing, use model's built-in loss
        if self.label_smoother is None or labels is None:
            loss = outputs.loss if isinstance(outputs, dict) else outputs["loss"]
        else:
            # Use custom segmentation label smoother
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            loss = self.label_smoother(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

