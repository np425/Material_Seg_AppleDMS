"""
Training script for Mask2Former on Apple-DMS material segmentation dataset.
Supports multi-GPU training with DDP via torchrun.

Run with: OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 training/mask2former/train.py
"""

import json
import os

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/transformers
# ============================================================================
# Prevent OpenMP thread oversubscription warning
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Solve CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_local_rank() -> int:
    """Get local rank from environment variable set by torchrun."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0) in DDP.
    Works BEFORE dist.init_process_group is called by checking LOCAL_RANK env var.
    """
    return get_local_rank() == 0


import sys
import torch.distributed as dist
from pathlib import Path

# Add training/mask2former/ to sys.path so local utils module is found
_MASK2FORMER_DIR = Path(__file__).resolve().parent
if str(_MASK2FORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_MASK2FORMER_DIR))

# Add dataset_helpers/ to sys.path so dataset_utils is found
_DATASET_DIR = _MASK2FORMER_DIR.parent.parent / "dataset_helpers"
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

# Import local utils AFTER setting up sys.path
from utils import Mask2FormerTrainer, Mask2FormerDataCollator

from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback
from dotenv import load_dotenv

from dataset_utils import (
    load_dms_from_hub,
    get_class_labels,
)

load_dotenv()


def train(train_config):
    """Train Mask2Former on Apple-DMS dataset."""

    # 1. Load processor first (needed for dataset preprocessing)
    pretrained_model_name = train_config.get(
        "pretrained_model_name", 
        "facebook/mask2former-swin-base-ade-semantic"
    )
    
    if is_main_process():
        print(f"Loading image processor from: {pretrained_model_name}")
    
    # Use slow processor to avoid grouping issues with variable-size images
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast=False)
    
    # 2. Load dataset with the processor
    if is_main_process():
        print("Loading dataset from Hub...")
    
    dataset = load_dms_from_hub(
        repo_id="AllanK24/apple-dms-materials-v2",
        processor=processor,
        augment_train=train_config.get("augment_train", False),
        model_type="mask2former",
        crop_size=train_config.get("augmentation_crop_size", (512, 512)),
    )
    
    # Get class labels (already cached from load_dms_from_hub)
    labels_info = get_class_labels(source="hub")
    id2label = labels_info["id2label"]
    label2id = labels_info["label2id"]
    num_labels = labels_info["num_labels"]
    
    if is_main_process():
        print(f"Number of classes: {num_labels}")

    # 3. Initialize the model with correct number of labels
    if is_main_process():
        print(f"Loading model from: {pretrained_model_name}")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # Required when changing num_labels
    )

    # 4. Initialize the training arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        learning_rate=train_config["learning_rate"],

        # Additional important hyperparams
        weight_decay=train_config["weight_decay"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        
        torch_compile=train_config["torch_compile"],
        torch_compile_backend=train_config["torch_compile_backend"],
        torch_compile_mode=train_config["torch_compile_mode"],

        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        save_total_limit=train_config["save_total_limit"],
        
        # Strategies must match for load_best_model_at_end
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        save_steps=train_config["save_steps"],
        eval_steps=train_config["eval_steps"],
        
        logging_steps=train_config["logging_steps"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        dataloader_num_workers=train_config["dataloader_num_workers"],
        seed=train_config.get("seed", 42),
        
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=True,  # IoU is better when higher
        
        push_to_hub=train_config.get("push_to_hub", True),
        hub_model_id=train_config["hub_model_id"],
        hub_strategy="end",
        hub_token=os.getenv("HF_TOKEN"),
        
        report_to="tensorboard",
        run_name=train_config["run_name"],
        
        # DDP settings
        ddp_find_unused_parameters=False,
        
        # Move predictions to CPU periodically during eval
        eval_accumulation_steps=train_config.get("eval_accumulation_steps", 8),
        
        # Important for custom datasets
        remove_unused_columns=False,
    )

    # 5. Initialize the Mask2FormerTrainer (custom trainer with confusion matrix evaluation)
    # Note: compute_metrics is handled internally via confusion matrix accumulation
    trainer = Mask2FormerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=Mask2FormerDataCollator(),  # Custom collator for variable-length masks
        image_processor=processor,  # Required for post-processing
        num_classes=num_labels,  # For confusion matrix
        id2label=id2label,  # For per-class metric names
        ignore_index=0,  # "No label" class to ignore in metrics
        label_smoothing=train_config.get("label_smoothing", 0.0),  # Label smoothing for classification
        backbone_lr=train_config.get("backbone_lr", None),  # Differential LR for backbone
        min_lr=train_config.get("min_lr", 0.0),  # Min LR for cosine scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.01)],
    )

    # 7. Train the model
    if is_main_process():
        print("Starting training...")
    trainer.train()

    # 8. Push the model to the hub (only main process should push)
    if is_main_process():
        hub_model_id = train_config["hub_model_id"]
        hf_dataset_identifier = "AllanK24/apple-dms-materials-v2"
        
        kwargs = {
            "tags": ["vision", "image-segmentation", "mask2former", "material-segmentation"],
            "finetuned_from": pretrained_model_name,
            "dataset": hf_dataset_identifier,
        }

        print(f"Pushing model to Hub: {hub_model_id}")
        processor.push_to_hub(hub_model_id)
        trainer.push_to_hub(**kwargs)
        
        print("Training complete!")
    
    return trainer


if __name__ == "__main__":
    # RUN number
    run_number = 2
    
    os.makedirs(f"/data/material_classification/checkpoints_v2/mask2former/swin-large/", exist_ok=True)

    train_config = {
        "output_dir": f"/data/material_classification/checkpoints_v2/mask2former/swin-large/run{run_number}",
        "pretrained_model_name": "facebook/mask2former-swin-large-ade-semantic",
        "learning_rate": 1e-3,  # Main LR for decoder/other components
        "backbone_lr": 1e-4,  # Lower LR for Swin backbone (10x lower)
        "min_lr": 1e-6,  # Min LR for cosine scheduler
        "num_train_epochs": 40,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 16,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "lr_scheduler_type": "cosine",  # Used as fallback, our custom scheduler handles it
        "label_smoothing": 0.1,  # Label smoothing for regularization

        # Compile
        "torch_compile": True,
        "torch_compile_backend": "inductor",
        "torch_compile_mode": "default",
        
        "metric_for_best_model": "eval_mean_iou",
        "greater_is_better": True,
        "fp16": False,
        "bf16": True,
        "save_total_limit": 2,
        "eval_strategy": "steps",
        "eval_steps": 350,
        "save_strategy": "steps",
        "save_steps": 350,
        "logging_steps": 350,
        "seed": 42,
        "dataloader_num_workers": 4,  # Reduced from 10; with 8 DDP procs = 32 workers total
        "load_best_model_at_end": True,
        "push_to_hub": True,
        "hub_model_id": f"AllanK24/mask2former-swin-large-apple-dms-v2-run{run_number}",
        "run_name": f"mask2former-swin-large-apple-dms-v2-run{run_number}",
        
        # CPU offload frequency for eval predictions
        "eval_accumulation_steps": 8,

        # Augmentation (set to True to enable material-specific augmentations on train set)
        "augment_train": True,
        "augmentation_crop_size": (512, 512),  # Uncomment to override LSJ crop size
    }

    # Only main process saves config and prints
    if is_main_process():
        os.makedirs(f"configs_v2/mask2former/swin-large/", exist_ok=True)
        with open(f"configs_v2/mask2former/swin-large/run{run_number}_train_config.json", "w") as f:
            json.dump(train_config, f, indent=2)
        print(f"Config saved to configs_v2/mask2former/swin-large/run{run_number}_train_config.json")

    train(train_config)