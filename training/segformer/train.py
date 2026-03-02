"""
Training script for SegFormer on Apple-DMS material segmentation dataset.
Supports multi-GPU training with DDP via torchrun.

Run with: OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 training/segformer/train.py
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


# Now import the rest - after env vars are set
import sys
from pathlib import Path

# Add training/segformer/ to sys.path so utils and label_smoother_utils are found
_SEGFORMER_DIR = Path(__file__).resolve().parent
if str(_SEGFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGFORMER_DIR))

# Add dataset_helpers/ to sys.path so dataset_utils is found
_DATASET_DIR = _SEGFORMER_DIR.parent.parent / "dataset_helpers"
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

# import wandb
import torch.distributed as dist
from transformers import SegformerForSemanticSegmentation, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from dotenv import load_dotenv

from dataset_utils import (
    load_dms_from_hub,
    get_class_labels,
    get_image_processor,
)

from utils import create_compute_metrics, preprocess_logits_for_metrics
from label_smoother_utils import SegmentationTrainer

load_dotenv()


def train(train_config):
    """Train SegFormer on Apple-DMS dataset."""

    # 0. Initialize wandb ONLY on main process to avoid duplicate runs
    # if is_main_process():
    #     wandb.init(
    #         project="apple-dms-material-segmentation",
    #         config=train_config,
    #     )
    
    # 1. Load dataset and class labels
    if is_main_process():
        print("Loading dataset from Hub...")
    dataset = load_dms_from_hub(
        repo_id="AllanK24/apple-dms-materials-v2",
        augment_train=train_config.get("augment_train", False),
        model_type="segformer",
        crop_size=train_config.get("augmentation_crop_size", (512, 512)),
    )
    
    # Get class labels (already cached from load_dms_from_hub)
    labels_info = get_class_labels(source="hub")
    id2label = labels_info["id2label"]
    label2id = labels_info["label2id"]
    num_labels = labels_info["num_labels"]
    
    if is_main_process():
        print(f"Number of classes: {num_labels}")
    
    # Get the image processor
    processor = get_image_processor()

    # 2. Initialize the model with correct number of labels
    pretrained_model_name = train_config.get(
        "pretrained_model_name", 
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    
    if is_main_process():
        print(f"Loading model from: {pretrained_model_name}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # Required when changing num_labels
    )

    # 3. Initialize the training arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        learning_rate=train_config["learning_rate"],

        # Additional important hyperparams
        weight_decay=train_config["weight_decay"],
        # max_grad_norm=train_config["max_grad_norm"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        lr_scheduler_kwargs=train_config["lr_scheduler_kwargs"],
        warmup_ratio=train_config["warmup_ratio"],
        # warmup_steps=train_config["warmup_steps"],
        # NOTE: label_smoothing_factor is passed to SegmentationTrainer instead
        label_smoothing_factor=train_config["label_smoothing_factor"],
        
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
        
        # WandB: only report from main process
        # report_to="wandb" if is_main_process() else "none",
        report_to="tensorboard" if is_main_process() else "none",
        run_name=train_config["run_name"],
        
        # DDP settings
        ddp_find_unused_parameters=False,  # No unused params in SegFormer, improves perf
        
        # Move predictions to CPU periodically during eval (works with preprocess_logits_for_metrics)
        eval_accumulation_steps=train_config.get("eval_accumulation_steps", 8),
        
        # Remove any None/empty values that could cause issues
        remove_unused_columns=False,  # Important for custom datasets
    )

    # 4. Create compute_metrics function with proper references
    compute_metrics = create_compute_metrics(
        id2label=id2label,
        num_labels=num_labels,
        ignore_index=0,  # "No label" class
        reduce_labels=processor.do_reduce_labels,
    )

    # 5. Initialize the trainer with custom label smoothing for segmentation
    trainer = SegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        # CRITICAL: Apply argmax BEFORE accumulating predictions to prevent RAM OOM
        # This reduces memory from [N,57,H,W] float32 to [N,H,W] int64 (~28× reduction)
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.01)],
        # Custom segmentation label smoothing (handles resolution mismatch)
        label_smoothing_factor=train_config["label_smoothing_factor"],
        ignore_index=0,  # "No label" class
        # Differential learning rates: backbone gets lower LR
        backbone_lr=train_config.get("backbone_lr"),
    )

    # 6. Train the model
    if is_main_process():
        print("Starting training...")
    trainer.train()

    # 7. Push the model to the hub (only main process should push)
    if is_main_process():
        hub_model_id = train_config["hub_model_id"]
        hf_dataset_identifier = "AllanK24/apple-dms-materials-v2"
        
        kwargs = {
            "tags": ["vision", "image-segmentation", "segformer", "material-segmentation"],
            "finetuned_from": pretrained_model_name,
            "dataset": hf_dataset_identifier,
        }

        print(f"Pushing model to Hub: {hub_model_id}")
        processor.push_to_hub(hub_model_id)
        trainer.push_to_hub(**kwargs)
        
        print("Training complete!")
    
    return trainer


if __name__ == "__main__":
    # RUN Number
    run_number = 2

    os.makedirs(f"/data/material_classification/checkpoints_v2/segformer/b5/", exist_ok=True)

    train_config = {
        "output_dir": f"/data/material_classification/checkpoints_v2/segformer/b5/run{run_number}",
        "pretrained_model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        "learning_rate": 1e-3,
        "num_train_epochs": 40,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 16,  # Lower than train to prevent OOM during evaluation
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_kwargs": {
            "min_lr": 1e-6,  # Custom scheduler in SegmentationTrainer supports this
        },
        "label_smoothing_factor": 0.1,  # Now works with SegmentationTrainer!

        # Differential learning rates
        "backbone_lr": 1e-4,  # Lower LR for pretrained backbone (encoder)
        # Note: decoder/head uses the main learning_rate (1e-4)

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
        "hub_model_id": f"AllanK24/segformer-b5-finetuned-apple-dms-v2-run{run_number}",
        "run_name": f"segformer-b5-finetuned-apple-dms-v2-run{run_number}",
        
        # CPU offload frequency for eval predictions (main memory fix is preprocess_logits_for_metrics)
        "eval_accumulation_steps": 8,

        # Augmentation (set to True to enable material-specific augmentations on train set)
        "augment_train": True,
        # "augmentation_crop_size": (512, 512),  # Uncomment to override LSJ crop size
    }

    # Only main process saves config and prints
    if is_main_process():
        os.makedirs("configs_v2/segformer/b5/", exist_ok=True)
        with open(f"configs_v2/segformer/b5/run{run_number}_train_config.json", "w") as f:
            json.dump(train_config, f, indent=2)
        print(f"Config saved to configs_v2/segformer/b5/run{run_number}_train_config.json")

    train(train_config)