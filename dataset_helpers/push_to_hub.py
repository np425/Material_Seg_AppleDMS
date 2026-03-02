#!/usr/bin/env python3
"""
Push the Apple-DMS dataset to HuggingFace Hub.

This script loads the prepared Apple-DMS dataset and pushes it to the HuggingFace Hub
as a semantic segmentation dataset suitable for training SegFormer models.

Usage:
    python push_to_hub.py --data_path /path/to/DMS_v1 --repo_id AllanK24/apple-dms-materials
    
    # Dry run to validate without uploading:
    python push_to_hub.py --data_path /path/to/DMS_v1 --repo_id AllanK24/apple-dms-materials --dry_run
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Image, ClassLabel
from PIL import Image as PILImage
from tqdm import tqdm


def load_taxonomy(data_path: str) -> Dict:
    """Load the taxonomy (class labels and colormap) from the dataset."""
    taxonomy_path = os.path.join(data_path, "taxonomy.json")
    with open(taxonomy_path, "r") as f:
        taxonomy = json.load(f)
    return taxonomy


def load_dataset_info(data_path: str) -> List[Dict]:
    """Load dataset metadata from info.json.gz."""
    info_path = os.path.join(data_path, "info.json.gz")
    with gzip.open(info_path, "rb") as f:
        data = json.loads(f.read())
    return data


def get_split_from_label_path(label_path: str) -> Optional[str]:
    """Extract the split (train/validation/test) from the label path."""
    if "train" in label_path:
        return "train"
    elif "validation" in label_path:
        return "validation"
    elif "test" in label_path:
        return "test"
    return None


def create_dataset_examples(
    data_path: str,
    data: List[Dict],
    split: str,
    skip_missing: bool = True
) -> List[Dict]:
    """Create examples for a specific split."""
    examples = []
    
    for datum in tqdm(data, desc=f"Processing {split}"):
        label_path = datum.get("label_path", "")
        datum_split = get_split_from_label_path(label_path)
        
        if datum_split != split:
            continue
            
        image_path = os.path.join(data_path, datum["image_path"])
        full_label_path = os.path.join(data_path, label_path)
        
        # Skip if files don't exist
        if not os.path.exists(image_path):
            if skip_missing:
                continue
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not os.path.exists(full_label_path):
            if skip_missing:
                continue
            else:
                raise FileNotFoundError(f"Label not found: {full_label_path}")
        
        # Extract image ID from path (e.g., "images/123456.jpg" -> "123456")
        image_id = Path(datum["image_path"]).stem
        
        examples.append({
            "image": image_path,
            "label": full_label_path,
            "image_id": image_id,
            "width": datum.get("width"),
            "height": datum.get("height"),
        })
    
    return examples


def create_hf_dataset(data_path: str, dry_run: bool = False) -> DatasetDict:
    """Create a HuggingFace DatasetDict from the DMS dataset."""
    print(f"Loading dataset from {data_path}")
    
    # Load metadata
    taxonomy = load_taxonomy(data_path)
    data = load_dataset_info(data_path)
    
    print(f"Dataset contains {len(data)} total samples")
    print(f"Number of classes: {len(taxonomy['names'])}")
    
    # Create examples for each split
    splits = {}
    for split_name in ["train", "validation", "test"]:
        examples = create_dataset_examples(data_path, data, split_name)
        print(f"  {split_name}: {len(examples)} samples")
        
        if examples:
            if dry_run:
                # In dry run, only use first 10 examples per split
                examples = examples[:10]
            
            splits[split_name] = Dataset.from_dict({
                "image": [ex["image"] for ex in examples],
                "label": [ex["label"] for ex in examples],
                "image_id": [ex["image_id"] for ex in examples],
            })
            
            # Cast image columns to Image feature
            splits[split_name] = splits[split_name].cast_column("image", Image())
            splits[split_name] = splits[split_name].cast_column("label", Image())
    
    dataset_dict = DatasetDict(splits)
    
    # Add class labels as dataset info
    print(f"\nClass names: {taxonomy['names'][:5]}... (showing first 5)")
    
    return dataset_dict, taxonomy


def main():
    parser = argparse.ArgumentParser(
        description="Push Apple-DMS dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the DMS_v1 dataset directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="AllanK24/apple-dms-materials",
        help="HuggingFace Hub repository ID (default: AllanK24/apple-dms-materials)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate dataset creation without pushing to Hub"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace Hub"
    )
    
    args = parser.parse_args()
    
    # Create dataset
    dataset_dict, taxonomy = create_hf_dataset(args.data_path, args.dry_run)
    
    print(f"\nDataset structure:")
    print(dataset_dict)
    
    if args.dry_run:
        print("\n[DRY RUN] Dataset created successfully!")
        print("  Sample from train split:")
        if "train" in dataset_dict:
            sample = dataset_dict["train"][0]
            print(f"    image_id: {sample['image_id']}")
            print(f"    image: {sample['image']}")
            print(f"    label: {sample['label']}")
        
        # Save class info locally for reference
        class_info_path = os.path.join(args.data_path, "class_info.json")
        with open(class_info_path, "w") as f:
            json.dump({
                "id2label": {i: name for i, name in enumerate(taxonomy["names"])},
                "label2id": {name: i for i, name in enumerate(taxonomy["names"])},
                "num_labels": len(taxonomy["names"]),
            }, f, indent=2)
        print(f"\n  Class info saved to: {class_info_path}")
        return
    
    # Push to Hub
    print(f"\nPushing dataset to {args.repo_id}...")
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
    )
    
    print(f"\n✅ Dataset successfully pushed to: https://huggingface.co/datasets/{args.repo_id}")
    
    # Also save class info as a separate file in the repo
    print("Saving class labels metadata...")
    from huggingface_hub import HfApi
    
    api = HfApi()
    class_info = {
        "id2label": {i: name for i, name in enumerate(taxonomy["names"])},
        "label2id": {name: i for i, name in enumerate(taxonomy["names"])},
        "num_labels": len(taxonomy["names"]),
        "colormap": taxonomy.get("srgb_colormap", taxonomy.get("colormap")),
    }
    
    # Save locally and upload
    class_info_path = "/tmp/class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(class_info, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=class_info_path,
        path_in_repo="class_info.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    
    print("✅ Class info uploaded!")


if __name__ == "__main__":
    main()
