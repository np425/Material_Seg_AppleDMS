#!/usr/bin/env python3
"""
Create a custom stratified 80/10/10 split of the Apple-DMS dataset.

This script:
1. Scans all valid image-label pairs
2. Determines the dominant material class per image (for stratification)
3. Performs stratified splitting: 80% train, 10% validation, 10% test
4. Compares class distributions between original Apple split and the new split
5. Optionally pushes the re-split dataset to HuggingFace Hub

Usage:
    python create_custom_split.py \
        --data_path /path/to/DMS_v1 \
        --output_dir /path/to/output \
        --push_to_hub \
        --repo_id AllanK24/apple-dms-materials-v2
"""

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def load_valid_samples(data_path: str) -> List[Dict]:
    """Load metadata and return only samples where both image and label exist."""
    info_path = os.path.join(data_path, "info.json.gz")
    with gzip.open(info_path, "rb") as f:
        data = json.loads(f.read())

    valid = []
    for datum in tqdm(data, desc="Checking valid samples"):
        img_p = os.path.join(data_path, datum["image_path"])
        lbl_p = os.path.join(data_path, datum["label_path"])
        if os.path.exists(img_p) and os.path.exists(lbl_p):
            # Also record original split
            lbl_path_str = datum.get("label_path", "")
            orig_split = None
            for s in ["train", "validation", "test"]:
                if s in lbl_path_str:
                    orig_split = s
                    break
            datum["_orig_split"] = orig_split
            datum["_image_path_abs"] = img_p
            datum["_label_path_abs"] = lbl_p
            valid.append(datum)

    print(f"Total valid image-label pairs: {len(valid)}")
    return valid


def compute_dominant_classes(
    samples: List[Dict],
    data_path: str,
) -> np.ndarray:
    """
    Compute the dominant (most frequent non-background) material class for each image.
    This is used as the stratification key.
    """
    dominant_classes = []

    for sample in tqdm(samples, desc="Computing dominant classes"):
        lbl = np.array(Image.open(sample["_label_path_abs"]))
        # Count pixels per class, excluding class 0 ("No label")
        class_counts = np.bincount(lbl.ravel(), minlength=57)
        class_counts[0] = 0  # Ignore "No label" for dominant class
        
        if class_counts.sum() == 0:
            # Image is entirely "No label" — use class 0
            dominant_classes.append(0)
        else:
            dominant_classes.append(int(np.argmax(class_counts)))

    return np.array(dominant_classes)


def stratified_split(
    samples: List[Dict],
    dominant_classes: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test using stratified sampling.
    
    Classes with fewer than 5 total samples are sent directly to train,
    since they can't reliably be stratified across the two-level split.
    After the first split, any classes with < 2 samples in the val+test
    pool are randomly assigned between val and test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    n = len(samples)
    indices = np.arange(n)

    # Need at least 5 samples to guarantee >=2 land in the 20% val+test pool
    class_counts = Counter(dominant_classes.tolist())
    rare_classes = {cls for cls, cnt in class_counts.items() if cnt < 5}
    
    if rare_classes:
        print(f"  {len(rare_classes)} rare classes (<5 samples) assigned directly to train.")

    # Separate rare-class samples (go straight to train)
    rare_mask = np.array([int(dc) in rare_classes for dc in dominant_classes])
    rare_indices = indices[rare_mask]
    normal_indices = indices[~rare_mask]
    normal_classes = dominant_classes[~rare_mask]

    # Step 1: Split normal samples into train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_test_ratio,
        random_state=seed,
    )
    train_idx_rel, valtest_idx_rel = next(sss1.split(normal_indices, normal_classes))

    train_indices_list = list(normal_indices[train_idx_rel]) + list(rare_indices)

    valtest_indices = normal_indices[valtest_idx_rel]
    valtest_classes = dominant_classes[valtest_indices]

    # Step 2: Split (val+test) into val and test (50/50 of the remainder)
    # Some classes may have only 1 sample in val+test; handle them separately
    valtest_class_counts = Counter(valtest_classes.tolist())
    vt_rare = {cls for cls, cnt in valtest_class_counts.items() if cnt < 2}

    if vt_rare:
        print(f"  {len(vt_rare)} classes have <2 samples in val+test pool; randomly assigned.")

    # Separate val+test rare samples — randomly assign to val or test
    vt_rare_mask = np.array([int(dc) in vt_rare for dc in valtest_classes])
    vt_rare_indices = valtest_indices[vt_rare_mask]
    vt_normal_indices = valtest_indices[~vt_rare_mask]
    vt_normal_classes = valtest_classes[~vt_rare_mask]

    # Stratified split on the normal portion
    relative_test = test_ratio / val_test_ratio
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test,
        random_state=seed,
    )
    val_idx_rel, test_idx_rel = next(sss2.split(vt_normal_indices, vt_normal_classes))
    val_indices_list = list(vt_normal_indices[val_idx_rel])
    test_indices_list = list(vt_normal_indices[test_idx_rel])

    # Randomly assign the rare val+test samples between val and test
    for idx in vt_rare_indices:
        if rng.random() < 0.5:
            val_indices_list.append(idx)
        else:
            test_indices_list.append(idx)

    # Gather samples
    train_samples = [samples[i] for i in train_indices_list]
    val_samples = [samples[i] for i in val_indices_list]
    test_samples = [samples[i] for i in test_indices_list]

    return train_samples, val_samples, test_samples


def compute_class_distribution(
    samples: List[Dict],
    num_classes: int = 57,
) -> np.ndarray:
    """
    Compute the total pixel count per class across all samples in a split.
    Returns a normalized distribution (sums to 1).
    """
    total_counts = np.zeros(num_classes, dtype=np.int64)

    for sample in tqdm(samples, desc="Computing distribution", leave=False):
        lbl = np.array(Image.open(sample["_label_path_abs"]))
        counts = np.bincount(lbl.ravel(), minlength=num_classes)
        total_counts += counts[:num_classes]

    # Normalize
    total_pixels = total_counts.sum()
    if total_pixels > 0:
        distribution = total_counts / total_pixels
    else:
        distribution = total_counts.astype(float)

    return distribution, total_counts


def compare_splits(
    original_splits: Dict[str, List[Dict]],
    custom_splits: Dict[str, List[Dict]],
    class_names: List[str],
    output_path: str,
):
    """
    Compare original and custom splits, producing a detailed JSON report.
    """
    report = {
        "summary": {},
        "original_split_sizes": {},
        "custom_split_sizes": {},
        "per_class_comparison": {},
    }

    # --- Split sizes ---
    for name, samples in original_splits.items():
        report["original_split_sizes"][name] = len(samples)
    for name, samples in custom_splits.items():
        report["custom_split_sizes"][name] = len(samples)

    # --- Class distributions ---
    print("\nComputing class distributions for original splits...")
    orig_dists = {}
    for name, samples in original_splits.items():
        if samples:
            dist, counts = compute_class_distribution(samples)
            orig_dists[name] = {"distribution": dist, "counts": counts}

    print("Computing class distributions for custom splits...")
    custom_dists = {}
    for name, samples in custom_splits.items():
        if samples:
            dist, counts = compute_class_distribution(samples)
            custom_dists[name] = {"distribution": dist, "counts": counts}

    # --- Measure distribution divergence ---
    # Jensen-Shannon divergence between train and val/test distributions
    # (lower = more similar = better stratification)
    from scipy.spatial.distance import jensenshannon

    for split_name in ["validation", "test"]:
        if split_name in orig_dists and "train" in orig_dists:
            jsd_orig = jensenshannon(
                orig_dists["train"]["distribution"],
                orig_dists[split_name]["distribution"],
            )
        else:
            jsd_orig = None

        if split_name in custom_dists and "train" in custom_dists:
            jsd_custom = jensenshannon(
                custom_dists["train"]["distribution"],
                custom_dists[split_name]["distribution"],
            )
        else:
            jsd_custom = None

        report["summary"][f"jsd_train_vs_{split_name}_original"] = float(jsd_orig) if jsd_orig is not None else None
        report["summary"][f"jsd_train_vs_{split_name}_custom"] = float(jsd_custom) if jsd_custom is not None else None

    # --- Per-class presence check ---
    # Count how many classes are present in each split
    for label_name, splits_dict in [("original", orig_dists), ("custom", custom_dists)]:
        for split_name, data in splits_dict.items():
            classes_present = int((data["counts"] > 0).sum())
            report["summary"][f"classes_present_{label_name}_{split_name}"] = classes_present

    # --- Per-class details ---
    for i, name in enumerate(class_names):
        cls_info = {"class_id": i, "class_name": name}
        for split_name in ["train", "validation", "test"]:
            if split_name in orig_dists:
                cls_info[f"original_{split_name}_pixels"] = int(orig_dists[split_name]["counts"][i])
                cls_info[f"original_{split_name}_pct"] = round(float(orig_dists[split_name]["distribution"][i]) * 100, 4)
            if split_name in custom_dists:
                cls_info[f"custom_{split_name}_pixels"] = int(custom_dists[split_name]["counts"][i])
                cls_info[f"custom_{split_name}_pct"] = round(float(custom_dists[split_name]["distribution"][i]) * 100, 4)
        report["per_class_comparison"][name] = cls_info

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SPLIT COMPARISON REPORT")
    print("=" * 70)

    print("\nSplit Sizes:")
    print(f"  {'Split':<14} {'Original':>10} {'Custom':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for split_name in ["train", "validation", "test"]:
        orig_n = report["original_split_sizes"].get(split_name, 0)
        cust_n = report["custom_split_sizes"].get(split_name, 0)
        print(f"  {split_name:<14} {orig_n:>10} {cust_n:>10}")

    print("\nJensen-Shannon Divergence (lower = better stratification):")
    for split_name in ["validation", "test"]:
        jsd_o = report["summary"].get(f"jsd_train_vs_{split_name}_original")
        jsd_c = report["summary"].get(f"jsd_train_vs_{split_name}_custom")
        improvement = ""
        if jsd_o is not None and jsd_c is not None:
            if jsd_c < jsd_o:
                pct = ((jsd_o - jsd_c) / jsd_o) * 100
                improvement = f" ✅ {pct:.1f}% better"
            else:
                pct = ((jsd_c - jsd_o) / jsd_o) * 100
                improvement = f" ⚠️ {pct:.1f}% worse"
        jsd_o_str = f"{jsd_o:.6f}" if jsd_o is not None else "N/A"
        jsd_c_str = f"{jsd_c:.6f}" if jsd_c is not None else "N/A"
        print(f"  Train vs {split_name:<12}: original={jsd_o_str}  custom={jsd_c_str}{improvement}")

    print("\nClasses Present in Each Split:")
    for label in ["original", "custom"]:
        parts = []
        for sn in ["train", "validation", "test"]:
            key = f"classes_present_{label}_{sn}"
            parts.append(f"{sn}={report['summary'].get(key, '?')}")
        print(f"  {label.capitalize():>10}: {', '.join(parts)}")

    print(f"\nDetailed report saved to: {output_path}")
    print("=" * 70)

    return report


def push_new_split_to_hub(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    repo_id: str,
    private: bool = False,
):
    """Push the re-split dataset to HuggingFace Hub."""
    from datasets import Dataset, DatasetDict, Image as HFImage

    def make_split(samples):
        return Dataset.from_dict({
            "image": [s["_image_path_abs"] for s in samples],
            "label": [s["_label_path_abs"] for s in samples],
            "image_id": [Path(s["image_path"]).stem for s in samples],
        }).cast_column("image", HFImage()).cast_column("label", HFImage())

    print("\nCreating HuggingFace DatasetDict...")
    ds = DatasetDict({
        "train": make_split(train_samples),
        "validation": make_split(val_samples),
        "test": make_split(test_samples),
    })

    print(ds)
    print(f"\nPushing to {repo_id}...")
    ds.push_to_hub(repo_id, private=private)
    print(f"✅ Pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a custom stratified 80/10/10 split of Apple-DMS"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to DMS_v1 directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save the split report and assignments",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push the re-split dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id", type=str, default="AllanK24/apple-dms-materials-v2",
        help="Hub repo ID for the re-split dataset",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the Hub dataset private",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load taxonomy
    taxonomy_path = os.path.join(args.data_path, "taxonomy.json")
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    class_names = taxonomy["names"]

    # Step 1: Load and validate samples
    print("Step 1: Loading valid samples...")
    samples = load_valid_samples(args.data_path)

    # Step 2: Compute dominant class per image
    print("\nStep 2: Computing dominant class per image...")
    dominant_classes = compute_dominant_classes(samples, args.data_path)
    print(f"  Dominant class distribution: {Counter(dominant_classes).most_common(10)} (top 10)")

    # Step 3: Create stratified split
    print("\nStep 3: Creating stratified 80/10/10 split...")
    train_samples, val_samples, test_samples = stratified_split(
        samples, dominant_classes, seed=args.seed,
    )
    print(f"  Train:      {len(train_samples):>6} ({100*len(train_samples)/len(samples):.1f}%)")
    print(f"  Validation: {len(val_samples):>6} ({100*len(val_samples)/len(samples):.1f}%)")
    print(f"  Test:       {len(test_samples):>6} ({100*len(test_samples)/len(samples):.1f}%)")

    # Step 4: Reconstruct original splits for comparison
    print("\nStep 4: Comparing with original Apple split...")
    original_splits = defaultdict(list)
    for s in samples:
        if s["_orig_split"]:
            original_splits[s["_orig_split"]].append(s)
    original_splits = dict(original_splits)

    custom_splits = {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples,
    }

    report = compare_splits(
        original_splits,
        custom_splits,
        class_names,
        output_path=os.path.join(args.output_dir, "split_comparison_report.json"),
    )

    # Step 5: Save split assignments
    split_assignments = {
        "seed": args.seed,
        "ratios": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "train_ids": [Path(s["image_path"]).stem for s in train_samples],
        "validation_ids": [Path(s["image_path"]).stem for s in val_samples],
        "test_ids": [Path(s["image_path"]).stem for s in test_samples],
    }
    assignments_path = os.path.join(args.output_dir, "split_assignments.json")
    with open(assignments_path, "w") as f:
        json.dump(split_assignments, f, indent=2)
    print(f"\nSplit assignments saved to: {assignments_path}")

    # Step 6: Push to Hub
    if args.push_to_hub:
        push_new_split_to_hub(
            train_samples, val_samples, test_samples,
            repo_id=args.repo_id,
            private=args.private,
        )


if __name__ == "__main__":
    main()
