---
annotations_creators:
- expert-generated
language:
- en
license: apple-ascl
multilinguality: monolingual
pretty_name: Apple Dense Material Segmentation (DMS) – Stratified Split
size_categories:
- 10K<n<100K
source_datasets:
- extended|open-images-v7
tags:
- material-segmentation
- semantic-segmentation
- dense-prediction
- materials
- segformer
- mask2former
- stratified-split
task_categories:
- image-segmentation
task_ids:
- semantic-segmentation
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: image
  - name: image_id
    dtype: string
  splits:
  - name: train
    num_examples: 33118
  - name: validation
    num_examples: 4138
  - name: test
    num_examples: 4140
---

# Apple Dense Material Segmentation (DMS) – Stratified 80/10/10 Split

A **pixel-level material segmentation** dataset containing ~41K images with dense annotations across **57 material categories**. Originally released by Apple as part of the [Dense Material Segmentation (DMS)](https://machinelearning.apple.com/research/dense-material-segmentation) research project.

This version uses a **custom stratified 80/10/10 split** (vs Apple's original 54/23/23) to maximise training data while maintaining representative validation and test sets.

## Why a Custom Split?

Apple's original split reserves nearly half the data for evaluation (23% val + 23% test). Our re-split allocates **80% to training** while using **stratified sampling** (based on the dominant material class per image) to keep val/test distributions aligned with the training set.

### Split Quality Comparison

| Metric | Original (Apple) | Custom (Stratified) | Improvement |
|--------|------------------|---------------------|-------------|
| Train size | 22,492 (54%) | **33,118 (80%)** | +47% more training data |
| JSD train↔val | 0.0524 | **0.0158** | ✅ **70% lower divergence** |
| JSD train↔test | 0.0526 | **0.0163** | ✅ **69% lower divergence** |
| Classes in all splits | 53/57 | 53/57 | Equal coverage |

> **JSD** = Jensen-Shannon Divergence between pixel-level class distributions. Lower values mean the evaluation sets better represent the training distribution, leading to more reliable metrics.

## Dataset Description

Each sample consists of:

| Field | Type | Description |
|-------|------|-------------|
| `image` | `PIL.Image` | RGB input image |
| `label` | `PIL.Image` | Single-channel segmentation mask (pixel values = class indices 0–56) |
| `image_id` | `string` | Unique image identifier |

### Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 33,118 | 80.0% |
| Validation | 4,138 | 10.0% |
| Test | 4,140 | 10.0% |
| **Total** | **41,396** | **100%** |

### Material Classes (57)

<details>
<summary>Click to expand full class list</summary>

| ID | Material | ID | Material | ID | Material |
|----|----------|----|----------|----|----------|
| 0 | No label | 19 | Gemstone/quartz | 38 | Sky |
| 1 | Animal skin | 20 | Glass | 39 | Snow |
| 2 | Bone/teeth/horn | 21 | Hair | 40 | Soap |
| 3 | Brickwork | 22 | I cannot tell | 41 | Soil/mud |
| 4 | Cardboard | 23 | Ice | 42 | Sponge |
| 5 | Carpet/rug | 24 | Leather | 43 | Stone, natural |
| 6 | Ceiling tile | 25 | Liquid, non-water | 44 | Stone, polished |
| 7 | Ceramic | 26 | Metal | 45 | Styrofoam |
| 8 | Chalkboard/blackboard | 27 | Mirror | 46 | Tile |
| 9 | Clutter | 28 | Not on list | 47 | Wallpaper |
| 10 | Concrete | 29 | Paint/plaster/enamel | 48 | Water |
| 11 | Cork/corkboard | 30 | Paper | 49 | Wax |
| 12 | Engineered stone | 31 | Pearl | 50 | Whiteboard |
| 13 | Fabric/cloth | 32 | Photograph/painting | 51 | Wicker |
| 14 | Fiberglass wool | 33 | Plastic, clear | 52 | Wood |
| 15 | Fire | 34 | Plastic, non-clear | 53 | Wood, tree |
| 16 | Foliage | 35 | Rubber/latex | 54 | Bad polygon |
| 17 | Food | 36 | Sand | 55 | Multiple materials |
| 18 | Fur | 37 | Skin/lips | 56 | Asphalt |

</details>

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("AllanK24/apple-dms-materials-v2")

# Access splits
train_ds = dataset["train"]      # 33,118 samples
val_ds = dataset["validation"]   #  4,138 samples
test_ds = dataset["test"]        #  4,140 samples

# View a sample
sample = train_ds[0]
sample["image"].show()    # RGB image
sample["label"].show()    # Segmentation mask
```

### Training with SegFormer / Mask2Former

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessorFast
import json
from huggingface_hub import hf_hub_download

# Load class info
class_info_path = hf_hub_download(
    repo_id="AllanK24/apple-dms-materials-v2",
    filename="class_info.json",
    repo_type="dataset",
)
with open(class_info_path) as f:
    class_info = json.load(f)

id2label = {int(k): v for k, v in class_info["id2label"].items()}
label2id = class_info["label2id"]
num_labels = class_info["num_labels"]

# Initialize model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Initialize processor
processor = SegformerImageProcessorFast.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

# Apply transforms
def transforms(batch):
    images = [x.convert("RGB") for x in batch["image"]]
    labels = [x for x in batch["label"]]
    return processor(images=images, segmentation_maps=labels, return_tensors="pt")

train_ds.set_transform(transforms)
```

## Stratification Method

The split was created using a two-level stratified sampling approach:
1. **Dominant class extraction** – For each image, the material class with the most pixels (excluding "No label") is identified.
2. **First split** – Images are stratified into 80% train vs 20% eval using `StratifiedShuffleSplit`.
3. **Second split** – The 20% eval pool is stratified 50/50 into validation and test.
4. **Rare class handling** – Classes with <5 total images go directly to train; classes with <2 images in the eval pool are randomly assigned between val/test.

Seed: `42` (for reproducibility).

## Source & Preparation

- **Original dataset**: [Apple DMS](https://github.com/apple/ml-dms-dataset) with images from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
- **Original split** (v1): [AllanK24/apple-dms-materials](https://huggingface.co/datasets/AllanK24/apple-dms-materials)
- **Preparation pipeline**: Download → resize/align (`prepare_images.py`) → validate (`check_images.py`, 41,385/41,396 passed) → stratified re-split

## Citation

```bibtex
@article{upchurch2022dense,
  title={Dense Material Segmentation with Context-Aware Network},
  author={Upchurch, Paul and Niu, Ransen},
  year={2022},
  url={https://machinelearning.apple.com/research/dense-material-segmentation}
}
```

## License

Released under the [Apple Sample Code License (ASCL)](https://developer.apple.com/sample-code/license/apple-sample-code-license/). Source images are from Open Images V7 (primarily CC BY 2.0). See the [original repository](https://github.com/apple/ml-dms-dataset) for full licensing details.
