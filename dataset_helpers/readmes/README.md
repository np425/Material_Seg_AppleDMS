---
annotations_creators:
- expert-generated
language:
- en
license: apple-ascl
multilinguality: monolingual
pretty_name: Apple Dense Material Segmentation (DMS)
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
    num_examples: 22492
  - name: validation
    num_examples: 9412
  - name: test
    num_examples: 9492
---

# Apple Dense Material Segmentation (DMS) Dataset

A **pixel-level material segmentation** dataset containing ~41K images with dense annotations across **57 material categories**. Originally released by Apple as part of the [Dense Material Segmentation (DMS)](https://machinelearning.apple.com/research/dense-material-segmentation) research project.

> **Note**: This is a mirror prepared for direct use with the HuggingFace 🤗 `datasets` library. The source images originate from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), and material annotations were created by Apple. Some images (~6%) from the original dataset could not be retrieved from Open Images and are therefore absent.

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
| Train | 22,492 | 54.3% |
| Validation | 9,412 | 22.7% |
| Test | 9,492 | 22.9% |
| **Total** | **41,396** | **100%** |

The split assignments follow the original Apple DMS partition.

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

dataset = load_dataset("AllanK24/apple-dms-materials")

# Access splits
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

# View a sample
sample = train_ds[0]
print(sample["image_id"])   # e.g. "22491"
sample["image"].show()      # RGB image
sample["label"].show()      # Segmentation mask
```

### Training with SegFormer

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessorFast
import json

# Load class info
from huggingface_hub import hf_hub_download
class_info_path = hf_hub_download(
    repo_id="AllanK24/apple-dms-materials",
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
```

### Applying Transforms

```python
def transforms(batch):
    images = [x.convert("RGB") for x in batch["image"]]
    labels = [x for x in batch["label"]]
    inputs = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return inputs

train_ds.set_transform(transforms)
```

## Dataset Preparation

This dataset was prepared from the original Apple DMS release using the following pipeline:

1. **Download** – Source images retrieved from Open Images V7 using URLs in Apple's metadata.
2. **Resize & align** – Images resized to match label dimensions using Apple's [`prepare_images.py`](https://github.com/apple/ml-dms-dataset).
3. **Validation** – Image–label consistency verified with Apple's `check_images.py` (41,385 / 41,396 passed; 11 minor rotation warnings).

## Citation

If you use this dataset, please cite the original Apple paper:

```bibtex
@article{upchurch2022dense,
  title={Dense Material Segmentation with Context-Aware Network},
  author={Upchurch, Paul and Niu, Ransen},
  year={2022},
  url={https://machinelearning.apple.com/research/dense-material-segmentation}
}
```

## License

This dataset is released under the [Apple Sample Code License (ASCL)](https://developer.apple.com/sample-code/license/apple-sample-code-license/). The source images are from Open Images V7 and are subject to their respective licenses (primarily CC BY 2.0). Please refer to the [original repository](https://github.com/apple/ml-dms-dataset) for full licensing details.
