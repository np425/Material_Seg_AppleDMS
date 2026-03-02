# Material Segmentation on Apple-DMS

> **Paper submitted to ICCSA 2026** — International Conference on Computational Science and Its Applications

Fine-tuning state-of-the-art semantic segmentation models on the [Apple Dense Material Segmentation (DMS)](https://machinelearning.apple.com/research/dms-dataset) dataset for per-pixel material recognition across **57 material classes**.

## Models

| Architecture | Backbone | Pretrained Checkpoint | Input Size |
|---|---|---|---|
| [SegFormer-B5](https://arxiv.org/abs/2105.15203) | MiT-B5 | `nvidia/segformer-b5-finetuned-ade-640-640` | 640 × 640 |
| [Mask2Former](https://arxiv.org/abs/2112.01527) | Swin-L | `facebook/mask2former-swin-large-ade-semantic` | 512 × 512 |

Both models are trained with multi-GPU DDP via `torchrun`, BF16 mixed precision, `torch.compile` (Inductor backend), and pushed to the 🤗 Hub after training.

## Datasets

Two versions of the Apple-DMS dataset are hosted on 🤗 Hugging Face:

| Dataset | Description | Link |
|---|---|---|
| **apple-dms-materials** | Original full-resolution Apple-DMS split (train / validation / test) with 57 material classes. | [AllanK24/apple-dms-materials](https://huggingface.co/datasets/AllanK24/apple-dms-materials) |
| **apple-dms-materials-v2** | Revised version with improved split balancing and custom stratified splitting logic. Used for the final training runs reported in the paper. | [AllanK24/apple-dms-materials-v2](https://huggingface.co/datasets/AllanK24/apple-dms-materials-v2) |

Each dataset includes `image` (RGB) and `label` (single-channel segmentation mask) columns, plus a `class_info.json` metadata file with `id2label`, `label2id`, and colormap.

<details>
<summary><b>57 Material Classes</b> (click to expand)</summary>

`No label` · `Animal skin` · `Bone/teeth/horn` · `Brickwork` · `Cardboard` · `Carpet/rug` · `Ceiling tile` · `Ceramic` · `Chalkboard/blackboard` · `Clutter` · `Concrete` · `Cork/corkboard` · `Engineered stone` · `Fabric/cloth` · `Fiberglass wool` · `Fire` · `Foliage` · `Food` · `Fur` · `Gemstone/quartz` · `Glass` · `Hair` · `I cannot tell` · `Ice` · `Leather` · `Liquid, non-water` · `Metal` · `Mirror` · `Not on list` · `Paint/plaster/enamel` · `Paper` · `Pearl` · `Photograph/painting` · `Plastic, clear` · `Plastic, non-clear` · `Rubber/latex` · `Sand` · `Skin/lips` · `Sky` · `Snow` · `Soap` · `Soil/mud` · `Sponge` · `Stone, natural` · `Stone, polished` · `Styrofoam` · `Tile` · `Wallpaper` · `Water` · `Wax` · `Whiteboard` · `Wicker` · `Wood` · `Wood, tree` · `Bad polygon` · `Multiple materials` · `Asphalt`

</details>

## Repository Structure

```
├── dataset_helpers/          # Dataset preparation & upload utilities
│   ├── dataset_utils.py      # Core dataset loading, class labels, image processors
│   ├── create_custom_split.py# Stratified train/val/test splitting
│   ├── download_dms_images.py# Download raw Apple-DMS images
│   └── push_to_hub.py        # Push processed dataset to HuggingFace Hub
│
├── training/                 # Model training scripts
│   ├── segformer/
│   │   ├── train.py          # SegFormer DDP training script
│   │   ├── utils.py          # Metric computation (mean IoU, per-class IoU)
│   │   ├── label_smoother_utils.py  # Custom segmentation label smoothing trainer
│   │   └── augmentations.py  # Physics-aware data augmentations
│   └── mask2former/
│       ├── train.py          # Mask2Former DDP training script
│       ├── utils.py          # Custom trainer with confusion matrix evaluation
│       └── predict_test_samples.py  # Qualitative test-set predictions
│
├── helpers/                  # Evaluation & visualization scripts
│   ├── evaluate_test_segformer.py   # Distributed evaluation (SegFormer)
│   ├── evaluate_test_mask2former.py # Distributed evaluation (Mask2Former)
│   ├── predict_custom.py     # Run inference on custom images
│   ├── predict_test_samples.py      # Visualize random test samples
│   ├── save_predictions.py   # Save raw prediction arrays
│   ├── plot_predictions.py   # Paper-quality plot generation
│   └── plot_metrics.py       # Plot training metrics from TensorBoard logs
```

## Key Features

### Training
- **Multi-GPU DDP** training via `torchrun` (tested on 8× GPU setups)
- **Differential learning rates** — lower LR for pretrained backbone, higher LR for decoder head
- **Cosine LR scheduler** with warmup and configurable minimum LR
- **Label smoothing** — custom implementation that handles SegFormer's 4× resolution mismatch between logits and labels
- **Early stopping** based on validation mean IoU
- **Automatic Hub push** — trained models are uploaded to 🤗 Hub at the end of training

### Data Augmentation
Physics-aware augmentation pipeline tailored for material recognition:
- **Large Scale Jittering (LSJ)** — random resize + crop, the standard for ViT-based dense prediction
- **Constrained Color Jitter** — conservative hue/saturation/contrast shifts to avoid semantic drift (e.g., wood → painted surface)
- **Specular Highlight Injection** — synthetic Gaussian blobs simulating view-dependent glare on shiny materials
- **Gaussian Noise** — sensor grain simulation (SegFormer only; disabled for Mask2Former to preserve mask boundaries)

### Evaluation
- **Mean IoU** and **per-class IoU/accuracy** on the test set
- **Boundary IoU** using morphological operations
- **Confusion matrix accumulation** for memory-efficient distributed evaluation (Mask2Former)
- Side-by-side visualizations: Original | Ground Truth | Prediction

## Usage

### Prerequisites

```bash
pip install torch transformers datasets evaluate huggingface_hub python-dotenv matplotlib
```

Create a `.env` file with your Hugging Face token:
```
HF_TOKEN=hf_your_token_here
```

### Training

**SegFormer-B5:**
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 training/segformer/train.py
```

**Mask2Former-Swin-L:**
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 training/mask2former/train.py
```

Training hyperparameters (learning rate, epochs, augmentation, etc.) are configured directly in each `train.py` file via the `train_config` dictionary.

### Evaluation

```bash
# SegFormer — distributed evaluation on test set
torchrun --nproc_per_node=8 helpers/evaluate_test_segformer.py \
    --run_dir /path/to/run/directory \
    --output_dir /path/to/output

# Mask2Former — distributed evaluation on test set
torchrun --nproc_per_node=8 helpers/evaluate_test_mask2former.py \
    --checkpoint /path/to/checkpoint \
    --output_dir /path/to/output
```

### Inference on Custom Images

```bash
python helpers/predict_custom.py \
    --run_dir /path/to/run/directory \
    --image_dir sample_images/ \
    --output_dir predictions/
```

## Citation

If you use this code or the datasets, please cite:

```bibtex
@inproceedings{material_seg_apple_dms_iccsa2026,
  title     = {Material Segmentation on Apple-DMS},
  author    = {Allan K.},
  booktitle = {International Conference on Computational Science and Its Applications (ICCSA)},
  year      = {2026},
}
```

## License

This project is for academic research purposes. The Apple-DMS dataset is subject to [Apple's original license terms](https://machinelearning.apple.com/research/dms-dataset).
