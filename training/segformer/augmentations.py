"""
Material-segmentation-specific data augmentations for Apple-DMS.

Based on "Material Segmentation Augmentation Strategies" research report.
Implements physics-aware augmentations tailored for material recognition:

Geometric augmentations (applied to BOTH image AND label):
  - Large Scale Jittering (LSJ): random resize + crop
  - Random Horizontal Flip

Photometric augmentations (applied to image ONLY, label untouched):
  - Constrained Color Jitter (hue, saturation, contrast, brightness)
  - Specular Highlight Injection (additive Gaussian blobs)
  - Gaussian Noise (SegFormer only)

Usage:
    from augmentations import get_material_augmentation

    aug = get_material_augmentation("segformer")   # or "mask2former"
    augmented_image, label = aug(image, label)      # PIL Image, PIL Image
"""

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class SpecularHighlightInjection:
    """
    Simulates specular highlights by adding random Gaussian blobs to the image.

    This is a physics-inspired augmentation: shiny materials (metal, glass, ceramic)
    exhibit view-dependent specular highlights. By synthetically adding highlights,
    the model learns to identify materials regardless of surface glare.

    The label is NOT modified — the material under the highlight remains the same.

    Args:
        prob: Probability of applying the augmentation
        num_highlights: Range (min, max) for number of Gaussian blobs
        intensity_range: Range (min, max) for peak intensity of each blob (0-255)
        sigma_range: Range (min, max) for standard deviation of each Gaussian (in pixels)
    """

    def __init__(
        self,
        prob: float = 0.3,
        num_highlights: Tuple[int, int] = (1, 5),
        intensity_range: Tuple[int, int] = (200, 255),
        sigma_range: Tuple[float, float] = (10.0, 60.0),
    ):
        self.prob = prob
        self.num_highlights = num_highlights
        self.intensity_range = intensity_range
        self.sigma_range = sigma_range

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply specular highlight injection to a PIL image."""
        if random.random() > self.prob:
            return image

        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]

        # Create highlight map
        highlight_map = np.zeros((h, w), dtype=np.float32)

        n_highlights = random.randint(*self.num_highlights)
        for _ in range(n_highlights):
            # Random center
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)

            # Random intensity and sigma
            intensity = random.randint(*self.intensity_range)
            sigma = random.uniform(*self.sigma_range)

            # Create coordinate grids (only compute in a bounding box for efficiency)
            radius = int(3 * sigma)
            y_min = max(0, cy - radius)
            y_max = min(h, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(w, cx + radius + 1)

            y_coords = np.arange(y_min, y_max)
            x_coords = np.arange(x_min, x_max)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

            # 2D Gaussian
            gaussian = intensity * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
            )
            highlight_map[y_min:y_max, x_min:x_max] += gaussian

        # Additive blending: I' = I + H (clamped to 255)
        highlight_3ch = np.stack([highlight_map] * 3, axis=-1)
        img_array = np.clip(img_array + highlight_3ch, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def __repr__(self):
        return (
            f"SpecularHighlightInjection(prob={self.prob}, "
            f"num_highlights={self.num_highlights}, "
            f"intensity_range={self.intensity_range}, "
            f"sigma_range={self.sigma_range})"
        )


class ConstrainedColorJitter:
    """
    Material-aware color jitter with tight constraints to avoid semantic drift.

    Unlike standard ImageNet color jitter, this uses very conservative ranges
    because material identity is often chroma-dependent (e.g., shifting wood's
    hue too far makes it look like painted surface or plastic).

    Applied to image ONLY — label is untouched.

    Args:
        brightness: (min, max) brightness factor range
        contrast: (min, max) contrast factor range
        saturation: (min, max) saturation factor range
        hue_delta: Maximum hue shift in [0, 0.5] (fraction of full hue wheel)
        prob: Probability of applying the augmentation
    """

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.8, 1.2),
        contrast: Tuple[float, float] = (0.7, 1.3),
        saturation: Tuple[float, float] = (0.8, 1.2),
        hue_delta: float = 0.02,
        prob: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_delta = hue_delta
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply constrained color jitter to a PIL image."""
        if random.random() > self.prob:
            return image

        # Randomize the order of color transforms for diversity
        transforms = [
            self._adjust_brightness,
            self._adjust_contrast,
            self._adjust_saturation,
            self._adjust_hue,
        ]
        random.shuffle(transforms)

        for transform in transforms:
            image = transform(image)

        return image

    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        factor = random.uniform(*self.brightness)
        return ImageEnhance.Brightness(image).enhance(factor)

    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        factor = random.uniform(*self.contrast)
        return ImageEnhance.Contrast(image).enhance(factor)

    def _adjust_saturation(self, image: Image.Image) -> Image.Image:
        factor = random.uniform(*self.saturation)
        return ImageEnhance.Color(image).enhance(factor)

    def _adjust_hue(self, image: Image.Image) -> Image.Image:
        """Shift hue by converting to HSV, rotating H, converting back."""
        delta = random.uniform(-self.hue_delta, self.hue_delta)
        if abs(delta) < 1e-6:
            return image

        img_array = np.array(image, dtype=np.uint8)
        # Convert RGB to HSV
        hsv = np.array(image.convert("HSV"), dtype=np.float32)
        # H channel is [0, 255] in PIL HSV mode (maps to [0°, 360°])
        # delta is a fraction of the full wheel, so multiply by 255
        hsv[:, :, 0] = (hsv[:, :, 0] + delta * 255) % 256
        hsv = hsv.astype(np.uint8)
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    def __repr__(self):
        return (
            f"ConstrainedColorJitter(brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, "
            f"hue_delta={self.hue_delta}, prob={self.prob})"
        )


class GaussianNoise:
    """
    Add Gaussian (ISO) noise to simulate sensor grain.

    Forces the model to distinguish "signal texture" (e.g., wood grain) from
    "noise texture" (sensor noise), improving the robustness of texture encoders.

    Recommended for SegFormer only (the MiT encoder benefits from this).
    Not recommended for Mask2Former (can destabilize mask boundary detection).

    Args:
        sigma_range: Range (min, max) for noise standard deviation
        prob: Probability of applying the augmentation
    """

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (5.0, 25.0),
        prob: float = 0.3,
    ):
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to a PIL image."""
        if random.random() > self.prob:
            return image

        img_array = np.array(image, dtype=np.float32)
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
        img_array = np.clip(img_array + noise, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def __repr__(self):
        return f"GaussianNoise(sigma_range={self.sigma_range}, prob={self.prob})"


class LargeScaleJitter:
    """
    Large Scale Jittering (LSJ) — the gold standard geometric augmentation
    for Vision Transformers on dense prediction tasks.

    Randomly resizes the image by a factor sampled from ratio_range, then
    takes a random crop of crop_size. If the resized image is smaller than
    crop_size, it is padded with zeros (image) / ignore_value (label).

    Applied to BOTH image AND label (geometric transform).

    Args:
        ratio_range: (min_ratio, max_ratio) for random resize
        crop_size: (height, width) of the output crop
        cat_max_ratio: Maximum fraction of pixels for a single class (not enforced currently)
    """

    def __init__(
        self,
        ratio_range: Tuple[float, float] = (0.5, 2.0),
        crop_size: Tuple[int, int] = (512, 512),
        ignore_value: int = 0,
    ):
        self.ratio_range = ratio_range
        self.crop_size = crop_size
        self.ignore_value = ignore_value

    def __call__(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply LSJ: random resize + random crop to both image and label."""
        w, h = image.size
        crop_h, crop_w = self.crop_size

        # Random scale factor
        scale = random.uniform(*self.ratio_range)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image (bilinear) and label (nearest — preserves class indices)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)

        # Random crop (or pad if resized image is smaller than crop)
        if new_h >= crop_h and new_w >= crop_w:
            # Standard random crop
            top = random.randint(0, new_h - crop_h)
            left = random.randint(0, new_w - crop_w)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            label = label.crop((left, top, left + crop_w, top + crop_h))
        else:
            # Pad then crop (image smaller than crop in at least one dimension)
            # Create padded canvas
            pad_h = max(crop_h, new_h)
            pad_w = max(crop_w, new_w)

            # Random offset for placing the image on the canvas
            offset_y = random.randint(0, max(0, pad_h - new_h))
            offset_x = random.randint(0, max(0, pad_w - new_w))

            # Pad image (zeros = black)
            padded_image = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
            padded_image.paste(image, (offset_x, offset_y))

            # Pad label (ignore_value)
            padded_label = Image.new("L", (pad_w, pad_h), self.ignore_value)
            padded_label.paste(label, (offset_x, offset_y))

            # Now crop to crop_size
            top = random.randint(0, max(0, pad_h - crop_h))
            left = random.randint(0, max(0, pad_w - crop_w))
            image = padded_image.crop((left, top, left + crop_w, top + crop_h))
            label = padded_label.crop((left, top, left + crop_w, top + crop_h))

        return image, label

    def __repr__(self):
        return (
            f"LargeScaleJitter(ratio_range={self.ratio_range}, "
            f"crop_size={self.crop_size})"
        )


class RandomHorizontalFlip:
    """
    Random horizontal flip applied to BOTH image AND label.

    Args:
        prob: Probability of flipping
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Randomly flip both image and label horizontally."""
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

    def __repr__(self):
        return f"RandomHorizontalFlip(prob={self.prob})"


class MaterialAugmentation:
    """
    Complete material-segmentation augmentation pipeline.

    Geometric augmentations are applied to BOTH image and label.
    Photometric augmentations are applied to image ONLY.

    Pipeline order:
        1. [Geometric] Large Scale Jitter (resize + crop)
        2. [Geometric] Random Horizontal Flip
        3. [Photometric] Constrained Color Jitter (image only)
        4. [Photometric] Specular Highlight Injection (image only)
        5. [Photometric] Gaussian Noise (image only, SegFormer only)

    Args:
        geometric_transforms: List of (image, label) -> (image, label) transforms
        photometric_transforms: List of image -> image transforms
    """

    def __init__(
        self,
        geometric_transforms: list,
        photometric_transforms: list,
    ):
        self.geometric_transforms = geometric_transforms
        self.photometric_transforms = photometric_transforms

    def __call__(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply augmentation pipeline.

        Args:
            image: RGB PIL Image
            label: Single-channel PIL Image (segmentation mask)

        Returns:
            (augmented_image, augmented_label) — both PIL Images
        """
        # Ensure correct modes
        image = image.convert("RGB")

        # 1. Geometric augmentations — transform BOTH image and label
        for transform in self.geometric_transforms:
            image, label = transform(image, label)

        # 2. Photometric augmentations — transform image ONLY, label untouched
        for transform in self.photometric_transforms:
            image = transform(image)

        return image, label

    def __repr__(self):
        lines = ["MaterialAugmentation("]
        lines.append("  geometric=[")
        for t in self.geometric_transforms:
            lines.append(f"    {t},")
        lines.append("  ],")
        lines.append("  photometric=[")
        for t in self.photometric_transforms:
            lines.append(f"    {t},")
        lines.append("  ]")
        lines.append(")")
        return "\n".join(lines)


def get_material_augmentation(
    model_type: str = "segformer",
    crop_size: Tuple[int, int] = (512, 512),
) -> MaterialAugmentation:
    """
    Factory function to create a MaterialAugmentation configured for the
    given model architecture.

    Args:
        model_type: "segformer" or "mask2former"
        crop_size: Output crop size (height, width). Default (512, 512).
                   Use (640, 640) for SegFormer-B5, or larger if GPU memory permits.

    Returns:
        MaterialAugmentation instance

    Hyperparameters (from research report, user-adjusted):
        SegFormer:
            - LSJ resize: 0.5–2.0 (preserves texture fidelity)
            - Color jitter: hue±0.02, sat 0.8–1.2, contrast 0.7–1.3
            - Gaussian noise: σ∈(5, 25), prob 0.3
            - Specular highlights: prob 0.3

        Mask2Former:
            - LSJ resize: 0.1–2.0 (extreme lower bound for global context)
            - Color jitter: hue±0.02, sat 0.8–1.2, contrast 0.6–1.4
            - NO Gaussian noise (can destabilize mask boundaries)
            - Specular highlights: prob 0.3
    """
    model_type = model_type.lower()

    if model_type == "segformer":
        geometric = [
            LargeScaleJitter(ratio_range=(0.5, 2.0), crop_size=crop_size),
            RandomHorizontalFlip(prob=0.5),
        ]
        photometric = [
            ConstrainedColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.7, 1.3),
                saturation=(0.8, 1.2),
                hue_delta=0.02,
                prob=0.5,
            ),
            SpecularHighlightInjection(prob=0.3),
            GaussianNoise(sigma_range=(5.0, 25.0), prob=0.3),
        ]

    elif model_type == "mask2former":
        geometric = [
            LargeScaleJitter(ratio_range=(0.1, 2.0), crop_size=crop_size),
            RandomHorizontalFlip(prob=0.5),
        ]
        photometric = [
            ConstrainedColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.6, 1.4),
                saturation=(0.8, 1.2),
                hue_delta=0.02,
                prob=0.5,
            ),
            SpecularHighlightInjection(prob=0.3),
            # No GaussianNoise for Mask2Former — can destabilize mask boundaries
        ]

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Use 'segformer' or 'mask2former'."
        )

    aug = MaterialAugmentation(geometric, photometric)
    return aug
