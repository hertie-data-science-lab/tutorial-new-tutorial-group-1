"""Data loading and preprocessing utilities for satellite imagery."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_dataset(
    data_path: str | Path,
    split: str = "train",
    category: str = "all",
) -> dict[str, Any]:
    """
    Load Geneva rooftop satellite imagery dataset.

    Args:
        data_path: Path to the root dataset directory (e.g. snapshot_download path).
        split: One of 'train', 'val', or 'test'.
        category: One of 'all', 'industrial', or 'residencial'.

    Returns:
        Dictionary containing:
            - "images": list of HxWxC numpy arrays (RGB)
            - "masks":  list of HxW numpy arrays (label masks)
            - "filenames": list of image file names (optional convenience)

    """
    data_path = Path(data_path)

    # Geneva dataset layout:
    # <root>/<split>/images/<category>/*.png
    # <root>/<split>/labels/<category>/*_label.png
    image_dir = data_path / split / "images" / category
    mask_dir = data_path / split / "labels" / category

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    filenames: list[str] = []

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {mask_dir}")

    for img_path in sorted(image_dir.glob("*.png")):
        # Build corresponding label filename: add "_label" before .png
        label_name = img_path.stem + "_label.png"
        mask_path = mask_dir / label_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        images.append(np.array(image))
        masks.append(np.array(mask))
        filenames.append(img_path.name)

    return {"images": images, "masks": masks, "filenames": filenames}


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess satellite image for model input.

    Args:
        image: Input image as numpy array.
        target_size: Target size for resizing (height, width).
        normalize: Whether to normalize the image values.

    Returns:
        Preprocessed image.

    """
    img = Image.fromarray(image)
    img = img.resize(target_size[::-1])  # PIL uses (width, height)
    result = np.array(img)

    return result


def create_dataloader(
    dataset: dict[str, Any],
    batch_size: int = 4,
    shuffle: bool = True,
) -> list[dict[str, np.ndarray]]:
    """
    Create data batches from dataset.

    Args:
        dataset: Dataset dictionary with images and masks.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.

    Returns:
        List of batch dictionaries.

    """
    images = dataset["images"]
    masks = dataset["masks"]

    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = {
            "images": np.array([images[j] for j in batch_indices]),
            "masks": np.array([masks[j] for j in batch_indices]),
        }
        batches.append(batch)

    return batches
