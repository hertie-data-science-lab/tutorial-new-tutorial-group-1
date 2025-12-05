"""Tutorial Group 1: Few-Shot Learning for Rooftop Detection in Satellite Imagery."""

from typing import Any

# Relative import from the same package
from .data import create_dataloader, load_dataset, preprocess_image

NAME = "few_shot_utils"

# What this package exposes at top level
__all__ = [
    "NAME",
    "FewShotSegmentationModel",
    "create_dataloader",
    "evaluate_model",
    "get_backbone",
    "load_dataset",
    "preprocess_image",
    "train_episode",
    "train_model",
    "calculate_iou",
    "calculate_dice",
    "print_evaluation_results",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """
    Lazy import for torch-dependent modules.

    This is called when an attribute is accessed on the package that
    isn't already defined above.
    """
    if name in ("FewShotSegmentationModel", "get_backbone", "create_model"):
        from . import models  # relative import

        return getattr(models, name)

    if name in ("train_episode", "train_model"):
        from . import train  # relative import

        return getattr(train, name)

    if name in (
        "calculate_iou",
        "calculate_dice",
        "evaluate_model",
        "print_evaluation_results",
    ):
        from . import evaluate  # relative import

        return getattr(evaluate, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
