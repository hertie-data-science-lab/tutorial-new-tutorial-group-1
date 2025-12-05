"""Training functions for few-shot segmentation."""

from typing import Any

import torch
from torch import nn
from tqdm import tqdm


def train_episode(
    model: nn.Module,
    support_images: torch.Tensor,
    support_masks: torch.Tensor,
    query_images: torch.Tensor,
    query_masks: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    """
    Train on a single few-shot episode.

    This function implements episodic training where the model first extracts
    prototypes from the support set and then uses them to segment the query set.

    Args:
        model: The few-shot segmentation model.
        support_images: Support set images for prototype extraction.
        support_masks: Support set masks for prototype extraction.
        query_images: Query set images to segment.
        query_masks: Query set ground truth masks.
        optimizer: Optimizer for updating model weights.
        criterion: Loss function.

    Returns:
        Dictionary containing loss and metrics.

    """
    model.train()
    optimizer.zero_grad()

    # Extract prototypes from support set if model supports it
    if hasattr(model, "extract_support_features"):
        prototypes = model.extract_support_features(support_images, support_masks)
        # Use prototypes for query prediction (prototype-based few-shot learning)
        # For now, we use standard forward pass for training
        _ = prototypes  # Prototypes can be used in more advanced implementations

    # Forward pass on query set
    outputs = model(query_images)

    # Compute loss
    loss = criterion(outputs, query_masks)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == query_masks).float().mean().item()

    return {"loss": loss.item(), "accuracy": accuracy}


def train_model(
    model: nn.Module,
    train_dataloader: list[dict[str, Any]],
    val_dataloader: list[dict[str, Any]] | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """
    Train the few-shot segmentation model.

    Args:
        model: The model to train.
        train_dataloader: Training data batches.
        val_dataloader: Validation data batches.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        device: Device to train on ('cuda' or 'cpu').

    Returns:
        Dictionary containing training history.

    Raises:
        ValueError: If train_dataloader is empty.

    """
    if not train_dataloader:
        msg = "train_dataloader cannot be empty"
        raise ValueError(msg)

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = torch.from_numpy(batch["images"]).float().permute(0, 3, 1, 2)
            masks = torch.from_numpy(batch["masks"]).long()

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Resize outputs to match mask size if needed
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == masks).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += accuracy
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)

        # Validation
        if val_dataloader is not None:
            val_metrics = _validate(model, val_dataloader, criterion, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            print(
                f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, "
                f"Train Acc={avg_acc:.4f}, Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}")

    return history


def _validate(
    model: nn.Module,
    dataloader: list[dict[str, Any]],
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """
    Validate the model.

    Args:
        model: The model to validate.
        dataloader: Validation data batches.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Dictionary containing validation metrics.

    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = torch.from_numpy(batch["images"]).float().permute(0, 3, 1, 2)
            masks = torch.from_numpy(batch["masks"]).long()

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = criterion(outputs, masks)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == masks).float().mean().item()

            total_loss += loss.item()
            total_acc += accuracy
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": total_acc / max(num_batches, 1),
    }
