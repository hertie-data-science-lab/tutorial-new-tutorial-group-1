"""Model architectures for few-shot segmentation."""

from typing import Any

import torch
from torch import nn


def get_backbone(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    Get a backbone network for feature extraction.

    Args:
        name: Name of the backbone architecture.
        pretrained: Whether to use pretrained weights.

    Returns:
        Backbone neural network module.

    """
    from torchvision import models

    if name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove the final fully connected layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
    elif name == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
    else:
        msg = f"Unknown backbone: {name}"
        raise ValueError(msg)

    return backbone


class FewShotSegmentationModel(nn.Module):
    """Few-shot segmentation model for rooftop detection."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the few-shot segmentation model.

        Args:
            backbone_name: Name of the backbone network.
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained backbone.

        """
        super().__init__()
        self.backbone = get_backbone(backbone_name, pretrained)
        self.num_classes = num_classes

        # Feature dimension from backbone (ResNet variants)
        feature_dim_map = {
            "resnet18": 512,
            "resnet34": 512,
        }
        feature_dim = feature_dim_map.get(backbone_name, 512)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output segmentation tensor of shape (B, num_classes, H, W).

        """
        features = self.backbone(x)
        output = self.decoder(features)
        return output

    def extract_support_features(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from support set for few-shot learning.

        Args:
            support_images: Support set images.
            support_masks: Support set masks.

        Returns:
            Prototype features for each class.

        """
        features = self.backbone(support_images)
        # Compute class prototypes by masked average pooling
        prototypes = []
        for c in range(self.num_classes):
            mask = (support_masks == c).float().unsqueeze(1)
            # Resize mask to feature size
            mask = nn.functional.interpolate(mask, size=features.shape[-2:], mode="nearest")
            masked_features = features * mask
            prototype = masked_features.sum(dim=(0, 2, 3)) / (mask.sum() + 1e-8)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def predict_with_prototypes(
        self,
        query_images: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict segmentation using class prototypes.

        Args:
            query_images: Query images to segment.
            prototypes: Class prototype features.

        Returns:
            Predicted segmentation masks.

        """
        features = self.backbone(query_images)
        # Compute distance to prototypes for each spatial location
        distances = []
        for prototype in prototypes:
            dist = ((features - prototype.view(1, -1, 1, 1)) ** 2).sum(dim=1)
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        # Return class with minimum distance
        predictions = distances.argmin(dim=1)
        # Resize to input size
        predictions = nn.functional.interpolate(
            predictions.unsqueeze(1).float(),
            size=query_images.shape[-2:],
            mode="nearest",
        ).squeeze(1)
        return predictions.long()


def create_model(config: dict[str, Any] | None = None) -> FewShotSegmentationModel:
    """
    Create a few-shot segmentation model from configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        Initialized model.

    """
    if config is None:
        config = {}

    return FewShotSegmentationModel(
        backbone_name=config.get("backbone", "resnet18"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
    )
