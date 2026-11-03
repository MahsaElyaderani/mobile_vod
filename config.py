from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Backbone, shared Bottleneck-LSTM, and SSD architecture settings."""

    image_size: int = 320
    width_mult: float = 1.0
    mobilenet_output_channels: int = 1280
    lstm_channels: int = 256
    lstm_kernel_size: int = 3
    lstm_forget_bias: float = 1.0
    sequence_length: int = 10
    num_classes: int = 30  # foreground ImageNet-VID classes

    # Detection feature maps in Table 1:
    # Bottleneck-LSTM output, FM1, FM2, FM3, FM4.
    feature_channels: Tuple[int, ...] = (256, 256, 128, 128, 64)

    # The paper equations use ReLU. The later released TensorFlow code uses
    # ReLU6 when constructing the cell; choose 'relu' for paper fidelity.
    lstm_activation: str = "relu"

    backbone: str = "timm_mobilenet_v2"

    @property
    def uses_timm(self):
        return self.backbone in ('timm_mobilenet_v2', 'tinynet_d', 'efficientnet_b0')

    @property
    def input_interpolation(self):
        return "bicubic" if self.uses_timm else "bilinear"

    def __post_init__(self) -> None:
        if self.backbone not in ("timm_mobilenet_v2", "tinynet_d", "efficientnet_b0"):
            raise ValueError("Unknown backbone family.")
        widths = (0.5, 1.0, 1.1, 1.2) if self.backbone == "timm_mobilenet_v2" else (1.0,)
        if self.width_mult not in widths:
            raise ValueError(f"Supported {self.backbone} widths are {widths}.")
        if self.image_size not in (160, 320):
            raise ValueError("Inputs must be 320 (big) or 160 (small).")
        expected = 1280
        if self.mobilenet_output_channels != expected:
            raise ValueError(f"Backbone output channels must be {expected} for this width.")
        if self.lstm_channels != 256:
            raise ValueError("Paper alpha=1 Bottleneck-LSTM output/state is 256 channels.")
        if len(self.feature_channels) != 5:
            raise ValueError("The paper architecture uses five detection feature levels.")
        if self.feature_channels[0] != self.lstm_channels:
            raise ValueError("First detection feature is the Bottleneck-LSTM output.")
        if self.lstm_activation not in {"relu", "relu6"}:
            raise ValueError("lstm_activation must be 'relu' or 'relu6'.")


@dataclass(frozen=True)
class SSDConfig:
    """SSD anchor/coder settings.

    The anchor settings and box-coder scales are taken from the authors'
    released ImageNet-VID TensorFlow configuration. TensorFlow's SSD anchor
    generator also adds an interpolated-scale square anchor on levels > 0 and,
    by default, uses only three boxes on the finest level.
    """

    num_levels: int = 5
    min_scale: float = 0.20
    max_scale: float = 0.95
    aspect_ratios: Tuple[float, ...] = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
    interpolated_scale_aspect_ratio: float = 1.0
    reduce_boxes_in_lowest_layer: bool = True

    match_iou_threshold: float = 0.50
    nms_iou_threshold: float = 0.50

    # TensorFlow FasterRcnnBoxCoder settings in the released config.
    y_scale: float = 10.0
    x_scale: float = 10.0
    height_scale: float = 5.0
    width_scale: float = 5.0

    # The predictor outputs one background logit plus the 30 foreground logits.
    include_background_class: bool = True

    @property
    def num_anchors_per_location(self) -> Tuple[int, ...]:
        # TensorFlow SSD default: finest level has 3 boxes when
        # reduce_boxes_in_lowest_layer=True. Other levels contain one anchor
        # per configured aspect ratio plus the interpolated-scale square box.
        first = 3 if self.reduce_boxes_in_lowest_layer else (
            len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        )
        other = len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        return (first,) + (other,) * (self.num_levels - 1)


@dataclass(frozen=True)
class TrainConfig:
    """Paper training settings for the alpha=1 experiment."""

    learning_rate: float = 0.003
    optimizer: str = "rmsprop"
    sequence_length: int = 10
    max_negatives_per_positive: int = 10
    negative_loss_weight: float = 0.3

    # RMSProp details from the authors' released configuration, not the paper.
    rmsprop_momentum: float = 0.9
    rmsprop_alpha: float = 0.9
    rmsprop_eps: float = 1.0

    def __post_init__(self) -> None:
        if self.optimizer.lower() != "rmsprop":
            raise ValueError("The paper's alpha=1 experiment uses RMSProp.")
        if self.sequence_length != 10:
            raise ValueError("The paper trains by unrolling 10-frame sequences.")


def model_config_dict(config):
    """Serialize the model configuration for checkpoints."""
    return asdict(config)


def checkpoint_model_config(values):
    """Do not interpret older unnamed V1 configurations as the new V2 default."""
    if values.get('backbone', 'mobilenet_v1') == 'mobilenet_v1':
        raise ValueError('MobileNetV1 checkpoints require the archived project version.')
    return ModelConfig(**values)
