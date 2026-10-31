from __future__ import annotations

from typing import Sequence, Tuple, List

import torch
from torch import nn, Tensor
from .padding import TFSameConv2d


class ConvBNReLU6(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_eps: float = 1e-3,
        bn_decay: float = 0.9997,
    ) -> None:
        super().__init__(
            TFSameConv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=1.0 - bn_decay),
            nn.ReLU6(inplace=True),
        )


class SSDExtraBlock(nn.Module):
    """One paper Table-1 SSD feature-map block.

    Pattern: 1x1 reduction -> 3x3 depthwise stride 2 -> 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        bn_eps: float = 1e-3,
        bn_decay: float = 0.9997,
    ) -> None:
        super().__init__()
        self.reduce = ConvBNReLU6(
            in_channels, bottleneck_channels, 1,
            bn_eps=bn_eps, bn_decay=bn_decay
        )
        self.depthwise = ConvBNReLU6(
            bottleneck_channels, bottleneck_channels, 3, stride=2, padding=1,
            groups=bottleneck_channels, bn_eps=bn_eps, bn_decay=bn_decay
        )
        self.project = ConvBNReLU6(
            bottleneck_channels, out_channels, 1,
            bn_eps=bn_eps, bn_decay=bn_decay
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.depthwise(self.reduce(x)))


class SSDFeatureMaps(nn.Module):
    """Build the four post-LSTM SSD feature maps from Table 1.

    Input level: Bottleneck-LSTM output [B,256,10,10] for 320x320 input.
    Additional levels:
      FM1: 256 -> 128 -> 256
      FM2: 256 ->  64 -> 128
      FM3: 128 ->  64 -> 128
      FM4: 128 ->  32 ->  64
    """

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.fm1 = SSDExtraBlock(input_channels, 128, 256)
        self.fm2 = SSDExtraBlock(256, 64, 128)
        self.fm3 = SSDExtraBlock(128, 64, 128)
        self.fm4 = SSDExtraBlock(128, 32, 64)

    def forward(self, lstm_feature: Tensor) -> List[Tensor]:
        if lstm_feature.ndim != 4 or lstm_feature.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected feature [B,{self.input_channels},H,W], got {tuple(lstm_feature.shape)}"
            )
        fm1 = self.fm1(lstm_feature)
        fm2 = self.fm2(fm1)
        fm3 = self.fm3(fm2)
        fm4 = self.fm4(fm3)
        return [lstm_feature, fm1, fm2, fm3, fm4]


class _SeparablePredictionConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 linear prediction convolution.

    This matches the released ImageNet-VID config's `kernel_size: 3` and
    `use_depthwise: true` setting for the box predictor, without adding hidden
    predictor layers (the same config has min_depth=max_depth=0).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SSDBoxPredictor(nn.Module):
    """Per-level depthwise-separable SSD class/box prediction heads.

    The paper states predictions are produced by an additional convolution on
    the Bottleneck-LSTM and each SSD feature map. The authors' released
    ImageNet-VID config specifies a 3x3 depthwise predictor (`use_depthwise:
    true`) with no inserted hidden feature depth (`min_depth=max_depth=0`).
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        num_classes: int,
        anchors_per_location: Sequence[int],
        include_background_class: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_channels) != len(anchors_per_location):
            raise ValueError("feature_channels and anchors_per_location must align.")
        self.feature_channels = tuple(feature_channels)
        self.num_foreground_classes = int(num_classes)
        self.num_classes_with_background = num_classes + int(include_background_class)
        self.anchors_per_location = tuple(int(v) for v in anchors_per_location)

        self.class_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        for channels, anchors in zip(self.feature_channels, self.anchors_per_location):
            self.class_heads.append(
                _SeparablePredictionConv(
                    channels, anchors * self.num_classes_with_background
                )
            )
            self.box_heads.append(
                _SeparablePredictionConv(channels, anchors * 4)
            )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in list(self.class_heads) + list(self.box_heads):
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    nn.init.normal_(submodule.weight, mean=0.0, std=0.03)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        if len(features) != len(self.feature_channels):
            raise ValueError(
                f"Expected {len(self.feature_channels)} feature maps, got {len(features)}"
            )
        cls_all: List[Tensor] = []
        box_all: List[Tensor] = []

        for level, (feature, expected_c, anchors, cls_head, box_head) in enumerate(
            zip(features, self.feature_channels, self.anchors_per_location,
                self.class_heads, self.box_heads)
        ):
            if feature.ndim != 4 or feature.shape[1] != expected_c:
                raise ValueError(
                    f"Level {level}: expected [B,{expected_c},H,W], got {tuple(feature.shape)}"
                )
            batch, _, height, width = feature.shape
            cls = cls_head(feature)
            box = box_head(feature)

            cls = cls.view(
                batch, anchors, self.num_classes_with_background, height, width
            ).permute(0, 3, 4, 1, 2).contiguous()
            box = box.view(batch, anchors, 4, height, width).permute(
                0, 3, 4, 1, 2
            ).contiguous()

            cls_all.append(cls.view(batch, -1, self.num_classes_with_background))
            box_all.append(box.view(batch, -1, 4))

        return torch.cat(cls_all, dim=1), torch.cat(box_all, dim=1)
