from __future__ import annotations

import math

import torch
from torch import nn
from torch import Tensor
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
        # TensorFlow moving-average decay d corresponds approximately to
        # PyTorch BatchNorm momentum (1-d).
        bn_momentum = 1.0 - bn_decay
        super().__init__(
            TFSameConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """MobileNetV1 depthwise 3x3 + pointwise 1x1 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bn_eps: float = 1e-3,
        bn_decay: float = 0.9997,
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU6(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bn_eps=bn_eps,
            bn_decay=bn_decay,
        )
        self.pointwise = ConvBNReLU6(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=bn_eps,
            bn_decay=bn_decay,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class MobileNetV1Backbone(nn.Module):
    """MobileNetV1 through Conv13_pointwise, with output stride 32.

    For a 320x320 input, output shape is [B, 1024, 10, 10].
    Layer naming follows Table 1 in Liu & Zhu (CVPR 2018):
      Conv1, Conv2, ..., Conv13.
    Conv8-Conv12 are five repeated 512->512 depthwise-separable blocks.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        bn_eps: float = 1e-3,
        bn_decay: float = 0.9997,
    ) -> None:
        super().__init__()
        if width_mult not in (0.35, 1.0):
            raise ValueError("Supported standalone widths are 0.35 and 1.0.")
        c32, c64, c128, c256, c512, c1024 = [
            max(8, int(c * width_mult)) for c in (32, 64, 128, 256, 512, 1024)
        ]

        self.conv1 = ConvBNReLU6(3, c32, 3, stride=2, padding=1,
                                 bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv2 = DepthwiseSeparableConv(c32, c64, stride=1,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv3 = DepthwiseSeparableConv(c64, c128, stride=2,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv4 = DepthwiseSeparableConv(c128, c128, stride=1,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv5 = DepthwiseSeparableConv(c128, c256, stride=2,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv6 = DepthwiseSeparableConv(c256, c256, stride=1,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv7 = DepthwiseSeparableConv(c256, c512, stride=2,
                                            bn_eps=bn_eps, bn_decay=bn_decay)
        self.conv8_12 = nn.Sequential(*[
            DepthwiseSeparableConv(c512, c512, stride=1,
                                   bn_eps=bn_eps, bn_decay=bn_decay)
            for _ in range(5)
        ])
        self.conv13 = DepthwiseSeparableConv(c512, c1024, stride=2,
                                             bn_eps=bn_eps, bn_decay=bn_decay)

        self.out_channels = c1024
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8_12(x)
        return self.conv13(x)


class FastMobileNetV1(nn.Module):
    """The native truncated V1 topology, with TF-Slim widths and output stride 16."""

    def __init__(self, width_mult=0.35, min_depth=8):
        super().__init__()
        if not math.isfinite(width_mult) or not 0 < width_mult <= 1 or min_depth < 1:
            raise ValueError("Invalid MobileNetV1 width.")
        depth = lambda value: max(int(value * width_mult), min_depth)
        c32, c64, c128, c256, c512, c1024 = [depth(x) for x in (32, 64, 128, 256, 512, 1024)]
        self.width_mult, self.min_depth = width_mult, min_depth
        self.conv1 = ConvBNReLU6(3, c32, 3, stride=2, padding=1)
        self.conv2 = DepthwiseSeparableConv(c32, c64)
        self.conv3 = DepthwiseSeparableConv(c64, c128, stride=2)
        self.conv4 = DepthwiseSeparableConv(c128, c128)
        self.conv5 = DepthwiseSeparableConv(c128, c256, stride=2)
        self.conv6 = DepthwiseSeparableConv(c256, c256)
        self.conv7 = DepthwiseSeparableConv(c256, c512, stride=2)
        self.conv8_12 = nn.Sequential(*[DepthwiseSeparableConv(c512, c512) for _ in range(5)])
        self.conv13 = DepthwiseSeparableConv(c512, c1024, stride=1)
        self.out_channels, self.output_stride = c1024, 16
        MobileNetV1Backbone._initialize_weights(self)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Fast backbone requires [B,3,H,W].")
        for module in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
                       self.conv6, self.conv7, self.conv8_12, self.conv13):
            x = module(x)
        return x


def initialize_by_channel_slicing(fast, big):
    """Initialization only; this is not a pretrained narrow-model checkpoint."""
    source, target = big.state_dict(), fast.state_dict()
    copied = {}
    if set(source) != set(target):
        raise ValueError("Channel slicing requires the native backbone's identical layer names.")
    for key, value in target.items():
        original = source[key]
        if original.ndim != value.ndim or any(a < b for a, b in zip(original.shape, value.shape)):
            raise ValueError(f"Cannot initialize {key} from the big backbone.")
        slices = tuple(slice(0, size) for size in value.shape)
        copied[key] = original[slices].clone() if value.ndim else original.clone()
    fast.load_state_dict(copied, strict=True)
    return {"method": "channel_slice_big", "copied_tensors": len(copied),
            "is_pretrained_fast_detector": False}
