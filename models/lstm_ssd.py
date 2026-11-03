from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor

try:
    from config import ModelConfig, SSDConfig
except ImportError:  # package-style import
    from ..config import ModelConfig, SSDConfig

from .backbones import build_backbone
from .bottleneck_lstm import BottleneckLSTMCell, State
from .ssd_head import SSDFeatureMaps, SSDBoxPredictor
from .anchors import SSDAnchorGenerator


class LSTMSSD(nn.Module):
    """Single-Bottleneck-LSTM detector from the paper's Table-1 design.

    One frame:
      RGB -> timm backbone -> Bottleneck-LSTM -> 5 SSD feature levels
          -> class logits and box offsets.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        ssd_cfg: SSDConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg or ModelConfig()
        self.ssd_cfg = ssd_cfg or SSDConfig()

        self.backbone = build_backbone(self.model_cfg)
        self.lstm = BottleneckLSTMCell(
            input_channels=self.model_cfg.mobilenet_output_channels,
            hidden_channels=self.model_cfg.lstm_channels,
            kernel_size=self.model_cfg.lstm_kernel_size,
            forget_bias=self.model_cfg.lstm_forget_bias,
            activation=self.model_cfg.lstm_activation,
        )
        self.feature_maps = SSDFeatureMaps()
        self.predictor = SSDBoxPredictor(
            feature_channels=self.model_cfg.feature_channels,
            num_classes=self.model_cfg.num_classes,
            anchors_per_location=self.ssd_cfg.num_anchors_per_location,
            include_background_class=self.ssd_cfg.include_background_class,
        )
        self.anchor_generator = SSDAnchorGenerator(
            num_levels=self.ssd_cfg.num_levels,
            min_scale=self.ssd_cfg.min_scale,
            max_scale=self.ssd_cfg.max_scale,
            aspect_ratios=self.ssd_cfg.aspect_ratios,
            interpolated_scale_aspect_ratio=self.ssd_cfg.interpolated_scale_aspect_ratio,
            reduce_boxes_in_lowest_layer=self.ssd_cfg.reduce_boxes_in_lowest_layer,
        )

    def _validate_frame(self, frame: Tensor) -> None:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(f"Expected frame [B,3,H,W], got {tuple(frame.shape)}")
        if frame.shape[-2:] != (self.model_cfg.image_size, self.model_cfg.image_size):
            raise ValueError(
                f"Expected {self.model_cfg.image_size}x{self.model_cfg.image_size} input; "
                f"got {tuple(frame.shape[-2:])}"
            )

    def forward_step(
        self,
        frame: Tensor,
        state: State | None = None,
        return_anchors: bool = True,
    ) -> Tuple[Dict[str, Tensor | List[Tensor]], State]:
        """Process one frame while carrying recurrent state online."""
        self._validate_frame(frame)
        backbone_feature = self.backbone(frame)
        return self._forward_from_backbone(
            backbone_feature, state, return_anchors, anchor_dtype=frame.dtype
        )

    def _forward_from_backbone(
        self,
        backbone_feature: Tensor,
        state: State | None,
        return_anchors: bool,
        *,
        anchor_dtype: torch.dtype,
    ) -> Tuple[Dict[str, Tensor | List[Tensor]], State]:
        if self.model_cfg.uses_timm and backbone_feature.shape[1:] != (self.model_cfg.mobilenet_output_channels, self.model_cfg.image_size // 32, self.model_cfg.image_size // 32):
            raise ValueError("timm ConvLSTM feature shape does not match the configured input size.")
        lstm_feature, new_state = self.lstm(backbone_feature, state)
        features = self.feature_maps(lstm_feature)
        class_logits, box_regression = self.predictor(features)

        output: Dict[str, Tensor | List[Tensor]] = {
            "class_logits": class_logits,
            "box_regression": box_regression,
            "features": features,
        }

        if return_anchors:
            shapes = [(f.shape[-2], f.shape[-1]) for f in features]
            anchors = self.anchor_generator.generate(
                shapes, device=backbone_feature.device, dtype=anchor_dtype
            )
            if anchors.shape[0] != class_logits.shape[1]:
                raise RuntimeError(
                    "Anchor/prediction count mismatch: "
                    f"anchors={anchors.shape[0]}, predictions={class_logits.shape[1]}"
                )
            output["anchors"] = anchors

        return output, new_state

    def forward_clip(
        self,
        frames: Tensor,
        state: State | None = None,
        return_anchors: bool = True,
        frozen_backbone_batch_size: int = 32,
    ) -> Tuple[Dict[str, Tensor], State]:
        """Process a clip sequentially and carry (c,h) across all frames.

        Args:
          frames: [B,T,3,H,W]. T may be any positive value at inference;
                  paper training uses T=10.
          state: optional initial (c,h). Use None to start from zeros.

        Returns:
          dict with class_logits [B,T,A,C], box_regression [B,T,A,4], and,
          when requested, anchors [A,4].
          final recurrent state after the last frame.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] <= 0:
            raise ValueError("Clip must contain at least one frame.")
        if frozen_backbone_batch_size < 0:
            raise ValueError("frozen_backbone_batch_size must be non-negative.")
        self._validate_frame(frames[:, 0])

        backbone_features: Tensor | None = None
        backbone_modules = tuple(self.backbone.modules())
        can_batch_backbone = (
            frozen_backbone_batch_size > 0
            and not frames.requires_grad
            and not any(module.training for module in backbone_modules)
            and all(not parameter.requires_grad for parameter in self.backbone.parameters())
            and all(
                not isinstance(module, nn.modules.batchnorm._BatchNorm)
                or module.track_running_stats
                for module in backbone_modules
            )
        )
        if can_batch_backbone:
            batch, time = frames.shape[:2]
            flat_frames = frames.reshape(batch * time, *frames.shape[2:])
            with torch.no_grad():
                chunks = [
                    self.backbone(chunk)
                    for chunk in flat_frames.split(frozen_backbone_batch_size)
                ]
                flat_features = torch.cat(chunks, dim=0)
            backbone_features = flat_features.reshape(
                batch, time, *flat_features.shape[1:]
            )

        cls_steps: List[Tensor] = []
        box_steps: List[Tensor] = []
        anchors: Tensor | None = None
        current_state = state

        for t in range(frames.shape[1]):
            if backbone_features is None:
                out, current_state = self.forward_step(
                    frames[:, t], current_state, return_anchors=return_anchors
                )
            else:
                out, current_state = self._forward_from_backbone(
                    backbone_features[:, t], current_state, return_anchors,
                    anchor_dtype=frames.dtype,
                )
            cls_steps.append(out["class_logits"])  # type: ignore[arg-type]
            box_steps.append(out["box_regression"])  # type: ignore[arg-type]
            if return_anchors and anchors is None:
                anchors = out["anchors"]  # type: ignore[assignment]

        result: Dict[str, Tensor] = {
            "class_logits": torch.stack(cls_steps, dim=1),
            "box_regression": torch.stack(box_steps, dim=1),
        }
        if anchors is not None:
            result["anchors"] = anchors
        assert current_state is not None
        return result, current_state

    def forward(self, frames: Tensor) -> Dict[str, Tensor]:
        """Convenience forward: accepts a single frame or a full clip."""
        if frames.ndim == 4:
            out, _ = self.forward_step(frames, state=None, return_anchors=True)
            return {
                "class_logits": out["class_logits"],  # type: ignore[dict-item]
                "box_regression": out["box_regression"],  # type: ignore[dict-item]
                "anchors": out["anchors"],  # type: ignore[dict-item]
            }
        if frames.ndim == 5:
            out, _ = self.forward_clip(frames, state=None, return_anchors=True)
            return out
        raise ValueError("LSTMSSD.forward expects [B,3,H,W] or [B,T,3,H,W].")
