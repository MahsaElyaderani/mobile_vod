from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence, Tuple, List

import torch
from torch import Tensor


@dataclass(frozen=True)
class BoxCoderScales:
    y: float = 10.0
    x: float = 10.0
    h: float = 5.0
    w: float = 5.0


def _box_iou_xyxy(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("boxes1 must have shape [N,4].")
    if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
        raise ValueError("boxes2 must have shape [M,4].")

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


class SSDAnchorGenerator:
    """PyTorch implementation of TensorFlow's SSD anchor construction.

    Anchors are returned in normalized xyxy format and in the same logical
    ordering used by SSD predictions: level, y, x, anchor-spec.
    """

    def __init__(
        self,
        num_levels: int = 5,
        min_scale: float = 0.2,
        max_scale: float = 0.95,
        aspect_ratios: Sequence[float] = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
        interpolated_scale_aspect_ratio: float = 1.0,
        reduce_boxes_in_lowest_layer: bool = True,
        clip: bool = False,
    ) -> None:
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        self.num_levels = int(num_levels)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.aspect_ratios = tuple(float(v) for v in aspect_ratios)
        self.interpolated_scale_aspect_ratio = float(interpolated_scale_aspect_ratio)
        self.reduce_boxes_in_lowest_layer = bool(reduce_boxes_in_lowest_layer)
        self.clip = bool(clip)

    @property
    def scales(self) -> Tuple[float, ...]:
        if self.num_levels == 1:
            return (self.min_scale, 1.0)
        values = [
            self.min_scale + (self.max_scale - self.min_scale) * i / (self.num_levels - 1)
            for i in range(self.num_levels)
        ]
        return tuple(values + [1.0])

    @property
    def num_anchors_per_location(self) -> Tuple[int, ...]:
        first = 3 if self.reduce_boxes_in_lowest_layer else (
            len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        )
        other = len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        return (first,) + (other,) * (self.num_levels - 1)

    def _specs_for_level(self, level: int) -> List[Tuple[float, float]]:
        scale = self.scales[level]
        next_scale = self.scales[level + 1]
        if level == 0 and self.reduce_boxes_in_lowest_layer:
            return [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]

        specs = [(scale, ratio) for ratio in self.aspect_ratios]
        if self.interpolated_scale_aspect_ratio > 0:
            specs.append((
                sqrt(scale * next_scale),
                self.interpolated_scale_aspect_ratio,
            ))
        return specs

    def generate(
        self,
        feature_map_shapes: Sequence[Tuple[int, int]],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if len(feature_map_shapes) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature-map shapes, got {len(feature_map_shapes)}"
            )

        all_anchors: List[Tensor] = []
        for level, (height, width) in enumerate(feature_map_shapes):
            if height <= 0 or width <= 0:
                raise ValueError("Feature-map dimensions must be positive.")

            specs = self._specs_for_level(level)
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            centers = torch.stack([xx, yy], dim=-1)  # [H,W,2]

            per_spec: List[Tensor] = []
            for scale, ratio in specs:
                anchor_h = scale / sqrt(ratio)
                anchor_w = scale * sqrt(ratio)
                half = torch.tensor(
                    [anchor_w / 2.0, anchor_h / 2.0], device=device, dtype=dtype
                )
                xy_min = centers - half
                xy_max = centers + half
                per_spec.append(torch.cat([xy_min, xy_max], dim=-1))

            # [H,W,A,4], with A varying fastest to match predictor flattening.
            level_anchors = torch.stack(per_spec, dim=2).reshape(-1, 4)
            if self.clip:
                level_anchors = level_anchors.clamp(0.0, 1.0)
            all_anchors.append(level_anchors)

        return torch.cat(all_anchors, dim=0)


def match_anchors(
    anchors: Tensor,
    gt_boxes: Tensor,
    iou_threshold: float = 0.5,
    force_match_for_each_gt: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Match normalized xyxy ground-truth boxes to anchors.

    Returns:
      matched_gt_indices: [A] index of best ground-truth box for every anchor.
      positive_mask:      [A] boolean positive-anchor mask.
      matched_iou:        [A] IoU with each anchor's assigned GT box.

    This mirrors the released config's 0.5 matched/unmatched threshold and
    force_match_for_each_row=True behavior at a practical level.
    """
    if anchors.ndim != 2 or anchors.shape[-1] != 4:
        raise ValueError("anchors must have shape [A,4].")
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must have shape [G,4].")
    if gt_boxes.shape[0] == 0:
        count = anchors.shape[0]
        return (
            torch.full((count,), -1, dtype=torch.long, device=anchors.device),
            torch.zeros(count, dtype=torch.bool, device=anchors.device),
            torch.zeros(count, dtype=anchors.dtype, device=anchors.device),
        )

    iou = _box_iou_xyxy(anchors, gt_boxes)
    matched_iou, matched_gt = iou.max(dim=1)
    positive = matched_iou >= iou_threshold

    if force_match_for_each_gt:
        # Ensure each GT owns at least one anchor. This is deterministic except
        # in exact-IoU ties, where PyTorch argmax chooses the first occurrence.
        best_anchor_for_gt = iou.argmax(dim=0)
        gt_indices = torch.arange(gt_boxes.shape[0], device=anchors.device)
        matched_gt[best_anchor_for_gt] = gt_indices
        matched_iou[best_anchor_for_gt] = iou[best_anchor_for_gt, gt_indices]
        positive[best_anchor_for_gt] = True

    return matched_gt, positive, matched_iou


def _xyxy_to_center(boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    xmin, ymin, xmax, ymax = boxes.unbind(dim=-1)
    w = (xmax - xmin).clamp(min=1e-8)
    h = (ymax - ymin).clamp(min=1e-8)
    x = xmin + 0.5 * w
    y = ymin + 0.5 * h
    return x, y, w, h


def encode_boxes(
    gt_boxes: Tensor,
    anchors: Tensor,
    scales: BoxCoderScales = BoxCoderScales(),
) -> Tensor:
    """Encode xyxy boxes using TensorFlow FasterRcnnBoxCoder conventions.

    Output order is [ty, tx, th, tw].
    """
    if gt_boxes.shape != anchors.shape or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes and anchors must both have shape [N,4].")
    gx, gy, gw, gh = _xyxy_to_center(gt_boxes)
    ax, ay, aw, ah = _xyxy_to_center(anchors)

    ty = (gy - ay) / ah * scales.y
    tx = (gx - ax) / aw * scales.x
    th = torch.log(gh / ah) * scales.h
    tw = torch.log(gw / aw) * scales.w
    return torch.stack([ty, tx, th, tw], dim=-1)


def decode_boxes(
    encoded: Tensor,
    anchors: Tensor,
    scales: BoxCoderScales = BoxCoderScales(),
    clip: bool = False,
) -> Tensor:
    """Inverse of encode_boxes; returns normalized xyxy boxes."""
    if encoded.shape != anchors.shape or encoded.shape[-1] != 4:
        raise ValueError("encoded and anchors must both have shape [N,4].")

    ax, ay, aw, ah = _xyxy_to_center(anchors)
    ty, tx, th, tw = encoded.unbind(dim=-1)

    y = ty / scales.y * ah + ay
    x = tx / scales.x * aw + ax
    h = torch.exp(th / scales.h) * ah
    w = torch.exp(tw / scales.w) * aw

    boxes = torch.stack([x - 0.5 * w, y - 0.5 * h,
                         x + 0.5 * w, y + 0.5 * h], dim=-1)
    return boxes.clamp(0.0, 1.0) if clip else boxes
