from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from models.anchors import BoxCoderScales, encode_boxes, match_anchors
except ImportError:
    from ..models.anchors import BoxCoderScales, encode_boxes, match_anchors


def hard_negative_mining(
    classification_loss_per_anchor: Tensor,
    positive_mask: Tensor,
    max_negatives_per_positive: int = 10,
) -> Tensor:
    """Select the hardest negative anchors for one image/frame.

    The paper modifies SSD hard-negative mining to allow at most 10 negatives
    per positive. The returned mask contains negatives only; positives are not
    included in this mask.
    """
    if classification_loss_per_anchor.ndim != 1 or positive_mask.ndim != 1:
        raise ValueError("loss and positive_mask must both be 1-D tensors.")
    if classification_loss_per_anchor.shape != positive_mask.shape:
        raise ValueError("loss and positive_mask must have the same shape.")
    if max_negatives_per_positive < 0:
        raise ValueError("max_negatives_per_positive must be non-negative.")

    num_positive = int(positive_mask.sum().item())
    num_negative_available = int((~positive_mask).sum().item())
    num_negative = min(max_negatives_per_positive * num_positive, num_negative_available)
    selected = torch.zeros_like(positive_mask)
    if num_negative == 0:
        return selected

    losses = classification_loss_per_anchor.detach().clone()
    losses[positive_mask] = -torch.inf
    indices = torch.topk(losses, k=num_negative, largest=True, sorted=False).indices
    selected[indices] = True
    return selected


def _flatten_targets(targets) -> List[dict]:
    if not isinstance(targets, list):
        raise TypeError("targets must be a list or nested list of dictionaries.")
    if not targets:
        return []
    if isinstance(targets[0], dict):
        return targets
    flat: List[dict] = []
    for clip in targets:
        if not isinstance(clip, list):
            raise TypeError("Nested targets must be list[list[dict]].")
        flat.extend(clip)
    return flat


class MultiBoxLoss(nn.Module):
    """SSD loss for the current LSTMSSD outputs.

    Classification uses independent sigmoid/BCE logits, matching the released
    TensorFlow config's ``weighted_sigmoid`` loss. Localization uses Smooth-L1.
    Losses are normalized by the number of matched positive anchors, as in the
    official SSD meta-architecture configuration.

    Paper-specific modification: up to 10 hard negatives per positive, with
    each selected negative classification loss multiplied by 0.3.
    The released ImageNet-VID config sets localization_weight=4.0 and
    classification_weight=1.0; those are the defaults here.
    """

    def __init__(
        self,
        num_classes: int = 30,
        match_iou_threshold: float = 0.5,
        max_negatives_per_positive: int = 10,
        negative_loss_weight: float = 0.3,
        classification_weight: float = 1.0,
        localization_weight: float = 4.0,
        box_coder_scales: BoxCoderScales = BoxCoderScales(),
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.match_iou_threshold = float(match_iou_threshold)
        self.max_negatives_per_positive = int(max_negatives_per_positive)
        self.negative_loss_weight = float(negative_loss_weight)
        self.classification_weight = float(classification_weight)
        self.localization_weight = float(localization_weight)
        self.box_coder_scales = box_coder_scales

    def forward(self, predictions: Dict[str, Tensor], targets) -> Dict[str, Tensor]:
        logits = predictions["class_logits"]
        box_regression = predictions["box_regression"]
        anchors = predictions["anchors"]
        if anchors.ndim != 2 or anchors.shape[-1] != 4:
            raise ValueError("predictions['anchors'] must have shape [A,4].")

        if logits.ndim == 4:
            # [B,T,A,C] -> [N,A,C]
            batch, time, anchors_count, classes = logits.shape
            logits_flat = logits.reshape(batch * time, anchors_count, classes)
            box_flat = box_regression.reshape(batch * time, anchors_count, 4)
        elif logits.ndim == 3:
            logits_flat = logits
            box_flat = box_regression
        else:
            raise ValueError("class_logits must have shape [B,A,C] or [B,T,A,C].")

        flat_targets = _flatten_targets(targets)
        if len(flat_targets) != logits_flat.shape[0]:
            raise ValueError(
                f"Target count ({len(flat_targets)}) does not match prediction samples "
                f"({logits_flat.shape[0]})."
            )
        if logits_flat.shape[1] != anchors.shape[0]:
            raise ValueError("Anchor count does not match predictions.")
        expected_classes = self.num_classes + 1
        if logits_flat.shape[-1] != expected_classes:
            raise ValueError(
                f"Expected {expected_classes} logits (background + foreground), "
                f"got {logits_flat.shape[-1]}."
            )

        total_cls = logits_flat.new_zeros(())
        total_loc = logits_flat.new_zeros(())
        total_positive = 0

        anchors = anchors.to(device=logits_flat.device, dtype=box_flat.dtype)
        for sample_idx, target in enumerate(flat_targets):
            gt_boxes = target["boxes"].to(device=anchors.device, dtype=anchors.dtype)
            gt_labels = target["labels"].to(device=anchors.device, dtype=torch.long)
            if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
                raise ValueError("target boxes must have shape [N,4] normalized xyxy.")
            if gt_labels.ndim != 1 or gt_labels.shape[0] != gt_boxes.shape[0]:
                raise ValueError("target labels must have shape [N] aligned with boxes.")
            if gt_labels.numel() and ((gt_labels < 1).any() or (gt_labels > self.num_classes).any()):
                raise ValueError("Foreground labels must be in 1..num_classes; 0 is background.")

            matched_gt, positive_mask, _ = match_anchors(
                anchors, gt_boxes, iou_threshold=self.match_iou_threshold,
                force_match_for_each_gt=True,
            )
            num_pos = int(positive_mask.sum().item())
            total_positive += num_pos

            cls_targets = logits_flat.new_zeros((anchors.shape[0], expected_classes))
            # Explicit background target for unmatched anchors.
            cls_targets[:, 0] = 1.0
            if num_pos:
                matched_labels = gt_labels[matched_gt[positive_mask]]
                cls_targets[positive_mask, 0] = 0.0
                cls_targets[positive_mask, matched_labels] = 1.0

            per_class_loss = F.binary_cross_entropy_with_logits(
                logits_flat[sample_idx], cls_targets, reduction="none"
            )
            per_anchor_cls = per_class_loss.sum(dim=-1)
            negative_mask = hard_negative_mining(
                per_anchor_cls,
                positive_mask,
                max_negatives_per_positive=self.max_negatives_per_positive,
            )
            cls_weights = per_anchor_cls.new_zeros(per_anchor_cls.shape)
            cls_weights[positive_mask] = 1.0
            cls_weights[negative_mask] = self.negative_loss_weight
            total_cls = total_cls + (per_anchor_cls * cls_weights).sum()

            if num_pos:
                matched_boxes = gt_boxes[matched_gt[positive_mask]]
                encoded_targets = encode_boxes(
                    matched_boxes, anchors[positive_mask], scales=self.box_coder_scales
                )
                loc = F.smooth_l1_loss(
                    box_flat[sample_idx][positive_mask], encoded_targets,
                    reduction="sum", beta=1.0,
                )
                total_loc = total_loc + loc

        normalizer = float(max(total_positive, 1))
        cls_loss = self.classification_weight * total_cls / normalizer
        loc_loss = self.localization_weight * total_loc / normalizer
        return {
            "total_loss": cls_loss + loc_loss,
            "classification_loss": cls_loss,
            "localization_loss": loc_loss,
            "num_positives": logits_flat.new_tensor(float(total_positive)),
        }
