from __future__ import annotations

from collections import defaultdict
import time
from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor
from torchvision.ops import nms

try:
    from utils.runtime import tqdm, LOSS_KEYS, loss_postfix
except ImportError:
    from ..utils.runtime import tqdm, LOSS_KEYS, loss_postfix

try:
    from models.anchors import BoxCoderScales, decode_boxes
    from models.lstm_ssd import LSTMSSD
except ImportError:
    from ..models.anchors import BoxCoderScales, decode_boxes
    from ..models.lstm_ssd import LSTMSSD


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (a1[:, None] + a2[None, :] - inter).clamp(min=1e-8)


def postprocess_detections(
    class_logits: Tensor,
    box_regression: Tensor,
    anchors: Tensor,
    num_classes: int = 30,
    score_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    max_detections_per_class: int = 100,
    max_total_detections: int = 100,
    box_coder_scales: BoxCoderScales = BoxCoderScales(),
) -> Dict[str, Tensor]:
    """Decode one frame of SSD predictions and apply class-wise NMS."""
    if class_logits.ndim == 3:
        if class_logits.shape[0] != 1:
            raise ValueError("Batched postprocess currently expects batch size 1.")
        class_logits = class_logits[0]
        box_regression = box_regression[0]
    if class_logits.ndim != 2 or box_regression.ndim != 2:
        raise ValueError("Expected class_logits [A,C] and box_regression [A,4].")
    if class_logits.shape[0] != anchors.shape[0] or box_regression.shape[0] != anchors.shape[0]:
        raise ValueError("Anchor and prediction counts must match.")
    if class_logits.shape[1] != num_classes + 1:
        raise ValueError("Expected background + num_classes logits.")

    boxes = decode_boxes(box_regression, anchors, scales=box_coder_scales, clip=True)
    scores_all = torch.sigmoid(class_logits)
    out_boxes: List[Tensor] = []
    out_scores: List[Tensor] = []
    out_labels: List[Tensor] = []

    for label in range(1, num_classes + 1):
        scores = scores_all[:, label]
        keep = scores > score_threshold
        if not keep.any():
            continue
        cls_boxes = boxes[keep]
        cls_scores = scores[keep]
        kept = nms(cls_boxes, cls_scores, nms_iou_threshold)[:max_detections_per_class]
        out_boxes.append(cls_boxes[kept])
        out_scores.append(cls_scores[kept])
        out_labels.append(torch.full((kept.numel(),), label, device=kept.device, dtype=torch.long))

    if not out_boxes:
        return {
            "boxes": boxes.new_zeros((0, 4)),
            "scores": boxes.new_zeros((0,)),
            "labels": torch.zeros((0,), device=boxes.device, dtype=torch.long),
        }
    boxes_out = torch.cat(out_boxes)
    scores_out = torch.cat(out_scores)
    labels_out = torch.cat(out_labels)
    order = torch.argsort(scores_out, descending=True)[:max_total_detections]
    return {"boxes": boxes_out[order], "scores": scores_out[order], "labels": labels_out[order]}


def _continuous_ap(recalls: Tensor, precisions: Tensor) -> float:
    """Area under the monotonic precision-recall envelope (VOC-style AP)."""
    mrec = torch.cat([recalls.new_tensor([0.0]), recalls, recalls.new_tensor([1.0])])
    mpre = torch.cat([precisions.new_tensor([0.0]), precisions, precisions.new_tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    change = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).flatten()
    return float(((mrec[change + 1] - mrec[change]) * mpre[change + 1]).sum().item())


def class_coverage(counts):
    counts = {i + 1: int(count) for i, count in enumerate(counts)}
    return {'positive_counts_per_class': counts,
            'present_classes': [i for i, count in counts.items() if count],
            'absent_classes': [i for i, count in counts.items() if not count]}


def classification_metrics(scores, targets):
    """Non-interpolated micro AP; equal scores share a single threshold."""
    if scores.shape != targets.shape or scores.ndim != 2 or not scores.numel():
        raise ValueError('Expected matching, nonempty [frames, classes] scores and targets.')
    if not torch.isfinite(scores).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('AP requires finite scores and binary targets.')
    order = scores.flatten().argsort(descending=True)
    ranked = scores.flatten()[order]
    positives = targets.flatten()[order].double().cumsum(0)
    ends = torch.cat((torch.where(ranked[:-1] != ranked[1:])[0], order.new_tensor([len(order)-1])))
    tp = positives[ends]
    precision = tp / (ends + 1)
    increments = torch.diff(tp, prepend=tp.new_zeros(1))
    ap = float((increments * precision).sum() / tp[-1]) if tp[-1] else 0.
    return {'micro_ap': ap, **class_coverage(targets.sum(0))}


def compute_map50(
    predictions: Sequence[Dict[str, Tensor]],
    ground_truths: Sequence[Dict[str, Tensor]],
    num_classes: int = 30,
    iou_threshold: float = 0.5,
    progress: bool = False,
) -> Dict[str, object]:
    """Compute per-class AP and mean AP at a fixed IoU threshold of 0.5.

    Classes with no ground-truth instances are omitted from the mean, which is
    important for small validation subsets. This implements standard detection
    AP matching and a continuous precision envelope; for publication-grade
    reproduction, results should still be cross-checked against the official
    ImageNet VID devkit.
    """
    gt_by_image_class: dict[tuple[int, int], Tensor] = {}
    gt_counts = [0] * (num_classes + 1)
    for gt in ground_truths:
        image_id = int(gt["image_id"].item()) if isinstance(gt["image_id"], Tensor) else int(gt["image_id"])
        boxes = gt["boxes"].detach().cpu()
        labels = gt["labels"].detach().cpu()
        for cls in range(1, num_classes + 1):
            cls_boxes = boxes[labels == cls]
            if cls_boxes.numel():
                gt_by_image_class[(image_id, cls)] = cls_boxes
                gt_counts[cls] += cls_boxes.shape[0]

    preds_by_class: dict[int, list[tuple[float, int, Tensor]]] = defaultdict(list)
    for pred in predictions:
        image_id = int(pred["image_id"].item()) if isinstance(pred["image_id"], Tensor) else int(pred["image_id"])
        boxes = pred["boxes"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        for box, score, label in zip(boxes, scores, labels):
            cls = int(label.item())
            preds_by_class[cls].append((float(score.item()), image_id, box))

    ap_per_class: Dict[int, float] = {}
    for cls in tqdm(range(1, num_classes + 1), desc="AP classes",
                    unit="class", disable=not progress, dynamic_ncols=True):
        num_gt = gt_counts[cls]
        if num_gt == 0:
            continue
        entries = sorted(preds_by_class.get(cls, []), key=lambda x: x[0], reverse=True)
        matched: dict[int, Tensor] = {
            image_id: torch.zeros(boxes.shape[0], dtype=torch.bool)
            for (image_id, c), boxes in gt_by_image_class.items() if c == cls
        }
        tp = torch.zeros(len(entries), dtype=torch.float32)
        fp = torch.zeros(len(entries), dtype=torch.float32)
        for idx, (_, image_id, box) in enumerate(entries):
            gt_boxes = gt_by_image_class.get((image_id, cls))
            if gt_boxes is None or gt_boxes.numel() == 0:
                fp[idx] = 1.0
                continue
            ious = _box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= iou_threshold and not matched[image_id][best_idx]:
                tp[idx] = 1.0
                matched[image_id][best_idx] = True
            else:
                fp[idx] = 1.0
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / float(num_gt)
        precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-12)
        ap_per_class[cls] = _continuous_ap(recalls, precisions)

    map50 = float(sum(ap_per_class.values()) / len(ap_per_class)) if ap_per_class else 0.0
    return {"mAP@0.5": map50, "AP@0.5_per_class": ap_per_class, "evaluated_classes": len(ap_per_class),
            **class_coverage(gt_counts[1:])}


@torch.no_grad()
def evaluate_vid(
    model: LSTMSSD,
    dataset,
    device: torch.device | str,
    score_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    max_detections_per_class: int = 100,
    max_total_detections: int = 100,
    criterion=None,
    progress: bool = True,
    description: str = "Eval (current/mean)",
    prediction_callback=None,
    per_video_metrics: bool = False,
) -> Dict[str, object]:
    """Evaluate online: recurrent state persists within each video only."""
    device = torch.device(device)
    if not 0 <= score_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
        raise ValueError('Score and NMS thresholds must be in [0,1].')
    if min(max_detections_per_class, max_total_detections) < 1:
        raise ValueError('Detection limits must be positive.')
    model.to(device).eval()
    predictions: List[Dict[str, Tensor]] = []
    ground_truths: List[Dict[str, Tensor]] = []

    totals = {key: 0.0 for key in LOSS_KEYS}
    num_frames = sum(len(dataset.images_by_video[v]) for v in dataset.video_ids)
    ranges, export_seconds = {}, 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with tqdm(total=num_frames, desc=description, unit="frame",
              disable=not progress, dynamic_ncols=True) as bar:
        for video_id in dataset.video_ids:
            begin = len(predictions)
            state = None  # reset exactly at the video boundary
            for frame, target in dataset.iter_video(video_id):
                frame = frame.unsqueeze(0).to(device)
                output, state = model.forward_step(frame, state=state, return_anchors=True)
                if not all(torch.isfinite(output[key]).all() for key in ('class_logits', 'box_regression')):
                    raise FloatingPointError('Non-finite detector outputs.')
                if criterion is not None:
                    loss_dict = criterion(output, [target])
                    current = {key: float(loss_dict[key].item()) for key in LOSS_KEYS}
                    for key in totals:
                        totals[key] += current[key]
                detections = postprocess_detections(
                    output["class_logits"], output["box_regression"], output["anchors"],
                    num_classes=model.model_cfg.num_classes,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    max_detections_per_class=max_detections_per_class,
                    max_total_detections=max_total_detections,
                )
                predictions.append({
                    "boxes": detections["boxes"].cpu(),
                    "scores": detections["scores"].cpu(),
                    "labels": detections["labels"].cpu(),
                    "image_id": target["image_id"].cpu(),
                })
                ground_truths.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                    "image_id": target["image_id"].cpu(),
                })
                if prediction_callback is not None:
                    export_started = time.perf_counter()
                    prediction_callback(video_id, target, predictions[-1])
                    export_seconds += time.perf_counter() - export_started
                bar.update(1)
                if criterion is not None:
                    bar.set_postfix(loss_postfix(current, totals, len(ground_truths)))
            ranges[video_id] = (begin, len(predictions))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_total = time.perf_counter() - started
    elapsed = max(elapsed_total - export_seconds, 1e-12)
    metrics = compute_map50(
        predictions, ground_truths,
        num_classes=model.model_cfg.num_classes,
        iou_threshold=0.5, progress=progress,
    )
    metrics["num_frames"] = len(ground_truths)
    count = model.model_cfg.num_classes
    object_counts = {label: 0 for label in range(1, count + 1)}
    frame_counts = {label: 0 for label in range(1, count + 1)}
    for target in ground_truths:
        labels = target['labels'].tolist()
        for label in labels:
            object_counts[label] += 1
        for label in set(labels):
            frame_counts[label] += 1
    metrics.update({
        'implementation': 'mobile_vod', 'num_videos': len(dataset.video_ids),
        'video_ids': dataset.video_ids,
        'frames_per_video': {v: len(dataset.images_by_video[v]) for v in dataset.video_ids},
        'evaluated_image_ids': [int(target['image_id']) for target in ground_truths],
        'score_threshold': score_threshold, 'nms_iou_threshold': nms_iou_threshold,
        'max_detections_per_class': max_detections_per_class,
        'max_total_detections': max_total_detections,
        'class_names': getattr(dataset, 'label_to_name', {}),
        'ground_truth_counts_per_class': object_counts,
        'ground_truth_frames_per_class': frame_counts,
        'inference_seconds': elapsed, 'FPS': len(predictions) / elapsed,
        'export_seconds': export_seconds, 'inference_plus_export_seconds': elapsed_total,
        'FPS_scope': 'image loading, preprocessing, transfer, inference, NMS, optional loss and progress; excludes AP and export callback',
    })
    if per_video_metrics:
        metrics['per_video_metrics'] = []
        for video_id, (begin, end) in ranges.items():
            video = compute_map50(predictions[begin:end], ground_truths[begin:end], count)
            video.update({'video_id': video_id, 'num_frames': end - begin})
            metrics['per_video_metrics'].append(video)
    if criterion is not None:
        metrics.update({
            f"mean_frame_{key}": value / max(len(ground_truths), 1)
            for key, value in totals.items()
        })
    return metrics
