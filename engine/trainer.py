from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import Tensor, nn

try:
    from utils.runtime import tqdm, loss_postfix
except ImportError:
    from ..utils.runtime import tqdm, loss_postfix

try:
    from models.lstm_ssd import LSTMSSD
except ImportError:
    from ..models.lstm_ssd import LSTMSSD


def _move_targets_to_device(targets, device: torch.device) -> list[list[dict]]:
    moved: list[list[dict]] = []
    for clip in targets:
        moved_clip: list[dict] = []
        for target in clip:
            out = {}
            for key, value in target.items():
                out[key] = value.to(device) if isinstance(value, Tensor) else value
            moved_clip.append(out)
        moved.append(moved_clip)
    return moved


def _mean_metrics(sums: Dict[str, float], batches: int) -> Dict[str, float]:
    if batches == 0:
        return {k: 0.0 for k in sums}
    return {k: v / batches for k, v in sums.items()}


def _check_losses(losses):
    if any(not torch.isfinite(losses[key]).all() for key in
           ('total_loss', 'classification_loss', 'localization_loss')):
        raise FloatingPointError('Non-finite loss; stopped before updating weights.')


def train_detector_stage1(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    max_grad_norm: float | None = 10.0,
    progress: bool = True,
    description: str = "Train (current/mean)",
) -> Dict[str, float]:
    """Train the non-recurrent MobileNetSSD on independent frames."""
    if isinstance(model, LSTMSSD):
        raise ValueError(
            "Stage 1 in the paper is a detector without LSTM. The current LSTMSSD "
            "must not be used as a substitute. Provide a true frame-based SSD model."
        )
    device = torch.device(device)
    model.to(device).train()
    totals = {"total_loss": 0.0, "classification_loss": 0.0, "localization_loss": 0.0}
    batches = 0

    with tqdm(dataloader, desc=description, unit="batch",
              disable=not progress, dynamic_ncols=True) as bar:
        for frames, targets in bar:
            frames = frames.to(device, non_blocking=True)
            batch, time = frames.shape[:2]
            frame_batch = frames.reshape(batch * time, *frames.shape[2:])
            flat_targets = [target for clip in targets for target in clip]
            flat_targets = [[t] for t in flat_targets]
            flat_targets = _move_targets_to_device(flat_targets, device)
            flat_targets = [clip[0] for clip in flat_targets]

            optimizer.zero_grad(set_to_none=True)
            predictions = model(frame_batch)
            loss_dict = criterion(predictions, flat_targets)
            _check_losses(loss_dict)
            loss_dict["total_loss"].backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()

            current = {key: float(loss_dict[key].detach().item()) for key in totals}
            for key in totals:
                totals[key] += current[key]
            batches += 1
            bar.set_postfix(loss_postfix(current, totals, batches))
    return _mean_metrics(totals, batches)


def freeze_backbone_through_conv13(model: LSTMSSD) -> None:
    """Freeze the MobileNetV1 backbone through Conv13."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    # Frozen BatchNorm statistics should not continue updating in stage 2.
    model.backbone.eval()


def train_recurrent_stage2(
    model: LSTMSSD,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    freeze_backbone: bool = True,
    max_grad_norm: float | None = 10.0,
    progress: bool = True,
    description: str = "Train (current/mean)",
    frozen_backbone_batch_size: int = 32,
) -> Dict[str, float]:
    """Train the recurrent detector on clips with full BPTT across each clip."""
    device = torch.device(device)
    model.to(device)
    model.train()
    if freeze_backbone:
        freeze_backbone_through_conv13(model)

    totals = {"total_loss": 0.0, "classification_loss": 0.0, "localization_loss": 0.0}
    batches = 0
    with tqdm(dataloader, desc=description, unit="batch",
              disable=not progress, dynamic_ncols=True) as bar:
        for frames, targets in bar:
            frames = frames.to(device, non_blocking=True)
            targets = _move_targets_to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model.forward_clip(
                frames, state=None, return_anchors=True,
                frozen_backbone_batch_size=frozen_backbone_batch_size,
            )
            loss_dict = criterion(predictions, targets)
            _check_losses(loss_dict)
            loss_dict["total_loss"].backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            if max_grad_norm is not None and trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
            optimizer.step()

            current = {key: float(loss_dict[key].detach().item()) for key in totals}
            for key in totals:
                totals[key] += current[key]
            batches += 1
            bar.set_postfix(loss_postfix(current, totals, batches))
    return _mean_metrics(totals, batches)
