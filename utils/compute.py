from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of scalar parameters in a model."""
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


@torch.no_grad()
def compute_macs_per_frame(
    model: nn.Module,
    image_size: int = 320,
    device: torch.device | str | None = None,
    return_breakdown: bool = False,
):
    """Count Conv2d/Linear multiply-accumulates for one online frame.

    One MAC is counted as one multiply-accumulate pair, matching the paper's
    reporting convention. Elementwise activations, sigmoid, recurrent state
    additions/multiplications, BatchNorm, NMS and anchor generation are not
    counted. This is therefore a transparent convolution/linear MAC count, not
    a hardware latency estimator.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    was_training = model.training
    model.to(device).eval()

    total = 0
    breakdown: Dict[str, int] = {}
    handles = []

    def register(name: str, module: nn.Module):
        def hook(mod, inputs, output):
            nonlocal total
            if isinstance(mod, nn.Conv2d):
                out = output
                batch, out_c, out_h, out_w = out.shape
                kernel_h, kernel_w = mod.kernel_size
                in_per_group = mod.in_channels // mod.groups
                macs = batch * out_c * out_h * out_w * in_per_group * kernel_h * kernel_w
            elif isinstance(mod, nn.Linear):
                out_elements = output.numel() // output.shape[-1]
                macs = out_elements * mod.in_features * mod.out_features
            else:
                return
            total += int(macs)
            breakdown[name] = breakdown.get(name, 0) + int(macs)
        handles.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            register(name, module)

    try:
        dummy = torch.zeros((1, 3, image_size, image_size), device=device)
        if hasattr(model, "forward_step"):
            model.forward_step(dummy, state=None, return_anchors=False)
        else:
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return (total, breakdown) if return_breakdown else total


@torch.no_grad()
def profile_conv_linear(module, forward):
    total, handles = [0], []
    modes = [(child, child.training) for child in module.modules()]
    def hook(child, inputs, output):
        if isinstance(child, nn.Conv2d):
            total[0] += output.numel() * (child.in_channels // child.groups) * child.kernel_size[0] * child.kernel_size[1]
        elif isinstance(child, nn.Linear):
            total[0] += output.numel() * child.in_features
    module.eval()
    try:
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                handles.append(child.register_forward_hook(hook))
        forward()
    finally:
        for handle in handles:
            handle.remove()
        for child, mode in modes:
            child.training = mode
    return total[0]


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
