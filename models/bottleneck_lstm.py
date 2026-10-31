from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

State = Tuple[Tensor, Tensor]  # (c, h), matching the authors' TensorFlow cell


class SeparableConv2d(nn.Module):
    """Depthwise convolution followed by a pointwise projection, no BN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial size.")
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class BottleneckLSTMCell(nn.Module):
    """PyTorch translation of the paper's Bottleneck-LSTM equations.

    b_t = phi( SepConv([x_t, h_{t-1}]) )
    [i,j,f,o] = SepConv(b_t)
    c_t = c_{t-1} * sigmoid(f + forget_bias) + sigmoid(i) * phi(j)
    h_t = phi(c_t) * sigmoid(o)

    The alpha=1 architecture uses input_channels=1024 and hidden_channels=256.
    """

    def __init__(
        self,
        input_channels: int = 1024,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        forget_bias: float = 1.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.forget_bias = float(forget_bias)
        self.activation_name = activation

        if activation not in {"relu", "relu6"}:
            raise ValueError("activation must be 'relu' or 'relu6'.")

        self.bottleneck = SeparableConv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gates = SeparableConv2d(
            hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Conservative initialization for a recurrent convolutional cell.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def _activation(self, x: Tensor) -> Tensor:
        if self.activation_name == "relu":
            return F.relu(x)
        return F.relu6(x)

    def init_state(
        self,
        batch: int,
        h: int,
        w: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> State:
        if batch <= 0 or h <= 0 or w <= 0:
            raise ValueError("batch, h and w must all be positive.")
        if dtype is None:
            dtype = next(self.parameters()).dtype
        shape = (batch, self.hidden_channels, h, w)
        c = torch.zeros(shape, device=device, dtype=dtype)
        hidden = torch.zeros(shape, device=device, dtype=dtype)
        return c, hidden

    def forward(self, x: Tensor, state: State | None = None) -> Tuple[Tensor, State]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[1]}"
            )

        batch, _, height, width = x.shape
        if state is None:
            c, h_prev = self.init_state(batch, height, width, x.device, x.dtype)
        else:
            c, h_prev = state
            expected = (batch, self.hidden_channels, height, width)
            if tuple(c.shape) != expected or tuple(h_prev.shape) != expected:
                raise ValueError(
                    f"State tensors must both have shape {expected}; "
                    f"got c={tuple(c.shape)}, h={tuple(h_prev.shape)}"
                )
            if c.device != x.device or h_prev.device != x.device:
                raise ValueError("Input and recurrent state must be on the same device.")
            if c.dtype != x.dtype or h_prev.dtype != x.dtype:
                raise ValueError("Input and recurrent state must have the same dtype.")

        bottleneck = self._activation(self.bottleneck(torch.cat([x, h_prev], dim=1)))
        gate_tensor = self.gates(bottleneck)
        i, j, f, o = torch.chunk(gate_tensor, chunks=4, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias)
        new_c = new_c + torch.sigmoid(i) * self._activation(j)
        new_h = self._activation(new_c) * torch.sigmoid(o)
        return new_h, (new_c, new_h)
