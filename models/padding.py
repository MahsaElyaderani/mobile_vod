import math

from torch import nn
from torch.nn import functional as F


class TFSameConv2d(nn.Conv2d):
    """TensorFlow SAME padding, including asymmetric stride-2 padding."""

    def __init__(self, *args, **kwargs):
        kwargs['padding'] = 0
        super().__init__(*args, **kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ph = max((math.ceil(h / sh) - 1) * sh + (kh - 1) * dh + 1 - h, 0)
        pw = max((math.ceil(w / sw) - 1) * sw + (kw - 1) * dw + 1 - w, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return super().forward(x)
