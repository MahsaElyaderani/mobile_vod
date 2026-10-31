"""Two RGB views resized from the same source, before backbone normalization."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PairedFrames:
    big: torch.Tensor
    small: torch.Tensor

    def __post_init__(self):
        if (self.big.shape[-3:] != (3, 320, 320) or self.small.shape[-3:] != (3, 160, 160)
                or self.big.shape[:-3] != self.small.shape[:-3]):
            raise ValueError('Paired RGB views require matching batch/time axes and sizes 320/160.')
        if self.big.device != self.small.device or self.big.dtype != self.small.dtype:
            raise ValueError('Paired views must have the same device and dtype.')

    # Training indexes batch/time axes; spatial axes refer to the big view.
    @property
    def shape(self):
        return self.big.shape

    @property
    def ndim(self):
        return self.big.ndim

    @property
    def device(self):
        return self.big.device

    @property
    def dtype(self):
        return self.big.dtype

    def __len__(self):
        return len(self.big)

    def __getitem__(self, index):
        return PairedFrames(self.big[index], self.small[index])

    def to(self, *args, **kwargs):
        return PairedFrames(self.big.to(*args, **kwargs), self.small.to(*args, **kwargs))

    def detach(self):
        return PairedFrames(self.big.detach(), self.small.detach())

    def cpu(self):
        return self.to('cpu')

    def pin_memory(self):
        return PairedFrames(self.big.pin_memory(), self.small.pin_memory())


def big_frames(frames):
    return frames.big if isinstance(frames, PairedFrames) else frames


def stack_frames(frames, dim=0):
    if isinstance(frames[0], PairedFrames):
        if not all(isinstance(frame, PairedFrames) for frame in frames):
            raise ValueError('Cannot mix single and paired views in a batch.')
        return PairedFrames(torch.stack([f.big for f in frames], dim),
                            torch.stack([f.small for f in frames], dim))
    return torch.stack(list(frames), dim)
