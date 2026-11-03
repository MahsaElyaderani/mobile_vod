"""Image classification through the same two backbones and shared detector LSTM."""
import torch
from torch import nn
import torch.nn.functional as F
from datasets.paired_frames import PairedFrames, big_frames
from models.lstm_ssd import LSTMSSD
from models.paired_vod import PairedVOD, select_memory
from routing.paired_policy import random_training_schedule


class JointClassifier(nn.Module):
    def __init__(self, model_config, paired_config, num_classes, *, detector=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError('Classification requires at least two classes.')
        # Reuse the exact detector topology, but retain no detection heads or anchors.
        pair = detector if detector is not None else PairedVOD(LSTMSSD(model_config), paired_config)
        self.standalone = getattr(pair, 'standalone', None)
        self.model_cfg, self.paired_config = model_config, paired_config
        self.ssd_cfg = pair.ssd_cfg
        self.big_backbone = pair.big.backbone
        self.small_backbone = pair.fast_backbone
        self.small_projection = pair.small_projection
        self.lstm = pair.big.lstm
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(model_config.lstm_channels, num_classes)
        self.num_classes = num_classes

    def train(self, mode=True):
        super().train(mode)
        if mode and self.paired_config.freeze_bn_stats:
            for layer in self.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, frames, steps=3, use_big=None):
        if steps < 1 or frames.ndim != 4 or big_frames(frames).shape[1:] != (3, 320, 320):
            raise ValueError('Expected a batch of 320 RGB views and positive unroll length.')
        if use_big is None:
            use_big, _ = random_training_schedule(len(frames), steps, frames.device)
        if use_big.dtype != torch.bool or use_big.shape != (len(frames), steps):
            raise ValueError('Branch choices must be boolean [batch, steps].')
        if self.model_cfg.uses_timm:
            if not isinstance(frames, PairedFrames):
                raise ValueError('V2 classification requires independently resized PairedFrames.')
            small = frames.small
        else:
            small = F.interpolate(big_frames(frames), (160, 160), mode='bilinear', align_corners=False)
        state = self.lstm.init_state(len(frames), 10, 10, frames.device, frames.dtype)
        logits = []
        reuse_features = (self.model_cfg.uses_timm
                          and self.paired_config.freeze_bn_stats)
        for step in range(steps):
            # Repeated image: retain one backbone graph for all recurrent steps.
            if step == 0 or not reuse_features:
                large = self.big_backbone(small if self.standalone == 'small' else big_frames(frames))
                if self.standalone == 'small':
                    large = self.small_projection(large)
                fast = large if self.standalone else self.small_projection(self.small_backbone(small))
            if large.shape != fast.shape or large.shape[1:] != (self.model_cfg.mobilenet_output_channels, 10, 10):
                raise ValueError('Joint backbone features do not match the shared LSTM input.')
            features = torch.where(use_big[:, step, None, None, None], large, fast)
            output, candidate = self.lstm(features, state)
            logits.append(self.classifier(self.pool(output).flatten(1)))
            state = candidate if self.standalone else select_memory(state, candidate, use_big[:, step])
        return torch.stack(logits, 1)

    def transfer_to_detector(self, detector):
        if self.standalone != getattr(detector, 'standalone', None):
            raise ValueError('Classification and detection branch layouts differ.')
        if (self.model_cfg != detector.model_cfg or
                self.paired_config.fast_width_mult != detector.paired_config.fast_width_mult or
                self.paired_config.fast_image_size != detector.paired_config.fast_image_size or
                self.paired_config.min_depth != detector.paired_config.min_depth):
            raise ValueError('Classification and detection architectures differ.')
        for target, source in ((detector.big.backbone, self.big_backbone),
                               (detector.fast_backbone, self.small_backbone),
                               (detector.small_projection, self.small_projection),
                               (detector.big.lstm, self.lstm)):
            target.load_state_dict(source.state_dict(), strict=True)
