import torch
from torch import nn
import torch.nn.functional as F
from paired_config import PairedConfig, MemoryCorruption
from .lstm_ssd import LSTMSSD
from datasets.paired_frames import PairedFrames, big_frames


def select_memory(previous, new, commit):
    if len(previous) != 2 or len(new) != 2:
        raise ValueError("A recurrent state is the pair (c,h).")
    if previous[0].shape != new[0].shape or previous[1].shape != new[1].shape:
        raise ValueError("Previous and new state shapes differ.")
    mask = torch.as_tensor(commit, device=new[0].device)
    if mask.dtype != torch.bool or mask.shape != (new[0].shape[0],):
        raise ValueError("Memory commit must be a boolean vector [B].")
    return tuple(torch.where(mask[:, None, None, None], b, a) for a, b in zip(previous, new))


def corrupt_memory(state, eligible, config: MemoryCorruption, generator=None):
    """Perturb only the transferred state; current detection outputs stay clean."""
    mask = torch.as_tensor(eligible, device=state[0].device, dtype=torch.bool)
    if mask.shape != (state[0].shape[0],):
        raise ValueError("Corruption eligibility must have shape [B].")
    selected = mask & (torch.rand(mask.shape, device=mask.device, generator=generator) < config.probability)
    if not selected.any():
        return state, selected
    # Shared non-negative scale/drop mask preserves c/h's relative magnitude.
    scale = (1 + config.noise_std * torch.randn(state[0].shape, device=state[0].device,
                                              dtype=state[0].dtype, generator=generator)).clamp(min=0)
    scale = scale * (torch.rand(scale.shape, device=scale.device, dtype=scale.dtype,
                               generator=generator) >= config.dropout).to(scale.dtype)
    scale = torch.where(selected[:, None, None, None], scale, torch.ones_like(scale))
    return tuple(value * scale for value in state), selected


class PairedVOD(nn.Module):
    def __init__(self, big: LSTMSSD, paired_config=None):
        super().__init__()
        if not isinstance(big, LSTMSSD):
            raise TypeError("The big branch must be a native stage-2 LSTMSSD.")
        self.big = big
        self.paired_config = paired_config or PairedConfig(
            fast_width_mult=(1.0 if big.model_cfg.backbone == 'efficientnet_b0' else
                             .5),
            fast_image_size=big.model_cfg.image_size // 2)
        cfg = self.paired_config
        if big.model_cfg.image_size != 320 or cfg.fast_image_size != 160:
            raise ValueError("Joint inputs must be big 320 and small 160.")
        if big.model_cfg.num_classes != 30 or not big.ssd_cfg.include_background_class:
            raise ValueError("This paired implementation requires the native 30-class VID detector.")
        if big.model_cfg.backbone == 'efficientnet_b0':
            if cfg.fast_width_mult != 1.0 or cfg.min_depth != 8:
                raise ValueError('EfficientNet-B0 pairs only with unscaled TinyNet-D.')
            from .timm_mobilenet_v2 import TimmMobileNetV2Backbone
            self.fast_backbone = TimmMobileNetV2Backbone(1.0, 16, model_name='tinynet_d.in1k')
            self.small_projection = nn.Identity()
        elif big.model_cfg.backbone == 'tinynet_d':
            raise ValueError('TinyNet-D is supported as a matched standalone small backbone.')
        elif big.model_cfg.uses_timm:
            if cfg.fast_width_mult not in (.5, 1.) or cfg.min_depth != 8:
                raise ValueError("The timm small branch requires width .5/1.0 and min_depth 8.")
            from .timm_mobilenet_v2 import TimmMobileNetV2Backbone
            self.fast_backbone = TimmMobileNetV2Backbone(cfg.fast_width_mult, 16)
            self.small_projection = nn.Identity()
        else:
            raise ValueError(f'Unsupported paired backbone: {big.model_cfg.backbone}')
        self.configure_detector_training()

    @property
    def model_cfg(self):
        return self.big.model_cfg

    @property
    def ssd_cfg(self):
        return self.big.ssd_cfg

    def train(self, mode=True):
        super().train(mode)
        if mode and self.paired_config.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if mode and getattr(self, "_freeze_big", False):
            self.big.eval()
        if mode and self.paired_config.fast_bn_momentum is not None:
            for module in self.fast_backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.momentum = self.paired_config.fast_bn_momentum
                    module.train(True)
        return self

    def configure_detector_training(self, freeze_big=False):
        self._freeze_big = freeze_big
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if freeze_big:
            for parameter in self.big.parameters():
                parameter.requires_grad_(False)
        return self.train(True)

    def reset_fast_bn_stats(self):
        """Reset only running estimates, preserving learned affine parameters."""
        for module in self.fast_backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()


    def initial_state(self, frames):
        self.big._validate_frame(big_frames(frames))
        side = self.model_cfg.image_size // 32
        return self.big.lstm.init_state(len(frames), side, side, frames.device, frames.dtype)

    def fast_features(self, frames):
        self.big._validate_frame(big_frames(frames))
        if self.model_cfg.uses_timm:
            if not isinstance(frames, PairedFrames):
                raise ValueError("timm small inference requires PairedFrames resized independently from the source RGB frame.")
            resized = frames.small
        else:
            resized = F.interpolate(frames, size=(self.paired_config.fast_image_size,) * 2,
                                    mode="bilinear", align_corners=False)
        return self.small_projection(self.fast_backbone(resized))

    def forward_big(self, frames, state=None):
        return self.big.forward_step(big_frames(frames), state, return_anchors=True)

    def forward_small(self, frames, state=None):
        feature = self.fast_features(frames)
        return self.big._forward_from_backbone(feature, state, True, anchor_dtype=frames.dtype)

    def forward_selected(self, frames, state, use_big, small_commit=None):
        mask = torch.as_tensor(use_big, device=frames.device)
        if mask.dtype != torch.bool or mask.shape != (len(frames),):
            raise ValueError("use_big must be a boolean vector [B].")
        state = self.initial_state(frames) if state is None else state
        large, small = self.big.backbone(big_frames(frames)), self.fast_features(frames)
        if large.shape != small.shape:
            raise RuntimeError("Aligned big and small features must have identical shapes.")
        features = torch.where(mask[:, None, None, None], large, small)
        output, new_state = self.big._forward_from_backbone(features, state, True,
                                                           anchor_dtype=frames.dtype)
        commit = mask
        if small_commit is not None:
            own = torch.as_tensor(small_commit, device=frames.device)
            if own.dtype != torch.bool or own.shape != mask.shape:
                raise ValueError("small_commit must be a boolean vector [B].")
            commit = mask | own
        return output, select_memory(state, new_state, commit)

    def forward_clip(self, frames, use_big, state=None, small_commit=None):
        if frames.ndim != 5 or frames.shape[1] < 1:
            raise ValueError("Expected a non-empty [B,T,3,H,W] clip.")
        if use_big.shape != frames.shape[:2]:
            raise ValueError("Branch selections must have shape [B,T].")
        logits, boxes, anchors = [], [], None
        for step in range(frames.shape[1]):
            own = None if small_commit is None else small_commit[:, step]
            output, state = self.forward_selected(frames[:, step], state, use_big[:, step], own)
            logits.append(output["class_logits"])
            boxes.append(output["box_regression"])
            anchors = output["anchors"]
        return {"class_logits": torch.stack(logits, 1),
                "box_regression": torch.stack(boxes, 1), "anchors": anchors}, state
