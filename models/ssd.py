from .lstm_ssd import LSTMSSD
from .timm_mobilenet_v2 import build_backbone
from .ssd_head import SSDFeatureMaps, SSDBoxPredictor
from .anchors import SSDAnchorGenerator
from torch import nn


class MobileNetSSD(nn.Module):
    """Non-recurrent stage-1 detector using the existing five SSD scales.

    Conv13 is the first prediction feature (1024 channels at alpha=1, 358 at .35).
    FM1 reduces its
    input to 128 channels. Remaining SSD widths match the recurrent variant.
    This baseline-width choice is explicit; the paper does not specify the
    complete stage-1 checkpoint graph.
    """

    def __init__(self, model_cfg=None, ssd_cfg=None):
        super().__init__()
        # Reuse exactly the recurrent detector's anchors/configuration without
        # allocating an LSTM or an unused second detector.
        try:
            from config import ModelConfig, SSDConfig
        except ImportError:
            from ..config import ModelConfig, SSDConfig
        self.model_cfg = model_cfg or ModelConfig()
        self.ssd_cfg = ssd_cfg or SSDConfig()
        self.backbone = build_backbone(self.model_cfg)
        self.feature_maps = SSDFeatureMaps(input_channels=self.backbone.out_channels)
        self.predictor = SSDBoxPredictor(
            (self.backbone.out_channels,) + tuple(self.model_cfg.feature_channels[1:]),
            self.model_cfg.num_classes, self.ssd_cfg.num_anchors_per_location,
            self.ssd_cfg.include_background_class,
        )
        c = self.ssd_cfg
        self.anchor_generator = SSDAnchorGenerator(
            num_levels=c.num_levels, min_scale=c.min_scale, max_scale=c.max_scale,
            aspect_ratios=c.aspect_ratios,
            interpolated_scale_aspect_ratio=c.interpolated_scale_aspect_ratio,
            reduce_boxes_in_lowest_layer=c.reduce_boxes_in_lowest_layer,
        )

    _validate_frame = LSTMSSD._validate_frame

    def forward(self, frame, return_anchors=True):
        self._validate_frame(frame)
        features = self.feature_maps(self.backbone(frame))
        cls, box = self.predictor(features)
        output = {'class_logits': cls, 'box_regression': box, 'features': features}
        if return_anchors:
            output['anchors'] = self.anchor_generator.generate(
                [f.shape[-2:] for f in features], device=frame.device, dtype=frame.dtype,
            )
            if output['anchors'].shape[0] != cls.shape[1]:
                raise RuntimeError('Anchor/prediction count mismatch')
        return output

    def forward_step(self, frame, state=None, return_anchors=True):
        if state is not None:
            raise ValueError('The stage-1 detector has no recurrent state.')
        return self(frame, return_anchors), None
