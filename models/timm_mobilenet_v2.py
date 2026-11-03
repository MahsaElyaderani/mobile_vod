"""Checkpoint-specific timm features; project inputs are RGB in [-1, 1]."""
import torch
from torch import nn


MODEL_NAMES = {1.0: 'mobilenetv2_100.ra_in1k', 0.5: 'mobilenetv2_050.lamb_in1k',
               1.1: 'mobilenetv2_110d.ra_in1k', 1.2: 'mobilenetv2_120d.ra_in1k'}


class TimmMobileNetV2Backbone(nn.Module):
    def __init__(self, width_mult=1.0, output_stride=32, *, model_name=None):
        super().__init__()
        if width_mult not in MODEL_NAMES or output_stride not in (16, 32):
            raise ValueError('Supported timm variants: .5/1.0/1.1d/1.2d; output strides: 32/16.')
        import timm
        if model_name not in (None, 'tinynet_d.in1k', 'efficientnet_b0.ra_in1k'):
            raise ValueError('Unsupported pretrained architecture.')
        if model_name is not None and width_mult != 1.0:
            raise ValueError('Named TinyNet/EfficientNet checkpoints cannot be width-scaled.')
        self.model_name = model_name or MODEL_NAMES[width_mult]
        self.width_mult, self.output_stride = width_mult, output_stride
        # Reconstructing saved detectors must never download classification weights.
        self.network = timm.create_model(self.model_name, pretrained=False)
        self.data_config = timm.data.resolve_model_data_config(self.network)
        self._validate_data_config(self.data_config)
        self.network.reset_classifier(0)
        self.out_channels = self.network.num_features
        if self.out_channels != 1280:
            raise ValueError('Expected the standard 1280-channel final convolution.')
        self.stride_changes = []
        if output_stride == 16:
            layer = self.network.blocks[5][0].conv_dw
            if layer.stride != (2, 2) or layer.dilation != (1, 1):
                raise ValueError('Unexpected timm downsampling architecture.')
            layer.stride = (1, 1)
            self.stride_changes = ['blocks.5.0.conv_dw: stride 2 -> 1; dilation unchanged']
        self.register_buffer('input_mean', torch.tensor(self.data_config['mean']).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('input_std', torch.tensor(self.data_config['std']).view(1, 3, 1, 1), persistent=False)

    def _validate_data_config(self, config):
        size = 152 if self.model_name == 'tinynet_d.in1k' else 224
        if (tuple(config['mean']) != (.485, .456, .406) or
                tuple(config['std']) != (.229, .224, .225) or
                config['interpolation'] != 'bicubic' or tuple(config['input_size']) != (3, size, size)):
            raise ValueError('Checkpoint preprocessing differs from the supported detection pipeline.')

    def normalize(self, frames):
        if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point():
            raise ValueError('Expected floating RGB [N,3,H,W] in project [-1,1] scaling.')
        return ((frames + 1) * .5 - self.input_mean) / self.input_std

    def forward(self, frames):
        # features_only omits conv_head; forward_features retains all 1280 channels.
        return self.network.forward_features(self.normalize(frames))

    def load_pretrained(self):
        import timm
        from path import TIMM_CACHE
        source = timm.create_model(self.model_name, pretrained=True, cache_dir=str(TIMM_CACHE))
        config = timm.data.resolve_model_data_config(source)
        self._validate_data_config(config)
        if config != self.data_config:
            raise ValueError('Loaded checkpoint preprocessing configuration changed.')
        source.reset_classifier(0)
        weights = source.state_dict()
        self.network.load_state_dict(weights, strict=True)
        if any(not torch.equal(value, self.network.state_dict()[key]) for key, value in weights.items()):
            raise RuntimeError('Pretrained parameter verification failed.')
        return {'type': 'timm', 'model_name': self.model_name, 'timm_version': timm.__version__,
                'data_config': config, 'feature_tensors_loaded': len(weights),
                'output_stride': self.output_stride, 'stride_changes': self.stride_changes,
                'detection_preprocessing': 'RGB; bicubic full-frame resize; no center crop; ImageNet normalization',
                'classifier': 'excluded; all feature parameters and buffers loaded strictly'}


def build_backbone(config):
    if config.backbone in ('tinynet_d', 'efficientnet_b0'):
        name = 'tinynet_d.in1k' if config.backbone == 'tinynet_d' else 'efficientnet_b0.ra_in1k'
        return TimmMobileNetV2Backbone(model_name=name)
    if config.backbone == 'timm_mobilenet_v2':
        return TimmMobileNetV2Backbone(config.width_mult, 32)
    raise ValueError(f'Unsupported backbone: {config.backbone}')
