"""Strict mapping of the original MobileNetV1 classification backbone."""
from pathlib import Path
import hashlib
import torch

HF_MODEL = 'google/mobilenet_v1_1.0_224'
HF_REVISION = '82c74e899e053f9ac5820a574f987d14f8230f10'
OFFICIAL_URL = ('https://storage.googleapis.com/download.tensorflow.org/models/'
                'mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz')
HF_SHA256 = '533912ba76c89fd19ae3572a74ed43965e514e6948c80939c7c3c72e12292f95'
TF_ARCHIVE_SHA256 = '2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def layer_mapping():
    # Paper Conv13 is TensorFlow Conv2d_12_pointwise, NOT Conv2d_13.
    yield 'conv1', 'MobilenetV1/Conv2d_0', 'mobilenet_v1.conv_stem', False
    blocks = ['conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7']
    blocks += [f'conv8_12.{i}' for i in range(5)] + ['conv13']
    for tf_index, name in enumerate(blocks, 1):
        for offset, kind in enumerate(['depthwise', 'pointwise']):
            yield (f'{name}.{kind}', f'MobilenetV1/Conv2d_{tf_index}_{kind}',
                   f'mobilenet_v1.layer.{2 * (tf_index - 1) + offset}',
                   kind == 'depthwise')


def _validate_state(backbone, state):
    expected = backbone.state_dict()
    if set(state) != set(expected):
        raise ValueError(f'Backbone keys differ: missing={set(expected)-set(state)}, '
                         f'unexpected={set(state)-set(expected)}')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Backbone shape/dtype mismatch: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite pretrained tensor: {name}')
    return state


def map_huggingface_state(backbone, source):
    state = {}
    for target, _, origin, _ in layer_mapping():
        state[target + '.0.weight'] = source[origin + '.convolution.weight'].clone()
        for key in ['weight', 'bias', 'running_mean', 'running_var']:
            state[target + '.1.' + key] = source[origin + '.normalization.' + key].clone()
        state[target + '.1.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)
    return _validate_state(backbone, state)


def map_tensorflow_checkpoint(backbone, checkpoint_prefix):
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError('TensorFlow is needed only to convert the original checkpoint.') from error
    reader = tf.train.load_checkpoint(str(checkpoint_prefix))
    state = {}
    for target, origin, _, depthwise in layer_mapping():
        suffix = '/depthwise_weights' if depthwise else '/weights'
        value = torch.from_numpy(reader.get_tensor(origin + suffix).copy())
        value = value.permute(2, 3, 0, 1) if depthwise else value.permute(3, 2, 0, 1)
        state[target + '.0.weight'] = value.contiguous()
        for target_key, tf_key in [('weight', 'gamma'), ('bias', 'beta'),
                                   ('running_mean', 'moving_mean'),
                                   ('running_var', 'moving_variance')]:
            value = reader.get_tensor(origin + '/BatchNorm/' + tf_key)
            state[target + '.1.' + target_key] = torch.from_numpy(value.copy())
        state[target + '.1.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)
    return _validate_state(backbone, state)


def load_pretrained_backbone(backbone, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'mobilenet_v1_backbone' or checkpoint.get('padding') != 'tf_same':
        raise ValueError('Use prepare_weights.py to create the validated backbone checkpoint.')
    state = _validate_state(backbone, checkpoint['backbone'])
    backbone.load_state_dict(state, strict=True)
    return checkpoint['source']


def initialize_standalone_backbone(backbone, path, method="exact"):
    """Initialize a new stage 1 only; slicing is not narrow-model pretraining."""
    if method == "exact":
        return load_pretrained_backbone(backbone, path)
    if method != "slice-pretrained" or backbone.out_channels != 358:
        raise ValueError("slice-pretrained requires a standalone alpha=0.35 backbone.")
    try:
        from models.mobilenet_v1 import MobileNetV1Backbone, initialize_by_channel_slicing
    except ImportError:
        from ..models.mobilenet_v1 import MobileNetV1Backbone, initialize_by_channel_slicing
    source = MobileNetV1Backbone()
    provenance = load_pretrained_backbone(source, path)
    report = initialize_by_channel_slicing(backbone, source)
    for layer in backbone.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.reset_running_stats()
    return {"type": "channel_slice_pretrained", "source": provenance,
            "source_checkpoint_sha256": sha256_file(path),
            "target_width_mult": 0.35, "transfer": report,
            "batchnorm_running_stats_reset": True}
