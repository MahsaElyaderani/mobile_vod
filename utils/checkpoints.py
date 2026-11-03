from dataclasses import asdict
import random
import torch

try:
    from config import ModelConfig, SSDConfig, checkpoint_model_config
    from models.ssd import MobileNetSSD
    from models.lstm_ssd import LSTMSSD
except ImportError:
    from ..config import ModelConfig, SSDConfig, checkpoint_model_config
    from ..models.ssd import MobileNetSSD
    from ..models.lstm_ssd import LSTMSSD


FORMAT_VERSION = 2


def model_from_checkpoint(checkpoint):
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError('Checkpoint lacks the current architecture/stage metadata. '
                         'Legacy checkpoints cannot be silently loaded after the padding correction.')
    stage = checkpoint.get('stage')
    if stage not in ['stage1', 'stage2']:
        raise ValueError(f'Unknown checkpoint stage: {stage}')
    cls = MobileNetSSD if stage == 'stage1' else LSTMSSD
    model = cls(checkpoint_model_config(checkpoint['model_config']), SSDConfig(**checkpoint['ssd_config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


def transfer_stage1(model, checkpoint):
    if checkpoint.get('stage') != 'stage1':
        raise ValueError('Stage 2 requires a completed stage-1 checkpoint; use --resume for stage 2.')
    source_model = model_from_checkpoint(checkpoint)
    if (asdict(model.model_cfg) != asdict(source_model.model_cfg) or
            asdict(model.ssd_cfg) != asdict(source_model.ssd_cfg)):
        raise ValueError('Stage-1 and stage-2 model/SSD configurations differ.')
    source = source_model.state_dict()
    destination = model.state_dict()
    expected_changed = {
        'feature_maps.fm1.reduce.0.weight',
        'predictor.class_heads.0.depthwise.weight',
        'predictor.class_heads.0.pointwise.weight',
        'predictor.box_heads.0.depthwise.weight',
        'predictor.box_heads.0.pointwise.weight',
    }
    transferred, changed, new = [], [], []
    for name, value in destination.items():
        if name.startswith('lstm.'):
            new.append(name)
        elif name in source and value.shape == source[name].shape:
            destination[name] = source[name].clone()
            transferred.append(name)
        else:
            changed.append(name)
    if set(changed) != expected_changed:
        raise ValueError(f'Unexpected stage-transition shape changes: {changed}')
    if set(source) - set(destination):
        raise ValueError('Stage-1 checkpoint contains unexpected tensors.')
    model.load_state_dict(destination, strict=True)
    return {'transferred': transferred, 'reinitialized_shape_changes': changed,
            'new_lstm_tensors': new}


def capture_rng():
    result = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
