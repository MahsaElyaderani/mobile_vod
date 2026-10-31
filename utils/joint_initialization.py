"""Fresh joint models and strict classification-to-detection transfer."""
from config import ModelConfig
from paired_config import PairedConfig
from models.joint_classification import JointClassifier
from utils.runtime import load_checkpoint
from utils.pretrained import initialize_standalone_backbone

CLASSIFICATION_FORMAT = 1


def make_joint_config(backbone='mobilenet_v1', width=1.):
    return ModelConfig(backbone=backbone, width_mult=width, image_size=320,
                       mobilenet_output_channels=1280 if backbone == 'timm_mobilenet_v2' else max(8, int(1024*width)))


def initialize_backbones(big, small, weights=None, random_init=False):
    if random_init:
        return {'initialization': 'random'}
    result = {}
    for name, backbone in (('big', big), ('small', small)):
        if hasattr(backbone, 'load_pretrained'):
            result[name] = backbone.load_pretrained()
        else:
            # Explicitly recorded channel slicing, never represented as exact narrow pretrained weights.
            method = 'exact' if backbone.out_channels == 1024 else 'slice-pretrained'
            result[name] = initialize_standalone_backbone(backbone, weights, method)
    return result


def classifier_from_payload(payload):
    if payload.get('classification_format_version') != CLASSIFICATION_FORMAT or payload.get('stage') != 'joint_pretrain':
        raise ValueError('Expected a joint_pretrain classification checkpoint.')
    if payload.get('objective') != 'vid_multilabel_bce':
        raise ValueError('Expected VID multi-label classification objective.')
    mapping = payload.get('class_to_idx', {})
    if len(mapping) != 30 or mapping != {str(k): k-1 for k in range(1, 31)}:
        raise ValueError('Invalid classification class mapping.')
    model = JointClassifier(ModelConfig(**payload['model_config']), PairedConfig(**payload['paired_config']), len(mapping))
    model.load_state_dict(payload['model'], strict=True)
    return model


def load_classification(path):
    payload = load_checkpoint(path)
    return classifier_from_payload(payload), payload
