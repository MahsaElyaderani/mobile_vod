"""Fresh joint models and strict classification-to-detection transfer."""
from config import ModelConfig, checkpoint_model_config
from paired_config import PairedConfig
from models.joint_classification import JointClassifier
from utils.runtime import load_checkpoint

CLASSIFICATION_FORMAT = 1


def make_joint_config(backbone='timm_mobilenet_v2', width=1.):
    return ModelConfig(backbone=backbone, width_mult=width, image_size=320,
                       mobilenet_output_channels=1280)


def initialize_backbones(big, small, weights=None, random_init=False):
    """Use timm weights; retain the unused weights argument for existing callers."""
    if random_init:
        return {'initialization': 'random'}
    result = {}
    for name, backbone in (('big', big), ('small', small)):
        result[name] = backbone.load_pretrained()
    return result


def classifier_from_payload(payload):
    if payload.get('classification_format_version') != CLASSIFICATION_FORMAT or payload.get('stage') != 'joint_pretrain':
        raise ValueError('Expected a joint_pretrain classification checkpoint.')
    if payload.get('objective') != 'vid_multilabel_bce':
        raise ValueError('Expected VID multi-label classification objective.')
    mapping = payload.get('class_to_idx', {})
    if len(mapping) != 30 or mapping != {str(k): k-1 for k in range(1, 31)}:
        raise ValueError('Invalid classification class mapping.')
    model = JointClassifier(checkpoint_model_config(payload['model_config']), PairedConfig(**payload['paired_config']), len(mapping))
    model.load_state_dict(payload['model'], strict=True)
    return model


def load_classification(path):
    payload = load_checkpoint(path)
    return classifier_from_payload(payload), payload
