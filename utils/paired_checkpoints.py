from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from config import ModelConfig, SSDConfig, model_config_dict
from models.lstm_ssd import LSTMSSD
from models.paired_vod import PairedVOD
from models.mobilenet_v1 import initialize_by_channel_slicing
from paired_config import PairedConfig
from routing.predictions import sha256_file
from utils.checkpoints import model_from_checkpoint
from utils.runtime import load_checkpoint, atomic_save_checkpoint, epoch_checkpoint_path


PAIRED_FORMAT_VERSION = 1


def paired_config_dict(model):
    config = asdict(model.paired_config)
    # Preserve fingerprints and resume definitions for pre-curriculum checkpoints.
    if config.get("fast_bn_momentum") is None:
        config.pop("fast_bn_momentum", None)
    return config


def detector_fingerprint(model):
    digest = hashlib.sha256()
    configuration = {"model": model_config_dict(model.model_cfg), "ssd": asdict(model.ssd_cfg),
                     "paired": paired_config_dict(model)}
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def paired_metadata(model):
    return {"paired_format_version": PAIRED_FORMAT_VERSION,
            "model_config": model_config_dict(model.model_cfg), "ssd_config": asdict(model.ssd_cfg),
            "paired_config": paired_config_dict(model)}


def paired_from_payload(payload, expected_stage=None):
    if payload.get("paired_format_version") != PAIRED_FORMAT_VERSION:
        raise ValueError("Not a supported paired-model checkpoint.")
    if payload.get("stage") != "detector":
        raise ValueError("Unknown paired checkpoint stage.")
    if expected_stage is not None and payload["stage"] != expected_stage:
        raise ValueError(f"Expected paired {expected_stage} checkpoint.")
    model = PairedVOD(LSTMSSD(ModelConfig(**payload["model_config"]),
                             SSDConfig(**payload["ssd_config"])),
                      PairedConfig(**payload["paired_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model


def initialize_pair(big_path, config, fast_init="slice-big", fast_path=None,
                    init_source="standalone_detection", model_config=None,
                    backbone_weights=None, random_backbones=False):
    if init_source == "joint_pretrain":
        from utils.joint_initialization import load_classification
        classifier, payload = load_classification(big_path)
        model = PairedVOD(LSTMSSD(classifier.model_cfg), config)
        classifier.transfer_to_detector(model)
        return model, payload, {"initialization": "joint_pretrain",
                               "big_checkpoint_sha256": sha256_file(big_path),
                               "classification_epoch": payload['epoch']}
    if init_source == "fresh":
        if big_path or fast_path:
            raise ValueError("Fresh initialization does not take detector checkpoints.")
        from utils.joint_initialization import initialize_backbones
        model = PairedVOD(LSTMSSD(model_config or ModelConfig()), config)
        source = initialize_backbones(model.big.backbone, model.fast_backbone, backbone_weights, random_backbones)
        return model, {}, {"initialization": "fresh", "backbones": source}
    if init_source != "standalone_detection":
        raise ValueError("Unknown joint initialization source.")
    payload = load_checkpoint(big_path)
    if payload.get("stage") != "stage2" or payload.get("validation_pending"):
        raise ValueError("standalone_detection initialization requires a native stage2 checkpoint with completed validation.")
    model = PairedVOD(model_from_checkpoint(payload), config)
    source = {"big_checkpoint_sha256": sha256_file(big_path),
              "big_checkpoint_epoch": payload.get("epoch"), "fast_initialization": fast_init}
    is_timm = model.model_cfg.backbone == "timm_mobilenet_v2"
    if is_timm and fast_init not in ("timm", "standalone", "checkpoint", "random"):
        raise ValueError("Unsupported timm small initialization.")
    if fast_init == "timm":
        if not is_timm or fast_path:
            raise ValueError("timm initialization requires a timm big checkpoint and no fast checkpoint.")
        source["fast_pretrained"] = model.fast_backbone.load_pretrained()
    elif fast_init == "slice-big":
        if fast_path:
            raise ValueError("--fast-checkpoint requires --fast-init checkpoint.")
        source["fast_transfer"] = initialize_by_channel_slicing(model.fast_backbone, model.big.backbone)
    elif fast_init == "checkpoint":
        if not fast_path:
            raise ValueError("--fast-init checkpoint requires --fast-checkpoint.")
        weights = load_checkpoint(fast_path)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        if not isinstance(weights, dict) or not all(isinstance(v, torch.Tensor) for v in weights.values()):
            raise ValueError("Fast checkpoint must contain a matching backbone state_dict only.")
        model.fast_backbone.load_state_dict(weights, strict=True)
        source["fast_checkpoint_sha256"] = sha256_file(fast_path)
    elif fast_init == "standalone":
        if not fast_path:
            raise ValueError("--fast-init standalone requires --fast-checkpoint.")
        weights = load_checkpoint(fast_path)
        if weights.get("validation_pending"):
            raise ValueError("Complete standalone validation before transferring its backbone.")
        native = model_from_checkpoint(weights)  # validates the entire source strictly
        if (native.model_cfg.width_mult != config.fast_width_mult or
                native.model_cfg.image_size not in (160, 320) or
                native.model_cfg.backbone != model.model_cfg.backbone):
            raise ValueError("Standalone family/width must match the small branch; resolution must be 160 or legacy 320.")
        for key in ("category_id_to_label", "label_to_name"):
            if weights.get(key) != payload.get(key):
                raise ValueError("Standalone and big checkpoint class definitions differ.")
        model.fast_backbone.load_state_dict(native.backbone.state_dict(), strict=True)
        source.update({"fast_checkpoint_sha256": sha256_file(fast_path),
                       "fast_checkpoint_epoch": weights.get("epoch"),
                       "fast_checkpoint_stage": weights["stage"],
                       "fast_transfer": {"components": ["backbone"],
                                         "source_input_size": native.model_cfg.image_size,
                                         "target_input_size": config.fast_image_size,
                                         "source_output_stride": 32,
                                         "target_output_stride": 16}})
    elif fast_init != "random" or fast_path:
        raise ValueError("Unknown or inconsistent fast initialization.")
    source["small_projection_initialization"] = "identity; 1280 channels" if is_timm else "random; trained during joint fine-tuning"
    return model, payload, source


def load_pair(path, expected_stage=None):
    payload = load_checkpoint(path)
    return paired_from_payload(payload, expected_stage), payload


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def run_lock(destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def save_training_payload(payload, destination):
    numbered = epoch_checkpoint_path(destination, payload["epoch"])
    if numbered.exists():
        previous = load_checkpoint(numbered)
        if previous.get("run_id") != payload["run_id"] or previous.get("epoch") != payload["epoch"]:
            raise FileExistsError(f"Different run owns checkpoint {numbered}.")
        if not previous.get("validation_pending"):
            raise FileExistsError(f"Completed epoch checkpoint is immutable: {numbered}.")
        atomic_save_checkpoint(payload, numbered)
    else:
        atomic_save_checkpoint(payload, numbered, overwrite=False)
    atomic_save_checkpoint(payload, destination)


def finish_validation(payload, metrics, destination):
    if not payload.get("validation_pending"):
        raise ValueError("No pending validation to finish.")
    current = float(metrics["mAP@0.5"])
    best_path = Path(destination).with_name(Path(destination).stem + "_best" + (Path(destination).suffix or ".pt"))
    known = payload.get("best_map50")
    if best_path.exists():
        stored = load_checkpoint(best_path)
        if stored.get("run_id") != payload["run_id"]:
            raise FileExistsError("Best checkpoint belongs to another run.")
        if known is None or stored["best_map50"] > known:
            known = stored["best_map50"]
    improved = known is None or current > known
    payload.update(validation_pending=False, validation_metrics=metrics,
                   best_map50=current if improved else known)
    payload.pop("validation_error", None)
    if improved:
        atomic_save_checkpoint(payload, best_path)
    save_training_payload(payload, destination)
    return payload
