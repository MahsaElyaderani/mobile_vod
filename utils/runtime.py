from __future__ import annotations

import os
import random
import tempfile
import inspect
from pathlib import Path

import torch
from tqdm.auto import tqdm


LOSS_KEYS = ("total_loss", "classification_loss", "localization_loss")


def loss_postfix(current, totals, count):
    names = ("total", "cls", "loc")
    return {
        name: f"{current[key]:.4f}/{totals[key] / count:.4f}"
        for name, key in zip(names, LOSS_KEYS)
    }


def select_video_subset(dataset, max_videos=None, seed=0):
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("max_videos must be positive.")
        chosen = random.Random(seed).sample(
            dataset.video_ids, min(max_videos, len(dataset.video_ids))
        )
        keep = set(chosen)
        dataset.video_ids = sorted(keep)
        dataset.clip_index = [
            pair for pair in dataset.clip_index if pair[0] in keep
        ]
    if not dataset.video_ids:
        raise ValueError("No videos selected.")
    return dataset


def select_validation_frames(dataset, frames_per_video=None, seed=0):
    """Select a fixed consecutive segment per video; None keeps all frames."""
    if frames_per_video is None:
        return dataset
    if frames_per_video < 1 or dataset.clip_length != 1:
        raise ValueError('Positive frames_per_video and clip_length=1 are required.')
    rng = random.Random(seed)
    selected = {}
    for video_id in dataset.video_ids:
        records = dataset.images_by_video[video_id]
        if len(records) < frames_per_video:
            selected[video_id] = records
            continue
        starts = []
        for start in range(len(records) - frames_per_video + 1):
            window = records[start:start + frames_per_video]
            ids = [int(r['frame_id']) for r in window if 'frame_id' in r]
            if len(ids) == len(window) and all(b == a + 1 for a, b in zip(ids, ids[1:])):
                starts.append(start)
        if not starts:
            raise ValueError(f'Video {video_id} has no {frames_per_video}-frame consecutive segment.')
        start = rng.choice(starts)
        selected[video_id] = records[start:start + frames_per_video]
    dataset.images_by_video.update(selected)
    dataset.clip_index = [(v, i) for v in dataset.video_ids
                          for i in range(len(dataset.images_by_video[v]))]
    return dataset


def atomic_save_checkpoint(payload, destination, overwrite=True):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # A hard link publishes the complete file and atomically rejects
            # an existing epoch checkpoint, even if another writer races us.
            os.link(temporary, destination)
            temporary.unlink()
        if os.name == "posix":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    try:
        from config import ModelConfig, SSDConfig
    except ImportError:
        from ..config import ModelConfig, SSDConfig
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    if safe_globals is not None:
        with safe_globals([ModelConfig, SSDConfig]):
            return torch.load(path, map_location="cpu", weights_only=True)
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def epoch_checkpoint_path(path, epoch):
    path = Path(path)
    suffix = path.suffix or ".pt"
    return path.with_name(f"{path.stem}_epoch_{epoch:04d}{suffix}")
