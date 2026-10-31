from __future__ import annotations

import json
from pathlib import Path

import torch

from datasets.imagenet_vid import ImageNetVIDDataset
from datasets.transforms import EvalTransform
from utils.runtime import select_video_subset
from .predictions import sha256_file


def add_data_arguments(parser):
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--frames-root', required=True)
    parser.add_argument('--video-ids', help='JSON list or split manifest containing video_ids.')
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--subset-seed', type=int, default=7)


def selected_dataset(args, image_size=320, clip_length=1, clip_stride=1, interpolation="bilinear", paired=False):
    raw = json.loads(Path(args.annotations).read_text())
    for field in ('videos', 'images', 'categories'):
        ids = [int(row['id']) for row in raw[field]]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate IDs in {field}.')
    # Do not retain a second full annotation tree while constructing the dataset.
    del raw, ids
    dataset = ImageNetVIDDataset(args.annotations, args.frames_root, EvalTransform(image_size, interpolation=interpolation, paired=paired),
                                clip_length=clip_length, clip_stride=clip_stride)
    if args.video_ids:
        selection = json.loads(Path(args.video_ids).read_text())
        if isinstance(selection, dict):
            if selection.get('annotations_sha256') != sha256_file(args.annotations):
                raise ValueError('Video split manifest refers to different annotations.')
            selection = selection['video_ids']
        if not isinstance(selection, list) or not selection or len(set(selection)) != len(selection):
            raise ValueError('Video selection must be a non-empty list of distinct IDs.')
        chosen = set(map(int, selection))
        if not chosen.issubset(dataset.video_ids):
            raise ValueError('Video split includes missing videos.')
        dataset.video_ids = sorted(chosen)
        dataset.clip_index = [pair for pair in dataset.clip_index if pair[0] in chosen]
    select_video_subset(dataset, args.max_videos, args.subset_seed)
    for video_id in dataset.video_ids:
        records = dataset.images_by_video[video_id]
        if not records:
            raise ValueError('Selected video is empty.')
        for record in records:
            if not all(key in record for key in ('frame_id', 'width', 'height')):
                raise ValueError('Routing requires explicit frame_id, width and height for every image.')
            if int(record['width']) < 1 or int(record['height']) < 1:
                raise ValueError('Invalid annotation image dimensions.')
        indices = [int(record['frame_id']) for record in records]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f'Video {video_id} contains frame gaps; provide consecutive video frames.')
    if clip_length > 1 and not dataset.clip_index:
        raise ValueError('No complete training clips for the requested length.')
    dataset.routing_records = {int(r['id']): r for v in dataset.video_ids for r in dataset.images_by_video[v]}
    return dataset


def data_signature(dataset):
    # Most frames share a video directory. Resolve that directory once rather than
    # walking every ancestor for every frame. Preserve individual file symlinks.
    parents, directories = {}, set()
    for record in dataset.routing_records.values():
        path = dataset.frames_root / str(record['file_name'])
        if path.is_symlink():
            directories.add(str(path.resolve().parent))
        else:
            parent = path.parent
            if parent not in parents:
                parents[parent] = str(parent.resolve())
            directories.add(parents[parent])
    return {'annotations_sha256': sha256_file(dataset.annotation_file),
            'video_ids': list(dataset.video_ids), 'clip_length': dataset.clip_length,
            'clip_stride': dataset.clip_stride, 'image_ids': list(dataset.routing_records),
            'video_directories': sorted(directories)}


def validate_native_categories(payload, dataset):
    labels = {int(key): int(value) for key, value in payload['category_id_to_label'].items()}
    names = {int(key): value for key, value in payload['label_to_name'].items()}
    if labels != dataset.category_id_to_label or names != dataset.label_to_name or dataset.num_classes != 30:
        raise ValueError('Native checkpoint and dataset must have identical 30-class definitions.')
    scales = payload['ssd_config']
    if tuple(scales[key] for key in ('y_scale', 'x_scale', 'height_scale', 'width_scale')) != (10, 10, 5, 5):
        raise ValueError('This evaluator requires native box-coder scales 10/10/5/5.')


def guard_held_out(training_signatures, dataset):
    current = data_signature(dataset)
    for trained in training_signatures:
        ids_overlap = (trained['annotations_sha256'] == current['annotations_sha256'] and
                       set(trained['video_ids']) & set(current['video_ids']))
        files_overlap = set(trained.get('video_directories', [])) & set(current['video_directories'])
        if ids_overlap or files_overlap:
            raise ValueError('Evaluation videos overlap joint training videos.')


def movement_between_frames(previous, current):
    from .paired_policy import frame_signature, frame_change
    signature = frame_signature(current)
    return signature, frame_change(signature, previous)
