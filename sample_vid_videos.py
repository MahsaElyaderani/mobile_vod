from path import parse_paths
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from utils.results import save_results
from vid_categories import VID_CLASSES


def select_class_samples(annotation_file, classes, videos_per_class=1, seed=7):
    if videos_per_class < 1:
        raise ValueError('videos_per_class must be positive.')
    annotation_file = Path(annotation_file)
    raw = annotation_file.read_bytes()
    data = json.loads(raw)
    for key in ('videos', 'images', 'annotations', 'categories'):
        if key not in data:
            raise ValueError(f'Missing annotation section: {key}')
    aliases = {alias: label for label, pair in enumerate(VID_CLASSES, 1) for alias in pair}
    names = [name.strip().lower().replace(' ', '_') for name in classes]
    if not names or any(name not in aliases for name in names):
        raise ValueError('Specify at least one recognized VID class name or synset ID.')
    requested = sorted({aliases[name] for name in names})
    category_to_fork = {}
    for category in data['categories']:
        name = str(category['name']).strip().lower().replace(' ', '_')
        if name not in aliases:
            raise ValueError(f'Unknown VID category {category["name"]!r}.')
        category_to_fork[int(category['id'])] = aliases[name]
    if len(category_to_fork) != 30 or set(category_to_fork.values()) != set(range(1, 31)):
        raise ValueError('Keep all 30 unambiguous VID category definitions in the source JSON.')
    valid_videos = {int(video['id']) for video in data['videos']}
    image_to_video = {int(image['id']): int(image['video_id']) for image in data['images']
                      if int(image['video_id']) in valid_videos}
    candidates = defaultdict(set)
    for annotation in data['annotations']:
        if annotation.get('iscrowd', False):
            continue
        if annotation['bbox'][2] <= 0 or annotation['bbox'][3] <= 0:
            continue
        video_id = image_to_video.get(int(annotation['image_id']))
        label = category_to_fork.get(int(annotation['category_id']))
        if video_id is not None and label is not None:
            candidates[label].add(video_id)
    rng = random.Random(seed)
    selected, picks = set(), {}
    for label in requested:
        choices = sorted(candidates[label])
        if len(choices) < videos_per_class:
            name = VID_CLASSES[label - 1][0]
            raise ValueError(f'{name}: only {len(choices)} candidate videos; requested {videos_per_class}.')
        chosen = rng.sample(choices, videos_per_class)
        selected.update(chosen)
        picks[VID_CLASSES[label - 1][0]] = chosen
    images = [image for image in data['images'] if int(image['video_id']) in selected]
    image_ids = {int(image['id']) for image in images}
    result = dict(data)
    result.update({
        'videos': [video for video in data['videos'] if int(video['id']) in selected],
        'images': images,
        'annotations': [annotation for annotation in data['annotations']
                        if int(annotation['image_id']) in image_ids],
        'sample_selection': {
            'source_annotation_sha256': hashlib.sha256(raw).hexdigest(),
            'requested_classes': [VID_CLASSES[label - 1][0] for label in requested],
            'videos_per_class': videos_per_class, 'subset_seed': seed,
            'selected_video_ids': sorted(selected), 'picks_per_class': picks,
            'class_coverage_selected_videos': {
                VID_CLASSES[label - 1][0]: sorted(candidates[label] & selected)
                for label in requested
            },
            'selection_scope': 'Whole videos selected using ground-truth class presence; diagnostic sample, not an unbiased full-validation benchmark.',
        },
    })
    return result


def main():
    parser = argparse.ArgumentParser(description='Save a reproducible VID video group covering selected classes.')
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--classes', nargs='+', default=[name for name, _ in VID_CLASSES])
    parser.add_argument('--videos-per-class', type=int, default=1)
    parser.add_argument('--subset-seed', type=int, default=7)
    parser.add_argument('--output', required=True)
    args = parse_paths(parser, "sample_vid_videos")
    result = select_class_samples(args.annotations, args.classes, args.videos_per_class, args.subset_seed)
    save_results(result, args.output)
    print(json.dumps(result['sample_selection'], indent=2))
    print(f'Saved {len(result["videos"])} distinct videos and {len(result["images"])} frames to {args.output}')


if __name__ == '__main__':
    main()
