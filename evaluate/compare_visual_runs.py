import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import parse_paths
import argparse
import csv
import json
import math
from itertools import zip_longest

import cv2
import numpy as np
import torch

from engine.visualizer import render_frame
from utils.results import save_results
from utils.runtime import tqdm


COMPARISON_KEYS = ('evaluated_image_ids', 'video_ids', 'frames_per_video', 'class_names',
                   'ground_truth_counts_per_class', 'score_threshold', 'nms_iou_threshold',
                   'max_detections_per_class', 'max_total_detections')


def validate_runs(run_a, run_b):
    metrics = []
    for directory in (run_a, run_b):
        if json.loads((directory / 'run_status.json').read_text())['status'] != 'complete':
            raise ValueError(f'Incomplete source run: {directory}')
        metrics.append(json.loads((directory / 'metrics.json').read_text()))
    for key in COMPARISON_KEYS:
        if metrics[0].get(key) != metrics[1].get(key) or key not in metrics[0]:
            raise ValueError(f'Source runs differ or lack comparison field: {key}')
    return metrics


def row_tensors(row):
    def tensors(entries, scores=False):
        result = {'boxes': torch.tensor([item['box_xyxy_normalized'] for item in entries],
                                         dtype=torch.float32).reshape(-1, 4),
                  'labels': torch.tensor([item['dataset_label'] for item in entries], dtype=torch.long)}
        if scores:
            result['scores'] = torch.tensor([item['score'] for item in entries], dtype=torch.float32)
        return result
    target = tensors(row['ground_truth'])
    target.update({'video_id': row['video_id'], 'frame_id': row['frame_id']})
    return target, tensors(row['predictions'], True)


def comparison_frame(image, row_a, row_b, class_names, label_a, label_b, threshold, max_side):
    target, pred_a = row_tensors(row_a)
    _, pred_b = row_tensors(row_b)
    a = render_frame(image, target, pred_a, class_names, label_a, threshold, 'side-by-side', max_side)
    b = render_frame(image, target, pred_b, class_names, label_b, threshold, 'side-by-side', max_side)
    width = a.shape[1] // 2
    result = np.concatenate((a[:, :width], a[:, width:], b[:, width:]), axis=1)
    return cv2.copyMakeBorder(result, 0, 0, 0, result.shape[1] % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    parser = argparse.ArgumentParser(description='Render matched saved detections as GT | model A | model B; no inference rerun.')
    parser.add_argument('--run-a', required=True)
    parser.add_argument('--run-b', required=True)
    parser.add_argument('--frames-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--label-a', default='stage1')
    parser.add_argument('--label-b', default='stage2')
    parser.add_argument('--draw-score-threshold', type=float, default=.3)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--playback-fps', type=float, default=30)
    parser.add_argument('--save-frames-every', type=int, default=30)
    parser.add_argument('--no-progress', action='store_true')
    args = parse_paths(parser, "compare_visual_runs")
    if args.label_a == args.label_b:
        raise ValueError('Use distinct names for the two models.')
    if (not 0 <= args.draw_score_threshold <= 1 or args.max_side < 1 or
            not math.isfinite(args.playback_fps) or args.playback_fps <= 0 or args.save_frames_every < 0):
        raise ValueError('Invalid rendering settings.')
    a, b = Path(args.run_a), Path(args.run_b)
    metrics = validate_runs(a, b)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'videos').mkdir()
    if args.save_frames_every:
        (output / 'frames').mkdir()
    names = {int(label): name for label, name in metrics[0]['class_names'].items()}
    writer, current, shape, indices, seen = None, None, None, [], []
    try:
        with (a / 'detections.jsonl').open() as stream_a, (b / 'detections.jsonl').open() as stream_b:
            rows = zip_longest(stream_a, stream_b)
            for line_a, line_b in tqdm(rows, total=metrics[0]['num_frames'], desc='Compare saved predictions',
                                       unit='frame', disable=args.no_progress):
                if line_a is None or line_b is None:
                    raise ValueError('Source detection files have different lengths.')
                row_a, row_b = json.loads(line_a), json.loads(line_b)
                for key in ('image_id', 'video_id', 'frame_id', 'file_name', 'original_size', 'ground_truth'):
                    if row_a[key] != row_b[key]:
                        raise ValueError(f'Detection rows differ at {key}; frames must match exactly.')
                image = cv2.imread(str(Path(args.frames_root) / row_a['file_name']))
                if image is None or [image.shape[1], image.shape[0]] != row_a['original_size']:
                    raise ValueError(f'Image missing or dimensions differ: {row_a["file_name"]}')
                canvas = comparison_frame(image, row_a, row_b, names, args.label_a, args.label_b,
                                           args.draw_score_threshold, args.max_side)
                video_id = row_a['video_id']
                if video_id != current:
                    if writer is not None:
                        writer.release()
                    current, shape = video_id, canvas.shape[:2]
                    filename = f'video_{video_id}_comparison.mp4'
                    writer = cv2.VideoWriter(str(output / 'videos' / filename),
                                             cv2.VideoWriter_fourcc(*'mp4v'), args.playback_fps,
                                             (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Could not open mp4v comparison writer.')
                    indices.append({'video_id': video_id, 'path': 'videos/' + filename, 'num_frames': 0,
                                    'encoded_size': [canvas.shape[1], canvas.shape[0]]})
                if canvas.shape[:2] != shape:
                    raise ValueError('Frame dimensions changed within a video.')
                writer.write(canvas)
                count = indices[-1]['num_frames']
                if args.save_frames_every and count % args.save_frames_every == 0:
                    path = output / 'frames' / f'video_{video_id}_image_{row_a["image_id"]}.jpg'
                    if not cv2.imwrite(str(path), canvas):
                        raise IOError(f'Could not save {path}')
                indices[-1]['num_frames'] += 1
                seen.append(row_a['image_id'])
        if writer is not None:
            writer.release()
            writer = None
        if seen != metrics[0]['evaluated_image_ids']:
            raise ValueError('Detection rows do not match the evaluated frame IDs.')
        save_results({'source_runs': [str(a.resolve()), str(b.resolve())],
                      'mAP@0.5': {args.label_a: metrics[0]['mAP@0.5'], args.label_b: metrics[1]['mAP@0.5']},
                      'verified_comparison_fields': list(COMPARISON_KEYS),
                      'draw_score_threshold': args.draw_score_threshold, 'num_frames': len(seen),
                      'playback_fps': args.playback_fps, 'inference_rerun': False,
                      'layout': 'ground truth | model A predictions | model B predictions'}, output / 'comparison.json')
        with (output / 'per_class_comparison.csv').open('x', newline='') as stream:
            table = csv.writer(stream)
            table.writerow(['class_name', args.label_a + '_AP50_percent', args.label_b + '_AP50_percent'])
            for label, name in sorted(names.items()):
                values = [item['AP@0.5_per_class'].get(str(label)) for item in metrics]
                table.writerow([name] + ['NA' if value is None else 100 * value for value in values])
        save_results({'videos': indices}, output / 'videos_index.json')
        save_results({'status': 'complete', 'num_frames': len(seen)}, output / 'run_status.json')
    except BaseException as error:
        if writer is not None:
            writer.release()
        save_results({'status': 'failed', 'error': str(error)}, output / 'run_status.json')
        raise
    print(f'Saved matched comparison: {output}')


if __name__ == '__main__':
    main()
