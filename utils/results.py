"""Shared result serialization and accuracy tables."""
import csv
import json
import os
import tempfile
from pathlib import Path

def save_results(metrics, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)

def write_accuracy_tables(metrics, output_dir):
    with (output_dir / 'per_class_ap.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset_label', 'class_name', 'ground_truth_objects',
                         'ground_truth_frames', 'AP50', 'AP50_percent'])
        for label, name in sorted(metrics['class_names'].items()):
            ap = metrics['AP@0.5_per_class'].get(label)
            writer.writerow([label, name, metrics['ground_truth_counts_per_class'][label],
                             metrics['ground_truth_frames_per_class'][label],
                             'NA' if ap is None else ap, 'NA' if ap is None else 100 * ap])
    with (output_dir / 'per_video_ap.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['video_id', 'num_frames', 'evaluated_classes', 'mAP50', 'mAP50_percent'])
        for video in metrics['per_video_metrics']:
            writer.writerow([video['video_id'], video['num_frames'], video['evaluated_classes'],
                             video['mAP@0.5'], 100 * video['mAP@0.5']])
