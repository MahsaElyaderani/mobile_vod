import json
import math
from pathlib import Path

import cv2
import numpy as np

from vid_categories import VID_CLASSES, vid_to_dataset_labels


GT_COLOR = (60, 220, 60)
PRED_COLOR = (0, 170, 255)


def draw_box(image, normalized_box, caption, color):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = normalized_box
    x1, x2 = np.clip([x1 * width, x2 * width], 0, width - 1).round().astype(int)
    y1, y2 = np.clip([y1 * height, y2 * height], 0, height - 1).round().astype(int)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(0, y1 - text_height - baseline - 4)
    cv2.rectangle(image, (x1, top), (min(width - 1, x1 + text_width + 4), top + text_height + baseline + 4), color, -1)
    cv2.putText(image, caption, (x1 + 2, top + text_height + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def captioned(image, caption):
    header = np.full((36, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(header, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def render_frame(bgr, target, prediction, class_names, variant,
                 draw_score_threshold=0.3, layout='side-by-side', max_side=960,
                 model_only_titles=False):
    if layout not in ('side-by-side', 'overlay'):
        raise ValueError('Unknown rendering layout.')
    if max_side < 1 or not 0 <= draw_score_threshold <= 1:
        raise ValueError('Invalid visualization dimensions/threshold.')
    factor = min(1.0, max_side / max(bgr.shape[:2]))
    if factor < 1:
        bgr = cv2.resize(bgr, (max(1, round(bgr.shape[1] * factor)),
                              max(1, round(bgr.shape[0] * factor))))
    ground_truth, predicted = bgr.copy(), bgr.copy()
    for box, label in zip(target['boxes'].tolist(), target['labels'].tolist()):
        draw_box(ground_truth, box, class_names[label], GT_COLOR)
    if layout == 'overlay':
        predicted = ground_truth.copy()
    shown = 0
    for box, label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(),
                                 prediction['scores'].tolist()):
        if score >= draw_score_threshold:
            draw_box(predicted, box, f'{class_names[label]} {score:.2f}', PRED_COLOR)
            shown += 1
    video_id, frame_id = int(target['video_id']), int(target['frame_id'])
    title = f'{variant} v{video_id} f{frame_id}'
    predicted = captioned(predicted, variant if model_only_titles else
                          f'{title} | score >= {draw_score_threshold:.2f} | {shown} boxes')
    if layout == 'side-by-side':
        ground_truth = captioned(ground_truth, 'Ground truth' if model_only_titles else
                                 f'Ground truth | video {video_id} frame {frame_id}')
        canvas = np.concatenate([ground_truth, predicted], axis=1)
    else:
        canvas = predicted
    return cv2.copyMakeBorder(canvas, 0, canvas.shape[0] % 2, 0, canvas.shape[1] % 2,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


class VideoExporter:
    def __init__(self, dataset, output_dir, variant, draw_score_threshold=0.3,
                 layout='side-by-side', playback_fps=30.0, max_side=960,
                 save_frames_every=0, class_names=None):
        if not math.isfinite(playback_fps) or playback_fps <= 0 or save_frames_every < 0:
            raise ValueError('playback_fps must be positive; save_frames_every must be non-negative.')
        self.dataset, self.variant = dataset, variant
        self.output_dir = Path(output_dir)
        self.video_dir = self.output_dir / 'videos'
        self.video_dir.mkdir()
        self.frame_dir = self.output_dir / 'frames'
        if save_frames_every:
            self.frame_dir.mkdir()
        self.records = {int(record['id']): record for video_id in dataset.video_ids
                        for record in dataset.images_by_video[video_id]}
        if class_names is None:
            mapping = vid_to_dataset_labels(dataset)
            self.class_names = {label: VID_CLASSES[fork_label - 1][0]
                                for fork_label, label in mapping.items()}
        else:
            self.class_names = dict(class_names)
        self.draw_threshold, self.layout = draw_score_threshold, layout
        self.playback_fps, self.max_side = playback_fps, max_side
        self.save_frames_every = save_frames_every
        self.writer, self.video_id, self.shape = None, None, None
        self.video_index = []
        self.detection_stream = (self.output_dir / 'detections.jsonl').open('x')

    def close_video(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def close(self):
        self.close_video()
        self.detection_stream.close()

    def __call__(self, video_id, target, prediction):
        image_id = int(target['image_id'])
        record = self.records[image_id]
        bgr = cv2.imread(str(self.dataset.frames_root / record['file_name']))
        if bgr is None:
            raise ValueError(f'Could not decode image {record["file_name"]}')
        canvas = render_frame(bgr, target, prediction, self.class_names, self.variant,
                              self.draw_threshold, self.layout, self.max_side)
        if video_id != self.video_id:
            self.close_video()
            self.video_id = video_id
            self.shape = canvas.shape[:2]
            filename = f'video_{video_id}_{self.variant}.mp4'
            self.writer = cv2.VideoWriter(str(self.video_dir / filename),
                                          cv2.VideoWriter_fourcc(*'mp4v'), self.playback_fps,
                                          (canvas.shape[1], canvas.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError('OpenCV could not open the mp4v video writer. Check codec availability.')
            self.video_index.append({'video_id': video_id, 'path': 'videos/' + filename,
                                     'num_frames': 0, 'encoded_size': [canvas.shape[1], canvas.shape[0]],
                                     'playback_fps': self.playback_fps, 'codec': 'mp4v'})
        if canvas.shape[:2] != self.shape:
            raise ValueError(f'Frame dimensions changed within video {video_id}.')
        self.writer.write(canvas)
        count = self.video_index[-1]['num_frames']
        if self.save_frames_every and count % self.save_frames_every == 0:
            filename = self.frame_dir / f'video_{video_id}_image_{image_id}.jpg'
            if not cv2.imwrite(str(filename), canvas):
                raise IOError(f'Could not save {filename}')
        self.video_index[-1]['num_frames'] += 1
        height, width = bgr.shape[:2]
        entries = []
        for box, label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(),
                                     prediction['scores'].tolist()):
            entries.append({'dataset_label': label, 'class_name': self.class_names[label],
                            'score': score, 'box_xyxy_normalized': box,
                            'box_xyxy_pixels': [box[0] * width, box[1] * height,
                                                box[2] * width, box[3] * height]})
        self.detection_stream.write(json.dumps({
            'video_id': video_id, 'frame_id': int(target['frame_id']), 'image_id': image_id,
            'file_name': record['file_name'], 'original_size': [width, height],
            'predictions': entries,
            'ground_truth': [{'dataset_label': label, 'class_name': self.class_names[label],
                              'box_xyxy_normalized': box}
                             for box, label in zip(target['boxes'].tolist(), target['labels'].tolist())],
        }, allow_nan=False) + '\n')


class PairedVideoExporter:
    def __init__(self, dataset, output_dir, variant, draw_score_threshold=.3, max_side=640,
                 playback_fps=30., save_frames_every=30):
        if (not 0 <= draw_score_threshold <= 1 or max_side < 1 or save_frames_every < 0 or
                not math.isfinite(playback_fps) or playback_fps <= 0):
            raise ValueError("Invalid visualization settings.")
        self.dataset, self.output, self.variant = dataset, Path(output_dir), variant
        self.threshold, self.max_side, self.fps, self.every = draw_score_threshold, max_side, playback_fps, save_frames_every
        (self.output / "videos").mkdir()
        if self.every:
            (self.output / "frames").mkdir()
        self.writer, self.current, self.shape, self.video_index = None, None, None, []

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __call__(self, record, target, prediction, route):
        image = cv2.imread(str(self.dataset.frames_root / record["file_name"]))
        if image is None or image.shape[:2] != (int(record["height"]), int(record["width"])):
            raise ValueError("Visualization frame missing or dimensions differ.")
        label = f"{self.variant}:{route['selected_model']} age={route['previous_contribution_age']}"
        canvas = render_frame(image, target, prediction, self.dataset.label_to_name, label,
                              self.threshold, "side-by-side", self.max_side)
        if self.current != int(record["video_id"]):
            self.close()
            self.current, self.shape = int(record["video_id"]), canvas.shape[:2]
            filename = f"video_{self.current}_{self.variant}.mp4"
            self.writer = cv2.VideoWriter(str(self.output / "videos" / filename), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (canvas.shape[1], canvas.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError("Could not open mp4v video writer.")
            self.video_index.append({"video_id": self.current, "path": "videos/" + filename,
                                     "num_frames": 0, "encoded_size": [canvas.shape[1], canvas.shape[0]]})
        if canvas.shape[:2] != self.shape:
            raise ValueError("Frame dimensions changed within a video.")
        self.writer.write(canvas)
        count = self.video_index[-1]["num_frames"]
        if self.every and count % self.every == 0:
            path = self.output / "frames" / f"video_{self.current}_image_{int(record['id'])}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise IOError(f"Could not save {path}.")
        self.video_index[-1]["num_frames"] += 1
