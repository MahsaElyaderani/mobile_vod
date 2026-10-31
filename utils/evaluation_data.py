"""Frame loading, target normalization and detection export for current VID evaluation."""
import torch
from PIL import Image

from datasets.transforms import resize_detection_views


def load_frame(dataset, record, image_size, interpolation="bilinear", paired=False):
    with Image.open(dataset.frames_root / record['file_name']) as image:
        if image.size != (int(record['width']), int(record['height'])):
            raise ValueError('Annotation dimensions differ from decoded small-model frame.')
        return resize_detection_views(image, image_size, interpolation, paired)


def normalized_target(dataset, record):
    boxes, labels = [], []
    for ann in dataset.anns_by_image.get(int(record['id']), []):
        category = int(ann['category_id'])
        x, y, width, height = map(float, ann['bbox'])
        if category not in dataset.category_id_to_label or width <= 0 or height <= 0:
            continue
        boxes.append([x, y, x + width, y + height])
        labels.append(dataset.category_id_to_label[category])
    absolute_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    normalized_boxes = (absolute_boxes / absolute_boxes.new_tensor(
        [record['width'], record['height'], record['width'], record['height']])).clamp(0, 1)
    return {'boxes': normalized_boxes,
            'labels': torch.tensor(labels, dtype=torch.long), 'image_id': torch.tensor(int(record['id'])),
            'video_id': torch.tensor(int(record['video_id'])), 'frame_id': torch.tensor(int(record['frame_id']))}


def raw_detection_row(record, target, prediction, names):
    width, height = int(record['width']), int(record['height'])
    return {'video_id': int(record['video_id']), 'frame_id': int(record['frame_id']), 'image_id': int(record['id']),
            'file_name': record['file_name'], 'original_size': [width, height],
            'predictions': [{'dataset_label': int(label), 'class_name': names[int(label)], 'score': float(score),
                              'box_xyxy_normalized': box,
                              'box_xyxy_pixels': [box[0]*width, box[1]*height, box[2]*width, box[3]*height]}
                             for box, label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(), prediction['scores'].tolist())],
            'ground_truth': [{'dataset_label': int(label), 'class_name': names[int(label)], 'box_xyxy_normalized': box}
                             for box, label in zip(target['boxes'].tolist(), target['labels'].tolist())]}
