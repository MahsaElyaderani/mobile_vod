from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch



def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed, *keys):
    data = json.dumps([int(seed), *keys], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little') % (2 ** 63 - 1)


def detection(boxes=(), scores=(), labels=(), num_classes=30):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().reshape(-1, 4)
    scores = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().reshape(-1)
    raw_labels = torch.as_tensor(labels).detach().cpu().reshape(-1)
    if raw_labels.numel() and (not torch.isfinite(raw_labels).all() or
                               not torch.equal(raw_labels, raw_labels.round())):
        raise ValueError('Detection labels must be finite integers.')
    labels = raw_labels.to(torch.long)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError('Detection boxes, scores and labels must align.')
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError('Non-finite detection data.')
    if ((boxes < 0) | (boxes > 1)).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Boxes must be normalized xyxy; scores must be in [0,1].')
    if len(boxes) and ((boxes[:, 2:] < boxes[:, :2]).any() or
                       (labels < 1).any() or (labels > num_classes).any()):
        raise ValueError('Reversed box corners or out-of-range foreground labels.')
    return {'boxes': boxes.clone(), 'scores': scores.clone(), 'labels': labels.clone()}


def validated(prediction, num_classes=30):
    return detection(prediction['boxes'], prediction['scores'], prediction['labels'], num_classes)


def serialize(prediction):
    prediction = validated(prediction)
    return {key: value.tolist() for key, value in prediction.items()}


def filtered(prediction, threshold=0.05, maximum=100):
    if not 0 <= threshold <= 1 or maximum < 1:
        raise ValueError('Invalid detection filter.')
    result = validated(prediction)
    keep = torch.nonzero(result['scores'] >= threshold).flatten()
    order = torch.argsort(result['scores'][keep], descending=True, stable=True)
    keep = keep[order[:maximum]]
    return {key: value[keep] for key, value in result.items()}
