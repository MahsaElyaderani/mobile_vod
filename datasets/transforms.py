from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as TF
from .paired_frames import PairedFrames, stack_frames


Target = dict[str, object]


def _normalize_boxes_xyxy(boxes: Tensor, width: int, height: int) -> Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)
    scale = boxes.new_tensor([width, height, width, height])
    return (boxes.to(dtype=torch.float32) / scale).clamp(0.0, 1.0)


def _box_iou_xyxy(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-8)


def _prepare_targets_normalized(
    images: Sequence[Image.Image], targets: Sequence[Target]
) -> List[Target]:
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length.")
    out: List[Target] = []
    for image, target in zip(images, targets):
        copied = deepcopy(target)
        boxes = copied.get("boxes")
        if not isinstance(boxes, Tensor):
            raise TypeError("Each target['boxes'] must be a torch.Tensor in absolute xyxy pixels.")
        copied["boxes"] = _normalize_boxes_xyxy(boxes, image.width, image.height)
        labels = copied.get("labels")
        if not isinstance(labels, Tensor):
            raise TypeError("Each target['labels'] must be a torch.Tensor.")
        copied["labels"] = labels.to(dtype=torch.long)
        copied["orig_size"] = torch.tensor([image.height, image.width], dtype=torch.long)
        out.append(copied)
    return out


def _resize_and_normalize_image(image: Image.Image, image_size: int, interpolation="bilinear") -> Tensor:
    # The authors' LSTM-SSD TensorFlow feature extractor maps [0,255] -> [-1,1].
    resized = TF.resize(image.convert("RGB"), [image_size, image_size], interpolation=TF.InterpolationMode(interpolation),
                        antialias=True)
    tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32) / 255.0
    return tensor.mul(2.0).sub(1.0)


def resize_detection_views(image, image_size=320, interpolation="bilinear", paired=False):
    """Resize each branch directly from one RGB image; never resize the big view again."""
    image = image.convert("RGB")
    big = _resize_and_normalize_image(image, image_size, interpolation)
    if not paired:
        return big
    if image_size != 320 or interpolation != "bicubic":
        raise ValueError("Paired timm views require bicubic resizing to 320 and 160.")
    return PairedFrames(big, _resize_and_normalize_image(image, 160, interpolation))


class EvalTransform:
    """Deterministic LSTM-SSD preprocessing.

    Images are warped to the fixed square resolution used by the paper's
    alpha=1 experiment. Boxes are returned as normalized xyxy coordinates;
    normalized coordinates remain valid under this fixed-shape warp.
    """

    def __init__(self, image_size: int = 320, interpolation="bilinear", paired=False) -> None:
        self.paired = paired
        self.interpolation = interpolation
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

    def __call__(
        self, images: Sequence[Image.Image], targets: Sequence[Target]
    ) -> Tuple[Tensor, List[Target]]:
        if not images:
            raise ValueError("A clip must contain at least one image.")
        normalized_targets = _prepare_targets_normalized(images, targets)
        frame_tensors = [resize_detection_views(im, self.image_size, self.interpolation, self.paired) for im in images]
        for target in normalized_targets:
            target["size"] = torch.tensor([self.image_size, self.image_size], dtype=torch.long)
        return stack_frames(frame_tensors, dim=0), normalized_targets


class TrainTransform:
    """SSD-style random crop + horizontal flip for a video clip.

    The CVPR 2018 paper states that SSD data augmentation is used. This class
    implements the standard SSD crop idea (sample an IoU-constrained patch)
    and a 0.5 horizontal flip. One geometric transform is shared across all
    frames of a clip so temporal correspondence is preserved. Sharing the
    random parameters across time is an explicit PyTorch design choice; the
    paper does not specify this detail.
    """

    def __init__(
        self,
        image_size: int = 320,
        flip_prob: float = 0.5,
        min_iou_options: Sequence[float | None] = (None, 0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_scale: float = 0.3,
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
        max_crop_trials: int = 50,
        interpolation="bilinear",
        paired=False,
    ) -> None:
        self.paired = paired
        self.interpolation = interpolation
        self.image_size = int(image_size)
        self.flip_prob = float(flip_prob)
        self.min_iou_options = tuple(min_iou_options)
        self.min_crop_scale = float(min_crop_scale)
        self.aspect_ratio_range = tuple(float(v) for v in aspect_ratio_range)
        self.max_crop_trials = int(max_crop_trials)
        if not 0.0 <= self.flip_prob <= 1.0:
            raise ValueError("flip_prob must be in [0,1].")
        if not 0.0 < self.min_crop_scale <= 1.0:
            raise ValueError("min_crop_scale must be in (0,1].")
        if self.aspect_ratio_range[0] <= 0 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError("Invalid aspect_ratio_range.")

    def _sample_crop(self, targets: Sequence[Target]) -> Tuple[float, float, float, float] | None:
        all_boxes = [t["boxes"] for t in targets if isinstance(t.get("boxes"), Tensor) and t["boxes"].numel()]
        if not all_boxes:
            return None
        boxes = torch.cat(all_boxes, dim=0)
        option = random.choice(self.min_iou_options)
        if option is None:
            return None

        for _ in range(self.max_crop_trials):
            crop_w = random.uniform(self.min_crop_scale, 1.0)
            crop_h = random.uniform(self.min_crop_scale, 1.0)
            ratio = crop_w / crop_h
            if ratio < self.aspect_ratio_range[0] or ratio > self.aspect_ratio_range[1]:
                continue
            left = random.uniform(0.0, 1.0 - crop_w)
            top = random.uniform(0.0, 1.0 - crop_h)
            crop = boxes.new_tensor([[left, top, left + crop_w, top + crop_h]])
            if _box_iou_xyxy(boxes, crop).max().item() < float(option):
                continue
            return left, top, left + crop_w, top + crop_h
        return None

    @staticmethod
    def _apply_crop_to_target(
        target: Target, crop: Tuple[float, float, float, float]
    ) -> Target:
        left, top, right, bottom = crop
        copied = deepcopy(target)
        boxes = copied["boxes"]
        labels = copied["labels"]
        assert isinstance(boxes, Tensor) and isinstance(labels, Tensor)
        if boxes.numel() == 0:
            copied["boxes"] = boxes.reshape(0, 4)
            copied["labels"] = labels.reshape(0)
            return copied

        centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        keep = (
            (centers[:, 0] >= left) & (centers[:, 0] <= right) &
            (centers[:, 1] >= top) & (centers[:, 1] <= bottom)
        )
        num_before = boxes.shape[0]
        boxes = boxes[keep].clone()
        for key in ("labels", "category_ids"):
            if key not in copied:
                continue
            value = copied[key]
            if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[0] != num_before:
                raise ValueError(f"target['{key}'] must align with the box dimension.")
            copied[key] = value[keep].clone()

        if boxes.numel():
            boxes[:, 0::2] = boxes[:, 0::2].clamp(left, right)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(top, bottom)
            crop_w = right - left
            crop_h = bottom - top
            boxes[:, 0::2] = (boxes[:, 0::2] - left) / crop_w
            boxes[:, 1::2] = (boxes[:, 1::2] - top) / crop_h
        copied["boxes"] = boxes
        return copied

    @staticmethod
    def _apply_flip_to_target(target: Target) -> Target:
        copied = deepcopy(target)
        boxes = copied["boxes"]
        assert isinstance(boxes, Tensor)
        if boxes.numel():
            flipped = boxes.clone()
            flipped[:, 0] = 1.0 - boxes[:, 2]
            flipped[:, 2] = 1.0 - boxes[:, 0]
            copied["boxes"] = flipped
        return copied

    def __call__(
        self, images: Sequence[Image.Image], targets: Sequence[Target]
    ) -> Tuple[Tensor, List[Target]]:
        if not images:
            raise ValueError("A clip must contain at least one image.")
        normalized_targets = _prepare_targets_normalized(images, targets)

        crop = self._sample_crop(normalized_targets)
        working_images = list(images)
        working_targets = normalized_targets
        if crop is not None:
            left, top, right, bottom = crop
            cropped_images: List[Image.Image] = []
            cropped_targets: List[Target] = []
            for image, target in zip(working_images, working_targets):
                x0 = int(round(left * image.width))
                y0 = int(round(top * image.height))
                x1 = int(round(right * image.width))
                y1 = int(round(bottom * image.height))
                x0 = max(0, min(x0, image.width - 1))
                y0 = max(0, min(y0, image.height - 1))
                x1 = max(x0 + 1, min(x1, image.width))
                y1 = max(y0 + 1, min(y1, image.height))
                cropped_images.append(TF.crop(image, y0, x0, y1 - y0, x1 - x0))
                # Use the exact pixel crop actually applied to this frame so
                # transformed boxes stay exactly aligned after rounding.
                actual_crop = (
                    x0 / image.width, y0 / image.height,
                    x1 / image.width, y1 / image.height,
                )
                cropped_targets.append(self._apply_crop_to_target(target, actual_crop))
            working_images = cropped_images
            working_targets = cropped_targets

        if random.random() < self.flip_prob:
            working_images = [TF.hflip(im) for im in working_images]
            working_targets = [self._apply_flip_to_target(t) for t in working_targets]

        frames = stack_frames(
            [resize_detection_views(im, self.image_size, self.interpolation, self.paired) for im in working_images], dim=0
        )
        for target in working_targets:
            target["size"] = torch.tensor([self.image_size, self.image_size], dtype=torch.long)
        return frames, working_targets
