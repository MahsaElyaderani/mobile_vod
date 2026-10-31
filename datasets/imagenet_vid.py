from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

from .transforms import EvalTransform, Target


class ImageNetVIDDataset(Dataset):
    """ JSON Annotations contain: ``videos``, ``images``, ``annotations`` and ``categories``.
    Image records must contain ``video_id`` and preferably ``frame_id``.

    Boxes are read from COCO ``[x, y, width, height]`` pixels, converted to
    absolute xyxy, then the transform returns normalized xyxy boxes.
    Foreground labels exposed to the detector are contiguous 1..K; 0 is
    reserved for background.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        frames_root: str | Path,
        transform=None,
        clip_length: int = 10,
        clip_stride: int = 1,
        drop_incomplete: bool = True,
    ) -> None:
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.frames_root = Path(frames_root)
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.drop_incomplete = bool(drop_incomplete)
        if self.clip_length <= 0 or self.clip_stride <= 0:
            raise ValueError("clip_length and clip_stride must be positive.")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(self.annotation_file)
        if not self.frames_root.is_dir():
            raise FileNotFoundError(self.frames_root)

        with self.annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("videos", "images", "annotations", "categories"):
            if key not in data:
                raise ValueError(f"Annotation JSON is missing required key: {key}")

        categories = sorted(data["categories"], key=lambda x: int(x["id"]))
        self.category_id_to_label: Dict[int, int] = {
            int(cat["id"]): idx + 1 for idx, cat in enumerate(categories)
        }
        self.label_to_category_id: Dict[int, int] = {
            label: cat_id for cat_id, label in self.category_id_to_label.items()
        }
        self.label_to_name: Dict[int, str] = {
            idx + 1: str(cat.get("name", cat["id"])) for idx, cat in enumerate(categories)
        }

        self.videos: Dict[int, dict] = {int(v["id"]): v for v in data["videos"]}
        images_by_video: Dict[int, List[dict]] = defaultdict(list)
        for image in data["images"]:
            video_id = int(image["video_id"])
            if video_id in self.videos:
                images_by_video[video_id].append(image)
        self.images_by_video: Dict[int, List[dict]] = {}
        for video_id, images in images_by_video.items():
            self.images_by_video[video_id] = sorted(
                images,
                key=lambda r: (int(r.get("frame_id", 0)), int(r["id"])),
            )
        self.video_ids = sorted(self.images_by_video)

        anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if bool(ann.get("iscrowd", False)):
                continue
            anns_by_image[int(ann["image_id"])].append(ann)
        self.anns_by_image = dict(anns_by_image)

        self.transform = transform or EvalTransform(image_size=320)
        self.clip_index: List[Tuple[int, int]] = []
        for video_id in self.video_ids:
            records = self.images_by_video[video_id]
            n = len(records)
            if n >= self.clip_length:
                for start in range(0, n - self.clip_length + 1, self.clip_stride):
                    window = records[start:start + self.clip_length]
                    # ImageNet-VID JSONs normally provide a frame_id for every
                    # frame. When present, require an actual consecutive run
                    # rather than merely ten adjacent JSON records.
                    frame_ids = [r.get("frame_id") for r in window]
                    if all(fid is not None for fid in frame_ids):
                        ints = [int(fid) for fid in frame_ids]
                        if any(b != a + 1 for a, b in zip(ints, ints[1:])):
                            continue
                    self.clip_index.append((video_id, start))
            elif not self.drop_incomplete and n > 0:
                self.clip_index.append((video_id, 0))

    @property
    def num_classes(self) -> int:
        return len(self.category_id_to_label)

    def __len__(self) -> int:
        return len(self.clip_index)

    def _load_image_and_target(self, record: dict) -> Tuple[Image.Image, Target]:
        image_path = self.frames_root / str(record["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        image_id = int(record["id"])
        boxes: List[List[float]] = []
        labels: List[int] = []
        category_ids: List[int] = []
        for ann in self.anns_by_image.get(image_id, []):
            category_id = int(ann["category_id"])
            if category_id not in self.category_id_to_label:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[category_id])
            category_ids.append(category_id)
        target: Target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "category_ids": torch.tensor(category_ids, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "video_id": torch.tensor(int(record["video_id"]), dtype=torch.long),
            "frame_id": torch.tensor(int(record.get("frame_id", 0)), dtype=torch.long),
        }
        return image, target

    def _records_to_clip(self, records: Sequence[dict]):
        images: List[Image.Image] = []
        targets: List[Target] = []
        for record in records:
            image, target = self._load_image_and_target(record)
            images.append(image)
            targets.append(target)
        return self.transform(images, targets)

    def __getitem__(self, index: int):
        video_id, start = self.clip_index[index]
        records = self.images_by_video[video_id]
        selected = records[start:start + self.clip_length]
        if len(selected) < self.clip_length:
            if self.drop_incomplete:
                raise RuntimeError("Internal clip index produced an incomplete clip.")
            # Repeat the final frame only when explicitly requested.
            selected = selected + [selected[-1]] * (self.clip_length - len(selected))
        return self._records_to_clip(selected)

    def iter_video(self, video_id: int) -> Iterator[Tuple[Tensor, Target]]:
        """Yield one transformed frame at a time in temporal order.

        This is intentionally separate from ``__getitem__`` so evaluation can
        maintain one recurrent state across an entire video and reset it only
        at video boundaries.
        """
        if video_id not in self.images_by_video:
            raise KeyError(f"Unknown video_id={video_id}")
        for record in self.images_by_video[video_id]:
            frames, targets = self._records_to_clip([record])
            yield frames[0], targets[0]
