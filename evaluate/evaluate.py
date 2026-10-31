from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import parse_paths

import argparse

import torch

from config import ModelConfig, SSDConfig, TrainConfig
from datasets.imagenet_vid import ImageNetVIDDataset
from datasets.transforms import EvalTransform
from models.lstm_ssd import LSTMSSD
from engine.evaluator import evaluate_vid
from utils.compute import count_parameters, compute_macs_per_frame
from losses.multibox_loss import MultiBoxLoss
from utils.runtime import select_video_subset, select_validation_frames, load_checkpoint, tqdm
from utils.checkpoints import model_from_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LSTM-SSD sequentially on ImageNet VID.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--frames-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--frames-per-video", type=int, help="Use 20 for the paper minival segment protocol.")
    p.add_argument("--no-progress", action="store_true")
    return parse_paths(p, "evaluate")


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(checkpoint)
    model_cfg, ssd_cfg = model.model_cfg, model.ssd_cfg
    dataset = ImageNetVIDDataset(args.annotations, args.frames_root,
                                 transform=EvalTransform(model_cfg.image_size, interpolation=model_cfg.input_interpolation),
                                 clip_length=1, clip_stride=1,)
    select_video_subset(dataset, args.max_videos, args.subset_seed)
    select_validation_frames(dataset, args.frames_per_video, args.subset_seed)
    tqdm.write(f"eval: {len(dataset.video_ids)} videos, "
               f"{sum(len(dataset.images_by_video[v]) for v in dataset.video_ids)} frames")
    if (dataset.category_id_to_label != checkpoint['category_id_to_label'] or
            dataset.label_to_name != checkpoint['label_to_name']):
        raise ValueError('Evaluation category definitions differ from checkpoint.')
    tqdm.write(f"Loaded {checkpoint['stage']} checkpoint at epoch {checkpoint['epoch']}")

    train_cfg = TrainConfig(**checkpoint['train_config'])
    criterion = MultiBoxLoss(
        num_classes=model_cfg.num_classes,
        match_iou_threshold=ssd_cfg.match_iou_threshold,
        max_negatives_per_positive=train_cfg.max_negatives_per_positive,
        negative_loss_weight=train_cfg.negative_loss_weight,
    )
    metrics = evaluate_vid(model, dataset, args.device,
                           score_threshold=0.0,
                           nms_iou_threshold=ssd_cfg.nms_iou_threshold,
                           criterion=criterion, progress=not args.no_progress)
    params = count_parameters(model)
    macs = compute_macs_per_frame(model, model_cfg.image_size, args.device)
    print(metrics)
    print(f"parameters={params:,}")
    print(f"MAC/frame={macs / 1e9:.4f} G")


if __name__ == "__main__":
    main()
