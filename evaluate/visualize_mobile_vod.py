import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import parse_paths
import argparse
from dataclasses import asdict
import hashlib
import json
import platform

import cv2
import torch

from config import TrainConfig
from datasets.imagenet_vid import ImageNetVIDDataset
from datasets.transforms import EvalTransform
from engine.evaluator import evaluate_vid
from engine.visualizer import VideoExporter
from utils.results import save_results
from losses.multibox_loss import MultiBoxLoss
from models.anchors import BoxCoderScales
from utils.checkpoints import model_from_checkpoint
from utils.compute import count_parameters, compute_macs_per_frame
from utils.runtime import load_checkpoint, select_video_subset, select_validation_frames, tqdm
from utils.results import write_accuracy_tables


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and visualize native stage-1 or stage-2 checkpoints.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--frames-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--subset-seed', type=int, default=7)
    parser.add_argument('--frames-per-video', type=int)
    parser.add_argument('--draw-score-threshold', type=float, default=.3)
    parser.add_argument('--layout', choices=['side-by-side', 'overlay'], default='side-by-side')
    parser.add_argument('--playback-fps', type=float, default=30)
    parser.add_argument('--max-side', type=int, default=960)
    parser.add_argument('--save-frames-every', type=int, default=30)
    parser.add_argument('--score-threshold', type=float, default=0)
    parser.add_argument('--nms-iou-threshold', type=float, default=.5)
    parser.add_argument('--max-detections-per-class', type=int, default=100)
    parser.add_argument('--max-total-detections', type=int, default=100)
    parser.add_argument('--warmup-frames', type=int, default=3)
    parser.add_argument('--no-loss', action='store_true', help='Skip loss computation; AP is unchanged.')
    parser.add_argument('--no-progress', action='store_true')
    return parse_paths(parser, "visualize_mobile_vod")


def main():
    args = parse_args()
    if args.warmup_frames < 0 or not 0 <= args.draw_score_threshold <= 1 or args.max_side < 1:
        raise ValueError('Invalid warmup/drawing settings.')
    payload = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(payload)
    dataset = ImageNetVIDDataset(args.annotations, args.frames_root,
                                transform=EvalTransform(model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation), clip_length=1)
    select_video_subset(dataset, args.max_videos, args.subset_seed)
    select_validation_frames(dataset, args.frames_per_video, args.subset_seed)
    if (dataset.category_id_to_label != payload['category_id_to_label'] or
            dataset.label_to_name != payload['label_to_name'] or
            dataset.num_classes != model.model_cfg.num_classes):
        raise ValueError('Evaluation category definitions differ from checkpoint.')
    scales = model.ssd_cfg
    if (scales.y_scale, scales.x_scale, scales.height_scale, scales.width_scale) != (10, 10, 5, 5):
        raise ValueError('The existing evaluator uses box-coder scales 10/10/5/5; this checkpoint differs.')
    criterion = None
    if not args.no_loss:
        cfg = TrainConfig(**payload['train_config'])
        criterion = MultiBoxLoss(num_classes=model.model_cfg.num_classes,
                                 match_iou_threshold=scales.match_iou_threshold,
                                 max_negatives_per_positive=cfg.max_negatives_per_positive,
                                 negative_loss_weight=cfg.negative_loss_weight)
    stage = payload['stage']
    model.to(args.device).eval()
    with torch.no_grad():
        state = None
        for _ in range(args.warmup_frames):
            _, state = model.forward_step(torch.zeros(1, 3, model.model_cfg.image_size,
                                                      model.model_cfg.image_size, device=args.device),
                                           state, return_anchors=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    exporter = None
    try:
        exporter = VideoExporter(dataset, output_dir, stage, args.draw_score_threshold,
                                     args.layout, args.playback_fps, args.max_side,
                                     args.save_frames_every, class_names=dataset.label_to_name)
        metrics = evaluate_vid(model, dataset, args.device, args.score_threshold,
                               args.nms_iou_threshold, args.max_detections_per_class,
                               args.max_total_detections, criterion, not args.no_progress,
                               description=f'{stage} inference', prediction_callback=exporter,
                               per_video_metrics=True)
        exporter.close()
        metrics.update({'stage': stage, 'variant': stage, 'checkpoint_epoch': payload.get('epoch'),
                        'checkpoint_path': str(Path(args.checkpoint).resolve()),
                        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                        'model_config': asdict(model.model_cfg), 'ssd_config': asdict(model.ssd_cfg),
                        'category_id_to_label': dataset.category_id_to_label,
                        'box_coder_scales': asdict(BoxCoderScales()), 'score_activation': 'sigmoid',
                        'preprocessing': 'PIL RGB, torchvision bilinear resize with antialias, pixel/255*2-1',
                        'annotations_sha256': hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                        'loss_computed': criterion is not None,
                        'state_policy': 'none' if stage == 'stage1' else 'carry within video; reset at each video boundary'})
        metrics['parameters'] = count_parameters(model)
        metrics['MACs_per_frame_conv_linear'] = compute_macs_per_frame(model, model.model_cfg.image_size, args.device)
        metrics['FLOPs_per_frame_conv_linear'] = 2 * metrics['MACs_per_frame_conv_linear']
        metrics['compute_scope'] = 'Conv2d/Linear only; FLOPs=2*MACs; recurrent elementwise operations and NMS excluded.'
        metrics['runtime'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'opencv': cv2.__version__, 'device': args.device, 'precision': 'float32'}
        metrics['visualization'] = {'draw_score_threshold': args.draw_score_threshold,
                                     'layout': args.layout, 'playback_fps': args.playback_fps,
                                     'max_side_per_panel': args.max_side}
        source = json.loads(Path(args.annotations).read_text())
        if 'sample_selection' in source:
            metrics['sample_selection'] = source['sample_selection']
        save_results(metrics, output_dir / 'metrics.json')
        save_results({'videos': exporter.video_index}, output_dir / 'videos_index.json')
        write_accuracy_tables(metrics, output_dir)
        save_results({'status': 'complete', 'num_frames': metrics['num_frames']}, output_dir / 'run_status.json')
    except BaseException as error:
        if exporter is not None:
            exporter.close()
        if not (output_dir / 'run_status.json').exists():
            save_results({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)},
                         output_dir / 'run_status.json')
        raise
    tqdm.write(f'{stage}: mAP@50 = {100 * metrics["mAP@0.5"]:.4f}%; outputs: {output_dir}')


if __name__ == '__main__':
    main()
