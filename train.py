from __future__ import annotations
from utils.resume import resume_defaults, saved_arguments, restore_optimizer

from path import parse_paths, standalone_output
import argparse
import random
from pathlib import Path
from dataclasses import asdict, replace
import torch
from config import ModelConfig, SSDConfig, TrainConfig, model_config_dict, checkpoint_model_config
from datasets.imagenet_vid import ImageNetVIDDataset
from datasets.transforms import TrainTransform, EvalTransform
from datasets.dataloader import build_dataloader
from losses.multibox_loss import MultiBoxLoss
from models.ssd import MobileNetSSD
from models.lstm_ssd import LSTMSSD
from engine.trainer import train_detector_stage1, train_recurrent_stage2, freeze_backbone_through_conv13
from engine.evaluator import evaluate_vid
from utils.runtime import (tqdm, select_video_subset, select_validation_frames,
                           atomic_save_checkpoint, load_checkpoint, epoch_checkpoint_path)
from routing.predictions import sha256_file
from utils.checkpoints import FORMAT_VERSION, model_from_checkpoint, transfer_stage1, capture_rng, restore_rng


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train non-recurrent SSD or recurrent LSTM-SSD.')
    phases = p.add_mutually_exclusive_group(required=True)
    phases.add_argument('--phase', dest='stage', type=lambda value: {
        'standalone_pretrain': '1', 'standalone_detection': '2'}.get(value, value),
        choices=['1', '2'], metavar='{standalone_pretrain,standalone_detection}',
        help='Non-recurrent detection pretraining or recurrent detection with frozen backbone.')
    phases.add_argument('--stage', choices=['1', '2'], help='Legacy alias: 1=standalone_pretrain, 2=standalone_detection.')
    p.add_argument('--backbone', choices=('timm_mobilenet_v2',), default=None)
    p.add_argument('--image-size', type=int, choices=(160, 320), default=None)
    p.add_argument('--model-size', choices=('big', 'small'), help='New model role: big=320, small=160; defaults to small for width <1.')
    p.add_argument('--width-mult', type=float, default=None,
                   help='New stage-1 V2 width (0.5/1.0/1.1/1.2); otherwise inherited.')
    p.add_argument('--backbone-init', choices=('timm',), default=None,
                   help='New stage 1 uses official timm pretrained weights.')
    p.add_argument('--annotations', required=True)
    p.add_argument('--frames-root', required=True)
    p.add_argument('--epochs', type=int, required=True, help='Total epoch target, including resumed epochs.')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--clip-stride', type=int, default=1, help='Stage-2 training clip-start stride.')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--allow-worker-count-change', action='store_true',
                   help='Allow a different worker count on resume; augmentation RNG sequences may change.')
    p.add_argument('--prefetch-factor', type=int, default=1,
                   help='Batches prepared per worker; ignored when num-workers=0.')
    p.add_argument('--sharing-strategy', choices=('file_descriptor', 'file_system'),
                   help='CPU tensor transport; file_system avoids cached per-storage file descriptors.')
    p.add_argument('--cudnn-benchmark', action='store_true',
                   help='Enable cuDNN autotuning, matching the existing PyCharm trainer.')
    p.add_argument('--frozen-backbone-batch-size', type=int, default=32,
                   help='Stage-2 frozen-backbone frame batch limit; 0 keeps sequential extraction.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--checkpoint-in', help='Completed stage-1 checkpoint for transition into stage 2.')
    g.add_argument('--resume', help='Resume the same stage, including optimizer and RNG.')
    p.add_argument('--resume-best-checkpoint',
                   help='Import the historical best when resuming into a new output directory.')
    p.add_argument('--backbone-weights', default=None, help=argparse.SUPPRESS)
    p.add_argument('--checkpoint-out', help='Latest checkpoint; numbered epochs and best share its directory.')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--max-videos', type=int)
    p.add_argument('--subset-seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val-annotations')
    p.add_argument('--val-frames-root')
    p.add_argument('--max-val-videos', type=int)
    p.add_argument('--val-frames-per-video', type=int, help='Use 20 for the paper minival segment protocol.')
    p.add_argument('--no-progress', action='store_true')
    a = resume_defaults(p, parse_paths(p, "train", argv), argv)
    if a.epochs < 1 or a.batch_size < 2 or a.clip_stride < 1 or a.num_workers < 0:
        p.error('epochs/clip-stride must be positive, batch-size >= 2, num-workers >= 0')
    if a.prefetch_factor < 1 or a.frozen_backbone_batch_size < 0:
        p.error('prefetch-factor must be positive; frozen-backbone-batch-size must be non-negative')
    if a.learning_rate is not None and a.learning_rate <= 0:
        p.error('learning-rate must be positive')
    if bool(a.val_annotations) != bool(a.val_frames_root):
        p.error('Supply both validation paths together')
    if (a.max_val_videos is not None or a.val_frames_per_video is not None) and not a.val_annotations:
        p.error('Validation subset options require validation paths')
    if a.stage == '1' and a.checkpoint_in:
        p.error('Stage 1 uses official timm weights or --resume')
    if a.stage == '2' and not (a.checkpoint_in or a.resume):
        p.error('Stage 2 requires --checkpoint-in from stage 1 or --resume from stage 2')
    if a.backbone_init is not None and (a.resume or a.stage != '1'):
        p.error('--backbone-init is only for a new stage-1 run')
    if a.resume_best_checkpoint and not a.resume:
        p.error('--resume-best-checkpoint requires --resume')
    a.checkpoint_out = a.checkpoint_out or standalone_output(a.stage)
    return a


def _same_value(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _same_training_snapshot(left, right):
    fields = ('format_version', 'stage', 'epoch', 'global_step', 'model', 'optimizer',
              'model_config', 'ssd_config', 'train_config', 'training_signature',
              'validation_signature', 'category_id_to_label', 'label_to_name', 'rng',
              'pretrained_source', 'transfer_report', 'train_metrics')
    return all(_same_value(left.get(key), right.get(key)) for key in fields)


# def _same_training_signature(left, right, allow_worker_change=False):
#     if not allow_worker_change:
#         return _same_value(left, right)
#     if not isinstance(left, dict) or not isinstance(right, dict):
#         return _same_value(left, right)
#     return _same_value({k: v for k, v in left.items() if k != 'num_workers'},
#                        {k: v for k, v in right.items() if k != 'num_workers'})

def _same_training_signature(left, right, allow_worker_change=False, allow_batch_change=False):
    if not allow_worker_change and not allow_batch_change:
        return _same_value(left, right)
    if not isinstance(left, dict) or not isinstance(right, dict):
        return _same_value(left, right)

    if allow_worker_change:
        left = {k: v for k, v in left.items() if k != 'num_workers'}
        right = {k: v for k, v in right.items() if k != 'num_workers'}

    if allow_batch_change:
        left = {k: v for k, v in left.items() if k != 'batch_size'}
        right = {k: v for k, v in right.items() if k != 'batch_size'}

    return _same_value(left, right)


def _same_run_definition(left, right):
    fields = ('format_version', 'stage', 'model_config', 'ssd_config',
              'validation_signature', 'category_id_to_label',
              'label_to_name', 'pretrained_source', 'transfer_report')
    # A historical best may predate an explicit learning-rate change.
    configs_match = _same_value(
        {k: v for k, v in left.get('train_config', {}).items() if k != 'learning_rate'},
        {k: v for k, v in right.get('train_config', {}).items() if k != 'learning_rate'})
    return (configs_match and all(_same_value(left.get(key), right.get(key)) for key in fields)
            and _same_training_signature(left.get('training_signature'),
                                         right.get('training_signature'), True))


def _import_resume_best(resume, source, destination):
    best = load_checkpoint(source)
    if (not _same_run_definition(resume, best) or best.get('validation_pending') or
            best['epoch'] > resume['epoch'] or
            best.get('val_metrics', {}).get('mAP@0.5') != resume.get('best_map50')):
        raise ValueError('Historical best does not match the resumed run and saved best mAP.')
    if destination.exists():
        existing = load_checkpoint(destination)
        if not _same_training_snapshot(existing, best):
            raise FileExistsError('Refusing to replace a different historical best checkpoint.')
    else:
        atomic_save_checkpoint(best, destination, overwrite=False)


def _finish_validation(payload, metrics):
    previous_best = payload.get('best_map50')
    improved = previous_best is None or metrics['mAP@0.5'] > previous_best
    payload['val_metrics'] = metrics
    payload['best_map50'] = metrics['mAP@0.5'] if improved else previous_best
    payload.pop('validation_pending', None)
    return improved


def _recover_pending_validation(resume, model, val_dataset, criterion, device,
                                latest, best_path, epoch_target, progress):
    if val_dataset is None:
        raise ValueError('Pending validation requires the original validation paths and selection.')
    numbered = epoch_checkpoint_path(latest, resume['epoch'])
    existing = {}
    for path in (numbered, latest):
        if path.exists():
            stored = load_checkpoint(path)
            if not _same_training_snapshot(stored, resume):
                raise ValueError(f'Recovery destination contains a different training snapshot: {path}')
            existing[path] = stored
    completed = next((stored for stored in existing.values()
                      if not stored.get('validation_pending') and 'val_metrics' in stored), None)
    stored_best = None
    if best_path.exists():
        stored_best = load_checkpoint(best_path)
        if not _same_run_definition(stored_best, resume):
            raise ValueError('Recovery best-checkpoint destination belongs to a different run.')
        if stored_best.get('epoch', -1) > resume['epoch']:
            raise ValueError('Recovery would overwrite a best checkpoint from a later epoch; use a new output path.')
        if stored_best.get('epoch') == resume['epoch'] and not _same_training_snapshot(stored_best, resume):
            raise ValueError('Recovery best-checkpoint destination contains a different training snapshot.')
    payload = dict(resume)
    if completed is not None:
        metrics = completed['val_metrics']
    else:
        training_rng = capture_rng()
        try:
            metrics = evaluate_vid(model, val_dataset, device, criterion=criterion,
                                   score_threshold=0.0, nms_iou_threshold=model.ssd_cfg.nms_iou_threshold,
                                   progress=progress,
                                   description=f'{resume["stage"]} Val {resume["epoch"]}/{epoch_target} recovery (current/mean)')
        finally:
            restore_rng(training_rng)
    known_best = [stored.get('best_map50') for stored in (resume, completed, stored_best)
                  if stored is not None and stored.get('best_map50') is not None]
    payload['best_map50'] = max(known_best) if known_best else None
    improved = _finish_validation(payload, metrics)
    if improved or (not best_path.exists() and metrics['mAP@0.5'] == payload['best_map50']):
        atomic_save_checkpoint(payload, best_path)
    stored_numbered = existing.get(numbered)
    if stored_numbered is None or stored_numbered.get('validation_pending'):
        atomic_save_checkpoint(payload, numbered, overwrite=stored_numbered is not None)
    atomic_save_checkpoint(payload, latest)
    tqdm.write(f'Recovered validation for epoch {resume["epoch"]}: {metrics}')
    return payload


def resolve_model_config(width, checkpoint=None, image_size=None, backbone=None, model_size=None):
    """Keep serialized configuration authoritative on transfer and resume."""
    if checkpoint is not None:
        mc = checkpoint_model_config(checkpoint['model_config'])
        sc = SSDConfig(**checkpoint['ssd_config'])
        if backbone is not None and backbone != mc.backbone:
            raise ValueError('--backbone differs from the checkpoint.')
        if width is not None and width != mc.width_mult:
            raise ValueError('--width-mult differs from the checkpoint.')
        if model_size is not None and mc.image_size != (320 if model_size == 'big' else 160):
            raise ValueError('--model-size differs from the checkpoint resolution.')
        if image_size is not None and image_size != mc.image_size:
            raise ValueError('--image-size differs from the checkpoint.')
        return mc, sc
    width = 1.0 if width is None else width
    backbone = backbone or "timm_mobilenet_v2"
    if model_size is None and image_size is not None:
        model_size = 'small' if image_size == 160 else 'big'
    role = model_size or ('small' if width < 1 else 'big')
    size = 160 if role == 'small' else 320
    if image_size is not None and image_size != size:
        raise ValueError('--image-size conflicts with model role; use --model-size big/small explicitly.')
    return ModelConfig(width_mult=width, backbone=backbone, image_size=size,
                       mobilenet_output_channels=1280), SSDConfig()


def main():
    a = parse_args()
    stage = 'stage' + a.stage
    latest = Path(a.checkpoint_out)
    best_path = latest.with_name(latest.stem + '_best' + (latest.suffix or '.pt'))
    if not a.resume and (latest.exists() or best_path.exists()):
        raise FileExistsError('Use a new run directory or --resume; existing checkpoints are protected.')
    resume = load_checkpoint(a.resume) if a.resume else None
    if resume and latest.exists() and not _same_training_snapshot(load_checkpoint(latest), resume):
        raise ValueError('Output latest contains a different training snapshot; use its latest checkpoint or a new directory.')
    if resume and best_path.exists():
        existing_best = load_checkpoint(best_path)
        if not _same_run_definition(existing_best, resume) or existing_best['epoch'] > resume['epoch']:
            raise ValueError('Output best belongs to a different run or a later epoch; use a new directory.')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    initial = load_checkpoint(a.checkpoint_in) if a.checkpoint_in else None
    mc, sc = resolve_model_config(a.width_mult, resume if resume is not None else initial, image_size=a.image_size, backbone=a.backbone, model_size=a.model_size)
    tc = TrainConfig()
    if resume:
        if resume.get('stage') != stage:
            raise ValueError('Cannot resume another stage; use --checkpoint-in for stage 1 -> 2.')
        model = model_from_checkpoint(resume)
        mc, sc = model.model_cfg, model.ssd_cfg
        tc = TrainConfig(**resume['train_config'])
    else:
        model = MobileNetSSD(mc, sc) if a.stage == '1' else LSTMSSD(mc, sc)
    if a.learning_rate is not None:
        tc = replace(tc, learning_rate=a.learning_rate)
    clip_length = 1 if a.stage == '1' else mc.sequence_length
    clip_stride = 1 if a.stage == '1' else a.clip_stride
    dataset = ImageNetVIDDataset(a.annotations, a.frames_root, transform=TrainTransform(mc.image_size, interpolation=mc.input_interpolation),
                                clip_length=clip_length, clip_stride=clip_stride)
    select_video_subset(dataset, a.max_videos, a.subset_seed)
    if dataset.num_classes != mc.num_classes:
        raise ValueError(f'Expected {mc.num_classes} category definitions; keep the full list for subsets.')
    if len(dataset) < a.batch_size:
        raise ValueError('Selected training data must contain at least one full batch.')
    signature = {'annotation_sha256': sha256_file(a.annotations), 'train_video_ids': dataset.video_ids,
                 'batch_size': a.batch_size, 'clip_stride': clip_stride, 'seed': a.seed,
                 'num_workers': a.num_workers}
    if resume and (resume.get('category_id_to_label') != dataset.category_id_to_label or
                   resume.get('label_to_name') != dataset.label_to_name):
        raise ValueError('Checkpoint and training data category definitions differ.')
    if resume and not _same_training_signature(signature, resume['training_signature'],
                                               allow_worker_change=True, allow_batch_change=True,):
        raise ValueError('Resume data/subset/stride/seed settings differ from checkpoint.')

    worker_changes = list(resume.get('worker_count_changes', [])) if resume else []
    if resume and signature['num_workers'] != resume['training_signature'].get('num_workers'):
        worker_changes.append({'after_epoch': resume['epoch'],
                               'previous': resume['training_signature'].get('num_workers'),
                               'current': signature['num_workers']})
        tqdm.write('Worker count changed explicitly; weights and optimizer are restored, '
                   'but future random augmentation sequences may differ.')
    tqdm.write(f'{stage}: {len(dataset.video_ids)} videos, {len(dataset)} samples; '
               f'{len(dataset) // a.batch_size} optimizer steps/epoch; '
               f'dropping {len(dataset) % a.batch_size} samples/epoch')
    val_dataset = None
    if a.val_annotations:
        val_dataset = ImageNetVIDDataset(a.val_annotations, a.val_frames_root,
                                        transform=EvalTransform(mc.image_size, interpolation=mc.input_interpolation), clip_length=1)
        select_video_subset(val_dataset, a.max_val_videos, a.subset_seed)
        select_validation_frames(val_dataset, a.val_frames_per_video, a.subset_seed)
        if (val_dataset.category_id_to_label != dataset.category_id_to_label or
                val_dataset.label_to_name != dataset.label_to_name):
            raise ValueError('Training and validation category definitions differ.')
        tqdm.write(f'val: {len(val_dataset.video_ids)} videos, '
                   f'{sum(len(val_dataset.images_by_video[v]) for v in val_dataset.video_ids)} frames')
    validation_signature = ({'annotation_sha256': sha256_file(a.val_annotations),
                             'image_ids': {v: [r['id'] for r in val_dataset.images_by_video[v]]
                                           for v in val_dataset.video_ids}} if val_dataset else None)
    if resume and validation_signature != resume.get('validation_signature'):
        raise ValueError('Resume validation split/segments differ; keep the same validation set.')
    loader = build_dataloader(dataset, a.batch_size, shuffle=True, num_workers=a.num_workers,
                              drop_last=True, pin_memory=a.device.startswith('cuda'),
                              prefetch_factor=a.prefetch_factor, sharing_strategy=a.sharing_strategy)
    performance_config = {'prefetch_factor': a.prefetch_factor,
                          'frozen_backbone_batch_size': a.frozen_backbone_batch_size,
                          'sharing_strategy': a.sharing_strategy, 'cudnn_benchmark': a.cudnn_benchmark}
    tqdm.write(f'Performance settings: {performance_config}')
    report = resume.get('transfer_report') if resume else None
    if resume:
        pretrained_source = resume['pretrained_source']
    elif a.stage == '1':
        if a.backbone_init not in (None, 'timm'):
            raise ValueError('The timm backbone requires --backbone-init timm.')
        pretrained_source = model.backbone.load_pretrained()
        tqdm.write(f'Loaded every backbone tensor: {pretrained_source}')
    else:
        if (initial.get('category_id_to_label') != dataset.category_id_to_label or
                initial.get('label_to_name') != dataset.label_to_name):
            raise ValueError('Stage-1 checkpoint and current data category definitions differ.')
        report = transfer_stage1(model, initial)
        pretrained_source = initial['pretrained_source']
        report['source_checkpoint_sha256'] = sha256_file(a.checkpoint_in)
        tqdm.write(f'Transferred {len(report["transferred"])} tensors; '
                   f'new LSTM tensors: {len(report["new_lstm_tensors"])}; '
                   f'shape changes: {report["reinitialized_shape_changes"]}')
    device = torch.device(a.device)
    model.to(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = a.cudnn_benchmark
    if a.stage == '2':
        freeze_backbone_through_conv13(model)
    criterion = MultiBoxLoss(num_classes=mc.num_classes, match_iou_threshold=sc.match_iou_threshold,
                             max_negatives_per_positive=tc.max_negatives_per_positive,
                             negative_loss_weight=tc.negative_loss_weight)
    # RMSProp details from released code; alpha=1 initial LR from the paper.
    optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad],
                                    lr=tc.learning_rate, alpha=tc.rmsprop_alpha,
                                    momentum=tc.rmsprop_momentum, eps=tc.rmsprop_eps)
    start = resume['epoch'] if resume else 0
    best_map = resume.get('best_map50') if resume else None
    if resume:
        restore_optimizer(optimizer, resume['optimizer'], a.learning_rate)
        tc = replace(tc, learning_rate=optimizer.param_groups[0]['lr'])
        restore_rng(resume['rng'])
        if a.resume_best_checkpoint:
            _import_resume_best(resume, a.resume_best_checkpoint, best_path)
    if resume and resume.get('validation_pending'):
        if start > a.epochs:
            raise ValueError('--epochs cannot be smaller than the saved epoch.')
        resume = _recover_pending_validation(resume, model, val_dataset, criterion, device,
                                             latest, best_path, a.epochs, not a.no_progress)
        best_map = resume['best_map50']
        if start == a.epochs:
            tqdm.write(f'Epoch target {a.epochs} already trained; pending validation is now complete.')
            return
    if start > a.epochs:
        raise ValueError('--epochs cannot be smaller than the saved epoch.')
    if start == a.epochs:
        tqdm.write(f'Epoch target {a.epochs} already reached; no additional training required.')
        return
    train_fn = train_detector_stage1 if a.stage == '1' else train_recurrent_stage2
    train_options = ({'frozen_backbone_batch_size': a.frozen_backbone_batch_size}
                     if a.stage == '2' else {})
    for epoch in range(start, a.epochs):
        numbered = epoch_checkpoint_path(latest, epoch + 1)
        if numbered.exists():
            raise FileExistsError(numbered)
        metrics = train_fn(model, loader, criterion, optimizer, device, progress=not a.no_progress,
                           description=f'{stage} Train {epoch + 1}/{a.epochs} (current/mean)',
                           **train_options)
        tqdm.write(f'epoch={epoch + 1} train: {metrics}')
        payload = {'format_version': FORMAT_VERSION, 'stage': stage, 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                   'global_step': (epoch + 1) * len(loader), 'model_config': model_config_dict(mc),
                   'ssd_config': asdict(sc), 'train_config': asdict(tc),
                   'pretrained_source': pretrained_source, 'transfer_report': report,
                   'train_metrics': metrics, 'training_signature': signature,
                   'performance_config': performance_config, 'resume_arguments': saved_arguments(a),
                   'validation_signature': validation_signature,
                   'category_id_to_label': dataset.category_id_to_label,
                   'label_to_name': dataset.label_to_name, 'best_map50': best_map,
                   'validation_video_ids': val_dataset.video_ids if val_dataset else None,
                   'validation_image_ids': ({v: [r['id'] for r in val_dataset.images_by_video[v]]
                                             for v in val_dataset.video_ids} if val_dataset else None),
                   'rng': capture_rng()}
        if worker_changes:
            payload['worker_count_changes'] = worker_changes
        improved = False
        if val_dataset is not None:
            try:
                val_metrics = evaluate_vid(model, val_dataset, device, criterion=criterion,
                                           score_threshold=0.0, nms_iou_threshold=sc.nms_iou_threshold,
                                           progress=not a.no_progress,
                                           description=f'{stage} Val {epoch + 1}/{a.epochs} (current/mean)')
            except BaseException:
                payload['validation_pending'] = True
                atomic_save_checkpoint(payload, numbered, overwrite=False)
                atomic_save_checkpoint(payload, latest)
                tqdm.write(f'Trained epoch saved despite validation failure: {numbered}')
                raise
            improved = _finish_validation(payload, val_metrics)
            best_map = payload['best_map50']
            payload['rng'] = capture_rng()
            tqdm.write(f'epoch={epoch + 1} val: {val_metrics}')
        atomic_save_checkpoint(payload, numbered, overwrite=False)
        atomic_save_checkpoint(payload, latest)
        if improved:
            atomic_save_checkpoint(payload, best_path)
        tqdm.write(f'Saved {numbered} and {latest}' + (f' and {best_path}' if improved else ''))


if __name__ == '__main__':
    main()
