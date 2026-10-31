"""Joint recurrent multi-label classification on VID frames (no detection heads)."""
import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import math
import random
import uuid
import torch
from torch.utils.data import DataLoader
from path import parse_paths
from engine.evaluator import classification_metrics
from config import model_config_dict, SSDConfig
from paired_config import PairedConfig
from models.joint_classification import JointClassifier
from datasets.classification import VIDClassificationDataset, classification_collate
from routing.data import add_data_arguments, selected_dataset, data_signature, guard_held_out, validate_native_categories
from utils.joint_initialization import make_joint_config, initialize_backbones, classifier_from_payload, CLASSIFICATION_FORMAT
from utils.checkpoints import capture_rng, restore_rng
from utils.resume import resume_defaults, saved_arguments, restore_optimizer
from utils.runtime import load_checkpoint, tqdm
from utils.paired_checkpoints import run_lock, save_training_payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument('--phase', choices=('joint_pretrain',), default='joint_pretrain')
    p.add_argument('--backbone', choices=('mobilenet_v1', 'timm_mobilenet_v2'), default='mobilenet_v1')
    p.add_argument('--big-width-mult', type=float, default=1.)
    p.add_argument('--fast-width-mult', type=float)
    p.add_argument('--backbone-weights')
    p.add_argument('--random-backbones', action='store_true', help='Explicitly skip pretrained backbone initialization.')
    p.add_argument('--freeze-bn', action='store_true')
    p.add_argument('--val-annotations')
    p.add_argument('--val-frames-root')
    p.add_argument('--max-val-videos', type=int)
    p.add_argument('--checkpoint-out', required=True)
    p.add_argument('--resume')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--grad-clip', type=float, default=10.)
    p.add_argument('--max-batches', type=int, help='Limit training and validation batches for smoke checks.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-progress', action='store_true')
    args = resume_defaults(p, parse_paths(p, 'train_joint_pretrain', argv), argv)
    if args.fast_width_mult is None:
        args.fast_width_mult = .5 if args.backbone == 'timm_mobilenet_v2' else .35
    if min(args.epochs, args.steps, args.batch_size) < 1 or args.num_workers < 0:
        p.error('epochs, steps and batch-size must be positive; num-workers nonnegative')
    if not args.freeze_bn and args.batch_size < 2:
        p.error('Use batch-size >=2 when training BatchNorm statistics')
    if args.max_batches is not None and args.max_batches < 1:
        p.error('max-batches must be positive')
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        p.error('grad-clip must be finite and positive')
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        p.error('learning-rate must be finite and positive')
    if bool(args.val_annotations) != bool(args.val_frames_root):
        p.error('Supply both validation paths')
    if args.max_val_videos is not None and not args.val_annotations:
        p.error('--max-val-videos requires validation annotations and frames root')
    return args


def classification_epoch(model, dataset, args, optimizer=None, loader=None):
    training = optimizer is not None
    model.train(training)
    if loader is None:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training, drop_last=training,
                            num_workers=args.num_workers, collate_fn=classification_collate)
    total, samples = 0., 0
    scores, labels = [], []
    with torch.set_grad_enabled(training):
        for index, (frames, target) in enumerate(tqdm(loader, disable=args.no_progress, desc='Classification train' if training else 'Classification val'), 1):
            frames, target = frames.to(args.device), target.to(args.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(frames, steps=args.steps)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite classification loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach()) * len(target)
            samples += len(target)
            if not training:
                scores.append(logits[:, -1].detach().cpu())
                labels.append(target.cpu())
            if args.max_batches and index >= args.max_batches:
                break
    if not samples:
        raise ValueError('No classification batches; check dataset size and batch-size')
    metrics = classification_metrics(torch.cat(scores), torch.cat(labels)) if not training else {}
    return {'bce_loss': total / samples, 'samples': samples, **metrics}


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    destination = Path(args.checkpoint_out).resolve()
    resume = load_checkpoint(args.resume) if args.resume else None
    cfg = make_joint_config(args.backbone, args.big_width_mult)
    paired = PairedConfig(fast_width_mult=args.fast_width_mult, freeze_bn_stats=args.freeze_bn)
    model = classifier_from_payload(resume) if resume else JointClassifier(cfg, paired, 30)
    if model.model_cfg != cfg or model.paired_config != paired:
        raise ValueError('Classification resume architecture/configuration changed')
    if resume and (resume.get('steps') != args.steps or resume.get('objective') != 'vid_multilabel_bce'):
        raise ValueError('Classification objective or recurrent steps changed')
    if destination.exists():
        existing = load_checkpoint(destination)
        if not resume or existing.get('run_id') != resume.get('run_id') or existing.get('epoch') != resume.get('epoch'):
            raise FileExistsError('Output belongs to another run or epoch; use a fresh destination')
        if any(not torch.equal(existing['model'][key], value) for key, value in resume['model'].items()):
            raise FileExistsError('Output contains different model weights')
    if not resume and list(destination.parent.glob(destination.stem + '_epoch_*')):
        raise FileExistsError('Output directory already has numbered checkpoints for this run name')
    base = selected_dataset(args, 320, interpolation=cfg.input_interpolation)
    metadata = {'category_id_to_label': base.category_id_to_label, 'label_to_name': base.label_to_name,
                'ssd_config': asdict(SSDConfig())}
    if resume:
        validate_native_categories(resume, base)
    training_signature = data_signature(base)
    val = None
    if args.val_annotations:
        val_args = SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames_root,
                                  max_videos=args.max_val_videos, video_ids=None, subset_seed=args.subset_seed)
        val = selected_dataset(val_args, 320, interpolation=cfg.input_interpolation)
        validate_native_categories(metadata, val)
        guard_held_out([training_signature], val)
    validation_signature = data_signature(val) if val else None
    if resume:
        # File identities and class mappings are strict; roots may move between machines.
        for old, new in ((resume['training_signature'], training_signature), (resume.get('validation_signature'), validation_signature)):
            if (old is None) != (new is None) or old and any(old[k] != new[k] for k in ('annotations_sha256', 'video_ids')):
                raise ValueError('Classification data selection changed on resume')
    sources = resume['sources'] if resume else initialize_backbones(model.big_backbone, model.small_backbone,
                                                                   args.backbone_weights, args.random_backbones)
    model.to(args.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate or .001, alpha=.9, momentum=.9, eps=1.)
    if resume:
        restore_optimizer(optimizer, resume['optimizer'], args.learning_rate)
        restore_rng(resume['rng_state'])
    start = resume['epoch'] if resume else 0
    if start > args.epochs:
        raise ValueError('Epoch target is below saved epoch')
    if start == args.epochs:
        print('Epoch target already reached; no additional training required.')
        return resume
    history = list(resume['history']) if resume else []
    run_id = resume['run_id'] if resume else uuid.uuid4().hex
    training = VIDClassificationDataset(base, cfg.backbone, True)
    validation = VIDClassificationDataset(val, cfg.backbone, False) if val else None
    for epoch in range(start + 1, args.epochs + 1):
        metrics = classification_epoch(model, training, args, optimizer)
        val_metrics = None
        if validation:
            rng = capture_rng()
            try:
                random.seed(0)
                torch.manual_seed(0)
                val_metrics = classification_epoch(model, validation, args)
            finally:
                restore_rng(rng)
        history.append({'epoch': epoch, 'train': metrics, 'validation': val_metrics})
        payload = {**metadata, 'classification_format_version': CLASSIFICATION_FORMAT, 'stage': 'joint_pretrain',
                   'objective': 'vid_multilabel_bce', 'class_to_idx': {str(k): k-1 for k in base.label_to_name},
                   'model_config': model_config_dict(cfg), 'paired_config': asdict(paired),
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng_state': capture_rng(),
                   'epoch': epoch, 'run_id': run_id, 'history': history, 'sources': sources, 'steps': args.steps,
                   'training_signature': training_signature, 'training_signatures': [training_signature],
                   'validation_signature': validation_signature, 'resume_arguments': saved_arguments(args)}
        save_training_payload(payload, destination)
        print(f'Saved joint_pretrain epoch {epoch}: BCE={metrics["bce_loss"]:.6f} -> {destination}', flush=True)
    return payload


if __name__ == '__main__':
    args = parse_args()
    with run_lock(args.checkpoint_out):
        train(args)
