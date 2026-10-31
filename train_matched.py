"""Matched V2 classification -> detection experiments; legacy trainers are unchanged."""
import argparse
from copy import copy
from pathlib import Path
import math
import random
import uuid

import torch
from torch import nn
from datasets.classification import VIDClassificationDataset
from datasets.dataloader import budget_loader
from engine.paired_trainer import (add_training_arguments, build_training_data, make_loader,
                                   criterion_for, train_detector_epoch)
from engine.paired_evaluator import evaluate_paired, EvaluationOptions
from models.joint_classification import JointClassifier
from models.lstm_ssd import LSTMSSD
from models.paired_vod import PairedVOD
from models.ssd_head import SSDFeatureMaps, SSDBoxPredictor
from paired_config import PairedConfig, PolicyConfig
from path import parse_paths, RUNS_DIR
from routing.data import data_signature, guard_held_out
from train_joint_pretrain import classification_epoch
from utils.checkpoints import capture_rng, restore_rng
from utils.joint_initialization import make_joint_config
from utils.paired_checkpoints import paired_metadata, run_lock
from utils.pretrained import sha256_file
from utils.resume import resume_defaults, saved_arguments, restore_optimizer
from utils.runtime import load_checkpoint, atomic_save_checkpoint

EXPERIMENTS = {'small_050': .5, 'big_100': 1., 'big_110d': 1.1, 'big_120d': 1.2,
               'joint_100': 1., 'joint_110d': 1.1, 'joint_120d': 1.2}


class MatchedDetector(PairedVOD):
    def __init__(self, experiment, seed=0):
        cfg = make_joint_config('timm_mobilenet_v2', EXPERIMENTS[experiment])
        self.standalone = None if experiment.startswith('joint_') else ('small' if experiment == 'small_050' else 'big')
        if self.standalone is None:
            super().__init__(LSTMSSD(cfg), PairedConfig(fast_width_mult=.5))
        else:
            nn.Module.__init__(self)
            self.big = LSTMSSD(cfg)
            self.paired_config = PairedConfig(fast_width_mult=.5)
            self.fast_backbone = nn.Identity()
            self.small_projection = nn.Identity()
            if self.standalone == 'small':
                self.big.backbone.network.blocks[5][0].conv_dw.stride = (1, 1)
                self.big.backbone.output_stride = 16
                self.big.backbone.stride_changes = ['blocks.5.0.conv_dw: stride 2 -> 1; dilation unchanged']
        # Separate initialization stream: changing backbone depth cannot change shared weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.big.lstm.reset_parameters()
            self.big.feature_maps = SSDFeatureMaps()
            self.big.predictor = SSDBoxPredictor(cfg.feature_channels, cfg.num_classes,
                self.ssd_cfg.num_anchors_per_location, self.ssd_cfg.include_background_class)
        self.configure_detector_training()

    def forward_big(self, frames, state=None):
        if self.standalone == 'small':
            return self.big._forward_from_backbone(self.big.backbone(frames.small), state, True,
                                                   anchor_dtype=frames.dtype)
        return super().forward_big(frames, state)

    def forward_small(self, frames, state=None):
        return self.forward_big(frames, state) if self.standalone else super().forward_small(frames, state)

    def forward_selected(self, frames, state, use_big, small_commit=None):
        if self.standalone:
            return self.forward_big(frames, state)
        return super().forward_selected(frames, state, use_big, small_commit)


def build_model(experiment, task, seed):
    detector = MatchedDetector(experiment, seed)
    if task == 'detection':
        return detector
    if task != 'classification':
        raise ValueError('Unknown matched task.')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 1)
        return JointClassifier(detector.model_cfg, detector.paired_config, 30, detector=detector)


def model_from_matched(payload):
    if payload.get('matched_format_version') != 1 or payload.get('stage') not in ('matched_classification', 'matched_detection'):
        raise ValueError('Expected a matched classification/detection checkpoint.')
    if payload.get('class_to_idx') != {str(i): i - 1 for i in range(1, 31)}:
        raise ValueError('Matched class mapping differs.')
    task = payload['stage'].removeprefix('matched_')
    model = build_model(payload['experiment'], task, payload['initialization_seed'])
    expected = paired_metadata(model)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Matched architecture/configuration differs.')
    if payload.get('protocol') != {'bn_stats': 'frozen', 'standalone_memory': 'update', 'joint_memory': 'big_only'}:
        raise ValueError('Matched training protocol differs.')
    model.load_state_dict(payload['model'], strict=True)
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    next(a for a in parser._actions if a.dest == 'checkpoint_out').required = False
    parser.set_defaults(clip_length=6, clip_stride=6, batch_size=2, val_frames_per_video=20)
    parser.add_argument('--task', choices=('classification', 'detection'), required=True)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument('--lr-schedule', choices=('constant', 'linear'), default='constant')
    parser.add_argument('--validate-every', type=int, default=5000, help='Budgeted validation interval in optimizer updates.')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--init-checkpoint')
    args = resume_defaults(parser, parse_paths(parser, 'train_matched', argv), argv)
    if args.validate_every < 1:
        parser.error('validate-every must be positive')
    if args.steps < 1 or args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('steps and learning-rate must be positive and finite')
    if not args.val_annotations or not args.val_frames_root:
        parser.error('Matched experiments require validation annotations and frames root')
    directory = Path(RUNS_DIR) / 'matched' / f'seed_{args.seed}' / args.experiment
    args.checkpoint_out = args.checkpoint_out or str(directory / (args.task + '.pt'))
    if args.task == 'classification' and args.init_checkpoint:
        parser.error('Classification starts from official timm weights, not another experiment')
    if args.task == 'detection' and not args.resume:
        args.init_checkpoint = args.init_checkpoint or str(directory / 'classification.pt')
    args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std = 0., .1, .1
    return args


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def checkpoint_position(payload):
    return payload['epoch'], payload.get('budget', {}).get('batch', 0)


def save_matched_checkpoints(payload, destination, select_best=True):
    if not select_best:
        atomic_save_checkpoint(payload, destination)
        return
    key = payload.get('selection_metric',
                      'bce_loss' if payload['stage'] == 'matched_classification' else 'mAP@0.5')
    scores = [row['validation'][key] for row in payload['history']]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('Cannot select a best checkpoint from non-finite validation scores.')
    best = min if key == 'bce_loss' else max
    improved = len(scores) == 1 or best(scores) != best(scores[:-1])
    if improved:
        best_path = destination.with_name(destination.stem + '_best' + (destination.suffix or '.pt'))
        atomic_save_checkpoint(payload, best_path)
    atomic_save_checkpoint(payload, destination)


def train_budgeted(model, train_data, args, optimizer, resume, checkpoint):
    batches = min(len(train_data) // args.batch_size, args.max_batches or len(train_data))
    budget = {'epochs': args.epochs, 'total_updates': args.epochs * batches,
              'updates': 0, 'epoch': 1, 'batch': 0, 'sampling_version': 1}
    if resume:
        saved = resume.get('budget')
        if not saved or any(saved[k] != budget[k] for k in ('epochs', 'total_updates', 'sampling_version')):
            raise ValueError('A linear schedule requires the original predetermined budget on resume.')
        budget.update(saved)
        if (not 0 <= budget['batch'] <= batches or
                budget['updates'] != (budget['epoch'] - 1) * batches + budget['batch'] or
                not 0 <= budget['updates'] <= budget['total_updates']):
            raise ValueError('Checkpoint batch position and completed update count disagree.')
        if args.learning_rate is not None and any(args.learning_rate != lr for lr in resume['scheduler']['base_lrs']):
            raise ValueError('Changing the initial LR would change the fixed-budget schedule; use a new run.')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: max(0., 1. - step / budget['total_updates']))
    if resume:
        scheduler.load_state_dict(resume['scheduler'])
        if scheduler.last_epoch != budget['updates']:
            raise ValueError('Scheduler and completed update counts disagree.')
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = lr
    print(f"Budget: {budget['updates']}/{budget['total_updates']} updates; linear LR, no warmup.", flush=True)
    if budget['updates'] >= budget['total_updates']:
        return resume
    # Step only after a successful optimizer update, including inside the reused loops.
    hook = optimizer.register_step_post_hook(lambda *_: scheduler.step())
    loop_args = copy(args)
    loop_args.max_batches = None  # The sliced loader already enforces the update budget.
    payload = resume
    try:
        while budget['updates'] < budget['total_updates']:
            if budget['batch'] == batches:
                budget['epoch'] += 1
                budget['batch'] = 0
            if budget['batch'] == 0:
                seed_training(args.seed + budget['epoch'])
            count = min(batches - budget['batch'], budget['total_updates'] - budget['updates'],
                        args.validate_every - budget['updates'] % args.validate_every)
            loader = budget_loader(train_data, args, budget['epoch'], budget['batch'], count)
            if args.task == 'classification':
                stats = classification_epoch(model, train_data, loop_args, optimizer, loader=loader)
            else:
                stats = train_detector_epoch(model, loader, optimizer, criterion_for(model), loop_args, budget['epoch'])
                if model.standalone:
                    stats.pop('big_selections')
            if scheduler.last_epoch != budget['updates'] + count:
                raise RuntimeError('Budgeted training must perform one optimizer update per consumed batch.')
            budget['updates'] += count
            budget['batch'] += count
            validate = budget['updates'] % args.validate_every == 0 or budget['updates'] == budget['total_updates']
            stats['interval_updates'] = count
            payload = checkpoint(budget['epoch'], stats, dict(budget), scheduler.state_dict(), validate)
    finally:
        hook.remove()
    return payload


def train(args):
    resume = load_checkpoint(args.resume) if args.resume else None
    source = load_checkpoint(args.init_checkpoint) if args.init_checkpoint and not resume else None
    if resume and (resume['experiment'] != args.experiment or resume['stage'] != 'matched_' + args.task):
        raise ValueError('Resume must use the same matched experiment and stage.')
    seed_training(args.seed)
    model = model_from_matched(resume) if resume else build_model(args.experiment, args.task, args.seed)
    sources, signatures = {}, []
    if source:
        if source.get('stage') != 'matched_classification' or source.get('experiment') != args.experiment:
            raise ValueError('Detection must initialize from classification of the same experiment.')
        if source.get('initialization_seed') != args.seed:
            raise ValueError('Use the same seed for matched classification and detection.')
        classifier = model_from_matched(source)
        classifier.transfer_to_detector(model)
        sources = {'classification_sha256': sha256_file(args.init_checkpoint)}
        signatures = list(source['training_signatures'])
    data_args = copy(args)
    if args.task == 'classification':
        data_args.clip_length = data_args.clip_stride = 1
    metadata = {} if not (source or resume) else dict(source or resume)
    train_data, val_data = build_training_data(data_args, model, metadata)
    for signature in signatures:
        guard_held_out([signature], val_data)
    own_signature = data_signature(train_data)
    if own_signature not in signatures:
        signatures.append(own_signature)
    definition = {'arguments': {k: v for k, v in saved_arguments(args).items() if k != 'init_checkpoint'},
                  'training': {k: v for k, v in own_signature.items() if k != 'video_directories'},
                  'validation': {k: v for k, v in data_signature(val_data).items() if k != 'video_directories'},
                  'num_workers': args.num_workers}
    if args.lr_schedule == 'constant':
        definition['arguments'].pop('lr_schedule')
        definition['arguments'].pop('validate_every')
    if resume and definition != resume['run_definition']:
        raise ValueError('Matched training settings or data changed on resume.')
    destination = Path(args.checkpoint_out)
    best_path = destination.with_name(destination.stem + '_best' + (destination.suffix or '.pt'))
    if best_path.exists():
        stored_best = load_checkpoint(best_path)
        if not resume or stored_best['run_id'] != resume['run_id'] or checkpoint_position(stored_best) > checkpoint_position(resume):
            raise FileExistsError('Best checkpoint belongs to another run or a later training position.')
        if checkpoint_position(stored_best) == checkpoint_position(resume) and (
                stored_best['model'].keys() != resume['model'].keys() or
                any(not torch.equal(v, stored_best['model'][k]) for k, v in resume['model'].items())):
            raise FileExistsError('Best checkpoint contains different model weights.')
    if destination.exists():
        stored = load_checkpoint(destination)
        if not resume or stored['run_id'] != resume['run_id'] or checkpoint_position(stored) != checkpoint_position(resume):
            raise FileExistsError('Output belongs to another run/epoch; use a fresh destination.')
        if stored['model'].keys() != resume['model'].keys() or any(not torch.equal(v, stored['model'][k]) for k, v in resume['model'].items()):
            raise FileExistsError('Output contains different model weights.')
    if not resume and list(destination.parent.glob(destination.stem + '_epoch_*')):
        raise FileExistsError('Numbered checkpoints already exist at this destination.')
    batches = len(train_data) // args.batch_size
    if args.max_batches and args.max_batches > batches:
        raise ValueError(f'Budget requests {args.max_batches} batches but dataset provides only {batches}.')
    if not resume and args.task == 'classification':
        for name in ('big_backbone', 'small_backbone'):
            backbone = getattr(model, name)
            if hasattr(backbone, 'load_pretrained'):
                sources[name] = backbone.load_pretrained()
    model.to(args.device)
    lr = args.learning_rate or (.001 if args.task == 'classification' else .0001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=.9, momentum=.9, eps=1.)
    if resume:
        restore_optimizer(optimizer, resume['optimizer'], args.learning_rate)
        restore_rng(resume['rng_state'])
        sources, signatures = resume['sources'], resume['training_signatures']
    start = resume['epoch'] if resume else 0
    if args.lr_schedule == 'constant' and start >= args.epochs:
        print('Epoch target already reached; no training required.')
        return resume
    history, run_id = (list(resume['history']), resume['run_id']) if resume else ([], uuid.uuid4().hex)
    if args.task == 'classification':
        train_data = VIDClassificationDataset(train_data, 'timm_mobilenet_v2', True)
        val_data = VIDClassificationDataset(val_data, 'timm_mobilenet_v2', False)

    def checkpoint(epoch, stats, budget=None, scheduler=None, validate=True):
        rng = capture_rng()
        was_training = model.training
        metrics = None
        if validate:
            try:
                seed_training(0)
                if args.task == 'classification':
                    val_args = copy(args)
                    val_args.max_batches = None
                    metrics = classification_epoch(model, val_data, val_args)
                else:
                    policy = PolicyConfig(mode='big' if model.standalone else 'fixed', interval=args.big_call_interval)
                    metrics, _ = evaluate_paired(model, val_data, policy,
                        EvaluationOptions(args.device, score_threshold=0. if budget else .001,
                                          warmup_frames=0, progress=not args.no_progress))
            finally:
                restore_rng(rng)
                model.train(was_training)
        if validate:
            history.append({'epoch': epoch, 'train': stats, 'validation': metrics,
                            **({'update': budget['updates']} if budget else {})})
        payload = {**paired_metadata(model), 'matched_format_version': 1, 'stage': 'matched_' + args.task,
                   'experiment': args.experiment, 'initialization_seed': args.seed,
                   'protocol': {'bn_stats': 'frozen', 'standalone_memory': 'update', 'joint_memory': 'big_only'},
                   'class_to_idx': {str(i): i - 1 for i in range(1, 31)},
                   'category_id_to_label': metadata['category_id_to_label'], 'label_to_name': metadata['label_to_name'],
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng_state': capture_rng(),
                   'selection_metric': (resume.get('selection_metric', 'bce_loss' if args.task == 'classification' else 'mAP@0.5')
                                        if resume else 'micro_ap' if args.task == 'classification' else 'mAP@0.5'),
                   'epoch': epoch, 'run_id': run_id, 'history': history, 'sources': sources,
                   'training_signatures': signatures, 'run_definition': definition,
                   'resume_arguments': saved_arguments(args)}
        if budget:
            payload.update(budget=budget, scheduler=scheduler, global_step=budget['updates'])
        save_matched_checkpoints(payload, destination, select_best=validate)
        suffix = f" update {budget['updates']}" if budget else ''
        print(f'Saved {args.experiment} {args.task} epoch {epoch}{suffix}: {destination}', flush=True)
        if metrics:
            key = payload['selection_metric']
            print(f"Validation {key}={metrics[key]:.6f}; class coverage {len(metrics['present_classes'])}/30", flush=True)
        return payload

    if args.lr_schedule == 'linear':
        return train_budgeted(model, train_data, args, optimizer, resume, checkpoint)
    for epoch in range(start + 1, args.epochs + 1):
        # Identical sampling/augmentation streams across architectures; exact epoch-boundary resume.
        seed_training(args.seed + epoch)
        if args.task == 'classification':
            stats = classification_epoch(model, train_data, args, optimizer)
        else:
            stats = train_detector_epoch(model, make_loader(train_data, args), optimizer, criterion_for(model), args, epoch)
            if model.standalone:
                stats.pop('big_selections')  # The random schedule is consumed for reproducibility, but unused.
        payload = checkpoint(epoch, stats)
    return payload


if __name__ == '__main__':
    args = parse_args()
    with run_lock(args.checkpoint_out):
        train(args)
