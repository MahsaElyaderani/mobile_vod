"""Prepare a separate random joint-training run from a completed paired checkpoint."""
from path import parse_paths, required_path, resume_launcher

import argparse
import copy
import hashlib
from pathlib import Path
import shlex
import uuid

from train_paired_vod import parse_args as training_args
from utils.paired_checkpoints import paired_from_payload
from utils.runtime import load_checkpoint, atomic_save_checkpoint, epoch_checkpoint_path
from utils.checkpoints import model_from_checkpoint


def prepare(source, destination, epochs, joint_mode='frozen', big_checkpoint=None):
    if joint_mode not in ('frozen', 'trainable'):
        raise ValueError('Joint mode must be frozen or trainable.')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = load_checkpoint(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed while reading; use a completed numbered epoch checkpoint.')
    if original.get('stage') != 'detector' or original.get('validation_pending'):
        raise ValueError('Use a completed paired detector checkpoint with no pending validation.')
    if epochs <= original['epoch']:
        raise ValueError('Epoch target must exceed the saved checkpoint epoch.')
    if not original.get('optimizer') or not original.get('rng_state'):
        raise ValueError('Source lacks optimizer or RNG state.')
    paired_from_payload(original, 'detector')  # Strict architecture/state validation.
    payload = copy.deepcopy(original)
    if big_checkpoint is not None:
        big_path = Path(big_checkpoint).resolve()
        big_digest = hashlib.sha256(big_path.read_bytes()).hexdigest()
        native = load_checkpoint(big_path)
        if hashlib.sha256(big_path.read_bytes()).hexdigest() != big_digest:
            raise ValueError('Big checkpoint changed while reading; retry after its save completes.')
        if native.get('stage') != 'stage2' or native.get('validation_pending'):
            raise ValueError('Big checkpoint must be native stage2 with completed validation.')
        for key in ('model_config', 'ssd_config', 'category_id_to_label', 'label_to_name'):
            if key not in native or native[key] != original[key]:
                raise ValueError(f'Big checkpoint {key} differs from the paired checkpoint.')
        replacement = model_from_checkpoint(native)
        for key, value in replacement.state_dict().items():
            payload['model']['big.' + key] = value.detach().cpu().clone()
        paired_from_payload(payload, 'detector')  # Verify the complete combined state strictly.
        payload['sources']['joint_big_replacement'] = {
            'checkpoint': str(big_path), 'sha256': big_digest, 'epoch': native.get('epoch'),
            'previous_big_checkpoint_sha256': payload['sources'].get('big_checkpoint_sha256'),
            'optimizer': 'All RMSprop accumulators reset; parameter groups and learning rates retained',
        }
        payload['sources']['big_checkpoint_sha256'] = big_digest
        payload['sources']['big_checkpoint_epoch'] = native.get('epoch')
        # This is a new combined detector, not an exact continuation of its old optimization.
        payload['optimizer']['state'] = {}
        payload.pop('validation_metrics', None)
    definition = payload['run_definition']
    definition['arguments'].update(payload.pop('resume_arguments', {}))
    definition['arguments'].update({
        'annotations': required_path('TRAIN_ANNOTATIONS'),
        'frames_root': required_path('TRAIN_FRAMES'),
        'val_annotations': required_path('VAL_ANNOTATIONS'),
        'val_frames_root': required_path('VAL_FRAMES'),
    })
    # Legacy phase options are provenance only; the new run starts joint training immediately.
    for key in ('schedules', 'zero_memory_epochs', 'clean_memory_epochs', 'warmup_interval'):
        definition['arguments'].pop(key, None)
    definition['arguments'].pop('freeze_big_in_joint', None)
    if joint_mode == 'frozen':
        definition['arguments']['freeze_big_in_joint'] = True
    definition['random_schedule_version'] = 3
    parent = {'checkpoint': str(source), 'sha256': digest, 'epoch': original['epoch'],
              'run_id': original['run_id'], 'change': f'random 50/50 joint training, big/shared {joint_mode}',
              'source_phase': original['history'][-1].get('phase') if original.get('history') else None,
              'best_selection': 'reset for the new experiment; model/optimizer/RNG/history retained'}
    payload['sources']['joint_training_parent'] = parent
    if big_checkpoint is not None:
        parent['best_selection'] = 'reset; small weights/RNG/history retained; big/shared replaced; optimizer accumulators reset'
    definition['sources'] = copy.deepcopy(payload['sources'])
    payload.update(run_id=uuid.uuid4().hex, checkpoint_out=str(destination), best_map50=None)
    # Reconstruct the full command from saved scientific settings, then validate it.
    cli = []
    for key, value in definition['arguments'].items():
        flag = '--' + key.replace('_', '-')
        if value is None or value is False:
            continue
        cli.append(flag)
        if value is not True:
            cli.extend(str(v) for v in value) if isinstance(value, (list, tuple)) else cli.append(str(value))
    cli += ['--resume', str(destination), '--checkpoint-out', str(destination), '--epochs', str(epochs),
            '--sharing-strategy', 'file_system', '--log-every', '500']
    training_args(cli, resume_payload=payload)
    script = resume_launcher(cli, destination)
    # Require a completely fresh directory: protect original and unrelated runs.
    destination.parent.mkdir(parents=True, exist_ok=False)
    atomic_save_checkpoint(payload, epoch_checkpoint_path(destination, payload['epoch']), overwrite=False)
    atomic_save_checkpoint(payload, destination, overwrite=False)
    launcher = destination.parent/'resume_joint_training.sh'
    launcher.write_text(script)
    print(f'Prepared from completed epoch {payload["epoch"]}; random joint training with '
          f'big/shared {joint_mode} runs at epochs {payload["epoch"]+1}–{epochs}.')
    print('No training launched. Start with:\n' + shlex.join(['bash', str(launcher)]))
    if big_checkpoint is not None:
        print(f'Loaded native big/shared weights from epoch {native.get("epoch")}: {big_path}')
    return payload, cli


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-out', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--joint-mode', choices=('frozen', 'trainable'), default='frozen')
    parser.add_argument('--big-checkpoint',
                        help='Replace big backbone and shared LSTM/SSD from native stage2; reset optimizer accumulators.')
    args = parse_paths(parser, "prepare_joint_training")
    prepare(args.checkpoint, args.checkpoint_out, args.epochs, args.joint_mode, big_checkpoint=args.big_checkpoint)
