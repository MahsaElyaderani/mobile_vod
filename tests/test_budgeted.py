"""Fixed-budget scheduling and interruption/resume integration checks."""
import contextlib
import io
import json
import copy
import os
import subprocess
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.dataloader import budget_loader
from train_matched import EXPERIMENTS, parse_args, train, model_from_matched, save_matched_checkpoints, train_budgeted
from models.timm_mobilenet_v2 import TimmMobileNetV2Backbone
from utils.runtime import load_checkpoint
from train import _same_value


class RandomSamples(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), torch.rand(()).item()]), torch.zeros(30)


class BudgetTests(unittest.TestCase):
    def test_bash_first_run_and_resume_commands(self):
        script = Path(__file__).resolve().parents[1] / 'run_matched.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'etc/profile.d').mkdir(parents=True)
            (root/'etc/profile.d/conda.sh').write_text('conda() { return 0; }\n')
            conda = root/'conda'
            conda.write_text(f'#!/bin/sh\necho "{root}"\n')
            python = root/'python'
            python.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
root = Path({str(root)!r})
if 'path.py' in sys.argv:
    print({str(script.parent / 'splits/train_25pct.json')!r} if sys.argv[-1] == 'MATCHED_TRAIN_SUBSET' else root / sys.argv[-1])
else:
    with (root/'calls.jsonl').open('a') as log:
        log.write(json.dumps(sys.argv[1:]) + '\\n')
    if '--checkpoint-out' in sys.argv:
        p = Path(sys.argv[sys.argv.index('--checkpoint-out') + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
''')
            for executable in (conda, python):
                executable.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH']}
            for _ in range(2):
                subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls), 18)
            trained = []
            for i, command in enumerate(calls):
                if 'train_matched.py' in command:
                    argv = command[command.index('train_matched.py')+1:]
                    self.assertEqual('--resume' in argv, i >= 9)
                    if i >= 9:
                        self.assertNotIn('_best', argv[argv.index('--resume')+1])
                        continue  # Fake checkpoint files intentionally contain no tensors.
                    args = parse_args(argv)
                    trained.append((args.experiment, args.task))
                    self.assertEqual(Path(args.video_ids), script.parent / 'splits/train_25pct.json')
                    self.assertIn('matched_subset25_linear_1epoch', args.checkpoint_out)
                    self.assertEqual((args.epochs, args.lr_schedule, args.validate_every), (1, 'linear', 5000))
                    self.assertIsNone(args.max_batches)
                    self.assertEqual(args.batch_size, 16 if args.task == 'classification' else 8)
                    self.assertEqual(args.learning_rate, .01 if args.task == 'classification' else .001)
                    if args.task == 'detection':
                        self.assertTrue(args.init_checkpoint.endswith('classification_best.pt'))
                else:
                    from evaluate.evaluate_paired_vod import parse_args as evaluation_args
                    args = evaluation_args(command[command.index('evaluate.evaluate_paired_vod')+1:])
                    self.assertTrue(args.checkpoint.endswith('detection_best.pt'))
                    self.assertEqual(args.backbone_benchmark_steps, 200)
                    self.assertIsNone(args.video_ids)
                    self.assertEqual(args.mode, 'fixed' if 'joint_120d' in args.checkpoint else 'big')
            self.assertEqual(trained, [(exp, task) for exp in ('small_050', 'big_120d', 'joint_120d')
                                      for task in ('classification', 'detection')])

    def test_budget_crosses_epoch_without_extra_validation(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=.01)
        args = SimpleNamespace(epochs=2, max_batches=2, batch_size=1, learning_rate=.01,
                               seed=0, validate_every=3, task='classification')
        checkpoints, rates = [], []

        def batches(model, data, args, optimizer, loader):
            for _ in loader:
                rates.append(optimizer.param_groups[0]['lr'])
                optimizer.zero_grad()
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()
            return {}

        def checkpoint(epoch, stats, budget, scheduler, validate):
            checkpoints.append((budget['updates'], validate))
            return copy.deepcopy({'budget': budget, 'scheduler': scheduler})

        with patch('train_matched.budget_loader', side_effect=lambda d, a, e, s, n: range(n)), \
             patch('train_matched.classification_epoch', side_effect=batches):
            result = train_budgeted(model, range(2), args, optimizer, None, checkpoint)
        self.assertEqual(checkpoints, [(2, False), (3, True), (4, True)])
        for actual, expected in zip(rates, [.01, .0075, .005, .0025]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result['scheduler']['_last_lr'], [0.])

    def test_loader_resume_with_prefetch(self):
        args = SimpleNamespace(seed=7, batch_size=2, num_workers=2, prefetch_factor=2,
                               sharing_strategy=None, device='cpu', task='classification')
        full = list(budget_loader(RandomSamples(), args, 1, 0, 6))
        tail = list(budget_loader(RandomSamples(), args, 1, 2, 4))
        self.assertTrue(all(torch.equal(a[0], b[0]) for a, b in zip(full[2:], tail)))
        args.num_workers = 0
        self.assertTrue(all(torch.equal(a[0], b[0]) for a, b in
                            zip(full, budget_loader(RandomSamples(), args, 1, 0, 6))))

    def test_seven_budgeted_pipelines_and_mid_epoch_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                (root / split).mkdir()
                images, annotations = [], []
                for i in range(6):
                    Image.new('RGB', (64, 64), (30 + 20*i, 80, 150)).save(root/split/f'{i}.png')
                    images.append(dict(id=i, video_id=1, frame_id=i, width=64, height=64, file_name=f'{split}/{i}.png'))
                    annotations.append(dict(id=i, image_id=i, category_id=1, bbox=[8, 8, 40, 40]))
                (root/f'{split}.json').write_text(json.dumps(dict(videos=[{'id': 1}], images=images,
                    annotations=annotations, categories=[{'id': i, 'name': str(i)} for i in range(1, 31)])))
            common = ['--annotations', str(root/'train.json'), '--frames-root', str(root),
                      '--val-annotations', str(root/'val.json'), '--val-frames-root', str(root),
                      '--epochs', '1', '--batch-size', '1', '--max-batches', '3', '--num-workers', '0',
                      '--clip-length', '2', '--clip-stride', '2', '--steps', '2', '--lr-schedule', 'linear',
                      '--validate-every', '2', '--val-frames-per-video', '2', '--device', 'cpu', '--no-progress']
            weights = (contextlib.nullcontext() if os.environ.get('TIMM_VERIFY_PRETRAINED') == '1' else
                       patch.object(TimmMobileNetV2Backbone, 'load_pretrained', return_value={'test': 'random'}))
            with contextlib.redirect_stdout(io.StringIO()), weights:
                for experiment in EXPERIMENTS:
                    for task, lr in [('classification', '.01'), ('detection', '.001')]:
                        with self.subTest(experiment=experiment, task=task):
                            destination = root/experiment/(task+'.pt')
                            options = common + ['--experiment', experiment, '--task', task,
                                '--checkpoint-out', str(destination), '--learning-rate', lr]
                            if task == 'detection':
                                options += ['--init-checkpoint', str(root/experiment/'classification_best.pt')]
                            snapshots = []
                            def record(payload, path, **kwargs):
                                save_matched_checkpoints(payload, path, **kwargs)
                                snapshots.append((payload['global_step'], payload['optimizer']['param_groups'][0]['lr']))
                            with patch('train_matched.save_matched_checkpoints', side_effect=record):
                                full = train(parse_args(options))
                            self.assertEqual([row['update'] for row in full['history']], [2, 3])
                            self.assertAlmostEqual(snapshots[0][1], float(lr)/3)
                            self.assertEqual(snapshots[-1][1], 0.)
                            self.assertEqual(full['scheduler']['last_epoch'], 3)
                            self.assertEqual(len(list(destination.parent.glob(task+'*.pt'))), 2)
                            model_from_matched(load_checkpoint(destination.with_stem(task+'_best')))
                            if experiment not in ('small_050', 'joint_100'):
                                continue
                            interrupted = root/experiment/(task+'_interrupted.pt')
                            interrupted_options = options + ['--checkpoint-out', str(interrupted)]
                            def stop(payload, path, **kwargs):
                                save_matched_checkpoints(payload, path, **kwargs)
                                raise InterruptedError('Simulated interruption after the periodic checkpoint')
                            with patch('train_matched.save_matched_checkpoints', side_effect=stop):
                                with self.assertRaises(InterruptedError):
                                    train(parse_args(interrupted_options))
                            mid = load_checkpoint(interrupted)
                            self.assertEqual(mid['global_step'], 2)
                            defaults = parse_args(['--experiment', experiment, '--task', task,
                                '--resume', str(interrupted)])
                            self.assertEqual(defaults.epochs, 1)
                            self.assertEqual(defaults.lr_schedule, 'linear')
                            restored = model_from_matched(mid).eval()
                            if task == 'classification':
                                from datasets.paired_frames import PairedFrames
                                with torch.no_grad():
                                    self.assertEqual(restored(PairedFrames(torch.zeros(1,3,320,320),
                                        torch.zeros(1,3,160,160)), steps=2).shape, (1,2,30))
                            else:
                                from evaluate.evaluate_paired_vod import main as evaluate
                                result = evaluate(['--checkpoint', str(interrupted), '--annotations', str(root/'val.json'),
                                    '--frames-root', str(root), '--frames-per-video', '2',
                                    '--mode', 'fixed' if experiment.startswith('joint') else 'big', '--score-threshold', '0',
                                    '--device', 'cpu', '--warmup-frames', '0', '--no-visuals', '--no-progress',
                                    '--output-dir', str(root/experiment/'mid_evaluation')])
                                self.assertEqual(next(iter(result.values()))['evaluation_source']['global_step'], 2)
                                self.assertEqual(next(iter(result.values()))['mAP@0.5'], mid['history'][-1]['validation']['mAP@0.5'])
                            resumed = train(parse_args(interrupted_options + ['--resume', str(interrupted)]))
                            for key in ('model', 'optimizer', 'scheduler', 'rng_state', 'budget', 'history'):
                                # Detector validation contains machine-dependent latency/FPS fields.
                                if key != 'history':
                                    self.assertTrue(_same_value(full[key], resumed[key]), (task, key))
                            metric = full['selection_metric']
                            self.assertEqual([r['validation'][metric] for r in full['history']],
                                             [r['validation'][metric] for r in resumed['history']])
                            self.assertEqual(train(parse_args(interrupted_options + ['--resume', str(interrupted)]))['global_step'], 3)
                            with self.assertRaises(ValueError):
                                train(parse_args(interrupted_options + ['--resume', str(interrupted), '--epochs', '2']))
                            with self.assertRaises(ValueError):
                                train(parse_args(interrupted_options + ['--resume', str(interrupted), '--learning-rate', '.2']))


if __name__ == '__main__':
    unittest.main()
