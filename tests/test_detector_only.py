"""Detector-only entry points and deterministic evaluation policies."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate.evaluate_paired_vod import parse_args
from paired_config import PolicyConfig
from routing.paired_policy import choose_big
from utils.paired_checkpoints import PAIRED_FORMAT_VERSION, paired_from_payload


class DetectorOnlyTests(unittest.TestCase):
    def test_backbone_benchmark_shapes_and_state(self):
        import torch
        from unittest.mock import patch
        from train_matched import EXPERIMENTS, build_model
        from evaluate.evaluate_paired_vod import benchmark_backbones
        torch.set_num_threads(2)
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                model = build_model(experiment, 'detection', 0)
                modes = [child.training for child in model.modules()]
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with patch('evaluate.evaluate_paired_vod.time.perf_counter', side_effect=range(8)):
                    timings = benchmark_backbones(model, steps=2, warmup=1)
                expected = {'big', 'small'} if experiment.startswith('joint') else (
                    {'small'} if experiment.startswith('small_') else {'big'})
                self.assertEqual(set(timings), expected)
                for name, row in timings.items():
                    self.assertEqual(row['input_shape'], [1, 3, 160, 160] if name == 'small' else [1, 3, 320, 320])
                    self.assertEqual(row['FPS'], 1.)
                    self.assertEqual(row['mean_ms'], 1000.)
                self.assertEqual([child.training for child in model.modules()], modes)
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaises(ValueError):
            benchmark_backbones(model, steps=0, warmup=1)
        with patch.object(model.big.backbone, 'forward', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                benchmark_backbones(model, steps=1, warmup=1)
        self.assertEqual([child.training for child in model.modules()], modes)

    def test_legacy_profiling_imports_and_cleanup(self):
        import torch
        from torch import nn
        from routing import compute as legacy
        from utils import compute
        self.assertIs(legacy.profile_conv_linear, compute.profile_conv_linear)
        self.assertIs(legacy.synchronize, compute.synchronize)
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4),
                              nn.Flatten(), nn.Linear(144, 2))
        model.train()
        model[1].eval()
        modes = [child.training for child in model.modules()]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        frame = torch.ones(1, 3, 8, 8)
        for fail in (False, True):
            def forward():
                self.assertFalse(model(frame).requires_grad)
                if fail:
                    raise RuntimeError('profiling failure')
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'profiling failure'):
                    legacy.profile_conv_linear(model, forward)
            else:
                self.assertEqual(legacy.profile_conv_linear(model, forward), 4176)
            self.assertEqual([child.training for child in model.modules()], modes)
            self.assertTrue(all(torch.equal(value, before[key])
                                for key, value in model.state_dict().items()))
            self.assertTrue(all(not child._forward_hooks for child in model.modules()))

    def test_all_evaluation_needs_only_detector(self):
        args = parse_args(['--checkpoint=x.pt', '--output-dir=unused', '--mode=all'])
        self.assertEqual(args.mode, 'all')
        for flag, value in (('--switch-checkpoint', 'x.pt'), ('--switch-threshold', '.5'),
                            ('--maximum-gap', '16'), ('--mode', 'learned')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--checkpoint=x.pt', '--output-dir=unused', flag, value])

    def test_only_detector_checkpoint_stage_accepted(self):
        for stage in ('switch', 'stage1', 'stage2', None):
            with self.assertRaises(ValueError):
                paired_from_payload({'paired_format_version': PAIRED_FORMAT_VERSION, 'stage': stage})

    def test_comparison_routes(self):
        for mode in ('big', 'small', 'fixed'):
            policy = PolicyConfig(mode=mode, interval=3)
            last = None
            chosen = []
            for step in range(7):
                big, _ = choose_big(policy, step, last)
                chosen.append(big)
                if big:
                    last = step
            expected = ([True] * 7 if mode == 'big' else [False] * 7 if mode == 'small'
                        else [True, False, False, True, False, False, True])
            self.assertEqual(chosen, expected)
        policy = PolicyConfig(interval=3)
        self.assertTrue(choose_big(policy, 1, 0, precision=True)[0])
        self.assertTrue(choose_big(policy, 1, 0, scene_change=True)[0])
        with self.assertRaises(ValueError):
            PolicyConfig(mode='learned')


if __name__ == '__main__':
    unittest.main()
