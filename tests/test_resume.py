"""Resume compatibility and optimizer-state regression tests."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.resume import resume_defaults, restore_optimizer
from utils.paired_checkpoints import finish_validation
from train import _same_run_definition
from train_paired_vod import parse_args


class ResumeTests(unittest.TestCase):
    def test_big_call_interval_rename_preserves_resume(self):
        from engine.paired_trainer import add_training_arguments
        from train_matched import parse_args as matched_args
        from utils.resume import saved_arguments
        parser = argparse.ArgumentParser()
        add_training_arguments(parser)
        self.assertIn('--big-call-interval', parser.format_help())
        self.assertNotIn('--validation-interval', parser.format_help())
        self.assertEqual(parser.parse_args(['--checkpoint-out=x', '--annotations=a', '--frames-root=b']).big_call_interval, 8)
        for parse, extra in ((parse_args, []), (matched_args, ['--task', 'detection', '--experiment', 'small_050'])):
            payload = {'run_definition': {'arguments': {'validation_interval': 4}},
                       'resume_arguments': {'validation_interval': 6}}
            for flag in (None, '--big-call-interval', '--validation-interval'):
                argv = extra + ['--resume=x', '--checkpoint-out=y'] + ([flag, '3'] if flag else [])
                with patch('utils.resume.load_checkpoint', return_value=payload):
                    args = parse(argv)
                expected = 3 if flag else 6
                self.assertEqual(args.big_call_interval, expected)
                self.assertFalse(hasattr(args, 'validation_interval'))
                self.assertEqual(saved_arguments(args)['validation_interval'], expected)

    def test_saved_settings_and_local_paths(self):
        payload = {'run_definition': {'arguments': {
            'clip_length': 3, 'batch_size': 2, 'freeze_big_in_joint': True,
            'annotations': '/old/server/train.json', 'device': 'cuda', 'learning_rate': .1}},
            'paired_config': {'fast_width_mult': .35, 'fast_image_size': 160,
                              'freeze_bn_stats': True}}
        with patch('utils.resume.load_checkpoint', return_value=payload):
            args = parse_args(['--resume', 'saved.pt', '--checkpoint-out', '/new/run.pt',
                               '--annotations', '/new/train.json', '--epochs', '50', '--device', 'cpu'])
        self.assertEqual((args.clip_length, args.batch_size, args.freeze_big_in_joint), (3, 2, True))
        self.assertEqual((args.annotations, args.device, args.epochs), ('/new/train.json', 'cpu', 50))
        self.assertIsNone(args.learning_rate)

    def test_explicit_equals_and_abbreviated_overrides(self):
        payload = {'resume_arguments': {'batch_size': 2, 'clip_length': 3}}
        with patch('utils.resume.load_checkpoint', return_value=payload):
            args = parse_args(['--resume=x', '--checkpoint-out=y', '--batch-size=4', '--clip-len', '5'])
        self.assertEqual((args.batch_size, args.clip_length), (4, 5))

    def optimizer(self):
        a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.RMSprop([{'params': [a], 'lr': .01}, {'params': [b], 'lr': .02}], momentum=.9)
        (a.sum() + b.sum()).backward()
        opt.step()
        return opt

    def test_restore_moments_then_override_rates(self):
        original = self.optimizer().state_dict()
        opt = self.optimizer()
        restore_optimizer(opt, original, .03, .04)
        self.assertEqual([g['lr'] for g in opt.param_groups], [.03, .04])
        for key, state in original['state'].items():
            for name, tensor in state.items():
                restored = opt.state_dict()['state'][key][name]
                if isinstance(tensor, torch.Tensor):
                    self.assertTrue(torch.equal(tensor, restored))
                else:
                    self.assertEqual(tensor, restored)
        restore_optimizer(opt, original)
        self.assertEqual([g['lr'] for g in opt.param_groups], [.01, .02])

    def test_invalid_rates_and_group_changes_rejected(self):
        opt = self.optimizer()
        for rate in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                restore_optimizer(opt, opt.state_dict(), rate)
        one = torch.optim.RMSprop([torch.nn.Parameter(torch.ones(2))])
        with self.assertRaisesRegex(ValueError, 'two-group'):
            restore_optimizer(one, one.state_dict(), fast_learning_rate=.1)

    def test_best_compatibility_allows_only_lr_config_change(self):
        original = {'train_config': {'learning_rate': .1, 'rmsprop_eps': 1},
                    'model_config': {'width_mult': 1}, 'category_id_to_label': {1: 0}}
        changed = copy.deepcopy(original)
        changed['train_config']['learning_rate'] = .2
        self.assertTrue(_same_run_definition(original, changed))
        changed['train_config']['rmsprop_eps'] = 2
        self.assertFalse(_same_run_definition(original, changed))
        changed = copy.deepcopy(original)
        changed['category_id_to_label'] = {2: 0}
        self.assertFalse(_same_run_definition(original, changed))

    def test_missing_historical_best_is_not_replaced_by_worse_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'detector.pt'
            payload = {'validation_pending': True, 'best_map50': .8, 'run_id': 'same', 'epoch': 2}
            with patch('utils.paired_checkpoints.save_training_payload'):
                result = finish_validation(payload, {'mAP@0.5': .5}, out)
            self.assertEqual(result['best_map50'], .8)
            self.assertFalse(out.with_name('detector_best.pt').exists())


if __name__ == '__main__':
    unittest.main()
