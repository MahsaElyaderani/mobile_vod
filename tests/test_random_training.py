"""Random-only training, legacy resume safety, and retained evaluation policies."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_paired_vod import parse_args
from routing.paired_policy import random_training_schedule, validate_random_training_resume
from paired_config import PolicyConfig


class RandomTrainingTests(unittest.TestCase):
    def test_cli_has_no_schedule_option(self):
        self.assertFalse(hasattr(parse_args(['--checkpoint-out=unused.pt']), 'schedules'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--checkpoint-out=unused.pt', '--schedules', 'random'])

    def test_warmup_options_removed(self):
        args = parse_args(['--checkpoint-out=unused.pt'])
        for flag in ('zero-memory-epochs', 'clean-memory-epochs', 'warmup-interval'):
            self.assertFalse(hasattr(args, flag.replace('-', '_')))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--checkpoint-out=unused.pt', '--' + flag, '1'])

    def test_legacy_warmup_resume_boundary(self):
        payload = {'epoch': 5, 'run_definition': {'random_schedule_version': 3,
                   'arguments': {'zero_memory_epochs': 4, 'clean_memory_epochs': 2}}}
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            validate_random_training_resume(payload)
        payload['epoch'] = 6
        validate_random_training_resume(payload)
        payload['resume_arguments'] = {'clean_memory_epochs': 3}
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            validate_random_training_resume(payload)

    def test_rng_matches_existing_random_only_runs(self):
        for seed in (0, 4, 100):
            torch.manual_seed(seed)
            for _ in range(4):
                torch.randint(1, ())
            expected = torch.stack([torch.rand(10) < .5 for _ in range(4)])
            expected_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            actual, small_commit = random_training_schedule(4, 10, 'cpu')
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertFalse(small_commit.any())

    def test_random_resume_accepts_legacy_and_new_metadata(self):
        for args in ({}, {'schedules': ['random']}):
            validate_random_training_resume({'run_definition': {
                'random_schedule_version': 3, 'arguments': args}})

    def test_incompatible_resume_rejected(self):
        for schedule in (['fixed4'], ['small'], ['big'], ['random', 'fixed8']):
            with self.assertRaisesRegex(ValueError, 'removed'):
                validate_random_training_resume({'run_definition': {
                    'random_schedule_version': 3, 'arguments': {'schedules': schedule}}})
        for version in (None, 1, 2):
            with self.assertRaisesRegex(ValueError, 'version 3'):
                validate_random_training_resume({'run_definition': {
                    'random_schedule_version': version, 'arguments': {'schedules': ['random']}}})

    def test_evaluation_comparisons_remain_available(self):
        for mode in ('small', 'big', 'fixed'):
            self.assertEqual(PolicyConfig(mode=mode).mode, mode)


if __name__ == '__main__':
    unittest.main()
