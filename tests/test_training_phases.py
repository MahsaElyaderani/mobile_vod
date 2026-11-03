"""Phase aliases preserve legacy checkpoint and initialization semantics."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import parse_args as standalone_args
from train_joint_pretrain import parse_args as classification_args
from train_paired_vod import parse_args as joint_args
from utils.paired_checkpoints import initialize_pair
from paired_config import PairedConfig


class TrainingPhaseTests(unittest.TestCase):
    def test_classification_validation_limit_requires_validation(self):
        with patch('path.TRAIN_VALIDATION_ANNOTATIONS', None), \
             patch('path.TRAIN_VALIDATION_FRAMES', None), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, '2'):
            classification_args(['--checkpoint-out', 'unused.pt', '--max-val-videos', '1'])

    def test_standalone_aliases_match_legacy_arguments(self):
        for phase, stage in [('standalone_pretrain', '1'), ('standalone_detection', '2')]:
            extra = ['--epochs', '1'] + (['--checkpoint-in', 'stage1.pt'] if stage == '2' else [])
            self.assertEqual(vars(standalone_args(['--phase', phase] + extra)),
                             vars(standalone_args(['--stage', stage] + extra)))

    def test_standalone_phase_stays_explicit_on_resume(self):
        with patch('utils.resume.load_checkpoint', return_value={'resume_arguments': {'stage': '1'}}):
            args = standalone_args(['--phase', 'standalone_detection', '--epochs', '2', '--resume', 'old.pt'])
        self.assertEqual(args.stage, '2')  # Actual checkpoint stage remains checked by main().

    def test_conflicting_standalone_flags_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            standalone_args(['--phase', 'standalone_detection', '--stage', '1', '--epochs', '1'])

    def test_joint_alias_and_legacy_initialization_match(self):
        common = ['--checkpoint-out', 'unused.pt', '--fast-init', 'timm', '--fast-width-mult', '.5']
        new = joint_args(common + ['--phase', 'joint_detection', '--init-source', 'standalone_detection', '--init-checkpoint', 'old_stage2.pt'])
        old = joint_args(common + ['--big-checkpoint', 'old_stage2.pt'])
        self.assertEqual(vars(new), vars(old))

    def test_invalid_initialization_stage_rejected(self):
        with patch('utils.joint_initialization.load_checkpoint', return_value={'stage': 'stage2'}):
            with self.assertRaisesRegex(ValueError, 'joint_pretrain'):
                initialize_pair('old.pt', PairedConfig(), init_source='joint_pretrain')


if __name__ == '__main__':
    unittest.main()
