"""Path integration tests; run: python3 -m unittest discover -s tests -v."""
import argparse
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import path as paths


class PathTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--annotations', required=True)
        parser.add_argument('--frames-root', required=True)
        parser.add_argument('--val-annotations')
        parser.add_argument('--val-frames-root')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--resume')
        group.add_argument('--checkpoint-in')
        parser.add_argument('--epochs', type=int, default=5)
        return parser

    def configured(self, argv=(), extra=None):
        with patch.object(paths, 'TRAIN_ANNOTATIONS', 'data with spaces/train.json'), \
             patch.object(paths, 'TRAIN_FRAMES', 'data with spaces/frames'), \
             patch.dict(paths.TASK_PATHS, {'train': extra or {}}):
            return paths.parse_paths(self.parser(), 'train', argv)

    def test_configured_paths_are_project_relative(self):
        args = self.configured()
        self.assertEqual(args.annotations, str(paths.PROJECT_ROOT / 'data with spaces/train.json'))
        self.assertEqual(args.frames_root, str(paths.PROJECT_ROOT / 'data with spaces/frames'))
        self.assertEqual(args.epochs, 5)

    def test_explicit_paths_keep_original_cli_semantics(self):
        args = self.configured(['--annotations=../manual.json', '--frames-root', './frames', '--epochs', '7'])
        self.assertEqual(args.annotations, '../manual.json')
        self.assertEqual(args.frames_root, './frames')
        self.assertEqual(args.epochs, 7)

    def test_abbreviated_cli_overrides_config(self):
        self.assertEqual(self.configured(['--annot', 'manual.json']).annotations, 'manual.json')

    def test_validation_can_be_disabled_and_resume_defaults_off(self):
        with patch.object(paths, 'TRAIN_VALIDATION_ANNOTATIONS', None), \
             patch.object(paths, 'TRAIN_VALIDATION_FRAMES', None):
            args = self.configured()
        self.assertIsNone(args.val_annotations)
        self.assertIsNone(args.val_frames_root)
        self.assertIsNone(args.resume)

    def test_training_validation_paths_are_used(self):
        with patch.object(paths, 'TRAIN_VALIDATION_ANNOTATIONS', '/validation.json'), \
             patch.object(paths, 'TRAIN_VALIDATION_FRAMES', '/validation'):
            args = self.configured()
        self.assertEqual(args.val_annotations, '/validation.json')
        self.assertEqual(args.val_frames_root, '/validation')

    def test_explicit_mutually_exclusive_selection_wins(self):
        args = self.configured(['--resume', 'saved.pt'], {'checkpoint_in': 'initial.pt'})
        self.assertEqual(args.resume, 'saved.pt')
        self.assertIsNone(args.checkpoint_in)

    def test_conflicting_configuration_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.configured(extra={'resume': 'saved.pt', 'checkpoint_in': 'initial.pt'})
        self.assertEqual(error.exception.code, 2)

    def test_missing_required_paths_still_error(self):
        with patch.object(paths, 'TRAIN_ANNOTATIONS', None), \
             patch.object(paths, 'TRAIN_FRAMES', None), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            paths.parse_paths(self.parser(), 'train', [])
        self.assertEqual(error.exception.code, 2)

    def test_comparison_path_lists_and_overrides(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--runs', nargs='+', required=True)
        with patch.dict(paths.TASK_PATHS, {'compare_paired_runs': {'runs': ['run one', 'run two']}}):
            args = paths.parse_paths(parser, 'compare_paired_runs', [])
            self.assertEqual(args.runs, [str(paths.PROJECT_ROOT / name) for name in ['run one', 'run two']])
            args = paths.parse_paths(parser, 'compare_paired_runs', ['--runs', 'custom'])
            self.assertEqual(args.runs, ['custom'])

    def test_weight_producer_and_consumer_agree(self):
        self.assertEqual(paths.task_paths('prepare_weights')['output'],
                         paths.task_paths('train')['backbone_weights'])

    def test_required_shell_setting_fails_when_unset(self):
        with patch.object(paths, 'TRAIN_ANNOTATIONS', None), self.assertRaisesRegex(ValueError, 'TRAIN_ANNOTATIONS'):
            paths.required_path('TRAIN_ANNOTATIONS')


if __name__ == '__main__':
    unittest.main()
