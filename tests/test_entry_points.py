"""Regression checks for direct and package evaluation entry points."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_comparison_help_as_script_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-B', str(ROOT / 'evaluate/compare_paired_runs.py'), '--help'],
                cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--runs', result.stdout)

    def test_comparison_help_as_module(self):
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'evaluate.compare_paired_runs', '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--runs', result.stdout)


if __name__ == '__main__':
    unittest.main()
